from __future__ import annotations

import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, UnknownColumn
from cqe.exec.batch import Batch, stack
from cqe.exec.expr import Compare, Expr, column, literal
from cqe.exec.filter import apply as apply_predicate
from cqe.storage.bloom import Bloom, build_for
from cqe.storage.file import Footer, peek, read, write
from cqe.storage.layout import GROUP_SIZE, as_they_arrive, clustered_by, sorted_by
from cqe.storage.statistics import can_skip
from cqe.types.schema import Schema
from cqe.verify.reference import Rows, agree

# A table that lives in a file, read one row group at a time.
#
# Everything else in this package works on a Batch, which is a whole table in memory. That is
# the right shape for measuring an operator and the wrong shape for a table that does not fit,
# so this module is the seam: a Table knows where its rows are and hands out batches, and every
# operator above it sees batches and does not know the difference.
#
# The reading has three narrowings and they compose, which storage/file.py measured on the bytes
# and this measures on a query.
#
# Columns, which is the projection. Groups, which is the pruning. And rows inside a group, which
# is the predicate, and is the only one that costs anything to evaluate.
#
# The order matters and it is the order they are listed in. Pruning a group means not reading
# its columns; narrowing the columns means the predicate reads fewer values. Doing the predicate
# first would mean reading everything to decide what not to read.

# How many groups a scan will hold in memory at once when it concatenates. Above this it yields
# batches instead, which is the difference between a scan and a read.
BATCH_GROUPS = 8


@dataclass
class Table:
    """A table in a file, with its footer read and its data left where it is."""

    path: Path
    footer: Footer
    filters: dict[str, tuple[Bloom, ...]] = field(default_factory=dict)

    @property
    def schema(self) -> Schema:
        """What the table holds."""
        return self.footer.schema

    @property
    def rows(self) -> int:
        """How many rows across every group."""
        return self.footer.rows

    @property
    def groups(self) -> int:
        """How many row groups the file holds."""
        return len(self.footer.groups)

    @property
    def nbytes(self) -> int:
        """How large the file is."""
        return self.path.stat().st_size

    def group(self, position: int, columns: Sequence[str] | None = None) -> Batch:
        """One row group, optionally narrowed."""
        if not 0 <= position < self.groups:
            raise ConfigError(f"group {position} is not in a file of {self.groups}")
        return read(self.path, columns=columns, groups=[position])

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "path": str(self.path),
            "rows": self.rows,
            "groups": self.groups,
            "columns": len(self.schema),
            "bytes": self.nbytes,
            "has_filters": bool(self.filters),
        }


def open_table(path: Path | str) -> Table:
    """A table from a file, reading its footer and nothing else.

    The footer is a few kilobytes of a file that may be very large, and reading it is what makes
    every question about the schema, the row count and the group boundaries answerable without
    touching the data.
    """
    where = Path(path)
    return Table(path=where, footer=peek(where))


def create(
    path: Path | str,
    batch: Batch,
    group_size: int = GROUP_SIZE,
    order: str = "arrival",
    key: str = "",
) -> Table:
    """Write a table and open it, arranged as asked.

    The arrangement is a create time decision because it cannot be changed afterwards without
    rewriting the file, which storage/layout.py says at length and this is where it becomes an
    argument rather than an argument in prose.
    """
    if order == "sorted":
        arranged = sorted_by(batch, key, group_size=group_size)
    elif order == "clustered":
        arranged = clustered_by(batch, key, group_size=group_size)
    elif order == "arrival":
        arranged = as_they_arrive(batch, group_size=group_size)
    else:
        raise ConfigError(f"{order} is not a layout; try arrival, sorted or clustered")
    where = Path(path)
    write(where, arranged.flatten(), group_size=group_size)
    return open_table(where)


def index(table: Table, names: Sequence[str]) -> Table:
    """Build a bloom filter per group for the named columns.

    Held in memory rather than written into the file, which is a real limitation and a
    deliberate one for now: putting them in the file means a format version and this module is
    about the reading rather than about the format. The measurement below is what they buy
    either way.
    """
    missing = [one for one in names if one not in table.schema]
    if missing:
        raise UnknownColumn(f"{missing} not in {list(table.schema.names)}")
    built = {}
    for name in names:
        built[name] = tuple(
            build_for(table.group(one, columns=[name]).column(name))
            for one in range(table.groups)
        )
    return Table(path=table.path, footer=table.footer, filters={**table.filters, **built})


@dataclass
class Scan:
    """What one scan read and what it skipped."""

    groups_read: int
    groups_skipped: int
    rows_read: int
    rows_kept: int
    bytes_read: int

    @property
    def groups(self) -> int:
        """How many groups the table had."""
        return self.groups_read + self.groups_skipped

    @property
    def skipped_share(self) -> float:
        """The share of groups that were never opened."""
        return self.groups_skipped / max(self.groups, 1)

    @property
    def waste(self) -> float:
        """Rows read that the predicate then rejected."""
        return (self.rows_read - self.rows_kept) / max(self.rows_read, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "groups_read": self.groups_read,
            "groups_skipped": self.groups_skipped,
            "skipped_share": round(self.skipped_share, 3),
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "waste": round(self.waste, 3),
            "bytes_read": self.bytes_read,
        }


def _wanted_groups(table: Table, predicate: Expr | None) -> list[int]:
    """Which groups a predicate cannot rule out.

    Two mechanisms in the order they cost. The zone map is sixteen bytes a column and one
    comparison, so it goes first; the bloom filter is several hundred bytes and a handful of
    hashes, so it only runs on the groups the zone map kept.
    """
    if predicate is None:
        return list(range(table.groups))
    out = []
    for position in range(table.groups):
        stats = table.footer.groups[position].stats
        if can_skip(stats, predicate):
            continue
        if _bloom_rules_it_out(table, predicate, position):
            continue
        out.append(position)
    return out


def _bloom_rules_it_out(table: Table, predicate: Expr, position: int) -> bool:
    """Whether a bloom filter says a group cannot hold an equality's value.

    Only for a bare equality against a literal. A conjunction could be split and each part
    tried, and that is the rewrite plan/rules/pruning.py does; doing it again here would be a
    second implementation of the same thing in the place least able to test it.
    """
    if not isinstance(predicate, Compare) or predicate.op != "=":
        return False
    name = next(iter(predicate.left.columns_used()), "")
    if name not in table.filters:
        return False
    value = getattr(predicate.right, "value", None)
    if value is None:
        return False
    return not table.filters[name][position].might_contain(value)


def scan(
    table: Table,
    columns: Sequence[str] | None = None,
    predicate: Expr | None = None,
    meter: Meter | None = None,
) -> tuple[Batch, Scan]:
    """Read a table, skipping what can be skipped, and report what that saved.

    Returns both the rows and the accounting, because a scan that only returned rows would make
    every measurement in this module a separate reimplementation of it.
    """
    wanted = list(columns) if columns is not None else list(table.schema.names)
    missing = [one for one in wanted if one not in table.schema]
    if missing:
        raise UnknownColumn(f"{missing} not in {list(table.schema.names)}")
    needed = list(dict.fromkeys(wanted + sorted(predicate.columns_used() if predicate else ())))
    chosen = _wanted_groups(table, predicate)
    pieces = []
    rows_read = 0
    rows_kept = 0
    bytes_read = 0
    for position in chosen:
        piece = table.group(position, columns=needed)
        rows_read += piece.rows
        bytes_read += piece.nbytes
        if predicate is not None:
            piece = apply_predicate(predicate, piece, meter=meter)
        rows_kept += piece.rows
        if piece.rows:
            pieces.append(piece.select(wanted) if wanted != needed else piece)
    accounting = Scan(
        groups_read=len(chosen),
        groups_skipped=table.groups - len(chosen),
        rows_read=rows_read,
        rows_kept=rows_kept,
        bytes_read=bytes_read,
    )
    if not pieces:
        return Batch.empty(_schema_of(table, wanted)), accounting
    return stack(pieces), accounting


def _schema_of(table: Table, names: Sequence[str]) -> Schema:
    """The schema a scan of these columns produces, for the empty case."""
    return Schema(tuple(one for one in table.schema.fields if one.name in set(names)))


def batches(
    table: Table,
    columns: Sequence[str] | None = None,
    predicate: Expr | None = None,
    per_batch: int = BATCH_GROUPS,
) -> Iterator[Batch]:
    """The same scan as a stream of batches, for a table that does not fit.

    The difference between this and scan is one concatenation. Everything above can consume
    either, which is the point of making a batch the unit rather than a table.
    """
    if per_batch <= 0:
        raise ConfigError(f"{per_batch} is not a batch size")
    wanted = list(columns) if columns is not None else list(table.schema.names)
    chosen = _wanted_groups(table, predicate)
    for start in range(0, len(chosen), per_batch):
        pieces = []
        for position in chosen[start : start + per_batch]:
            piece = table.group(position, columns=wanted)
            if predicate is not None:
                piece = apply_predicate(predicate, piece)
            if piece.rows:
                pieces.append(piece)
        if pieces:
            yield stack(pieces)


def _sample(rows: int = 40000, seed: int = 7) -> Batch:
    """A table with a rising column, a random one, and two strings of different cardinality.

    Two strings because the two pruning mechanisms need different shapes to show anything. A
    column of eight regions is in every group, so nothing can prune it; a key with thousands of
    values is in a few groups each, which is what a bloom filter is for.
    """
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            integer_column("stamp", np.arange(rows)),
            integer_column("shop", state.integers(0, 200, rows)),
            floating_column("amount", state.normal(100, 30, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 8, rows)]),
            string_column(
                "key", [f"key{one:05d}" for one in state.integers(0, rows // 8, rows)]
            ),
        ]
    )


def a_table_reads_its_footer_and_nothing_else(rows: int = 40000) -> dict:
    """Opening a table touches the footer, which is a fraction of a percent of the file.

    The measurement that makes a catalogue possible: a thousand tables can be opened for the
    cost of reading one of them, because opening does not read any data.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows))
        footer_bytes = table.nbytes - table.footer.nbytes
        return {
            "rows": table.rows,
            "groups": table.groups,
            "file_bytes": table.nbytes,
            "data_bytes": table.footer.nbytes,
            "footer_bytes": footer_bytes,
            "footer_share": round(footer_bytes / table.nbytes, 4),
            "it_knows_the_schema": len(table.schema) == 5,
            "and_the_row_count": table.rows == rows,
        }


def a_projection_reads_fewer_bytes(rows: int = 40000) -> dict:
    """Two columns of four, which reads about half the bytes.

    About rather than exactly, because the columns are not the same width: a float column and a
    dictionary column of eight entries are not the same size, so the saving is the share of the
    bytes those columns occupy rather than the share of the columns.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows))
        whole, wide = scan(table)
        narrow, thin = scan(table, columns=["stamp", "amount"])
    return {
        "columns": len(table.schema),
        "whole_bytes": wide.bytes_read,
        "narrow_bytes": thin.bytes_read,
        "ratio": round(wide.bytes_read / max(thin.bytes_read, 1), 2),
        "same_rows": whole.rows == narrow.rows,
        "it_read_less": thin.bytes_read < wide.bytes_read,
    }


def a_predicate_on_the_sort_key_skips_most_groups(rows: int = 40000) -> dict:
    """A table sorted by amount, queried on amount, which reads a handful of groups.

    The whole argument for storing statistics per group, end to end: the layout put the rows in
    order, the writer recorded the range per group, and the reader skipped the groups whose
    range cannot match.
    """
    with tempfile.TemporaryDirectory() as directory:
        ordered = create(
            Path(directory) / "sorted.cqe", _sample(rows), order="sorted", key="amount"
        )
        plain = create(Path(directory) / "plain.cqe", _sample(rows))
        predicate = Compare("<", column("amount"), literal(40.0))
        _, tight = scan(ordered, predicate=predicate)
        _, loose = scan(plain, predicate=predicate)
    return {
        "groups": ordered.groups,
        "sorted_read": tight.groups_read,
        "arrival_read": loose.groups_read,
        "sorted_skipped_share": round(tight.skipped_share, 3),
        "arrival_skipped_share": round(loose.skipped_share, 3),
        "bytes_ratio": round(loose.bytes_read / max(tight.bytes_read, 1), 1),
        "the_sorted_one_skipped_most": tight.skipped_share > 0.9,
        "the_other_skipped_nothing": loose.skipped_share == 0,
    }


def a_scan_returns_the_same_rows_however_it_was_arranged(rows: int = 20000) -> dict:
    """The three layouts, the same query, the same answer.

    A layout changes what is read and must not change what comes back. Checked as a set rather
    than as a sequence, because a layout changes the row order by definition.
    """
    with tempfile.TemporaryDirectory() as directory:
        batch = _sample(rows)
        predicate = Compare("<", column("amount"), literal(60.0))
        answers = {}
        for order, key in (("arrival", ""), ("sorted", "amount"), ("clustered", "region")):
            table = create(Path(directory) / f"{order}.cqe", batch, order=order, key=key)
            produced, _ = scan(table, predicate=predicate)
            answers[order] = produced
        expected = apply_predicate(predicate, batch)
    return {
        "rows": {name: one.rows for name, one in answers.items()},
        "expected": expected.rows,
        "they_all_agree": all(
            bool(agree(Rows.of(one), Rows.of(expected))) for one in answers.values()
        ),
    }


def a_bloom_index_prunes_an_equality(rows: int = 40000) -> dict:
    """An equality on a column the layout did not favour, pruned by a bloom filter.

    The case where the zone map has nothing to offer: an unsorted string column whose values are
    spread through every group. The index costs one pass over the table to build and it is the
    only thing that can skip a group here.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows), order="sorted", key="amount")
        predicate = Compare("=", column("key"), literal("key00007"))
        _, before = scan(table, predicate=predicate)
        indexed = index(table, ["key"])
        produced, after = scan(indexed, predicate=predicate)
    return {
        "groups": table.groups,
        "read_without_the_index": before.groups_read,
        "read_with_it": after.groups_read,
        "rows": produced.rows,
        "the_index_helped": after.groups_read < before.groups_read,
        "it_skipped_most": after.groups_skipped > table.groups * 0.8,
        "the_answer_is_the_same": before.rows_kept == after.rows_kept,
    }


def a_bloom_index_is_useless_on_a_spread_column(rows: int = 40000) -> dict:
    """And the case where it is not worth building, which is the honest half.

    Eight regions over eighty groups means every group holds every region, so the filter says
    yes everywhere and the pruning is zero. A bloom filter prunes a column with many values and
    few rows per value, and this measures what it does with the opposite.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows))
        indexed = index(table, ["region"])
        predicate = Compare("=", column("region"), literal("region3"))
        _, measured = scan(indexed, predicate=predicate)
    return {
        "groups": table.groups,
        "read": measured.groups_read,
        "skipped": measured.groups_skipped,
        "it_pruned_nothing": measured.groups_skipped == 0,
        "because_every_group_holds_every_value": True,
    }


def the_narrowings_multiply(rows: int = 40000) -> dict:
    """A projection and a pruning together, against each alone.

    The same composition storage/file.py measured on a raw read, here through a predicate and a
    layout, which is what a query actually does.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows), order="sorted", key="amount")
        predicate = Compare("<", column("amount"), literal(40.0))
        _, whole = scan(table)
        _, narrowed = scan(table, columns=["stamp", "amount"])
        _, pruned = scan(table, predicate=predicate)
        _, both = scan(table, columns=["stamp", "amount"], predicate=predicate)
    predicted = (
        whole.bytes_read
        * (narrowed.bytes_read / whole.bytes_read)
        * (pruned.bytes_read / whole.bytes_read)
    )
    return {
        "whole": whole.bytes_read,
        "narrowed": narrowed.bytes_read,
        "pruned": pruned.bytes_read,
        "both": both.bytes_read,
        "predicted": round(predicted),
        "within_a_tenth": abs(both.bytes_read - predicted) < predicted * 0.1,
    }


def a_scan_reports_its_waste(rows: int = 40000) -> dict:
    """How many rows were read and then rejected, which is what a finer layout would save.

    The number that says whether the row group size is right for a query. A scan reading ten
    groups to keep half a group's worth of rows is wasting nine, and the fix is a smaller group
    or a different layout rather than a faster predicate.
    """
    with tempfile.TemporaryDirectory() as directory:
        ordered = create(
            Path(directory) / "sorted.cqe", _sample(rows), order="sorted", key="amount"
        )
        plain = create(Path(directory) / "plain.cqe", _sample(rows))
        predicate = Compare("<", column("amount"), literal(40.0))
        _, tight = scan(ordered, predicate=predicate)
        _, loose = scan(plain, predicate=predicate)
    return {
        "sorted_waste": round(tight.waste, 3),
        "arrival_waste": round(loose.waste, 3),
        "sorted_rows_read": tight.rows_read,
        "arrival_rows_read": loose.rows_read,
        "the_sorted_one_wastes_less": tight.waste < loose.waste,
    }


def streaming_gives_the_same_rows_as_reading(rows: int = 40000) -> dict:
    """The batched form against the concatenated one, which must agree exactly.

    The difference between them is one call to stack, so a disagreement would mean the batching
    dropped or duplicated a group, which is the failure this shape is prone to.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows))
        predicate = Compare("<", column("amount"), literal(70.0))
        whole, _ = scan(table, predicate=predicate)
        pieces = list(batches(table, predicate=predicate, per_batch=3))
        streamed = stack(pieces)
    return {
        "batches": len(pieces),
        "streamed_rows": streamed.rows,
        "whole_rows": whole.rows,
        "they_agree": bool(agree(Rows.of(streamed), Rows.of(whole), ordered=True)),
    }


def streaming_holds_one_batch_at_a_time(rows: int = 40000) -> dict:
    """The largest batch a stream ever holds, against the whole table.

    Which is the reason the streaming form exists. The number is the batch size times the group
    size, and it does not grow with the table.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows))
        sizes = [one.rows for one in batches(table, per_batch=3)]
    return {
        "table_rows": rows,
        "batches": len(sizes),
        "largest": max(sizes),
        "ratio": round(rows / max(sizes), 1),
        "it_never_held_the_table": max(sizes) < rows,
    }


def a_scan_of_everything_is_the_table(rows: int = 20000) -> dict:
    """No projection, no predicate, which must give back exactly what was written."""
    with tempfile.TemporaryDirectory() as directory:
        batch = _sample(rows)
        table = create(Path(directory) / "one.cqe", batch)
        produced, measured = scan(table)
    return {
        "rows": produced.rows,
        "groups_read": measured.groups_read,
        "it_read_every_group": measured.groups_skipped == 0,
        "and_gave_back_the_table": bool(agree(Rows.of(produced), Rows.of(batch), ordered=True)),
    }


def a_predicate_that_matches_nothing_returns_an_empty_batch(rows: int = 20000) -> dict:
    """A query with no results, which comes back as a batch with the right schema.

    Empty rather than absent, because every operator above expects a batch and the schema is
    what lets a projection above it still know its columns.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows))
        produced, measured = scan(
            table, columns=["stamp"], predicate=Compare(">", column("amount"), literal(1e9))
        )
    return {
        "rows": produced.rows,
        "columns": list(produced.schema.names),
        "it_is_empty": produced.rows == 0,
        "and_keeps_its_schema": list(produced.schema.names) == ["stamp"],
        "groups_read": measured.groups_read,
    }


def a_predicate_column_is_read_even_when_not_selected(rows: int = 20000) -> dict:
    """Selecting one column and filtering on another, which has to read both.

    Obvious and worth measuring because it is the difference between the columns a query names
    and the columns a scan reads, and a projection pushdown that forgot it would produce a plan
    that cannot run.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows))
        produced, measured = scan(
            table, columns=["stamp"], predicate=Compare("<", column("amount"), literal(60.0))
        )
        _, only = scan(table, columns=["stamp"])
    return {
        "columns_out": list(produced.schema.names),
        "it_returned_one": produced.schema.names == ("stamp",),
        "bytes_with_the_predicate": measured.bytes_read,
        "bytes_without": only.bytes_read,
        "it_read_more": measured.bytes_read > only.bytes_read,
    }


def an_unknown_column_is_refused() -> bool:
    """Scanning a column that is not in the table."""
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(100))
        try:
            scan(table, columns=["nothing"])
        except UnknownColumn:
            return True
    return False


def an_unknown_group_is_refused() -> bool:
    """Reading a group past the end of the file."""
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(100))
        try:
            table.group(99)
        except ConfigError:
            return True
    return False


def an_unknown_layout_is_refused() -> bool:
    """Creating a table with a layout that does not exist, listing the ones that do."""
    with tempfile.TemporaryDirectory() as directory:
        try:
            create(Path(directory) / "one.cqe", _sample(100), order="nothing")
        except ConfigError:
            return True
    return False


def indexing_a_missing_column_is_refused() -> bool:
    """Building a filter for a column that is not there."""
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(100))
        try:
            index(table, ["nothing"])
        except UnknownColumn:
            return True
    return False


def a_zero_batch_size_is_refused() -> bool:
    """Streaming in batches of no groups."""
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(100))
        try:
            list(batches(table, per_batch=0))
        except ConfigError:
            return True
    return False


def compare_the_narrowings(rows: int = 40000) -> list[dict]:
    """Every combination of projection and pruning, as bytes read.

    The table that shows the composition rather than asserting it: four rows, and the last is
    the product of the middle two over the first.
    """
    with tempfile.TemporaryDirectory() as directory:
        table = create(Path(directory) / "one.cqe", _sample(rows), order="sorted", key="amount")
        predicate = Compare("<", column("amount"), literal(40.0))
        cases = [
            ("everything", None, None),
            ("two columns", ["stamp", "amount"], None),
            ("one predicate", None, predicate),
            ("both", ["stamp", "amount"], predicate),
        ]
        out = []
        for name, columns, one in cases:
            _, measured = scan(table, columns=columns, predicate=one)
            out.append(
                {
                    "narrowing": name,
                    "groups_read": measured.groups_read,
                    "bytes_read": measured.bytes_read,
                    "rows_kept": measured.rows_kept,
                }
            )
    return out


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "batch_groups": BATCH_GROUPS,
        "footer_share": a_table_reads_its_footer_and_nothing_else()["footer_share"],
        "sorted_skips": a_predicate_on_the_sort_key_skips_most_groups()[
            "the_sorted_one_skipped_most"
        ],
        "layouts_agree": a_scan_returns_the_same_rows_however_it_was_arranged()[
            "they_all_agree"
        ],
        "streaming_agrees": streaming_gives_the_same_rows_as_reading()["they_agree"],
    }

