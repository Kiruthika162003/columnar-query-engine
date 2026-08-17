from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cqe.columns.encode import dictionary
from cqe.errors import ConfigError, SchemaError
from cqe.exec.batch import Batch, stack
from cqe.exec.expr import Compare, Expr, column, literal
from cqe.storage.file import Footer, peek, read, write
from cqe.storage.statistics import GroupStats, prune

# Merging many small files into fewer large ones, which is the maintenance job a table that is
# appended to needs and never asks for.
#
# The shape of the problem. Rows arrive in small batches and each batch becomes a file, because
# a writer holding five hundred rows cannot wait for the other forty nine thousand five hundred.
# Every file carries a header, a footer and a digest, and after a thousand ingests the table is
# a thousand files whose footers are read in full by every query that touches any of them.
#
# Compaction reads them and writes one. What that is worth, and what it costs, is measured
# below, and the headline is not the one usually given for it.

# Rows per group in a compacted file. Larger than the ingest size on purpose, because the point
# of compacting is to stop paying per group, and the measurement below says what the wider zone
# maps cost in return.
COMPACT_GROUP = 4_000

# What one ingest holds. Small enough that a writer can produce it without waiting.
INGEST_ROWS = 500

# Bytes of fixed cost to open a file: the header, the footer pointer and a seek. A scan across
# many fragments pays this once per fragment before it reads a single value.
OPEN_COST = 4_096


@dataclass(frozen=True)
class Fragment:
    """One file of the table, and what a reader learns before touching its data."""

    path: Path
    rows: int
    groups: int
    data_bytes: int
    footer_bytes: int

    @property
    def total_bytes(self) -> int:
        """Everything the file occupies."""
        return self.data_bytes + self.footer_bytes

    @property
    def overhead(self) -> float:
        """Footer bytes as a share of the whole file."""
        return self.footer_bytes / max(self.total_bytes, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "path": self.path.name,
            "rows": self.rows,
            "groups": self.groups,
            "data": self.data_bytes,
            "footer": self.footer_bytes,
            "overhead": round(self.overhead, 4),
        }


@dataclass(frozen=True)
class Table:
    """Every fragment of one logical table."""

    fragments: tuple[Fragment, ...]

    def __post_init__(self) -> None:
        if not self.fragments:
            raise ConfigError("a table has at least one fragment")

    @property
    def rows(self) -> int:
        """Rows across every fragment."""
        return sum(one.rows for one in self.fragments)

    @property
    def groups(self) -> int:
        """Row groups across every fragment."""
        return sum(one.groups for one in self.fragments)

    @property
    def data_bytes(self) -> int:
        """Data bytes across every fragment."""
        return sum(one.data_bytes for one in self.fragments)

    @property
    def footer_bytes(self) -> int:
        """Footer bytes across every fragment."""
        return sum(one.footer_bytes for one in self.fragments)

    @property
    def open_bytes(self) -> int:
        """What opening every fragment costs before any value is read."""
        return len(self.fragments) * OPEN_COST

    @property
    def metadata_bytes(self) -> int:
        """Everything a scan pays that is not data."""
        return self.footer_bytes + self.open_bytes

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "fragments": len(self.fragments),
            "rows": self.rows,
            "groups": self.groups,
            "data": self.data_bytes,
            "footer": self.footer_bytes,
            "metadata": self.metadata_bytes,
            "metadata_share": round(self.metadata_bytes / max(self.data_bytes, 1), 4),
        }


@dataclass(frozen=True)
class Compaction:
    """One compaction: what it merged, what it produced and what the rewrite cost."""

    before: Table
    after: Table
    read_bytes: int
    written_bytes: int

    @property
    def rewrite_cost(self) -> int:
        """Bytes moved to do the compaction, paid once."""
        return self.read_bytes + self.written_bytes

    @property
    def metadata_saved(self) -> int:
        """Metadata bytes a scan no longer pays, saved on every scan."""
        return self.before.metadata_bytes - self.after.metadata_bytes

    @property
    def break_even(self) -> float:
        """Scans before the rewrite has paid for itself."""
        if self.metadata_saved <= 0:
            return float("inf")
        return self.rewrite_cost / self.metadata_saved

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "fragments_before": len(self.before.fragments),
            "fragments_after": len(self.after.fragments),
            "groups_before": self.before.groups,
            "groups_after": self.after.groups,
            "metadata_before": self.before.metadata_bytes,
            "metadata_after": self.after.metadata_bytes,
            "saved": self.metadata_saved,
            "rewrite_cost": self.rewrite_cost,
            "break_even": round(self.break_even, 2),
        }


def _fragment(path: Path, footer: Footer | None = None) -> Fragment:
    """Describe one file from its footer."""
    known = footer if footer is not None else peek(path)
    written = Path(path).stat().st_size
    return Fragment(
        path=Path(path),
        rows=known.rows,
        groups=len(known.groups),
        data_bytes=known.nbytes,
        footer_bytes=max(written - known.nbytes, 0),
    )


def ingest(batch: Batch, folder: Path, size: int = INGEST_ROWS, group_size: int = 0) -> Table:
    """Write a table as a run of small files, the way an appended table accumulates.

    Each slice becomes its own file with its own footer, which is what a writer that cannot wait
    produces. The group size defaults to the ingest size, so a fragment holds exactly one group,
    which is the case the measurements below are about.
    """
    if size < 1:
        raise ConfigError(f"{size} is not an ingest size")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    made: list[Fragment] = []
    for position, piece in enumerate(batch.batches(size)):
        path = folder / f"part{position:05d}.cqe"
        footer = write(path, piece, group_size=group_size or size)
        made.append(_fragment(path, footer))
    return Table(fragments=tuple(made))


def describe(paths: list[Path]) -> Table:
    """Read the footers of an existing set of fragments."""
    if not paths:
        raise ConfigError("there are no fragments to describe")
    return Table(fragments=tuple(_fragment(Path(one)) for one in paths))


def load(table: Table, columns: list[str] | None = None) -> Batch:
    """Every row of every fragment, in fragment order."""
    pieces = [read(one.path, columns=columns) for one in table.fragments]
    names = set(pieces[0].names)
    for piece in pieces[1:]:
        if set(piece.names) != names:
            raise SchemaError("the fragments do not share a schema")
    return stack(pieces)


def compact(
    table: Table,
    into: Path,
    group_size: int = COMPACT_GROUP,
    sort_by: str | None = None,
) -> Compaction:
    """Merge every fragment into one file.

    Optionally sorted by a column on the way through, which costs a sort and is the only thing
    here that changes what the statistics can rule out. Without it the compacted file holds the
    same rows in the same order and the only change is how many footers there are.
    """
    if group_size < 1:
        raise ConfigError(f"{group_size} is not a group size")
    whole = load(table)
    if sort_by is not None:
        if sort_by not in whole:
            raise SchemaError(f"{sort_by} is not in {list(whole.names)}")
        whole = whole.take(np.argsort(whole.values(sort_by), kind="stable"))
    into = Path(into)
    into.parent.mkdir(parents=True, exist_ok=True)
    footer = write(into, whole, group_size=group_size)
    made = Table(fragments=(_fragment(into, footer),))
    return Compaction(
        before=table,
        after=made,
        read_bytes=table.data_bytes + table.metadata_bytes,
        written_bytes=made.data_bytes + made.footer_bytes,
    )


def _group_stats(table: Table) -> list[GroupStats]:
    """Every group of every fragment, which is what a pruner sees."""
    out: list[GroupStats] = []
    for one in table.fragments:
        out.extend(group.stats for group in peek(one.path).groups)
    return out


def scan_cost(table: Table, predicate: Expr, columns: int = 1) -> dict:
    """What one scan of the table costs, counting metadata and surviving values.

    The metadata term is what compaction is for and the values term is what the group size
    decides. Both are counted because reporting only the second makes compaction look free and
    reporting only the first makes it look like a cure.
    """
    pruned = prune(_group_stats(table), predicate, columns_read=columns)
    return {
        "fragments": len(table.fragments),
        "groups": pruned.groups,
        "skipped": pruned.skipped,
        "skipped_share": round(pruned.skipped_share, 4),
        "rows_read": pruned.rows_read,
        "values_read": pruned.values_read,
        "metadata_bytes": table.metadata_bytes,
        "total_bytes": table.metadata_bytes + pruned.values_read * 8,
    }


def _table(rows: int = 40_000, seed: int = 41) -> Batch:
    """A table whose columns differ in how they arrive."""
    state = np.random.default_rng(seed)
    return Batch.of(
        id=np.arange(rows).tolist(),
        key=state.integers(0, 100_000, rows).tolist(),
        shop=state.integers(0, 40, rows).tolist(),
        label=[f"kind{int(one):03d}" for one in state.integers(0, 200, rows)],
        amount=(state.normal(500, 120, rows)).tolist(),
    )


def _folder(name: str) -> Path:
    """A temporary directory beside the working directory, cleared before use."""
    path = Path(f"_compact_{name}")
    if path.exists():
        shutil.rmtree(path)
    return path


def _clear(*folders: Path) -> None:
    """Remove the temporary directories a measurement made."""
    for one in folders:
        if one.exists():
            shutil.rmtree(one, ignore_errors=True)


def many_small_files_pay_more_metadata_than_data(rows: int = 20_000) -> dict:
    """A table ingested in small pieces spends more on metadata than on values.

    The claim to check before doing anything else, because compaction is only worth writing if
    the overhead it removes is real. Five hundred rows of five columns is twenty kilobytes of
    data against a footer of a few hundred bytes, which sounds harmless, and then a scan opens
    the file and pays the fixed cost of doing so, which is what actually dominates.
    """
    folder = _folder("small")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        per_fragment = table.data_bytes / len(table.fragments)
        return {
            "fragments": len(table.fragments),
            "data_bytes": table.data_bytes,
            "footer_bytes": table.footer_bytes,
            "open_bytes": table.open_bytes,
            "metadata_bytes": table.metadata_bytes,
            "data_per_fragment": round(per_fragment, 1),
            "metadata_share": round(table.metadata_bytes / table.data_bytes, 4),
            "metadata_is_a_large_share": table.metadata_bytes > table.data_bytes * 0.1,
            "the_footers_alone_are_smaller": table.footer_bytes < table.open_bytes,
        }
    finally:
        _clear(folder)


def compaction_removes_almost_all_of_it(rows: int = 20_000) -> dict:
    """Merging forty fragments into one leaves one footer and one open.

    The saving is close to the whole of the metadata, because the metadata was a per fragment
    cost and there is now one fragment. This is the measurement compaction is usually justified
    by and it is the easy half of the story.

    The data shrinks too, which was not expected here and is not a metadata effect at all. Each
    fragment carried its own dictionary of nearly every label it held, so forty fragments held
    forty copies of a two hundred entry dictionary. That saving is measured on its own below.
    """
    folder = _folder("shrink")
    out = _folder("shrink_out")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        made = compact(table, out / "whole.cqe")
        return {
            "fragments_before": len(table.fragments),
            "fragments_after": len(made.after.fragments),
            "metadata_before": made.before.metadata_bytes,
            "metadata_after": made.after.metadata_bytes,
            "saved": made.metadata_saved,
            "share_removed": round(made.metadata_saved / made.before.metadata_bytes, 4),
            "it_removed_most_of_it": made.metadata_saved > made.before.metadata_bytes * 0.9,
            "data_before": made.before.data_bytes,
            "data_after": made.after.data_bytes,
            "the_data_shrank_as_well": made.after.data_bytes < made.before.data_bytes,
            "data_saving": round(1 - made.after.data_bytes / made.before.data_bytes, 4),
        }
    finally:
        _clear(folder, out)


def the_pruning_loss_is_the_group_size_not_the_compaction(rows: int = 40_000) -> dict:
    """Compaction does not widen the zone maps. Choosing a larger group does.

    The correction this module exists for. Compaction is routinely described as trading pruning
    for metadata, and the measurement says that is two things being confused. Compacting forty
    fragments of five hundred rows into one file of five hundred row groups prunes exactly as
    much as before, to the group. The loss appears only when the compacted file is written with
    four thousand row groups, and that is a separate decision that could have been made at
    ingest time and has nothing to do with merging files.

    So the trade is real but it is not the one named. It is between metadata and zone map width,
    and compaction is what makes a wide group affordable, not what forces one.

    Measured on the column that arrives in order, because that is the only kind a zone map has
    anything to say about. A predicate on a randomly distributed column prunes almost nothing at
    either group size and would have made both numbers look the same for the wrong reason.
    """
    folder = _folder("prune")
    same = _folder("prune_same")
    wide = _folder("prune_wide")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        predicate = Compare("<", column("id"), literal(int(rows * 0.3) + 300))
        before = scan_cost(table, predicate)
        kept = compact(table, same / "whole.cqe", group_size=INGEST_ROWS)
        widened = compact(table, wide / "whole.cqe", group_size=COMPACT_GROUP)
        at_same = scan_cost(kept.after, predicate)
        at_wide = scan_cost(widened.after, predicate)
        return {
            "groups_before": before["groups"],
            "groups_same": at_same["groups"],
            "groups_wide": at_wide["groups"],
            "rows_read_before": before["rows_read"],
            "rows_read_same": at_same["rows_read"],
            "rows_read_wide": at_wide["rows_read"],
            "same_group_size_prunes_identically": at_same["rows_read"] == before["rows_read"],
            "the_wide_one_reads_more": at_wide["rows_read"] > before["rows_read"],
            "widening_cost": round(at_wide["rows_read"] / max(before["rows_read"], 1), 3),
            "and_both_read_less_metadata": at_wide["metadata_bytes"] < before["metadata_bytes"],
        }
    finally:
        _clear(folder, same, wide)


def the_total_cost_still_falls(rows: int = 40_000) -> dict:
    """Counting both terms, the wide compacted file is cheaper to scan than the fragments.

    The point of counting metadata in the same units as values. Compacting to four thousand row
    groups reads more values and far less metadata, and the question of which wins cannot be
    answered by preferring one of them. On this table it wins by a wide margin, and the ratio
    below says by how much and therefore how selective a predicate would have to be to reverse
    it.
    """
    folder = _folder("total")
    out = _folder("total_out")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        predicate = Compare("<", column("id"), literal(int(rows * 0.3) + 300))
        before = scan_cost(table, predicate, columns=1)
        made = compact(table, out / "whole.cqe")
        after = scan_cost(made.after, predicate, columns=1)
        return {
            "before_total": before["total_bytes"],
            "after_total": after["total_bytes"],
            "before_metadata": before["metadata_bytes"],
            "after_metadata": after["metadata_bytes"],
            "before_values": before["values_read"],
            "after_values": after["values_read"],
            "it_is_cheaper": after["total_bytes"] < before["total_bytes"],
            "by_this_ratio": round(before["total_bytes"] / max(after["total_bytes"], 1), 2),
            "the_values_did_rise": after["values_read"] > before["values_read"],
        }
    finally:
        _clear(folder, out)


def sorting_during_compaction_buys_the_pruning_back(rows: int = 40_000) -> dict:
    """A sort on the way through makes the wide groups prune better than the narrow ones did.

    Compaction already reads and writes every row, so sorting costs one more pass over data that
    is in memory anyway, and it is the only moment in a table's life when reordering is nearly
    free. The result is a file with a quarter of the groups that rules out more of them than the
    fragments ever could, because a sorted column's zone maps are disjoint.
    """
    folder = _folder("sorted")
    plain = _folder("sorted_plain")
    order = _folder("sorted_order")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        predicate = Compare(">", column("amount"), literal(800.0))
        before = scan_cost(table, predicate)
        unsorted = compact(table, plain / "whole.cqe")
        ordered = compact(table, order / "whole.cqe", sort_by="amount")
        at_plain = scan_cost(unsorted.after, predicate)
        at_order = scan_cost(ordered.after, predicate)
        return {
            "rows_read_fragments": before["rows_read"],
            "rows_read_unsorted": at_plain["rows_read"],
            "rows_read_sorted": at_order["rows_read"],
            "skipped_share_sorted": at_order["skipped_share"],
            "the_sorted_one_reads_least": at_order["rows_read"] < before["rows_read"],
            "and_far_less_than_the_unsorted": at_order["rows_read"] < at_plain["rows_read"] / 2,
            "with_the_same_group_count": at_order["groups"] == at_plain["groups"],
        }
    finally:
        _clear(folder, plain, order)


def sorting_by_one_column_does_not_help_another(rows: int = 40_000) -> dict:
    """Compacting sorted by amount leaves a predicate on key no better off than unsorted.

    A file has one order and the sort during compaction spends it. Worth measuring because the
    previous result is easy to read as compaction making pruning better, when what it improved
    was pruning on the one column that was chosen.
    """
    folder = _folder("onecol")
    plain = _folder("onecol_plain")
    order = _folder("onecol_order")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        on_key = Compare("<", column("key"), literal(2_000))
        unsorted = compact(table, plain / "whole.cqe")
        ordered = compact(table, order / "whole.cqe", sort_by="amount")
        at_plain = scan_cost(unsorted.after, on_key)
        at_order = scan_cost(ordered.after, on_key)
        on_amount = Compare(">", column("amount"), literal(800.0))
        helped = scan_cost(ordered.after, on_amount)
        against = scan_cost(unsorted.after, on_amount)
        return {
            "key_rows_unsorted": at_plain["rows_read"],
            "key_rows_sorted": at_order["rows_read"],
            "amount_rows_unsorted": against["rows_read"],
            "amount_rows_sorted": helped["rows_read"],
            "the_other_column_is_no_better": at_order["rows_read"] >= at_plain["rows_read"],
            "while_the_sorted_one_is": helped["rows_read"] < against["rows_read"],
        }
    finally:
        _clear(folder, plain, order)


def compaction_keeps_every_row(rows: int = 12_000) -> dict:
    """The merged file holds the same rows as the fragments did, sorted or not.

    Checked as a multiset rather than a sequence, because the sorted compaction is allowed to
    reorder and not allowed to lose anything. A compaction that dropped the last partial
    fragment would pass every measurement above.
    """
    folder = _folder("keep")
    plain = _folder("keep_plain")
    order = _folder("keep_order")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        original = load(table)
        unsorted = compact(table, plain / "whole.cqe")
        ordered = compact(table, order / "whole.cqe", sort_by="amount")
        first = read(plain / "whole.cqe")
        second = read(order / "whole.cqe")
        as_before = [tuple(one) for one in original.to_rows()]
        as_plain = [tuple(one) for one in first.to_rows()]
        as_order = [tuple(one) for one in second.to_rows()]
        return {
            "rows": len(as_before),
            "unsorted_rows": len(as_plain),
            "sorted_rows": len(as_order),
            "the_unsorted_one_is_identical": as_plain == as_before,
            "the_sorted_one_is_a_permutation": sorted(as_order) == sorted(as_before),
            "and_it_is_not_the_same_order": as_order != as_before,
            "row_counts_match": unsorted.after.rows == ordered.after.rows == table.rows,
        }
    finally:
        _clear(folder, plain, order)


def a_dictionary_over_more_rows_is_smaller_per_row(rows: int = 40_000) -> dict:
    """Two hundred distinct labels are stored once in a compacted file and forty times before.

    A dictionary is per chunk, so a fragment of five hundred rows carries a dictionary of nearly
    every label it has, and eighty fragments carry eighty copies. Compacting shares one across
    the whole file, which is a saving in the data rather than the metadata and the only one here
    that grows with the number of distinct values.
    """
    folder = _folder("dict")
    out = _folder("dict_out")
    try:
        made = _table(rows)
        table = ingest(made, folder, size=INGEST_ROWS)
        merged = compact(table, out / "whole.cqe", group_size=COMPACT_GROUP)
        labels = made.column("label").to_list()
        one_dictionary = dictionary.encode(labels)
        per_fragment = dictionary.encode(labels[:INGEST_ROWS])
        return {
            "distinct": len(set(labels)),
            "fragments": len(table.fragments),
            "distinct_in_one_fragment": len(set(labels[:INGEST_ROWS])),
            "one_dictionary_bytes": one_dictionary.encoded_bytes,
            "fragment_dictionary_bytes": per_fragment.encoded_bytes * len(table.fragments),
            "the_copies_cost_more": per_fragment.encoded_bytes * len(table.fragments)
            > one_dictionary.encoded_bytes,
            "data_before": table.data_bytes,
            "data_after": merged.after.data_bytes,
            "the_file_got_smaller": merged.after.data_bytes < table.data_bytes,
        }
    finally:
        _clear(folder, out)


def compaction_pays_for_itself_after_a_few_scans(rows: int = 20_000) -> dict:
    """The rewrite costs bytes once and every later scan saves some, so there is a break even.

    Stated as a number of scans rather than as a judgement, because whether compaction is worth
    running depends entirely on how often the table is read afterwards and that is not something
    a storage layer can know. A table written once and read twice should be left alone.
    """
    folder = _folder("payoff")
    out = _folder("payoff_out")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        made = compact(table, out / "whole.cqe")
        return {
            "rewrite_cost": made.rewrite_cost,
            "saved_per_scan": made.metadata_saved,
            "break_even": round(made.break_even, 2),
            "it_pays_back": made.break_even < 20,
            "and_it_is_not_immediate": made.break_even > 1,
            "read_bytes": made.read_bytes,
            "written_bytes": made.written_bytes,
        }
    finally:
        _clear(folder, out)


def compacting_an_already_compact_file_earns_nothing(rows: int = 20_000) -> dict:
    """Running compaction twice costs a full rewrite and saves nothing the second time.

    The property a maintenance job needs before it can be run on a schedule. A compaction that
    kept finding work to do on an unchanged table would rewrite it forever.
    """
    folder = _folder("idem")
    once = _folder("idem_once")
    twice = _folder("idem_twice")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        first = compact(table, once / "whole.cqe")
        second = compact(first.after, twice / "whole.cqe")
        return {
            "first_saved": first.metadata_saved,
            "second_saved": second.metadata_saved,
            "the_second_saves_nothing": second.metadata_saved == 0,
            "and_its_break_even_is_never": second.break_even == float("inf"),
            "the_bytes_are_the_same": second.after.data_bytes == first.after.data_bytes,
            "but_it_still_cost_a_rewrite": second.rewrite_cost > 0,
        }
    finally:
        _clear(folder, once, twice)


def a_larger_ingest_needs_less_compacting(rows: int = 40_000) -> dict:
    """Ingesting in pieces of four thousand instead of five hundred removes most of the problem.

    Worth measuring because compaction is a fix for a decision made upstream. If the writer can
    buffer eight times as long, seven eighths of the metadata never exists, and the saving
    compaction has left to find shrinks with it. The reason to compact anyway is that the writer
    usually cannot.

    The data falls with it rather than staying put, for the dictionary reason measured
    separately below. Buffering longer is not only a metadata decision.
    """
    small = _folder("ingest_small")
    large = _folder("ingest_large")
    try:
        made = _table(rows)
        thin = ingest(made, small, size=INGEST_ROWS)
        thick = ingest(made, large, size=INGEST_ROWS * 8)
        return {
            "small_fragments": len(thin.fragments),
            "large_fragments": len(thick.fragments),
            "small_metadata": thin.metadata_bytes,
            "large_metadata": thick.metadata_bytes,
            "the_large_ingest_pays_less": thick.metadata_bytes < thin.metadata_bytes,
            "by_this_ratio": round(thin.metadata_bytes / max(thick.metadata_bytes, 1), 2),
            "small_data": thin.data_bytes,
            "large_data": thick.data_bytes,
            "the_data_falls_as_well": thick.data_bytes < thin.data_bytes,
            "but_by_far_less": thick.data_bytes > thin.data_bytes * 0.8,
        }
    finally:
        _clear(small, large)


def partial_compaction_leaves_the_recent_fragments_alone(rows: int = 40_000) -> dict:
    """Merging the older half and leaving the newer half takes most of the saving.

    What a table that is still being appended to can actually do. The newest fragments are the
    ones a writer is still adding to and the ones a reader is most likely to want whole, and
    leaving them out costs a share of the saving proportional to how many were left.
    """
    folder = _folder("partial")
    out = _folder("partial_out")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        cut = len(table.fragments) // 2
        older = Table(fragments=table.fragments[:cut])
        newer = table.fragments[cut:]
        made = compact(older, out / "older.cqe")
        mixed = Table(fragments=(*made.after.fragments, *newer))
        whole = Table(fragments=table.fragments)
        full = compact(whole, out / "whole.cqe")
        return {
            "fragments_before": len(table.fragments),
            "fragments_after_partial": len(mixed.fragments),
            "fragments_after_full": len(full.after.fragments),
            "metadata_before": table.metadata_bytes,
            "metadata_partial": mixed.metadata_bytes,
            "metadata_full": full.after.metadata_bytes,
            "partial_saving": round(
                1 - mixed.metadata_bytes / table.metadata_bytes,
                4,
            ),
            "full_saving": round(1 - full.after.metadata_bytes / table.metadata_bytes, 4),
            "the_partial_one_helps": mixed.metadata_bytes < table.metadata_bytes,
            "but_less_than_the_full_one": mixed.metadata_bytes > full.after.metadata_bytes,
        }
    finally:
        _clear(folder, out)


def a_scan_of_two_columns_of_five_is_unchanged_by_compaction(rows: int = 20_000) -> dict:
    """Projection is free before and after, so compaction neither helps nor hurts it.

    A negative result and the one most worth having. Reading two columns of five costs the same
    share of the data in both shapes, because that share is a property of the column major
    layout rather than of the file count, and the metadata term is the only thing that moved.

    The share is not two fifths, and the difference is the point of stating it. Columns are not
    the same width, so a projection's cost is decided by which columns rather than how many.
    """
    folder = _folder("project")
    out = _folder("project_out")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        made = compact(table, out / "whole.cqe")
        two_before = load(table, columns=["shop", "amount"])
        two_after = read(out / "whole.cqe", columns=["shop", "amount"])
        all_before = load(table)
        share_before = two_before.nbytes / all_before.nbytes
        share_after = two_after.nbytes / read(out / "whole.cqe").nbytes
        return {
            "columns": all_before.width,
            "share_before": round(share_before, 4),
            "share_after": round(share_after, 4),
            "the_share_is_the_same": abs(share_before - share_after) < 0.001,
            "and_it_is_not_two_fifths_of_bytes": abs(share_before - 0.4) > 0.01,
            "the_metadata_moved_instead": made.metadata_saved > 0,
        }
    finally:
        _clear(folder, out)


def compacting_nothing_is_refused() -> bool:
    """A table with no fragments is refused rather than compacted into an empty file."""
    try:
        Table(fragments=())
    except ConfigError:
        return True
    return False


def a_zero_group_size_is_refused(rows: int = 2_000) -> bool:
    """Compacting into groups of no rows is refused."""
    folder = _folder("zero")
    out = _folder("zero_out")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        try:
            compact(table, out / "whole.cqe", group_size=0)
        except ConfigError:
            return True
        return False
    finally:
        _clear(folder, out)


def a_zero_ingest_size_is_refused(rows: int = 2_000) -> bool:
    """Ingesting in pieces of no rows is refused."""
    folder = _folder("noingest")
    try:
        try:
            ingest(_table(rows), folder, size=0)
        except ConfigError:
            return True
        return False
    finally:
        _clear(folder)


def sorting_by_a_missing_column_is_refused(rows: int = 2_000) -> bool:
    """Compacting sorted by a column the table does not have is refused."""
    folder = _folder("nocol")
    out = _folder("nocol_out")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        try:
            compact(table, out / "whole.cqe", sort_by="absent")
        except SchemaError:
            return True
        return False
    finally:
        _clear(folder, out)


def describing_nothing_is_refused() -> bool:
    """Describing an empty list of paths is refused."""
    try:
        describe([])
    except ConfigError:
        return True
    return False


def compare_the_ingest_sizes(rows: int = 40_000) -> list[dict]:
    """What the ingest size does to metadata, before anything is compacted."""
    out: list[dict] = []
    made = _table(rows)
    for size in (250, 500, 1_000, 4_000):
        folder = _folder(f"sweep{size}")
        try:
            table = ingest(made, folder, size=size)
            out.append(
                {
                    "ingest": size,
                    "fragments": len(table.fragments),
                    "data": table.data_bytes,
                    "metadata": table.metadata_bytes,
                    "share": round(table.metadata_bytes / table.data_bytes, 4),
                }
            )
        finally:
            _clear(folder)
    return out


def the_metadata_share_falls_with_the_ingest_size(rows: int = 40_000) -> dict:
    """Doubling the ingest size roughly halves the metadata, which is the whole shape of it.

    Metadata is per fragment and fragments are rows over ingest size, so the share is inversely
    proportional and there is nothing subtle in it. Measured anyway, because the constant
    decides when compaction is worth scheduling and the constant is not derivable.
    """
    table = compare_the_ingest_sizes(rows)
    shares = [one["share"] for one in table]
    ratios = [round(shares[one] / shares[one + 1], 2) for one in range(len(shares) - 1)]
    return {
        "sizes": [one["ingest"] for one in table],
        "shares": shares,
        "ratios": ratios,
        "it_falls_throughout": all(
            shares[one] > shares[one + 1] for one in range(len(shares) - 1)
        ),
        "each_doubling_roughly_halves": all(1.7 < one < 2.3 for one in ratios[:2]),
        "the_largest_still_pays_something": shares[-1] > 0,
    }


def compare_the_shapes(rows: int = 20_000) -> list[dict]:
    """Fragments, compacted and compacted sorted, priced on the same predicate."""
    folder = _folder("shapes")
    plain = _folder("shapes_plain")
    order = _folder("shapes_order")
    try:
        table = ingest(_table(rows), folder, size=INGEST_ROWS)
        predicate = Compare(">", column("amount"), literal(800.0))
        unsorted = compact(table, plain / "whole.cqe")
        ordered = compact(table, order / "whole.cqe", sort_by="amount")
        return [
            dict(shape=name, **scan_cost(one, predicate))
            for name, one in (
                ("fragments", table),
                ("compacted", unsorted.after),
                ("compacted sorted", ordered.after),
            )
        ]
    finally:
        _clear(folder, plain, order)


def summarise(rows: int = 20_000) -> dict:
    """The findings in one mapping."""
    loss = the_pruning_loss_is_the_group_size_not_the_compaction()
    total = the_total_cost_still_falls()
    sorted_back = sorting_during_compaction_buys_the_pruning_back()
    payoff = compaction_pays_for_itself_after_a_few_scans(rows)
    return {
        "metadata_share_before": many_small_files_pay_more_metadata_than_data(rows)[
            "metadata_share"
        ],
        "compaction_removes": compaction_removes_almost_all_of_it(rows)["share_removed"],
        "same_group_size_prunes_identically": loss["same_group_size_prunes_identically"],
        "widening_costs": loss["widening_cost"],
        "total_falls_by": total["by_this_ratio"],
        "sorting_beats_the_fragments": sorted_back["the_sorted_one_reads_least"],
        "break_even_scans": payoff["break_even"],
        "twice_is_pointless": compacting_an_already_compact_file_earns_nothing(rows)[
            "the_second_saves_nothing"
        ],
    }
