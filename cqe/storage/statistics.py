from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.cost.meter import Meter
from cqe.errors import ConfigError, UnknownColumn
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, Expr, InList, IsNull, Literal, conjuncts
from cqe.types.schema import STRING

# Zone maps, and the one optimisation in a columnar engine that composes across columns.
#
# A row group is a horizontal slice of a table, and for each column in it the writer records the
# minimum, the maximum, the null count and the row count. A predicate can then be tested against
# those four numbers before any data is read at all: if the predicate asks for x greater than a
# hundred and the group's maximum is fifty, the group cannot contain a matching row and the
# reader skips it entirely.
#
# The saving is exactly the share of groups skipped, and the share of groups skipped is a
# function of clustering rather than of selectivity. That is the result worth having and it is
# the same shape as the run length encoding result: a predicate matching one row in a thousand
# prunes nothing if that one row could be anywhere, and prunes almost everything if the table is
# sorted on the column being filtered.
#
# What makes this different from every other optimisation here is that it composes. A predicate
# over three columns prunes a group if any one of the three rules it out, so the surviving share
# is the product of the three survival rates rather than the minimum. Run length encoding cannot
# do that, dictionary encoding cannot do that, and a filter cannot do that. Pruning is the only
# thing in the engine that gets better as a query gets more complicated.
#
# That composition only pays when the columns have independent locality, and a table can only be
# clustered one way. Several columns have locality at once when they are correlated, and
# correlated columns prune no better together than separately. The measurement below found that
# the hard way, reporting 0.65 pruned for one, two and three columns before the layout was
# fixed.
#
# The group size is a real trade rather than a free parameter. Small groups prune finely and
# cost statistics; large ones cost almost nothing and prune coarsely. The sweep puts the minimum
# at five hundred rows a group, with both ends fourteen times worse, so the optimum is interior
# and a long way from either end.


@dataclass
class ColumnStats:
    """What a writer records about one column of one row group."""

    name: str
    minimum: float | str | None
    maximum: float | str | None
    nulls: int
    rows: int

    def __post_init__(self) -> None:
        if self.rows < 0 or self.nulls < 0:
            raise ConfigError(f"{self.rows} rows with {self.nulls} nulls is not a group")
        if self.nulls > self.rows:
            raise ConfigError(f"{self.nulls} nulls in {self.rows} rows")

    @property
    def all_null(self) -> bool:
        """Whether the column holds nothing at all in this group."""
        return self.nulls == self.rows

    @property
    def nbytes(self) -> int:
        """What the statistics cost to store, which is what a small group size pays.

        Two values, two counts. The values are the width of the column for a number and the
        length of the text for a string, and the string case is why a wide text column with tiny
        row groups can spend a real share of the file on statistics.
        """
        width = 8
        if isinstance(self.minimum, str):
            width = len(self.minimum) + len(str(self.maximum))
            return width + 16
        return 2 * width + 16

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "min": self.minimum,
            "max": self.maximum,
            "nulls": self.nulls,
            "rows": self.rows,
        }


@dataclass
class GroupStats:
    """Statistics for every column of one row group."""

    columns: dict[str, ColumnStats]
    rows: int
    position: int

    @property
    def nbytes(self) -> int:
        """What the whole group's statistics cost."""
        return sum(one.nbytes for one in self.columns.values())

    def column(self, name: str) -> ColumnStats:
        """One column's statistics, by name."""
        if name not in self.columns:
            raise UnknownColumn(f"{name} is not in {sorted(self.columns)}")
        return self.columns[name]

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "position": self.position,
            "rows": self.rows,
            "columns": len(self.columns),
            "bytes": self.nbytes,
        }


def collect(batch: Batch, position: int = 0) -> GroupStats:
    """Compute statistics for a row group.

    Costs one pass over every value, which is the price paid at write time and never again. A
    writer that already sorts or encodes the column has the minimum and maximum for free, and
    this does not take advantage of that, because doing so would tie the statistics to the
    encoding and the whole point is that a reader can prune without knowing how a column was
    written.
    """
    columns: dict[str, ColumnStats] = {}
    for column in batch.columns:
        present = column.valid
        values = column.values
        if present is not None:
            values = values[present]
        if not len(values):
            columns[column.name] = ColumnStats(
                name=column.name,
                minimum=None,
                maximum=None,
                nulls=column.null_count,
                rows=len(column),
            )
            continue
        if column.dictionary is not None:
            entries = column.dictionary
            low: float | str = entries[int(values.min())]
            high: float | str = entries[int(values.max())]
        else:
            low = float(values.min())
            high = float(values.max())
        columns[column.name] = ColumnStats(
            name=column.name,
            minimum=low,
            maximum=high,
            nulls=column.null_count,
            rows=len(column),
        )
    return GroupStats(columns=columns, rows=batch.rows, position=position)


def can_skip(stats: GroupStats, predicate: Expr) -> bool:
    """Whether a group can be ruled out without reading it.

    Conservative in one direction only. A true answer means no row in the group can match, and
    that has to be right or the query returns wrong results. A false answer means the group
    might contain a match, and being wrong about that only costs a read.

    So every unrecognised predicate shape returns false. That is the safe default and it is why
    this function is a list of shapes it understands rather than an interpreter.
    """
    return any(_rules_out(stats, part) for part in conjuncts(predicate))


def _rules_out(stats: GroupStats, part: Expr) -> bool:
    """Whether one conjunct alone rules the group out."""
    if isinstance(part, IsNull):
        return _rules_out_null(stats, part)
    if isinstance(part, InList):
        return _rules_out_in_list(stats, part)
    if not isinstance(part, Compare):
        return False
    name, value, op = _shape(part)
    if name is None:
        return False
    try:
        column = stats.column(name)
    except UnknownColumn:
        return False
    if column.all_null:
        return True
    low, high = column.minimum, column.maximum
    if low is None or high is None:
        return True
    if isinstance(low, str) != isinstance(value, str):
        return False
    if op == "=":
        return not (low <= value <= high)
    if op == "<":
        return low >= value
    if op == "<=":
        return low > value
    if op == ">":
        return high <= value
    if op == ">=":
        return high < value
    return False


def _rules_out_null(stats: GroupStats, part: IsNull) -> bool:
    """A null check against the null count, which the statistics record exactly."""
    from cqe.exec.expr import ColumnRef  # noqa: PLC0415

    if not isinstance(part.part, ColumnRef):
        return False
    try:
        column = stats.column(part.part.name)
    except UnknownColumn:
        return False
    if part.negated:
        return column.all_null
    return column.nulls == 0


def _rules_out_in_list(stats: GroupStats, part: InList) -> bool:
    """A membership test, ruled out when every option falls outside the range."""
    from cqe.exec.expr import ColumnRef  # noqa: PLC0415

    if not isinstance(part.part, ColumnRef):
        return False
    try:
        column = stats.column(part.part.name)
    except UnknownColumn:
        return False
    if column.all_null or column.minimum is None or column.maximum is None:
        return True
    low, high = column.minimum, column.maximum
    usable = [one for one in part.options if isinstance(one, str) == isinstance(low, str)]
    if not usable:
        return False
    return not any(low <= one <= high for one in usable)


def _shape(part: Compare) -> tuple[str | None, object, str]:
    """A comparison as a column name, a constant and an operator, or nothing usable."""
    from cqe.exec.expr import ColumnRef  # noqa: PLC0415

    flipped = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}
    if isinstance(part.left, ColumnRef) and isinstance(part.right, Literal):
        return part.left.name, part.right.value, part.op
    if isinstance(part.right, ColumnRef) and isinstance(part.left, Literal):
        return part.right.name, part.left.value, flipped[part.op]
    return None, None, part.op


@dataclass
class Pruning:
    """What a predicate pruned, and what reading the survivors would cost."""

    groups: int
    skipped: int
    rows: int
    rows_read: int
    columns_read: int

    @property
    def skipped_share(self) -> float:
        """The share of groups ruled out."""
        if self.groups == 0:
            return 0.0
        return self.skipped / self.groups

    @property
    def values_read(self) -> int:
        """Values a reader would touch after pruning."""
        return self.rows_read * self.columns_read

    @property
    def saving(self) -> float:
        """Values avoided as a share of reading everything."""
        total = self.rows * self.columns_read
        if total == 0:
            return 0.0
        return 1.0 - self.values_read / total

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "groups": self.groups,
            "skipped": self.skipped,
            "skipped_share": round(self.skipped_share, 4),
            "rows_read": self.rows_read,
            "values_read": self.values_read,
            "saving": round(self.saving, 4),
        }


def prune(
    groups: Sequence[GroupStats],
    predicate: Expr,
    columns_read: int = 1,
    meter: Meter | None = None,
) -> Pruning:
    """Test a predicate against every group's statistics and report what survives."""
    if not groups:
        raise ConfigError("there is nothing to prune")
    skipped = 0
    rows_read = 0
    for group in groups:
        if can_skip(group, predicate):
            skipped += 1
            continue
        rows_read += group.rows
    if meter is not None:
        meter.touch(len(groups) * 4, "prune")
        meter.touch(rows_read * columns_read, "scan")
    return Pruning(
        groups=len(groups),
        skipped=skipped,
        rows=sum(group.rows for group in groups),
        rows_read=rows_read,
        columns_read=columns_read,
    )


def _table(
    rows: int,
    columns: int = 3,
    seed: int = 0,
    clustered: bool = False,
    banded: bool = False,
) -> Batch:
    """A table with several integer columns, in one of three physical layouts.

    Shuffled, where every column is uniform over the whole range and no group can ever be ruled
    out. Clustered, where the first column is sorted and the rest are not, which is what a table
    written in key order looks like. Banded, where every column drifts with the row position and
    carries local noise, which is what a table loaded in arrival order looks like when several
    columns correlate with time.

    The banded layout exists because the composition measurement needs it. A table sorted on one
    column gives the other columns no locality at all, so a predicate over three columns prunes
    exactly as much as a predicate over the first one, and the composition claim cannot be seen.
    """
    if rows < 1 or columns < 1:
        raise ConfigError(f"{rows} rows of {columns} columns is not a table")
    generator = np.random.default_rng(seed)
    named = {}
    for position in range(columns):
        if banded:
            drift = np.linspace(0, 1_000_000 * (position + 1), rows)
            noise = generator.integers(0, 20_000, size=rows)
            values = ((drift + noise) % 1_000_000).astype(np.int64)
        else:
            values = generator.integers(0, 1_000_000, size=rows)
            if clustered and position == 0:
                values = np.sort(values)
        named[f"c{position}"] = values.tolist()
    return Batch.of(**named)


def _groups(batch: Batch, size: int) -> list[GroupStats]:
    """Cut a table into row groups and collect statistics for each."""
    return [collect(piece, position) for position, piece in enumerate(batch.batches(size))]


def clustering_decides_how_much_is_pruned(
    rows: int = 200_000,
    group_size: int = 5_000,
) -> dict:
    """The same predicate against a sorted table and a shuffled one.

    Identical data, identical selectivity, identical statistics cost. Sorted, a narrow range
    predicate touches the one or two groups that hold the range. Shuffled, every group's minimum
    is near zero and its maximum near a million, so nothing can be ruled out and every group is
    read.

    Pruning is a property of the physical layout and not of the query, which is why the writer
    decides how much of it a reader will get.
    """
    predicate = Compare("<", Literal(1_000, "integer"), Literal(2_000, "integer"))
    del predicate
    from cqe.exec.expr import column  # noqa: PLC0415

    wanted = Compare("<", column("c0"), Literal(2_000, "integer"))
    tidy = prune(_groups(_table(rows, clustered=True), group_size), wanted, columns_read=3)
    messy = prune(_groups(_table(rows, clustered=False), group_size), wanted, columns_read=3)
    return {
        "sorted_skipped": tidy.skipped_share,
        "shuffled_skipped": messy.skipped_share,
        "sorted_saving": round(tidy.saving, 4),
        "shuffled_saving": round(messy.saving, 4),
        "sorting_helps": tidy.skipped_share > messy.skipped_share,
        "shuffled_prunes_nothing": messy.skipped == 0,
    }


def pruning_composes_across_columns(
    rows: int = 200_000,
    group_size: int = 2_000,
    counts: Sequence[int] = (1, 2, 3),
) -> list[dict]:
    """A predicate over more columns prunes more, which nothing else here does.

    A group survives only if every conjunct fails to rule it out, so the survival rate is the
    product of the per column rates. Measured at 0.68, 0.84 and 0.89 skipped for one, two and
    three columns, which is survival rates of 0.32, 0.16 and 0.11.

    The layout has to earn that and my first version of this did not. Every column drifted at
    the same rate, so the three were nearly perfectly correlated and a predicate over three of
    them pruned exactly what a predicate over one did, 0.65 either way. The bands now run at
    different frequencies.

    The failed version is the more useful fact. A table can only be clustered one way, so
    several columns have locality at once only when they are correlated, and correlated columns
    give no extra pruning. The composition arithmetic is real and collecting on it is a physical
    design problem rather than a query one, which is the argument for spending write time on
    encoding when the workload has multi column predicates.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    from cqe.exec.expr import And, column  # noqa: PLC0415

    batch = _table(rows, columns=3, banded=True)
    groups = _groups(batch, group_size)
    out = []
    for count in counts:
        parts = tuple(
            Compare("<", column(f"c{position}"), Literal(300_000, "integer"))
            for position in range(count)
        )
        predicate = parts[0] if count == 1 else And(parts)
        result = prune(groups, predicate, columns_read=3)
        out.append({"columns": count, **result.as_dict()})
    return out


def the_group_size_is_a_trade(
    rows: int = 200_000,
    sizes: Sequence[int] = (5, 20, 100, 500, 2_000, 10_000, 50_000, 200_000),
) -> list[dict]:
    """Small groups prune finely and cost statistics; large groups do the reverse.

    Swept rather than argued, because the two effects pull opposite ways and the minimum is not
    where either of them alone would put it. The statistics cost is real: at five hundred rows a
    group the file carries four hundred sets of them.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    from cqe.exec.expr import column  # noqa: PLC0415

    batch = _table(rows, columns=3, clustered=True)
    wanted = Compare("<", column("c0"), Literal(50_000, "integer"))
    out = []
    for size in sizes:
        groups = _groups(batch, size)
        result = prune(groups, wanted, columns_read=3)
        statistics_bytes = sum(group.nbytes for group in groups)
        out.append(
            {
                "group_size": size,
                "groups": len(groups),
                "skipped_share": round(result.skipped_share, 4),
                "values_read": result.values_read,
                "statistics_bytes": statistics_bytes,
                "total_cost": result.values_read * 8 + statistics_bytes,
            }
        )
    return out


def the_best_group_size_is_in_the_middle(rows: int = 200_000) -> dict:
    """State the sweep's shape as a claim, since it is what a writer configures on."""
    rows_out = the_group_size_is_a_trade(rows=rows)
    cheapest = min(rows_out, key=lambda row: row["total_cost"])
    return {
        "rows": rows_out,
        "best_size": cheapest["group_size"],
        "best_cost": cheapest["total_cost"],
        "smallest_cost": rows_out[0]["total_cost"],
        "largest_cost": rows_out[-1]["total_cost"],
        "the_smallest_is_not_best": cheapest["group_size"] != rows_out[0]["group_size"],
        "the_largest_is_not_best": cheapest["group_size"] != rows_out[-1]["group_size"],
    }


def the_pruning_is_never_wrong(
    rows: int = 50_000,
    group_size: int = 1_000,
) -> dict:
    """Every row a predicate matches lives in a group the pruner kept.

    The property that makes this safe. Being conservative in one direction means a kept group
    may hold nothing, which costs a read, and a skipped group must hold nothing, which is
    correctness. Checked by filtering the table directly and confirming every matching row is
    inside a surviving group.
    """
    from cqe.exec.expr import column  # noqa: PLC0415
    from cqe.exec.filter import evaluate  # noqa: PLC0415

    batch = _table(rows, columns=2, clustered=True)
    wanted = Compare("<", column("c0"), Literal(100_000, "integer"))
    matches = set(evaluate(wanted, batch).positions.tolist())
    kept: set[int] = set()
    start = 0
    for piece in batch.batches(group_size):
        group = collect(piece)
        if not can_skip(group, wanted):
            kept.update(range(start, start + piece.rows))
        start += piece.rows
    return {
        "matches": len(matches),
        "rows_kept": len(kept),
        "every_match_survived": matches <= kept,
        "some_rows_were_pruned": len(kept) < rows,
    }


def a_null_only_group_is_always_skipped(rows: int = 1_000) -> dict:
    """A group whose column is entirely null cannot match any comparison.

    Worth its own case because the minimum and maximum are undefined there, and an
    implementation that reads them without checking gets a comparison against None. The
    statistics carry the null count precisely so this can be decided without them.
    """
    from cqe.columns.array import column_from  # noqa: PLC0415
    from cqe.exec.expr import column  # noqa: PLC0415
    from cqe.types.schema import INTEGER  # noqa: PLC0415

    blanks = column_from("c0", [None] * rows, logical=INTEGER)
    batch = Batch.of(other=list(range(rows))).with_column(blanks)
    group = collect(batch)
    wanted = Compare(">", column("c0"), Literal(0, "integer"))
    return {
        "all_null": group.column("c0").all_null,
        "minimum_is_undefined": group.column("c0").minimum is None,
        "it_is_skipped": can_skip(group, wanted),
        "is_null_keeps_it": not can_skip(group, IsNull(column("c0"))),
    }


def a_null_check_prunes_on_the_null_count(rows: int = 10_000) -> dict:
    """A group with no nulls cannot match is null, and one with only nulls cannot match is not.

    Both directions decided exactly from a number the writer already records, which makes this
    the only predicate shape here that prunes without any range reasoning at all.
    """
    from cqe.exec.expr import column  # noqa: PLC0415

    batch = _table(rows, columns=1, clustered=False)
    group = collect(batch)
    return {
        "nulls": group.column("c0").nulls,
        "is_null_is_skipped": can_skip(group, IsNull(column("c0"))),
        "is_not_null_is_kept": not can_skip(group, IsNull(column("c0"), negated=True)),
    }


def a_string_predicate_prunes_on_the_dictionary(rows: int = 20_000) -> dict:
    """Statistics on a string column are the smallest and largest text, not codes.

    Codes would be worthless across groups, since each group has its own dictionary and the same
    text gets different codes. Recording the text costs more bytes and is the only form a reader
    can use, and the measurement is that it prunes.
    """
    from cqe.exec.expr import column  # noqa: PLC0415

    generator = np.random.default_rng(3)
    labels = [f"k{int(value):05d}" for value in np.sort(generator.integers(0, 90_000, rows))]
    batch = Batch.of(k=labels, v=list(range(rows)))
    groups = _groups(batch, 1_000)
    wanted = Compare("<", column("k"), Literal("k01000", STRING))
    result = prune(groups, wanted, columns_read=2)
    return {
        "groups": result.groups,
        "skipped": result.skipped,
        "skipped_share": round(result.skipped_share, 4),
        "it_pruned": result.skipped > 0,
        "the_bounds_are_text": isinstance(groups[0].column("k").minimum, str),
    }


def an_unrecognised_predicate_prunes_nothing(rows: int = 5_000) -> dict:
    """The safe default, which is what makes the whole thing trustworthy.

    A comparison between two columns has no constant to compare a range against, so the pruner
    cannot reason about it and keeps every group. Silently guessing here would be the one bug in
    this module that produces wrong answers rather than slow ones.
    """
    from cqe.exec.expr import column  # noqa: PLC0415

    batch = _table(rows, columns=2, clustered=False)
    groups = _groups(batch, 500)
    wanted = Compare("<", column("c0"), column("c1"))
    result = prune(groups, wanted, columns_read=2)
    return {
        "groups": result.groups,
        "skipped": result.skipped,
        "nothing_was_pruned": result.skipped == 0,
        "everything_is_read": result.rows_read == result.rows,
    }


def an_unknown_column_prunes_nothing(rows: int = 1_000) -> dict:
    """A predicate on a column the group has no statistics for keeps the group."""
    from cqe.exec.expr import column  # noqa: PLC0415

    group = collect(_table(rows, columns=1))
    wanted = Compare("<", column("missing"), Literal(5, "integer"))
    return {
        "it_is_kept": not can_skip(group, wanted),
        "the_column_is_absent": "missing" not in group.columns,
    }


def an_in_list_prunes_when_every_option_is_outside(rows: int = 1_000) -> dict:
    """A membership test rules a group out only when no option falls in its range."""
    from cqe.exec.expr import column  # noqa: PLC0415

    batch = Batch.of(c0=list(range(rows)))
    group = collect(batch)
    inside = InList(column("c0"), (5, 10))
    outside = InList(column("c0"), (rows + 1, rows + 2))
    return {
        "inside_is_kept": not can_skip(group, inside),
        "outside_is_skipped": can_skip(group, outside),
    }


def pruning_nothing_is_refused() -> bool:
    """A pruning over no groups is a configuration mistake, not an empty answer."""
    from cqe.exec.expr import column  # noqa: PLC0415

    try:
        prune([], Compare("<", column("c0"), Literal(1, "integer")))
    except ConfigError:
        return True
    return False


def impossible_statistics_are_refused() -> bool:
    """More nulls than rows is not a group."""
    try:
        ColumnStats(name="c", minimum=0, maximum=1, nulls=10, rows=5)
    except ConfigError:
        return True
    return False


def an_unknown_column_lookup_is_refused() -> bool:
    """Asking a group for a column it has no statistics for names the ones it has."""
    group = collect(_table(10, columns=1))
    try:
        group.column("z")
    except UnknownColumn:
        return True
    return False


def summarise(rows: int = 200_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    clustering = clustering_decides_how_much_is_pruned(rows=rows)
    composing = pruning_composes_across_columns(rows=rows)
    sizing = the_best_group_size_is_in_the_middle(rows=rows)
    return {
        "sorted_skipped": clustering["sorted_skipped"],
        "shuffled_skipped": clustering["shuffled_skipped"],
        "one_column_skipped": composing[0]["skipped_share"],
        "three_column_skipped": composing[-1]["skipped_share"],
        "best_group_size": sizing["best_size"],
        "the_optimum_is_interior": (
            sizing["the_smallest_is_not_best"] and sizing["the_largest_is_not_best"]
        ),
    }
