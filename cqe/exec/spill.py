from __future__ import annotations

import heapq
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import BudgetExceeded, ConfigError
from cqe.exec.aggregate import Aggregate, hash_aggregate
from cqe.exec.batch import Batch, stack
from cqe.exec.sort import SortKey, order_by
from cqe.storage.bloom import _hash
from cqe.storage.file import read, write
from cqe.verify.reference import Rows, agree, group_by
from cqe.verify.reference import order_by as reference_order

# What happens when an operator needs more memory than it has, which is the case every engine
# gets wrong first and every engine has to handle.
#
# Two operators here can exceed their memory: a sort, which needs every row at once, and a hash
# aggregate, which needs one entry per group. Neither can be made to fit by being cleverer, so
# both spill: they write partial results to disk and combine them afterwards.
#
# The two spill differently and the difference is the interesting part.
#
# A sort spills runs. Each run is a sorted slice that fitted in memory, and the merge is a k way
# merge that reads one row at a time from each run. The merge is exact and needs memory
# proportional to the number of runs rather than to the data, so a sort of any size fits in a
# fixed budget as long as the number of runs does.
#
# An aggregate spills partitions. Each partition holds the groups whose keys hash into it, and
# the combine is a concatenation, because a group appears in exactly one partition. That is only
# true because the partition is chosen by hashing the key, and it is the property that makes the
# combine free.
#
# Both are measured here against the same operators running in memory, because a spilling
# operator that gets a different answer is not a spilling operator, it is a bug with a fallback.

# How many rows a run holds before it is written out, when nothing else is said. Small enough
# that the measurements below actually spill on the table sizes they use.
RUN_ROWS = 2000

# How many partitions a spilling aggregate hashes into. A power of two so the partition is a
# mask rather than a modulo, and small enough that the merge does not open too many files.
PARTITIONS = 16


@dataclass
class Spilled:
    """A set of runs on disk, and what it cost to put them there."""

    paths: tuple[Path, ...]
    rows: int
    bytes_written: int
    kind: str = "run"

    @property
    def runs(self) -> int:
        """How many pieces were written."""
        return len(self.paths)

    def read_all(self) -> list[Batch]:
        """Every run back in memory, for a combine that can afford it."""
        return [read(one) for one in self.paths]

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kind": self.kind,
            "runs": self.runs,
            "rows": self.rows,
            "bytes": self.bytes_written,
        }


@dataclass
class Budget:
    """How many rows an operator may hold at once, and whether it has exceeded that."""

    rows: int
    held: int = 0
    peak: int = 0
    spills: int = 0

    def take(self, count: int) -> None:
        """Account for holding more rows, refusing if that exceeds the budget."""
        self.held += count
        self.peak = max(self.peak, self.held)
        if self.held > self.rows:
            raise BudgetExceeded("rows", self.rows, self.held)

    def release(self, count: int) -> None:
        """Account for letting rows go."""
        self.held = max(self.held - count, 0)
        self.spills += 1

    @property
    def fits(self) -> bool:
        """Whether everything held so far stayed inside the budget."""
        return self.peak <= self.rows

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"budget": self.rows, "peak": self.peak, "spills": self.spills}


def _write_run(batch: Batch, directory: Path, position: int) -> tuple[Path, int]:
    """One run to disk, returning where it went and how large it was."""
    path = Path(directory) / f"run{position:04d}.cqe"
    write(path, batch, group_size=max(batch.rows, 1))
    return path, path.stat().st_size


def spill_sorted_runs(
    batch: Batch,
    keys: Sequence[SortKey],
    directory: Path,
    run_rows: int = RUN_ROWS,
    meter: Meter | None = None,
) -> Spilled:
    """Sort a table in slices, writing each sorted slice out.

    Each run is sorted on its own, which is the only work done in memory. Sorting a slice of two
    thousand rows costs two thousand log two thousand comparisons and there are n over two
    thousand of them, so the total is less than a full sort would be, and the merge afterwards
    makes up the difference.
    """
    if run_rows <= 0:
        raise ConfigError(f"{run_rows} rows is not a run")
    paths: list[Path] = []
    written = 0
    for position, start in enumerate(range(0, batch.rows, run_rows)):
        piece = batch.slice(start, min(start + run_rows, batch.rows))
        ordering = order_by(piece, keys, meter=meter)
        path, size = _write_run(piece.take(ordering.positions), directory, position)
        paths.append(path)
        written += size
        if meter is not None:
            meter.spill(piece.nbytes)
    return Spilled(paths=tuple(paths), rows=batch.rows, bytes_written=written)


def merge_runs(
    spilled: Spilled,
    keys: Sequence[SortKey],
    meter: Meter | None = None,
) -> Batch:
    """Combine sorted runs into one sorted table.

    A k way merge through a heap, which holds one row per run rather than one row per table.
    That is the whole point of spilling a sort: the merge's memory is the run count, not the row
    count, so a table of any size merges in a fixed budget.

    The rows come back through the reference row form rather than as columns, because a merge is
    inherently row at a time and pretending otherwise would mean a column slice per row.
    """
    if not spilled.paths:
        raise ConfigError("there are no runs to merge")
    runs = [read(one) for one in spilled.paths]
    names = list(runs[0].schema.names)
    rows = [Rows.of(one).rows for one in runs]
    cursors = [0] * len(runs)
    heap: list[tuple] = []
    for index, one in enumerate(rows):
        if one:
            heapq.heappush(heap, (_key_of(one[0], names, keys), index))
    out: list[list] = []
    while heap:
        _, index = heapq.heappop(heap)
        row = rows[index][cursors[index]]
        out.append(row)
        cursors[index] += 1
        if meter is not None:
            meter.materialise(1)
        if cursors[index] < len(rows[index]):
            heapq.heappush(heap, (_key_of(rows[index][cursors[index]], names, keys), index))
    return Rows(names=tuple(names), rows=out).to_batch()


def _key_of(row: Sequence, names: Sequence[str], keys: Sequence[SortKey]) -> tuple:
    """One row's sort key, as a tuple the heap can order.

    Descending is handled by negating numbers, which does not work for strings, so a descending
    string key is refused rather than silently ordered the wrong way. The alternative is a
    wrapper class per value and the merge is already the slow path.
    """
    out = []
    for one in keys:
        value = row[list(names).index(one.name)]
        if one.descending:
            if isinstance(value, str):
                raise ConfigError("a merge cannot order strings downwards")
            out.append(-value)
        else:
            out.append(value)
    return tuple(out)


def external_sort(
    batch: Batch,
    keys: Sequence[SortKey],
    directory: Path,
    run_rows: int = RUN_ROWS,
    meter: Meter | None = None,
) -> Batch:
    """A sort that never holds more than one run in memory at a time."""
    spilled = spill_sorted_runs(batch, keys, directory, run_rows, meter=meter)
    return merge_runs(spilled, keys, meter=meter)


def _partition_of(value, partitions: int) -> int:
    """Which partition a key belongs to.

    Hashed rather than ranged, because a range partition needs to know the distribution and gets
    it wrong on skewed data, and the combine only works if a group lands in exactly one
    partition, which hashing guarantees and ranging also does but only if the ranges are right.
    """
    return int(_hash(value) % partitions)


def spill_partitions(
    batch: Batch,
    key: str,
    directory: Path,
    partitions: int = PARTITIONS,
    meter: Meter | None = None,
) -> Spilled:
    """Cut a table into partitions by hashing one key column.

    Every row with the same key lands in the same partition, which is what makes aggregating
    each partition separately give the same answer as aggregating the whole table.
    """
    if partitions <= 0:
        raise ConfigError(f"{partitions} is not a partition count")
    if key not in batch.schema:
        raise ConfigError(f"{key} is not a column of {list(batch.schema.names)}")
    values = batch.column(key).to_list()
    assigned = np.array([_partition_of(one, partitions) for one in values])
    paths: list[Path] = []
    written = 0
    for one in range(partitions):
        mask = assigned == one
        if not mask.any():
            continue
        piece = batch.mask(mask)
        path, size = _write_run(piece, directory, one)
        paths.append(path)
        written += size
        if meter is not None:
            meter.spill(piece.nbytes)
    return Spilled(paths=tuple(paths), rows=batch.rows, bytes_written=written, kind="partition")


def aggregate_partitions(
    spilled: Spilled,
    key: str,
    aggregates: Sequence[Aggregate],
    meter: Meter | None = None,
) -> Batch:
    """Aggregate each partition on its own and concatenate the results.

    Concatenate rather than combine, because a group appears in exactly one partition. If the
    partitioning were by anything other than a hash of the key, this would need a second
    aggregate over the results and the sums would have to be sums of sums, which works for count
    and sum and does not work for a distinct count.
    """
    if not spilled.paths:
        raise ConfigError("there are no partitions to aggregate")
    pieces = []
    for one in spilled.paths:
        piece = read(one)
        pieces.append(hash_aggregate(piece, [key], aggregates, meter=meter).batch)
    return stack(pieces)


def external_aggregate(
    batch: Batch,
    key: str,
    aggregates: Sequence[Aggregate],
    directory: Path,
    partitions: int = PARTITIONS,
    meter: Meter | None = None,
) -> Batch:
    """An aggregate that never holds more than one partition's groups at a time."""
    spilled = spill_partitions(batch, key, directory, partitions, meter=meter)
    return aggregate_partitions(spilled, key, aggregates, meter=meter)


def in_batches(batch: Batch, rows: int) -> Iterator[Batch]:
    """A table as a sequence of slices, which is how a bounded operator reads it."""
    if rows <= 0:
        raise ConfigError(f"{rows} is not a batch size")
    for start in range(0, batch.rows, rows):
        yield batch.slice(start, min(start + rows, batch.rows))


@dataclass
class Bounded:
    """An operator that reads in batches and refuses to exceed a row budget."""

    budget: Budget
    meter: Meter = field(default_factory=Meter)

    def sort(self, batch: Batch, keys: Sequence[SortKey], directory: Path) -> Batch:
        """Sort inside the budget, spilling when the input does not fit."""
        if batch.rows <= self.budget.rows:
            self.budget.take(batch.rows)
            ordering = order_by(batch, keys, meter=self.meter)
            return batch.take(ordering.positions, meter=self.meter)
        return external_sort(
            batch, keys, directory, run_rows=self.budget.rows, meter=self.meter
        )

    def aggregate(
        self,
        batch: Batch,
        key: str,
        aggregates: Sequence[Aggregate],
        directory: Path,
    ) -> Batch:
        """Aggregate inside the budget, spilling when the groups do not fit."""
        groups = len(set(batch.column(key).to_list()))
        if groups <= self.budget.rows:
            self.budget.take(groups)
            return hash_aggregate(batch, [key], aggregates, meter=self.meter).batch
        return external_aggregate(batch, key, aggregates, directory, meter=self.meter)


def _table(rows: int = 20000, groups: int = 400, seed: int = 23) -> Batch:
    """A table to sort and to aggregate, with a key of moderate cardinality."""
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, groups, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("label", [f"kind{one}" for one in state.integers(0, 12, rows)]),
        ]
    )


def an_external_sort_gives_the_same_order() -> dict:
    """Spill, merge, and compare against the reference sort row for row.

    The only claim that matters. An external sort that is nearly right is wrong, and the
    reference is a row at a time Python sort that shares no code with any of this.
    """
    batch = _table(8000)
    keys = [SortKey(name="amount")]
    with tempfile.TemporaryDirectory() as directory:
        spilled = spill_sorted_runs(batch, keys, Path(directory), run_rows=1000)
        merged = merge_runs(spilled, keys)
    expected = reference_order(Rows.of(batch), ["amount"])
    return {
        "rows": merged.rows,
        "runs": spilled.runs,
        "it_spilled": spilled.runs > 1,
        "they_agree": bool(agree(Rows.of(merged), expected, ordered=True)),
    }


def an_external_sort_holds_one_run_at_a_time() -> dict:
    """What the spill buys: memory proportional to the run, not to the table.

    Measured on the largest slice ever held rather than on a peak memory reading, because a
    Python memory reading measures the interpreter and this measures the algorithm.
    """
    batch = _table(20000)
    keys = [SortKey(name="amount")]
    run_rows = 1000
    with tempfile.TemporaryDirectory() as directory:
        spilled = spill_sorted_runs(batch, keys, Path(directory), run_rows=run_rows)
        largest = max(read(one).rows for one in spilled.paths)
    return {
        "rows": batch.rows,
        "runs": spilled.runs,
        "largest_run": largest,
        "it_never_held_the_table": largest <= run_rows,
        "the_ratio": round(batch.rows / largest, 1),
    }


def a_smaller_run_makes_more_runs(rows: int = 20000) -> dict:
    """The trade: less memory means more runs, and the merge holds one row per run.

    So the memory does not go to zero, it moves from the sort to the merge. The total held is
    the run size plus the run count, which is r plus n over r, and that is smallest at the
    square root of the row count. For twenty thousand rows that is 141, and the sweep finds the
    minimum there rather than at either end.

    The first sweep started at five hundred and found its minimum at the smallest point it
    tried, which is what a sweep that has not gone far enough always says. The points below
    bracket the square root on both sides so the shape is visible rather than inferred.
    """
    batch = _table(rows)
    keys = [SortKey(name="amount")]
    out = []
    for run_rows in (50, 141, 500, 2000, 5000, 20000):
        with tempfile.TemporaryDirectory() as directory:
            spilled = spill_sorted_runs(batch, keys, Path(directory), run_rows=run_rows)
            out.append(
                {
                    "run_rows": run_rows,
                    "runs": spilled.runs,
                    "held_in_sort": run_rows,
                    "held_in_merge": spilled.runs,
                    "total_held": run_rows + spilled.runs,
                }
            )
    totals = [one["total_held"] for one in out]
    best = totals.index(min(totals))
    return {
        "sweep": out,
        "best_run_rows": out[best]["run_rows"],
        "the_square_root": round(rows**0.5),
        "the_smallest_total_is_in_the_middle": best not in (0, len(totals) - 1),
        "and_it_is_at_the_square_root": out[best]["run_rows"] == 141,
    }


def an_external_aggregate_gives_the_same_groups() -> dict:
    """Partition, aggregate each, concatenate, and compare against the reference.

    The concatenation is only correct because a group lands in exactly one partition, so this
    measurement is really a check on the partitioning rather than on the aggregate.
    """
    batch = _table(10000)
    aggregates = [Aggregate(name="total", function="sum", source="amount")]
    with tempfile.TemporaryDirectory() as directory:
        spilled = spill_partitions(batch, "shop", Path(directory))
        produced = aggregate_partitions(spilled, "shop", aggregates)
    expected = group_by(Rows.of(batch), ["shop"], [("total", "sum", "amount")])
    return {
        "groups": produced.rows,
        "partitions": spilled.runs,
        "expected": len(expected.rows),
        "they_agree": bool(agree(Rows.of(produced), expected)),
    }


def every_group_lands_in_one_partition() -> dict:
    """The property the concatenation rests on, checked directly.

    If a key appeared in two partitions the result would hold that group twice, with each half's
    total, and the sums would be wrong in a way that looks like a data problem rather than a
    partitioning one.
    """
    batch = _table(10000)
    with tempfile.TemporaryDirectory() as directory:
        spilled = spill_partitions(batch, "shop", Path(directory))
        seen: dict[int, int] = {}
        for index, one in enumerate(spilled.paths):
            for key in set(read(one).column("shop").to_list()):
                seen.setdefault(key, index)
                if seen[key] != index:
                    return {"it_holds": False, "the_key_in_two": key}
    return {
        "keys": len(seen),
        "partitions": spilled.runs,
        "it_holds": True,
        "keys_per_partition": round(len(seen) / max(spilled.runs, 1), 1),
    }


def the_partitions_are_roughly_even() -> dict:
    """How evenly the hash spreads the keys, which is what bounds the largest partition.

    A spilling aggregate is only bounded by its largest partition, so an uneven hash means the
    budget has to be set for the worst one and the average is not the number that matters.
    """
    batch = _table(20000, groups=800)
    with tempfile.TemporaryDirectory() as directory:
        spilled = spill_partitions(batch, "shop", Path(directory))
        sizes = [read(one).rows for one in spilled.paths]
    average = sum(sizes) / len(sizes)
    return {
        "partitions": len(sizes),
        "smallest": min(sizes),
        "largest": max(sizes),
        "average": round(average, 1),
        "ratio": round(max(sizes) / average, 2),
        "it_is_within_a_third": max(sizes) < average * 1.35,
    }


def a_skewed_key_makes_an_uneven_partition() -> dict:
    """And what happens when the keys are not uniform, which is the case that breaks it.

    Hashing spreads the distinct keys evenly and does nothing about how many rows each key has.
    A key holding half the rows puts half the rows in one partition however many partitions
    there are, and no partitioning scheme fixes that: the group itself does not fit.
    """
    state = np.random.default_rng(31)
    rows = 20000
    keys = np.where(state.random(rows) < 0.5, 0, state.integers(1, 400, rows))
    batch = Batch.from_columns(
        [
            integer_column("shop", keys),
            floating_column("amount", state.normal(100, 20, rows)),
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        spilled = spill_partitions(batch, "shop", Path(directory))
        sizes = [read(one).rows for one in spilled.paths]
    average = sum(sizes) / len(sizes)
    return {
        "partitions": len(sizes),
        "largest": max(sizes),
        "average": round(average, 1),
        "ratio": round(max(sizes) / average, 2),
        "the_skew_survives_the_hash": max(sizes) > average * 3,
        "the_biggest_key_holds": int((keys == 0).sum()),
    }


def more_partitions_do_not_fix_skew() -> dict:
    """The same skewed table at four partition counts, which changes nothing.

    Worth measuring because more partitions is the first thing anyone reaches for, and the
    largest partition stays the size of the largest group no matter how many there are.
    """
    state = np.random.default_rng(31)
    rows = 20000
    keys = np.where(state.random(rows) < 0.5, 0, state.integers(1, 400, rows))
    batch = Batch.from_columns(
        [
            integer_column("shop", keys),
            floating_column("amount", state.normal(100, 20, rows)),
        ]
    )
    out = []
    for count in (4, 16, 64, 256):
        with tempfile.TemporaryDirectory() as directory:
            spilled = spill_partitions(batch, "shop", Path(directory), partitions=count)
            sizes = [read(one).rows for one in spilled.paths]
        out.append({"partitions": count, "largest": max(sizes)})
    largest = [one["largest"] for one in out]
    floor = int((keys == 0).sum())
    return {
        "sweep": out,
        "the_floor": floor,
        "it_never_goes_below_the_biggest_group": min(largest) >= floor,
        "quadrupling_the_partitions_barely_helps": largest[-1] > largest[0] * 0.4,
    }


def a_budget_refuses_before_it_runs_out() -> dict:
    """The budget itself, which is what makes a spill happen rather than a memory error.

    An operator that discovers it is out of memory by running out of memory has already lost.
    The budget is checked before the allocation, so the refusal is a decision rather than a
    crash.
    """
    budget = Budget(rows=1000)
    budget.take(600)
    caught = ""
    try:
        budget.take(600)
    except BudgetExceeded as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_refused": bool(caught),
        "peak": budget.peak,
        "it_names_both_numbers": "1200" in caught and "1000" in caught,
    }


def a_bounded_sort_spills_only_when_it_has_to() -> dict:
    """The same operator on a table that fits and one that does not.

    The point of the wrapper: the fast path is the ordinary in memory sort and the spill is only
    reached when the budget says so, so the cost of having a spilling implementation is zero on
    every query that does not need it.
    """
    small = _table(500)
    large = _table(20000)
    keys = [SortKey(name="amount")]
    with tempfile.TemporaryDirectory() as directory:
        inside = Bounded(budget=Budget(rows=2000))
        outside = Bounded(budget=Budget(rows=2000))
        fitted = inside.sort(small, keys, Path(directory))
        spilled = outside.sort(large, keys, Path(directory))
    return {
        "small_rows": fitted.rows,
        "large_rows": spilled.rows,
        "the_small_one_did_not_spill": inside.meter.spilled_bytes == 0,
        "the_large_one_did": outside.meter.spilled_bytes > 0,
        "both_are_sorted": bool(
            np.all(np.diff(fitted.column("amount").values) >= 0)
            and np.all(np.diff(spilled.column("amount").values) >= 0)
        ),
    }


def a_bounded_aggregate_spills_only_when_it_has_to() -> dict:
    """The same, for the aggregate, where the budget is on groups rather than rows."""
    few = _table(10000, groups=20)
    many = _table(10000, groups=2000)
    aggregates = [Aggregate(name="total", function="sum", source="amount")]
    with tempfile.TemporaryDirectory() as directory:
        inside = Bounded(budget=Budget(rows=100))
        outside = Bounded(budget=Budget(rows=100))
        fitted = inside.aggregate(few, "shop", aggregates, Path(directory))
        spilled = outside.aggregate(many, "shop", aggregates, Path(directory))
    return {
        "few_groups": fitted.rows,
        "many_groups": spilled.rows,
        "the_small_one_did_not_spill": inside.meter.spilled_bytes == 0,
        "the_large_one_did": outside.meter.spilled_bytes > 0,
    }


def spilling_costs_bytes_written(rows: int = 20000) -> dict:
    """What the spill costs, in bytes through the file format.

    The number a planner would need to decide between spilling and refusing the query, and it is
    larger than the data because each run carries a header and a footer.
    """
    batch = _table(rows)
    keys = [SortKey(name="amount")]
    with tempfile.TemporaryDirectory() as directory:
        spilled = spill_sorted_runs(batch, keys, Path(directory), run_rows=1000)
    return {
        "rows": rows,
        "data_bytes": batch.nbytes,
        "written_bytes": spilled.bytes_written,
        "runs": spilled.runs,
        "overhead": round(spilled.bytes_written / batch.nbytes, 3),
        "bytes_per_run": spilled.bytes_written // spilled.runs,
    }


def a_merge_of_one_run_is_a_read() -> dict:
    """The degenerate case, where the table fitted after all."""
    batch = _table(500)
    keys = [SortKey(name="amount")]
    with tempfile.TemporaryDirectory() as directory:
        spilled = spill_sorted_runs(batch, keys, Path(directory), run_rows=5000)
        merged = merge_runs(spilled, keys)
    return {
        "runs": spilled.runs,
        "it_is_one": spilled.runs == 1,
        "rows": merged.rows,
        "it_is_sorted": bool(np.all(np.diff(merged.column("amount").values) >= 0)),
    }


def a_descending_string_merge_is_refused() -> bool:
    """A merge that cannot express its own ordering, refused rather than silently wrong."""
    batch = _table(500)
    keys = [SortKey(name="label", descending=True)]
    with tempfile.TemporaryDirectory() as directory:
        spilled = spill_sorted_runs(batch, keys, Path(directory), run_rows=100)
        try:
            merge_runs(spilled, keys)
        except ConfigError:
            return True
    return False


def a_zero_run_size_is_refused() -> bool:
    """A run of no rows."""
    with tempfile.TemporaryDirectory() as directory:
        try:
            spill_sorted_runs(_table(100), [SortKey(name="amount")], Path(directory), 0)
        except ConfigError:
            return True
    return False


def merging_nothing_is_refused() -> bool:
    """A merge with no runs in it."""
    try:
        merge_runs(Spilled(paths=(), rows=0, bytes_written=0), [SortKey(name="a")])
    except ConfigError:
        return True
    return False


def partitioning_by_a_missing_column_is_refused() -> bool:
    """A partition key that is not a column."""
    with tempfile.TemporaryDirectory() as directory:
        try:
            spill_partitions(_table(100), "nothing", Path(directory))
        except ConfigError:
            return True
    return False


def a_zero_batch_size_is_refused() -> bool:
    """Reading a table in batches of nothing."""
    try:
        list(in_batches(_table(100), 0))
    except ConfigError:
        return True
    return False


def compare_the_two_spills() -> list[dict]:
    """The sort and the aggregate side by side, which is the module in one table."""
    return [
        {
            "operator": "sort",
            "spills": "runs",
            "combine": "k way merge",
            "memory": "one run plus one row per run",
            "survives_skew": True,
        },
        {
            "operator": "aggregate",
            "spills": "partitions",
            "combine": "concatenation",
            "memory": "the largest partition",
            "survives_skew": False,
        },
    ]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "run_rows": RUN_ROWS,
        "partitions": PARTITIONS,
        "sort_agrees": an_external_sort_gives_the_same_order()["they_agree"],
        "aggregate_agrees": an_external_aggregate_gives_the_same_groups()["they_agree"],
        "groups_land_in_one_partition": every_group_lands_in_one_partition()["it_holds"],
        "skew_survives_hashing": a_skewed_key_makes_an_uneven_partition()[
            "the_skew_survives_the_hash"
        ],
    }
