from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, SchemaError, UnknownColumn
from cqe.exec.aggregate import Aggregate, hash_aggregate
from cqe.exec.batch import Batch
from cqe.exec.sort import SortKey, order_by
from cqe.types.schema import FLOATING, INTEGER, Field
from cqe.verify.reference import Rows

# Window functions, which are the operations that need a row's neighbours.
#
# An aggregate collapses a group to one row and a window function does not: it produces one
# value per row, computed over the rows around it. That difference is the whole implementation
# difficulty, because the fast path for an aggregate is a scatter into one slot per group and
# the fast path for a window is a scan along each group in order.
#
# Three kinds are here and they need different things.
#
# A ranking, which needs the rows in order within their partition and nothing else. Row number,
# rank and dense rank differ only in how they treat ties, and the difference is the thing that
# gets implemented wrong: row number never ties, rank ties and then skips, dense rank ties and
# does not skip.
#
# A running total, which needs a prefix sum along each partition. Vectorised with cumsum per
# partition rather than a loop, and the measurement below is what that is worth.
#
# A neighbour, which is the value one row earlier or later. A shift within the partition, with
# the rows that fall off the end becoming null rather than wrapping, and wrapping is what a
# naive roll would do.
#
# Everything here is checked against a row at a time reference in verify/reference.py the same
# way every operator is, because a window function is the operator where an off by one is
# easiest to write and hardest to see.

FUNCTIONS = (
    "row_number",
    "rank",
    "dense_rank",
    "running_sum",
    "running_max",
    "lag",
    "lead",
)


@dataclass(frozen=True)
class Window:
    """One window function: what to compute, over what partition, in what order."""

    name: str
    function: str
    source: str = ""
    partition: tuple[str, ...] = ()
    order: tuple[SortKey, ...] = ()
    offset: int = 1

    def __post_init__(self) -> None:
        if self.function not in FUNCTIONS:
            raise ConfigError(
                f"{self.function} is not a window function; try one of {sorted(FUNCTIONS)}"
            )
        if self.function in ("running_sum", "running_max", "lag", "lead") and not self.source:
            raise ConfigError(f"{self.function} needs a source column")
        if self.function in ("rank", "dense_rank") and not self.order:
            raise ConfigError(f"{self.function} needs an order to rank by")
        if self.offset < 1:
            raise ConfigError(f"{self.offset} is not an offset")

    @property
    def needs_order(self) -> bool:
        """Whether the result depends on the order within the partition.

        Everything except a partition wide maximum does, which is why there is no such function
        here: a window aggregate with no ordering is a group by joined back to its input, and
        exec/aggregate.py already does that better.
        """
        return True

    def describe(self) -> str:
        """One line, as it would be written."""
        pieces = [f"{self.function}({self.source})" if self.source else f"{self.function}()"]
        if self.partition:
            pieces.append(f"partition by {', '.join(self.partition)}")
        if self.order:
            pieces.append(f"order by {', '.join(one.name for one in self.order)}")
        return " ".join(pieces) + f" as {self.name}"

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "function": self.function,
            "source": self.source,
            "partition": list(self.partition),
            "order": [one.name for one in self.order],
        }


def _partition_starts(batch: Batch, names: Sequence[str], order: np.ndarray) -> np.ndarray:
    """Where each partition begins, given rows already arranged by partition.

    A boolean per row rather than a list of slices, because every one of the functions below is
    written as a vectorised pass with a reset at the boundaries, and a boolean array is what
    resets them. A list of slices would mean a Python loop per partition and the whole point is
    not to have one.
    """
    if not names:
        starts = np.zeros(batch.rows, dtype=bool)
        if batch.rows:
            starts[0] = True
        return starts
    starts = np.zeros(batch.rows, dtype=bool)
    if not batch.rows:
        return starts
    starts[0] = True
    for name in names:
        values = batch.column(name).values[order]
        starts[1:] |= values[1:] != values[:-1]
    return starts


def _arranged(batch: Batch, window: Window, meter: Meter | None = None) -> np.ndarray:
    """The row order a window needs: partition first, then the window's own order.

    One sort rather than two. The partition columns are the most significant keys and the order
    columns follow, which gives rows grouped by partition and ordered inside each, and that is
    exactly what every function below walks.
    """
    keys = [SortKey(name=one) for one in window.partition] + list(window.order)
    if not keys:
        return np.arange(batch.rows, dtype=np.int64)
    return order_by(batch, keys, meter=meter).positions


def _row_number(starts: np.ndarray) -> np.ndarray:
    """One, two, three within each partition.

    Computed as the position minus the position of the partition's first row, which is a running
    maximum over the start positions. No loop and no ties, because a row number never ties even
    when the ordering does.
    """
    positions = np.arange(len(starts), dtype=np.int64)
    first = np.maximum.accumulate(np.where(starts, positions, -1))
    return positions - first + 1


def _rank(starts: np.ndarray, ties: np.ndarray, dense: bool) -> np.ndarray:
    """Rank, which ties, and dense rank, which ties without leaving a gap.

    The difference is one line and it is the line everybody gets wrong. After three rows tied at
    rank one, plain rank gives the next row four and dense rank gives it two. Plain rank is the
    row number of the first row of the tie group; dense rank counts the tie groups.
    """
    numbers = _row_number(starts)
    fresh = starts | ~ties
    if dense:
        groups = np.cumsum(fresh)
        first = np.maximum.accumulate(np.where(starts, groups, 0))
        return groups - first + 1
    # Carry each row the row number of the first row of its tie group. Accumulated over the
    # global position rather than over the row number, because a row number restarts at one in
    # every partition and a running maximum over it would carry the previous partition's largest
    # value across the boundary. The differential check found exactly that: rank agreed on a
    # single partition and disagreed as soon as there were two.
    positions = np.arange(len(starts), dtype=np.int64)
    governing = np.maximum.accumulate(np.where(fresh, positions, 0))
    return numbers[governing] if len(positions) else numbers


def _ties(batch: Batch, window: Window, order: np.ndarray) -> np.ndarray:
    """Whether each row equals the one before it on every ordering column."""
    same = np.zeros(batch.rows, dtype=bool)
    if batch.rows < 2 or not window.order:
        return same
    same[1:] = True
    for key in window.order:
        values = batch.column(key.name).values[order]
        same[1:] &= values[1:] == values[:-1]
    return same


def _running(values: np.ndarray, starts: np.ndarray, function: str) -> np.ndarray:
    """A running sum or maximum along each partition.

    The sum is a cumulative sum with the partition's opening total subtracted back off, which is
    two passes and no loop. The maximum has no such trick because a maximum has no inverse, so
    it is a cumulative maximum per partition, done by resetting at the boundaries.
    """
    if function == "running_sum":
        totals = np.cumsum(values)
        opening = np.where(starts, totals - values, 0.0)
        return totals - np.maximum.accumulate(opening)
    out = values.astype(np.float64, copy=True)
    boundaries = np.flatnonzero(starts)
    edges = np.append(boundaries, len(values))
    for start, stop in pairwise(edges):
        out[start:stop] = np.maximum.accumulate(values[start:stop])
    return out


def _shifted(
    values: np.ndarray, starts: np.ndarray, offset: int, forward: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The value some rows away in the same partition, and where there was not one.

    The rows that fall off either end of a partition get a null rather than the value from the
    neighbouring partition, which is what a plain roll would give and is the bug this function
    exists to not have.
    """
    rows = len(values)
    out = np.zeros(rows, dtype=values.dtype)
    valid = np.zeros(rows, dtype=bool)
    positions = np.arange(rows)
    first = np.maximum.accumulate(np.where(starts, positions, -1))
    ends = np.append(np.flatnonzero(starts)[1:], rows)
    last = np.repeat(ends - 1, np.diff(np.append(np.flatnonzero(starts), rows)))
    wanted = positions - offset if forward else positions + offset
    inside = (wanted >= first) & (wanted <= last)
    out[inside] = values[wanted[inside]]
    valid[inside] = True
    return out, valid


def evaluate(batch: Batch, window: Window, meter: Meter | None = None) -> Column:
    """One window function over a batch, as a column in the batch's own row order.

    The result is put back in the input's order rather than left in the window's order, which is
    what makes a window function composable: two windows with different partitions can be added
    to the same batch, and neither reorders it.
    """
    _check(batch, window)
    order = _arranged(batch, window, meter=meter)
    starts = _partition_starts(batch, window.partition, order)
    if meter is not None:
        meter.touch(batch.rows, f"window_{window.function}")
    if window.function == "row_number":
        values, valid, logical = _row_number(starts), None, INTEGER
    elif window.function in ("rank", "dense_rank"):
        ties = _ties(batch, window, order)
        values = _rank(starts, ties, dense=window.function == "dense_rank")
        valid, logical = None, INTEGER
    elif window.function in ("running_sum", "running_max"):
        source = batch.column(window.source).values[order].astype(np.float64)
        values = _running(source, starts, window.function)
        valid, logical = None, FLOATING
    else:
        source = batch.column(window.source).values[order]
        values, valid = _shifted(
            source, starts, window.offset, forward=window.function == "lag"
        )
        logical = batch.column(window.source).field.logical
    back = np.empty(batch.rows, dtype=np.int64)
    back[order] = np.arange(batch.rows)
    dictionary = (
        batch.column(window.source).dictionary if window.function in ("lag", "lead") else None
    )
    if meter is not None:
        meter.materialise(batch.rows)
    return Column(
        field=Field(name=window.name, logical=logical, nullable=valid is not None),
        values=values[back],
        valid=None if valid is None else valid[back],
        dictionary=dictionary,
    )


def _check(batch: Batch, window: Window) -> None:
    """Every column a window reads exists, named against the part that reads it."""
    if window.source and window.source not in batch.schema:
        raise UnknownColumn(f"{window.source} is not a column of {list(batch.schema.names)}")
    missing = [one for one in window.partition if one not in batch.schema]
    if missing:
        raise UnknownColumn(f"{missing} are partitioned by and are not columns")
    missing = [one.name for one in window.order if one.name not in batch.schema]
    if missing:
        raise UnknownColumn(f"{missing} are ordered by and are not columns")
    if window.name in batch.schema:
        raise SchemaError(f"{window.name} is already a column")


def apply(batch: Batch, windows: Sequence[Window], meter: Meter | None = None) -> Batch:
    """Several window functions, each added as a column."""
    made = list(batch.columns)
    for one in windows:
        made.append(evaluate(batch, one, meter=meter))
    return Batch.from_columns(made)


def _reference(batch: Batch, window: Window) -> list:
    """The same window computed row at a time in Python.

    The reference every measurement below compares against. It shares no code with the
    vectorised path: it sorts a list of dictionaries, walks it keeping a running state, and puts
    the answers back by row index.
    """
    rows = Rows.of(batch)
    names = list(rows.names)
    indexed = list(enumerate(rows.rows))
    keys = list(window.partition) + [one.name for one in window.order]
    directions = [False] * len(window.partition) + [one.descending for one in window.order]

    def sorter(pair):
        out = []
        for name, down in zip(keys, directions, strict=True):
            value = pair[1][names.index(name)]
            out.append(_flip(value, down))
        return tuple(out)

    indexed.sort(key=sorter)
    out: list = [None] * len(indexed)
    partition_of = [
        tuple(pair[1][names.index(one)] for one in window.partition) for pair in indexed
    ]
    running = 0.0
    highest = None
    number = 0
    rank = 0
    dense = 0
    previous_key = None
    for position, (index, row) in enumerate(indexed):
        fresh = position == 0 or partition_of[position] != partition_of[position - 1]
        if fresh:
            running, highest, number, rank, dense, previous_key = 0.0, None, 0, 0, 0, None
        number += 1
        key = tuple(row[names.index(one.name)] for one in window.order)
        if key != previous_key:
            rank = number
            dense += 1
            previous_key = key
        if window.function == "row_number":
            out[index] = number
        elif window.function == "rank":
            out[index] = rank
        elif window.function == "dense_rank":
            out[index] = dense
        elif window.function == "running_sum":
            running += float(row[names.index(window.source)])
            out[index] = running
        elif window.function == "running_max":
            value = float(row[names.index(window.source)])
            highest = value if highest is None else max(highest, value)
            out[index] = highest
        else:
            step = -window.offset if window.function == "lag" else window.offset
            wanted = position + step
            same = 0 <= wanted < len(indexed) and partition_of[wanted] == partition_of[position]
            out[index] = indexed[wanted][1][names.index(window.source)] if same else None
    return out


def _flip(value, descending: bool):
    """One sort key value, negated when the ordering is descending."""
    if not descending:
        return value
    if isinstance(value, str):
        raise ConfigError("the reference cannot order strings downwards")
    return -value


def _table(rows: int = 4000, partitions: int = 12, seed: int = 9) -> Batch:
    """A table with a partition column, an ordering column and a value."""
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, partitions, rows)),
            integer_column("day", state.integers(0, 30, rows)),
            floating_column("amount", state.normal(100, 20, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 4, rows)]),
        ]
    )


def _agrees(batch: Batch, window: Window) -> bool:
    """Whether the vectorised path and the reference produce the same column."""
    produced = evaluate(batch, window).to_list()
    expected = _reference(batch, window)
    for one, other in zip(produced, expected, strict=True):
        if one is None or other is None:
            if one is not other:
                return False
            continue
        if abs(float(one) - float(other)) > 1e-9:
            return False
    return True


def every_function_agrees_with_the_reference(rows: int = 2000) -> dict:
    """All seven functions against the row at a time interpreter.

    The measurement that matters. A window function is where an off by one is easiest to write
    and hardest to see, and only a second implementation walking the rows one at a time finds
    one.
    """
    batch = _table(rows)
    order = (SortKey(name="day"),)
    windows = {
        "row_number": Window("v", "row_number", partition=("shop",), order=order),
        "rank": Window("v", "rank", partition=("shop",), order=order),
        "dense_rank": Window("v", "dense_rank", partition=("shop",), order=order),
        "running_sum": Window("v", "running_sum", "amount", ("shop",), order),
        "running_max": Window("v", "running_max", "amount", ("shop",), order),
        "lag": Window("v", "lag", "amount", ("shop",), order),
        "lead": Window("v", "lead", "amount", ("shop",), order),
    }
    out = {name: _agrees(batch, one) for name, one in windows.items()}
    return {**out, "they_all_agree": all(out.values())}


def the_three_rankings_differ_on_ties(rows: int = 40) -> dict:
    """Row number, rank and dense rank over a column with ties in it.

    The one measurement that makes the difference between them concrete. Three rows tied at the
    first position: row number gives one two three, rank gives one one one and then four, dense
    rank gives one one one and then two.
    """
    values = [1, 1, 1, 2, 2, 3]
    batch = Batch.from_columns(
        [
            integer_column("day", values),
            integer_column("id", list(range(len(values)))),
        ]
    )
    order = (SortKey(name="day"),)
    produced = {
        one: evaluate(batch, Window("v", one, order=order)).to_list()
        for one in ("row_number", "rank", "dense_rank")
    }
    return {
        **produced,
        "row_number_never_ties": len(set(produced["row_number"])) == len(values),
        "rank_ties_and_skips": produced["rank"] == [1, 1, 1, 4, 4, 6],
        "dense_rank_ties_without_skipping": produced["dense_rank"] == [1, 1, 1, 2, 2, 3],
    }


def a_window_does_not_collapse_its_input(rows: int = 2000) -> dict:
    """A window function returns one row per input row, unlike an aggregate.

    The definitional difference, checked rather than stated, because the easiest wrong
    implementation of a window function is a group by joined back and it would fail here on a
    partition holding no rows.
    """
    batch = _table(rows)
    windowed = apply(
        batch,
        [Window("total", "running_sum", "amount", ("shop",), (SortKey(name="day"),))],
    )
    grouped = hash_aggregate(
        batch, ["shop"], [Aggregate(name="total", function="sum", source="amount")]
    ).batch
    return {
        "input_rows": batch.rows,
        "window_rows": windowed.rows,
        "aggregate_rows": grouped.rows,
        "the_window_kept_every_row": windowed.rows == batch.rows,
        "the_aggregate_collapsed_them": grouped.rows < batch.rows,
        "and_it_added_a_column": windowed.width == batch.width + 1,
    }


def the_running_sum_ends_at_the_group_total(rows: int = 4000) -> dict:
    """The last row of each partition holds that partition's whole sum.

    The property that ties a window back to an aggregate: a running sum's final value per
    partition is the group sum, so the two operators must agree at the boundary even though they
    produce different shapes.
    """
    batch = _table(rows)
    order = (SortKey(name="day"), SortKey(name="id"))
    windowed = apply(batch, [Window("total", "running_sum", "amount", ("shop",), order)])
    grouped = hash_aggregate(
        batch, ["shop"], [Aggregate(name="sum", function="sum", source="amount")]
    ).batch
    finals = {}
    rows_out = windowed.to_rows()
    names = list(windowed.schema.names)
    for row in rows_out:
        shop = row[names.index("shop")]
        finals[shop] = max(finals.get(shop, float("-inf")), row[names.index("total")])
    totals = dict(
        zip(
            grouped.column("shop").to_list(),
            grouped.column("sum").to_list(),
            strict=True,
        )
    )
    return {
        "partitions": len(totals),
        "they_agree": all(abs(finals[shop] - totals[shop]) < 1e-6 for shop in totals),
        "the_largest_difference": round(
            max(abs(finals[shop] - totals[shop]) for shop in totals), 9
        ),
    }


def a_neighbour_does_not_cross_a_partition(rows: int = 2000) -> dict:
    """The first row of each partition has no earlier neighbour, and gets a null.

    The bug a roll produces: without the check, the first row of a partition takes the last row
    of the one before it, which is a value from a different shop and looks entirely plausible.
    """
    batch = _table(rows)
    order = (SortKey(name="day"), SortKey(name="id"))
    lagged = evaluate(batch, Window("before", "lag", "amount", ("shop",), order))
    partitions = len(set(batch.column("shop").to_list()))
    nulls = int((~lagged.valid).sum()) if lagged.valid is not None else 0
    return {
        "rows": rows,
        "partitions": partitions,
        "nulls": nulls,
        "one_null_per_partition": nulls == partitions,
        "and_it_agrees_with_the_reference": _agrees(
            batch, Window("before", "lag", "amount", ("shop",), order)
        ),
    }


def a_lead_is_a_lag_backwards(rows: int = 2000) -> dict:
    """Lead and lag over the same window, where each row's lead is its neighbour's lag.

    An internal consistency check rather than a reference one, and it catches the case where
    both are wrong in the same direction, which a reference written by the same hand might
    share.
    """
    batch = _table(rows)
    order = (SortKey(name="day"), SortKey(name="id"))
    lagged = evaluate(batch, Window("before", "lag", "amount", ("shop",), order))
    led = evaluate(batch, Window("after", "lead", "amount", ("shop",), order))
    return {
        "lag_nulls": int((~lagged.valid).sum()) if lagged.valid is not None else 0,
        "lead_nulls": int((~led.valid).sum()) if led.valid is not None else 0,
        "they_have_the_same_null_count": (
            int((~lagged.valid).sum()) == int((~led.valid).sum())
        ),
        "the_lag_agrees": _agrees(batch, Window("v", "lag", "amount", ("shop",), order)),
        "the_lead_agrees": _agrees(batch, Window("v", "lead", "amount", ("shop",), order)),
    }


def a_window_with_no_partition_is_one_partition(rows: int = 1000) -> dict:
    """Without a partition clause the whole table is one window.

    Which is the standard's reading and is also the one that makes a running total over a table
    mean what a reader expects.
    """
    batch = _table(rows)
    order = (SortKey(name="id"),)
    numbers = evaluate(batch, Window("v", "row_number", order=order)).to_list()
    running = evaluate(batch, Window("v", "running_sum", "amount", order=order)).to_list()
    return {
        "rows": rows,
        "first_number": numbers[0],
        "last_number": max(numbers),
        "it_counts_the_whole_table": max(numbers) == rows,
        "the_sum_ends_at_the_total": abs(
            max(running) - float(sum(batch.column("amount").to_list()))
        )
        < 1e-6,
    }


def the_result_comes_back_in_the_input_order(rows: int = 1000) -> dict:
    """A window sorts internally and returns its column in the batch's own order.

    Which is what lets two windows with different partitions be added to the same batch. Checked
    by adding a window and confirming the other columns did not move.
    """
    batch = _table(rows)
    windowed = apply(
        batch,
        [
            Window("by_shop", "row_number", partition=("shop",), order=(SortKey(name="day"),)),
            Window(
                "by_region", "row_number", partition=("region",), order=(SortKey(name="id"),)
            ),
        ],
    )
    return {
        "columns": windowed.width,
        "it_added_two": windowed.width == batch.width + 2,
        "the_input_did_not_move": bool(
            np.array_equal(windowed.column("id").values, batch.column("id").values)
        ),
        "both_partitions_are_right": (
            max(windowed.column("by_shop").to_list()) < rows
            and max(windowed.column("by_region").to_list()) < rows
        ),
    }


def the_vectorised_path_beats_the_reference(rows: int = 4000) -> dict:
    """How much work each does, counted rather than timed.

    The reference does one Python operation per row and the fast path does a handful of numpy
    passes over the whole column, so the ratio is the row count divided by the number of passes.
    Counting the passes rather than timing them keeps the number a property of the algorithm.
    """
    batch = _table(rows)
    order = (SortKey(name="day"),)
    meter = Meter()
    evaluate(batch, Window("v", "running_sum", "amount", ("shop",), order), meter=meter)
    return {
        "rows": rows,
        "values_touched": meter.values_touched,
        "passes": round(meter.values_touched / max(rows, 1), 1),
        "the_reference_does_one_per_row": rows,
        "it_is_a_few_passes": meter.values_touched < rows * 10,
    }


def a_string_column_can_be_lagged(rows: int = 500) -> dict:
    """Lag over a dictionary encoded column, which keeps the dictionary.

    The case that catches a lag implemented on the values array alone: the codes shift correctly
    and the result is meaningless without the dictionary that decodes them.
    """
    batch = _table(rows)
    order = (SortKey(name="id"),)
    lagged = evaluate(batch, Window("before", "lag", "region", ("shop",), order))
    values = lagged.to_list()
    return {
        "rows": rows,
        "it_kept_the_dictionary": lagged.dictionary is not None,
        "the_values_are_strings": all(one is None or isinstance(one, str) for one in values),
        "and_they_are_real_regions": {one for one in values if one is not None}
        <= set(batch.column("region").to_list()),
    }


def an_offset_of_two_looks_two_rows_back(rows: int = 500) -> dict:
    """Lag with an offset, which needs two nulls per partition rather than one."""
    batch = _table(rows, partitions=5)
    order = (SortKey(name="id"),)
    one_back = evaluate(batch, Window("v", "lag", "amount", ("shop",), order, offset=1))
    two_back = evaluate(batch, Window("v", "lag", "amount", ("shop",), order, offset=2))
    return {
        "partitions": 5,
        "one_back_nulls": int((~one_back.valid).sum()),
        "two_back_nulls": int((~two_back.valid).sum()),
        "it_is_twice_as_many": int((~two_back.valid).sum()) == 2 * int((~one_back.valid).sum()),
    }


def a_descending_order_reverses_the_ranking(rows: int = 500) -> dict:
    """Rank by a column downwards, where the largest value ranks first."""
    batch = _table(rows, partitions=4)
    up = evaluate(
        batch, Window("v", "row_number", partition=("shop",), order=(SortKey(name="amount"),))
    ).to_list()
    down = evaluate(
        batch,
        Window(
            "v",
            "row_number",
            partition=("shop",),
            order=(SortKey(name="amount", descending=True),),
        ),
    ).to_list()
    amounts = batch.column("amount").to_list()
    shops = batch.column("shop").to_list()
    first_up = amounts[up.index(1)]
    smallest = min(
        one for one, shop in zip(amounts, shops, strict=True) if shop == shops[up.index(1)]
    )
    largest = max(
        one for one, shop in zip(amounts, shops, strict=True) if shop == shops[down.index(1)]
    )
    return {
        "ascending_first": round(first_up, 3),
        "it_is_the_smallest": abs(first_up - smallest) < 1e-9,
        "descending_first": round(amounts[down.index(1)], 3),
        "it_is_the_largest": abs(amounts[down.index(1)] - largest) < 1e-9,
    }


def an_empty_batch_produces_an_empty_column() -> dict:
    """A window over no rows, which must still produce a column of the right type."""
    batch = _table(10).slice(0, 0)
    made = evaluate(batch, Window("v", "row_number", order=(SortKey(name="id"),)))
    return {
        "rows": len(made),
        "it_is_empty": len(made) == 0,
        "and_it_has_a_type": made.field.logical == INTEGER,
    }


def a_partition_of_one_row_works(rows: int = 40) -> dict:
    """Every partition holding exactly one row, where every neighbour is null."""
    batch = Batch.from_columns(
        [
            integer_column("shop", list(range(rows))),
            integer_column("day", [1] * rows),
            floating_column("amount", [float(one) for one in range(rows)]),
        ]
    )
    order = (SortKey(name="day"),)
    numbers = evaluate(
        batch, Window("v", "row_number", partition=("shop",), order=order)
    ).to_list()
    lagged = evaluate(batch, Window("v", "lag", "amount", ("shop",), order))
    return {
        "partitions": rows,
        "every_number_is_one": set(numbers) == {1},
        "every_neighbour_is_null": int((~lagged.valid).sum()) == rows,
    }


def an_unknown_function_is_refused() -> bool:
    """A window function that does not exist, with the list in the message."""
    try:
        Window("v", "median")
    except ConfigError:
        return True
    return False


def a_running_sum_without_a_source_is_refused() -> bool:
    """A running total over nothing."""
    try:
        Window("v", "running_sum")
    except ConfigError:
        return True
    return False


def a_rank_without_an_order_is_refused() -> bool:
    """A ranking with nothing to rank by, which would be arbitrary."""
    try:
        Window("v", "rank")
    except ConfigError:
        return True
    return False


def a_zero_offset_is_refused() -> bool:
    """A lag of no rows, which is the column itself."""
    try:
        Window("v", "lag", "amount", order=(SortKey(name="id"),), offset=0)
    except ConfigError:
        return True
    return False


def a_missing_source_column_is_refused() -> bool:
    """A window over a column that is not there."""
    try:
        evaluate(_table(10), Window("v", "running_sum", "nothing", order=(SortKey(name="id"),)))
    except UnknownColumn:
        return True
    return False


def a_missing_partition_column_is_refused() -> bool:
    """A partition by a column that is not there."""
    try:
        evaluate(
            _table(10),
            Window("v", "row_number", partition=("nothing",), order=(SortKey(name="id"),)),
        )
    except UnknownColumn:
        return True
    return False


def a_name_that_already_exists_is_refused() -> bool:
    """A window named after an existing column, which would shadow it."""
    try:
        evaluate(_table(10), Window("amount", "row_number", order=(SortKey(name="id"),)))
    except SchemaError:
        return True
    return False


def compare_the_functions(rows: int = 2000) -> list[dict]:
    """Every function, what it needs and what it produces."""
    batch = _table(rows)
    order = (SortKey(name="day"),)
    out = []
    for name in FUNCTIONS:
        source = "amount" if name not in ("row_number", "rank", "dense_rank") else ""
        window = Window("v", name, source, ("shop",), order)
        made = evaluate(batch, window)
        out.append(
            {
                "function": name,
                "type": made.field.logical,
                "nullable": made.valid is not None,
                "nulls": 0 if made.valid is None else int((~made.valid).sum()),
                "agrees": _agrees(batch, window),
            }
        )
    return out


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "functions": len(FUNCTIONS),
        "all_agree": every_function_agrees_with_the_reference()["they_all_agree"],
        "rankings_differ": the_three_rankings_differ_on_ties()["rank_ties_and_skips"],
        "keeps_every_row": a_window_does_not_collapse_its_input()["the_window_kept_every_row"],
        "neighbours_stay_inside": a_neighbour_does_not_cross_a_partition()[
            "one_null_per_partition"
        ],
        "passes": the_vectorised_path_beats_the_reference()["passes"],
    }
