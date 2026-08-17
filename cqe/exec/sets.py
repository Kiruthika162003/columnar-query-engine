from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import SchemaError
from cqe.exec.batch import Batch, stack
from cqe.exec.join.outer import semi
from cqe.exec.sort import SortKey, order_by
from cqe.verify.reference import Rows, agree
from cqe.verify.reference import distinct as reference_distinct
from cqe.verify.reference import union_all as reference_union

# Distinct and the set operations, which are all one thing with different bookkeeping.
#
# Every operation here is built on identifying which rows are equal to which. Once that is done,
# distinct keeps the first of each group, union keeps the rows in either side, intersect keeps
# the rows in both, and except keeps the rows in the left and not the right. Four operations,
# one grouping.
#
# Two decisions run through all of them.
#
# A null equals a null here. That is the opposite of the join rule and it is not an
# inconsistency: a join asks whether two rows match and a set operation asks whether two rows
# are the same row, and two rows that are both missing a value are the same row. Every standard
# says so and it still surprises people, so the measurement below states it in both directions.
#
# The comparison is over every column, so a set operation is only defined between two tables
# with the same schema. Refusing a mismatch is the whole of the type checking here and the
# message names both schemas, because a set operation between the wrong two tables is usually a
# query that meant to project one of them first.
#
# The implementation sorts rather than hashes, and the measurement below is why: a sort gives
# the duplicate groups and the order at once, and a hash gives the groups and then needs a sort
# to make the answer deterministic.

# How many rows a set operation will compare with the row at a time path before it stops being
# worth measuring against. Above this the reference is slow enough to dominate a measurement.
REFERENCE_LIMIT = 20000


@dataclass
class Produced:
    """What one set operation produced and what it cost."""

    batch: Batch
    kind: str
    left_rows: int
    right_rows: int
    duplicates: int = 0

    @property
    def rows(self) -> int:
        """Rows the operation produced."""
        return self.batch.rows

    @property
    def share(self) -> float:
        """The share of the input that survived."""
        return self.rows / max(self.left_rows + self.right_rows, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kind": self.kind,
            "left": self.left_rows,
            "right": self.right_rows,
            "rows": self.rows,
            "duplicates": self.duplicates,
            "share": round(self.share, 3),
        }


def _keys(batch: Batch) -> np.ndarray:
    """One comparable value per row, over every column.

    A tuple per row as an object array, which is the slow and correct thing. A hash of the row
    would be faster and would need a second pass to resolve collisions, and this module is about
    the semantics rather than the speed.

    A null becomes a sentinel object rather than being skipped, which is what makes two rows
    that are both missing a value compare equal. Skipping it would make a null row equal to
    every row, and using the underlying value would make it equal to whatever leftover sat in
    the array.
    """
    out = np.empty(batch.rows, dtype=object)
    columns = [one.to_list() for one in batch.columns]
    for position in range(batch.rows):
        out[position] = tuple(
            _MISSING if one[position] is None else one[position] for one in columns
        )
    return out


class _Missing:
    """The stand in for a null inside a row key, so that two nulls compare equal."""

    def __repr__(self) -> str:
        return "missing"

    def __lt__(self, other) -> bool:
        return not isinstance(other, _Missing)


_MISSING = _Missing()


def _check(left: Batch, right: Batch) -> None:
    """Both sides have the same schema, which every set operation needs."""
    if list(left.schema.names) != list(right.schema.names):
        raise SchemaError(
            f"{list(left.schema.names)} against {list(right.schema.names)}; "
            "a set operation needs the same columns in the same order"
        )
    for one, other in zip(left.schema.fields, right.schema.fields, strict=True):
        if one.logical != other.logical:
            raise SchemaError(
                f"{one.name} is {one.logical} on the left and {other.logical} on the right"
            )


def _ordered(batch: Batch, meter: Meter | None = None) -> np.ndarray:
    """The row positions in a deterministic order, sorted on every column.

    Sorted rather than hashed. A hash would find the duplicate groups faster and would leave the
    output in whatever order the table happened to be in, and then a caller comparing two runs
    would see different row orders for the same query. Sorting costs n log n and makes the
    answer a function of the input rather than of the memory layout.
    """
    if not batch.rows:
        return np.array([], dtype=np.int64)
    keys = [SortKey(name=one) for one in batch.schema.names]
    return order_by(batch, keys, meter=meter).positions


def distinct(batch: Batch, meter: Meter | None = None) -> Produced:
    """One row of each distinct row, in sorted order."""
    if not batch.rows:
        return Produced(batch=batch, kind="distinct", left_rows=0, right_rows=0)
    order = _ordered(batch, meter=meter)
    keys = _keys(batch)[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.array(
        [keys[one] != keys[one - 1] for one in range(1, len(keys))], dtype=bool
    )
    if meter is not None:
        meter.materialise(int(first.sum()))
    return Produced(
        batch=batch.take(order[first]),
        kind="distinct",
        left_rows=batch.rows,
        right_rows=0,
        duplicates=int((~first).sum()),
    )


def union_all(left: Batch, right: Batch, meter: Meter | None = None) -> Produced:
    """Every row from both sides, keeping the duplicates.

    The only operation here that does not need the keys at all, which is why it is the one a
    planner should reach for when the duplicates do not matter: it is a concatenation and costs
    nothing but the copy.
    """
    _check(left, right)
    made = stack([left, right])
    if meter is not None:
        meter.materialise(made.rows)
    return Produced(batch=made, kind="union all", left_rows=left.rows, right_rows=right.rows)


def union(left: Batch, right: Batch, meter: Meter | None = None) -> Produced:
    """Every row in either side, once each."""
    _check(left, right)
    made = distinct(stack([left, right]), meter=meter)
    return Produced(
        batch=made.batch,
        kind="union",
        left_rows=left.rows,
        right_rows=right.rows,
        duplicates=made.duplicates,
    )


def intersect(left: Batch, right: Batch, meter: Meter | None = None) -> Produced:
    """The rows in both sides, once each."""
    _check(left, right)
    wanted = set(_keys(right).tolist())
    keys = _keys(left)
    kept = np.array([one in wanted for one in keys], dtype=bool)
    if meter is not None:
        meter.probe(left.rows)
    made = distinct(left.mask(kept), meter=meter) if kept.any() else None
    return Produced(
        batch=made.batch if made else Batch.empty(left.schema),
        kind="intersect",
        left_rows=left.rows,
        right_rows=right.rows,
        duplicates=made.duplicates if made else 0,
    )


def difference(left: Batch, right: Batch, meter: Meter | None = None) -> Produced:
    """The rows in the left and not the right, once each."""
    _check(left, right)
    unwanted = set(_keys(right).tolist())
    keys = _keys(left)
    kept = np.array([one not in unwanted for one in keys], dtype=bool)
    if meter is not None:
        meter.probe(left.rows)
    made = distinct(left.mask(kept), meter=meter) if kept.any() else None
    return Produced(
        batch=made.batch if made else Batch.empty(left.schema),
        kind="except",
        left_rows=left.rows,
        right_rows=right.rows,
        duplicates=made.duplicates if made else 0,
    )


def _table(rows: int = 4000, distinct_rows: int = 1000, seed: int = 151) -> Batch:
    """A table with a set number of distinct rows, repeated to fill it.

    The first version built the columns as picks modulo fifty, two hundred and eight, which caps
    the distinct rows at two hundred whatever the argument says. Every sweep over the distinct
    count then produced the same number and the measurement said the saving was flat when it was
    the generator that was flat. The pick itself is one of the columns now, so the distinct
    count is what was asked for.
    """
    state = np.random.default_rng(seed)
    picks = state.integers(0, distinct_rows, rows)
    return Batch.from_columns(
        [
            integer_column("shop", picks),
            floating_column("amount", (picks % 200).astype(np.float64)),
            string_column("region", [f"region{one % 8}" for one in picks]),
        ]
    )


def _with_nulls(rows: int = 1000, seed: int = 157) -> Batch:
    """A table where a share of one column is null."""
    state = np.random.default_rng(seed)
    values = state.integers(0, 20, rows)
    made = integer_column("shop", values)
    valid = state.random(rows) > 0.3
    return Batch.from_columns(
        [
            Column(field=made.field, values=values, valid=valid),
            integer_column("day", state.integers(0, 5, rows)),
        ]
    )


def distinct_removes_the_repeats(rows: int = 4000) -> dict:
    """A table of four thousand rows holding a thousand distinct ones.

    The base case, checked against the reference. Everything else in the module is this
    operation with different bookkeeping around it.
    """
    batch = _table(rows, distinct_rows=1000)
    made = distinct(batch)
    expected = reference_distinct(Rows.of(batch))
    return {
        **made.as_dict(),
        "expected": len(expected.rows),
        "it_agrees_with_the_reference": bool(agree(Rows.of(made.batch), expected)),
        "it_removed_the_repeats": made.rows < batch.rows,
    }


def distinct_is_sorted(rows: int = 2000) -> dict:
    """The output is in sorted order, which makes it a function of the input.

    A hash based distinct would return the same rows in whatever order the table happened to be
    in, and two runs over differently ordered copies of the same data would produce differently
    ordered answers. Sorting costs n log n and buys a result a caller can compare.
    """
    batch = _table(rows, distinct_rows=500)
    made = distinct(batch).batch
    shuffled = batch.take(np.random.default_rng(163).permutation(batch.rows))
    again = distinct(shuffled).batch
    return {
        "rows": made.rows,
        "it_is_sorted": bool(np.all(np.diff(made.column("shop").values) >= 0)),
        "and_the_shuffled_copy_gives_the_same_order": bool(
            np.array_equal(made.column("shop").values, again.column("shop").values)
        ),
    }


def two_nulls_are_the_same_row(rows: int = 1000) -> dict:
    """Two rows both missing the same value are one distinct row.

    The opposite of the join rule and the one everybody trips over. A join asks whether two rows
    match and a set operation asks whether two rows are the same row, and two rows that are both
    missing a value are the same row. Every standard says so.
    """
    batch = _with_nulls(rows)
    made = distinct(batch)
    column = made.batch.column("shop")
    null_rows = 0 if column.valid is None else int((~column.valid).sum())
    days = len(set(batch.column("day").to_list()))
    return {
        "rows": rows,
        "distinct_rows": made.rows,
        "distinct_null_rows": null_rows,
        "distinct_days": days,
        "the_nulls_collapsed": null_rows <= days,
        "and_they_did_not_all_vanish": null_rows > 0,
        "it_agrees_with_the_reference": bool(
            agree(Rows.of(made.batch), reference_distinct(Rows.of(batch)))
        ),
    }


def a_null_matches_nothing_in_a_join(rows: int = 1000) -> dict:
    """And the same data through a join, where the nulls match nothing at all.

    The two rules side by side, which is the only way to see that they are not a contradiction.
    The same null column groups with itself under distinct and matches nothing under a join.
    """
    batch = _with_nulls(rows)
    nulls = int((~batch.column("shop").valid).sum())
    joined = semi(batch, batch, ["shop"], ["shop"])
    grouped = distinct(batch)
    return {
        "rows": rows,
        "null_rows": nulls,
        "a_self_join_kept": joined.rows,
        "the_nulls_were_dropped": joined.rows == rows - nulls,
        "and_distinct_kept_them": grouped.rows > 0,
        "the_two_rules_differ": joined.rows != rows,
    }


def union_all_keeps_everything(rows: int = 2000) -> dict:
    """A concatenation, which is the cheapest operation here and the only one without keys.

    Worth measuring against union: the duplicates are what cost, so a query that does not care
    about them should say so.
    """
    left = _table(rows, distinct_rows=500, seed=167)
    right = _table(rows, distinct_rows=500, seed=173)
    kept = union_all(left, right)
    once = union(left, right)
    expected = reference_union(Rows.of(left), Rows.of(right))
    return {
        "left": left.rows,
        "right": right.rows,
        "union_all_rows": kept.rows,
        "union_rows": once.rows,
        "it_is_the_sum": kept.rows == left.rows + right.rows,
        "and_the_union_is_far_smaller": once.rows < kept.rows / 4,
        "it_agrees_with_the_reference": bool(agree(Rows.of(kept.batch), expected)),
    }


def a_union_is_a_distinct_over_a_concatenation(rows: int = 2000) -> dict:
    """Which is what it is implemented as, and the measurement says the two agree.

    Not a tautology: the implementation could have deduplicated each side first and then merged,
    which is faster and is wrong when a row appears once on each side, since it appears twice in
    the concatenation and once in the answer.
    """
    left = _table(rows, distinct_rows=400, seed=179)
    right = _table(rows, distinct_rows=400, seed=181)
    made = union(left, right)
    naive = stack([distinct(left).batch, distinct(right).batch])
    return {
        "union_rows": made.rows,
        "deduplicating_each_side_first": naive.rows,
        "the_naive_way_is_too_large": naive.rows > made.rows,
        "by_this_many_rows": naive.rows - made.rows,
        "which_are_the_rows_in_both": naive.rows - made.rows > 0,
    }


def an_intersection_keeps_what_is_in_both(rows: int = 2000) -> dict:
    """The rows in both sides, once each, checked against a set built in Python."""
    left = _table(rows, distinct_rows=300, seed=191)
    right = _table(rows, distinct_rows=300, seed=193)
    made = intersect(left, right)
    lefts = {tuple(one) for one in Rows.of(left).rows}
    rights = {tuple(one) for one in Rows.of(right).rows}
    return {
        **made.as_dict(),
        "expected": len(lefts & rights),
        "it_matches_a_python_set": made.rows == len(lefts & rights),
        "it_is_smaller_than_either_side": made.rows < min(left.rows, right.rows),
    }


def a_difference_keeps_what_is_only_on_the_left(rows: int = 2000) -> dict:
    """The rows in the left and not the right."""
    left = _table(rows, distinct_rows=300, seed=197)
    right = _table(rows, distinct_rows=300, seed=199)
    made = difference(left, right)
    lefts = {tuple(one) for one in Rows.of(left).rows}
    rights = {tuple(one) for one in Rows.of(right).rows}
    return {
        **made.as_dict(),
        "expected": len(lefts - rights),
        "it_matches_a_python_set": made.rows == len(lefts - rights),
    }


def the_three_set_operations_partition(rows: int = 2000) -> dict:
    """Intersect, and the two differences, which together cover the union exactly.

    One property covering all three, and the one that catches an implementation where a row
    equal on some columns and not others falls out of all of them.
    """
    left = _table(rows, distinct_rows=250, seed=211)
    right = _table(rows, distinct_rows=250, seed=223)
    both = intersect(left, right).rows
    only_left = difference(left, right).rows
    only_right = difference(right, left).rows
    whole = union(left, right).rows
    return {
        "in_both": both,
        "only_left": only_left,
        "only_right": only_right,
        "the_union": whole,
        "they_sum_to_the_union": both + only_left + only_right == whole,
    }


def a_set_operation_with_itself_is_a_distinct(rows: int = 2000) -> dict:
    """Every operation against the same table, which has a known answer for each.

    A table unioned with itself is its distinct rows; intersected with itself is the same; and
    subtracted from itself is empty. Three answers from one input, and any implementation that
    got the null rule wrong would fail at least one of them.
    """
    batch = _table(rows, distinct_rows=400, seed=227)
    unique = distinct(batch).rows
    return {
        "rows": rows,
        "distinct": unique,
        "union_with_itself": union(batch, batch).rows,
        "intersect_with_itself": intersect(batch, batch).rows,
        "except_itself": difference(batch, batch).rows,
        "the_union_is_the_distinct": union(batch, batch).rows == unique,
        "and_so_is_the_intersection": intersect(batch, batch).rows == unique,
        "and_the_difference_is_empty": difference(batch, batch).rows == 0,
    }


def a_set_operation_with_nothing_is_the_table(rows: int = 1000) -> dict:
    """Against an empty table, where each operation has its identity."""
    batch = _table(rows, distinct_rows=200, seed=229)
    empty = Batch.empty(batch.schema)
    return {
        "rows": rows,
        "distinct": distinct(batch).rows,
        "union_with_nothing": union(batch, empty).rows,
        "intersect_with_nothing": intersect(batch, empty).rows,
        "except_nothing": difference(batch, empty).rows,
        "the_union_is_the_distinct": union(batch, empty).rows == distinct(batch).rows,
        "the_intersection_is_empty": intersect(batch, empty).rows == 0,
        "and_the_difference_is_the_distinct": difference(batch, empty).rows
        == distinct(batch).rows,
    }


def the_duplicate_share_decides_the_saving(rows: int = 4000) -> dict:
    """How much distinct removes, against how many duplicates there were.

    The table a planner needs: a distinct over a column that is already unique costs a sort and
    saves nothing, and over a column with ten copies of everything it removes ninety percent.
    """
    out = []
    for unique in (100, 500, 2000, 4000):
        batch = _table(rows, distinct_rows=unique, seed=233)
        made = distinct(batch)
        out.append(
            {
                "distinct_rows_asked_for": unique,
                "rows_in": rows,
                "rows_out": made.rows,
                "removed": made.duplicates,
                "share_removed": round(made.duplicates / rows, 3),
            }
        )
    shares = [one["share_removed"] for one in out]
    return {
        "sweep": out,
        "the_saving_falls": shares == sorted(shares, reverse=True),
        "at_the_most_repeated": shares[0],
        "at_the_least": shares[-1],
    }


def a_mismatched_schema_is_refused() -> dict:
    """Two tables with different columns, refused with both schemas named.

    Usually a query that meant to project one side first, so the message names both rather than
    saying they differ.
    """
    left = _table(100)
    right = Batch.from_columns([integer_column("shop", [1, 2, 3])])
    caught = ""
    try:
        union(left, right)
    except SchemaError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_both": "shop" in caught and "amount" in caught,
    }


def a_mismatched_type_is_refused() -> bool:
    """The same columns with a different type on one side."""
    left = Batch.from_columns([integer_column("v", [1, 2, 3])])
    right = Batch.from_columns([floating_column("v", [1.0, 2.0])])
    try:
        union(left, right)
    except SchemaError:
        return True
    return False


def an_empty_table_distincts_to_nothing() -> dict:
    """A distinct over no rows, which is no rows and keeps its schema."""
    batch = Batch.empty(_table(10).schema)
    made = distinct(batch)
    return {
        "rows": made.rows,
        "it_is_empty": made.rows == 0,
        "and_it_kept_its_schema": list(made.batch.schema.names) == ["shop", "amount", "region"],
    }


def every_operation_agrees_with_a_python_set(rows: int = 2000) -> dict:
    """All four against sets of tuples built in Python, which is a second implementation.

    Not the row at a time reference, because the reference has no intersect or except. A Python
    set is a third implementation and it shares nothing with either.
    """
    left = _table(rows, distinct_rows=300, seed=239)
    right = _table(rows, distinct_rows=300, seed=241)
    lefts = {tuple(one) for one in Rows.of(left).rows}
    rights = {tuple(one) for one in Rows.of(right).rows}
    results = {
        "distinct": distinct(left).rows == len(lefts),
        "union": union(left, right).rows == len(lefts | rights),
        "intersect": intersect(left, right).rows == len(lefts & rights),
        "except": difference(left, right).rows == len(lefts - rights),
    }
    return {**results, "they_all_agree": all(results.values())}


def compare_the_operations(rows: int = 2000) -> list[dict]:
    """Every operation on the same pair of tables, which is the module in one table."""
    left = _table(rows, distinct_rows=300, seed=251)
    right = _table(rows, distinct_rows=300, seed=257)
    return [
        one.as_dict()
        for one in (
            distinct(left),
            union_all(left, right),
            union(left, right),
            intersect(left, right),
            difference(left, right),
        )
    ]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "operations": 5,
        "distinct_agrees": distinct_removes_the_repeats()["it_agrees_with_the_reference"],
        "the_output_is_sorted": distinct_is_sorted()["it_is_sorted"],
        "two_nulls_are_one_row": two_nulls_are_the_same_row()["the_nulls_collapsed"],
        "and_a_join_disagrees": a_null_matches_nothing_in_a_join()["the_two_rules_differ"],
        "they_partition": the_three_set_operations_partition()["they_sum_to_the_union"],
        "python_sets_agree": every_operation_agrees_with_a_python_set()["they_all_agree"],
    }
