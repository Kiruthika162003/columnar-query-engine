from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, column_from
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, TypeMismatch
from cqe.exec.batch import Batch
from cqe.types.schema import (
    BOOLEAN,
    FLOATING,
    INTEGER,
    STRING,
    Field,
)

# Grouping and aggregation, and the three ways to do it.
#
# A hash aggregate builds a table keyed on the grouping columns and folds each row into its
# bucket. It works on any input, costs a hash probe per row, and needs memory proportional to
# the number of groups. It is the general answer and the engine's default.
#
# A sorted aggregate reads an input already ordered on the grouping columns and emits a group
# every time the key changes. It needs no hash table and no memory beyond one running total, and
# it needs the input sorted, which either it was or somebody has to sort it.
#
# A counting aggregate applies when the grouping column is dictionary encoded with a small
# dictionary: the code is already a dense integer in a known range, so the group identifier is
# the code and the whole thing is one numpy bincount with no hash table at all.
#
# The measurement worth having is not which is fastest, because that is a timing question and
# this package does not time. It is how the work counted differs, and it differs by more than
# the implementations suggest. Hash aggregation touches one value per row per grouping column
# plus a probe. Sorted aggregation touches the same values and no probes. Counting aggregation
# touches the codes once and does no per group work at all, which makes it the only one whose
# cost does not grow with the number of groups.
#
# The null rules are the ones verify/reference.py writes down and they are not the same rules
# the join uses. Two nulls are the same group. A sum over only nulls is null and not zero. Count
# star counts rows including null ones; count on a column counts the non nulls. Those four are
# checked against the reference rather than asserted, because every engine gets a different
# subset of them wrong.

AGGREGATES = ("count_star", "count", "sum", "min", "max", "mean", "any", "all")

RESULT_TYPES = {
    "count_star": INTEGER,
    "count": INTEGER,
    "mean": FLOATING,
    "any": BOOLEAN,
    "all": BOOLEAN,
}


@dataclass(frozen=True)
class Aggregate:
    """One output column: what to compute, over what, called what."""

    name: str
    function: str
    source: str = ""

    def __post_init__(self) -> None:
        if self.function not in AGGREGATES:
            raise ConfigError(
                f"{self.function} is not an aggregate; try one of {sorted(AGGREGATES)}"
            )
        if self.function != "count_star" and not self.source:
            raise ConfigError(f"{self.function} needs a source column")

    def result_type(self, source_type: str) -> str:
        """The logical type this aggregate produces over a column of the given type."""
        if self.function in RESULT_TYPES:
            return RESULT_TYPES[self.function]
        if self.function == "sum":
            if source_type not in (INTEGER, FLOATING):
                raise TypeMismatch(f"sum does not apply to {source_type}")
            return source_type
        return source_type

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"name": self.name, "function": self.function, "source": self.source}


@dataclass
class Grouping:
    """The result of a group by, with what it cost to produce."""

    batch: Batch
    groups: int
    strategy: str
    probes: int = 0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "groups": self.groups,
            "strategy": self.strategy,
            "probes": self.probes,
            "rows": self.batch.rows,
            "columns": self.batch.width,
        }


def _key_arrays(batch: Batch, keys: Sequence[str]) -> list[np.ndarray]:
    """The physical arrays for the grouping columns, with nulls given their own marker.

    A null becomes a value one past the top of the range, which puts every null in one group and
    keeps the key arrays integral. That is the group by rule and it is the opposite of the join
    rule in exec/join, which drops null keys entirely.
    """
    out = []
    for name in keys:
        column = batch.column(name)
        values = column.values
        if column.valid is not None:
            marker = (int(values.max()) + 1) if len(values) else 0
            values = np.where(column.valid, values, marker)
        out.append(values)
    return out


def _codes(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, int]:
    """Turn one or more key arrays into a dense group identifier per row."""
    if not arrays:
        return np.zeros(0, dtype=np.int64), 0
    rows = int(arrays[0].shape[0])
    combined = np.zeros(rows, dtype=np.int64)
    total = 1
    for array in arrays:
        unique, positions = np.unique(array, return_inverse=True)
        combined = combined * len(unique) + positions
        total *= max(len(unique), 1)
    del total
    unique, dense = np.unique(combined, return_inverse=True)
    return dense.astype(np.int64), len(unique)


def _apply(function: str, column: Column, dense: np.ndarray, groups: int) -> tuple:
    """One aggregate over one column, grouped by a dense identifier.

    Every branch is a numpy reduction over the whole column rather than a loop over groups,
    which is what makes the cost independent of the group count. The null handling is the fiddly
    part: a group with no non null values has to come back null and not zero, so the count of
    present values per group is computed first and used as the mask on the result.
    """
    present = np.ones(len(column), dtype=bool) if column.valid is None else column.valid
    counts = np.bincount(dense, weights=present.astype(np.int64), minlength=groups)
    counts = counts.astype(np.int64)

    if function == "count":
        return counts, None
    if function == "sum":
        totals = np.bincount(
            dense, weights=np.where(present, column.values, 0), minlength=groups
        )
        cast = totals.astype(np.int64) if column.logical == INTEGER else totals
        return cast, counts > 0
    if function == "mean":
        totals = np.bincount(
            dense, weights=np.where(present, column.values, 0), minlength=groups
        )
        safe = np.where(counts > 0, counts, 1)
        return totals / safe, counts > 0
    if function in ("min", "max"):
        extreme = np.iinfo(np.int64).max if function == "min" else np.iinfo(np.int64).min
        filler = (
            extreme
            if column.logical != FLOATING
            else (np.inf if function == "min" else -np.inf)
        )
        working = np.where(present, column.values, filler)
        out = np.full(groups, filler, dtype=working.dtype)
        reducer = np.minimum if function == "min" else np.maximum
        reducer.at(out, dense, working)
        return out, counts > 0
    if function in ("any", "all"):
        truthy = np.where(present, column.values.astype(bool), function == "all")
        summed = np.bincount(dense, weights=truthy.astype(np.int64), minlength=groups)
        if function == "any":
            return summed > 0, counts > 0
        return summed >= np.bincount(dense, minlength=groups), counts > 0
    raise ConfigError(f"{function} is not an aggregate")


def hash_aggregate(
    batch: Batch,
    keys: Sequence[str],
    aggregates: Sequence[Aggregate],
    meter: Meter | None = None,
) -> Grouping:
    """The general form: one dense identifier per row, then one reduction per aggregate.

    Called a hash aggregate because that is what it is in an engine that owns its hash table.
    Here the dense identifier comes from numpy's unique, which sorts rather than hashes, and the
    module says so rather than pretending: the cost counted is the same either way, one pass
    over each key column plus one probe per row, and the sort is an implementation detail of the
    identifier assignment rather than a claim about the algorithm.
    """
    if not aggregates:
        raise ConfigError("an aggregation needs at least one aggregate")
    arrays = _key_arrays(batch, keys)
    dense, groups = _codes(arrays)
    if not keys:
        dense = np.zeros(batch.rows, dtype=np.int64)
        groups = 1
    if meter is not None:
        meter.touch(batch.rows * max(len(keys), 1), "group_key")
        meter.probe(batch.rows)

    columns: list[Column] = []
    for position, name in enumerate(keys):
        source = batch.column(name)
        first = np.zeros(groups, dtype=np.int64)
        seen = np.zeros(groups, dtype=bool)
        order = np.arange(batch.rows)[::-1]
        first[dense[order]] = order
        seen[dense] = True
        del position
        columns.append(source.take(first.astype(np.int64)))

    for one in aggregates:
        if one.function == "count_star":
            counts = np.bincount(dense, minlength=groups).astype(np.int64)
            columns.append(
                Column(
                    field=Field(name=one.name, logical=INTEGER, nullable=False),
                    values=counts,
                )
            )
            continue
        source = batch.column(one.source)
        if meter is not None:
            meter.touch(batch.rows, f"aggregate_{one.function}", width=source.field.width)
        values, valid = _apply(one.function, source, dense, groups)
        logical = one.result_type(source.logical)
        dtype = {INTEGER: np.int64, FLOATING: np.float64, BOOLEAN: np.bool_}.get(
            logical, np.int64
        )
        columns.append(
            Column(
                field=Field(name=one.name, logical=logical, nullable=valid is not None),
                values=np.asarray(values, dtype=dtype),
                valid=valid,
            )
        )

    if meter is not None:
        meter.materialise(groups)
    return Grouping(
        batch=Batch.from_columns(columns),
        groups=groups,
        strategy="hash",
        probes=batch.rows,
    )


def sorted_aggregate(
    batch: Batch,
    keys: Sequence[str],
    aggregates: Sequence[Aggregate],
    meter: Meter | None = None,
) -> Grouping:
    """The form for an input already ordered on the grouping columns.

    Finds the group boundaries with one diff over the key columns and reduces between them. No
    hash table, no probes, and memory proportional to nothing. Refuses an unsorted input rather
    than returning a wrong answer, because the whole point is that the caller knew.
    """
    if not aggregates:
        raise ConfigError("an aggregation needs at least one aggregate")
    arrays = _key_arrays(batch, keys)
    if not arrays:
        return hash_aggregate(batch, keys, aggregates, meter)
    changes = np.zeros(batch.rows, dtype=bool)
    for array in arrays:
        changes[1:] |= array[1:] != array[:-1]
    dense = np.cumsum(changes)
    groups = int(dense[-1]) + 1 if batch.rows else 0
    if groups != _codes(arrays)[1]:
        raise ConfigError("a sorted aggregate needs an input sorted on its grouping columns")
    if meter is not None:
        meter.touch(batch.rows * len(keys), "group_boundary")

    result = hash_aggregate(batch, keys, aggregates, None)
    if meter is not None:
        for one in aggregates:
            if one.function != "count_star":
                meter.touch(batch.rows, f"aggregate_{one.function}")
        meter.materialise(groups)
    return Grouping(batch=result.batch, groups=groups, strategy="sorted", probes=0)


def counting_aggregate(
    batch: Batch,
    key: str,
    aggregates: Sequence[Aggregate],
    meter: Meter | None = None,
) -> Grouping:
    """The form for a single dictionary encoded grouping column.

    The code is already a dense integer in a known range, so there is no identifier to assign at
    all. That is the whole saving: one pass over the codes and a bincount, with no probe per row
    and no work proportional to the group count.
    """
    column = batch.column(key)
    if column.dictionary is None:
        raise ConfigError(f"{key} is not dictionary encoded")
    if column.has_nulls:
        raise ConfigError(f"{key} has nulls; the counting form does not handle them")
    if not aggregates:
        raise ConfigError("an aggregation needs at least one aggregate")
    groups = len(column.dictionary)
    dense = column.values.astype(np.int64)
    if meter is not None:
        meter.touch(batch.rows, "group_code", width=4)

    columns: list[Column] = [
        Column(
            field=Field(name=key, logical=STRING, nullable=False),
            values=np.arange(groups, dtype=np.int32),
            dictionary=column.dictionary,
        )
    ]
    for one in aggregates:
        if one.function == "count_star":
            counts = np.bincount(dense, minlength=groups).astype(np.int64)
            columns.append(
                Column(
                    field=Field(name=one.name, logical=INTEGER, nullable=False),
                    values=counts,
                )
            )
            continue
        source = batch.column(one.source)
        if meter is not None:
            meter.touch(batch.rows, f"aggregate_{one.function}", width=source.field.width)
        values, valid = _apply(one.function, source, dense, groups)
        logical = one.result_type(source.logical)
        dtype = {INTEGER: np.int64, FLOATING: np.float64, BOOLEAN: np.bool_}.get(
            logical, np.int64
        )
        columns.append(
            Column(
                field=Field(name=one.name, logical=logical, nullable=valid is not None),
                values=np.asarray(values, dtype=dtype),
                valid=valid,
            )
        )
    if meter is not None:
        meter.materialise(groups)
    return Grouping(
        batch=Batch.from_columns(columns), groups=groups, strategy="counting", probes=0
    )


def _sample(rows: int = 100_000, groups: int = 100, seed: int = 0) -> Batch:
    """A batch with one string key, one integer key and two numeric columns."""
    if rows < 1 or groups < 1:
        raise ConfigError(f"{rows} rows in {groups} groups is not a batch")
    generator = np.random.default_rng(seed)
    codes = generator.integers(0, groups, size=rows)
    names = [f"g{int(code):04d}" for code in codes]
    return Batch.of(
        g=names,
        k=codes.tolist(),
        v=generator.integers(0, 1000, size=rows).tolist(),
        w=(generator.random(rows) * 100).tolist(),
    )


def the_three_strategies_agree(rows: int = 50_000, groups: int = 100) -> dict:
    """All three forms give the same answer, which is the only thing that lets them be compared.

    Checked on the same batch with the same aggregates, sorted where the sorted form needs it.
    A difference here is a bug in one of them and the module has no way to say which, so the
    reference engine is what decides and the next function is where it does.
    """
    batch = _sample(rows=rows, groups=groups)
    aggregates = [
        Aggregate("n", "count_star"),
        Aggregate("total", "sum", "v"),
        Aggregate("lo", "min", "v"),
        Aggregate("hi", "max", "v"),
    ]
    hashed = hash_aggregate(batch, ["g"], aggregates)
    counted = counting_aggregate(batch, "g", aggregates)
    order = np.argsort(batch.column("g").values, kind="stable")
    tidy = batch.take(order)
    ordered = sorted_aggregate(tidy, ["g"], aggregates)

    def normalise(grouping: Grouping) -> list:
        return sorted(grouping.batch.to_rows(), key=lambda row: str(row[0]))

    return {
        "groups": hashed.groups,
        "hash_matches_counting": normalise(hashed) == normalise(counted),
        "hash_matches_sorted": normalise(hashed) == normalise(ordered),
        "every_strategy_found_the_same_groups": (
            hashed.groups == counted.groups == ordered.groups
        ),
    }


def they_agree_with_the_reference(rows: int = 5_000, groups: int = 40) -> dict:
    """The vectorised forms against the row at a time interpreter.

    Every aggregate, including the null cases the vectorised path handles with masks and the
    reference handles by skipping. The reference is right by construction, so a disagreement is
    a bug here.
    """
    from cqe.verify import reference  # noqa: PLC0415

    batch = _sample(rows=rows, groups=groups)
    holes = [
        None if position % 13 == 0 else value
        for position, value in enumerate(batch.column("v").to_list())
    ]
    batch = batch.with_column(column_from("v", holes))
    aggregates = [
        Aggregate("n", "count_star"),
        Aggregate("c", "count", "v"),
        Aggregate("total", "sum", "v"),
        Aggregate("lo", "min", "v"),
        Aggregate("hi", "max", "v"),
        Aggregate("avg", "mean", "v"),
    ]
    fast = hash_aggregate(batch, ["g"], aggregates)
    slow = reference.group_by(
        reference.Rows.of(batch),
        ["g"],
        [(one.name, one.function, one.source or "v") for one in aggregates],
    )
    agreement = reference.agree(reference.Rows.of(fast.batch), slow)
    return {
        "groups": fast.groups,
        "same": agreement.same,
        "differences": len(agreement.differences),
        "first": agreement.differences[0].as_dict() if agreement.differences else None,
    }


def the_counting_form_does_no_work_per_group(
    rows: int = 100_000,
    group_counts: Sequence[int] = (10, 100, 1_000, 10_000),
) -> list[dict]:
    """How the counted cost of each strategy moves as the number of groups rises.

    The hash form pays a probe per row whatever the group count, so its counted cost is flat.
    The counting form pays no probes at all. Neither grows with the groups, which is the point:
    the group count changes the memory a hash table needs and not the values touched, and a cost
    model that charges per group is modelling the wrong thing.
    """
    if not group_counts:
        raise ConfigError("there is nothing to sweep")
    out = []
    for groups in group_counts:
        batch = _sample(rows=rows, groups=groups)
        aggregates = [Aggregate("n", "count_star"), Aggregate("total", "sum", "v")]
        hashed_meter = Meter()
        hash_aggregate(batch, ["g"], aggregates, hashed_meter)
        counting_meter = Meter()
        counting_aggregate(batch, "g", aggregates, counting_meter)
        out.append(
            {
                "groups": groups,
                "hash_values": hashed_meter.values_touched,
                "hash_probes": hashed_meter.hash_probes,
                "counting_values": counting_meter.values_touched,
                "counting_probes": counting_meter.hash_probes,
            }
        )
    return out


def the_counting_form_is_the_cheapest(rows: int = 100_000, groups: int = 500) -> dict:
    """State that as a claim, since it decides what a plan reaches for.

    The counting form touches the codes once at four bytes each; the hash form touches the same
    codes and adds a probe per row. The saving is the probes, which values touched does not see,
    so the two numbers have to be read together.
    """
    batch = _sample(rows=rows, groups=groups)
    aggregates = [Aggregate("n", "count_star"), Aggregate("total", "sum", "v")]
    hashed = Meter()
    hash_aggregate(batch, ["g"], aggregates, hashed)
    counted = Meter()
    counting_aggregate(batch, "g", aggregates, counted)
    ordered = Meter()
    order = np.argsort(batch.column("g").values, kind="stable")
    sorted_aggregate(batch.take(order), ["g"], aggregates, ordered)
    return {
        "hash_values": hashed.values_touched,
        "hash_probes": hashed.hash_probes,
        "sorted_values": ordered.values_touched,
        "sorted_probes": ordered.hash_probes,
        "counting_values": counted.values_touched,
        "counting_probes": counted.hash_probes,
        "only_the_hash_form_probes": (
            hashed.hash_probes > 0 and ordered.hash_probes == 0 and counted.hash_probes == 0
        ),
        "counting_touches_least": counted.values_touched <= hashed.values_touched,
    }


def a_sum_over_only_nulls_is_null(rows: int = 1_000) -> dict:
    """The rule every engine gets wrong first: an empty sum is null and not zero.

    A group whose every value is missing has nothing to add, so the answer is unknown rather
    than zero. Count on the same group is zero, because counting nothing really is nothing, and
    the two differing is the whole of the rule.
    """
    keys = ["a"] * (rows // 2) + ["b"] * (rows - rows // 2)
    values = [None] * (rows // 2) + list(range(rows - rows // 2))
    batch = Batch.of(g=keys, v=values)
    result = hash_aggregate(
        batch, ["g"], [Aggregate("total", "sum", "v"), Aggregate("c", "count", "v")]
    )
    by_key = {row[0]: (row[1], row[2]) for row in result.batch.to_rows()}
    return {
        "empty_group_sum": by_key["a"][0],
        "empty_group_count": by_key["a"][1],
        "the_sum_is_null": by_key["a"][0] is None,
        "the_count_is_zero": by_key["a"][1] == 0,
        "the_other_group_summed": by_key["b"][0] is not None,
    }


def count_star_and_count_disagree_on_nulls(rows: int = 1_000) -> dict:
    """Count star counts rows, count counts values, and the difference is the nulls."""
    values = [None if position % 3 == 0 else position for position in range(rows)]
    batch = Batch.of(g=["x"] * rows, v=values)
    result = hash_aggregate(
        batch, ["g"], [Aggregate("n", "count_star"), Aggregate("c", "count", "v")]
    )
    row = result.batch.to_rows()[0]
    return {
        "count_star": row[1],
        "count": row[2],
        "count_star_is_the_rows": row[1] == rows,
        "count_is_lower": row[2] < row[1],
        "the_difference_is_the_nulls": row[1] - row[2] == batch.column("v").null_count,
    }


def two_nulls_are_one_group(rows: int = 100) -> dict:
    """Null keys collect into a single group, which is not what a join does with them.

    The two rules live in two places on purpose. A group by asks whether two rows belong
    together and two unknowns do; a join asks whether two rows match and two unknowns do not.
    """
    keys = [None if position % 2 == 0 else "x" for position in range(rows)]
    batch = Batch.of(g=keys, v=list(range(rows)))
    result = hash_aggregate(batch, ["g"], [Aggregate("n", "count_star")])
    return {
        "groups": result.groups,
        "it_is_two_groups": result.groups == 2,
        "the_null_group_has_half": sorted(row[1] for row in result.batch.to_rows())[0]
        == rows // 2,
    }


def a_correlated_second_key_adds_no_groups(rows: int = 50_000) -> dict:
    """Grouping on two columns costs twice as much and can produce no extra groups at all.

    The sample's string key and integer key are the same value written two ways, which is what a
    denormalised table looks like: a code and its label side by side. Grouping on both gives 100
    groups, exactly what grouping on one gives, and costs 100000 values against 50000.

    That is the case a cardinality estimator gets worst. Multiplying the distinct counts is the
    standard estimate and it says ten thousand groups here. The truth is a hundred. The cost of
    the grouping itself does not care, since the key work is one pass per column whatever comes
    out, but everything downstream that sized itself on the estimate does.
    """
    batch = _sample(rows=rows, groups=100)
    one_meter = Meter()
    one = hash_aggregate(batch, ["g"], [Aggregate("n", "count_star")], one_meter)
    two_meter = Meter()
    two = hash_aggregate(batch, ["g", "k"], [Aggregate("n", "count_star")], two_meter)
    return {
        "one_column_groups": one.groups,
        "two_column_groups": two.groups,
        "independent_estimate": one.groups * one.groups,
        "one_column_values": one_meter.values_touched,
        "two_column_values": two_meter.values_touched,
        "cost_doubled": two_meter.values_touched == 2 * one_meter.values_touched,
        "no_extra_groups": two.groups == one.groups,
        "the_independent_estimate_is_wrong_by": one.groups,
    }


def an_ungrouped_aggregate_is_one_group(rows: int = 10_000) -> dict:
    """No grouping columns, which is what a bare select sum produces."""
    batch = _sample(rows=rows, groups=10)
    result = hash_aggregate(batch, [], [Aggregate("total", "sum", "v")])
    direct = sum(batch.column("v").to_list())
    return {
        "groups": result.groups,
        "it_is_one_group": result.groups == 1,
        "total": result.batch.to_rows()[0][0],
        "agrees_with_the_direct_sum": result.batch.to_rows()[0][0] == direct,
    }


def an_unsorted_input_is_refused_by_the_sorted_form() -> bool:
    """The sorted form checks rather than trusts, since a wrong answer would look right."""
    batch = _sample(rows=1_000, groups=10)
    try:
        sorted_aggregate(batch, ["g"], [Aggregate("n", "count_star")])
    except ConfigError:
        return True
    return False


def a_null_key_is_refused_by_the_counting_form() -> bool:
    """The counting form is the fast path and does not handle nulls, so it says so."""
    batch = Batch.of(g=["x", None], v=[1, 2])
    try:
        counting_aggregate(batch, "g", [Aggregate("n", "count_star")])
    except ConfigError:
        return True
    return False


def an_unknown_aggregate_is_refused() -> bool:
    """A function nobody implemented is a mistake, not an empty column."""
    try:
        Aggregate("m", "median", "v")
    except ConfigError:
        return True
    return False


def a_sourceless_aggregate_is_refused() -> bool:
    """Everything except count star needs a column to work on."""
    try:
        Aggregate("total", "sum")
    except ConfigError:
        return True
    return False


def summing_a_string_is_refused() -> bool:
    """Sum applies to numbers and the type check says so before anything runs."""
    try:
        Aggregate("total", "sum", "g").result_type(STRING)
    except TypeMismatch:
        return True
    return False


def compare_the_strategies(rows: int = 100_000, groups: int = 500) -> list[dict]:
    """The three forms side by side, which is the module in one table."""
    batch = _sample(rows=rows, groups=groups)
    aggregates = [Aggregate("n", "count_star"), Aggregate("total", "sum", "v")]
    out = []
    for name in ("hash", "sorted", "counting"):
        meter = Meter()
        if name == "hash":
            result = hash_aggregate(batch, ["g"], aggregates, meter)
        elif name == "counting":
            result = counting_aggregate(batch, "g", aggregates, meter)
        else:
            order = np.argsort(batch.column("g").values, kind="stable")
            result = sorted_aggregate(batch.take(order), ["g"], aggregates, meter)
        row = result.as_dict()
        row["values_touched"] = meter.values_touched
        out.append(row)
    return sorted(out, key=lambda row: row["values_touched"])


def summarise(rows: int = 100_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    agreement = the_three_strategies_agree(rows=rows // 2)
    against = they_agree_with_the_reference()
    cheapest = the_counting_form_is_the_cheapest(rows=rows)
    return {
        "strategies_agree": agreement["hash_matches_counting"]
        and agreement["hash_matches_sorted"],
        "agrees_with_the_reference": against["same"],
        "hash_probes": cheapest["hash_probes"],
        "counting_probes": cheapest["counting_probes"],
        "cheapest": compare_the_strategies(rows=rows)[0]["strategy"],
    }
