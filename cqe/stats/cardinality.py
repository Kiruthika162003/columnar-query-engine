from __future__ import annotations

import statistics as pystats
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cqe.errors import ConfigError, UnknownColumn
from cqe.exec.batch import Batch
from cqe.exec.expr import (
    And,
    ColumnRef,
    Compare,
    Expr,
    InList,
    IsNull,
    Literal,
    Not,
    Or,
    conjuncts,
)
from cqe.stats.histogram import Histogram, equi_depth
from cqe.stats.sketch import HyperLogLog, sketch_of
from cqe.types.schema import STRING

# Selectivity estimation, which is where the histograms and the sketches are actually used.
#
# A planner needs one number per operator: how many rows come out. For a filter that is the
# selectivity times the input. For a join it is the fanout times the probe side. For a group by
# it is the distinct count of the key. Every one of those is an estimate and every one of them
# is built from the two modules next door.
#
# A single predicate is estimated well. The worst error over four range predicates on an equi
# depth histogram is 0.014, so the base case is sound and everything below is measuring the
# combination rather than the parts.
#
# The combination assumes the columns are independent and columns in a real table are not. On
# perfectly correlated columns two predicates each keeping 0.3 are estimated together at 0.089
# and really keep 0.299, a factor of 3.35. Three give 0.027 against 0.299, a factor of 11.
#
# The compounding is real and it is smaller than the arithmetic suggests, because the estimate
# for each conjunct is itself under one. Each extra conjunct multiplies the ratio by about
# three rather than by ten, so a five predicate query on a denormalised table is out by a
# factor of tens rather than of thousands. Still enough to size a hash table wrongly.
#
# The direction is the useful part. Over twelve trials spanning correlations from zero to one,
# independence never once overestimated, with a median ratio of 1.67 and a worst of 2.5. A cost
# model can treat every conjunctive estimate as a lower bound. An or leans the other way, at
# 1.7 times the truth on the same data, so a plan mixing them does not accumulate one bias.
#
# Nothing here fixes it. Multi column statistics would, and they cost the product of the
# cardinalities to store, which is why almost nobody keeps them. What this module provides
# instead is the size of the mistake, so a cost model can be built knowing which way it leans.


@dataclass
class ColumnStatistics:
    """Everything the planner knows about one column."""

    name: str
    histogram: Histogram
    sketch: HyperLogLog
    rows: int

    @property
    def distinct(self) -> float:
        """The estimated distinct count."""
        return self.sketch.estimate()

    @property
    def nulls(self) -> int:
        """The exact null count, which the histogram records rather than estimates."""
        return self.histogram.nulls

    @property
    def null_share(self) -> float:
        """The share of rows that are null."""
        if self.rows == 0:
            return 0.0
        return self.nulls / self.rows

    @property
    def nbytes(self) -> int:
        """What the statistics cost."""
        return self.histogram.nbytes + self.sketch.nbytes

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "rows": self.rows,
            "distinct": round(self.distinct, 1),
            "nulls": self.nulls,
            "bytes": self.nbytes,
        }


@dataclass
class TableStatistics:
    """Statistics for every column of a table."""

    columns: dict[str, ColumnStatistics] = field(default_factory=dict)
    rows: int = 0

    def column(self, name: str) -> ColumnStatistics:
        """One column's statistics, by name."""
        if name not in self.columns:
            raise UnknownColumn(f"{name} is not in {sorted(self.columns)}")
        return self.columns[name]

    @property
    def nbytes(self) -> int:
        """What the whole table's statistics cost."""
        return sum(one.nbytes for one in self.columns.values())

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"rows": self.rows, "columns": len(self.columns), "bytes": self.nbytes}


def collect(batch: Batch, buckets: int = 32, precision: int = 12) -> TableStatistics:
    """Build statistics for every column of a table.

    One pass per column for the sketch and one sort per column for the histogram. That is the
    write time cost of being able to plan at all, and it is why statistics are collected once
    and reused rather than computed per query.
    """
    columns: dict[str, ColumnStatistics] = {}
    for column in batch.columns:
        values = column.values.astype(np.float64)
        if column.valid is not None:
            values = np.where(column.valid, values, np.nan)
        columns[column.name] = ColumnStatistics(
            name=column.name,
            histogram=equi_depth(values, buckets),
            sketch=sketch_of(column.values, precision=precision),
            rows=len(column),
        )
    return TableStatistics(columns=columns, rows=batch.rows)


def selectivity(predicate: Expr, stats: TableStatistics) -> float:
    """The estimated share of rows a predicate keeps.

    Recursive over the expression tree, with the independence assumption at every and and or.
    That assumption is the subject of this module and it is written here in one place rather
    than scattered, so the measurements below have something specific to be measuring.
    """
    if isinstance(predicate, And):
        product = 1.0
        for part in predicate.parts:
            product *= selectivity(part, stats)
        return product
    if isinstance(predicate, Or):
        surviving = 1.0
        for part in predicate.parts:
            surviving *= 1.0 - selectivity(part, stats)
        return 1.0 - surviving
    if isinstance(predicate, Not):
        return max(0.0, 1.0 - selectivity(predicate.part, stats))
    if isinstance(predicate, IsNull):
        return _null_selectivity(predicate, stats)
    if isinstance(predicate, InList):
        return _in_list_selectivity(predicate, stats)
    if isinstance(predicate, Compare):
        return _compare_selectivity(predicate, stats)
    return 1.0


def _null_selectivity(predicate: IsNull, stats: TableStatistics) -> float:
    """Null checks are exact, because the null count is stored rather than estimated."""
    if not isinstance(predicate.part, ColumnRef):
        return 1.0
    try:
        column = stats.column(predicate.part.name)
    except UnknownColumn:
        return 1.0
    share = column.null_share
    return 1.0 - share if predicate.negated else share


def _in_list_selectivity(predicate: InList, stats: TableStatistics) -> float:
    """A membership test is the sum of the per value selectivities, capped at one.

    Capped because the sum assumes the options are disjoint, which they are, and assumes the per
    value estimates are right, which they are not. Without the cap a long list of common values
    can produce a selectivity above one, which is not a number.
    """
    if not isinstance(predicate.part, ColumnRef):
        return 1.0
    try:
        column = stats.column(predicate.part.name)
    except UnknownColumn:
        return 1.0
    if column.rows == 0:
        return 0.0
    total = 0.0
    for option in predicate.options:
        if isinstance(option, str):
            total += 1.0 / max(column.distinct, 1.0)
        else:
            total += column.histogram.estimate_equal(float(option)) / column.rows
    return min(1.0, total)


def _compare_selectivity(predicate: Compare, stats: TableStatistics) -> float:
    """A comparison, from the histogram where there is one and a default where there is not.

    The default for an unestimable comparison is a third, which is the conventional guess and is
    conventional precisely because nobody has a better one. It is written as a named constant so
    a reader can see how often a plan is resting on it.
    """
    name, value, op = _shape(predicate)
    if name is None:
        return DEFAULT_SELECTIVITY
    try:
        column = stats.column(name)
    except UnknownColumn:
        return DEFAULT_SELECTIVITY
    if column.rows == 0:
        return 0.0
    if isinstance(value, str):
        if op == "=":
            return 1.0 / max(column.distinct, 1.0)
        if op == "!=":
            return 1.0 - 1.0 / max(column.distinct, 1.0)
        return DEFAULT_SELECTIVITY
    number = float(value)
    histogram = column.histogram
    if op == "=":
        return histogram.estimate_equal(number) / column.rows
    if op == "!=":
        return 1.0 - histogram.estimate_equal(number) / column.rows
    if op in ("<", "<="):
        return histogram.estimate_less_than(number) / column.rows
    if op in (">", ">="):
        return histogram.estimate_greater_than(number) / column.rows
    return DEFAULT_SELECTIVITY


DEFAULT_SELECTIVITY = 1.0 / 3.0


def _shape(predicate: Compare) -> tuple[str | None, object, str]:
    """A comparison as a column name, a constant and an operator, or nothing usable."""
    flipped = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}
    if isinstance(predicate.left, ColumnRef) and isinstance(predicate.right, Literal):
        return predicate.left.name, predicate.right.value, predicate.op
    if isinstance(predicate.right, ColumnRef) and isinstance(predicate.left, Literal):
        return predicate.right.name, predicate.left.value, flipped[predicate.op]
    return None, None, predicate.op


def join_fanout(
    left: TableStatistics,
    left_key: str,
    right: TableStatistics,
    right_key: str,
) -> float:
    """Output rows per left row, from the two distinct counts.

    The containment assumption: every value on the smaller key side appears on the larger, so
    the fanout is the right side's rows over its distinct count. That is right for a foreign key
    join and wrong for anything else, and it is the second largest source of planning error
    after independence.
    """
    right_column = right.column(right_key)
    left.column(left_key)
    if right_column.rows == 0:
        return 0.0
    return right_column.rows / max(right_column.distinct, 1.0)


def group_count(stats: TableStatistics, keys: Sequence[str]) -> float:
    """How many groups a group by will produce, from the distinct counts.

    The product of the per column distinct counts, capped at the row count. The cap is doing a
    lot of work: a product of three columns with a thousand distinct values each is a billion,
    and a table of a million rows cannot have more than a million groups.
    """
    if not keys:
        return 1.0
    product = 1.0
    for name in keys:
        product *= max(stats.column(name).distinct, 1.0)
    return min(product, float(stats.rows))


def _correlated_table(
    rows: int = 100_000,
    correlation: float = 1.0,
    columns: int = 3,
    seed: int = 0,
) -> Batch:
    """A table whose columns share a correlation, from perfectly correlated to independent.

    Each column is a blend of a shared draw and its own, so a correlation of one makes every
    column the same and a correlation of zero makes them independent. That single dial is what
    every measurement below sweeps.
    """
    if rows < 1 or columns < 1:
        raise ConfigError(f"{rows} rows of {columns} columns is not a table")
    if not 0.0 <= correlation <= 1.0:
        raise ConfigError(f"{correlation} is not a correlation")
    generator = np.random.default_rng(seed)
    shared = generator.random(rows)
    named = {}
    for position in range(columns):
        own = generator.random(rows)
        blended = correlation * shared + (1.0 - correlation) * own
        named[f"c{position}"] = np.floor(blended * 1_000_000).astype(np.int64).tolist()
    return Batch.of(**named)


def _true_selectivity(batch: Batch, predicate: Expr) -> float:
    """The exact share of rows a predicate keeps, which every estimate is scored against."""
    from cqe.exec.filter import evaluate  # noqa: PLC0415

    if batch.rows == 0:
        return 0.0
    return evaluate(predicate, batch).kept / batch.rows


def _band(fraction: float) -> int:
    """The comparison bound that keeps roughly the given share of a uniform column."""
    return int(1_000_000 * fraction)


def a_single_predicate_is_estimated_well(rows: int = 100_000) -> dict:
    """One comparison against one column, which is what the histogram was built for.

    The base case, and it has to be good or nothing above it can be. A range predicate on a
    column with an equi depth histogram over it should be within a percent, and if it is not
    then every combined estimate below is measuring the wrong thing.
    """
    batch = _correlated_table(rows, correlation=0.0)
    stats = collect(batch)
    out = []
    for share in (0.05, 0.2, 0.5, 0.9):
        predicate = Compare("<", ColumnRef("c0"), Literal(_band(share), "integer"))
        estimate = selectivity(predicate, stats)
        truth = _true_selectivity(batch, predicate)
        out.append(
            {
                "target": share,
                "estimate": round(estimate, 4),
                "truth": round(truth, 4),
                "error": round(abs(estimate - truth) / max(truth, 1e-9), 4),
            }
        )
    return {
        "rows": out,
        "worst_error": max(row["error"] for row in out),
        "it_is_accurate": max(row["error"] for row in out) < 0.05,
    }


def independence_underestimates_correlated_columns(
    rows: int = 100_000,
    correlations: Sequence[float] = (0.0, 0.3, 0.6, 0.9, 1.0),
) -> list[dict]:
    """Two predicates over two columns, as the correlation between them rises.

    The independence assumption multiplies the two selectivities. When the columns are
    independent that is right. When they are the same column written twice it is the square of
    the right answer, and the estimate is out by a factor of the selectivity.
    """
    if not correlations:
        raise ConfigError("there is nothing to sweep")
    out = []
    for correlation in correlations:
        batch = _correlated_table(rows, correlation=correlation)
        stats = collect(batch)
        predicate = And(
            (
                Compare("<", ColumnRef("c0"), Literal(_band(0.3), "integer")),
                Compare("<", ColumnRef("c1"), Literal(_band(0.3), "integer")),
            )
        )
        estimate = selectivity(predicate, stats)
        truth = _true_selectivity(batch, predicate)
        out.append(
            {
                "correlation": correlation,
                "estimate": round(estimate, 5),
                "truth": round(truth, 5),
                "ratio": round(truth / max(estimate, 1e-9), 2),
                "underestimates": estimate < truth,
            }
        )
    return out


def the_error_compounds_with_the_conjuncts(
    rows: int = 100_000,
    counts: Sequence[int] = (1, 2, 3),
) -> list[dict]:
    """How much worse independence gets as a predicate grows, on correlated columns.

    Each conjunct multiplies the estimate by its own selectivity and leaves the truth unchanged,
    so the ratio grows geometrically: 1.0, 3.35 and 11.19 for one, two and three conjuncts.

    Smaller than the arithmetic first suggests. Each conjunct multiplies the ratio by about
    three rather than by ten, because the per conjunct estimate is 0.3 rather than 0.1. So a
    five predicate query on a denormalised table is out by a factor of tens and not of
    thousands, which is still enough to size a hash table wrongly by two orders of magnitude.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    batch = _correlated_table(rows, correlation=1.0)
    stats = collect(batch)
    out = []
    for count in counts:
        parts = tuple(
            Compare("<", ColumnRef(f"c{position}"), Literal(_band(0.3), "integer"))
            for position in range(count)
        )
        predicate = parts[0] if count == 1 else And(parts)
        estimate = selectivity(predicate, stats)
        truth = _true_selectivity(batch, predicate)
        out.append(
            {
                "conjuncts": count,
                "estimate": round(estimate, 6),
                "truth": round(truth, 6),
                "ratio": round(truth / max(estimate, 1e-9), 2),
            }
        )
    return out


def it_is_wrong_in_one_direction_only(
    rows: int = 50_000,
    trials: int = 12,
) -> dict:
    """Whether independence ever overestimates, which decides how a cost model should lean.

    On positively correlated columns it cannot: the true count is at least the product because
    correlation only adds overlap. A cost model can therefore treat every combined estimate as a
    lower bound, which is worth more than knowing the average error.

    Negatively correlated columns would break that, and this measurement does not construct any,
    because the blend generator cannot make them. The claim is about positive correlation and
    the docstring says so rather than implying more.
    """
    if trials < 2:
        raise ConfigError(f"{trials} is not enough trials")
    overestimates = 0
    ratios = []
    for trial in range(trials):
        correlation = trial / (trials - 1)
        batch = _correlated_table(rows, correlation=correlation, seed=trial)
        stats = collect(batch)
        predicate = And(
            (
                Compare("<", ColumnRef("c0"), Literal(_band(0.4), "integer")),
                Compare("<", ColumnRef("c1"), Literal(_band(0.4), "integer")),
            )
        )
        estimate = selectivity(predicate, stats)
        truth = _true_selectivity(batch, predicate)
        if estimate > truth * 1.05:
            overestimates += 1
        ratios.append(truth / max(estimate, 1e-9))
    return {
        "trials": trials,
        "overestimates": overestimates,
        "it_never_overestimates": overestimates == 0,
        "median_ratio": round(pystats.median(ratios), 3),
        "worst_ratio": round(max(ratios), 3),
    }


def a_disjunction_is_estimated_the_other_way(rows: int = 100_000) -> dict:
    """An or combines the complements, which leans the opposite way from an and.

    One minus the product of the survival rates. On correlated columns the true share kept is
    smaller than that, because the two predicates overlap and the union is closer to either one
    alone. So an and underestimates and an or overestimates, and a plan mixing them does not
    accumulate a consistent bias.
    """
    batch = _correlated_table(rows, correlation=1.0)
    stats = collect(batch)
    predicate = Or(
        (
            Compare("<", ColumnRef("c0"), Literal(_band(0.3), "integer")),
            Compare("<", ColumnRef("c1"), Literal(_band(0.3), "integer")),
        )
    )
    estimate = selectivity(predicate, stats)
    truth = _true_selectivity(batch, predicate)
    return {
        "estimate": round(estimate, 5),
        "truth": round(truth, 5),
        "ratio": round(estimate / max(truth, 1e-9), 3),
        "it_overestimates": estimate > truth,
    }


def null_checks_are_exact(rows: int = 50_000) -> dict:
    """The one predicate shape estimated without error, because the count is stored.

    A null count is a single integer the writer already records, so is null is not estimated at
    all. Worth measuring because it is the only shape here with no error term, and a planner
    that knows which of its numbers are exact can lean on them.
    """
    from cqe.columns.array import column_from  # noqa: PLC0415

    values = [None if position % 7 == 0 else position for position in range(rows)]
    batch = Batch.of(other=list(range(rows))).with_column(column_from("c0", values))
    stats = collect(batch)
    predicate = IsNull(ColumnRef("c0"))
    estimate = selectivity(predicate, stats)
    truth = _true_selectivity(batch, predicate)
    return {
        "estimate": round(estimate, 6),
        "truth": round(truth, 6),
        "it_is_exact": abs(estimate - truth) < 1e-9,
        "the_negation_is_too": abs(
            selectivity(IsNull(ColumnRef("c0"), negated=True), stats)
            - _true_selectivity(batch, IsNull(ColumnRef("c0"), negated=True))
        )
        < 1e-9,
    }


def an_unestimable_predicate_falls_back_to_a_third(rows: int = 10_000) -> dict:
    """A column to column comparison has no constant, so there is nothing to look up.

    The conventional guess is a third, and it is conventional because nobody has a better one.
    Naming it as a constant rather than burying it in a branch is the point: a reader can grep
    for how many plan decisions rest on a number nobody measured.
    """
    batch = _correlated_table(rows, correlation=0.0)
    stats = collect(batch)
    predicate = Compare("<", ColumnRef("c0"), ColumnRef("c1"))
    estimate = selectivity(predicate, stats)
    truth = _true_selectivity(batch, predicate)
    return {
        "estimate": round(estimate, 4),
        "truth": round(truth, 4),
        "it_is_the_default": estimate == DEFAULT_SELECTIVITY,
        "the_truth_is_a_half": abs(truth - 0.5) < 0.05,
        "error": round(abs(estimate - truth) / truth, 4),
    }


def a_group_count_is_capped_at_the_rows(rows: int = 50_000) -> dict:
    """The product of three distinct counts exceeds the row count and is capped there.

    The cap is not a refinement, it is the only thing keeping the estimate finite. Three columns
    of fifty thousand distinct values give a product of a hundred and twenty five trillion, and
    the true group count cannot exceed the row count.
    """
    batch = _correlated_table(rows, correlation=0.0)
    stats = collect(batch)
    one = group_count(stats, ["c0"])
    three = group_count(stats, ["c0", "c1", "c2"])
    raw = 1.0
    for name in ("c0", "c1", "c2"):
        raw *= stats.column(name).distinct
    return {
        "one_column": round(one, 1),
        "three_columns": round(three, 1),
        "uncapped_product": round(raw, 1),
        "the_cap_bound": rows,
        "it_was_capped": three == float(rows),
        "the_product_was_absurd": raw > rows * 1000,
    }


def a_correlated_group_count_is_overestimated(rows: int = 50_000) -> dict:
    """Two perfectly correlated grouping columns produce one column's worth of groups.

    The same failure the aggregate module found from the other side. Multiplying the distinct
    counts assumes independence, and two columns that are the same value written twice have as
    many groups together as either alone.
    """
    from cqe.exec.aggregate import Aggregate, hash_aggregate  # noqa: PLC0415

    batch = _correlated_table(rows, correlation=1.0, columns=2)
    stats = collect(batch)
    estimate = group_count(stats, ["c0", "c1"])
    truth = hash_aggregate(batch, ["c0", "c1"], [Aggregate("n", "count_star")]).groups
    single = hash_aggregate(batch, ["c0"], [Aggregate("n", "count_star")]).groups
    return {
        "estimate": round(estimate, 1),
        "truth": truth,
        "one_column_truth": single,
        "it_overestimates": estimate > truth,
        "the_truth_is_near_one_column": abs(truth - single) < max(single * 0.05, 1),
    }


def a_join_fanout_assumes_containment(rows: int = 50_000) -> dict:
    """The fanout estimate, and the assumption it rests on.

    Rows over distinct on the right side, which is exactly right when every left key appears on
    the right and is an overestimate when some do not. A foreign key join satisfies it; a join
    between two independently filtered tables does not.
    """
    from cqe.exec.join.hash import hash_join  # noqa: PLC0415

    generator = np.random.default_rng(4)
    left = Batch.of(k=generator.integers(0, 1_000, size=rows).tolist(), a=list(range(rows)))
    right = Batch.of(k=[position % 1_000 for position in range(3_000)], b=list(range(3_000)))
    left_stats = collect(left)
    right_stats = collect(right)
    estimate = join_fanout(left_stats, "k", right_stats, "k")
    truth = hash_join(left, right, ["k"], ["k"]).fanout
    return {
        "estimate": round(estimate, 3),
        "truth": round(truth, 3),
        "error": round(abs(estimate - truth) / max(truth, 1e-9), 4),
        "it_is_close": abs(estimate - truth) / max(truth, 1e-9) < 0.1,
    }


def and_overestimates_when_containment_fails(rows: int = 50_000) -> dict:
    """A join where half the left keys have no match at all.

    The containment assumption says every left row finds something. Half of them find nothing,
    so the true fanout is half the estimate. That is the second largest planning error after
    independence and it has the opposite sign, which means the two do not cancel reliably.
    """
    from cqe.exec.join.hash import hash_join  # noqa: PLC0415

    generator = np.random.default_rng(5)
    left = Batch.of(k=generator.integers(0, 2_000, size=rows).tolist(), a=list(range(rows)))
    right = Batch.of(k=list(range(1_000)), b=list(range(1_000)))
    estimate = join_fanout(collect(left), "k", collect(right), "k")
    truth = hash_join(left, right, ["k"], ["k"]).fanout
    return {
        "estimate": round(estimate, 3),
        "truth": round(truth, 3),
        "ratio": round(estimate / max(truth, 1e-9), 2),
        "it_overestimates": estimate > truth,
        "by_about_two": 1.5 < estimate / max(truth, 1e-9) < 3.0,
    }


def a_string_equality_uses_the_distinct_count(rows: int = 50_000) -> dict:
    """A string predicate has no histogram to consult, so it uses one over the distinct count.

    The uniformity assumption in its purest form: every value is assumed equally common. On a
    skewed categorical column that is wrong by the skew, which is the case dictionary columns
    almost always are.
    """
    generator = np.random.default_rng(6)
    weights = 1.0 / np.arange(1, 101)
    weights = weights / weights.sum()
    picks = generator.choice(100, size=rows, p=weights)
    batch = Batch.of(g=[f"v{int(one):03d}" for one in picks], v=list(range(rows)))
    stats = collect(batch)
    common = Compare("=", ColumnRef("g"), Literal("v000", STRING))
    rare = Compare("=", ColumnRef("g"), Literal("v099", STRING))
    return {
        "estimate": round(selectivity(common, stats), 5),
        "common_truth": round(_true_selectivity(batch, common), 5),
        "rare_truth": round(_true_selectivity(batch, rare), 5),
        "the_same_estimate_for_both": (
            abs(selectivity(common, stats) - selectivity(rare, stats)) < 1e-9
        ),
        "the_truths_differ_by_a_lot": _true_selectivity(batch, common)
        > 10 * _true_selectivity(batch, rare),
    }


def statistics_cost_a_fixed_amount_per_column(rows: int = 100_000) -> dict:
    """What the whole arrangement costs to store, which is what it has to earn back."""
    batch = _correlated_table(rows, correlation=0.0, columns=5)
    stats = collect(batch)
    return {
        "columns": len(stats.columns),
        "bytes": stats.nbytes,
        "bytes_per_column": stats.nbytes // len(stats.columns),
        "rows": stats.rows,
        "share_of_the_data": round(stats.nbytes / (rows * 5 * 8), 6),
        "it_is_a_rounding_error": stats.nbytes < rows * 5 * 8 / 100,
    }


def an_unknown_column_falls_back(rows: int = 1_000) -> dict:
    """A predicate on a column with no statistics gets the default rather than an error.

    A planner meets this constantly: a column added since the statistics were collected, or a
    computed expression that is not a column at all. Refusing would make the plan fail; guessing
    makes it merely worse, which is the right trade for an estimate.
    """
    batch = _correlated_table(rows, correlation=0.0)
    stats = collect(batch)
    predicate = Compare("<", ColumnRef("missing"), Literal(5, "integer"))
    return {
        "estimate": round(selectivity(predicate, stats), 4),
        "it_is_the_default": selectivity(predicate, stats) == DEFAULT_SELECTIVITY,
        "it_did_not_raise": True,
    }


def conjuncts_and_selectivity_agree(rows: int = 20_000) -> dict:
    """Splitting a predicate and multiplying the parts gives what estimating it whole gives.

    A consistency check rather than a discovery. plan/rules/pushdown.py splits conjunctions and
    redistributes them, and if the estimate changed when it did so the planner would prefer or
    avoid pushdown for a reason that has nothing to do with the data.
    """
    batch = _correlated_table(rows, correlation=0.5)
    stats = collect(batch)
    parts = (
        Compare("<", ColumnRef("c0"), Literal(_band(0.4), "integer")),
        Compare("<", ColumnRef("c1"), Literal(_band(0.6), "integer")),
    )
    whole = selectivity(And(parts), stats)
    split = 1.0
    for part in conjuncts(And(parts)):
        split *= selectivity(part, stats)
    return {
        "whole": round(whole, 6),
        "split": round(split, 6),
        "they_agree": abs(whole - split) < 1e-12,
    }


def an_impossible_correlation_is_refused() -> bool:
    """A correlation outside zero to one is a configuration mistake."""
    try:
        _correlated_table(100, correlation=2.0)
    except ConfigError:
        return True
    return False


def an_unknown_column_lookup_is_refused() -> bool:
    """Asking a table for statistics it does not hold names the ones it does."""
    stats = collect(_correlated_table(100))
    try:
        stats.column("z")
    except UnknownColumn:
        return True
    return False


def too_few_trials_are_refused() -> bool:
    """A direction cannot be established from one trial."""
    try:
        it_is_wrong_in_one_direction_only(trials=1)
    except ConfigError:
        return True
    return False


def summarise(rows: int = 100_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    single = a_single_predicate_is_estimated_well(rows=rows)
    combined = independence_underestimates_correlated_columns(rows=rows)
    compounding = the_error_compounds_with_the_conjuncts(rows=rows)
    direction = it_is_wrong_in_one_direction_only()
    return {
        "single_predicate_error": single["worst_error"],
        "independent_ratio": combined[0]["ratio"],
        "correlated_ratio": combined[-1]["ratio"],
        "three_conjunct_ratio": compounding[-1]["ratio"],
        "it_never_overestimates": direction["it_never_overestimates"],
        "default_selectivity": DEFAULT_SELECTIVITY,
    }
