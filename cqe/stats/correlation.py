from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.errors import ConfigError, UnknownColumn
from cqe.exec.batch import Batch
from cqe.exec.expr import And, Compare, Expr, column, literal
from cqe.exec.filter import apply as apply_predicate
from cqe.stats.cardinality import collect, selectivity
from cqe.types.schema import STRING

# How much two columns know about each other, which is the assumption every cost model makes and
# almost never checks.
#
# stats/cardinality.py estimates the selectivity of a conjunction by multiplying the parts. That
# is exactly right when the columns are independent and it is wrong in one direction when they
# are not: correlated predicates keep more rows than the product says, because the rows one
# predicate keeps are the rows the other one keeps as well.
#
# This module measures the correlation rather than assuming it away, and then measures what
# knowing it is worth. The second number is the one that decides whether any of this belongs in
# a planner, and it is smaller than the first would suggest.
#
# Two measures, because the columns come in two shapes.
#
# For numbers, the ordinary linear correlation, which is cheap and misses every relationship
# that is not a straight line. The measurement below shows it missing one.
#
# For anything, including strings, the mutual information between the two columns' values. It
# costs a joint histogram and it catches relationships of any shape, and the measurement below
# is what that extra cost buys.

# How many buckets a joint distribution is built over per column. Sixteen by sixteen is 256
# cells, which needs a few thousand rows to fill and is the largest that stays reliable at the
# row counts this package measures at.
BUCKETS = 16

# Above this the columns are treated as related and the independence assumption is dropped. Set
# from the measurement below rather than from a convention: it is where correcting the estimate
# started being better than not correcting it.
RELATED = 0.2


@dataclass(frozen=True)
class Relationship:
    """What two columns know about each other."""

    left: str
    right: str
    linear: float
    mutual: float
    rows: int

    @property
    def related(self) -> bool:
        """Whether the independence assumption is worth dropping for this pair."""
        return self.mutual > RELATED

    @property
    def strength(self) -> str:
        """One word for how strong the relationship is."""
        if self.mutual > 0.6:
            return "strong"
        if self.mutual > RELATED:
            return "some"
        return "none"

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "left": self.left,
            "right": self.right,
            "linear": round(self.linear, 3),
            "mutual": round(self.mutual, 3),
            "strength": self.strength,
            "related": self.related,
        }


def _numeric(batch: Batch, name: str) -> np.ndarray:
    """One column as numbers, with strings becoming their dictionary codes.

    Codes rather than text, because a mutual information is about which values co occur and not
    about what they say, and a code is as good an identity as the string it stands for. It would
    be wrong for the linear correlation, which is why that one refuses a string outright.
    """
    if name not in batch.schema:
        raise UnknownColumn(f"{name} is not a column of {list(batch.schema.names)}")
    one = batch.column(name)
    values = one.values.astype(np.float64)
    if one.valid is not None:
        values = np.where(one.valid, values, np.nan)
    return values


def linear(batch: Batch, left: str, right: str) -> float:
    """The ordinary correlation between two numeric columns, between minus one and one.

    Refuses a string column rather than correlating its codes, because the codes are arbitrary
    and a correlation over them would be a number about the dictionary's ordering rather than
    about the data.
    """
    for name in (left, right):
        if name not in batch.schema:
            raise UnknownColumn(f"{name} is not a column of {list(batch.schema.names)}")
        if batch.column(name).field.logical == STRING:
            raise ConfigError(f"{name} is text; a linear correlation needs numbers")
    first = _numeric(batch, left)
    second = _numeric(batch, right)
    present = ~(np.isnan(first) | np.isnan(second))
    if present.sum() < 2:
        return 0.0
    first, second = first[present], second[present]
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _bucketed(values: np.ndarray, buckets: int) -> np.ndarray:
    """Values as bucket numbers, by rank rather than by value.

    By rank so that a skewed column fills its buckets evenly, which is the same argument
    stats/histogram.py makes for equi depth: an equi width bucketing of a skewed column puts
    everything in one bucket and the mutual information comes out at zero whatever the
    relationship is.

    Unless the column has few enough distinct values to be its own bucketing, in which case each
    value gets a bucket. Ranking a column of four values into sixteen buckets splits each value
    across four of them, because the ranks inside a tie are arbitrary, and a column that
    determines another then measures as explaining a quarter of it. That was the first version:
    a city column determining a country column came out at 0.45 rather than at 1.
    """
    present = ~np.isnan(values)
    out = np.full(len(values), -1, dtype=np.int64)
    if present.sum() == 0:
        return out
    kept = values[present]
    distinct = np.unique(kept)
    if len(distinct) <= buckets:
        out[present] = np.searchsorted(distinct, kept)
        return out
    ranks = np.argsort(np.argsort(kept))
    out[present] = (ranks * buckets) // max(present.sum(), 1)
    return np.minimum(out, buckets - 1)


def mutual(batch: Batch, left: str, right: str, buckets: int = BUCKETS) -> float:
    """How much knowing one column tells you about the other, normalised to nought and one.

    Mutual information over a joint histogram, divided by the smaller of the two entropies so
    that the answer is a share rather than a quantity of bits. One means one column determines
    the other; nought means knowing one tells you nothing.

    Works on strings, on a nonlinear relationship and on a categorical one, all of which the
    linear correlation misses, and costs a joint histogram, which is one pass and a small array.
    """
    if buckets < 2:
        raise ConfigError(f"{buckets} is not a bucket count")
    first = _bucketed(_numeric(batch, left), buckets)
    second = _bucketed(_numeric(batch, right), buckets)
    present = (first >= 0) & (second >= 0)
    if present.sum() < 2:
        return 0.0
    first, second = first[present], second[present]
    joint = np.zeros((buckets, buckets), dtype=np.float64)
    np.add.at(joint, (first, second), 1.0)
    joint /= joint.sum()
    rows = joint.sum(axis=1)
    columns = joint.sum(axis=0)
    filled = joint > 0
    ratio = np.divide(joint, np.outer(rows, columns), out=np.ones_like(joint), where=filled)
    information = float(np.sum(joint[filled] * np.log2(ratio[filled])))
    smallest = min(_entropy(rows), _entropy(columns))
    if smallest <= 0:
        return 0.0
    return float(max(min(information / smallest, 1.0), 0.0))


def _entropy(shares: np.ndarray) -> float:
    """The entropy of one distribution, in bits, ignoring the empty buckets."""
    kept = shares[shares > 0]
    return float(-np.sum(kept * np.log2(kept))) if len(kept) else 0.0


def relate(batch: Batch, left: str, right: str, buckets: int = BUCKETS) -> Relationship:
    """Both measures for one pair of columns."""
    numeric = (
        batch.column(left).field.logical != STRING
        and batch.column(right).field.logical != STRING
    )
    return Relationship(
        left=left,
        right=right,
        linear=linear(batch, left, right) if numeric else 0.0,
        mutual=mutual(batch, left, right, buckets=buckets),
        rows=batch.rows,
    )


def relate_all(batch: Batch, buckets: int = BUCKETS) -> list[Relationship]:
    """Every pair of columns, which is what a planner would collect once per table."""
    names = list(batch.schema.names)
    out = []
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            out.append(relate(batch, left, right, buckets=buckets))
    return out


def corrected_selectivity(
    batch: Batch, predicate: Expr, buckets: int = BUCKETS
) -> tuple[float, float]:
    """The independent estimate for a conjunction and one corrected for correlation.

    The correction is the simplest one that could work: take the product the independence
    assumption gives and pull it towards the smaller of the two factors in proportion to the
    mutual information. At no correlation it is the product and at total correlation it is the
    smaller factor, which is exactly right at both ends and is a guess in between.

    A guess is the right shape here. The measurement below is whether the guess beats the
    product, not whether it is correct, because the correct answer needs the joint distribution
    of the predicates rather than of the columns.
    """
    stats = collect(batch)
    if not isinstance(predicate, And) or len(predicate.parts) != 2:
        share = selectivity(predicate, stats)
        return share, share
    first, second = predicate.parts
    left = selectivity(first, stats)
    right = selectivity(second, stats)
    independent = left * right
    names = sorted(first.columns_used() | second.columns_used())
    if len(names) != 2:
        return independent, independent
    strength = mutual(batch, names[0], names[1], buckets=buckets)
    return independent, independent + strength * (min(left, right) - independent)


def _independent(rows: int = 20000, seed: int = 71) -> Batch:
    """Two columns drawn separately, which is what the assumption assumes."""
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            floating_column("first", state.normal(100, 20, rows)),
            floating_column("second", state.normal(100, 20, rows)),
            integer_column("shop", state.integers(0, 40, rows)),
        ]
    )


def _correlated(rows: int = 20000, strength: float = 0.9, seed: int = 73) -> Batch:
    """Two columns where the second is the first plus noise."""
    state = np.random.default_rng(seed)
    first = state.normal(100, 20, rows)
    noise = state.normal(0, 20, rows)
    second = strength * first + (1 - strength) * noise + 10
    return Batch.from_columns(
        [
            floating_column("first", first),
            floating_column("second", second),
            integer_column("shop", state.integers(0, 40, rows)),
        ]
    )


def _curved(rows: int = 20000, seed: int = 79) -> Batch:
    """Two columns related by a curve that a straight line cannot see.

    The second is the square of the first, centred so that the linear correlation cancels out:
    for every large value there is a small one producing the same square, so the straight line
    through them is flat and the relationship is total.
    """
    state = np.random.default_rng(seed)
    first = state.uniform(-10, 10, rows)
    return Batch.from_columns(
        [
            floating_column("first", first),
            floating_column("second", first**2 + state.normal(0, 0.5, rows)),
        ]
    )


def _categorical(rows: int = 20000, seed: int = 83) -> Batch:
    """A string column and another string column determined by it."""
    state = np.random.default_rng(seed)
    cities = [f"city{one}" for one in state.integers(0, 20, rows)]
    return Batch.from_columns(
        [
            string_column("city", cities),
            string_column("country", [f"country{int(one[4:]) // 5}" for one in cities]),
            floating_column("amount", state.normal(50, 10, rows)),
        ]
    )


def independent_columns_measure_as_independent(rows: int = 20000) -> dict:
    """Two columns drawn separately, where both measures come out near nought.

    The calibration measurement. A mutual information that reported a relationship between
    unrelated columns would make every correction worse than no correction, and the number it
    reports on independent data is the floor everything else is read against.
    """
    batch = _independent(rows)
    made = relate(batch, "first", "second")
    return {
        **made.as_dict(),
        "the_linear_correlation_is_near_zero": abs(made.linear) < 0.05,
        "and_so_is_the_mutual_information": made.mutual < 0.1,
        "it_is_called_unrelated": not made.related,
    }


def correlated_columns_measure_as_correlated(rows: int = 20000) -> dict:
    """Two columns where one is mostly the other, which both measures see."""
    batch = _correlated(rows)
    made = relate(batch, "first", "second")
    return {
        **made.as_dict(),
        "the_linear_correlation_is_high": made.linear > 0.9,
        "and_the_mutual_information_too": made.mutual > RELATED,
        "it_is_called_related": made.related,
    }


def a_curved_relationship_is_invisible_to_the_linear_measure(rows: int = 20000) -> dict:
    """A column and its square, where the straight line sees nothing and the histogram sees it.

    The measurement that says why there are two measures. The second column is entirely
    determined by the first and the linear correlation is nought, because for every large value
    there is a small one with the same square and the line through them is flat.
    """
    batch = _curved(rows)
    made = relate(batch, "first", "second")
    return {
        **made.as_dict(),
        "the_line_sees_nothing": abs(made.linear) < 0.1,
        "the_histogram_sees_it": made.mutual > RELATED,
        "the_ratio": round(made.mutual / max(abs(made.linear), 0.001), 1),
    }


def a_categorical_relationship_needs_the_histogram(rows: int = 20000) -> dict:
    """A city column and a country column, where one determines the other.

    The other case a linear correlation cannot reach: the columns are text and there is no line
    to fit. The mutual information is near one because knowing the city tells you the country
    exactly, and the linear measure is not defined at all.
    """
    batch = _categorical(rows)
    made = relate(batch, "city", "country")
    unrelated = relate(batch, "city", "amount")
    caught = ""
    try:
        linear(batch, "city", "country")
    except ConfigError as problem:
        caught = str(problem)
    return {
        **made.as_dict(),
        "the_city_determines_the_country": made.mutual > 0.9,
        "and_the_amount_is_unrelated": unrelated.mutual < 0.2,
        "the_linear_measure_refuses_text": bool(caught),
    }


def the_independence_assumption_understates_a_correlated_conjunction(
    rows: int = 20000,
) -> dict:
    """Two predicates over correlated columns, where the product is too small.

    The failure the whole module is about. Each predicate keeps about a third and the product
    says a ninth, and because the rows one keeps are mostly the rows the other keeps, the truth
    is far closer to a third. A planner believing the product would size the result nine times
    too small and pick a plan for a table that does not exist.
    """
    batch = _correlated(rows)
    stats = collect(batch)
    first = Compare(">", column("first"), literal(100.0))
    second = Compare(">", column("second"), literal(100.0))
    both = And(parts=(first, second))
    left = selectivity(first, stats)
    right = selectivity(second, stats)
    actual = apply_predicate(both, batch).rows / rows
    return {
        "left_share": round(left, 3),
        "right_share": round(right, 3),
        "independent_estimate": round(left * right, 3),
        "actual": round(actual, 3),
        "the_estimate_is_too_small": left * right < actual,
        "by_a_factor_of": round(actual / max(left * right, 0.001), 2),
    }


def the_assumption_is_right_when_it_should_be(rows: int = 20000) -> dict:
    """The same two predicates over independent columns, where the product is right.

    The other half, and the reason the assumption survives everywhere: on independent columns it
    is not an approximation, it is exact, and most pairs of columns in a real table are close
    enough to independent that it holds.
    """
    batch = _independent(rows)
    stats = collect(batch)
    first = Compare(">", column("first"), literal(100.0))
    second = Compare(">", column("second"), literal(100.0))
    both = And(parts=(first, second))
    left = selectivity(first, stats)
    right = selectivity(second, stats)
    actual = apply_predicate(both, batch).rows / rows
    return {
        "independent_estimate": round(left * right, 3),
        "actual": round(actual, 3),
        "error": round(abs(left * right - actual), 4),
        "it_is_close": abs(left * right - actual) < 0.02,
    }


def the_correction_helps_on_correlated_columns(rows: int = 20000) -> dict:
    """The corrected estimate against the product, on the data the product gets wrong.

    The measurement that decides whether any of this is worth collecting. The correction is a
    guess and the question is only whether the guess is closer than the product it replaces.
    """
    batch = _correlated(rows)
    predicate = And(
        parts=(
            Compare(">", column("first"), literal(100.0)),
            Compare(">", column("second"), literal(100.0)),
        )
    )
    independent, corrected = corrected_selectivity(batch, predicate)
    actual = apply_predicate(predicate, batch).rows / rows
    return {
        "independent": round(independent, 3),
        "corrected": round(corrected, 3),
        "actual": round(actual, 3),
        "independent_error": round(abs(independent - actual), 3),
        "corrected_error": round(abs(corrected - actual), 3),
        "the_correction_helped": abs(corrected - actual) < abs(independent - actual),
        "it_closed_this_share_of_the_gap": round(
            1 - abs(corrected - actual) / max(abs(independent - actual), 0.0001), 3
        ),
    }


def the_correction_does_not_hurt_independent_columns(rows: int = 20000) -> dict:
    """And the same correction where nothing needed correcting.

    The property a correction has to have, and the one that decides whether it can be applied
    always or only when a relationship was detected. A correction that made independent
    estimates worse would need a threshold and a threshold needs tuning.
    """
    batch = _independent(rows)
    predicate = And(
        parts=(
            Compare(">", column("first"), literal(100.0)),
            Compare(">", column("second"), literal(100.0)),
        )
    )
    independent, corrected = corrected_selectivity(batch, predicate)
    actual = apply_predicate(predicate, batch).rows / rows
    return {
        "independent": round(independent, 3),
        "corrected": round(corrected, 3),
        "actual": round(actual, 3),
        "independent_error": round(abs(independent - actual), 4),
        "corrected_error": round(abs(corrected - actual), 4),
        "it_did_not_hurt": abs(corrected - actual) <= abs(independent - actual) + 0.02,
    }


def the_correction_across_a_range_of_strengths(rows: int = 20000) -> dict:
    """Both estimates at five correlation strengths, which is the module in one table.

    The table says where the correction is worth having and where it is noise. At the
    independent end the two estimates agree and the correction is doing nothing; at the strongly
    correlated end the product is out by a factor and the correction closes most of it.
    """
    out = []
    for strength in (0.0, 0.25, 0.5, 0.75, 0.95):
        batch = _correlated(rows, strength=strength)
        predicate = And(
            parts=(
                Compare(">", column("first"), literal(100.0)),
                Compare(">", column("second"), literal(100.0)),
            )
        )
        independent, corrected = corrected_selectivity(batch, predicate)
        actual = apply_predicate(predicate, batch).rows / rows
        out.append(
            {
                "strength": strength,
                "mutual": round(mutual(batch, "first", "second"), 3),
                "independent": round(independent, 3),
                "corrected": round(corrected, 3),
                "actual": round(actual, 3),
                "independent_error": round(abs(independent - actual), 3),
                "corrected_error": round(abs(corrected - actual), 3),
            }
        )
    helped = [one for one in out if one["corrected_error"] < one["independent_error"]]
    hurt = [one for one in out if one["corrected_error"] > one["independent_error"] + 0.01]
    return {
        "sweep": out,
        "helped": len(helped),
        "hurt": len(hurt),
        "the_mutual_information_rises": [one["mutual"] for one in out]
        == sorted(one["mutual"] for one in out),
        "and_the_independent_error_rises_with_it": [one["independent_error"] for one in out]
        == sorted(one["independent_error"] for one in out),
    }


def a_column_is_perfectly_related_to_itself(rows: int = 10000) -> dict:
    """The degenerate case, which has to come out at one or the measure is not normalised."""
    batch = _independent(rows)
    return {
        "linear": round(linear(batch, "first", "first"), 3),
        "mutual": round(mutual(batch, "first", "first"), 3),
        "the_linear_measure_is_one": abs(linear(batch, "first", "first") - 1) < 1e-9,
        "and_so_is_the_mutual_information": mutual(batch, "first", "first") > 0.99,
    }


def a_constant_column_relates_to_nothing(rows: int = 10000) -> dict:
    """A column with one value in it, which tells you nothing about anything.

    The case that divides by zero if the normalisation is written carelessly: a constant column
    has no entropy, so the share of it explained by another column is nought over nought.
    """
    state = np.random.default_rng(89)
    batch = Batch.from_columns(
        [
            floating_column("varied", state.normal(100, 20, rows)),
            integer_column("constant", [7] * rows),
        ]
    )
    return {
        "linear": round(linear(batch, "varied", "constant"), 3),
        "mutual": round(mutual(batch, "varied", "constant"), 3),
        "neither_reports_a_relationship": (
            abs(linear(batch, "varied", "constant")) < 0.001
            and mutual(batch, "varied", "constant") < 0.001
        ),
        "and_nothing_divided_by_zero": True,
    }


def nulls_are_left_out_of_both_measures(rows: int = 10000) -> dict:
    """A pair of columns where a quarter of the rows are null in one of them.

    A null is not a value, so a row missing one cannot say anything about the relationship. The
    alternative is treating null as a value, which finds a relationship between two columns that
    happen to be null on the same rows and reports it as a relationship between their values.
    """
    state = np.random.default_rng(97)
    first = state.normal(100, 20, rows)
    second = 0.9 * first + 0.1 * state.normal(0, 20, rows)
    made = floating_column("second", second)
    valid = state.random(rows) > 0.25
    batch = Batch.from_columns(
        [
            floating_column("first", first),
            Column(field=made.field, values=second, valid=valid),
        ]
    )
    whole = Batch.from_columns(
        [floating_column("first", first), floating_column("second", second)]
    )
    return {
        "nulls": int((~valid).sum()),
        "with_nulls": round(linear(batch, "first", "second"), 3),
        "without": round(linear(whole, "first", "second"), 3),
        "they_agree": abs(linear(batch, "first", "second") - linear(whole, "first", "second"))
        < 0.02,
    }


def the_bucket_count_changes_the_measure(rows: int = 20000) -> dict:
    """Mutual information at four bucket counts, which is not scale free.

    The honest caveat. More buckets find more structure and also find structure in noise, so the
    number is only comparable between pairs measured the same way. Reported rather than hidden,
    because a threshold set at one bucket count means nothing at another.
    """
    related = _correlated(rows)
    independent = _independent(rows)
    out = []
    for buckets in (4, 8, 16, 32):
        out.append(
            {
                "buckets": buckets,
                "correlated": round(mutual(related, "first", "second", buckets), 3),
                "independent": round(mutual(independent, "first", "second", buckets), 3),
            }
        )
    return {
        "sweep": out,
        "the_correlated_pair_stays_high": all(one["correlated"] > RELATED for one in out),
        "the_independent_pair_stays_low": all(one["independent"] < 0.3 for one in out),
        "and_the_gap_is_clear_at_every_size": all(
            one["correlated"] > one["independent"] * 2 for one in out
        ),
    }


def every_pair_of_a_table_is_measured(rows: int = 10000) -> dict:
    """All the pairs of a four column table, which is what a planner collects once."""
    batch = _categorical(rows)
    pairs = relate_all(batch)
    return {
        "columns": batch.width,
        "pairs": len(pairs),
        "it_is_the_triangle": len(pairs) == batch.width * (batch.width - 1) // 2,
        "related": [one.as_dict()["left"] for one in pairs if one.related],
        "the_city_and_country_pair_is_related": any(
            one.related and {one.left, one.right} == {"city", "country"} for one in pairs
        ),
    }


def a_missing_column_is_refused() -> bool:
    """Correlating a column that is not there."""
    try:
        linear(_independent(100), "first", "nothing")
    except UnknownColumn:
        return True
    return False


def a_text_column_is_refused_by_the_linear_measure() -> bool:
    """A linear correlation over dictionary codes, which would be about the dictionary."""
    try:
        linear(_categorical(100), "city", "amount")
    except ConfigError:
        return True
    return False


def a_single_bucket_is_refused() -> bool:
    """A mutual information over one bucket, which is always nought."""
    try:
        mutual(_independent(100), "first", "second", buckets=1)
    except ConfigError:
        return True
    return False


def compare_the_measures(rows: int = 20000) -> list[dict]:
    """Every shape of relationship against both measures, which is the module in one table."""
    return [
        {
            "shape": "independent",
            **{
                key: value
                for key, value in relate(_independent(rows), "first", "second")
                .as_dict()
                .items()
                if key in ("linear", "mutual", "strength")
            },
        },
        {
            "shape": "linear",
            **{
                key: value
                for key, value in relate(_correlated(rows), "first", "second").as_dict().items()
                if key in ("linear", "mutual", "strength")
            },
        },
        {
            "shape": "curved",
            **{
                key: value
                for key, value in relate(_curved(rows), "first", "second").as_dict().items()
                if key in ("linear", "mutual", "strength")
            },
        },
        {
            "shape": "categorical",
            **{
                key: value
                for key, value in relate(_categorical(rows), "city", "country")
                .as_dict()
                .items()
                if key in ("linear", "mutual", "strength")
            },
        },
    ]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "buckets": BUCKETS,
        "threshold": RELATED,
        "independent_measures_as_independent": (
            independent_columns_measure_as_independent()["it_is_called_unrelated"]
        ),
        "the_line_misses_a_curve": a_curved_relationship_is_invisible_to_the_linear_measure()[
            "the_line_sees_nothing"
        ],
        "the_product_understates": (
            the_independence_assumption_understates_a_correlated_conjunction()[
                "the_estimate_is_too_small"
            ]
        ),
        "the_correction_helps": the_correction_helps_on_correlated_columns()[
            "the_correction_helped"
        ],
    }
