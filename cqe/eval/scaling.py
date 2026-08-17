from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError
from cqe.exec.aggregate import Aggregate, hash_aggregate
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.exec.filter import apply as apply_predicate
from cqe.exec.join.hash import hash_join, nested_loop_join
from cqe.exec.sort import SortKey, order_by, top_k

# How each operator's cost grows with the input, measured rather than asserted.
#
# Every operator in this package has a complexity everybody knows: a filter is linear, a sort is
# n log n, a hash join is linear in both sides. Those are claims about an algorithm and this
# module checks them against the implementation, which is a different thing: an implementation
# with an accidental quadratic in it has the right algorithm and the wrong cost, and nothing
# short of measuring several sizes finds one.
#
# The method is a fit rather than a ratio. Two sizes give a ratio and a ratio cannot tell linear
# from n log n, because over a factor of two they differ by ten percent. Four sizes over a
# factor of sixteen give an exponent, and the exponent tells them apart: linear is one, n log n
# comes out at about 1.1 at these sizes, and quadratic is two.
#
# Nothing here is timed. The counts come from cost/meter.py and are exactly reproducible, so the
# exponent is a property of the code rather than of the machine, and a fit over four points is
# meaningful in a way that a fit over four timings would not be.

# The sizes every measurement runs at. A factor of sixteen from end to end, which is enough to
# separate the exponents and small enough that the suite stays quick.
SIZES = (2500, 5000, 10000, 40000)

# How far from the expected exponent a measurement may be before it counts as a different growth
# rate. Generous, because the fit is over four points and the constant term is not zero.
TOLERANCE = 0.25


@dataclass(frozen=True)
class Growth:
    """How one operator's cost grew with its input."""

    name: str
    sizes: tuple[int, ...]
    costs: tuple[int, ...]
    expected: float

    @property
    def exponent(self) -> float:
        """The fitted exponent, which is the slope of the log log line.

        Fitted rather than taken from the endpoints, because the endpoints are the two points a
        constant term distorts most and the middle points are what say whether the line is
        straight.
        """
        if len(self.sizes) < 2 or min(self.costs) <= 0:
            return 0.0
        logs = np.log(np.array(self.sizes, dtype=np.float64))
        values = np.log(np.array(self.costs, dtype=np.float64))
        return float(np.polyfit(logs, values, 1)[0])

    @property
    def straight(self) -> float:
        """How well a straight line fits the log log points, as one minus the residual share.

        A number near one means the growth really is a power of the size. A number well below it
        means the cost has two terms of different orders and the exponent is an average of them,
        which is the case a single fitted number would otherwise hide.
        """
        if len(self.sizes) < 3 or min(self.costs) <= 0:
            return 0.0
        logs = np.log(np.array(self.sizes, dtype=np.float64))
        values = np.log(np.array(self.costs, dtype=np.float64))
        slope, intercept = np.polyfit(logs, values, 1)
        predicted = slope * logs + intercept
        residual = float(np.sum((values - predicted) ** 2))
        total = float(np.sum((values - values.mean()) ** 2))
        return 1.0 if total == 0 else max(0.0, 1 - residual / total)

    @property
    def matches(self) -> bool:
        """Whether the fit is close enough to what was expected."""
        return abs(self.exponent - self.expected) <= TOLERANCE

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "operator": self.name,
            "expected": self.expected,
            "measured": round(self.exponent, 3),
            "straightness": round(self.straight, 4),
            "matches": self.matches,
            "costs": list(self.costs),
        }


def _table(rows: int, groups: int = 200, seed: int = 149) -> Batch:
    """A table of a given size, with the same shape at every size.

    The group count is fixed rather than scaled, because an aggregate whose group count grows
    with its input is measuring two things at once and the exponent comes out between them.
    """
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, groups, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 8, rows)]),
        ]
    )


def measure(
    name: str,
    action: Callable[[Batch, Meter], object],
    expected: float,
    sizes: Sequence[int] = SIZES,
) -> Growth:
    """Run one operator at several sizes and fit the exponent."""
    if len(sizes) < 2:
        raise ConfigError("a growth measurement needs at least two sizes")
    costs = []
    for rows in sizes:
        meter = Meter()
        action(_table(rows), meter)
        costs.append(meter.values_touched + meter.comparisons + meter.hash_probes)
    return Growth(name=name, sizes=tuple(sizes), costs=tuple(costs), expected=expected)


def _filter(batch: Batch, meter: Meter) -> object:
    """A filter over one column."""
    return apply_predicate(Compare(">", column("amount"), literal(100.0)), batch, meter=meter)


def _sort(batch: Batch, meter: Meter) -> object:
    """A full sort on one key."""
    return order_by(batch, [SortKey(name="amount")], meter=meter)


def _top_k(batch: Batch, meter: Meter) -> object:
    """A partial sort keeping ten rows."""
    return top_k(batch, [SortKey(name="amount")], 10, meter=meter)


def _aggregate(batch: Batch, meter: Meter) -> object:
    """A hash aggregate over a fixed number of groups."""
    return hash_aggregate(
        batch, ["shop"], [Aggregate(name="n", function="count_star", source="")], meter=meter
    )


def _join(batch: Batch, meter: Meter) -> object:
    """A hash join against a fixed dimension."""
    dimension = Batch.from_columns(
        [
            integer_column("shop", np.arange(200)),
            string_column("name", [f"shop{one}" for one in range(200)]),
        ]
    )
    return hash_join(batch, dimension, ["shop"], ["shop"], meter=meter)


def _project(batch: Batch, meter: Meter) -> object:
    """A projection, which should not grow at all."""
    return batch.select(["id", "amount"], meter=meter)


OPERATORS: tuple[tuple[str, Callable, float], ...] = (
    ("filter", _filter, 1.0),
    ("aggregate", _aggregate, 1.0),
    ("join", _join, 1.0),
    ("top k", _top_k, 1.0),
    ("sort", _sort, 1.1),
    ("project", _project, 0.0),
)


def measure_all(sizes: Sequence[int] = SIZES) -> list[Growth]:
    """Every operator, at every size."""
    return [measure(name, action, expected, sizes) for name, action, expected in OPERATORS]


def a_filter_is_linear() -> dict:
    """One pass over one column, which is the simplest thing here and the baseline.

    Every other exponent is read against this one. If a filter came out at anything but one, the
    measurement would be wrong rather than the filter.
    """
    made = measure("filter", _filter, 1.0)
    return {
        **made.as_dict(),
        "it_is_linear": made.matches,
        "and_the_fit_is_straight": made.straight > 0.99,
    }


def a_projection_does_not_grow() -> dict:
    """A projection at four sizes, whose cost does not move at all.

    The claim exec/project.py makes, measured across sizes rather than at one. A projection that
    copied would grow linearly and would look free at any single size.
    """
    made = measure("project", _project, 0.0)
    return {
        **made.as_dict(),
        "every_cost_is_the_same": len(set(made.costs)) == 1,
        "and_it_is_zero": made.costs[0] == 0,
    }


def a_sort_grows_faster_than_linear() -> dict:
    """A sort at four sizes, which comes out above one and well below two.

    The measurement that separates n log n from linear, and the reason four sizes are needed: at
    two points the difference between them is inside the noise of the constant term.
    """
    made = measure("sort", _sort, 1.1)
    linear = measure("filter", _filter, 1.0)
    return {
        **made.as_dict(),
        "it_is_above_linear": made.exponent > linear.exponent,
        "and_below_quadratic": made.exponent < 1.5,
        "the_filter_exponent": round(linear.exponent, 3),
        "the_gap": round(made.exponent - linear.exponent, 3),
    }


def a_partial_sort_is_linear() -> dict:
    """A top ten at four sizes, which is linear where the full sort is not.

    The whole argument for having a partial sort. An argpartition is linear in the input however
    small the limit is, so the exponent drops back to one and the constant is smaller too.
    """
    partial = measure("top k", _top_k, 1.0)
    whole = measure("sort", _sort, 1.1)
    return {
        **partial.as_dict(),
        "it_is_linear": partial.matches,
        "the_full_sort_is_not": whole.exponent > partial.exponent,
        "and_it_costs_less_at_every_size": all(
            one < other for one, other in zip(partial.costs, whole.costs, strict=True)
        ),
        "the_ratio_at_the_largest": round(whole.costs[-1] / max(partial.costs[-1], 1), 2),
    }


def an_aggregate_is_linear_in_its_rows() -> dict:
    """A hash aggregate over a fixed group count, which is linear in the rows.

    Fixed groups on purpose. An aggregate whose group count grows with the input is measuring
    two things at once and its exponent lands between them, which is the shape of measurement
    that produces a number nobody can act on.
    """
    made = measure("aggregate", _aggregate, 1.0)
    return {
        **made.as_dict(),
        "it_is_linear": made.matches,
        "and_the_fit_is_straight": made.straight > 0.99,
    }


def an_aggregate_grows_with_its_groups(rows: int = 20000) -> dict:
    """And the other dimension, held the other way round.

    Written expecting the cost to rise a little with the group count and it did not rise at all:
    a thousandfold more groups came to exactly the same number. The reason is in the accounting
    rather than in the operator. The per row work is a probe and the per group work is a
    materialisation, and the meter keeps those in different fields, so the sum this measurement
    was adding up left the group work out entirely.

    Counting the materialised rows as well makes it visible, and the point survives: the per row
    work does not move at all and the per group work does, and the total rises by well under a
    factor for a thousandfold change in groups. That is the property making a hash aggregate
    usable on a column with many distinct values, and it needed the third field to see.
    """
    out = []
    for groups in (10, 100, 1000, 10000):
        batch = _table(rows, groups=groups)
        meter = Meter()
        _aggregate(batch, meter)
        per_row = meter.values_touched + meter.hash_probes
        out.append(
            {
                "groups": groups,
                "per_row_work": per_row,
                "per_group_work": meter.rows_materialised,
                "total": per_row + meter.rows_materialised,
            }
        )
    totals = [one["total"] for one in out]
    return {
        "rows": rows,
        "sweep": out,
        "the_per_row_work_does_not_move": len({one["per_row_work"] for one in out}) == 1,
        "the_per_group_work_does": len({one["per_group_work"] for one in out}) > 1,
        "the_total_rises": totals == sorted(totals),
        "but_not_by_much": totals[-1] / max(totals[0], 1) < 2,
        "a_thousandfold_in_groups_costs": round(totals[-1] / max(totals[0], 1), 2),
    }


def a_join_is_linear_in_the_larger_side() -> dict:
    """A hash join against a fixed dimension, which is linear in the fact table.

    The property that makes a star schema query affordable: the dimension is built once and
    every fact row is one probe, so the cost is the fact table's size and not the product.
    """
    made = measure("join", _join, 1.0)
    return {
        **made.as_dict(),
        "it_is_linear": made.matches,
        "and_not_quadratic": made.exponent < 1.5,
    }


def a_nested_loop_join_is_quadratic() -> dict:
    """And the algorithm that is quadratic, which is the control.

    Without it the module would only show measurements that came out as expected, and a method
    that has never produced a two cannot be trusted to distinguish one from two.

    The first version of this held the smaller side at four hundred rows while the larger one
    grew, which makes a nested loop linear rather than quadratic: it is quadratic in the product
    and one factor was constant. Scaling both sides together is what the claim needs.
    """

    def action(batch: Batch, meter: Meter) -> object:
        smaller = batch.slice(0, max(batch.rows // 10, 1)).select(["shop"])
        return nested_loop_join(batch, smaller, ["shop"], ["shop"], meter=meter)

    made = measure("nested loop", action, 2.0, sizes=(500, 1000, 2000, 4000))
    hashed = measure("hash", _join, 1.0, sizes=(500, 1000, 2000, 4000))
    return {
        **made.as_dict(),
        "the_hash_join_exponent": round(hashed.exponent, 3),
        "the_nested_loop_is_steeper": made.exponent > hashed.exponent + 0.3,
        "and_the_method_can_see_a_two": made.exponent > 1.5,
    }


def two_sizes_cannot_tell_linear_from_n_log_n() -> dict:
    """Why four sizes rather than two, measured rather than argued.

    Over a factor of two a sort and a filter differ by about ten percent, which is inside what a
    constant term moves. The same two operators over a factor of sixteen are unmistakable, and
    the ratio of ratios is the number that says so.
    """
    small = (5000, 10000)
    wide = (2500, 40000)
    return {
        "over_a_factor_of_two": {
            "filter": round(measure("filter", _filter, 1.0, small).exponent, 3),
            "sort": round(measure("sort", _sort, 1.1, small).exponent, 3),
        },
        "over_a_factor_of_sixteen": {
            "filter": round(measure("filter", _filter, 1.0, wide).exponent, 3),
            "sort": round(measure("sort", _sort, 1.1, wide).exponent, 3),
        },
        "the_narrow_gap": round(
            measure("sort", _sort, 1.1, small).exponent
            - measure("filter", _filter, 1.0, small).exponent,
            3,
        ),
        "the_wide_gap": round(
            measure("sort", _sort, 1.1, wide).exponent
            - measure("filter", _filter, 1.0, wide).exponent,
            3,
        ),
    }


def every_operator_matches_its_expected_growth() -> dict:
    """All six at once, which is the module in one table.

    The measurement a change to any operator would break. An accidental quadratic has the right
    algorithm and the wrong cost, and nothing short of several sizes finds one.
    """
    made = measure_all()
    return {
        "operators": len(made),
        "table": [one.as_dict() for one in made],
        "they_all_match": all(one.matches for one in made),
        "which_did_not": [one.name for one in made if not one.matches],
    }


def every_fit_is_a_straight_line() -> dict:
    """How well a power law describes each operator, which is what the exponent assumes.

    A number near one means the cost really is a power of the size. Anything well below it means
    two terms of different orders and an exponent that is an average of them, and the fit
    quality is what makes that visible rather than hidden inside a single number.
    """
    made = [one for one in measure_all() if min(one.costs) > 0]
    return {
        "operators": len(made),
        "straightness": {one.name: round(one.straight, 4) for one in made},
        "the_worst": round(min(one.straight for one in made), 4),
        "they_are_all_straight": all(one.straight > 0.98 for one in made),
    }


def the_costs_are_reproducible() -> dict:
    """The same measurement twice, which must give identical counts.

    Nothing here is timed, so two runs of the same code over the same generated data produce the
    same numbers exactly. That is what makes a four point fit meaningful: with timings the
    residual would be noise and the exponent would need many more points.
    """
    first = measure_all()
    second = measure_all()
    return {
        "operators": len(first),
        "they_are_identical": all(
            one.costs == other.costs for one, other in zip(first, second, strict=True)
        ),
        "the_exponents_match": all(
            one.exponent == other.exponent for one, other in zip(first, second, strict=True)
        ),
    }


def the_exponent_survives_a_different_size_range() -> dict:
    """The same operators over a different set of sizes, where the exponents hold.

    A fit that changed with the range would be describing the constant term rather than the
    growth, and the number would mean nothing outside the sizes it was fitted on.
    """
    small = measure_all(sizes=(1000, 2000, 4000, 8000))
    large = measure_all(sizes=(5000, 10000, 20000, 40000))
    pairs = list(zip(small, large, strict=True))
    return {
        "small_range": {one.name: round(one.exponent, 3) for one in small},
        "large_range": {one.name: round(one.exponent, 3) for one in large},
        "the_largest_difference": round(
            max(abs(one.exponent - other.exponent) for one, other in pairs), 3
        ),
        "they_agree": all(abs(one.exponent - other.exponent) < 0.2 for one, other in pairs),
    }


def a_single_size_is_refused() -> bool:
    """A growth measurement with one point, which has no slope."""
    try:
        measure("filter", _filter, 1.0, sizes=(1000,))
    except ConfigError:
        return True
    return False


def compare_the_operators() -> list[dict]:
    """Every operator with its expected and measured exponent."""
    return [one.as_dict() for one in measure_all()]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "sizes": len(SIZES),
        "range": SIZES[-1] // SIZES[0],
        "all_match": every_operator_matches_its_expected_growth()["they_all_match"],
        "fits_are_straight": every_fit_is_a_straight_line()["they_are_all_straight"],
        "reproducible": the_costs_are_reproducible()["they_are_identical"],
        "a_quadratic_is_visible": a_nested_loop_join_is_quadratic()[
            "and_the_method_can_see_a_two"
        ],
    }
