from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.cost.model import estimate
from cqe.errors import ConfigError, PlanError
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.plan.logical import (
    Aggregate,
    Filter,
    Group,
    Join,
    Limit,
    Plan,
    Project,
    Sort,
    SortKey,
    table,
    walk,
)
from cqe.plan.physical import execute

# Charging a plan's measured cost to its individual nodes, which is the question a total does
# not answer.
#
# cost/model.py compares a whole plan's estimate against a whole run's meter and gives a ratio.
# That ratio is what a planner needs to order two plans, and it is nearly useless for improving
# the model, because a total that is out by a fifth could be five nodes each out by a fifth or
# one node out by a factor of three with the others cancelling it. The measurements below say
# which, and the answer is the second one.
#
# The method. Every node of the plan is run as a subtree of its own with a fresh meter, so a
# subtree's cost is measured directly. A node's own cost is then that subtree's cost less its
# children's, which is exact as long as running a subtree twice costs the same. That assumption
# is not taken on trust: it is measured first, because everything after it depends on it.
#
# This costs one run per node rather than one per plan, so it is a diagnostic and not something
# a planner does. Nothing in the engine calls it.


# Cost in the meter's units, which has to match what the model predicts. A probe is counted
# alongside a touch because the model charges for both and the ratio between them is the model's
# business, not the meter's.
def _counted(meter: Meter) -> int:
    """One number for what a run did, in the units the model predicts."""
    return meter.values_touched + meter.hash_probes


# How far a node's estimate may be out before it is called wrong. Wide, because the point is to
# find the nodes that are out by a factor rather than the ones that are out by a fifth.
TOLERABLE = 2.0


@dataclass(frozen=True)
class NodeCost:
    """One node of a plan: what it was predicted to cost and what it cost."""

    node: str
    depth: int
    predicted: float
    counted: int
    predicted_rows: float
    counted_rows: int

    @property
    def independent(self) -> bool:
        """Whether the node's cost can be separated from its children's at all.

        False when the subtraction comes out negative, which means the node made its children
        cheaper than they are on their own. That is a fact about the runner rather than a bug in
        the arithmetic, so it is reported instead of clamped.
        """
        return self.counted >= 0

    @property
    def ratio(self) -> float:
        """Predicted over counted, so above one is an overestimate."""
        if self.predicted == 0 and self.counted == 0:
            return 1.0
        return self.predicted / max(self.counted, 1)

    @property
    def row_ratio(self) -> float:
        """The same for the row count, which is where a cost error usually starts."""
        return self.predicted_rows / max(self.counted_rows, 1)

    @property
    def wrong(self) -> bool:
        """Whether the estimate is out by more than a factor worth reporting."""
        return self.error > TOLERABLE

    @property
    def error(self) -> float:
        """How far out, as a factor, in whichever direction."""
        if not self.independent:
            return float("inf")
        return max(self.ratio, 1 / max(self.ratio, 1e-9))

    def describe(self) -> str:
        """One line for a report."""
        pad = "  " * self.depth
        return f"{pad}{self.node}: predicted {self.predicted:.0f}, counted {self.counted}"

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "node": self.node,
            "depth": self.depth,
            "predicted": round(self.predicted),
            "counted": self.counted,
            "ratio": round(self.ratio, 3),
            "rows_predicted": round(self.predicted_rows),
            "rows_counted": self.counted_rows,
            "row_ratio": round(self.row_ratio, 3),
            "independent": self.independent,
            "wrong": self.wrong,
        }


@dataclass(frozen=True)
class Attribution:
    """Every node of one plan, priced twice."""

    parts: tuple[NodeCost, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ConfigError("a plan has at least one node")

    @property
    def predicted(self) -> float:
        """The whole plan's estimate."""
        return sum(one.predicted for one in self.parts)

    @property
    def counted(self) -> int:
        """The whole plan's measured cost."""
        return sum(one.counted for one in self.parts)

    @property
    def ratio(self) -> float:
        """The whole plan's estimate over its measured cost."""
        return self.predicted / max(self.counted, 1)

    @property
    def separable(self) -> tuple[NodeCost, ...]:
        """The nodes whose own cost the subtraction could recover."""
        return tuple(one for one in self.parts if one.independent)

    @property
    def dependent(self) -> tuple[NodeCost, ...]:
        """The nodes that made their children cheaper, so their own cost came out negative."""
        return tuple(one for one in self.parts if not one.independent)

    @property
    def worst(self) -> NodeCost:
        """The separable node whose estimate is furthest out."""
        candidates = self.separable or self.parts
        return max(candidates, key=lambda one: one.error)

    @property
    def dearest(self) -> NodeCost:
        """The node that actually cost the most."""
        return max(self.parts, key=lambda one: one.counted)

    @property
    def predicted_dearest(self) -> NodeCost:
        """The node the model expected to cost the most."""
        return max(self.parts, key=lambda one: one.predicted)

    @property
    def wrong_nodes(self) -> tuple[NodeCost, ...]:
        """Every separable node out by more than the tolerance."""
        return tuple(one for one in self.separable if one.wrong)

    def of(self, node: str) -> tuple[NodeCost, ...]:
        """Every node of one kind."""
        return tuple(one for one in self.parts if one.node == node)

    def explain(self) -> str:
        """The plan with both numbers against each node."""
        return "\n".join(one.describe() for one in self.parts)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "nodes": len(self.parts),
            "predicted": round(self.predicted),
            "counted": self.counted,
            "ratio": round(self.ratio, 3),
            "wrong_nodes": len(self.wrong_nodes),
            "dependent_nodes": len(self.dependent),
            "worst": self.worst.node,
            "worst_error": round(self.worst.error, 2),
        }


def _depths(plan: Plan) -> dict[int, int]:
    """How far each node sits below the root, keyed by identity."""
    out: dict[int, int] = {id(plan): 0}
    pending = [plan]
    while pending:
        one = pending.pop()
        for child in one.children():
            out[id(child)] = out[id(one)] + 1
            pending.append(child)
    return out


def _subtree_cost(one: Plan, catalogue: Mapping[str, Batch]) -> tuple[int, int]:
    """What running this node and everything under it costs, and how many rows it gives."""
    meter = Meter()
    made = execute(one, catalogue, meter)
    return _counted(meter), made.rows


def attribute(plan: Plan, catalogue: Mapping[str, Batch]) -> Attribution:
    """Charge the measured cost of a plan to its nodes.

    Every subtree is run on its own, and a node's own cost is its subtree's less its children's.
    That is one run per node, which is why this is a diagnostic. The alternative, threading a
    per node meter through the runner, would put the cost of measurement into the thing being
    measured and would need every operator to cooperate.
    """
    depths = _depths(plan)
    order = walk(plan)
    subtree: dict[int, tuple[int, int]] = {}
    predicted_subtree: dict[int, tuple[float, float]] = {}
    for one in order:
        subtree[id(one)] = _subtree_cost(one, catalogue)
        costed = estimate(one)
        predicted_subtree[id(one)] = (costed.total, costed.rows)

    parts: list[NodeCost] = []
    for one in order:
        counted, rows = subtree[id(one)]
        predicted, predicted_rows = predicted_subtree[id(one)]
        own = counted - sum(subtree[id(child)][0] for child in one.children())
        own_predicted = predicted - sum(
            predicted_subtree[id(child)][0] for child in one.children()
        )
        parts.append(
            NodeCost(
                node=type(one).__name__,
                depth=depths[id(one)],
                predicted=own_predicted,
                counted=own,
                predicted_rows=predicted_rows,
                counted_rows=rows,
            )
        )
    parts.sort(key=lambda one: one.depth)
    return Attribution(parts=tuple(parts))


def _tables(rows: int = 8_000, shops: int = 40, seed: int = 11) -> dict[str, Batch]:
    """Two tables, one large and one small, which is the shape most plans have."""
    state = np.random.default_rng(seed)
    facts = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, shops, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("label", [f"kind{one % 8}" for one in range(rows)]),
        ]
    )
    dimension = Batch.from_columns(
        [
            integer_column("shop", np.arange(shops)),
            string_column("region", [f"region{one % 5}" for one in range(shops)]),
        ]
    )
    return {"facts": facts, "shops": dimension}


def _deep(catalogue: Mapping[str, Batch]) -> Plan:
    """A plan with one of nearly every node, which is what makes the errors compound."""
    scan = table("facts", catalogue["facts"])
    filtered = Filter(input=scan, predicate=Compare(">", column("amount"), literal(90.0)))
    joined = Join(
        left=filtered,
        right=table("shops", catalogue["shops"]),
        left_keys=("shop",),
        right_keys=("shop",),
    )
    grouped = Group(
        input=joined,
        keys=("region",),
        aggregates=(Aggregate(name="total", function="sum", source="amount"),),
    )
    ordered = Sort(input=grouped, keys=(SortKey(name="region"),))
    return Limit(input=ordered, count=5)


def a_subtree_costs_the_same_twice(rows: int = 8_000) -> dict:
    """Running the same subtree twice counts the same, which is what the subtraction needs.

    Checked first because every number below is a difference of two measured subtrees, and a
    runner with a cache or a random strategy choice would make those differences meaningless
    without anything looking wrong. It is not obviously true: physical.py picks join and
    aggregate strategies from the data it is given, and if that choice depended on anything
    outside the data the second run could differ.
    """
    catalogue = _tables(rows)
    plan = _deep(catalogue)
    counts = [[_subtree_cost(one, catalogue)[0] for one in walk(plan)] for _ in range(3)]
    return {
        "nodes": len(counts[0]),
        "first": counts[0],
        "they_all_agree": counts[0] == counts[1] == counts[2],
        "total": sum(counts[0]),
    }


def a_node_can_cost_less_than_nothing(rows: int = 8_000) -> dict:
    """A limit above a sort has a negative own cost, and that is the right answer.

    The assumption I started with was that a subtree always costs at least as much as its
    children, so no own cost could come out below zero. It is false, and the plan that breaks
    it is the commonest one there is. physical.py turns a sort under a limit into a partial
    sort, so the sort measured inside the limit does a fifth of the work it does on its own,
    and the subtraction charges the difference to the limit as a negative number.

    Clamping that to zero would have been the obvious repair and it would have been wrong. The
    limit is what caused the saving, so the saving belongs to it, and the own costs still sum to
    what the whole plan cost. What is lost is not correctness but separability: the sort's own
    number is now the cost it has in isolation rather than the cost it had in this plan, and no
    amount of arithmetic recovers both.
    """
    catalogue = _tables(rows)
    ordered = Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="amount"),))
    made = attribute(Limit(input=ordered, count=10), catalogue)
    whole = _subtree_cost(Limit(input=ordered, count=10), catalogue)[0]
    limit = made.of("Limit")[0]
    return {
        "own_costs": [one.counted for one in made.parts],
        "the_limit_is_negative": limit.counted < 0,
        "the_sort_alone_costs": made.of("Sort")[0].counted,
        "the_plan_costs": whole,
        "and_they_still_sum_to_the_plan": made.counted == whole,
        "the_limit_is_not_separable": not limit.independent,
        "separable_nodes": len(made.separable),
        "of": len(made.parts),
    }


def the_total_hides_the_per_node_error(rows: int = 8_000) -> dict:
    """The plan's total is out by a small factor and its worst node is out by a large one.

    The measurement this module exists for. A model checked only at the root looks better than
    it is, because a scan that is overcharged and a join that is undercharged cancel, and the
    node that would be worth fixing is invisible. The spread below is the range of per node
    ratios, and it is wide enough that the total says almost nothing about any of them.
    """
    catalogue = _tables(rows)
    made = attribute(_deep(catalogue), catalogue)
    ratios = [one.ratio for one in made.parts]
    return {
        "total_ratio": round(made.ratio, 3),
        "node_ratios": [round(one, 3) for one in ratios],
        "worst_node": made.worst.node,
        "worst_error": round(made.worst.error, 2),
        "spread": round(max(ratios) / max(min(ratios), 1e-9), 2),
        "the_total_is_out_by_a_factor_of_two": 0.4 < made.ratio < 2.5,
        "while_the_worst_node_is_out_by_thousands": made.worst.error > 1_000,
        "the_total_is_far_closer_than_the_worst_node": made.worst.error
        > 100 * max(made.ratio, 1 / made.ratio),
        "wrong_nodes": [one.node for one in made.wrong_nodes],
        "right_nodes": [one.node for one in made.separable if not one.wrong],
    }


def the_model_and_the_meter_disagree_about_what_a_scan_is(rows: int = 8_000) -> dict:
    """The model charges rows times columns for a scan and the meter charges nothing.

    I expected the scan to be the one node the model gets exactly right, since its row count and
    its column count are both known and neither has to be guessed. The ratio is thirty two
    thousand, which is the largest error anywhere in this module, and it is not an error in the
    formula. The model prices a scan as bytes pulled off storage, which is what storage/file.py
    does. The runner is handed a Batch that is already in memory and an unprojected scan returns
    it, touching nothing. Both are right about the system they describe.

    The consequence reaches back into cost/model.py. Every ratio reported there includes a scan
    term the meter never sees, which is most of why the model looks like a systematic
    overestimate on small plans. The ratios below with the scans removed are the model's real
    accuracy on the part of the plan the runner actually executes.
    """
    catalogue = _tables(rows)
    made = attribute(_deep(catalogue), catalogue)
    scans = made.of("Scan")
    others = [one for one in made.separable if one.node != "Scan"]
    with_scans = made.predicted / max(made.counted, 1)
    predicted_without = sum(one.predicted for one in made.parts if one.node != "Scan")
    counted_without = sum(one.counted for one in made.parts if one.node != "Scan")
    return {
        "scans": len(scans),
        "scan_predicted": [round(one.predicted) for one in scans],
        "scan_counted": [one.counted for one in scans],
        "every_scan_is_charged_nothing": all(one.counted == 0 for one in scans),
        "while_the_model_charges_for_all_of_them": all(one.predicted > 0 for one in scans),
        "the_worst_node_is_a_scan": made.worst.node == "Scan",
        "ratio_with_scans": round(with_scans, 3),
        "ratio_without_scans": round(predicted_without / max(counted_without, 1), 3),
        "the_model_looks_better_without_them": abs(
            predicted_without / max(counted_without, 1) - 1
        )
        < abs(with_scans - 1),
        "other_ratios": [round(one.ratio, 3) for one in others],
    }


def the_row_error_arrives_before_the_cost_error(rows: int = 8_000) -> dict:
    """A node whose row estimate is out is a node whose cost estimate is out, in that order.

    The reason the fix for a cost model is usually a fix to stats/cardinality.py. A node's cost
    is a function of the rows it is given and the rows it produces, so a selectivity guess that
    is out by a factor of two makes every node above it out by about the same factor without any
    of them having a wrong formula.
    """
    catalogue = _tables(rows)
    made = attribute(_deep(catalogue), catalogue)
    usable = [one for one in made.parts if one.counted_rows > 0 and one.counted > 0]
    pairs = [(one.node, round(one.row_ratio, 3), round(one.ratio, 3)) for one in usable]
    row_errors = [abs(np.log(max(one.row_ratio, 1e-9))) for one in usable]
    cost_errors = [abs(np.log(max(one.ratio, 1e-9))) for one in usable]
    together = float(np.corrcoef(row_errors, cost_errors)[0, 1]) if len(usable) > 2 else 0.0
    return {
        "pairs": pairs,
        "correlation": round(together, 3),
        "they_move_together": together > 0.5,
        "the_worst_row_estimate": max(usable, key=lambda one: abs(np.log(one.row_ratio))).node,
    }


def the_error_compounds_up_the_tree(rows: int = 8_000) -> dict:
    """A node's row estimate is out by roughly the product of the errors below it.

    Which is why a plan of six nodes can be out by a factor at the root while no single node has
    a formula that is wrong. Measured as the row error by depth from the leaves up, in logs, so
    that compounding shows as addition.

    It grows to the aggregate and then the limit throws all of it away. A limit's output is the
    smaller of its count and its input, and once the input is comfortably larger than the count
    the estimate is exactly right however wrong everything below it was. So the compounding is
    real and the root is not where to look for it: a plan ending in a limit reports a perfect
    row estimate over five nodes of accumulated error.
    """
    catalogue = _tables(rows)
    made = attribute(_deep(catalogue), catalogue)
    by_depth: dict[int, list[float]] = {}
    for one in made.parts:
        if one.counted_rows > 0:
            by_depth.setdefault(one.depth, []).append(abs(np.log(max(one.row_ratio, 1e-9))))
    depths = sorted(by_depth, reverse=True)
    errors = [round(float(np.mean(by_depth[one])), 4) for one in depths]
    root = made.parts[0]
    return {
        "depths_from_the_leaves": depths,
        "row_errors": errors,
        "the_leaves_are_exact": errors[0] < 0.01,
        "it_grows_on_the_way_up": max(errors) > errors[0],
        "peak": max(errors),
        "peak_at_depth": depths[errors.index(max(errors))],
        "the_root_is_a_limit": root.node == "Limit",
        "and_the_root_is_exact_again": abs(root.row_ratio - 1.0) < 0.01,
        "so_the_root_hides_it": errors[-1] < max(errors),
    }


def the_model_names_the_wrong_dominant_node(rows: int = 8_000) -> dict:
    """The node the model expects to dominate is not always the node that does.

    The practical consequence of a per node error, and the reason a planner that reports where
    a query spent its time from the estimate rather than the meter will send someone to the
    wrong place. Reported as a comparison rather than a claim, because on some plans the model
    gets this right and on others it does not, and which is which is the useful part.

    Plans that cost nothing to run are left out. A projection over a scan has no dearest node in
    the meter's eyes and comparing the model against a table of zeroes would count as a
    disagreement without anything having disagreed.
    """
    catalogue = _tables(rows)
    plans = _plan_set(catalogue)
    rows_out = []
    for name, plan in plans:
        made = attribute(plan, catalogue)
        if made.counted == 0:
            continue
        rows_out.append(
            {
                "plan": name,
                "predicted_dearest": made.predicted_dearest.node,
                "actually_dearest": made.dearest.node,
                "agrees": made.predicted_dearest.node == made.dearest.node,
            }
        )
    agreed = sum(1 for one in rows_out if one["agrees"])
    return {
        "plans": rows_out,
        "agreed": agreed,
        "of": len(rows_out),
        "it_is_usually_right": agreed >= len(rows_out) / 2,
        "but_not_always": agreed < len(rows_out),
    }


def _plan_set(catalogue: Mapping[str, Batch]) -> list[tuple[str, Plan]]:
    """Plans that differ in which node dominates them."""
    facts = table("facts", catalogue["facts"])
    shops = table("shops", catalogue["shops"])
    by_amount = (SortKey(name="amount"),)
    return [
        ("scan", facts),
        (
            "filter",
            Filter(input=facts, predicate=Compare(">", column("amount"), literal(90.0))),
        ),
        ("project", Project(input=facts, names=("id", "amount"))),
        (
            "join",
            Join(left=facts, right=shops, left_keys=("shop",), right_keys=("shop",)),
        ),
        (
            "group",
            Group(
                input=facts,
                keys=("shop",),
                aggregates=(Aggregate(name="total", function="sum", source="amount"),),
            ),
        ),
        ("sort", Sort(input=facts, keys=by_amount)),
        ("limit", Limit(input=Sort(input=facts, keys=by_amount), count=10)),
        ("deep", _deep(catalogue)),
    ]


def the_limit_hole_is_a_pushdown_the_method_cannot_see(rows: int = 8_000) -> dict:
    """The model overcharges a sort under a limit, and attribution cannot say by how much.

    cost/model.py measured this at the root and called it the limit hole: the model prices a
    full sort and the runner does a partial one. Attribution was supposed to say which node the
    hole is in, and it cannot, because the node it is in only behaves that way when its parent
    is present and this method measures every node on its own.

    So the honest report is two numbers that do not reconcile. The sort measured alone costs
    what a full sort costs. The sort measured inside the limit costs a fifth of that, and the
    difference lands on the limit. A method that measures parts in isolation cannot attribute an
    optimisation that only exists when the parts are together, and saying so is more useful than
    picking whichever of the two numbers looks tidier.
    """
    catalogue = _tables(rows)
    ordered = Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="amount"),))
    made = attribute(Limit(input=ordered, count=10), catalogue)
    sort = made.of("Sort")[0]
    limit = made.of("Limit")[0]
    in_context = _subtree_cost(Limit(input=ordered, count=10), catalogue)[0]
    alone = _subtree_cost(ordered, catalogue)[0]
    return {
        "sort_alone": alone,
        "plan_with_the_limit": in_context,
        "the_limit_makes_the_plan_cheaper": in_context < alone,
        "by_this_factor": round(alone / max(in_context, 1), 2),
        "sort_ratio_measured_alone": round(sort.ratio, 3),
        "the_sort_looks_overcharged_alone": sort.ratio > 1.5,
        "the_limit_absorbs_the_difference": limit.counted == in_context - alone,
        "which_makes_it_inseparable": not limit.independent,
    }


def a_projection_costs_nothing_and_is_charged_nothing(rows: int = 8_000) -> dict:
    """A projection selects columns already in memory, and both sides agree it is free.

    The one node where the model and the meter agree exactly, because there is nothing to
    estimate. Included as the control.

    A scan with a projection above it is a plan that costs nothing to run at all, which was not
    the intent and is worth stating: both nodes are free for different reasons. The projection
    is free because selecting columns of a batch copies no values. The scan is free because the
    batch was already in memory, which the model does not believe.
    """
    catalogue = _tables(rows)
    plan = Project(input=table("facts", catalogue["facts"]), names=("id", "amount"))
    made = attribute(plan, catalogue)
    projection = made.of("Project")[0]
    scan = made.of("Scan")[0]
    return {
        "predicted": projection.predicted,
        "counted": projection.counted,
        "both_are_nothing": projection.predicted == 0 and projection.counted == 0,
        "the_scan_below_is_free_too": scan.counted == 0,
        "but_the_model_charges_for_it": scan.predicted > 0,
        "the_whole_plan_runs_for_nothing": made.counted == 0,
    }


def a_subtree_cost_is_not_a_node_cost(rows: int = 8_000) -> dict:
    """A join over a filter, measured as a subtree, is half the filter.

    Reporting a subtree's cost as its root's cost is the mistake this module is a defence
    against, and the number below says how large it is. Measured over a filter rather than over
    two bare scans, because bare scans are free here and a join over two of them would have made
    the subtraction look unnecessary for the wrong reason.
    """
    catalogue = _tables(rows)
    filtered = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(90.0)),
    )
    plan = Join(
        left=filtered,
        right=table("shops", catalogue["shops"]),
        left_keys=("shop",),
        right_keys=("shop",),
    )
    made = attribute(plan, catalogue)
    join = made.of("Join")[0]
    whole = _subtree_cost(plan, catalogue)[0]
    return {
        "subtree_cost": whole,
        "join_own_cost": join.counted,
        "children_cost": whole - join.counted,
        "the_children_are_a_large_share": (whole - join.counted) > whole * 0.3,
        "share_that_is_the_join": round(join.counted / max(whole, 1), 3),
        "reporting_the_subtree_would_overstate_by": round(whole / max(join.counted, 1), 2),
    }


def attribution_costs_a_run_per_node(rows: int = 4_000) -> dict:
    """Attributing a six node plan runs it six times over, which is why nothing calls it.

    Stated as a measurement so the cost is on the record. A per node meter threaded through the
    runner would cost one run, and would need every operator to know which node it belongs to,
    which is a change to nine files to support a diagnostic. This is the cheaper trade for how
    often it is used.
    """
    catalogue = _tables(rows)
    plan = _deep(catalogue)
    once = Meter()
    execute(plan, catalogue, once)
    made = attribute(plan, catalogue)
    total = sum(_subtree_cost(one, catalogue)[0] for one in walk(plan))
    return {
        "nodes": len(made.parts),
        "one_run": _counted(once),
        "attribution_ran": total,
        "it_costs_more_than_one_run": total > _counted(once),
        "by_this_factor": round(total / max(_counted(once), 1), 2),
        "and_the_answer_is_still_one_run_of_work": made.counted == _counted(once),
    }


def a_plan_the_model_cannot_cost_is_refused() -> bool:
    """A node the model has no term for is refused rather than charged nothing."""

    class Strange(Plan):
        @property
        def schema(self):
            raise PlanError("no schema")

    try:
        estimate(Strange())
    except PlanError:
        return True
    return False


def an_empty_attribution_is_refused() -> bool:
    """An attribution with no nodes is refused."""
    try:
        Attribution(parts=())
    except ConfigError:
        return True
    return False


def compare_the_plans(rows: int = 8_000) -> list[dict]:
    """Every plan in the set, priced at the root and at its worst node.

    Reported twice, once with the scans and once without. With them, every plan's worst node is
    a scan and the table says nothing else, because the scan disagreement is a difference of
    definition rather than of accuracy and it is thirty two thousand times larger than anything
    the model gets wrong about the nodes the runner actually runs.
    """
    catalogue = _tables(rows)
    out = []
    for name, plan in _plan_set(catalogue):
        made = attribute(plan, catalogue)
        others = [one for one in made.separable if one.node != "Scan"]
        predicted = sum(one.predicted for one in made.parts if one.node != "Scan")
        counted = sum(one.counted for one in made.parts if one.node != "Scan")
        worst_other = max(others, key=lambda one: one.error) if others else None
        out.append(
            {
                "plan": name,
                "nodes": len(made.parts),
                "total_ratio": round(made.ratio, 3),
                "worst": made.worst.node,
                "worst_error": round(made.worst.error, 2),
                "wrong_nodes": len(made.wrong_nodes),
                "ratio_without_scans": (
                    round(predicted / max(counted, 1), 3) if counted > 0 else 0.0
                ),
                "worst_without_scans": worst_other.node if worst_other else "",
                "worst_error_without_scans": (
                    round(worst_other.error, 2) if worst_other else 0.0
                ),
            }
        )
    return out


def the_root_flatters_the_model(rows: int = 8_000) -> dict:
    """With the scans set aside, a plan's total is still closer to one than its worst node is.

    The finding generalised past the one plan it was found on, and measured without the scan
    term so that it is about the model's accuracy rather than about the definition of a scan. If
    the total were a fair summary these two numbers would be the same. They are not, and the gap
    is what a planner's own explain output would hide.

    One plan goes the other way and it is worth naming. A sort under a limit has a root out by
    twenty one and no separable node out by more than four, because the discrepancy lives in the
    limit, whose own cost the method refuses to state. So a root that looks worse than every one
    of its parts is a signal too: it means the parts do not account for the plan.
    """
    table_of = [
        one
        for one in compare_the_plans(rows)
        if one["worst_without_scans"] and one["ratio_without_scans"] > 0
    ]
    totals = [
        max(one["ratio_without_scans"], 1 / max(one["ratio_without_scans"], 1e-9))
        for one in table_of
    ]
    worst = [one["worst_error_without_scans"] for one in table_of]
    return {
        "plans": len(table_of),
        "totals": [round(one, 2) for one in totals],
        "worst_errors": worst,
        "mean_total_error": round(float(np.mean(totals)), 3),
        "mean_worst_error": round(float(np.mean(worst)), 3),
        "the_root_looks_better": float(np.mean(totals)) < float(np.mean(worst)),
        "by_this_factor": round(float(np.mean(worst)) / max(float(np.mean(totals)), 1e-9), 2),
        "plans_where_the_worst_node_is_worse": sum(
            1 for one, other in zip(totals, worst, strict=True) if other > one * 1.5
        ),
        "plans_where_the_root_is_worse": sum(
            1 for one, other in zip(totals, worst, strict=True) if one > other * 1.5
        ),
    }


def summarise(rows: int = 8_000) -> dict:
    """The findings in one mapping."""
    hidden = the_total_hides_the_per_node_error(rows)
    compounding = the_error_compounds_up_the_tree(rows)
    negative = a_node_can_cost_less_than_nothing(rows)
    scans = the_model_and_the_meter_disagree_about_what_a_scan_is(rows)
    return {
        "the_subtraction_is_repeatable": a_subtree_costs_the_same_twice(rows)["they_all_agree"],
        "a_limit_costs_a_negative_amount": negative["the_limit_is_negative"],
        "a_scan_is_charged_nothing": scans["every_scan_is_charged_nothing"],
        "total_ratio": hidden["total_ratio"],
        "worst_node": hidden["worst_node"],
        "worst_error": hidden["worst_error"],
        "the_leaves_are_exact": compounding["the_leaves_are_exact"],
        "the_root_hides_the_row_error": compounding["so_the_root_hides_it"],
        "the_root_flatters_by": the_root_flatters_the_model(rows)["by_this_factor"],
    }
