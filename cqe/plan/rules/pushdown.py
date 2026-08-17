from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cqe.errors import ConfigError, PlanError
from cqe.exec.batch import Batch
from cqe.exec.expr import Expr, all_of, conjuncts
from cqe.plan.logical import (
    Filter,
    Group,
    Join,
    Limit,
    Plan,
    Project,
    Scan,
    Sort,
    check_schema,
    scans,
    table,
    transform,
    walk,
)

# Predicate and projection pushdown, which are the two rewrites that pay for themselves.
#
# A query says filter after join because that is how it reads. An engine that runs it that way
# joins every row and then throws most of them away. Pushing the predicate below the join means
# the join sees fewer rows, and since a join costs the sum of its inputs and produces the
# product, moving a filter down is the largest single change a planner can make to a plan.
#
# Projection pushdown is the same idea on the other axis. A query selecting two columns of forty
# does not need the other thirty eight read from disk, and in a columnar layout not reading them
# is free rather than merely cheap. That is the whole argument for the storage format, and it is
# worth nothing unless the planner works out which columns the query touches.
#
# Both are conditional and the conditions are where the interest is.
#
# A predicate can only be pushed below a join if every column it reads comes from one side. A
# predicate over both sides is a join condition and belongs where it is. Splitting a conjunction
# first is what makes this useful: a and b, where a reads the left and b reads both, pushes a
# down and leaves b behind, and a planner that could not split would push neither.
#
# A predicate cannot be pushed below an aggregate unless it reads only grouping columns. One
# reading an aggregate output is asking about a value that does not exist yet. That is the rule
# that separates where from having, and it is one line here.
#
# The measurements are all of the same shape: build a plan, rewrite it, run both against the
# same data through the executor, and check the answers match while the values touched fall.

# A filter commutes with a projection and with a sort and does not commute with a limit. Limit
# then filter takes the first hundred rows and keeps the matching ones; filter then limit takes
# the first hundred matching rows. Those are different queries and the measurement below shows
# the size of the difference on real data, which is how the omission was found.
MOVABLE_THROUGH = (Project, Sort)


@dataclass
class Rewrite:
    """What a rule did to a plan."""

    before: Plan
    after: Plan
    moved: int
    rule: str

    @property
    def changed(self) -> bool:
        """Whether the plan is different at all."""
        return self.before != self.after

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rule": self.rule,
            "moved": self.moved,
            "changed": self.changed,
            "nodes_before": self.before.nodes(),
            "nodes_after": self.after.nodes(),
        }


def push_predicates(plan: Plan) -> Rewrite:
    """Move every conjunct as far down the tree as it can legally go.

    One pass, children first, so a conjunct pushed into a subtree is then pushed further by the
    rewrite of that subtree. The count of conjuncts that reached a scan is the number worth
    reporting, since a conjunct that stopped at a join has not bought anything.
    """
    moved = _Counter()
    rewritten = transform(plan, lambda node: _push_one(node, moved))
    check_schema(rewritten)
    return Rewrite(before=plan, after=rewritten, moved=moved.value, rule="predicate pushdown")


class _Counter:
    """A mutable count, since transform takes a function and not a closure over a list."""

    def __init__(self) -> None:
        self.value = 0

    def add(self, amount: int = 1) -> None:
        """Record that something moved."""
        self.value += amount


def _push_one(node: Plan, moved: _Counter) -> Plan:
    """Push the conjuncts of one filter as far down as each can go."""
    if not isinstance(node, Filter):
        return node
    parts = conjuncts(node.predicate)
    below = node.input
    stayed: list[Expr] = []
    pushed: list[Expr] = []
    for part in parts:
        placed = _place(below, part)
        if placed is None:
            stayed.append(part)
        else:
            below = placed
            pushed.append(part)
            moved.add()
    if not pushed:
        return node
    if not stayed:
        return below
    return Filter(input=below, predicate=all_of(stayed))


def _place(node: Plan, part: Expr) -> Plan | None:
    """Put one conjunct as far below a node as it can legally go, or refuse.

    Returns None when the conjunct cannot move at all, which is what tells the caller to leave
    it where it was. Returning the unchanged node instead would make a rule that never moves
    anything look like one that moved everything.
    """
    needed = part.columns_used()
    if isinstance(node, Scan):
        if needed <= set(node.schema().names):
            return node.with_predicate(part)
        return None
    if isinstance(node, Filter):
        placed = _place(node.input, part)
        if placed is None:
            return None
        return Filter(input=placed, predicate=node.predicate)
    if isinstance(node, MOVABLE_THROUGH):
        if not needed <= set(node.input.schema().names):
            return None
        placed = _place(node.input, part)
        if placed is None:
            return None
        return _rebuild_single(node, placed)
    if isinstance(node, Join):
        left_names = set(node.left.schema().names)
        right_names = set(node.right.schema().names)
        if needed <= left_names:
            placed = _place(node.left, part)
            if placed is None:
                return None
            return Join(
                left=placed,
                right=node.right,
                left_keys=node.left_keys,
                right_keys=node.right_keys,
                suffix=node.suffix,
            )
        if needed <= right_names:
            placed = _place(node.right, part)
            if placed is None:
                return None
            return Join(
                left=node.left,
                right=placed,
                left_keys=node.left_keys,
                right_keys=node.right_keys,
                suffix=node.suffix,
            )
        return None
    if isinstance(node, Group):
        if needed <= set(node.keys):
            placed = _place(node.input, part)
            if placed is None:
                return None
            return Group(input=placed, keys=node.keys, aggregates=node.aggregates)
        return None
    return None


def _rebuild_single(node: Plan, child: Plan) -> Plan:
    """A copy of a single input node with a different input."""
    if isinstance(node, Project):
        return Project(input=child, names=node.names)
    if isinstance(node, Sort):
        return Sort(input=child, keys=node.keys)
    if isinstance(node, Limit):
        return Limit(input=child, count=node.count, offset=node.offset)
    raise PlanError(f"{type(node).__name__} is not a single input node")


def push_projections(plan: Plan) -> Rewrite:
    """Narrow every scan to the columns something above it reads.

    Computed from the top down, since what a scan needs is the union of what every node above it
    reads, and a node only knows what it reads itself. The traversal collects the requirement on
    the way down and applies it at the leaves.
    """
    needed = _required(plan, set(plan.schema().names))
    dropped = _Counter()

    def narrow(node: Plan) -> Plan:
        if not isinstance(node, Scan):
            return node
        wanted = [name for name in node.table_schema.names if name in needed]
        if not wanted:
            wanted = [node.table_schema.names[0]]
        if len(wanted) < len(node.columns_read()):
            dropped.add(len(node.columns_read()) - len(wanted))
            return node.with_projection(wanted)
        return node

    rewritten = transform(plan, narrow)
    check_schema(rewritten)
    return Rewrite(
        before=plan, after=rewritten, moved=dropped.value, rule="projection pushdown"
    )


def _required(plan: Plan, wanted: set[str]) -> set[str]:
    """Every column any node in a plan reads, given what the top of it produces."""
    needed = set(wanted)
    for node in walk(plan):
        needed |= node.columns_used()
        if isinstance(node, Join):
            needed |= set(node.left_keys) | set(node.right_keys)
    return needed


def push_everything(plan: Plan) -> Rewrite:
    """Both rules, predicates first.

    The order matters and only in one direction. Pushing predicates first lets a scan see the
    columns a pushed predicate reads, so projection pushdown keeps them. The other order would
    narrow a scan to the columns the query outputs, and then the predicate could not be pushed
    into it at all.
    """
    first = push_predicates(plan)
    second = push_projections(first.after)
    return Rewrite(
        before=plan,
        after=second.after,
        moved=first.moved + second.moved,
        rule="both",
    )


def _tables(rows: int = 20_000, keys: int = 500, seed: int = 0) -> tuple[Batch, Batch]:
    """A wide fact table and a narrow dimension table, which is the shape most queries have."""
    import numpy as np  # noqa: PLC0415

    if rows < 1 or keys < 1:
        raise ConfigError(f"{rows} rows over {keys} keys is not a table")
    generator = np.random.default_rng(seed)
    fact = Batch.of(
        k=generator.integers(0, keys, size=rows).tolist(),
        a=generator.integers(0, 1_000, size=rows).tolist(),
        b=generator.integers(0, 1_000, size=rows).tolist(),
        c=generator.integers(0, 1_000, size=rows).tolist(),
        d=generator.integers(0, 1_000, size=rows).tolist(),
    )
    dimension = Batch.of(
        k=list(range(keys)),
        label=[f"d{position:04d}" for position in range(keys)],
        weight=generator.integers(0, 100, size=keys).tolist(),
    )
    return fact, dimension


def _run(plan: Plan, tables: dict[str, Batch]):
    """Execute a logical plan directly, which is enough to check a rewrite.

    A tiny interpreter rather than the physical planner, because the point here is whether the
    rewrite changed the answer and a second implementation would let a shared bug hide. It walks
    the plan and calls the operators in exec directly.
    """
    from cqe.cost.meter import Meter  # noqa: PLC0415

    meter = Meter()
    result = _evaluate(plan, tables, meter)
    return result, meter


def _evaluate(plan: Plan, tables: dict[str, Batch], meter):
    """One node of a logical plan against real data."""
    from cqe.exec import filter as filtering  # noqa: PLC0415
    from cqe.exec import sort as sorting  # noqa: PLC0415
    from cqe.exec.aggregate import hash_aggregate  # noqa: PLC0415
    from cqe.exec.join.hash import hash_join  # noqa: PLC0415

    if isinstance(plan, Scan):
        batch = tables[plan.name]
        if plan.projected is not None:
            batch = batch.select(list(plan.projected))
        meter.touch(batch.rows * batch.width, "scan")
        for part in plan.pushed:
            batch = filtering.apply(part, batch, meter)
        return batch
    if isinstance(plan, Filter):
        return filtering.apply(plan.predicate, _evaluate(plan.input, tables, meter), meter)
    if isinstance(plan, Project):
        return _evaluate(plan.input, tables, meter).select(list(plan.names), meter)
    if isinstance(plan, Join):
        left = _evaluate(plan.left, tables, meter)
        right = _evaluate(plan.right, tables, meter)
        return hash_join(left, right, list(plan.left_keys), list(plan.right_keys), meter).batch
    if isinstance(plan, Group):
        below = _evaluate(plan.input, tables, meter)
        return hash_aggregate(below, list(plan.keys), list(plan.aggregates), meter).batch
    if isinstance(plan, Sort):
        below = _evaluate(plan.input, tables, meter)
        return sorting.sort(below, list(plan.keys), meter)
    if isinstance(plan, Limit):
        below = _evaluate(plan.input, tables, meter)
        return below.slice(plan.offset, plan.offset + plan.count)
    raise PlanError(f"{type(plan).__name__} cannot be evaluated")


def _same(left: Batch, right: Batch) -> bool:
    """Whether two results hold the same rows, ignoring order."""
    if left.names != right.names:
        return False
    return sorted(left.to_rows(), key=str) == sorted(right.to_rows(), key=str)


def a_predicate_below_a_join_is_the_largest_win(rows: int = 20_000) -> dict:
    """Filtering before joining rather than after, which is the rewrite that pays most.

    The join sees the filtered rows instead of all of them, so its build, its probe and its
    output all shrink by the selectivity. Both plans are run against the same data and the rows
    are compared, because a rewrite that is fast and wrong is the failure this arrangement is
    built to catch.
    """
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    fact, dimension = _tables(rows=rows)
    tables = {"f": fact, "d": dimension}
    plan = Filter(
        input=Join(
            left=table("f", fact),
            right=table("d", dimension),
            left_keys=("k",),
            right_keys=("k",),
        ),
        predicate=Compare("<", column("a"), literal(50)),
    )
    rewrite = push_predicates(plan)
    before, before_meter = _run(plan, tables)
    after, after_meter = _run(rewrite.after, tables)
    return {
        "moved": rewrite.moved,
        "same_rows": _same(before, after),
        "before_values": before_meter.values_touched,
        "after_values": after_meter.values_touched,
        "ratio": round(before_meter.values_touched / max(after_meter.values_touched, 1), 2),
        "it_helped": after_meter.values_touched < before_meter.values_touched,
    }


def a_conjunction_splits_and_only_part_moves(rows: int = 20_000) -> dict:
    """A predicate reading both sides of a join stays; one reading a single side moves.

    Splitting first is what makes pushdown useful at all. A planner that treated the conjunction
    as one unit would find it reads both sides and push nothing, so the entire saving depends on
    a two line function in exec/expr.py.
    """
    from cqe.exec.expr import And, Compare, column, literal  # noqa: PLC0415

    fact, dimension = _tables(rows=rows)
    tables = {"f": fact, "d": dimension}
    predicate = And(
        (
            Compare("<", column("a"), literal(200)),
            Compare("<", column("weight"), literal(50)),
            Compare("<", column("a"), column("weight")),
        )
    )
    plan = Filter(
        input=Join(
            left=table("f", fact),
            right=table("d", dimension),
            left_keys=("k",),
            right_keys=("k",),
        ),
        predicate=predicate,
    )
    rewrite = push_predicates(plan)
    before, before_meter = _run(plan, tables)
    after, after_meter = _run(rewrite.after, tables)
    remaining = [node for node in walk(rewrite.after) if isinstance(node, Filter)]
    return {
        "conjuncts": len(conjuncts(predicate)),
        "moved": rewrite.moved,
        "filters_left": len(remaining),
        "same_rows": _same(before, after),
        "ratio": round(before_meter.values_touched / max(after_meter.values_touched, 1), 2),
        "one_stayed_behind": rewrite.moved == 2 and len(remaining) == 1,
    }


def a_predicate_on_an_aggregate_cannot_move(rows: int = 20_000) -> dict:
    """The rule that separates where from having, measured rather than asserted.

    A predicate reading a grouping column can be pushed below the aggregate, because the value
    exists before the grouping. One reading an aggregate output cannot, because the value does
    not exist until the grouping has happened.
    """
    from cqe.exec.aggregate import Aggregate  # noqa: PLC0415
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    fact, _ = _tables(rows=rows)
    tables = {"f": fact}
    grouped = Group(
        input=table("f", fact),
        keys=("k",),
        aggregates=(Aggregate("n", "count_star"), Aggregate("total", "sum", "a")),
    )
    on_key = Filter(input=grouped, predicate=Compare("<", column("k"), literal(100)))
    on_result = Filter(input=grouped, predicate=Compare(">", column("n"), literal(30)))
    key_rewrite = push_predicates(on_key)
    result_rewrite = push_predicates(on_result)
    key_before, _ = _run(on_key, tables)
    key_after, _ = _run(key_rewrite.after, tables)
    return {
        "key_predicate_moved": key_rewrite.moved,
        "result_predicate_moved": result_rewrite.moved,
        "the_key_one_moved": key_rewrite.moved == 1,
        "the_result_one_did_not": result_rewrite.moved == 0,
        "the_answer_survived": _same(key_before, key_after),
    }


def a_projection_narrows_every_scan(rows: int = 20_000) -> dict:
    """Reading two columns of five instead of five, which in a columnar layout is free.

    The saving is the ratio of the column counts and it is exact rather than estimated, because
    a column not read costs nothing at all. That is the property the whole storage format exists
    to provide and it is worth nothing without this rule.
    """
    from cqe.exec.aggregate import Aggregate  # noqa: PLC0415

    fact, _ = _tables(rows=rows)
    tables = {"f": fact}
    plan = Group(
        input=table("f", fact),
        keys=("k",),
        aggregates=(Aggregate("total", "sum", "a"),),
    )
    rewrite = push_projections(plan)
    before, before_meter = _run(plan, tables)
    after, after_meter = _run(rewrite.after, tables)
    narrowed = scans(rewrite.after)[0]
    return {
        "columns_before": len(scans(plan)[0].columns_read()),
        "columns_after": len(narrowed.columns_read()),
        "dropped": rewrite.moved,
        "same_rows": _same(before, after),
        "before_values": before_meter.values_touched,
        "after_values": after_meter.values_touched,
        "ratio": round(before_meter.values_touched / max(after_meter.values_touched, 1), 2),
    }


def the_order_of_the_two_rules_does_not_matter(rows: int = 20_000) -> dict:
    """Both orders reach the same plan, once a scan reports what it reads.

    I expected predicates first to be necessary, on the grounds that narrowing a scan to the
    query's output columns would remove the column a predicate needs and the predicate could
    then never be pushed into it.

    That is true of a scan that does not report the columns its pushed predicates read, and the
    first version of the plan node did not report them. The failure was worse than a missed
    rewrite: predicates first narrowed the scan afterwards and dropped the column its own pushed
    predicate needed, so the plan built cleanly and failed when the predicate was evaluated.

    With Scan.columns_used reporting them, both orders push the predicate, both reach 106081
    values touched, and the order is genuinely free. The measurement is kept because it is the
    thing that would notice if that stopped being true.
    """
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    fact, _ = _tables(rows=rows)
    tables = {"f": fact}
    plan = Project(
        input=Filter(input=table("f", fact), predicate=Compare("<", column("a"), literal(100))),
        names=("k", "b"),
    )
    right_way = push_everything(plan)
    projections_first = push_projections(plan)
    then_predicates = push_predicates(projections_first.after)

    good, good_meter = _run(right_way.after, tables)
    other, other_meter = _run(then_predicates.after, tables)
    reference, _ = _run(plan, tables)
    return {
        "right_way_moved": right_way.moved,
        "other_way_moved": projections_first.moved + then_predicates.moved,
        "right_way_pushed_the_predicate": len(scans(right_way.after)[0].pushed) == 1,
        "other_way_pushed_it": len(scans(then_predicates.after)[0].pushed) == 1,
        "both_are_correct": _same(good, reference) and _same(other, reference),
        "right_way_values": good_meter.values_touched,
        "other_way_values": other_meter.values_touched,
    }


def pushing_through_a_sort_is_safe(rows: int = 20_000) -> dict:
    """A filter below a sort gives the same answer and sorts fewer rows.

    Legal because filtering and sorting commute: the rows that survive are the same set either
    way and the order among them is the same order. Worth measuring rather than assuming, since
    the same is not true of a limit and the next function shows why.
    """
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415
    from cqe.exec.sort import SortKey  # noqa: PLC0415

    fact, _ = _tables(rows=rows)
    tables = {"f": fact}
    plan = Filter(
        input=Sort(input=table("f", fact), keys=(SortKey("b"),)),
        predicate=Compare("<", column("a"), literal(100)),
    )
    rewrite = push_predicates(plan)
    before, before_meter = _run(plan, tables)
    after, after_meter = _run(rewrite.after, tables)
    return {
        "moved": rewrite.moved,
        "same_rows": before.to_rows() == after.to_rows(),
        "before_values": before_meter.values_touched,
        "after_values": after_meter.values_touched,
        "it_helped": after_meter.values_touched < before_meter.values_touched,
    }


def pushing_through_a_limit_is_refused(rows: int = 20_000) -> dict:
    """The case the rule has to refuse, and the size of the difference if it did not.

    A filter below a limit is a different query. Limit then filter takes the first thousand rows
    and keeps the matching ones, which is 98 here. Filter then limit takes the first thousand
    matching rows, which is 1000. Ten times as many rows, and a different set of them.

    The first version of this module listed Limit alongside Project and Sort as a node a filter
    could move through, and this measurement is what found it. The rule now refuses, this checks
    that nothing moved, and the numbers stay in the docstring so the reason is visible.
    """
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    fact, _ = _tables(rows=rows)
    tables = {"f": fact}
    plan = Filter(
        input=Limit(input=table("f", fact), count=1_000),
        predicate=Compare("<", column("a"), literal(100)),
    )
    rewrite = push_predicates(plan)
    before, _ = _run(plan, tables)
    after, _ = _run(rewrite.after, tables)
    return {
        "moved": rewrite.moved,
        "before_rows": before.rows,
        "after_rows": after.rows,
        "the_answers_match": before.rows == after.rows,
        "nothing_moved": rewrite.moved == 0,
        "the_plan_is_unchanged": rewrite.after == plan,
    }


def nothing_moves_when_nothing_can(rows: int = 5_000) -> dict:
    """A plan with no filters, where the rule correctly does nothing.

    Worth its own case because a rule that reports having moved something when it has not is
    indistinguishable from one that works, until a plan gets slower for no visible reason.
    """
    fact, _ = _tables(rows=rows)
    plan = Project(input=table("f", fact), names=("k", "a"))
    rewrite = push_predicates(plan)
    return {
        "moved": rewrite.moved,
        "changed": rewrite.changed,
        "nothing_moved": rewrite.moved == 0,
        "the_plan_is_identical": rewrite.after == plan,
    }


def a_rewrite_never_changes_the_schema(rows: int = 5_000) -> dict:
    """Every rewrite here produces a plan with the same output columns as the one it replaced.

    The property that makes a rewrite a rewrite. Checked by comparing schemas rather than by
    running, since a schema change would be caught by the plan builder anyway and this says the
    builder was not merely lucky.
    """
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    fact, dimension = _tables(rows=rows)
    plan = Project(
        input=Filter(
            input=Join(
                left=table("f", fact),
                right=table("d", dimension),
                left_keys=("k",),
                right_keys=("k",),
            ),
            predicate=Compare("<", column("a"), literal(100)),
        ),
        names=("k", "label"),
    )
    both = push_everything(plan)
    return {
        "before": list(plan.schema().names),
        "after": list(both.after.schema().names),
        "they_match": plan.schema().names == both.after.schema().names,
        "something_moved": both.moved > 0,
    }


def a_rewrite_that_breaks_a_plan_is_refused() -> bool:
    """The schema check runs after every rewrite and fails on a plan that cannot stand."""
    fact, _ = _tables(rows=100)
    plan = Project(input=table("f", fact), names=("k", "a"))
    try:
        check_schema(Project(input=plan, names=("b",)))
    except Exception as problem:
        return "not in" in str(problem) or "produces nothing" in str(problem)
    return False


def an_unknown_node_cannot_be_evaluated() -> bool:
    """The interpreter refuses a node it does not implement rather than returning nothing."""
    fact, _ = _tables(rows=10)
    try:
        _evaluate(Plan(), {"f": fact}, None)
    except PlanError:
        return True
    return False


def an_impossible_table_is_refused() -> bool:
    """The generator refuses a table of no rows."""
    try:
        _tables(rows=0)
    except ConfigError:
        return True
    return False


def rebuilding_a_join_as_a_single_input_is_refused() -> bool:
    """The single input helper is for single input nodes."""
    fact, dimension = _tables(rows=10)
    joined = Join(
        left=table("f", fact),
        right=table("d", dimension),
        left_keys=("k",),
        right_keys=("k",),
    )
    try:
        _rebuild_single(joined, table("f", fact))
    except PlanError:
        return True
    return False


def compare_the_rules(rows: int = 20_000) -> list[dict]:
    """Each rule and both together on one plan, which is the module in one table."""
    from cqe.exec.aggregate import Aggregate  # noqa: PLC0415
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    fact, dimension = _tables(rows=rows)
    tables = {"f": fact, "d": dimension}
    plan = Group(
        input=Filter(
            input=Join(
                left=table("f", fact),
                right=table("d", dimension),
                left_keys=("k",),
                right_keys=("k",),
            ),
            predicate=Compare("<", column("a"), literal(100)),
        ),
        keys=("label",),
        aggregates=(Aggregate("n", "count_star"),),
    )
    reference, base_meter = _run(plan, tables)
    out = [
        {
            "rule": "none",
            "moved": 0,
            "values": base_meter.values_touched,
            "correct": True,
        }
    ]
    for name, rule in (
        ("predicates", push_predicates),
        ("projections", push_projections),
        ("both", push_everything),
    ):
        rewrite = rule(plan)
        result, meter = _run(rewrite.after, tables)
        out.append(
            {
                "rule": name,
                "moved": rewrite.moved,
                "values": meter.values_touched,
                "correct": _same(result, reference),
            }
        )
    return out


def summarise(rows: int = 20_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    join = a_predicate_below_a_join_is_the_largest_win(rows=rows)
    split = a_conjunction_splits_and_only_part_moves(rows=rows)
    columns = a_projection_narrows_every_scan(rows=rows)
    rows_out = compare_the_rules(rows=rows)
    return {
        "join_ratio": join["ratio"],
        "split_moved": split["moved"],
        "projection_ratio": columns["ratio"],
        "every_rule_is_correct": all(row["correct"] for row in rows_out),
        "best_rule": min(rows_out, key=lambda row: row["values"])["rule"],
        "best_values": min(row["values"] for row in rows_out),
    }


def rules() -> Sequence:
    """Every rule in this module, in the order a planner should apply them."""
    return (push_predicates, push_projections)
