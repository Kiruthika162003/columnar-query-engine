from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import PlanError, UnsupportedPlan
from cqe.exec.aggregate import (
    Aggregate,
    counting_aggregate,
    hash_aggregate,
    sorted_aggregate,
)
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, literal
from cqe.exec.expr import column as column_ref
from cqe.exec.filter import apply as apply_predicate
from cqe.exec.join.hash import hash_join, merge_join, nested_loop_join
from cqe.exec.sort import SortKey, order_by, top_k
from cqe.plan.logical import (
    Filter,
    Group,
    Join,
    Limit,
    Plan,
    Project,
    Scan,
    Sort,
    render,
    table,
    walk,
)
from cqe.plan.rules.pushdown import push_everything
from cqe.types.schema import STRING
from cqe.verify.reference import Rows, agree, group_by, inner_join, select, where

# Turning a logical plan into a run, which means picking a strategy for every node that has more
# than one and then executing them bottom up.
#
# The choices are all local. A join looks at its two inputs and picks hash, merge or nested
# loop; an aggregate looks at its key column and picks hash, sorted or counting; a sort looks at
# whether a limit sits above it and picks a full sort or a partial one. None of them looks at
# the whole plan.
#
# That is a real limitation and it is worth being explicit about it rather than pretending the
# choices are optimal. A global chooser would consider that a sorted aggregate leaves its output
# sorted, which a sort above it could then skip, and would sometimes pay for a sort it does not
# need in order to save a larger one later. This does not do that. The reason is not that it is
# hard, it is that plan/rules/ordering.py measured the join reordering as the decision worth
# making and every local choice here as within a factor of two of the best one. A second global
# pass to recover the last factor of two is not worth the way it makes every choice depend on
# every other one.
#
# Every choice is recorded rather than just taken, so a run can be explained afterwards. That is
# the difference between an engine that is fast and one that can be made fast: the first tells
# you the answer and the second tells you why it took as long as it did.

HASH_JOIN = "hash"
MERGE_JOIN = "merge"
NESTED_LOOP_JOIN = "nested loop"
HASH_GROUP = "hash"
SORTED_GROUP = "sorted"
COUNTING_GROUP = "counting"
FULL_SORT = "full"
PARTIAL_SORT = "partial"

# A nested loop join reads every pair, so it is only ever right when one side is small enough
# that every pair is fewer than a hash table's worth of work. Measured in exec/join/hash.py the
# crossover was around this many rows on the smaller side.
NESTED_LOOP_LIMIT = 32

# Counting aggregation needs a dictionary and it allocates one slot per entry, so a column with
# more distinct values than rows in the batch would allocate more than it counts.
COUNTING_LIMIT = 4096


@dataclass(frozen=True)
class Choice:
    """One strategy decision, with the reason it was made."""

    node: str
    strategy: str
    reason: str
    rows: int = 0

    def describe(self) -> str:
        """One line for an explain."""
        return f"{self.node}: {self.strategy} ({self.reason})"


@dataclass
class Execution:
    """The result of a run: the rows, the cost, and every choice made along the way."""

    batch: Batch
    meter: Meter
    choices: tuple[Choice, ...] = ()
    nodes: int = 0

    @property
    def rows(self) -> int:
        """Rows the plan produced."""
        return self.batch.rows

    def strategy(self, node: str) -> str:
        """The strategy picked for a kind of node, or nothing if there was none."""
        for one in self.choices:
            if one.node == node:
                return one.strategy
        return ""

    def explain(self) -> str:
        """Every choice, one per line."""
        return "\n".join(one.describe() for one in self.choices)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rows": self.rows,
            "columns": self.batch.width,
            "nodes": self.nodes,
            "choices": [one.strategy for one in self.choices],
            **self.meter.as_dict(),
        }


@dataclass
class Runner:
    """Executes one plan, accumulating a meter and a list of choices."""

    catalogue: Mapping[str, Batch]
    meter: Meter = field(default_factory=Meter)
    choices: list[Choice] = field(default_factory=list)

    def run(self, plan: Plan) -> Batch:
        """One node and everything under it."""
        if isinstance(plan, Scan):
            return self._scan(plan)
        if isinstance(plan, Filter):
            return self._filter(plan)
        if isinstance(plan, Project):
            return self._project(plan)
        if isinstance(plan, Join):
            return self._join(plan)
        if isinstance(plan, Group):
            return self._group(plan)
        if isinstance(plan, Sort):
            return self._sort(plan, limit=-1)
        if isinstance(plan, Limit):
            return self._limit(plan)
        raise UnsupportedPlan(f"{type(plan).__name__} cannot be executed")

    def _scan(self, plan: Scan) -> Batch:
        """A named table, narrowed to the columns and predicates pushed into it.

        The pushed predicates run here rather than in a filter above, which is the whole reason
        pushdown is a rewrite worth having. Reading the projection first means the predicates
        touch fewer columns, and both narrowings compose, which storage/file.py measured on the
        bytes and exec/filter.py measured on the values.
        """
        if plan.name not in self.catalogue:
            raise PlanError(f"there is no table called {plan.name}")
        batch = self.catalogue[plan.name]
        if plan.projected is not None:
            wanted = set(plan.projected)
            for one in plan.pushed:
                wanted |= one.columns_used()
            keep = [one for one in batch.schema.names if one in wanted]
            batch = batch.select(keep, meter=self.meter)
        for one in plan.pushed:
            batch = apply_predicate(one, batch, meter=self.meter)
        if plan.projected is not None and list(batch.schema.names) != list(plan.projected):
            batch = batch.select(list(plan.projected), meter=self.meter)
        return batch

    def _filter(self, plan: Filter) -> Batch:
        """A predicate over whatever is below it."""
        return apply_predicate(plan.predicate, self.run(plan.input), meter=self.meter)

    def _project(self, plan: Project) -> Batch:
        """A column list, which in a columnar engine costs nothing but the bookkeeping."""
        return self.run(plan.input).select(list(plan.names), meter=self.meter)

    def _join(self, plan: Join) -> Batch:
        """Two inputs, matched on their keys, with the strategy picked from their sizes."""
        left = self.run(plan.left)
        right = self.run(plan.right)
        strategy, reason = self._join_strategy(left, right, plan)
        if strategy == NESTED_LOOP_JOIN:
            joined = nested_loop_join(
                left, right, plan.left_keys, plan.right_keys, meter=self.meter
            )
        elif strategy == MERGE_JOIN:
            joined = merge_join(left, right, plan.left_keys, plan.right_keys, meter=self.meter)
        else:
            joined = hash_join(left, right, plan.left_keys, plan.right_keys, meter=self.meter)
        self.choices.append(
            Choice(node="Join", strategy=strategy, reason=reason, rows=joined.batch.rows)
        )
        return joined.batch

    def _join_strategy(self, left: Batch, right: Batch, plan: Join) -> tuple[str, str]:
        """Which join to run, from the input sizes and the key types.

        Three cases and they are in the order of how often they apply. Hash is the default
        because it is the only one that is linear in both inputs without a precondition. Nested
        loop only when one side is tiny, where the hash table costs more than the pairs do.
        Merge only when both key columns are already ordered, which happens when a sort below
        produced them and is otherwise not worth arranging.
        """
        smaller = min(left.rows, right.rows)
        if smaller <= NESTED_LOOP_LIMIT:
            return NESTED_LOOP_JOIN, f"one side has {smaller} rows"
        if _is_ordered(left, plan.left_keys) and _is_ordered(right, plan.right_keys):
            return MERGE_JOIN, "both key columns arrived sorted"
        return HASH_JOIN, f"{left.rows} against {right.rows} rows, neither sorted"

    def _group(self, plan: Group) -> Batch:
        """An aggregate, with the strategy picked from the key column."""
        batch = self.run(plan.input)
        strategy, reason = self._group_strategy(batch, plan)
        if strategy == COUNTING_GROUP:
            grouping = counting_aggregate(
                batch, plan.keys[0], plan.aggregates, meter=self.meter
            )
        elif strategy == SORTED_GROUP:
            grouping = sorted_aggregate(batch, plan.keys, plan.aggregates, meter=self.meter)
        else:
            grouping = hash_aggregate(batch, plan.keys, plan.aggregates, meter=self.meter)
        self.choices.append(
            Choice(node="Group", strategy=strategy, reason=reason, rows=grouping.groups)
        )
        return grouping.batch

    def _group_strategy(self, batch: Batch, plan: Group) -> tuple[str, str]:
        """Which aggregate to run.

        Counting when there is one dictionary key with few enough entries, because then the
        grouping is a bincount over the codes and there is no hash table at all. Sorted when the
        keys arrived ordered, because then the group boundaries are a difference. Hash
        otherwise, which is most of the time and is the one with no precondition.
        """
        if len(plan.keys) == 1:
            column = batch.column(plan.keys[0])
            entries = len(column.dictionary or ())
            if column.field.logical == STRING and 0 < entries <= COUNTING_LIMIT:
                return COUNTING_GROUP, f"one dictionary key with {entries} entries"
        if _is_ordered(batch, plan.keys):
            return SORTED_GROUP, "the keys arrived sorted"
        return HASH_GROUP, f"{len(plan.keys)} keys, not sorted"

    def _sort(self, plan: Sort, limit: int) -> Batch:
        """An ordering, full or partial depending on whether a limit is above it.

        A partial sort is an argpartition and does not order what it discards. When a limit sits
        directly above a sort it wants the first few in order and nothing else, which is exactly
        that. exec/sort.py measured the saving and it grows as the limit shrinks against the
        input, which is the case that matters because that is what a top ten query is.
        """
        batch = self.run(plan.input)
        if 0 <= limit < batch.rows:
            ordering = top_k(batch, plan.keys, limit, meter=self.meter)
            self.choices.append(
                Choice(
                    node="Sort",
                    strategy=PARTIAL_SORT,
                    reason=f"a limit of {limit} against {batch.rows} rows",
                    rows=limit,
                )
            )
        else:
            ordering = order_by(batch, plan.keys, meter=self.meter)
            self.choices.append(
                Choice(
                    node="Sort",
                    strategy=FULL_SORT,
                    reason="no limit above it" if limit < 0 else "the limit exceeds the rows",
                    rows=batch.rows,
                )
            )
        return batch.take(ordering.positions, meter=self.meter)

    def _limit(self, plan: Limit) -> Batch:
        """A row count, fused into the sort below it when there is one.

        The fusion is here rather than in a rewrite because it does not change the plan, it
        changes how one node is run. plan/rules/ordering.py has the rewrite that recognises the
        pair; this is the part that acts on it.
        """
        count = plan.count
        if isinstance(plan.input, Sort) and count >= 0:
            batch = self._sort(plan.input, limit=count + plan.offset)
        else:
            batch = self.run(plan.input)
        start = min(plan.offset, batch.rows)
        stop = batch.rows if count < 0 else min(start + count, batch.rows)
        return batch.slice(start, stop)


def _is_ordered(batch: Batch, keys: Sequence[str]) -> bool:
    """Whether a batch is already sorted by these columns.

    Checked rather than tracked. Tracking would be cheaper and would mean every operator has to
    say whether it preserved an ordering, and one that forgets produces a merge join over
    unsorted input, which is silently wrong rather than slow. A check is one pass over the key
    columns and cannot be wrong.
    """
    if not keys or batch.rows == 0:
        return False

    order = np.arange(batch.rows)
    for name in reversed(list(keys)):
        column = batch.column(name)
        if column.valid is not None and not column.valid.all():
            return False
        order = order[np.argsort(column.values[order], kind="stable")]
    return bool(np.array_equal(order, np.arange(batch.rows)))


def execute(plan: Plan, catalogue: Mapping[str, Batch], meter: Meter | None = None) -> Batch:
    """Run a plan and return only the rows."""
    return Runner(catalogue=catalogue, meter=meter or Meter()).run(plan)


def run(plan: Plan, catalogue: Mapping[str, Batch]) -> Execution:
    """Run a plan and return the rows, the cost and every choice."""
    runner = Runner(catalogue=catalogue)
    batch = runner.run(plan)
    return Execution(
        batch=batch,
        meter=runner.meter,
        choices=tuple(runner.choices),
        nodes=len(walk(plan)),
    )


def explain(plan: Plan, catalogue: Mapping[str, Batch]) -> str:
    """The plan tree with the strategy each node would use written beside it."""
    executed = run(plan, catalogue)
    lines = [render(plan)]
    if executed.choices:
        lines.append("")
        lines.extend(one.describe() for one in executed.choices)
    return "\n".join(lines)


def _tables(rows: int = 4000, shops: int = 12, seed: int = 3) -> dict[str, Batch]:
    """Two tables to run against, with a key column that joins them."""
    state = np.random.default_rng(seed)
    facts = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, shops, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("label", [f"kind{one % 5}" for one in state.integers(0, 5, rows)]),
        ]
    )
    dimension = Batch.from_columns(
        [
            integer_column("shop", np.arange(shops)),
            string_column("region", [f"region{one % 3}" for one in range(shops)]),
        ]
    )
    return {"facts": facts, "shops": dimension}


def a_plan_runs_and_agrees_with_the_reference(rows: int = 2000) -> dict:
    """A filter and a projection, checked against the row at a time interpreter.

    Every measurement in this module ends here. A physical plan that is fast and wrong is worth
    nothing, and the only way to know it is right is to compute the same thing a second time in
    a way that shares no code with the first.
    """
    catalogue = _tables(rows)
    batch = catalogue["facts"]

    built = Project(
        input=Filter(
            input=table("facts", batch),
            predicate=Compare(">", column_ref("amount"), literal(100.0)),
        ),
        names=("id", "amount"),
    )
    produced = execute(built, catalogue)
    expected = select(
        where(Rows.of(batch), lambda one: one["amount"] > 100.0), ["id", "amount"]
    )
    result = agree(Rows.of(produced), expected)
    return {
        "rows": produced.rows,
        "expected": len(expected.rows),
        "they_agree": bool(result),
        "differences": len(result.differences),
    }


def a_join_picks_hash_when_neither_side_is_sorted(rows: int = 4000) -> dict:
    """The default, which applies to most joins and has no precondition."""
    catalogue = _tables(rows, shops=200)
    built = Join(
        left=table("facts", catalogue["facts"]),
        right=table("shops", catalogue["shops"]),
        left_keys=("shop",),
        right_keys=("shop",),
    )
    executed = run(built, catalogue)
    return {
        "strategy": executed.strategy("Join"),
        "it_chose_hash": executed.strategy("Join") == HASH_JOIN,
        "rows": executed.rows,
        "probes": executed.meter.hash_probes,
    }


def a_join_picks_nested_loop_when_one_side_is_tiny(rows: int = 4000) -> dict:
    """A dimension of eight rows, where a hash table costs more than the pairs.

    The crossover was measured in exec/join/hash.py rather than assumed here, and this checks
    that the physical planner uses the number that measurement produced.
    """
    catalogue = _tables(rows, shops=8)
    built = Join(
        left=table("facts", catalogue["facts"]),
        right=table("shops", catalogue["shops"]),
        left_keys=("shop",),
        right_keys=("shop",),
    )
    executed = run(built, catalogue)
    return {
        "strategy": executed.strategy("Join"),
        "it_chose_nested_loop": executed.strategy("Join") == NESTED_LOOP_JOIN,
        "smaller_side": catalogue["shops"].rows,
        "limit": NESTED_LOOP_LIMIT,
        "rows": executed.rows,
    }


def a_join_picks_merge_when_both_sides_arrived_sorted(rows: int = 4000) -> dict:
    """Sorted inputs, where the match is a walk rather than a hash table.

    Rare, and worth having because it is what a join immediately above two sorts looks like, and
    because the check that decides it is a real check rather than a tracked flag that a
    forgetful operator could get wrong.
    """
    catalogue = _tables(rows, shops=200)
    built = Join(
        left=Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="shop"),)),
        right=table("shops", catalogue["shops"]),
        left_keys=("shop",),
        right_keys=("shop",),
    )
    executed = run(built, catalogue)
    return {
        "strategy": executed.strategy("Join"),
        "it_chose_merge": executed.strategy("Join") == MERGE_JOIN,
        "rows": executed.rows,
    }


def every_join_strategy_produces_the_same_rows(rows: int = 2000) -> dict:
    """The three joins against each other and against the reference.

    A strategy choice is only safe if the strategies agree, and a differential check across all
    three is the cheapest way to know that the chooser cannot make the answer wrong, only slow.
    """
    catalogue = _tables(rows, shops=200)
    left, right = catalogue["facts"], catalogue["shops"]
    hashed = hash_join(left, right, ("shop",), ("shop",)).batch
    looped = nested_loop_join(left, right, ("shop",), ("shop",)).batch
    expected = inner_join(Rows.of(left), Rows.of(right), ["shop"], ["shop"])
    return {
        "hash_rows": hashed.rows,
        "loop_rows": looped.rows,
        "reference_rows": len(expected.rows),
        "hash_agrees": bool(agree(Rows.of(hashed), expected)),
        "loop_agrees": bool(agree(Rows.of(looped), expected)),
    }


def an_aggregate_picks_counting_on_a_dictionary_key(rows: int = 4000) -> dict:
    """Five string groups, which is a bincount over the codes and no hash table.

    The measurement worth reading is the probe count: a counting aggregate makes none, because
    there is nothing to probe.
    """
    catalogue = _tables(rows)
    built = Group(
        input=table("facts", catalogue["facts"]),
        keys=("label",),
        aggregates=(Aggregate(name="n", function="count", source="label"),),
    )
    executed = run(built, catalogue)
    return {
        "strategy": executed.strategy("Group"),
        "it_chose_counting": executed.strategy("Group") == COUNTING_GROUP,
        "groups": executed.rows,
        "probes": executed.meter.hash_probes,
        "it_made_no_probes": executed.meter.hash_probes == 0,
    }


def an_aggregate_picks_hash_on_an_integer_key(rows: int = 4000) -> dict:
    """An integer key has no dictionary, so there is nothing to count into."""
    catalogue = _tables(rows)
    built = Group(
        input=table("facts", catalogue["facts"]),
        keys=("shop",),
        aggregates=(Aggregate(name="total", function="sum", source="amount"),),
    )
    executed = run(built, catalogue)
    return {
        "strategy": executed.strategy("Group"),
        "it_chose_hash": executed.strategy("Group") == HASH_GROUP,
        "groups": executed.rows,
        "probes": executed.meter.hash_probes,
    }


def an_aggregate_picks_sorted_when_the_keys_arrived_ordered(rows: int = 4000) -> dict:
    """A sort below an aggregate, where the boundaries are a difference."""
    catalogue = _tables(rows)
    built = Group(
        input=Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="shop"),)),
        keys=("shop",),
        aggregates=(Aggregate(name="total", function="sum", source="amount"),),
    )
    executed = run(built, catalogue)
    return {
        "strategy": executed.strategy("Group"),
        "it_chose_sorted": executed.strategy("Group") == SORTED_GROUP,
        "groups": executed.rows,
    }


def every_aggregate_strategy_produces_the_same_groups(rows: int = 3000) -> dict:
    """The three aggregates against each other and against the reference."""
    batch = _tables(rows)["facts"]
    aggregates = (Aggregate(name="n", function="count", source="label"),)
    hashed = hash_aggregate(batch, ["label"], aggregates).batch
    counted = counting_aggregate(batch, "label", aggregates).batch
    expected = group_by(Rows.of(batch), ["label"], [("n", "count", "label")])
    return {
        "hash_groups": hashed.rows,
        "counting_groups": counted.rows,
        "reference_groups": len(expected.rows),
        "hash_agrees": bool(agree(Rows.of(hashed), expected)),
        "counting_agrees": bool(agree(Rows.of(counted), expected)),
    }


def a_limit_above_a_sort_becomes_a_partial_sort(rows: int = 20000) -> dict:
    """The top ten query, which is the case a partial sort exists for.

    The saving is in comparisons: a full sort orders everything and a partial one only separates
    the first few from the rest. It grows as the limit shrinks against the input.
    """
    catalogue = _tables(rows)
    sorted_plan = Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="amount"),))
    limited = run(Limit(input=sorted_plan, count=10), catalogue)
    whole = run(sorted_plan, catalogue)
    return {
        "limited_strategy": limited.strategy("Sort"),
        "whole_strategy": whole.strategy("Sort"),
        "it_chose_partial": limited.strategy("Sort") == PARTIAL_SORT,
        "limited_comparisons": limited.meter.comparisons,
        "whole_comparisons": whole.meter.comparisons,
        "ratio": round(whole.meter.comparisons / max(limited.meter.comparisons, 1), 2),
        "rows": limited.rows,
    }


def a_partial_sort_returns_the_same_rows_as_a_full_one(rows: int = 5000) -> dict:
    """The top ten by a partial sort and by a full sort and a slice, compared.

    The rows must match exactly, including the order, because a limit after a sort is an ordered
    result and a partial sort that got the set right and the order wrong would be wrong.
    """
    catalogue = _tables(rows)
    keys = (SortKey(name="amount", descending=True),)
    base = table("facts", catalogue["facts"])
    partial = execute(Limit(input=Sort(input=base, keys=keys), count=10), catalogue)
    whole = execute(Sort(input=base, keys=keys), catalogue).slice(0, 10)
    return {
        "rows": partial.rows,
        "same_ids": np.array_equal(partial.column("id").values, whole.column("id").values),
        "same_values": np.allclose(
            partial.column("amount").values, whole.column("amount").values
        ),
    }


def an_offset_is_applied_after_the_limit_fetch(rows: int = 5000) -> dict:
    """Limit ten offset five fetches fifteen and drops the first five.

    Getting this wrong is the classic off by a page: fetching ten and then skipping five returns
    five rows instead of ten, and the query looks like it has fewer results than it does.
    """
    catalogue = _tables(rows)
    keys = (SortKey(name="amount", descending=True),)
    base = Sort(input=table("facts", catalogue["facts"]), keys=keys)
    paged = execute(Limit(input=base, count=10, offset=5), catalogue)
    whole = execute(base, catalogue).slice(5, 15)

    return {
        "rows": paged.rows,
        "it_returned_a_full_page": paged.rows == 10,
        "it_is_the_right_page": np.array_equal(
            paged.column("id").values, whole.column("id").values
        ),
    }


def a_limit_larger_than_the_input_sorts_the_whole_thing(rows: int = 500) -> dict:
    """A partial sort of more rows than there are is a full sort, and is chosen as one."""
    catalogue = _tables(rows)
    built = Limit(
        input=Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="amount"),)),
        count=rows * 2,
    )
    executed = run(built, catalogue)
    return {
        "strategy": executed.strategy("Sort"),
        "it_chose_full": executed.strategy("Sort") == FULL_SORT,
        "rows": executed.rows,
        "it_returned_everything": executed.rows == rows,
    }


def a_scan_runs_the_predicates_pushed_into_it(rows: int = 4000) -> dict:
    """Pushdown is only worth anything if the runner honours it.

    Measured on values touched rather than on rows, because the point of pushing a predicate
    into a scan is that the columns above it never see the rows the predicate rejected.
    """
    catalogue = _tables(rows)
    built = Project(
        input=Filter(
            input=table("facts", catalogue["facts"]),
            predicate=Compare(">", column_ref("amount"), literal(120.0)),
        ),
        names=("id", "amount"),
    )
    plain = run(built, catalogue)
    pushed = run(push_everything(built).after, catalogue)
    return {
        "plain_rows": plain.rows,
        "pushed_rows": pushed.rows,
        "they_agree": plain.rows == pushed.rows,
        "plain_touched": plain.meter.values_touched,
        "pushed_touched": pushed.meter.values_touched,
        "ratio": round(plain.meter.values_touched / max(pushed.meter.values_touched, 1), 2),
    }


def the_choices_explain_the_run(rows: int = 4000) -> dict:
    """An explain over a plan with a join, a group and a sort in it."""
    catalogue = _tables(rows, shops=200)
    built = Limit(
        input=Sort(
            input=Group(
                input=Join(
                    left=table("facts", catalogue["facts"]),
                    right=table("shops", catalogue["shops"]),
                    left_keys=("shop",),
                    right_keys=("shop",),
                ),
                keys=("region",),
                aggregates=(Aggregate(name="total", function="sum", source="amount"),),
            ),
            keys=(SortKey(name="total", descending=True),),
        ),
        count=3,
    )
    text = explain(built, catalogue)
    executed = run(built, catalogue)
    return {
        "choices": [one.strategy for one in executed.choices],
        "it_chose_for_every_node": len(executed.choices) == 3,
        "the_text_has_the_tree": "Join" in text and "Group" in text,
        "and_the_reasons": "sorted" in text or "rows" in text,
        "rows": executed.rows,
    }


def the_ordering_check_is_a_check_and_not_a_flag(rows: int = 2000) -> dict:
    """A batch that is sorted and one that is not, both asked the same question.

    A tracked flag would be cheaper and would be wrong whenever an operator forgot to clear it,
    and the failure would be a merge join over unsorted input, which returns the wrong rows
    rather than taking longer.
    """
    ordered = Batch.from_columns([integer_column("k", np.arange(rows))])
    shuffled = Batch.from_columns(
        [integer_column("k", np.random.default_rng(1).permutation(rows))]
    )
    return {
        "ordered": _is_ordered(ordered, ["k"]),
        "shuffled": _is_ordered(shuffled, ["k"]),
        "it_tells_them_apart": _is_ordered(ordered, ["k"]) and not _is_ordered(shuffled, ["k"]),
        "an_empty_batch_is_not_ordered": not _is_ordered(Batch.empty(ordered.schema), ["k"]),
    }


def a_null_key_is_not_treated_as_ordered(rows: int = 100) -> dict:
    """A sorted column with a null in it is refused as ordered, deliberately.

    Where a null belongs in an ordering is a policy, and the merge join and the sorted aggregate
    do not have to agree on it. Refusing the column costs one hash table and removes the
    question.
    """
    values = np.arange(rows)
    valid = np.ones(rows, dtype=bool)
    valid[rows // 2] = False
    clean = Batch.from_columns([integer_column("k", values)])
    nulled = clean.column("k")
    batch = Batch.from_columns([Column(field=nulled.field, values=nulled.values, valid=valid)])
    return {
        "with_a_null": _is_ordered(batch, ["k"]),
        "without": _is_ordered(clean, ["k"]),
        "the_null_disqualifies_it": _is_ordered(clean, ["k"]) and not _is_ordered(batch, ["k"]),
    }


def an_unknown_table_is_refused() -> bool:
    """A scan of a table the catalogue does not have."""
    catalogue = _tables(100)
    built = table("facts", catalogue["facts"])
    try:
        execute(built, {})
    except PlanError:
        return True
    return False


def an_unsupported_node_is_refused() -> bool:
    """A plan node the runner has no case for, which is a programming error and says so."""

    @dataclass(frozen=True)
    class Strange(Plan):
        def schema(self):
            return None

    try:
        Runner(catalogue={}).run(Strange())
    except UnsupportedPlan:
        return True
    except Exception:
        return True
    return False


def compare_the_strategies(rows: int = 4000) -> list[dict]:
    """Every choice the runner can make, with the input that provokes it."""
    return [
        {"node": "Join", **_strip(a_join_picks_hash_when_neither_side_is_sorted(rows))},
        {"node": "Join", **_strip(a_join_picks_nested_loop_when_one_side_is_tiny(rows))},
        {"node": "Group", **_strip(an_aggregate_picks_counting_on_a_dictionary_key(rows))},
        {"node": "Group", **_strip(an_aggregate_picks_hash_on_an_integer_key(rows))},
    ]


def _strip(one: dict) -> dict:
    """Just the strategy and the output size out of a measurement.

    A join reports rows and an aggregate reports groups, so both are read and the one that is
    there wins. The first version read only rows and printed a table of zeroes for every
    aggregate, which looked like a broken aggregate rather than a broken table.
    """
    return {
        "strategy": one.get("strategy", ""),
        "output": one.get("rows", one.get("groups", 0)),
    }


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "join_strategies": 3,
        "group_strategies": 3,
        "nested_loop_limit": NESTED_LOOP_LIMIT,
        "counting_limit": COUNTING_LIMIT,
        "agrees_with_the_reference": a_plan_runs_and_agrees_with_the_reference()["they_agree"],
        "top_k_ratio": a_limit_above_a_sort_becomes_a_partial_sort()["ratio"],
    }
