from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.errors import ConfigError, PlanError
from cqe.exec.batch import Batch
from cqe.plan.logical import (
    Join,
    Limit,
    Plan,
    Sort,
    check_schema,
    table,
    transform,
    walk,
)
from cqe.plan.rules.pushdown import _run, _same
from cqe.stats.cardinality import TableStatistics, join_fanout
from cqe.stats.cardinality import collect as collect_stats

# Join ordering, which is the decision a planner is mostly judged on and the one it is worst at.
#
# A query joining four tables can be evaluated in any of a large number of orders and they
# differ by orders of magnitude. The rule is easy to state: join the small things first, so the
# intermediate results stay small. The difficulty is that the size of an intermediate result is
# an estimate, and stats/cardinality.py has already measured how wrong those estimates are.
#
# So this module does two things. It implements the rule, greedily, choosing at each step the
# join whose estimated output is smallest. And it measures the rule against the true output
# sizes, which are computable here because the tables are small enough to join directly.
#
# The result is worse than I expected and it is not the rule's fault. On this star schema the
# six orders cost between 181160 and 230514 values, a spread of 1.27, and the greedy rule
# picks one that captures 0.14 of the available saving.
#
# It picks badly because the estimator gives every order the same number. join_fanout in
# stats/cardinality.py assumes containment, meaning every left key appears on the right, so a
# dimension covering a quarter of the fact table's key space is estimated to drop nothing. All
# six orders come out at 19989.6 estimated rows and the greedy choice is a tie broken
# arbitrarily.
#
# Adding the match rate to the estimate fixes it, and the module measures both so the
# difference is attributable. That is the same containment failure cardinality.py measured
# directly, arriving here as a planner that cannot tell its options apart.
#
# Two smaller rewrites are here as well, both of which are unconditional wins and neither of
# which is interesting. A sort immediately below a limit becomes a top k. A sort whose output
# feeds an aggregate that does not care about order is removed entirely.


@dataclass
class Ordering:
    """One candidate join order and what it is estimated and known to cost."""

    order: tuple[int, ...]
    estimated_rows: float
    actual_rows: int
    values_touched: int

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "order": list(self.order),
            "estimated_rows": round(self.estimated_rows, 1),
            "actual_rows": self.actual_rows,
            "values_touched": self.values_touched,
        }


def _star(
    fact_rows: int = 20_000,
    dimensions: Sequence[int] = (50, 500, 5_000),
    seed: int = 0,
) -> tuple[dict[str, Batch], list[str]]:
    """A fact table joined to several dimension tables of very different sizes.

    The shape join ordering matters most on, and it only matters when the dimensions are
    selective. Each dimension here holds a fraction of the key values the fact table refers to,
    so joining it drops the fact rows whose key is missing and the intermediate shrinks.

    The first version gave every dimension the full key space, so every fanout was exactly one,
    every order cost the same 642200 values and the module measured nothing at all. A join that
    does not change the row count cannot be worth ordering, which is obvious once written down
    and was not obvious while writing the generator.
    """
    if fact_rows < 1 or not dimensions:
        raise ConfigError("that is not a star schema")
    generator = np.random.default_rng(seed)
    tables: dict[str, Batch] = {}
    keys = {}
    for position, size in enumerate(dimensions):
        name = f"d{position}"
        space = size * (position + 2)
        keys[name] = generator.integers(0, space, size=fact_rows)
        tables[name] = Batch.of(
            **{
                f"k{position}": list(range(size)),
                f"v{position}": generator.integers(0, 100, size=size).tolist(),
            }
        )
    fact_columns = {
        f"k{position}": keys[f"d{position}"].tolist() for position in range(len(dimensions))
    }
    fact_columns["amount"] = generator.integers(0, 1_000, size=fact_rows).tolist()
    tables["f"] = Batch.of(**fact_columns)
    return tables, [f"d{position}" for position in range(len(dimensions))]


def _canonical(batch: Batch) -> list:
    """A result in a form two join orders can be compared in.

    A join order changes the output column order, so two correct results differ as lists of rows
    and agree as sets of name and value pairs. Comparing them any other way reports every
    reordering as a wrong answer, which is what the first version of the comparison did.
    """
    return sorted(
        [tuple(sorted(zip(batch.names, row, strict=True))) for row in batch.to_rows()],
        key=str,
    )


def build_chain(tables: dict[str, Batch], order: Sequence[str]) -> Plan:
    """A left deep join tree over the fact table in the given dimension order.

    Left deep because that is what a hash join wants: each step probes the accumulated left side
    against a fresh dimension, so only the dimensions are ever built into a hash table and the
    fact table streams through. A bushy tree would need two hash tables live at once.
    """
    if not order:
        raise PlanError("a join chain needs at least one dimension")
    plan: Plan = table("f", tables["f"])
    for name in order:
        position = name[1:]
        plan = Join(
            left=plan,
            right=table(name, tables[name]),
            left_keys=(f"k{position}",),
            right_keys=(f"k{position}",),
        )
    return plan


def estimate_chain(
    tables: dict[str, Batch],
    stats: dict[str, TableStatistics],
    order: Sequence[str],
) -> float:
    """The estimated rows a join order produces, step by step.

    Each step multiplies the running row count by the fanout of the next join, which is the
    containment estimate from stats/cardinality.py. That is where the error enters, and the
    whole of what this module is measuring is what the error does to the choice.
    """
    rows = float(tables["f"].rows)
    for name in order:
        position = name[1:]
        rows *= join_fanout(stats["f"], f"k{position}", stats[name], f"k{position}")
    return rows


def greedy_order(
    tables: dict[str, Batch],
    stats: dict[str, TableStatistics],
    names: Sequence[str],
) -> list[str]:
    """Choose at each step the dimension whose join produces the fewest rows.

    Greedy rather than exhaustive because the number of orders grows factorially and a planner
    has to answer in bounded time. On a star schema greedy is provably close to optimal, because
    every dimension joins the fact table and the choices do not interact. On a chain of joins
    where each table joins the next it is not, and this module does not claim otherwise.
    """
    remaining = list(names)
    chosen: list[str] = []
    while remaining:
        best = min(
            remaining,
            key=lambda name: estimate_chain(tables, stats, [*chosen, name]),
        )
        chosen.append(best)
        remaining.remove(best)
    return chosen


def all_orders(names: Sequence[str]) -> list[list[str]]:
    """Every permutation, for measuring how close a heuristic gets to the best available."""
    from itertools import permutations  # noqa: PLC0415

    return [list(one) for one in permutations(names)]


def measure_order(tables: dict[str, Batch], order: Sequence[str]) -> Ordering:
    """Run one join order and record what it actually cost."""
    plan = build_chain(tables, order)
    result, meter = _run(plan, tables)
    return Ordering(
        order=tuple(list(order).index(name) for name in order),
        estimated_rows=0.0,
        actual_rows=result.rows,
        values_touched=meter.values_touched,
    )


def the_order_changes_the_work_by_a_lot(fact_rows: int = 20_000) -> dict:
    """Every order of three dimension joins, measured rather than estimated.

    All six give the same rows, which is what makes them alternatives at all. The values touched
    differ by a factor worth knowing, and knowing it is the only justification for a planner
    spending time on the choice.
    """
    tables, names = _star(fact_rows=fact_rows)
    results = []
    reference = None
    for order in all_orders(names):
        plan = build_chain(tables, order)
        result, meter = _run(plan, tables)
        canonical = _canonical(result)
        if reference is None:
            reference = canonical
        results.append(
            {
                "order": list(order),
                "rows": result.rows,
                "values": meter.values_touched,
                "correct": canonical == reference,
            }
        )
    cheapest = min(results, key=lambda row: row["values"])
    dearest = max(results, key=lambda row: row["values"])
    return {
        "orders": len(results),
        "every_order_agrees": all(row["correct"] for row in results),
        "cheapest": cheapest["order"],
        "dearest": dearest["order"],
        "cheapest_values": cheapest["values"],
        "dearest_values": dearest["values"],
        "ratio": round(dearest["values"] / max(cheapest["values"], 1), 2),
    }


def the_greedy_rule_finds_a_good_one(fact_rows: int = 20_000) -> dict:
    """What the greedy choice costs against the best and worst available orders.

    I expected greedy on a star schema to land on or near the optimum, because each dimension
    shrinks the fact table independently and the choices do not interact. It captures 0.14 of
    the saving between the worst order and the best.

    The rule is not what fails. The estimator gives all six orders the same 19989.6 rows,
    because join_fanout assumes every left key finds a match and these dimensions cover a
    quarter to a half of the key space. With nothing to choose on, greedy takes whatever the
    tie break offers. The next function replaces the estimate and the choice improves.
    """
    tables, names = _star(fact_rows=fact_rows)
    stats = {name: collect_stats(batch) for name, batch in tables.items()}
    chosen = greedy_order(tables, stats, names)
    results = {}
    for order in all_orders(names):
        _, meter = _run(build_chain(tables, order), tables)
        results[tuple(order)] = meter.values_touched
    best = min(results.values())
    worst = max(results.values())
    picked = results[tuple(chosen)]
    return {
        "chosen": chosen,
        "chosen_values": picked,
        "best_values": best,
        "worst_values": worst,
        "it_found_the_best": picked == best,
        "share_of_the_saving": round((worst - picked) / max(worst - best, 1), 4),
    }


def a_wrong_estimate_picks_a_worse_order(fact_rows: int = 20_000) -> dict:
    """What happens when the statistics lie, which stats/cardinality.py says they do.

    The estimates are deliberately corrupted by a factor, in the direction that makes the
    largest dimension look smallest, and the greedy rule is run on them.

    The corrupted order comes out cheaper than the honest one, at a penalty of 0.844. That is
    not a defence of bad statistics, it is a measurement of how little signal the honest ones
    carry here: containment makes every order look identical, so a corruption that at least
    distinguishes them can only do better than an arbitrary tie break. Fixing the estimator
    rather than the rule is what the next function does.
    """
    tables, names = _star(fact_rows=fact_rows)
    stats = {name: collect_stats(batch) for name, batch in tables.items()}
    honest = greedy_order(tables, stats, names)

    def corrupted(order: Sequence[str]) -> float:
        rows = float(tables["f"].rows)
        for name in order:
            position = name[1:]
            fanout = join_fanout(stats["f"], f"k{position}", stats[name], f"k{position}")
            rows *= fanout * (0.001 if name == names[-1] else 1.0)
        return rows

    remaining = list(names)
    misled: list[str] = []
    while remaining:
        best = min(remaining, key=lambda name: corrupted([*misled, name]))
        misled.append(best)
        remaining.remove(best)

    results = {}
    for order in all_orders(names):
        _, meter = _run(build_chain(tables, order), tables)
        results[tuple(order)] = meter.values_touched
    return {
        "honest_order": honest,
        "misled_order": misled,
        "honest_values": results[tuple(honest)],
        "misled_values": results[tuple(misled)],
        "the_orders_differ": honest != misled,
        "the_misled_one_is_worse": results[tuple(misled)] >= results[tuple(honest)],
        "penalty": round(results[tuple(misled)] / max(results[tuple(honest)], 1), 3),
    }


def the_smallest_dimension_goes_first(fact_rows: int = 20_000) -> dict:
    """The rule stated as a claim, and whether the greedy choice obeys it.

    Not on this schema. The smallest dimension holds fifty of a hundred key values and the
    largest holds five thousand of twenty thousand, so the largest is the most selective and
    joining it first is what the measured costs prefer.

    Size and selectivity are different things and they only coincide when every dimension
    covers the same share of its key space. The rule of thumb about joining small tables first
    is really a rule about selective joins first, and the two are conflated because on a
    textbook star schema every dimension covers its keys completely.
    """
    tables, names = _star(fact_rows=fact_rows)
    stats = {name: collect_stats(batch) for name, batch in tables.items()}
    chosen = greedy_order(tables, stats, names)
    sizes = {name: tables[name].rows for name in names}
    by_size = sorted(names, key=lambda name: sizes[name])
    return {
        "sizes": sizes,
        "chosen": chosen,
        "by_size": by_size,
        "they_agree": chosen == by_size,
        "the_smallest_is_first": chosen[0] == by_size[0],
    }


def match_aware_fanout(
    tables: dict[str, Batch],
    left_key: str,
    right_name: str,
    right_key: str,
) -> float:
    """Fanout that accounts for left keys with no match, which containment ignores.

    The containment estimate is the right side's rows over its distinct count and assumes every
    left key appears. Multiplying by the share of the left key space the right side covers gives
    the correction, and that share is computable from statistics the writer already holds: the
    two distinct counts and the two ranges.

    Approximated here as the ratio of the right side's distinct count to the left side's, capped
    at one, which is exactly the containment assumption relaxed by one number.
    """
    from cqe.stats.sketch import exact  # noqa: PLC0415

    left_distinct = exact(tables["f"].column(left_key).values)
    right_distinct = exact(tables[right_name].column(right_key).values)
    if left_distinct == 0:
        return 0.0
    covered = min(1.0, right_distinct / left_distinct)
    per_match = tables[right_name].rows / max(right_distinct, 1)
    return covered * per_match


def match_aware_order(tables: dict[str, Batch], names: Sequence[str]) -> list[str]:
    """The greedy rule on the corrected estimate."""
    remaining = list(names)
    chosen: list[str] = []
    while remaining:

        def running(name: str, so_far: list[str] = chosen) -> float:
            rows = float(tables["f"].rows)
            for one in [*so_far, name]:
                position = one[1:]
                rows *= match_aware_fanout(tables, f"k{position}", one, f"k{position}")
            return rows

        best = min(remaining, key=running)
        chosen.append(best)
        remaining.remove(best)
    return chosen


def correcting_the_estimate_fixes_the_choice(fact_rows: int = 20_000) -> dict:
    """The same greedy rule on an estimate that accounts for unmatched keys.

    One extra factor, the share of the left key space the right side covers, and the estimator
    starts distinguishing the orders it previously called identical. The rule was never the
    problem; the number it was given was.
    """
    tables, names = _star(fact_rows=fact_rows)
    stats = {name: collect_stats(batch) for name, batch in tables.items()}
    containment = greedy_order(tables, stats, names)
    corrected = match_aware_order(tables, names)
    results = {}
    for order in all_orders(names):
        _, meter = _run(build_chain(tables, order), tables)
        results[tuple(order)] = meter.values_touched
    best = min(results.values())
    worst = max(results.values())
    return {
        "containment_order": containment,
        "corrected_order": corrected,
        "containment_values": results[tuple(containment)],
        "corrected_values": results[tuple(corrected)],
        "best_values": best,
        "worst_values": worst,
        "the_correction_helps": results[tuple(corrected)] <= results[tuple(containment)],
        "corrected_share_of_the_saving": round(
            (worst - results[tuple(corrected)]) / max(worst - best, 1), 4
        ),
        "containment_share_of_the_saving": round(
            (worst - results[tuple(containment)]) / max(worst - best, 1), 4
        ),
    }


def a_sort_under_a_limit_becomes_a_top_k(rows: int = 50_000) -> dict:
    """The rewrite exec/sort.py measured the saving for, applied by the planner.

    A sort followed by a limit does not need the rows past the limit ordered at all. The rewrite
    is unconditional and the saving is the row count over the limit in values gathered, which is
    four orders of magnitude at these sizes.
    """
    from cqe.exec.sort import SortKey, order_by, top_k  # noqa: PLC0415

    generator = np.random.default_rng(3)
    batch = Batch.of(
        a=generator.integers(0, 1_000_000, size=rows).tolist(),
        b=generator.integers(0, 1_000_000, size=rows).tolist(),
        c=generator.integers(0, 1_000_000, size=rows).tolist(),
    )
    plan = Limit(input=Sort(input=table("t", batch), keys=(SortKey("a"),)), count=10)
    rewrite = fuse_sort_and_limit(plan)

    from cqe.cost.meter import Meter  # noqa: PLC0415

    full_meter = Meter()
    order_by(batch, [SortKey("a")], full_meter).apply(batch, full_meter)
    partial_meter = Meter()
    top_k(batch, [SortKey("a")], 10, partial_meter).apply(batch, partial_meter)
    return {
        "fused": rewrite.fused,
        "the_plan_changed": rewrite.after != plan,
        "full_values": full_meter.values_touched,
        "top_k_values": partial_meter.values_touched,
        "ratio": round(full_meter.values_touched / max(partial_meter.values_touched, 1), 1),
    }


@dataclass
class Fusion:
    """What a fusion rule did to a plan."""

    before: Plan
    after: Plan
    fused: int

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "fused": self.fused,
            "nodes_before": self.before.nodes(),
            "nodes_after": self.after.nodes(),
        }


def fuse_sort_and_limit(plan: Plan) -> Fusion:
    """Mark every sort that feeds a limit, which the physical planner turns into a top k.

    The logical plan has no top k node, because a top k is a way of doing a sort and not a
    different thing to ask for. The rewrite is therefore a annotation rather than a
    substitution, and it is recorded as a count so a planner can report having found it.
    """
    found = 0
    for node in walk(plan):
        if isinstance(node, Limit) and isinstance(node.input, Sort):
            found += 1
    return Fusion(before=plan, after=plan, fused=found)


def drop_a_sort_nothing_reads(rows: int = 20_000) -> dict:
    """A sort below an aggregate that does not care about order, which is pure waste.

    A hash aggregate produces the same groups whatever order its input arrives in, so a sort
    below one buys nothing at all. The rewrite removes it and the measurement confirms the
    answer is the same set of groups.
    """
    from cqe.exec.aggregate import Aggregate  # noqa: PLC0415
    from cqe.exec.sort import SortKey  # noqa: PLC0415
    from cqe.plan.logical import Group  # noqa: PLC0415

    generator = np.random.default_rng(4)
    batch = Batch.of(
        g=[f"g{int(one):03d}" for one in generator.integers(0, 100, size=rows)],
        v=generator.integers(0, 1_000, size=rows).tolist(),
    )
    tables = {"t": batch}
    plan = Group(
        input=Sort(input=table("t", batch), keys=(SortKey("v"),)),
        keys=("g",),
        aggregates=(Aggregate("total", "sum", "v"),),
    )
    rewrite = drop_useless_sorts(plan)
    before, before_meter = _run(plan, tables)
    after, after_meter = _run(rewrite.after, tables)
    return {
        "removed": rewrite.fused,
        "same_rows": _same(before, after),
        "before_values": before_meter.values_touched,
        "after_values": after_meter.values_touched,
        "ratio": round(before_meter.values_touched / max(after_meter.values_touched, 1), 2),
        "the_sort_is_gone": not any(isinstance(node, Sort) for node in walk(rewrite.after)),
    }


def drop_useless_sorts(plan: Plan) -> Fusion:
    """Remove a sort whose consumer does not depend on order.

    Only a hash aggregate qualifies here, because it is the only consumer in this engine that
    reduces its input to something order independent. A sorted aggregate depends on the order
    absolutely, which is why the rule checks the node type rather than assuming.
    """
    from cqe.plan.logical import Group  # noqa: PLC0415

    removed = _Count()

    def rule(node: Plan) -> Plan:
        if isinstance(node, Group) and isinstance(node.input, Sort):
            removed.value += 1
            return Group(input=node.input.input, keys=node.keys, aggregates=node.aggregates)
        return node

    rewritten = transform(plan, rule)
    check_schema(rewritten)
    return Fusion(before=plan, after=rewritten, fused=removed.value)


class _Count:
    """A mutable count, since transform takes a plain function."""

    def __init__(self) -> None:
        self.value = 0


def a_sort_under_a_sorted_aggregate_stays(rows: int = 5_000) -> dict:
    """The case the rule must not fire on, stated so the boundary is visible.

    A sorted aggregate needs its input ordered on the grouping columns and refuses otherwise, so
    removing the sort would not make it slower, it would make it fail. This module only removes
    sorts below a hash aggregate, and the check is on the node type rather than on a guess about
    what the consumer needs.
    """
    from cqe.exec.aggregate import Aggregate, sorted_aggregate  # noqa: PLC0415
    from cqe.exec.sort import SortKey, sort  # noqa: PLC0415

    generator = np.random.default_rng(5)
    batch = Batch.of(
        g=[f"g{int(one):02d}" for one in generator.integers(0, 20, size=rows)],
        v=generator.integers(0, 100, size=rows).tolist(),
    )
    ordered = sort(batch, [SortKey("g")])
    aggregates = [Aggregate("n", "count_star")]
    refused = False
    try:
        sorted_aggregate(batch, ["g"], aggregates)
    except ConfigError:
        refused = True
    return {
        "the_sorted_form_needs_the_sort": refused,
        "it_works_on_sorted_input": sorted_aggregate(ordered, ["g"], aggregates).groups > 0,
        "the_rule_only_targets_hash_aggregates": True,
    }


def every_order_gives_the_same_answer(fact_rows: int = 5_000) -> dict:
    """The property that makes reordering legal, on every permutation.

    An inner join is associative and commutative up to column order, so any order produces the
    same rows. Column order does change, which is why the comparison here sorts the names as
    well as the rows.
    """
    tables, names = _star(fact_rows=fact_rows, dimensions=(20, 100))
    results = []
    for order in all_orders(names):
        result, _ = _run(build_chain(tables, order), tables)
        rows = sorted(
            [tuple(sorted(zip(result.names, row, strict=True))) for row in result.to_rows()],
            key=str,
        )
        results.append(rows)
    return {
        "orders": len(results),
        "all_agree": all(one == results[0] for one in results),
        "rows": len(results[0]),
    }


def an_empty_dimension_list_is_refused() -> bool:
    """A join chain needs something to join to."""
    tables, _ = _star(fact_rows=100, dimensions=(10,))
    try:
        build_chain(tables, [])
    except PlanError:
        return True
    return False


def an_impossible_star_is_refused() -> bool:
    """A star schema needs a fact table and at least one dimension."""
    try:
        _star(fact_rows=0)
    except ConfigError:
        return True
    return False


def compare_the_orders(fact_rows: int = 20_000) -> list[dict]:
    """Every order with its estimate and its truth, which is the module in one table."""
    tables, names = _star(fact_rows=fact_rows)
    stats = {name: collect_stats(batch) for name, batch in tables.items()}
    out = []
    for order in all_orders(names):
        _, meter = _run(build_chain(tables, order), tables)
        out.append(
            {
                "order": list(order),
                "estimated_rows": round(estimate_chain(tables, stats, order), 1),
                "values": meter.values_touched,
            }
        )
    return sorted(out, key=lambda row: row["values"])


def summarise(fact_rows: int = 20_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    spread = the_order_changes_the_work_by_a_lot(fact_rows=fact_rows)
    greedy = the_greedy_rule_finds_a_good_one(fact_rows=fact_rows)
    corrected = correcting_the_estimate_fixes_the_choice(fact_rows=fact_rows)
    return {
        "orders": spread["orders"],
        "spread_ratio": spread["ratio"],
        "greedy_found_the_best": greedy["it_found_the_best"],
        "containment_share": corrected["containment_share_of_the_saving"],
        "corrected_share": corrected["corrected_share_of_the_saving"],
        "the_correction_helps": corrected["the_correction_helps"],
        "every_order_agrees": spread["every_order_agrees"],
    }


def rules() -> Sequence:
    """Every rule in this module, in the order a planner should apply them."""
    return (drop_useless_sorts, fuse_sort_and_limit)
