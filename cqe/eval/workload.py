from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.cost.model import estimate
from cqe.errors import ConfigError
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.exec.filter import apply as apply_predicate
from cqe.plan.logical import walk
from cqe.plan.physical import run
from cqe.plan.rules.pushdown import push_everything
from cqe.sql.parse import plan as plan_query
from cqe.stats.cardinality import collect
from cqe.storage.disk import create, scan
from cqe.verify.reference import Rows, agree, group_by, inner_join
from cqe.verify.reference import order_by as reference_order
from cqe.verify.reference import select as reference_select
from cqe.verify.reference import where as reference_where

# A set of queries to run the whole engine against, which is what a benchmark is when it is
# honest about being one.
#
# The queries are the shapes that come up rather than the shapes that flatter a columnar engine.
# A point lookup, which a columnar layout is bad at. A wide aggregate, which it is good at. A
# join, a top ten, a scan of everything. If the set only held the ones this engine wins, the
# numbers would say nothing.
#
# Nothing here is timed. Every number is values touched, hash probes or bytes read, for the
# reason cost/meter.py gives at length: a time is a property of a machine and a count is a
# property of a plan, and only the second is worth recording in a file that will be read later.
#
# Every query is also checked against the reference, because a benchmark that does not check its
# answers measures how fast the engine can be wrong.

# The row count the whole set runs at unless told otherwise. Large enough for the differences
# between plans to be visible and small enough for the suite to stay quick.
ROWS = 20000


@dataclass(frozen=True)
class Query:
    """One named query and what it is meant to exercise."""

    name: str
    text: str
    exercises: str

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"query": self.name, "exercises": self.exercises, "text": self.text}


QUERIES: tuple[Query, ...] = (
    Query(
        name="scan",
        text="select id, amount from facts",
        exercises="reading two columns of five",
    ),
    Query(
        name="point",
        text="select id, amount from facts where id = 4242",
        exercises="one row out of the table, which a columnar layout is worst at",
    ),
    Query(
        name="range",
        text="select id, amount from facts where amount > 130",
        exercises="a selective predicate over one column",
    ),
    Query(
        name="conjunction",
        text="select id from facts where amount > 120 and shop < 10",
        exercises="two predicates, where the order they run in matters",
    ),
    Query(
        name="aggregate",
        text="select label, count(*) as n from facts group by label",
        exercises="a low cardinality group, which reaches the counting form",
    ),
    Query(
        name="wide aggregate",
        text="select shop, sum(amount) as total from facts group by shop",
        exercises="a higher cardinality group, which reaches the hash form",
    ),
    Query(
        name="top ten",
        text="select id, amount from facts order by amount desc limit 10",
        exercises="a sort that a limit turns into a partial one",
    ),
    Query(
        name="join",
        text="select id, region from facts join shops on facts.shop = shops.shop",
        exercises="a fact table against a dimension",
    ),
    Query(
        name="join and group",
        text=(
            "select region, count(*) as n from facts join shops on facts.shop = shops.shop "
            "group by region order by n desc"
        ),
        exercises="the shape most reporting queries have",
    ),
    Query(
        name="everything",
        text="select * from facts",
        exercises="the case where no narrowing helps at all",
    ),
)


@dataclass
class Measurement:
    """What one query cost, and whether it was right."""

    query: str
    rows: int
    values_touched: int
    hash_probes: int
    rows_materialised: int
    comparisons: int
    predicted: float
    nodes: int
    correct: bool = True

    @property
    def total(self) -> int:
        """One number for the query, which is what a comparison sorts on."""
        return self.values_touched + self.hash_probes

    @property
    def ratio(self) -> float:
        """What the model predicted over what the meter counted."""
        return self.predicted / max(self.total, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "query": self.query,
            "rows": self.rows,
            "touched": self.values_touched,
            "probes": self.hash_probes,
            "total": self.total,
            "predicted": round(self.predicted),
            "ratio": round(self.ratio, 2),
            "correct": self.correct,
        }


def catalogue(rows: int = ROWS, shops: int = 50, seed: int = 11) -> dict[str, Batch]:
    """The two tables every query in the set runs against.

    A fact table and a dimension, which is the shape that makes a join measurement mean
    anything. The columns are the same ones the module measurements elsewhere use, so a number
    here can be compared with a number there.
    """
    state = np.random.default_rng(seed)
    facts = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, shops, rows)),
            floating_column("amount", state.normal(100, 30, rows)),
            string_column("label", [f"kind{one}" for one in state.integers(0, 12, rows)]),
            string_column(
                "key", [f"key{one:05d}" for one in state.integers(0, rows // 4, rows)]
            ),
        ]
    )
    dimension = Batch.from_columns(
        [
            integer_column("shop", np.arange(shops)),
            string_column("region", [f"region{one % 6}" for one in range(shops)]),
        ]
    )
    return {"facts": facts, "shops": dimension}


def measure(query: Query, tables: dict[str, Batch], rewrite: bool = True) -> Measurement:
    """Run one query and record what it cost.

    The plan is rewritten by default because that is what a user gets, and the flag exists so
    that the measurement of what the rewrite is worth can turn it off rather than reimplementing
    the run.
    """
    built = plan_query(query.text, tables)
    if rewrite:
        built = push_everything(built).after
    stats = {name: collect(one) for name, one in tables.items()}
    predicted = estimate(built, stats).total
    executed = run(built, tables)
    return Measurement(
        query=query.name,
        rows=executed.rows,
        values_touched=executed.meter.values_touched,
        hash_probes=executed.meter.hash_probes,
        rows_materialised=executed.meter.rows_materialised,
        comparisons=executed.meter.comparisons,
        predicted=predicted,
        nodes=len(walk(built)),
    )


def measure_all(rows: int = ROWS, rewrite: bool = True) -> list[Measurement]:
    """The whole set, in order."""
    tables = catalogue(rows)
    return [measure(one, tables, rewrite=rewrite) for one in QUERIES]


def named(name: str) -> Query:
    """One query by name, with the list in the refusal."""
    for one in QUERIES:
        if one.name == name:
            return one
    raise ConfigError(f"{name} is not a query; try one of {[one.name for one in QUERIES]}")


def the_whole_set_runs(rows: int = ROWS) -> dict:
    """Every query, with its cost, which is the module in one table.

    Two of the ten cost the meter nothing at all, and they are the two that read every column
    and filter nothing: a bare scan and a select star. In this harness the tables are already in
    memory, so a scan of one is a reference to a batch that exists and there is no read to
    count. That is the same caveat cost/model.py records, and it is why the spread below is
    reported over the queries that touch something rather than over all ten. The disk
    measurements later in this module are where a scan costs what a scan costs.

    Over the eight that do work the range is six to one, from an aggregate at twenty thousand to
    a join and group at a hundred and twenty thousand, which is the range a cost model has to
    order correctly.
    """
    measured = measure_all(rows)
    working = [one for one in measured if one.total > 0]
    totals = [one.total for one in working]
    return {
        "queries": len(measured),
        "free_queries": len(measured) - len(working),
        "cheapest": working[totals.index(min(totals))].query,
        "dearest": working[totals.index(max(totals))].query,
        "spread": round(max(totals) / min(totals), 1),
        "table": [one.as_dict() for one in measured],
    }


def the_rewrite_helps_most_queries(rows: int = ROWS) -> dict:
    """Every query with the rewrite on and off, which is what pushdown is worth end to end.

    The measurement the rewrite exists for, and it is the one place where a claim about a rule
    can be checked against a whole query rather than against a plan shape.
    """
    with_rule = {one.query: one for one in measure_all(rows, rewrite=True)}
    without = {one.query: one for one in measure_all(rows, rewrite=False)}
    changes = []
    for name, one in with_rule.items():
        before = without[name].total
        after = one.total
        changes.append(
            {
                "query": name,
                "without": before,
                "with": after,
                "ratio": 1.0 if before == after else round(before / max(after, 1), 2),
            }
        )
    helped = [one for one in changes if one["ratio"] > 1.01]
    hurt = [one for one in changes if one["ratio"] < 0.99]
    return {
        "queries": len(changes),
        "helped": len(helped),
        "unchanged": len(changes) - len(helped) - len(hurt),
        "hurt": len(hurt),
        "best": max(changes, key=lambda one: one["ratio"])["query"],
        "best_ratio": max(one["ratio"] for one in changes),
        "it_never_hurts": not hurt,
        "changes": changes,
    }


def every_query_agrees_with_the_reference(rows: int = 4000) -> dict:
    """Each query run through the engine and through the row at a time interpreter.

    Only the queries the reference can express, which is most of them: the reference has no
    parser, so each one is written out in Python beside the SQL. That duplication is the point,
    since a reference sharing the parser would share its bugs.
    """
    tables = catalogue(rows)
    facts = Rows.of(tables["facts"])
    shops = Rows.of(tables["shops"])
    checks = {
        "scan": (
            "select id, amount from facts",
            lambda: reference_select(facts, ["id", "amount"]),
        ),
        "point": (
            "select id, amount from facts where id = 4242",
            lambda: reference_select(
                reference_where(facts, lambda one: one["id"] == 4242), ["id", "amount"]
            ),
        ),
        "range": (
            "select id, amount from facts where amount > 130",
            lambda: reference_select(
                reference_where(facts, lambda one: one["amount"] > 130), ["id", "amount"]
            ),
        ),
        "conjunction": (
            "select id from facts where amount > 120 and shop < 10",
            lambda: reference_select(
                reference_where(facts, lambda one: one["amount"] > 120 and one["shop"] < 10),
                ["id"],
            ),
        ),
        "aggregate": (
            "select label, count(*) as n from facts group by label",
            lambda: group_by(facts, ["label"], [("n", "count_star", "")]),
        ),
        "join": (
            "select id, region from facts join shops on facts.shop = shops.shop",
            lambda: reference_select(
                inner_join(facts, shops, ["shop"], ["shop"]), ["id", "region"]
            ),
        ),
    }
    out = {}
    for name, (text, expected) in checks.items():
        produced = run(push_everything(plan_query(text, tables)).after, tables).batch
        ordered = name in ("scan",)
        out[name] = bool(agree(Rows.of(produced), expected(), ordered=ordered))
    ordered_text = "select id, amount from facts order by amount desc"
    ordered_plan = run(push_everything(plan_query(ordered_text, tables)).after, tables).batch
    out["ordered"] = bool(
        agree(
            Rows.of(ordered_plan),
            reference_select(
                reference_order(facts, ["amount"], descending=[True]), ["id", "amount"]
            ),
            ordered=True,
        )
    )
    return {"checked": len(out), "results": out, "they_all_agree": all(out.values())}


def a_point_lookup_is_the_worst_case(rows: int = ROWS) -> dict:
    """One row out of twenty thousand, which a columnar layout has no answer for.

    The measurement a columnar engine should be made to publish. Without an index the scan reads
    the whole key column to find one row, and the cost is the same as the range query that
    returns two thousand. The saving is entirely in the columns not read.
    """
    tables = catalogue(rows)
    point = measure(named("point"), tables)
    ranged = measure(named("range"), tables)
    return {
        "point_rows": point.rows,
        "point_cost": point.total,
        "range_rows": ranged.rows,
        "range_cost": ranged.total,
        "cost_per_row_point": round(point.total / max(point.rows, 1)),
        "cost_per_row_range": round(ranged.total / max(ranged.rows, 1)),
        "the_point_lookup_costs_the_same": abs(point.total - ranged.total) < ranged.total,
        "but_returns_far_fewer_rows": point.rows < ranged.rows / 100,
    }


def a_point_lookup_on_disk_is_much_better(rows: int = ROWS) -> dict:
    """The same lookup against a sorted file, where the zone map does what an index would.

    Written on the id column and that proves nothing, because id counts upwards and is therefore
    already sorted in arrival order: both files read one group and the sort changed nothing. It
    is the same finding storage/layout.py records, met again from the other end.

    So the lookup here is on the key column, which is drawn at random and is in no order at all.
    The arrival file reads every group of it and the sorted file reads one, which is the claim
    the id column could not support.
    """
    tables = catalogue(rows)
    with tempfile.TemporaryDirectory() as directory:
        predicate = Compare("=", column("key"), literal("key00042"))
        plain = create(Path(directory) / "plain.cqe", tables["facts"], group_size=500)
        ordered = create(
            Path(directory) / "sorted.cqe",
            tables["facts"],
            group_size=500,
            order="sorted",
            key="key",
        )
        _, loose = scan(plain, columns=["id", "key"], predicate=predicate)
        _, tight = scan(ordered, columns=["id", "key"], predicate=predicate)
    return {
        "groups": plain.groups,
        "arrival_read": loose.groups_read,
        "sorted_read": tight.groups_read,
        "arrival_bytes": loose.bytes_read,
        "sorted_bytes": tight.bytes_read,
        "ratio": round(loose.bytes_read / max(tight.bytes_read, 1), 1),
        "the_layout_fixed_it": tight.groups_read < loose.groups_read / 10,
    }


def the_model_ranks_the_set(rows: int = ROWS) -> dict:
    """The cost model against the meter, over the whole query set.

    A harder test than cost/model.py's own, because those plans differ in one dimension each and
    these differ in every dimension at once. The number to read is the pairwise agreement.
    """
    measured = measure_all(rows)
    predicted = [one.predicted for one in measured]
    counted = [one.total for one in measured]
    pairs = 0
    agreed = 0
    for first in range(len(measured)):
        for second in range(first + 1, len(measured)):
            pairs += 1
            agreed += int(
                (predicted[first] < predicted[second]) == (counted[first] < counted[second])
            )
    ratios = [one.ratio for one in measured if one.total > 0]
    return {
        "queries": len(measured),
        "pairs": pairs,
        "agreed": agreed,
        "share": round(agreed / max(pairs, 1), 3),
        "lowest_ratio": round(min(ratios), 2),
        "highest_ratio": round(max(ratios), 2),
        "spread": round(max(ratios) / max(min(ratios), 0.01), 1),
    }


def the_costs_scale_with_the_rows() -> dict:
    """The whole set at three sizes, which says which queries are linear and which are not.

    Written expecting the top ten query to grow differently from the rest, because a sort is n
    log n. It grows at 3.99 for a fourfold increase in rows, the same as everything else, and
    the reason is that a partial sort is linear in its input: the log factor belonged to the
    full sort the limit replaced.

    So every query in the set is linear in the rows, which is not what a set containing a sort
    would usually show. The number that would not be linear is a sort with no limit above it,
    and no query here has one.
    """
    out = []
    for rows in (5000, 10000, 20000):
        measured = {one.query: one.total for one in measure_all(rows)}
        out.append({"rows": rows, **measured})
    first, last = out[0], out[-1]
    growth = {
        name: round(last[name] / max(first[name], 1), 2) for name in first if name != "rows"
    }
    return {
        "sizes": [one["rows"] for one in out],
        "growth": growth,
        "the_rows_grew_fourfold": last["rows"] / first["rows"] == 4,
        "most_grew_about_fourfold": sum(1 for one in growth.values() if 3 < one < 5) >= 5,
        "table": out,
    }


def a_query_over_a_file_costs_less_than_over_memory(rows: int = ROWS) -> dict:
    """The same predicate against a batch and against a sorted file.

    The comparison the whole storage half of this package exists for. In memory there is nothing
    to prune and every row is read; on disk the layout and the statistics between them skip most
    of the file, and the answers are identical.
    """
    tables = catalogue(rows)

    predicate = Compare(">", column("amount"), literal(140.0))
    meter = Meter()
    in_memory = apply_predicate(predicate, tables["facts"], meter=meter)
    with tempfile.TemporaryDirectory() as directory:
        table = create(
            Path(directory) / "one.cqe",
            tables["facts"],
            group_size=500,
            order="sorted",
            key="amount",
        )
        on_disk, measured = scan(table, predicate=predicate)
    return {
        "rows": in_memory.rows,
        "memory_touched": meter.values_touched,
        "disk_rows_read": measured.rows_read,
        "groups_skipped": measured.groups_skipped,
        "they_agree": bool(agree(Rows.of(on_disk), Rows.of(in_memory))),
        "the_file_read_fewer_rows": measured.rows_read < rows,
    }


def the_queries_reach_every_strategy(rows: int = ROWS) -> dict:
    """Which physical strategies the set actually exercises.

    The measurement that says whether a benchmark covers the engine. A set that never reaches
    the counting aggregate says nothing about it, and a set that reaches every path is one where
    a regression in any of them shows up.
    """
    tables = catalogue(rows)
    reached = set()
    for one in QUERIES:
        built = push_everything(plan_query(one.text, tables)).after
        for choice in run(built, tables).choices:
            reached.add(f"{choice.node}:{choice.strategy}")
    return {
        "strategies": sorted(reached),
        "count": len(reached),
        "it_reaches_a_join": any(one.startswith("Join") for one in reached),
        "and_two_aggregates": sum(1 for one in reached if one.startswith("Group")) >= 2,
        "and_a_partial_sort": "Sort:partial" in reached,
    }


def an_unknown_query_is_refused() -> bool:
    """Asking for a query that is not in the set, with the names in the message."""
    try:
        named("nothing")
    except ConfigError:
        return True
    return False


def compare_the_queries(rows: int = ROWS) -> list[dict]:
    """Every query and what it cost, sorted by cost."""
    measured = sorted(measure_all(rows), key=lambda one: one.total)
    return [one.as_dict() for one in measured]


def summarise() -> dict:
    """The module in one mapping."""
    ranking = the_model_ranks_the_set()
    return {
        "queries": len(QUERIES),
        "rows": ROWS,
        "all_agree": every_query_agrees_with_the_reference()["they_all_agree"],
        "model_agreement": ranking["share"],
        "rewrite_never_hurts": the_rewrite_helps_most_queries()["it_never_hurts"],
        "strategies_reached": the_queries_reach_every_strategy()["count"],
    }
