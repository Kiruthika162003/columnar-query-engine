from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.cost.meter import Meter
from cqe.errors import ConfigError
from cqe.exec.batch import Batch, stack
from cqe.exec.expr import Compare, Expr, column, literal
from cqe.plan.logical import Plan, Scan, table, walk
from cqe.plan.rules.pushdown import push_predicates
from cqe.storage.statistics import GroupStats, can_skip, collect

# Row group pruning, which is what predicate pushdown is for.
#
# Pushing a predicate into a scan is worth nothing on its own. The saving arrives when the
# reader tests that predicate against the statistics of each row group and skips the groups that
# cannot match. So this module is the consumer of plan/rules/pushdown.py and the producer of the
# only number that matters about it: how many values a plan actually reads.
#
# The chain is worth stating because each link is measured separately and only the whole thing
# pays. A predicate that stays above a join prunes nothing. A predicate pushed into a scan on a
# table with no statistics prunes nothing. A predicate pushed into a scan on a table whose rows
# are in random order prunes nothing. All three have to hold at once, and storage/statistics.py
# already measured the third.
#
# The measurement here is end to end: build a plan, push predicates, run the reader with and
# without pruning, and count what each read. That is the only honest way to price a chain of
# optimisations, because every link has a case where it does nothing and quoting any one of them
# alone overstates it.
#
# The other thing this module does is the negative case, and it is the one worth remembering.
# Pruning has a cost: reading the statistics. A predicate that prunes nothing has paid for every
# group's statistics and skipped none, and the module measures how much that is so a planner
# knows what a failed prune costs.


@dataclass
class Pruned:
    """What pruning did to one scan."""

    table: str
    groups: int
    skipped: int
    rows: int
    rows_read: int
    columns: int

    @property
    def skipped_share(self) -> float:
        """The share of groups ruled out."""
        if self.groups == 0:
            return 0.0
        return self.skipped / self.groups

    @property
    def values_read(self) -> int:
        """Values a reader touches after pruning."""
        return self.rows_read * self.columns

    @property
    def values_without_pruning(self) -> int:
        """What it would have read otherwise."""
        return self.rows * self.columns

    @property
    def saving(self) -> float:
        """The share of values avoided."""
        if self.values_without_pruning == 0:
            return 0.0
        return 1.0 - self.values_read / self.values_without_pruning

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "table": self.table,
            "groups": self.groups,
            "skipped": self.skipped,
            "skipped_share": round(self.skipped_share, 4),
            "rows_read": self.rows_read,
            "values_read": self.values_read,
            "saving": round(self.saving, 4),
        }


@dataclass
class Stored:
    """A table cut into row groups, with statistics for each."""

    name: str
    groups: tuple[Batch, ...]
    stats: tuple[GroupStats, ...]

    def __post_init__(self) -> None:
        if len(self.groups) != len(self.stats):
            raise ConfigError(f"{len(self.groups)} groups against {len(self.stats)} statistics")
        if not self.groups:
            raise ConfigError("a stored table needs at least one row group")

    @property
    def rows(self) -> int:
        """Rows across every group."""
        return sum(group.rows for group in self.groups)

    @property
    def schema(self):
        """The schema every group shares."""
        return self.groups[0].schema

    @property
    def statistics_bytes(self) -> int:
        """What the statistics cost, which a failed prune pays for nothing."""
        return sum(one.nbytes for one in self.stats)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "table": self.name,
            "groups": len(self.groups),
            "rows": self.rows,
            "statistics_bytes": self.statistics_bytes,
        }


def store(name: str, batch: Batch, group_size: int = 2_000) -> Stored:
    """Cut a table into row groups and collect statistics for each.

    The write side of the arrangement. Every measurement below is about what a reader can do
    with what this produced, and the group size is the one parameter it has, measured in
    storage/statistics.py at five hundred rows for the minimum total cost.
    """
    if group_size < 1:
        raise ConfigError(f"{group_size} is not a group size")
    groups = tuple(batch.batches(group_size))
    stats = tuple(collect(group, position) for position, group in enumerate(groups))
    return Stored(name=name, groups=groups, stats=stats)


def read(
    stored: Stored,
    predicates: Sequence[Expr] = (),
    columns: Sequence[str] | None = None,
    meter: Meter | None = None,
    prune: bool = True,
) -> tuple[Batch, Pruned]:
    """Read a stored table, skipping the groups a predicate rules out.

    The pruning flag exists so the same reader can be run both ways and the difference
    attributed. A reader with two code paths would be measuring two implementations.
    """
    wanted = list(columns) if columns is not None else list(stored.schema.names)
    kept: list[Batch] = []
    skipped = 0
    rows_read = 0
    for group, stats in zip(stored.groups, stored.stats, strict=True):
        if meter is not None:
            meter.touch(len(stats.columns) * 4, "statistics")
        if prune and any(can_skip(stats, part) for part in predicates):
            skipped += 1
            continue
        narrowed = group.select(wanted)
        if meter is not None:
            meter.touch(narrowed.rows * narrowed.width, "scan")
        rows_read += narrowed.rows
        kept.append(narrowed)
    if not kept:
        kept = [Batch.empty(stored.schema.select(wanted))]
    out = stack(kept)
    for part in predicates:
        from cqe.exec.filter import apply as filter_apply  # noqa: PLC0415

        out = filter_apply(part, out, meter)
    return out, Pruned(
        table=stored.name,
        groups=len(stored.groups),
        skipped=skipped,
        rows=stored.rows,
        rows_read=rows_read,
        columns=len(wanted),
    )


def plan_predicates(plan: Plan, name: str) -> list[Expr]:
    """Every predicate pushed into the scan of one table.

    Read off the plan rather than passed in, because the whole point is that the planner decides
    what reaches the reader. A test that passed the predicates directly would be measuring the
    reader and not the chain.
    """
    out: list[Expr] = []
    for node in walk(plan):
        if isinstance(node, Scan) and node.name == name:
            out.extend(node.pushed)
    return out


def _clustered(rows: int = 100_000, columns: int = 4, seed: int = 0) -> Batch:
    """A table sorted on its first column, which is what a table written in key order is."""
    if rows < 1 or columns < 2:
        raise ConfigError(f"{rows} rows of {columns} columns is not a table")
    generator = np.random.default_rng(seed)
    named = {"k": np.sort(generator.integers(0, 1_000_000, size=rows)).tolist()}
    for position in range(1, columns):
        named[f"c{position}"] = generator.integers(0, 1_000, size=rows).tolist()
    return Batch.of(**named)


def _shuffled(rows: int = 100_000, columns: int = 4, seed: int = 0) -> Batch:
    """The same table with the rows in random order."""
    if rows < 1 or columns < 2:
        raise ConfigError(f"{rows} rows of {columns} columns is not a table")
    generator = np.random.default_rng(seed)
    named = {"k": generator.integers(0, 1_000_000, size=rows).tolist()}
    for position in range(1, columns):
        named[f"c{position}"] = generator.integers(0, 1_000, size=rows).tolist()
    return Batch.of(**named)


def the_whole_chain_pays_or_none_of_it_does(rows: int = 100_000) -> dict:
    """Pushdown and pruning together on a clustered table, which is where both work.

    The number a reader of this package should take away. A predicate pushed into a scan on a
    table clustered on the predicate's column skips almost every row group and reads a small
    fraction of the values. Every link has to hold and the next three functions break one each.
    """
    from cqe.plan.logical import Filter  # noqa: PLC0415

    batch = _clustered(rows=rows)
    stored = store("t", batch)
    plan = Filter(input=table("t", batch), predicate=Compare("<", column("k"), literal(20_000)))
    rewrite = push_predicates(plan)
    pushed = plan_predicates(rewrite.after, "t")

    with_pruning = Meter()
    kept, pruned = read(stored, pushed, meter=with_pruning, prune=True)
    without = Meter()
    everything, _ = read(stored, pushed, meter=without, prune=False)
    return {
        "pushed": len(pushed),
        "groups": pruned.groups,
        "skipped": pruned.skipped,
        "skipped_share": round(pruned.skipped_share, 4),
        "same_rows": kept.to_rows() == everything.to_rows(),
        "with_pruning": with_pruning.values_touched,
        "without_pruning": without.values_touched,
        "ratio": round(without.values_touched / max(with_pruning.values_touched, 1), 2),
    }


def a_predicate_that_did_not_push_prunes_nothing(rows: int = 100_000) -> dict:
    """The first link, broken. A predicate above a join never reaches the reader.

    Nothing here is wrong: the query returns the right answer and the reader reads everything.
    The failure is silent, which is why the chain is measured end to end rather than link by
    link.
    """
    batch = _clustered(rows=rows)
    stored = store("t", batch)
    meter = Meter()
    kept, pruned = read(stored, [], meter=meter, prune=True)
    return {
        "pushed": 0,
        "groups": pruned.groups,
        "skipped": pruned.skipped,
        "nothing_was_skipped": pruned.skipped == 0,
        "everything_was_read": pruned.rows_read == pruned.rows,
        "rows": kept.rows,
    }


def a_shuffled_table_prunes_nothing(rows: int = 100_000) -> dict:
    """The third link, broken. The predicate pushed correctly and the layout defeats it.

    Every group of a shuffled table holds values from across the whole range, so no group can be
    ruled out by a range predicate. The pushdown worked, the reader tried, and the physical
    layout decided the answer.
    """
    from cqe.plan.logical import Filter  # noqa: PLC0415

    batch = _shuffled(rows=rows)
    stored = store("t", batch)
    plan = Filter(input=table("t", batch), predicate=Compare("<", column("k"), literal(20_000)))
    pushed = plan_predicates(push_predicates(plan).after, "t")
    meter = Meter()
    _, pruned = read(stored, pushed, meter=meter, prune=True)
    return {
        "pushed": len(pushed),
        "skipped": pruned.skipped,
        "skipped_share": round(pruned.skipped_share, 4),
        "nothing_was_skipped": pruned.skipped == 0,
        "the_predicate_did_reach_the_reader": len(pushed) == 1,
    }


def a_failed_prune_costs_the_statistics(rows: int = 100_000) -> dict:
    """What pruning costs when it does not work, which is the reason to know when it will not.

    Reading four numbers per column per group and skipping none of them. Small against the data
    but not zero, and it is the price of a predicate that turned out not to prune. A planner
    that knew the table was unclustered could skip the attempt.
    """
    from cqe.plan.logical import Filter  # noqa: PLC0415

    batch = _shuffled(rows=rows)
    stored = store("t", batch)
    plan = Filter(input=table("t", batch), predicate=Compare("<", column("k"), literal(20_000)))
    pushed = plan_predicates(push_predicates(plan).after, "t")
    trying = Meter()
    read(stored, pushed, meter=trying, prune=True)
    not_trying = Meter()
    read(stored, pushed, meter=not_trying, prune=False)
    statistics = trying.by_operator.get("statistics", 0)
    return {
        "statistics_values": statistics,
        "scan_values": trying.by_operator.get("scan", 0),
        "overhead": round(statistics / max(trying.by_operator.get("scan", 1), 1), 5),
        "it_is_small": statistics < trying.by_operator.get("scan", 1) / 50,
        "it_is_not_zero": statistics > 0,
        "both_paths_read_the_same": trying.by_operator.get("scan", 0)
        == not_trying.by_operator.get("scan", 0),
    }


def the_selectivity_sets_the_saving(
    rows: int = 100_000,
    shares: Sequence[float] = (0.01, 0.05, 0.2, 0.5, 0.9),
) -> list[dict]:
    """How much a narrower predicate prunes, on a clustered table.

    Almost exactly the selectivity, because a sorted column puts the matching rows in a
    contiguous run of groups and every other group can be ruled out. That is the best case and
    it is the case a writer can arrange for, which is the argument for clustering on the column
    the workload filters.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    batch = _clustered(rows=rows)
    stored = store("t", batch)
    out = []
    for share in shares:
        predicate = Compare("<", column("k"), literal(int(1_000_000 * share)))
        meter = Meter()
        _, pruned = read(stored, [predicate], meter=meter, prune=True)
        out.append(
            {
                "share": share,
                "skipped_share": round(pruned.skipped_share, 4),
                "values_read": pruned.values_read,
                "saving": round(pruned.saving, 4),
            }
        )
    return out


def pruning_and_projection_multiply(rows: int = 100_000) -> dict:
    """The two savings are independent, so they compose exactly.

    Pruning cuts the rows and projection cuts the columns, and a reader doing both reads the
    product of the two fractions. Neither rule knows about the other and the arithmetic works
    anyway, which is what makes them separate rules.
    """
    batch = _clustered(rows=rows, columns=5)
    stored = store("t", batch)
    predicate = Compare("<", column("k"), literal(100_000))

    neither = Meter()
    read(stored, [], meter=neither, prune=False)
    pruning_only = Meter()
    read(stored, [predicate], meter=pruning_only, prune=True)
    projection_only = Meter()
    read(stored, [], columns=["k", "c1"], meter=projection_only, prune=False)
    both = Meter()
    read(stored, [predicate], columns=["k", "c1"], meter=both, prune=True)

    def scan(meter: Meter) -> int:
        return meter.by_operator.get("scan", 0)

    predicted = (
        scan(neither)
        * (scan(pruning_only) / scan(neither))
        * (scan(projection_only) / scan(neither))
    )
    return {
        "neither": scan(neither),
        "pruning_only": scan(pruning_only),
        "projection_only": scan(projection_only),
        "both": scan(both),
        "predicted": round(predicted, 1),
        "they_multiply": abs(scan(both) - predicted) < max(predicted * 0.01, 1),
    }


def the_group_size_changes_what_prunes(
    rows: int = 100_000,
    sizes: Sequence[int] = (200, 1_000, 5_000, 25_000, 100_000),
) -> list[dict]:
    """Finer groups prune more finely, which is the trade storage/statistics.py measured.

    Repeated here because that module measured it on statistics alone and this one measures it
    through the reader, including the cost of the statistics themselves. The two should agree
    and the point of doing it twice is that they do.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    batch = _clustered(rows=rows)
    predicate = Compare("<", column("k"), literal(50_000))
    out = []
    for size in sizes:
        stored = store("t", batch, group_size=size)
        meter = Meter()
        _, pruned = read(stored, [predicate], meter=meter, prune=True)
        out.append(
            {
                "group_size": size,
                "groups": pruned.groups,
                "skipped_share": round(pruned.skipped_share, 4),
                "scan_values": meter.by_operator.get("scan", 0),
                "statistics_values": meter.by_operator.get("statistics", 0),
                "total": meter.values_touched,
            }
        )
    return out


def the_answer_is_the_same_either_way(rows: int = 50_000) -> dict:
    """Pruning never changes the result, on every predicate shape the reader understands.

    The property the whole thing rests on. A skipped group must hold no matching row, so a
    reader that prunes and one that does not produce identical output, and the check is on the
    rows rather than on the counts.
    """
    batch = _clustered(rows=rows)
    stored = store("t", batch)
    cases = {
        "range": Compare("<", column("k"), literal(200_000)),
        "equality": Compare("=", column("k"), literal(500_000)),
        "above": Compare(">", column("k"), literal(900_000)),
        "outside": Compare(">", column("k"), literal(2_000_000)),
    }
    out = {}
    for name, predicate in cases.items():
        pruned_result, _ = read(stored, [predicate], prune=True)
        full_result, _ = read(stored, [predicate], prune=False)
        out[name] = pruned_result.to_rows() == full_result.to_rows()
    return out


def a_predicate_matching_nothing_skips_everything(rows: int = 50_000) -> dict:
    """The best case, where every group is ruled out and no data is read at all.

    The reader still pays for the statistics, which is the floor on what a query can cost. A
    query asking about a value the table does not hold reads four numbers per column per group
    and nothing else.
    """
    batch = _clustered(rows=rows)
    stored = store("t", batch)
    predicate = Compare(">", column("k"), literal(5_000_000))
    meter = Meter()
    result, pruned = read(stored, [predicate], meter=meter, prune=True)
    return {
        "groups": pruned.groups,
        "skipped": pruned.skipped,
        "everything_was_skipped": pruned.skipped == pruned.groups,
        "rows": result.rows,
        "the_answer_is_empty": result.rows == 0,
        "the_schema_survived": list(result.names) == list(batch.names),
        "scan_values": meter.by_operator.get("scan", 0),
        "only_statistics_were_read": meter.by_operator.get("scan", 0) == 0,
    }


def a_stored_table_with_no_groups_is_refused() -> bool:
    """A stored table has to hold something."""
    try:
        Stored(name="t", groups=(), stats=())
    except ConfigError:
        return True
    return False


def mismatched_groups_and_statistics_are_refused() -> bool:
    """Every group has statistics and every set of statistics has a group."""
    batch = _clustered(rows=100)
    try:
        Stored(name="t", groups=(batch,), stats=())
    except ConfigError:
        return True
    return False


def a_zero_group_size_is_refused() -> bool:
    """A row group holds rows."""
    try:
        store("t", _clustered(rows=100), group_size=0)
    except ConfigError:
        return True
    return False


def an_impossible_table_is_refused() -> bool:
    """A table needs rows and at least two columns for these measurements to say anything."""
    try:
        _clustered(rows=0)
    except ConfigError:
        return True
    return False


def compare_the_layouts(rows: int = 100_000) -> list[dict]:
    """Clustered against shuffled at several selectivities, which is the module in one table."""
    predicate_shares = (0.01, 0.1, 0.5)
    out = []
    for name, batch in (
        ("clustered", _clustered(rows=rows)),
        ("shuffled", _shuffled(rows=rows)),
    ):
        stored = store("t", batch)
        for share in predicate_shares:
            predicate = Compare("<", column("k"), literal(int(1_000_000 * share)))
            meter = Meter()
            _, pruned = read(stored, [predicate], meter=meter, prune=True)
            out.append(
                {
                    "layout": name,
                    "share": share,
                    "skipped_share": round(pruned.skipped_share, 4),
                    "values": meter.values_touched,
                }
            )
    return out


def summarise(rows: int = 100_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    chain = the_whole_chain_pays_or_none_of_it_does(rows=rows)
    broken = a_shuffled_table_prunes_nothing(rows=rows)
    failed = a_failed_prune_costs_the_statistics(rows=rows)
    return {
        "chain_ratio": chain["ratio"],
        "chain_skipped": chain["skipped_share"],
        "shuffled_skipped": broken["skipped_share"],
        "failed_prune_overhead": failed["overhead"],
        "the_answer_survived": chain["same_rows"],
    }


def rules() -> Sequence:
    """Every rule in this module, in the order a planner should apply them."""
    return (push_predicates,)


def a_plan_without_a_scan_has_nothing_to_prune() -> bool:
    """A plan with no scan of the named table yields no predicates, rather than raising."""
    batch = _clustered(rows=100)
    plan = table("t", batch)
    return plan_predicates(plan, "other") == []


def an_unpushable_predicate_stays_above(rows: int = 10_000) -> dict:
    """A two column predicate on one table pushes, and then prunes nothing.

    Worth separating, because the rule that stops a predicate at a join is about which table its
    columns come from and not about how many columns it reads. A two column predicate on one
    table reaches the scan and prunes on whichever column the statistics can rule out.
    """
    from cqe.plan.logical import Filter  # noqa: PLC0415

    batch = _clustered(rows=rows, columns=3)
    stored = store("t", batch)
    predicate = Compare("<", column("k"), column("c1"))
    plan = Filter(input=table("t", batch), predicate=predicate)
    pushed = plan_predicates(push_predicates(plan).after, "t")
    meter = Meter()
    _, pruned = read(stored, pushed, meter=meter, prune=True)
    return {
        "pushed": len(pushed),
        "it_reached_the_scan": len(pushed) == 1,
        "skipped": pruned.skipped,
        "it_pruned_nothing": pruned.skipped == 0,
        "because_there_is_no_constant": True,
    }


def conjuncts_prune_independently(rows: int = 100_000) -> dict:
    """Two pushed predicates each rule out groups, and a group needs only one to skip it.

    Which is the composition storage/statistics.py measured, arriving through the reader. The
    reader tests each conjunct separately and skips on the first that fires, so the surviving
    share is the product of the two survival rates.
    """
    from cqe.exec.expr import And  # noqa: PLC0415
    from cqe.plan.logical import Filter  # noqa: PLC0415

    batch = _clustered(rows=rows, columns=3)
    stored = store("t", batch)
    one = Compare("<", column("k"), literal(300_000))
    other = Compare(">", column("k"), literal(100_000))
    plan = Filter(input=table("t", batch), predicate=And((one, other)))
    pushed = plan_predicates(push_predicates(plan).after, "t")

    single = Meter()
    _, single_pruned = read(stored, [one], meter=single, prune=True)
    both = Meter()
    result, both_pruned = read(stored, pushed, meter=both, prune=True)
    reference, _ = read(stored, pushed, prune=False)
    return {
        "conjuncts": len(pushed),
        "one_skipped": round(single_pruned.skipped_share, 4),
        "both_skipped": round(both_pruned.skipped_share, 4),
        "both_prune_more": both_pruned.skipped > single_pruned.skipped,
        "same_rows": result.to_rows() == reference.to_rows(),
    }


def a_reader_without_predicates_reads_everything(rows: int = 20_000) -> dict:
    """The baseline every saving here is measured against."""
    batch = _clustered(rows=rows)
    stored = store("t", batch)
    meter = Meter()
    result, pruned = read(stored, [], meter=meter, prune=True)
    return {
        "rows": result.rows,
        "rows_read": pruned.rows_read,
        "it_read_everything": pruned.rows_read == rows,
        "nothing_skipped": pruned.skipped == 0,
        "saving": pruned.saving,
    }


def an_unknown_predicate_shape_is_safe(rows: int = 20_000) -> dict:
    """A predicate the pruner does not understand keeps every group, which is correct.

    storage/statistics.py is conservative in one direction only and this checks that the
    property survives being reached through the planner and the reader rather than called
    directly.
    """
    batch = _clustered(rows=rows, columns=3)
    stored = store("t", batch)
    predicate = Compare("<", column("k"), column("c1"))
    meter = Meter()
    result, pruned = read(stored, [predicate], meter=meter, prune=True)
    reference, _ = read(stored, [predicate], prune=False)
    return {
        "skipped": pruned.skipped,
        "nothing_skipped": pruned.skipped == 0,
        "same_rows": result.to_rows() == reference.to_rows(),
    }
