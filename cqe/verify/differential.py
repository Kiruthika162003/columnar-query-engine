from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cqe.errors import ConfigError, QueryEngineError
from cqe.exec.aggregate import Aggregate, counting_aggregate, hash_aggregate, sorted_aggregate
from cqe.exec.batch import Batch
from cqe.exec.expr import to_callable
from cqe.exec.filter import apply as apply_predicate
from cqe.exec.join.hash import hash_join, nested_loop_join
from cqe.exec.sort import SortKey, order_by, top_k
from cqe.types.schema import FLOATING, INTEGER, STRING
from cqe.verify.fuzz import Case, cases, shrink
from cqe.verify.reference import Rows, agree, group_by, inner_join
from cqe.verify.reference import order_by as reference_order
from cqe.verify.reference import where as reference_where

# The harness that ties the fuzzer to the reference.
#
# Every operator in this package has a fast path built out of numpy and a reference built out of
# Python lists, and they share no code. That is the whole verification strategy: two independent
# implementations of the same definition, checked against each other on inputs neither author
# chose.
#
# The properties checked here are of three kinds and they are worth naming separately.
#
# Agreement. The fast path and the reference return the same rows. This is most of what is
# checked and it is the only one that can catch a wrong answer.
#
# Equivalence between fast paths. Three aggregate strategies, three join strategies, two sort
# strategies, all of which must agree with each other as well as with the reference. This
# catches the case where the strategy chooser picks a path that nobody exercises.
#
# Invariants. Properties that must hold of a result whatever the input: a filter never returns
# more rows than it was given, a sort is a permutation, a join of a key against itself returns
# every row. These catch failures the reference would share, which agreement cannot.
#
# The last kind matters more than it looks. If both implementations misread the same definition
# they agree perfectly and are both wrong, and only an invariant that does not mention either of
# them can find it.

# How many generated cases each check runs by default. Enough that a rare path is reached and
# few enough that the whole suite stays quick.
CASES = 40


@dataclass(frozen=True)
class Report:
    """What one differential check found."""

    name: str
    cases: int
    failures: tuple[Case, ...]
    messages: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """Whether every case agreed."""
        return not self.failures

    @property
    def rate(self) -> float:
        """The share of cases that failed."""
        return len(self.failures) / max(self.cases, 1)

    def first(self) -> str:
        """The first failure, shrunk and rendered, or nothing."""
        if not self.failures:
            return ""
        return f"{self.failures[0].describe()}: {self.messages[0]}"

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "cases": self.cases,
            "failures": len(self.failures),
            "passed": self.passed,
            "first": self.first(),
        }


def _run(name: str, check: Callable[[Case], bool], count: int, seed: int) -> Report:
    """Run one check over generated cases, shrinking every failure."""
    failures = []
    messages = []
    for one in cases(count=count, seed=seed):
        message = _why(check, one)
        if message:
            smaller = shrink(one, lambda case: bool(_why(check, case)))
            failures.append(smaller)
            messages.append(message)
    return Report(name=name, cases=count, failures=tuple(failures), messages=tuple(messages))


def _why(check: Callable[[Case], bool], case: Case) -> str:
    """Empty when a case passes, and why when it does not.

    An engine error counts as a failure. A generated table is a legal table, so an operator that
    refuses one is either wrong or has a refusal that is too broad, and both are worth finding.
    The exception type is kept in the message because the two look different.
    """
    try:
        return "" if check(case) else "they disagreed"
    except QueryEngineError as problem:
        return f"{type(problem).__name__}: {problem}"


def filter_agrees(case: Case) -> bool:
    """A filter against the reference, on the same table and the same predicate."""
    produced = apply_predicate(case.predicate, case.batch)
    predicate = to_callable(case.predicate)
    expected = reference_where(Rows.of(case.batch), predicate)
    return bool(agree(Rows.of(produced), expected))


def filter_never_grows(case: Case) -> bool:
    """A filter returns at most what it was given.

    An invariant rather than an agreement, so it holds even if the reference is wrong in the
    same way. It is the cheapest property in the package and it has caught a selection vector
    applied twice.
    """
    return apply_predicate(case.predicate, case.batch).rows <= case.batch.rows


def filter_is_idempotent(case: Case) -> bool:
    """Filtering twice by the same predicate gives the same rows as filtering once.

    True for a two valued predicate and also for a three valued one, because a null that was
    rejected is not in the second input to be rejected again. The property is worth checking
    because a predicate evaluated against a filtered batch is evaluated against different rows,
    and an operator holding a stale mask would fail here and nowhere else.
    """
    once = apply_predicate(case.predicate, case.batch)
    twice = apply_predicate(case.predicate, once)
    return once.rows == twice.rows


def sort_agrees(case: Case) -> bool:
    """A sort against the reference, on the first column that can be ordered."""
    name = _sortable(case.batch)
    if name is None:
        return True
    keys = [SortKey(name=name)]
    ordering = order_by(case.batch, keys)
    produced = case.batch.take(ordering.positions)
    expected = reference_order(Rows.of(case.batch), [name])
    return bool(agree(Rows.of(produced), expected, ordered=True))


def sort_is_a_permutation(case: Case) -> bool:
    """A sort returns the same multiset of rows it was given.

    The invariant that catches a sort dropping or duplicating a row, which agreement would only
    catch if the reference did not make the same mistake. Checked on the sorted positions rather
    than on the values, so it holds for a column with repeats in it.
    """
    name = _sortable(case.batch)
    if name is None:
        return True
    ordering = order_by(case.batch, [SortKey(name=name)])
    return sorted(ordering.positions.tolist()) == list(range(case.batch.rows))


def descending_reverses_ascending(case: Case) -> bool:
    """A descending sort is the ascending one reversed, for a column with no nulls.

    Only for a column with no nulls, because nulls go last in both directions by policy and a
    reversal would move them to the front. That policy is exec/sort.py's and this respects it
    rather than testing against it.
    """
    name = _sortable(case.batch, no_nulls=True)
    if name is None or case.batch.rows == 0:
        return True
    up = order_by(case.batch, [SortKey(name=name)]).positions
    down = order_by(case.batch, [SortKey(name=name, descending=True)]).positions
    values = case.batch.column(name).values
    return bool(np.array_equal(values[up], values[down][::-1]))


def top_k_agrees_with_a_full_sort(case: Case) -> bool:
    """The first few of a partial sort are the first few of a full one."""
    name = _sortable(case.batch)
    if name is None or case.batch.rows < 3:
        return True
    keys = [SortKey(name=name)]
    count = max(case.batch.rows // 3, 1)
    partial = top_k(case.batch, keys, count).positions
    whole = order_by(case.batch, keys).positions[:count]
    values = case.batch.column(name).values
    return bool(np.array_equal(values[partial], values[whole]))


def aggregate_agrees(case: Case) -> bool:
    """A hash aggregate against the reference, counting rows per group."""
    name = _groupable(case.batch)
    if name is None:
        return True
    aggregates = [Aggregate(name="n", function="count_star", source="")]
    produced = hash_aggregate(case.batch, [name], aggregates).batch
    expected = group_by(Rows.of(case.batch), [name], [("n", "count_star", "")])
    return bool(agree(Rows.of(produced), expected))


def the_aggregate_strategies_agree(case: Case) -> bool:
    """Hash, sorted and counting produce the same groups.

    The check that catches a strategy nobody exercises, and it did on its first run. It reached
    the counting form with a nullable string key, which the counting form refuses, and the
    refusal came back as an engine error rather than as a disagreement. The bug was in
    plan/physical.py, which chose counting for any dictionary column without checking for nulls,
    so a query grouping by a nullable string crashed. Every hand written test passed because
    none of them grouped by a nullable string.

    This check now respects the same precondition the chooser does, so a failure here from now
    on is a real disagreement rather than a precondition ignored.
    """
    name = _groupable(case.batch)
    if name is None or case.batch.rows == 0:
        return True
    aggregates = [Aggregate(name="n", function="count_star", source="")]
    hashed = hash_aggregate(case.batch, [name], aggregates).batch
    ordering = order_by(case.batch, [SortKey(name=name)])
    ordered = sorted_aggregate(case.batch.take(ordering.positions), [name], aggregates).batch
    if not agree(Rows.of(hashed), Rows.of(ordered)):
        return False
    key = case.batch.column(name)
    if key.field.logical != STRING or key.has_nulls:
        return True
    counted = counting_aggregate(case.batch, name, aggregates).batch
    return bool(agree(Rows.of(hashed), Rows.of(counted)))


def the_group_counts_sum_to_the_rows(case: Case) -> bool:
    """Every row belongs to exactly one group, so the counts add up to the table.

    Except for the null rows, which group/aggregate treats as one group rather than dropping.
    That is exec/aggregate.py's policy and this invariant is written to match it, because an
    invariant that contradicts a stated policy tests the policy rather than the code.
    """
    name = _groupable(case.batch)
    if name is None:
        return True
    aggregates = [Aggregate(name="n", function="count_star", source="")]
    produced = hash_aggregate(case.batch, [name], aggregates).batch
    return int(sum(produced.column("n").values)) == case.batch.rows


def join_agrees(case: Case) -> bool:
    """A hash join of a table against itself, against the reference join."""
    name = _joinable(case.batch)
    if name is None or case.batch.rows > 60:
        return True
    left = case.batch.select([name])
    produced = hash_join(left, left, [name], [name]).batch
    expected = inner_join(Rows.of(left), Rows.of(left), [name], [name])
    return bool(agree(Rows.of(produced), expected))


def the_join_strategies_agree(case: Case) -> bool:
    """A hash join and a nested loop join of the same inputs."""
    name = _joinable(case.batch)
    if name is None or case.batch.rows > 60:
        return True
    left = case.batch.select([name])
    hashed = hash_join(left, left, [name], [name]).batch
    looped = nested_loop_join(left, left, [name], [name]).batch
    return bool(agree(Rows.of(hashed), Rows.of(looped)))


def a_self_join_returns_at_least_the_rows(case: Case) -> bool:
    """A join of a key against itself matches every row with at least itself.

    Unless the key is null, which matches nothing, which is the three valued rule and is the
    single most commonly broken invariant in a join. Stated as an inequality because a key with
    repeats matches more than once.
    """
    name = _joinable(case.batch)
    if name is None or case.batch.rows > 60:
        return True
    left = case.batch.select([name])
    produced = hash_join(left, left, [name], [name]).batch
    column = left.column(name)
    present = case.batch.rows if column.valid is None else int(column.valid.sum())
    return produced.rows >= present


def projection_keeps_the_rows(case: Case) -> bool:
    """A projection changes the columns and never the rows."""
    names = list(case.batch.schema.names)[:1]
    return case.batch.select(names).rows == case.batch.rows


def slicing_partitions_the_table(case: Case) -> bool:
    """A table cut in two has the two halves adding up to it."""
    if case.batch.rows < 2:
        return True
    middle = case.batch.rows // 2
    return (
        case.batch.slice(0, middle).rows + case.batch.slice(middle, case.batch.rows).rows
        == case.batch.rows
    )


def _sortable(batch: Batch, no_nulls: bool = False) -> str | None:
    """The first column that can be ordered, or nothing."""
    for name in batch.schema.names:
        one = batch.column(name)
        if no_nulls and one.valid is not None:
            continue
        if one.field.logical in (INTEGER, FLOATING, STRING):
            return name
    return None


def _groupable(batch: Batch) -> str | None:
    """The first column that can be grouped by."""
    return _sortable(batch)


def _joinable(batch: Batch) -> str | None:
    """The first column that can be a join key.

    Integers only. A float key is legal and comparing floats for equality is a different
    question from whether the join works, and a string key joins through its dictionary, which
    exec/join/hash.py checks separately and at length.
    """
    for name in batch.schema.names:
        if batch.column(name).field.logical == INTEGER:
            return name
    return None


CHECKS: dict[str, Callable[[Case], bool]] = {
    "filter agrees": filter_agrees,
    "filter never grows": filter_never_grows,
    "filter is idempotent": filter_is_idempotent,
    "sort agrees": sort_agrees,
    "sort is a permutation": sort_is_a_permutation,
    "descending reverses ascending": descending_reverses_ascending,
    "top k agrees": top_k_agrees_with_a_full_sort,
    "aggregate agrees": aggregate_agrees,
    "aggregate strategies agree": the_aggregate_strategies_agree,
    "group counts sum": the_group_counts_sum_to_the_rows,
    "join agrees": join_agrees,
    "join strategies agree": the_join_strategies_agree,
    "self join returns the rows": a_self_join_returns_at_least_the_rows,
    "projection keeps the rows": projection_keeps_the_rows,
    "slicing partitions": slicing_partitions_the_table,
}


def run_one(name: str, count: int = CASES, seed: int = 0) -> Report:
    """One named check over generated cases."""
    if name not in CHECKS:
        raise ConfigError(f"{name} is not a check; try one of {sorted(CHECKS)}")
    return _run(name, CHECKS[name], count, seed)


def run_all(count: int = CASES, seed: int = 0) -> list[Report]:
    """Every check over generated cases."""
    return [
        _run(name, check, count, seed + position)
        for position, (name, check) in enumerate(CHECKS.items())
    ]


def failures(count: int = CASES, seed: int = 0) -> list[Report]:
    """Only the checks that found something, which is what a command line prints."""
    return [one for one in run_all(count=count, seed=seed) if not one.passed]


def every_check_passes(count: int = CASES, seed: int = 0) -> dict:
    """The whole harness in one call, which is what the test suite runs.

    Fifteen checks over generated tables, and the number worth watching is not whether it passes
    today but how much it covers: the case count, the column types reached, and the share of
    cases where the check did real work rather than skipping.
    """
    reports = run_all(count=count, seed=seed)
    return {
        "checks": len(reports),
        "cases_each": count,
        "total_cases": len(reports) * count,
        "failing_checks": [one.name for one in reports if not one.passed],
        "they_all_passed": all(one.passed for one in reports),
    }


def a_broken_filter_is_caught(count: int = 30) -> dict:
    """A deliberately wrong filter, to check the harness can fail.

    Every verification harness needs this measurement. Without it there is no evidence that the
    checks would notice a wrong answer, and a harness that cannot fail is a harness that passes.
    """

    def broken(case: Case) -> bool:
        produced = apply_predicate(case.predicate, case.batch)
        wrong = produced.slice(0, max(produced.rows - 1, 0))
        expected = reference_where(Rows.of(case.batch), to_callable(case.predicate))
        return bool(agree(Rows.of(wrong), expected))

    report = _run("broken filter", broken, count, seed=101)
    return {
        "cases": count,
        "failures": len(report.failures),
        "it_was_caught": not report.passed,
        "the_rate": round(report.rate, 3),
        "the_first_is_small": report.failures[0].batch.rows < 20 if report.failures else False,
    }


def a_broken_sort_is_caught(count: int = 30) -> dict:
    """A sort that drops its last row, which the permutation invariant catches.

    Worth separating from the filter case because it is caught by an invariant rather than by
    agreement, and the two failure modes are different: agreement needs a correct reference and
    an invariant does not.
    """

    def broken(case: Case) -> bool:
        name = _sortable(case.batch)
        if name is None or case.batch.rows < 2:
            return True
        ordering = order_by(case.batch, [SortKey(name=name)])
        dropped = ordering.positions[:-1]
        return sorted(dropped.tolist()) == list(range(case.batch.rows))

    report = _run("broken sort", broken, count, seed=103)
    return {
        "cases": count,
        "failures": len(report.failures),
        "it_was_caught": not report.passed,
        "the_rate": round(report.rate, 3),
    }


def a_broken_aggregate_is_caught(count: int = 30) -> dict:
    """A group count that drops the nulls, which the sum invariant catches.

    The specific failure is the one every aggregate gets wrong first: nulls are a group, not an
    absence, and an implementation that skips them returns counts that do not add up.
    """

    def broken(case: Case) -> bool:
        name = _groupable(case.batch)
        if name is None:
            return True
        one = case.batch.column(name)
        kept = case.batch if one.valid is None else case.batch.mask(one.valid)
        aggregates = [Aggregate(name="n", function="count_star", source="")]
        produced = hash_aggregate(kept, [name], aggregates).batch
        return int(sum(produced.column("n").values)) == case.batch.rows

    report = _run("broken aggregate", broken, count, seed=107)
    return {
        "cases": count,
        "failures": len(report.failures),
        "it_was_caught": not report.passed,
        "the_rate": round(report.rate, 3),
        "the_nulls_are_what_it_found": report.rate > 0.2,
    }


def the_checks_do_real_work(count: int = 40) -> dict:
    """How often each check does something rather than skipping.

    A check that returns true because it found no column of the right type has tested nothing,
    and a harness reporting fifteen passing checks where half of them skipped is worse than one
    reporting seven. This counts the skips explicitly.
    """
    made = cases(count=count, seed=109)
    return {
        "cases": count,
        "sortable": sum(1 for one in made if _sortable(one.batch) is not None),
        "joinable": sum(1 for one in made if _joinable(one.batch) is not None),
        "small_enough_to_join": sum(
            1 for one in made if _joinable(one.batch) is not None and one.batch.rows <= 60
        ),
        "with_nulls": sum(
            1 for one in made if any(column.valid is not None for column in one.batch.columns)
        ),
        "most_are_sortable": sum(1 for one in made if _sortable(one.batch) is not None)
        > count * 0.8,
    }


def a_failure_is_reported_small(count: int = 40) -> dict:
    """How small the shrunk failures come out, which is what makes a report readable.

    Measured against the cases that produced them rather than in absolute terms, because a case
    that started at three rows cannot shrink much and should not count against the shrinker.
    """

    def broken(case: Case) -> bool:
        return case.batch.rows < 15

    report = _run("always fails above fifteen rows", broken, count, seed=113)
    if not report.failures:
        return {"failures": 0, "it_found_nothing": True}
    sizes = [one.batch.rows for one in report.failures]
    return {
        "failures": len(report.failures),
        "largest_shrunk": max(sizes),
        "smallest_shrunk": min(sizes),
        "average": round(sum(sizes) / len(sizes), 1),
        "they_are_all_near_the_boundary": max(sizes) <= 20,
    }


def an_unknown_check_is_refused() -> bool:
    """Asking for a check that does not exist, with the list in the message."""
    try:
        run_one("nothing")
    except ConfigError:
        return True
    return False


def a_report_renders_its_first_failure() -> dict:
    """What a failure looks like when it is printed."""

    def broken(case: Case) -> bool:
        return case.batch.rows < 5

    report = _run("small tables only", broken, count=20, seed=127)
    return {
        "failures": len(report.failures),
        "rendered": report.first(),
        "it_has_a_predicate": "(" in report.first(),
        "it_has_a_row_count": "rows" in report.first(),
        "a_passing_report_renders_nothing": Report(name="none", cases=1, failures=()).first()
        == "",
    }


def compare_the_checks(count: int = 20) -> list[dict]:
    """Every check and what it found, which is the module in one table."""
    return [one.as_dict() for one in run_all(count=count, seed=131)]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "checks": len(CHECKS),
        "cases": CASES,
        "all_pass": every_check_passes(count=20)["they_all_passed"],
        "a_broken_filter_is_caught": a_broken_filter_is_caught(count=20)["it_was_caught"],
        "a_broken_sort_is_caught": a_broken_sort_is_caught(count=20)["it_was_caught"],
        "a_broken_aggregate_is_caught": a_broken_aggregate_is_caught(count=20)["it_was_caught"],
    }
