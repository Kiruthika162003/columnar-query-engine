from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cqe.columns.array import (
    Column,
    boolean_column,
    floating_column,
    integer_column,
    string_column,
)
from cqe.errors import ConfigError, QueryEngineError
from cqe.exec.batch import Batch
from cqe.exec.expr import (
    And,
    Compare,
    Expr,
    InList,
    IsNull,
    Not,
    Or,
    column,
    describe,
    literal,
)
from cqe.exec.filter import apply as apply_predicate
from cqe.types.schema import BOOLEAN, FLOATING, INTEGER, STRING, Field

# Random tables and random predicates, for finding the cases nobody thought to write a test for.
#
# A fuzzer is only useful if it generates the awkward cases more often than chance would, so the
# generators here are weighted rather than uniform. A uniform random float column has no nulls,
# no repeated values, no zeros and nothing at the boundary of anything, and every one of those
# is where the bugs are. The generators below produce nulls at a set rate, repeat values on
# purpose, and put values exactly on the boundaries a predicate will compare against.
#
# The second half of being useful is shrinking. A failing case of four hundred rows and a
# predicate six levels deep says almost nothing; the same failure on three rows and one
# comparison says what is wrong. Every failure found here is shrunk before it is reported, and
# the shrinker is measured on how much it removes.
#
# Nothing here checks correctness on its own. It generates cases and hands them to
# verify/differential.py, which is what compares the fast path against the reference.

# How often a generated column has nulls in it at all, and what share of its rows are null when
# it does. Both are high on purpose: a null is one branch in most operators and both sides of it
# need reaching.
NULL_COLUMN_RATE = 0.4
NULL_VALUE_RATE = 0.25

# How often a generated integer column repeats its values heavily, which is what makes groups
# and join matches happen rather than being one row each.
REPEATED_RATE = 0.5

# The deepest predicate the generator will build. Deeper ones are not more likely to find a bug
# and are much harder to read when they do.
MAX_DEPTH = 3


@dataclass
class Generator:
    """A source of random tables and predicates, with everything it produces reproducible."""

    seed: int = 0
    state: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.state = np.random.default_rng(self.seed)

    def rows(self, low: int = 1, high: int = 200) -> int:
        """How many rows the next table has."""
        return int(self.state.integers(low, high + 1))

    def column(self, rows: int, logical: str = "") -> Column:
        """One column of a random type, with nulls at the set rate."""
        kind = logical or str(self.state.choice([INTEGER, FLOATING, STRING, BOOLEAN]))
        made = self._values(kind, rows)
        if self.state.random() < NULL_COLUMN_RATE and rows:
            valid = self.state.random(rows) > NULL_VALUE_RATE
            return Column(
                field=made.field, values=made.values, valid=valid, dictionary=made.dictionary
            )
        return made

    def _values(self, kind: str, rows: int) -> Column:
        """The values themselves, weighted towards the awkward ones."""
        if kind == INTEGER:
            if self.state.random() < REPEATED_RATE:
                return integer_column("v", self.state.integers(0, max(rows // 8, 2), rows))
            return integer_column("v", self.state.integers(-1000, 1000, rows))
        if kind == FLOATING:
            values = self.state.normal(0, 100, rows)
            if rows and self.state.random() < 0.3:
                values[self.state.integers(0, rows)] = 0.0
            return floating_column("v", values)
        if kind == BOOLEAN:
            return boolean_column("v", self.state.random(rows) < 0.5)
        entries = max(rows // 6, 2)
        return string_column("v", [f"s{one}" for one in self.state.integers(0, entries, rows)])

    def batch(self, columns: int = 3, rows: int | None = None) -> Batch:
        """A whole table, with column names that are stable across runs."""
        count = rows if rows is not None else self.rows()
        made = []
        for one in range(columns):
            built = self.column(count)
            made.append(
                Column(
                    field=Field(
                        name=f"c{one}",
                        logical=built.field.logical,
                        nullable=built.valid is not None,
                    ),
                    values=built.values,
                    valid=built.valid,
                    dictionary=built.dictionary,
                )
            )
        return Batch.from_columns(made)

    def predicate(self, batch: Batch, depth: int = 0) -> Expr:
        """A random predicate over a table's columns.

        Weighted towards leaves as the depth grows, so the expected size is small and the tail
        is long. A generator that split evenly at every level would produce predicates of size
        two to the depth and almost never produce a bare comparison, which is the shape most
        real predicates have.
        """
        if depth >= MAX_DEPTH or self.state.random() < 0.4 + 0.2 * depth:
            return self._leaf(batch)
        choice = self.state.random()
        if choice < 0.4:
            return And(
                parts=(self.predicate(batch, depth + 1), self.predicate(batch, depth + 1))
            )
        if choice < 0.8:
            return Or(
                parts=(self.predicate(batch, depth + 1), self.predicate(batch, depth + 1))
            )
        return Not(part=self.predicate(batch, depth + 1))

    def _leaf(self, batch: Batch) -> Expr:
        """One comparison, null test or membership over a random column.

        The literal is drawn from the column's own values half the time, which is what makes an
        equality ever match. A literal drawn from the type's whole range would make every
        equality false and the fuzzer would only ever test the empty result.
        """
        name = str(self.state.choice(list(batch.schema.names)))
        one = batch.column(name)
        choice = self.state.random()
        if choice < 0.15:
            return IsNull(part=column(name), negated=bool(self.state.random() < 0.5))
        if choice < 0.3 and len(one):
            values = tuple(self._sample(one, 3))
            return InList(part=column(name), options=values)
        operator = str(self.state.choice(["=", "!=", "<", "<=", ">", ">="]))
        if one.field.logical == STRING:
            operator = str(self.state.choice(["=", "!="]))
        return Compare(operator, column(name), literal(self._one_value(one)))

    def _sample(self, one: Column, count: int) -> list:
        """A few of a column's own values, so a membership test can match."""
        present = [value for value in one.to_list() if value is not None]
        if not present:
            return [0]
        positions = self.state.integers(0, len(present), min(count, len(present)))
        return [present[int(position)] for position in positions]

    def _one_value(self, one: Column):
        """A literal to compare against, from the column half the time.

        The boundary matters: comparing against a value the column actually holds is what
        exercises the difference between less than and less than or equal, and a value drawn at
        random from the range almost never lands on one.
        """
        present = [value for value in one.to_list() if value is not None]
        if present and self.state.random() < 0.5:
            return present[int(self.state.integers(0, len(present)))]
        if one.field.logical == INTEGER:
            return int(self.state.integers(-1000, 1000))
        if one.field.logical == FLOATING:
            return float(self.state.normal(0, 100))
        if one.field.logical == BOOLEAN:
            return bool(self.state.random() < 0.5)
        return f"s{int(self.state.integers(0, 20))}"


@dataclass(frozen=True)
class Case:
    """One generated case: a table and a predicate over it."""

    batch: Batch
    predicate: Expr

    @property
    def size(self) -> int:
        """How large the case is, as rows times predicate nodes.

        One number so that the shrinker has something to minimise. Rows and nodes are not really
        comparable, and a shrinker needs a total order, so a product is used and the two are
        reported separately as well.
        """
        return self.batch.rows * _nodes(self.predicate)

    def describe(self) -> str:
        """The case as one line."""
        return f"{self.batch.rows} rows, {self.batch.width} columns, {describe(self.predicate)}"

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rows": self.batch.rows,
            "columns": self.batch.width,
            "nodes": _nodes(self.predicate),
            "predicate": describe(self.predicate),
        }


def _nodes(one: Expr) -> int:
    """How many nodes a predicate has, for measuring the shrinker."""
    if isinstance(one, (And, Or)):
        return 1 + sum(_nodes(part) for part in one.parts)
    if isinstance(one, Not):
        return 1 + _nodes(one.part)
    if isinstance(one, (Compare,)):
        return 1 + _nodes(one.left) + _nodes(one.right)
    if isinstance(one, (IsNull,)):
        return 1 + _nodes(one.part)
    return 1


def cases(count: int = 50, seed: int = 0, columns: int = 3) -> list[Case]:
    """A batch of generated cases."""
    if count <= 0:
        raise ConfigError(f"{count} is not a case count")
    maker = Generator(seed=seed)
    out = []
    for _ in range(count):
        batch = maker.batch(columns=columns)
        out.append(Case(batch=batch, predicate=maker.predicate(batch)))
    return out


def shrink(case: Case, fails, limit: int = 200) -> Case:
    """The smallest failing case reachable by removing rows and predicate parts.

    Greedy rather than exhaustive. At each step it tries a handful of smaller cases and keeps
    the first that still fails, and stops when nothing smaller fails. That is not a minimum and
    it is close enough to read, which is the only thing a shrunk case is for.

    The predicate is shrunk before the rows, because a predicate with one comparison in it and
    four hundred rows is far easier to read than a six node predicate over three rows.
    """
    if not fails(case):
        raise ConfigError("the case does not fail, so there is nothing to shrink")
    current = case
    for _ in range(limit):
        smaller = _smaller(current)
        for one in smaller:
            if fails(one):
                current = one
                break
        else:
            return current
    return current


def _smaller(case: Case) -> list[Case]:
    """Candidate smaller cases, predicate first and then rows."""
    out = []
    for one in _simpler_predicates(case.predicate):
        out.append(Case(batch=case.batch, predicate=one))
    rows = case.batch.rows
    if rows > 1:
        out.append(Case(batch=case.batch.slice(0, rows // 2), predicate=case.predicate))
        out.append(Case(batch=case.batch.slice(rows // 2, rows), predicate=case.predicate))
        if rows > 2:
            out.append(Case(batch=case.batch.slice(0, rows - 1), predicate=case.predicate))
    return out


def _simpler_predicates(one: Expr) -> list[Expr]:
    """Each way of making a predicate smaller by one step."""
    if isinstance(one, (And, Or)):
        return [*list(one.parts), *_children_replaced(one)]
    if isinstance(one, Not):
        return [one.part]
    return []


def _children_replaced(one: Expr) -> list[Expr]:
    """The same conjunction with one part simplified, for the deeper cases."""
    out = []
    for index, part in enumerate(one.parts):
        for simpler in _simpler_predicates(part):
            parts = list(one.parts)
            parts[index] = simpler
            out.append(type(one)(parts=tuple(parts)))
    return out


@dataclass
class Failure:
    """One case that failed, before and after shrinking."""

    original: Case
    shrunk: Case
    message: str

    @property
    def reduction(self) -> float:
        """How much smaller the shrunk case is, as a share removed."""
        if self.original.size == 0:
            return 0.0
        return 1 - self.shrunk.size / self.original.size

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "original": self.original.as_dict(),
            "shrunk": self.shrunk.as_dict(),
            "reduction": round(self.reduction, 3),
            "message": self.message,
        }


def search(check, count: int = 50, seed: int = 0, columns: int = 3) -> list[Failure]:
    """Generate cases, keep the ones that fail, and shrink each.

    The check returns true when the case is fine, which is the convention that makes a passing
    run report nothing. A check that raises is a failure too, with the exception as the message,
    because an operator that crashes on generated input is exactly what this is looking for.
    """
    out = []
    for one in cases(count=count, seed=seed, columns=columns):
        message = _failure_message(check, one)
        if message:
            out.append(
                Failure(
                    original=one,
                    shrunk=shrink(one, lambda case: bool(_failure_message(check, case))),
                    message=message,
                )
            )
    return out


def _failure_message(check, case: Case) -> str:
    """Empty when the case passes, and why when it does not."""
    try:
        return "" if check(case) else "the check returned false"
    except QueryEngineError as problem:
        return f"{type(problem).__name__}: {problem}"


def the_generator_is_reproducible(count: int = 20) -> dict:
    """The same seed gives the same cases, which is what makes a failure worth reporting.

    A fuzzer that cannot reproduce its own findings is a random crash reporter. Checked on the
    rendered predicates and the row counts rather than on object identity.
    """
    first = cases(count=count, seed=7)
    second = cases(count=count, seed=7)
    third = cases(count=count, seed=8)
    return {
        "cases": count,
        "same_seed_matches": [one.describe() for one in first]
        == [one.describe() for one in second],
        "different_seed_differs": [one.describe() for one in first]
        != [one.describe() for one in third],
    }


def the_generator_makes_nulls(count: int = 200) -> dict:
    """How many generated columns have nulls in them, and how many values are null.

    Both are checked because a generator that puts one null in one column of two hundred is not
    testing the null paths. The target is set by NULL_COLUMN_RATE and this is what came out.
    """
    made = cases(count=count, seed=11, columns=3)
    columns = [one for case in made for one in case.batch.columns]
    with_nulls = [one for one in columns if one.valid is not None]
    nulls = sum(int((~one.valid).sum()) for one in with_nulls)
    inside = sum(len(one) for one in with_nulls)
    everywhere = sum(len(one) for one in columns)
    return {
        "columns": len(columns),
        "with_nulls": len(with_nulls),
        "column_rate": round(len(with_nulls) / len(columns), 3),
        "column_target": NULL_COLUMN_RATE,
        "value_rate_in_those_columns": round(nulls / max(inside, 1), 3),
        "value_target": NULL_VALUE_RATE,
        "value_rate_overall": round(nulls / max(everywhere, 1), 3),
        "the_column_rate_is_near_its_target": abs(
            len(with_nulls) / len(columns) - NULL_COLUMN_RATE
        )
        < 0.1,
        "the_value_rate_is_near_its_target": abs(
            nulls / max(inside, 1) - NULL_VALUE_RATE
        )
        < 0.05,
    }


def the_generator_makes_predicates_that_match(count: int = 200) -> dict:
    """What share of generated predicates keep something and reject something.

    The measurement that says whether the fuzzer is testing anything. A predicate that keeps
    every row exercises no branch, and one that keeps none exercises the empty path over and
    over. The interesting ones are in between, and drawing literals from the column's own values
    is what puts them there.
    """
    kept = 0
    partial = 0
    empty = 0
    everything = 0
    for one in cases(count=count, seed=13):
        try:
            rows = apply_predicate(one.predicate, one.batch).rows
        except QueryEngineError:
            continue
        kept += 1
        if rows == 0:
            empty += 1
        elif rows == one.batch.rows:
            everything += 1
        else:
            partial += 1
    return {
        "evaluated": kept,
        "kept_some_rows": partial,
        "kept_none": empty,
        "kept_all": everything,
        "partial_share": round(partial / max(kept, 1), 3),
        "most_are_partial": partial > max(empty, everything),
    }


def the_generator_makes_every_type(count: int = 300) -> dict:
    """Every logical type appears among the generated columns."""
    made = cases(count=count, seed=17, columns=3)
    types = {one.field.logical for case in made for one in case.batch.columns}
    return {
        "types": sorted(types),
        "count": len(types),
        "it_made_all_four": len(types) == 4,
    }


def the_predicates_are_mostly_small(count: int = 300) -> dict:
    """The distribution of predicate sizes, which should be small with a long tail.

    A generator that split evenly at every level would make every predicate the maximum size and
    would never test a bare comparison, which is the shape most real predicates have.
    """
    sizes = [_nodes(one.predicate) for one in cases(count=count, seed=19)]
    return {
        "cases": count,
        "smallest": min(sizes),
        "largest": max(sizes),
        "median": int(np.median(sizes)),
        "mean": round(float(np.mean(sizes)), 2),
        "the_median_is_small": int(np.median(sizes)) <= 5,
        "and_the_tail_is_long": max(sizes) > int(np.median(sizes)) * 2,
    }


def shrinking_removes_most_of_a_case() -> dict:
    """A case that fails on one row, shrunk from a large one.

    The failing condition is artificial and is the shape a real one has: something is wrong with
    one particular value, and the case that found it also holds two hundred rows that are fine.
    """
    maker = Generator(seed=23)
    batch = maker.batch(columns=3, rows=200)
    predicate = And(
        parts=(
            Compare(">", column("c0"), literal(-100000)),
            Or(
                parts=(
                    Compare("<", column("c0"), literal(100000)),
                    Compare("=", column("c0"), literal(0)),
                )
            ),
        )
    )
    case = Case(batch=batch, predicate=predicate)
    smaller = shrink(case, lambda one: one.batch.rows >= 1)
    return {
        "original": case.as_dict(),
        "shrunk": smaller.as_dict(),
        "rows_removed": case.batch.rows - smaller.batch.rows,
        "nodes_removed": _nodes(case.predicate) - _nodes(smaller.predicate),
        "reduction": round(1 - smaller.size / case.size, 3),
        "it_removed_most_of_it": smaller.size < case.size * 0.1,
    }


def shrinking_keeps_the_failure() -> dict:
    """The shrunk case still fails, which is the only property a shrinker must have.

    A shrinker that produced a smaller passing case would be worse than none at all, because the
    report would point at code that works.
    """
    maker = Generator(seed=29)
    batch = maker.batch(columns=3, rows=100)
    predicate = And(
        parts=(
            Compare(">", column("c0"), literal(-100000)),
            Compare("<", column("c0"), literal(100000)),
        )
    )
    case = Case(batch=batch, predicate=predicate)

    def fails(one: Case) -> bool:
        return one.batch.rows >= 3

    smaller = shrink(case, fails)
    return {
        "original_rows": case.batch.rows,
        "shrunk_rows": smaller.batch.rows,
        "it_still_fails": fails(smaller),
        "and_it_is_at_the_boundary": smaller.batch.rows <= 4,
    }


def shrinking_a_passing_case_is_refused() -> bool:
    """Shrinking something that does not fail, which is a caller error."""
    maker = Generator(seed=31)
    batch = maker.batch(columns=2, rows=10)
    case = Case(batch=batch, predicate=Compare(">", column("c0"), literal(0)))
    try:
        shrink(case, _never)
    except ConfigError:
        return True
    return False


def a_search_over_a_sound_check_finds_nothing(count: int = 100) -> dict:
    """The fuzzer against a check that is always true, which must report nothing.

    The measurement that says the fuzzer does not invent failures. Every reported failure costs
    somebody an investigation, and a fuzzer with a false positive rate above zero gets turned
    off.
    """
    found = search(_always, count=count, seed=37)
    return {
        "cases": count,
        "failures": len(found),
        "it_found_nothing": not found,
    }


def a_search_over_a_broken_check_finds_it(count: int = 100) -> dict:
    """And against a check that fails on tables above ten rows, which it must find.

    Both halves are needed. A fuzzer that never reports anything passes the first measurement
    perfectly.
    """
    found = search(lambda one: one.batch.rows <= 10, count=count, seed=41)
    return {
        "cases": count,
        "failures": len(found),
        "it_found_some": bool(found),
        "every_one_was_shrunk": all(one.shrunk.size <= one.original.size for one in found),
        "average_reduction": round(sum(one.reduction for one in found) / max(len(found), 1), 3),
    }


def a_search_reports_an_exception_as_a_failure() -> dict:
    """A check that raises is a failure with the exception as its message."""

    def broken(one: Case) -> bool:
        raise ConfigError(f"this always breaks, on {one.batch.rows} rows")

    found = search(broken, count=5, seed=43)
    return {
        "failures": len(found),
        "it_caught_them": len(found) == 5,
        "the_message_names_the_error": "ConfigError" in found[0].message,
    }


def _always(one: Case) -> bool:
    """A check that never reports a failure, for the false positive measurement."""
    return bool(one.batch.rows >= 0)


def _never(one: Case) -> bool:
    """A check that reports every case as passing, for the shrinker refusal."""
    return not one.batch.rows >= 0


def a_zero_case_count_is_refused() -> bool:
    """Generating no cases at all."""
    try:
        cases(count=0)
    except ConfigError:
        return True
    return False


def compare_the_generators(count: int = 200) -> list[dict]:
    """What the generator produces, as one table."""
    made = cases(count=count, seed=47)
    types = {}
    for case in made:
        for one in case.batch.columns:
            types[one.field.logical] = types.get(one.field.logical, 0) + 1
    return [
        {"type": name, "columns": total, "share": round(total / sum(types.values()), 3)}
        for name, total in sorted(types.items())
    ]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "max_depth": MAX_DEPTH,
        "null_column_rate": NULL_COLUMN_RATE,
        "reproducible": the_generator_is_reproducible()["same_seed_matches"],
        "partial_share": the_generator_makes_predicates_that_match()["partial_share"],
        "no_false_positives": a_search_over_a_sound_check_finds_nothing()["it_found_nothing"],
        "finds_a_real_one": a_search_over_a_broken_check_finds_it()["it_found_some"],
    }

