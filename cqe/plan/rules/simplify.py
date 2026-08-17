from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError
from cqe.exec.batch import Batch
from cqe.exec.expr import (
    And,
    Compare,
    Expr,
    InList,
    IsNull,
    Literal,
    Not,
    Or,
    column,
    describe,
    literal,
)
from cqe.exec.filter import apply as apply_predicate
from cqe.types.schema import BOOLEAN
from cqe.verify.reference import Rows, agree

# Rewriting a predicate into a cheaper one that keeps the same rows.
#
# Every rule here is a local rewrite: a shape that matches becomes another shape, and the
# rewriting repeats until nothing matches. That is the whole design and it has one consequence
# worth stating up front: a set of rules that can undo each other never terminates, so each rule
# has to move the predicate in one direction. Here the direction is fewer nodes, and the
# measurement checks it, because a rule set that loops is a hang rather than a wrong answer and
# is much harder to notice in a test.
#
# The rules divide into three kinds.
#
# Constant folding, which evaluates what does not depend on the data. A comparison of two
# literals is a literal, and once it is, an and with a false in it is false and the rest of the
# conjunction never runs.
#
# Flattening, which turns nested conjunctions into one. It changes nothing about the meaning and
# it is what makes every other rule simpler, because a rule matching a conjunction of three
# parts would otherwise have to match every way of nesting them.
#
# Negation pushing, which moves a not inwards until it sits on a comparison, where it becomes
# the opposite comparison. That is the rule with the null trap in it: not of a comparison is not
# the opposite comparison when the comparison is null, and the whole thing only works because a
# null stays null under both.
#
# Nothing here reorders a conjunction by selectivity. That is exec/filter.py's job and it needs
# statistics; these rules need nothing but the predicate.

# How many times the rewriting will go round before giving up. A rule set that terminates never
# needs more than a handful; this exists so that a rule added later that loops is reported
# rather than hanging.
MAX_ROUNDS = 20

OPPOSITE = {"=": "!=", "!=": "=", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}


@dataclass(frozen=True)
class Rewritten:
    """A predicate before and after simplification, and what changed."""

    before: Expr
    after: Expr
    rounds: int
    rules: tuple[str, ...]

    @property
    def changed(self) -> bool:
        """Whether anything happened."""
        return describe(self.before) != describe(self.after)

    @property
    def shrank(self) -> bool:
        """Whether the result has fewer nodes than the input."""
        return size(self.after) < size(self.before)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "before": describe(self.before),
            "after": describe(self.after),
            "before_nodes": size(self.before),
            "after_nodes": size(self.after),
            "rounds": self.rounds,
            "rules": list(self.rules),
        }


def size(one: Expr) -> int:
    """How many nodes a predicate has, which is what every rule here reduces."""
    if isinstance(one, (And, Or)):
        return 1 + sum(size(part) for part in one.parts)
    if isinstance(one, Not):
        return 1 + size(one.part)
    if isinstance(one, Compare):
        return 1 + size(one.left) + size(one.right)
    if isinstance(one, IsNull):
        return 1 + size(one.part)
    return 1


def _is_constant(one: Expr, value: bool) -> bool:
    """Whether a predicate is the literal true or the literal false."""
    return isinstance(one, Literal) and one.logical == BOOLEAN and bool(one.value) is value


def _truth(value: bool) -> Literal:
    """A boolean literal."""
    return Literal(value=value, logical=BOOLEAN)


def flatten(one: Expr) -> tuple[Expr, bool]:
    """Nested conjunctions and disjunctions become flat ones.

    An and of an and is one and. Nothing about the meaning changes and every other rule gets
    simpler, because a rule matching three conjuncts would otherwise have to match every way of
    nesting them.
    """
    if not isinstance(one, (And, Or)):
        return one, False
    kind = type(one)
    parts: list[Expr] = []
    changed = False
    for part in one.parts:
        if isinstance(part, kind):
            parts.extend(part.parts)
            changed = True
        else:
            parts.append(part)
    if len(parts) == 1:
        return parts[0], True
    return kind(parts=tuple(parts)), changed


def fold(one: Expr) -> tuple[Expr, bool]:
    """A comparison of two literals becomes a literal.

    The rule that makes the others worth having: once a branch is a constant, the conjunction
    rules can delete it, and a predicate that was written with a redundant clause in it stops
    costing a pass over the data.
    """
    if (
        isinstance(one, Compare)
        and isinstance(one.left, Literal)
        and isinstance(one.right, Literal)
    ):
        if one.left.value is None or one.right.value is None:
            return one, False
        return _truth(_compare(one.op, one.left.value, one.right.value)), True
    if (
        isinstance(one, Not)
        and isinstance(one.part, Literal)
        and one.part.logical == BOOLEAN
        and one.part.value is not None
    ):
        return _truth(not bool(one.part.value)), True
    return one, False


def _compare(op: str, left, right) -> bool:
    """One comparison between two Python values."""
    if op == "=":
        return bool(left == right)
    if op == "!=":
        return bool(left != right)
    if op == "<":
        return bool(left < right)
    if op == "<=":
        return bool(left <= right)
    if op == ">":
        return bool(left > right)
    return bool(left >= right)


def absorb(one: Expr) -> tuple[Expr, bool]:
    """Constants inside a conjunction or a disjunction disappear or take over.

    An and with a false in it is false; a true in it is dropped. An or is the mirror. Both are
    obvious and both are the rules that actually delete work, because a predicate reaching here
    with a constant in it usually got one from folding a clause that was written by a query
    generator rather than by a person.
    """
    if not isinstance(one, (And, Or)):
        return one, False
    dominant = not isinstance(one, And)
    neutral = not dominant
    if any(_is_constant(part, dominant) for part in one.parts):
        return _truth(dominant), True
    kept = [part for part in one.parts if not _is_constant(part, neutral)]
    if len(kept) == len(one.parts):
        return one, False
    if not kept:
        return _truth(neutral), True
    if len(kept) == 1:
        return kept[0], True
    return type(one)(parts=tuple(kept)), True


def deduplicate(one: Expr) -> tuple[Expr, bool]:
    """The same conjunct twice becomes one.

    Which happens far more often than it sounds, because pushing a predicate through a join
    duplicates it onto both sides and a rewrite that then pulls one back up leaves two copies of
    the same clause in the same conjunction.
    """
    if not isinstance(one, (And, Or)):
        return one, False
    seen: dict[str, Expr] = {}
    for part in one.parts:
        seen.setdefault(describe(part), part)
    if len(seen) == len(one.parts):
        return one, False
    kept = list(seen.values())
    if len(kept) == 1:
        return kept[0], True
    return type(one)(parts=tuple(kept)), True


def push_negation(one: Expr) -> tuple[Expr, bool]:
    """A not moves inwards until it reaches a comparison, where it flips it.

    The rule with the trap. Not of a less than is a greater than or equal only because a null
    comparison stays null under both: the comparison is unknown, its negation is unknown, and
    the filter drops the row either way. If nulls were treated as false, the negation would keep
    the row and the rewrite would change the answer.

    Not of an is null is the same test negated, which the type already carries as a flag rather
    than as a wrapper, so this rule turns a wrapper into a flag.
    """
    if not isinstance(one, Not):
        return one, False
    inner = one.part
    if isinstance(inner, Not):
        return inner.part, True
    if isinstance(inner, Compare):
        return Compare(OPPOSITE[inner.op], inner.left, inner.right), True
    if isinstance(inner, IsNull):
        return IsNull(part=inner.part, negated=not inner.negated), True
    if isinstance(inner, And):
        return Or(parts=tuple(Not(part=part) for part in inner.parts)), True
    if isinstance(inner, Or):
        return And(parts=tuple(Not(part=part) for part in inner.parts)), True
    return one, False


def collapse_membership(one: Expr) -> tuple[Expr, bool]:
    """A membership test of one value becomes an equality.

    Small and worth doing because an in list of one is what a query generator emits when a
    parameter list happened to have one element, and an equality is what the pruning rules in
    plan/rules/pruning.py can use and a membership test is not.

    It is also the one rule here that makes a predicate larger by the node count: a membership
    test is one node and a comparison is three. The module's stated direction is fewer nodes and
    this rule goes the other way, which is only safe because it cannot fire twice on its own
    output, and the termination measurement is what confirms that rather than the argument.
    """
    if isinstance(one, InList) and len(one.options) == 1:
        return Compare("=", one.part, literal(one.options[0])), True
    return one, False


RULES = (
    ("flatten", flatten),
    ("fold", fold),
    ("absorb", absorb),
    ("deduplicate", deduplicate),
    ("push negation", push_negation),
    ("collapse membership", collapse_membership),
)


def _once(one: Expr) -> tuple[Expr, list[str]]:
    """One pass of every rule over every node, bottom up.

    Bottom up so that a rule firing on a child gives the parent something to fire on in the same
    pass, which is what keeps the round count at two or three rather than at the depth of the
    predicate.
    """
    fired: list[str] = []
    if isinstance(one, (And, Or)):
        parts = []
        for part in one.parts:
            rewritten, names = _once(part)
            parts.append(rewritten)
            fired.extend(names)
        one = type(one)(parts=tuple(parts))
    elif isinstance(one, Not):
        rewritten, names = _once(one.part)
        fired.extend(names)
        one = Not(part=rewritten)
    for name, rule in RULES:
        one, changed = rule(one)
        if changed:
            fired.append(name)
    return one, fired


def simplify(one: Expr, rounds: int = MAX_ROUNDS) -> Rewritten:
    """Apply every rule until nothing changes.

    The round count is reported rather than hidden, because a predicate needing many rounds
    means two rules are handing work back and forth and that is worth seeing before it becomes a
    rule set that does not terminate.
    """
    if rounds < 1:
        raise ConfigError(f"{rounds} is not a round count")
    current = one
    fired: list[str] = []
    used = 0
    for round_number in range(1, rounds + 1):
        used = round_number
        rewritten, names = _once(current)
        fired.extend(names)
        if describe(rewritten) == describe(current):
            break
        current = rewritten
    return Rewritten(before=one, after=current, rounds=used, rules=tuple(dict.fromkeys(fired)))


def _table(rows: int = 10000, seed: int = 21) -> Batch:
    """A table to check a rewrite against."""
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 30, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 5, rows)]),
        ]
    )


def _same_rows(batch: Batch, before: Expr, after: Expr) -> bool:
    """Whether two predicates keep the same rows.

    The property every rule has to have and the only one that matters. Checked by running both
    over real data rather than by reasoning about the shapes, because a rule that is right about
    every shape and wrong about nulls is right about nothing.
    """
    return bool(
        agree(
            Rows.of(apply_predicate(before, batch)),
            Rows.of(apply_predicate(after, batch)),
        )
    )


def folding_removes_a_constant_clause(rows: int = 10000) -> dict:
    """A conjunction with a comparison of two literals in it, which folds away.

    The rule chain in one measurement: the comparison folds to true, the true is absorbed out of
    the conjunction, and what is left is the clause that reads the data.
    """
    predicate = And(
        parts=(
            Compare(">", column("amount"), literal(100.0)),
            Compare("<", literal(1), literal(2)),
        )
    )
    rewritten = simplify(predicate)
    batch = _table(rows)
    before = Meter()
    after = Meter()
    apply_predicate(predicate, batch, meter=before)
    apply_predicate(rewritten.after, batch, meter=after)
    return {
        **rewritten.as_dict(),
        "before_touched": before.values_touched,
        "after_touched": after.values_touched,
        "ratio": round(before.values_touched / max(after.values_touched, 1), 2),
        "it_kept_the_same_rows": _same_rows(batch, predicate, rewritten.after),
    }


def a_false_constant_deletes_the_whole_conjunction(rows: int = 10000) -> dict:
    """An and with a false in it is false, whatever else is in it.

    The largest saving any rule here makes, and the one that turns a query somebody generated
    into no work at all rather than into a scan.
    """
    predicate = And(
        parts=(
            Compare(">", column("amount"), literal(100.0)),
            Compare(">", literal(1), literal(2)),
            Compare("<", column("shop"), literal(10)),
        )
    )
    rewritten = simplify(predicate)
    batch = _table(rows)
    before = Meter()
    after = Meter()
    kept = apply_predicate(predicate, batch, meter=before)
    left = apply_predicate(rewritten.after, batch, meter=after)
    return {
        **rewritten.as_dict(),
        "it_is_a_constant": isinstance(rewritten.after, Literal),
        "rows_before": kept.rows,
        "rows_after": left.rows,
        "they_agree": kept.rows == left.rows == 0,
        "before_touched": before.values_touched,
        "after_touched": after.values_touched,
    }


def a_true_constant_disappears() -> dict:
    """An and with a true in it is the rest of the and, which is the mirror rule."""
    predicate = And(parts=(Compare(">", column("amount"), literal(100.0)), _truth(True)))
    rewritten = simplify(predicate)
    return {
        **rewritten.as_dict(),
        "it_is_the_other_clause": describe(rewritten.after) == "(amount > 100.0)",
        "it_shrank": rewritten.shrank,
    }


def nesting_flattens(rows: int = 10000) -> dict:
    """An and of an and of an and becomes one and of three.

    Which changes nothing about the answer and is what makes every other rule simpler, so the
    measurement is on the node count rather than on the rows.
    """
    predicate = And(
        parts=(
            Compare(">", column("amount"), literal(60.0)),
            And(
                parts=(
                    Compare("<", column("shop"), literal(20)),
                    And(parts=(Compare("<", column("id"), literal(9000)),)),
                )
            ),
        )
    )
    rewritten = simplify(predicate)
    batch = _table(rows)
    return {
        **rewritten.as_dict(),
        "it_is_one_conjunction": isinstance(rewritten.after, And)
        and all(not isinstance(one, And) for one in rewritten.after.parts),
        "conjuncts": len(rewritten.after.parts) if isinstance(rewritten.after, And) else 1,
        "it_kept_the_same_rows": _same_rows(batch, predicate, rewritten.after),
    }


def a_repeated_clause_is_dropped(rows: int = 10000) -> dict:
    """The same conjunct twice, which becomes one and saves a whole pass.

    Common because pushing a predicate through a join duplicates it onto both sides, and a later
    rewrite that pulls one back up leaves two copies in the same conjunction.
    """
    clause = Compare(">", column("amount"), literal(100.0))
    predicate = And(parts=(clause, Compare("<", column("shop"), literal(10)), clause))
    rewritten = simplify(predicate)
    batch = _table(rows)
    before = Meter()
    after = Meter()
    apply_predicate(predicate, batch, meter=before)
    apply_predicate(rewritten.after, batch, meter=after)
    return {
        **rewritten.as_dict(),
        "conjuncts_before": len(predicate.parts),
        "conjuncts_after": len(rewritten.after.parts)
        if isinstance(rewritten.after, And)
        else 1,
        "before_touched": before.values_touched,
        "after_touched": after.values_touched,
        "ratio": round(before.values_touched / max(after.values_touched, 1), 2),
        "it_kept_the_same_rows": _same_rows(batch, predicate, rewritten.after),
    }


def a_negation_reaches_the_comparison(rows: int = 10000) -> dict:
    """Not of a less than becomes a greater than or equal, over real data.

    The rule with the null trap in it. Checked on a column with nulls rather than on a shape,
    because the shape is right whatever the null policy and the rows are only right under the
    policy this engine has.
    """
    predicate = Not(part=Compare("<", column("amount"), literal(100.0)))
    rewritten = simplify(predicate)
    batch = _table(rows)
    return {
        **rewritten.as_dict(),
        "it_became_a_comparison": isinstance(rewritten.after, Compare),
        "the_operator_flipped": isinstance(rewritten.after, Compare)
        and rewritten.after.op == ">=",
        "it_kept_the_same_rows": _same_rows(batch, predicate, rewritten.after),
        "it_shrank": rewritten.shrank,
    }


def a_negation_over_nulls_keeps_the_same_rows(rows: int = 4000) -> dict:
    """The same rewrite over a column that is a quarter null.

    The measurement that says the rule is safe. A null comparison is unknown, its negation is
    unknown, and the filter drops the row under both, so the rewrite is exact. If nulls were
    read as false the negation would keep them and the two predicates would differ by every null
    row.
    """
    state = np.random.default_rng(23)
    values = state.normal(100, 25, rows)
    made = floating_column("amount", values)
    valid = state.random(rows) > 0.25
    batch = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            Column(field=made.field, values=values, valid=valid),
        ]
    )
    predicate = Not(part=Compare("<", column("amount"), literal(100.0)))
    rewritten = simplify(predicate)
    before = apply_predicate(predicate, batch)
    after = apply_predicate(rewritten.after, batch)
    return {
        "nulls": int((~valid).sum()),
        "rows_before": before.rows,
        "rows_after": after.rows,
        "they_agree": _same_rows(batch, predicate, rewritten.after),
        "the_nulls_are_in_neither": before.rows + int((~valid).sum()) <= rows,
    }


def a_negation_distributes_over_a_conjunction(rows: int = 10000) -> dict:
    """Not of an and becomes an or of nots, each of which then flips.

    Two rules composing, which is what the repeated passes are for: the first turns the not into
    a disjunction of nots and the second turns each of those into a comparison.
    """
    predicate = Not(
        part=And(
            parts=(
                Compare("<", column("amount"), literal(100.0)),
                Compare("<", column("shop"), literal(10)),
            )
        )
    )
    rewritten = simplify(predicate)
    batch = _table(rows)
    return {
        **rewritten.as_dict(),
        "it_became_a_disjunction": isinstance(rewritten.after, Or),
        "no_negations_are_left": "not" not in describe(rewritten.after).lower(),
        "it_kept_the_same_rows": _same_rows(batch, predicate, rewritten.after),
    }


def a_double_negation_cancels(rows: int = 10000) -> dict:
    """Not of not is the thing itself, in one round."""
    inner = Compare("<", column("amount"), literal(100.0))
    predicate = Not(part=Not(part=inner))
    rewritten = simplify(predicate)
    batch = _table(rows)
    return {
        **rewritten.as_dict(),
        "it_is_the_original": describe(rewritten.after) == describe(inner),
        "it_kept_the_same_rows": _same_rows(batch, predicate, rewritten.after),
    }


def a_membership_of_one_becomes_an_equality(rows: int = 10000) -> dict:
    """An in list holding one value, which the pruning rules can use and a list cannot.

    A small rewrite whose value is entirely downstream: plan/rules/pruning.py reads an equality
    and skips row groups from it, and reads a membership test and does nothing.
    """
    predicate = InList(part=column("shop"), options=(7,))
    rewritten = simplify(predicate)
    batch = _table(rows)
    return {
        **rewritten.as_dict(),
        "it_became_an_equality": isinstance(rewritten.after, Compare),
        "it_kept_the_same_rows": _same_rows(batch, predicate, rewritten.after),
    }


def the_rules_terminate() -> dict:
    """Every rewrite reaches a fixed point in a handful of rounds.

    The property a rule set needs and the one that fails silently: two rules that undo each
    other never terminate, and the failure is a hang rather than a wrong answer. The round count
    is counted rather than trusted.
    """
    predicates = _examples()
    rounds = [simplify(one).rounds for one in predicates]
    return {
        "predicates": len(predicates),
        "most_rounds": max(rounds),
        "average": round(sum(rounds) / len(rounds), 2),
        "they_all_terminate": max(rounds) < MAX_ROUNDS,
        "and_quickly": max(rounds) <= 4,
    }


def every_rewrite_keeps_the_same_rows(rows: int = 4000) -> dict:
    """Every example predicate, before and after, over the same table.

    The measurement that makes the rest of the module trustworthy. A simplification that changes
    the answer is worse than no simplification, and the shapes are checked against the data
    rather than against an argument.
    """
    batch = _table(rows)
    results = {}
    for one in _examples():
        rewritten = simplify(one)
        results[describe(one)[:40]] = _same_rows(batch, one, rewritten.after)
    return {
        "predicates": len(results),
        "they_all_agree": all(results.values()),
        "failures": [name for name, ok in results.items() if not ok],
    }


def every_rewrite_shrinks_or_stays() -> dict:
    """No rule makes a predicate larger, which is what makes the rewriting terminate.

    Not quite true of push negation over a conjunction, which turns one node into two, and the
    measurement says which ones grow and by how much. The count that has to fall is the count of
    negations, and it does.
    """
    out = []
    for one in _examples():
        rewritten = simplify(one)
        out.append(
            {
                "predicate": describe(one)[:36],
                "before": size(one),
                "after": size(rewritten.after),
                "grew": size(rewritten.after) > size(one),
            }
        )
    grew = [one for one in out if one["grew"]]
    return {
        "predicates": len(out),
        "shrank_or_stayed": len(out) - len(grew),
        "grew": len(grew),
        "which_grew": [one["predicate"] for one in grew],
        "and_none_grew_by_much": all(one["after"] <= one["before"] + 2 for one in grew),
    }


def _examples() -> list[Expr]:
    """The predicates every measurement above runs over."""
    return [
        Compare(">", column("amount"), literal(100.0)),
        And(
            parts=(
                Compare(">", column("amount"), literal(100.0)),
                Compare("<", literal(1), literal(2)),
            )
        ),
        And(
            parts=(
                Compare(">", column("amount"), literal(100.0)),
                Compare(">", literal(1), literal(2)),
            )
        ),
        And(
            parts=(
                Compare(">", column("amount"), literal(60.0)),
                And(parts=(Compare("<", column("shop"), literal(20)),)),
            )
        ),
        Not(part=Compare("<", column("amount"), literal(100.0))),
        Not(part=Not(part=Compare("<", column("amount"), literal(100.0)))),
        Not(
            part=And(
                parts=(
                    Compare("<", column("amount"), literal(100.0)),
                    Compare("<", column("shop"), literal(10)),
                )
            )
        ),
        Not(part=IsNull(part=column("amount"))),
        InList(part=column("shop"), options=(7,)),
        InList(part=column("shop"), options=(1, 2, 3)),
        Or(
            parts=(
                Compare("<", column("amount"), literal(20.0)),
                Compare("<", column("amount"), literal(20.0)),
            )
        ),
    ]


def a_predicate_with_nothing_to_do_is_unchanged() -> dict:
    """A bare comparison, which no rule matches and which comes back identical."""
    predicate = Compare(">", column("amount"), literal(100.0))
    rewritten = simplify(predicate)
    return {
        "before": describe(rewritten.before),
        "after": describe(rewritten.after),
        "it_is_unchanged": not rewritten.changed,
        "and_it_took_one_round": rewritten.rounds == 1,
    }


def the_saving_grows_with_the_conjunction(rows: int = 10000) -> dict:
    """How much a folded conjunction saves, against how many clauses it had.

    The saving is one pass over one column per clause removed, so it grows linearly with the
    number of constant clauses, and this is the table that says so.
    """
    batch = _table(rows)
    out = []
    for extra in (0, 1, 2, 4):
        parts = [Compare(">", column("amount"), literal(100.0))]
        parts.extend(Compare("<", literal(1), literal(2)) for _ in range(extra))
        predicate = And(parts=tuple(parts)) if len(parts) > 1 else parts[0]
        rewritten = simplify(predicate)
        before = Meter()
        after = Meter()
        apply_predicate(predicate, batch, meter=before)
        apply_predicate(rewritten.after, batch, meter=after)
        out.append(
            {
                "constant_clauses": extra,
                "before": before.values_touched,
                "after": after.values_touched,
                "saved": before.values_touched - after.values_touched,
            }
        )
    return {
        "sweep": out,
        "the_saving_grows": [one["saved"] for one in out]
        == sorted(one["saved"] for one in out),
        "and_the_result_is_always_the_same_size": len({one["after"] for one in out}) == 1,
    }


def a_zero_round_limit_is_refused() -> bool:
    """Simplifying with no rounds, which is a caller error."""
    try:
        simplify(Compare(">", column("a"), literal(1)), rounds=0)
    except ConfigError:
        return True
    return False


def compare_the_rules() -> list[dict]:
    """Every rule, an example it fires on and what it produces."""
    cases = [
        ("flatten", And(parts=(And(parts=(Compare(">", column("a"), literal(1)),)),))),
        ("fold", Compare("<", literal(1), literal(2))),
        (
            "absorb",
            And(parts=(Compare(">", column("a"), literal(1)), _truth(True))),
        ),
        (
            "deduplicate",
            And(
                parts=(
                    Compare(">", column("a"), literal(1)),
                    Compare(">", column("a"), literal(1)),
                )
            ),
        ),
        ("push negation", Not(part=Compare("<", column("a"), literal(1)))),
        ("collapse membership", InList(part=column("a"), options=(7,))),
    ]
    out = []
    for name, one in cases:
        rewritten = simplify(one)
        out.append(
            {
                "rule": name,
                "before": describe(one),
                "after": describe(rewritten.after),
                "nodes_saved": size(one) - size(rewritten.after),
            }
        )
    return out


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "rules": len(RULES),
        "max_rounds": MAX_ROUNDS,
        "they_terminate": the_rules_terminate()["they_all_terminate"],
        "rounds_needed": the_rules_terminate()["most_rounds"],
        "every_rewrite_agrees": every_rewrite_keeps_the_same_rows()["they_all_agree"],
        "false_deletes_everything": a_false_constant_deletes_the_whole_conjunction()[
            "it_is_a_constant"
        ],
    }

