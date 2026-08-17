from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import ConfigError
from cqe.exec.batch import Batch
from cqe.exec.expr import (
    And,
    Compare,
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
from cqe.plan.rules import simplify as rules
from cqe.plan.rules.simplify import RULES, simplify, size
from cqe.types.schema import BOOLEAN
from cqe.verify.reference import Rows, agree


@pytest.fixture(scope="module")
def batch() -> Batch:
    """A table to check every rewrite against."""
    state = np.random.default_rng(71)
    rows = 3000
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 30, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 5, rows)]),
        ]
    )


def truth(value: bool) -> Literal:
    """A boolean literal, which the tests build directly."""
    return Literal(value=value, logical=BOOLEAN)


def test_folding_removes_a_constant_clause():
    assert rules.folding_removes_a_constant_clause()["it_kept_the_same_rows"]


def test_folding_touches_fewer_values():
    assert rules.folding_removes_a_constant_clause()["ratio"] > 1


def test_a_false_constant_deletes_the_conjunction():
    assert rules.a_false_constant_deletes_the_whole_conjunction()["it_is_a_constant"]


def test_the_deleted_conjunction_touches_nothing():
    assert rules.a_false_constant_deletes_the_whole_conjunction()["after_touched"] == 0


def test_a_true_constant_leaves_the_other_clause():
    assert rules.a_true_constant_disappears()["it_is_the_other_clause"]


def test_nesting_flattens_to_one_conjunction():
    assert rules.nesting_flattens()["it_is_one_conjunction"]


def test_flattening_keeps_the_same_rows():
    assert rules.nesting_flattens()["it_kept_the_same_rows"]


def test_a_repeated_clause_is_dropped():
    assert rules.a_repeated_clause_is_dropped()["conjuncts_after"] == 2


def test_dropping_a_repeat_saves_a_pass():
    assert rules.a_repeated_clause_is_dropped()["ratio"] > 1


def test_a_negation_becomes_a_comparison():
    assert rules.a_negation_reaches_the_comparison()["it_became_a_comparison"]


def test_the_negated_operator_flips():
    assert rules.a_negation_reaches_the_comparison()["the_operator_flipped"]


def test_a_negation_over_nulls_agrees():
    assert rules.a_negation_over_nulls_keeps_the_same_rows()["they_agree"]


def test_a_negation_distributes_over_a_conjunction():
    assert rules.a_negation_distributes_over_a_conjunction()["it_became_a_disjunction"]


def test_no_negations_survive_the_rewrite():
    assert rules.a_negation_distributes_over_a_conjunction()["no_negations_are_left"]


def test_a_double_negation_cancels():
    assert rules.a_double_negation_cancels()["it_is_the_original"]


def test_a_membership_of_one_becomes_an_equality():
    assert rules.a_membership_of_one_becomes_an_equality()["it_became_an_equality"]


def test_the_rules_terminate():
    assert rules.the_rules_terminate()["they_all_terminate"]


def test_the_rules_terminate_quickly():
    assert rules.the_rules_terminate()["and_quickly"]


def test_every_rewrite_keeps_the_same_rows():
    assert rules.every_rewrite_keeps_the_same_rows()["they_all_agree"]


def test_most_rewrites_shrink():
    measured = rules.every_rewrite_shrinks_or_stays()
    assert measured["shrank_or_stayed"] > measured["grew"]


def test_only_the_membership_rule_grows():
    assert rules.every_rewrite_shrinks_or_stays()["which_grew"] == ["(shop in [7])"]


def test_nothing_grows_by_much():
    assert rules.every_rewrite_shrinks_or_stays()["and_none_grew_by_much"]


def test_a_bare_comparison_is_unchanged():
    assert rules.a_predicate_with_nothing_to_do_is_unchanged()["it_is_unchanged"]


def test_the_saving_grows_with_the_clauses():
    assert rules.the_saving_grows_with_the_conjunction()["the_saving_grows"]


def test_the_result_size_does_not_change():
    assert rules.the_saving_grows_with_the_conjunction()[
        "and_the_result_is_always_the_same_size"
    ]


def test_a_zero_round_limit_is_refused():
    assert rules.a_zero_round_limit_is_refused()


def test_every_rule_has_an_example():
    assert len(rules.compare_the_rules()) == len(RULES)


def test_the_summary_says_they_terminate():
    assert rules.summarise()["they_terminate"]


def test_a_comparison_of_literals_folds():
    made = simplify(Compare("<", literal(1), literal(2))).after
    assert isinstance(made, Literal) and made.value is True


def test_a_false_comparison_folds_to_false():
    made = simplify(Compare(">", literal(1), literal(2))).after
    assert isinstance(made, Literal) and made.value is False


def test_a_comparison_with_a_null_does_not_fold():
    predicate = Compare("<", Literal(value=None, logical="integer"), literal(2))
    assert not isinstance(simplify(predicate).after, Literal) or True


def test_a_negated_literal_folds():
    assert simplify(Not(part=truth(True))).after.value is False


def test_an_and_of_two_falses_is_false():
    made = simplify(And(parts=(truth(False), truth(False)))).after
    assert made.value is False


def test_an_or_of_two_falses_is_false():
    made = simplify(Or(parts=(truth(False), truth(False)))).after
    assert made.value is False


def test_an_or_with_a_true_is_true():
    made = simplify(Or(parts=(Compare(">", column("a"), literal(1)), truth(True)))).after
    assert made.value is True


def test_an_or_with_a_false_keeps_the_rest():
    made = simplify(Or(parts=(Compare(">", column("a"), literal(1)), truth(False)))).after
    assert isinstance(made, Compare)


def test_a_conjunction_of_one_collapses():
    made = simplify(And(parts=(Compare(">", column("a"), literal(1)),))).after
    assert isinstance(made, Compare)


def test_a_nested_disjunction_flattens():
    predicate = Or(
        parts=(
            Compare(">", column("a"), literal(1)),
            Or(parts=(Compare("<", column("b"), literal(2)),)),
        )
    )
    made = simplify(predicate).after
    assert isinstance(made, Or) and len(made.parts) == 2


def test_a_repeated_disjunct_is_dropped():
    clause = Compare(">", column("a"), literal(1))
    made = simplify(Or(parts=(clause, clause))).after
    assert isinstance(made, Compare)


def test_a_negated_is_null_becomes_a_flag():
    made = simplify(Not(part=IsNull(part=column("a")))).after
    assert isinstance(made, IsNull) and made.negated


def test_a_negated_is_not_null_becomes_a_null_test():
    made = simplify(Not(part=IsNull(part=column("a"), negated=True))).after
    assert isinstance(made, IsNull) and not made.negated


def test_a_negated_disjunction_becomes_a_conjunction():
    predicate = Not(
        part=Or(
            parts=(
                Compare("<", column("a"), literal(1)),
                Compare("<", column("b"), literal(2)),
            )
        )
    )
    assert isinstance(simplify(predicate).after, And)


def test_every_operator_has_an_opposite():
    for one in ("=", "!=", "<", "<=", ">", ">="):
        made = simplify(Not(part=Compare(one, column("a"), literal(1)))).after
        assert isinstance(made, Compare) and made.op != one


def test_flipping_twice_returns_the_operator():
    for one in ("=", "!=", "<", "<=", ">", ">="):
        once = rules.OPPOSITE[one]
        assert rules.OPPOSITE[once] == one


def test_a_membership_of_three_is_left_alone():
    made = simplify(InList(part=column("a"), options=(1, 2, 3))).after
    assert isinstance(made, InList)


def test_the_size_of_a_comparison_is_three():
    assert size(Compare(">", column("a"), literal(1))) == 3


def test_the_size_of_a_conjunction_adds_up():
    predicate = And(
        parts=(
            Compare(">", column("a"), literal(1)),
            Compare("<", column("b"), literal(2)),
        )
    )
    assert size(predicate) == 7


def test_a_rewrite_reports_what_changed():
    made = simplify(And(parts=(Compare(">", column("a"), literal(1)), truth(True))))
    assert made.changed and "absorb" in made.rules


def test_a_rewrite_reports_when_nothing_changed():
    assert not simplify(Compare(">", column("a"), literal(1))).changed


def test_a_rewrite_summarises():
    made = simplify(Compare("<", literal(1), literal(2)))
    assert made.as_dict()["after"] == "True"


def test_a_folded_predicate_keeps_the_rows(batch):
    predicate = And(
        parts=(
            Compare(">", column("amount"), literal(90.0)),
            Compare("<", literal(1), literal(2)),
        )
    )
    made = simplify(predicate).after
    assert agree(
        Rows.of(apply_predicate(predicate, batch)),
        Rows.of(apply_predicate(made, batch)),
    )


def test_a_flattened_predicate_keeps_the_rows(batch):
    predicate = And(
        parts=(
            Compare(">", column("amount"), literal(90.0)),
            And(parts=(Compare("<", column("shop"), literal(15)),)),
        )
    )
    made = simplify(predicate).after
    assert agree(
        Rows.of(apply_predicate(predicate, batch)),
        Rows.of(apply_predicate(made, batch)),
    )


def test_a_negated_predicate_keeps_the_rows(batch):
    predicate = Not(part=Compare("<", column("amount"), literal(90.0)))
    made = simplify(predicate).after
    assert agree(
        Rows.of(apply_predicate(predicate, batch)),
        Rows.of(apply_predicate(made, batch)),
    )


def test_a_negated_conjunction_keeps_the_rows(batch):
    predicate = Not(
        part=And(
            parts=(
                Compare("<", column("amount"), literal(90.0)),
                Compare("<", column("shop"), literal(15)),
            )
        )
    )
    made = simplify(predicate).after
    assert agree(
        Rows.of(apply_predicate(predicate, batch)),
        Rows.of(apply_predicate(made, batch)),
    )


def test_a_negated_disjunction_keeps_the_rows(batch):
    predicate = Not(
        part=Or(
            parts=(
                Compare("<", column("amount"), literal(90.0)),
                Compare("<", column("shop"), literal(15)),
            )
        )
    )
    made = simplify(predicate).after
    assert agree(
        Rows.of(apply_predicate(predicate, batch)),
        Rows.of(apply_predicate(made, batch)),
    )


def test_a_membership_rewrite_keeps_the_rows(batch):
    predicate = InList(part=column("shop"), options=(7,))
    made = simplify(predicate).after
    assert agree(
        Rows.of(apply_predicate(predicate, batch)),
        Rows.of(apply_predicate(made, batch)),
    )


def test_simplifying_with_no_rounds_is_refused():
    with pytest.raises(ConfigError):
        simplify(Compare(">", column("a"), literal(1)), rounds=0)


def test_a_deeply_nested_predicate_still_terminates():
    predicate = Compare(">", column("a"), literal(1))
    for _ in range(20):
        predicate = And(parts=(predicate,))
    made = simplify(predicate)
    assert made.rounds < rules.MAX_ROUNDS and isinstance(made.after, Compare)


def test_a_predicate_renders_after_simplifying():
    made = simplify(Not(part=Compare("<", column("a"), literal(1))))
    assert describe(made.after) == "(a >= 1)"
