from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import UnsupportedPlan
from cqe.exec.batch import Batch
from cqe.exec.expr import And, Compare, InList, IsNull, Not, Or, column, literal
from cqe.plan.physical import run
from cqe.sql import render as printer
from cqe.sql.parse import parse
from cqe.sql.parse import plan as plan_query
from cqe.sql.render import QUERIES, expression, render
from cqe.verify.reference import Rows, agree


@pytest.fixture(scope="module")
def catalogue() -> dict[str, Batch]:
    """Two tables to render plans against."""
    state = np.random.default_rng(293)
    rows = 1000
    facts = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 20, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("label", [f"kind{one}" for one in state.integers(0, 6, rows)]),
        ]
    )
    shops = Batch.from_columns(
        [
            integer_column("shop", np.arange(20)),
            string_column("region", [f"region{one % 4}" for one in range(20)]),
        ]
    )
    return {"facts": facts, "shops": shops}


def test_a_plan_renders_with_every_clause():
    assert printer.a_plan_renders_as_a_query()["it_has_every_clause"]


def test_a_rendered_query_is_short():
    assert printer.a_plan_renders_as_a_query()["and_it_is_shorter_than_the_tree"]


def test_every_query_round_trips():
    assert printer.every_query_round_trips()["they_all_round_trip"]


def test_no_query_failed_to_round_trip():
    assert printer.every_query_round_trips()["failures"] == []


def test_every_round_trip_keeps_its_answer():
    assert printer.a_round_trip_keeps_the_answer()["they_all_agree"]


def test_a_pushed_predicate_survives_the_render():
    assert printer.a_pushed_predicate_comes_back()["the_predicate_survived"]


def test_a_pushed_and_unpushed_plan_render_the_same():
    assert printer.a_pushed_predicate_comes_back()["and_both_render_the_same"]


def test_the_two_plans_really_differ():
    assert printer.a_pushed_predicate_comes_back()["the_plans_differ"]


def test_a_predicate_is_fully_bracketed():
    assert printer.a_predicate_is_fully_bracketed()["it_brackets_the_or"]


def test_the_bracketed_predicate_round_trips():
    assert printer.a_predicate_is_fully_bracketed()["and_it_round_trips"]


def test_a_quote_is_doubled():
    assert printer.a_string_with_a_quote_survives()["it_doubled_the_quote"]


def test_no_backslash_is_used():
    assert printer.a_string_with_a_quote_survives()["and_no_backslash"]


def test_the_quoted_value_survives():
    assert printer.a_string_with_a_quote_survives()["to_the_same_value"]


def test_three_literals_come_back_as_literals():
    assert printer.every_literal_type_renders()["three_come_back_as_literals"]


def test_a_negative_literal_comes_back_as_arithmetic():
    assert printer.every_literal_type_renders()["the_negative_one_is_an_arithmetic"]


def test_every_literal_value_survives():
    assert printer.every_literal_type_renders()["and_every_value_survives"]


def test_a_null_renders_as_null():
    assert printer.a_null_literal_renders_as_null()["it_says_null"]


def test_a_null_does_not_render_as_none():
    assert printer.a_null_literal_renders_as_null()["and_not_none"]


def test_the_negated_null_test_differs():
    assert printer.a_null_literal_renders_as_null()["the_negated_form_differs"]


def test_the_plan_starts_with_the_limit():
    assert printer.the_clause_order_is_not_the_plan_order()["the_plan_starts_with_the_limit"]


def test_the_query_ends_with_the_limit():
    assert printer.the_clause_order_is_not_the_plan_order()["and_the_query_ends_with_it"]


def test_a_join_qualifies_both_sides():
    assert printer.a_join_qualifies_its_keys()["it_qualified_both_sides"]


def test_a_rendered_join_round_trips():
    assert printer.a_join_qualifies_its_keys()["and_it_round_trips"]


def test_an_aggregate_without_a_projection_still_renders():
    assert printer.an_aggregate_renders_its_select_list()["and_the_select_list_is_still_right"]


def test_the_aggregate_plan_has_no_projection():
    assert printer.an_aggregate_renders_its_select_list()["it_has_no_projection"]


def test_a_rendered_query_is_shorter_than_its_tree():
    assert printer.a_rendered_query_is_shorter_than_its_tree()["the_query_is_shorter"]


def test_an_unrenderable_node_is_refused():
    assert printer.an_unrenderable_node_is_refused()


def test_an_unrenderable_expression_is_refused():
    assert printer.an_unrenderable_expression_is_refused()


def test_the_query_table_covers_every_query():
    assert len(printer.compare_the_queries()) == len(QUERIES)


def test_every_query_in_the_table_round_trips():
    assert all(one["round_trips"] for one in printer.compare_the_queries())


def test_the_summary_says_they_all_round_trip():
    assert printer.summarise()["they_all_round_trip"]


def test_a_comparison_renders_with_brackets():
    assert expression(Compare(">", column("a"), literal(1))) == "(a > 1)"


def test_a_conjunction_renders_with_and():
    made = expression(
        And(
            parts=(Compare(">", column("a"), literal(1)), Compare("<", column("b"), literal(2)))
        )
    )
    assert " and " in made


def test_a_disjunction_renders_with_or():
    made = expression(
        Or(parts=(Compare(">", column("a"), literal(1)), Compare("<", column("b"), literal(2))))
    )
    assert " or " in made


def test_a_negation_renders_with_not():
    assert expression(Not(part=Compare(">", column("a"), literal(1)))).startswith("(not")


def test_a_null_test_renders():
    assert expression(IsNull(part=column("a"))) == "(a is null)"


def test_a_negated_null_test_renders():
    assert expression(IsNull(part=column("a"), negated=True)) == "(a is not null)"


def test_a_membership_test_renders():
    made = expression(InList(part=column("a"), options=(1, 2, 3)))
    assert made == "(a in (1, 2, 3))"


def test_a_string_membership_test_renders():
    made = expression(InList(part=column("a"), options=("x", "y")))
    assert made == "(a in ('x', 'y'))"


def test_an_integer_literal_renders_plainly():
    assert expression(literal(42)) == "42"


def test_a_string_literal_is_quoted():
    assert expression(literal("text")) == "'text'"


def test_a_column_renders_as_its_name():
    assert expression(column("amount")) == "amount"


def test_a_scan_renders_as_a_star_query(catalogue):
    made = render(plan_query("select * from facts", catalogue))
    assert made.text == "select * from facts"


def test_a_projection_renders_its_columns(catalogue):
    made = render(plan_query("select id, amount from facts", catalogue))
    assert made.text == "select id, amount from facts"


def test_a_filter_renders_a_where_clause(catalogue):
    made = render(plan_query("select id from facts where amount > 100", catalogue))
    assert "where" in made.text


def test_a_sort_renders_an_order_clause(catalogue):
    made = render(plan_query("select id from facts order by amount desc", catalogue))
    assert "order by amount desc" in made.text


def test_a_limit_renders_a_limit_clause(catalogue):
    made = render(plan_query("select id from facts limit 5", catalogue))
    assert "limit 5" in made.text


def test_an_offset_renders_an_offset_clause(catalogue):
    made = render(plan_query("select id from facts limit 5 offset 3", catalogue))
    assert "offset 3" in made.text


def test_a_group_renders_a_group_clause(catalogue):
    made = render(plan_query("select shop, count(*) as n from facts group by shop", catalogue))
    assert "group by shop" in made.text


def test_a_rendered_plan_counts_its_nodes(catalogue):
    made = render(plan_query("select id from facts where amount > 100", catalogue))
    assert made.nodes >= 2


def test_a_rendered_plan_lists_its_clauses(catalogue):
    made = render(plan_query("select id from facts where amount > 100 limit 5", catalogue))
    assert "where" in made.clauses and "limit" in made.clauses


def test_a_rendered_plan_summarises(catalogue):
    made = render(plan_query("select id from facts", catalogue))
    assert made.as_dict()["length"] == len(made.text)


def test_every_query_parses_after_rendering(catalogue):
    for one in QUERIES:
        text = render(plan_query(one, catalogue)).text
        assert parse(text) is not None


def test_every_query_gives_the_same_rows(catalogue):
    for one in QUERIES:
        first = plan_query(one, catalogue)
        second = plan_query(render(first).text, catalogue)
        assert agree(
            Rows.of(run(first, catalogue).batch), Rows.of(run(second, catalogue).batch)
        )


def test_a_multi_key_sort_renders_both(catalogue):
    made = render(plan_query("select id from facts order by shop, amount desc", catalogue))
    assert "shop, amount desc" in made.text


def test_a_conjunction_of_filters_joins_with_and(catalogue):
    made = render(plan_query("select id from facts where amount > 100 and shop < 5", catalogue))
    assert " and " in made.text


def test_an_unknown_expression_raises():
    class Strange:
        pass

    with pytest.raises(UnsupportedPlan):
        expression(Strange())
