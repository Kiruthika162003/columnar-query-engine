from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import ParseError, UnknownColumn
from cqe.exec.batch import Batch
from cqe.exec.expr import And, Arithmetic, Compare, InList, IsNull, Not, Or
from cqe.plan.logical import Group, Join, Limit, Project, Sort, walk
from cqe.plan.physical import execute
from cqe.sql import parse as parser
from cqe.sql import tokenise as lexer
from cqe.verify.reference import Rows, agree, order_by, select, where


@pytest.fixture(scope="module")
def catalogue() -> dict[str, Batch]:
    """Two tables shared by every test in this file."""
    state = np.random.default_rng(19)
    rows = 600
    facts = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 12, rows)),
            floating_column("amount", state.normal(100, 20, rows)),
            string_column("label", [f"kind{one % 4}" for one in range(rows)]),
        ]
    )
    shops = Batch.from_columns(
        [
            integer_column("shop", np.arange(12)),
            string_column("region", [f"region{one % 3}" for one in range(12)]),
        ]
    )
    return {"facts": facts, "shops": shops}


def test_a_query_becomes_tokens():
    assert lexer.a_query_tokenises()["it_ends_with_an_end_token"]


def test_every_token_position_indexes_the_source():
    assert lexer.positions_point_at_the_source()["every_position_lands_on_its_token"]


def test_positions_increase():
    assert lexer.positions_point_at_the_source()["they_increase"]


def test_two_character_symbols_are_one_token():
    assert lexer.the_longest_symbol_wins()["they_are_single_tokens"]


def test_a_one_character_symbol_still_works():
    assert lexer.the_longest_symbol_wins()["and_a_bare_one_still_works"]


def test_a_keyword_is_a_word_like_any_other():
    assert lexer.a_keyword_is_just_a_word()["they_are_all_words"]


def test_a_reserved_word_can_name_a_column():
    assert lexer.a_keyword_is_just_a_word()["and_still_usable_as_a_name"]


def test_keyword_matching_ignores_case():
    measured = lexer.matching_is_case_insensitive_on_words_only()
    assert measured["select_matches_lowercase"] and measured["and_uppercase"]


def test_string_matching_does_not_ignore_case():
    assert lexer.matching_is_case_insensitive_on_words_only()["a_string_is_exact"]


def test_two_quotes_become_one():
    assert lexer.a_string_with_a_quote_in_it_survives()["it_became_one_quote"]


def test_an_empty_string_is_a_token():
    assert lexer.a_string_with_a_quote_in_it_survives()["an_empty_string_is_a_token"]


def test_a_comment_runs_to_the_end_of_the_line():
    assert lexer.a_comment_is_skipped()["the_query_survived"]


def test_a_decimal_is_a_single_number():
    assert lexer.a_decimal_is_one_token()["the_decimal_is_one_token"]


def test_a_qualified_name_is_not_a_decimal():
    assert lexer.a_decimal_is_one_token()["a_qualified_name_is_three_tokens"]


def test_an_unterminated_string_points_at_its_quote():
    measured = lexer.an_unterminated_string_is_refused()
    assert measured["it_was_refused"] and measured["it_points_at_the_quote"]


def test_an_unknown_character_names_itself():
    assert lexer.an_unknown_character_is_refused()["it_names_the_character"]


def test_a_parse_error_draws_a_caret():
    assert lexer.an_error_marks_the_position()["the_caret_is_under_the_character"]


def test_an_empty_query_is_only_an_end_token():
    assert lexer.an_empty_query_is_one_end_token()["it_is_one_token"]


def test_a_stream_refuses_with_both_names():
    assert lexer.a_stream_walks_and_refuses()["the_refusal_names_both"]


def test_peeking_leaves_the_cursor_alone():
    assert lexer.a_stream_walks_and_refuses()["peeking_does_not_move"]


def test_reading_past_the_end_is_safe():
    assert lexer.a_stream_ends_cleanly()["reading_past_the_end_is_safe"]


def test_expect_keyword_lists_every_option():
    assert lexer.expect_keyword_lists_the_options()["it_lists_them"]


def test_the_token_shapes_grow_with_the_query():
    counts = [one["tokens"] for one in lexer.compare_the_shapes()]
    assert counts == sorted(counts)


def test_a_query_with_no_tokens_is_refused():
    with pytest.raises(ParseError):
        parser.parse("   ")


def test_the_simplest_query_names_its_table():
    assert parser.the_simplest_query_parses()["table"] == "facts"


def test_a_star_builds_no_projection():
    assert parser.a_star_is_not_a_projection()["the_star_has_no_project"]


def test_a_named_list_does_build_one():
    assert parser.a_star_is_not_a_projection()["the_named_one_does"]


def test_naming_every_column_in_order_is_a_star():
    assert parser.selecting_every_column_by_name_is_also_not_a_projection()[
        "the_ordered_one_is_bare"
    ]


def test_reordering_the_columns_is_a_real_projection():
    assert parser.selecting_every_column_by_name_is_also_not_a_projection()[
        "the_reordered_one_projects"
    ]


def test_and_binds_tighter_than_or():
    assert parser.precedence_is_the_standard_one()["and_binds_tighter"]


def test_not_binds_tighter_than_and():
    assert parser.precedence_is_the_standard_one()["not_binds_tighter"]


def test_brackets_change_the_tree():
    assert parser.parentheses_override_precedence()["they_differ"]


def test_a_bracketed_or_sits_under_an_and():
    assert parser.parentheses_override_precedence()["and_holds_an_or"]


def test_arithmetic_is_below_comparison():
    assert parser.arithmetic_binds_tighter_than_comparison()["its_left_is_arithmetic"]


def test_a_product_is_below_a_sum():
    assert parser.multiplication_binds_tighter_than_addition()["its_right_is_a_product"]


def test_one_plus_two_times_three_is_seven():
    assert parser.multiplication_binds_tighter_than_addition()["it_is_seven"]


def test_a_chained_comparison_is_refused():
    assert parser.comparisons_do_not_chain()["it_was_refused"]


def test_the_chain_refusal_suggests_a_conjunction():
    assert parser.comparisons_do_not_chain()["it_says_what_to_write_instead"]


def test_the_spelled_out_conjunction_parses():
    assert parser.comparisons_do_not_chain()["and_the_spelled_out_form_works"]


def test_an_implicit_alias_is_refused():
    assert parser.a_missing_as_is_refused()["it_was_refused"]


def test_an_explicit_alias_is_accepted():
    assert parser.a_missing_as_is_refused()["and_the_explicit_form_works"]


def test_a_keyword_alias_names_the_keyword():
    assert parser.a_keyword_cannot_be_an_alias()["it_names_the_word"]


def test_a_join_written_either_way_builds_the_same_keys():
    assert parser.a_join_orients_itself()["they_agree"]


def test_the_build_order_puts_the_limit_on_top():
    assert parser.the_build_order_is_scan_join_filter_group_sort_limit()[
        "it_starts_at_the_limit"
    ]


def test_the_build_order_puts_the_join_at_the_bottom():
    assert parser.the_build_order_is_scan_join_filter_group_sort_limit()["then_the_join"]


def test_a_matching_select_list_needs_no_projection():
    assert parser.the_build_order_is_scan_join_filter_group_sort_limit()[
        "there_is_no_projection"
    ]


def test_a_swapped_select_list_needs_one():
    assert parser.the_build_order_is_scan_join_filter_group_sort_limit()[
        "and_swapping_the_select_list_puts_one_back"
    ]


def test_ordering_by_an_unselected_column_is_allowed():
    assert parser.ordering_by_an_unselected_column_works()["and_sorted_by_another"]


def test_the_result_still_has_one_column():
    assert parser.ordering_by_an_unselected_column_works()["it_returns_one_column"]


def test_an_aggregate_without_a_group_by_is_a_group():
    assert parser.an_aggregate_with_no_group_by_is_one_group()["it_is_a_group"]


def test_an_ungrouped_column_is_refused():
    assert parser.selecting_an_ungrouped_column_is_refused()["it_names_the_column"]


def test_grouping_by_that_column_fixes_it():
    assert parser.selecting_an_ungrouped_column_is_refused()["and_grouping_by_it_works"]


def test_a_misspelled_column_lists_the_real_ones():
    assert parser.a_misspelled_column_names_the_alternatives()["and_lists_the_real_ones"]


def test_the_misspelling_parses_before_it_is_refused():
    assert parser.a_misspelled_column_names_the_alternatives()["the_parse_itself_was_fine"]


def test_a_missing_table_lists_the_catalogue():
    assert parser.a_missing_table_lists_the_catalogue()["it_lists_what_there_is"]


def test_text_after_the_query_is_refused():
    assert parser.trailing_text_is_refused()["it_names_the_leftover"]


def test_an_in_list_keeps_its_values():
    assert parser.an_in_list_parses()["options"] == [1, 2, 3]


def test_an_in_list_of_strings_parses():
    assert parser.an_in_list_parses()["strings_work"] == ["a", "b"]


def test_is_null_and_is_not_null_differ():
    assert parser.is_null_parses_both_ways()["one_is_negated"]


def test_a_description_reparses_to_itself():
    assert parser.a_query_round_trips_through_its_own_description()["it_is_stable"]


def test_the_description_keeps_every_clause():
    measured = parser.a_query_round_trips_through_its_own_description()
    assert measured["the_group"] and measured["the_order"] and measured["the_limit"]


def test_a_parsed_query_agrees_with_the_reference():
    assert parser.a_parsed_query_evaluates_to_the_same_rows_as_the_reference()["they_agree"]


def test_the_parsed_query_also_agrees_on_the_order():
    assert parser.a_parsed_query_evaluates_to_the_same_rows_as_the_reference()[
        "and_in_the_same_order"
    ]


def test_the_clause_table_grows():
    nodes = [one["nodes"] for one in parser.compare_the_clauses()]
    assert min(nodes) == 1 and max(nodes) == 4


def test_the_summary_reports_the_precedence():
    assert parser.summarise()["precedence_holds"]


def test_a_where_clause_becomes_a_filter(catalogue):
    built = parser.plan("select id from facts where amount > 100", catalogue)
    kinds = [type(one).__name__ for one in walk(built)]
    assert "Filter" in kinds


def test_a_group_by_becomes_a_group(catalogue):
    built = parser.plan("select shop, count(*) as n from facts group by shop", catalogue)
    assert isinstance(built, Group)


def test_an_order_by_becomes_a_sort(catalogue):
    built = parser.plan("select id from facts order by amount", catalogue)
    assert any(isinstance(one, Sort) for one in walk(built))


def test_a_limit_becomes_a_limit(catalogue):
    built = parser.plan("select id from facts limit 5", catalogue)
    assert any(isinstance(one, Limit) for one in walk(built))


def test_a_join_becomes_a_join(catalogue):
    built = parser.plan("select id from facts join shops on facts.shop = shops.shop", catalogue)
    assert any(isinstance(one, Join) for one in walk(built))


def test_an_offset_is_carried_into_the_plan(catalogue):
    built = parser.plan("select id from facts limit 5 offset 3", catalogue)
    limits = [one for one in walk(built) if isinstance(one, Limit)]
    assert limits[0].offset == 3


def test_a_fractional_limit_is_refused():
    with pytest.raises(ParseError):
        parser.parse("select id from facts limit 2.5")


def test_a_sum_of_a_star_is_refused():
    with pytest.raises(ParseError):
        parser.parse("select sum(*) from facts")


def test_a_count_of_a_star_is_not():
    assert parser.parse("select count(*) as n from facts").items[0].function == "count"


def test_an_aggregate_over_a_missing_column_is_refused(catalogue):
    with pytest.raises(UnknownColumn):
        parser.plan("select shop, sum(nothing) as t from facts group by shop", catalogue)


def test_a_group_key_that_is_not_a_column_is_refused(catalogue):
    with pytest.raises(UnknownColumn):
        parser.plan("select count(*) as n from facts group by nothing", catalogue)


def test_an_order_key_that_is_not_a_column_is_refused(catalogue):
    with pytest.raises(UnknownColumn):
        parser.plan("select id from facts order by nothing", catalogue)


def test_a_selected_column_that_is_not_there_is_refused(catalogue):
    with pytest.raises(UnknownColumn):
        parser.plan("select nothing from facts", catalogue)


def test_a_join_on_a_missing_key_is_refused(catalogue):
    with pytest.raises(UnknownColumn):
        parser.plan("select id from facts join shops on facts.nothing = shops.shop", catalogue)


def test_a_query_with_no_from_is_refused():
    with pytest.raises(ParseError):
        parser.parse("select 1")


def test_an_unclosed_bracket_is_refused():
    with pytest.raises(ParseError):
        parser.parse("select id from facts where (amount > 1")


def test_a_predicate_over_a_string_parses(catalogue):
    built = parser.plan("select id from facts where label = 'kind1'", catalogue)
    assert execute(built, catalogue).rows > 0


def test_a_string_predicate_agrees_with_the_reference(catalogue):
    built = parser.plan("select id from facts where label = 'kind1'", catalogue)
    produced = execute(built, catalogue)
    expected = select(
        where(Rows.of(catalogue["facts"]), lambda one: one["label"] == "kind1"), ["id"]
    )
    assert agree(Rows.of(produced), expected)


def test_an_in_list_agrees_with_the_reference(catalogue):
    built = parser.plan("select id from facts where shop in (1, 2, 3)", catalogue)
    produced = execute(built, catalogue)
    expected = select(
        where(Rows.of(catalogue["facts"]), lambda one: one["shop"] in (1, 2, 3)), ["id"]
    )
    assert agree(Rows.of(produced), expected)


def test_a_conjunction_agrees_with_the_reference(catalogue):
    built = parser.plan("select id from facts where amount > 100 and shop < 6", catalogue)
    produced = execute(built, catalogue)
    expected = select(
        where(
            Rows.of(catalogue["facts"]),
            lambda one: one["amount"] > 100 and one["shop"] < 6,
        ),
        ["id"],
    )
    assert agree(Rows.of(produced), expected)


def test_a_disjunction_agrees_with_the_reference(catalogue):
    built = parser.plan("select id from facts where amount > 130 or shop = 0", catalogue)
    produced = execute(built, catalogue)
    expected = select(
        where(
            Rows.of(catalogue["facts"]),
            lambda one: one["amount"] > 130 or one["shop"] == 0,
        ),
        ["id"],
    )
    assert agree(Rows.of(produced), expected)


def test_a_negation_agrees_with_the_reference(catalogue):
    built = parser.plan("select id from facts where not shop = 0", catalogue)
    produced = execute(built, catalogue)
    expected = select(where(Rows.of(catalogue["facts"]), lambda one: one["shop"] != 0), ["id"])
    assert agree(Rows.of(produced), expected)


def test_an_ordered_query_agrees_row_for_row(catalogue):
    built = parser.plan("select id, amount from facts order by amount desc", catalogue)
    produced = execute(built, catalogue)
    expected = select(
        order_by(Rows.of(catalogue["facts"]), ["amount"], descending=[True]),
        ["id", "amount"],
    )
    assert agree(Rows.of(produced), expected, ordered=True)


def test_two_sort_keys_parse(catalogue):
    built = parser.plan("select id from facts order by shop, amount desc", catalogue)
    keys = next(one for one in walk(built) if isinstance(one, Sort)).keys
    assert len(keys) == 2 and keys[1].descending


def test_the_first_sort_key_is_ascending_by_default(catalogue):
    built = parser.plan("select id from facts order by shop, amount desc", catalogue)
    keys = next(one for one in walk(built) if isinstance(one, Sort)).keys
    assert not keys[0].descending


def test_an_explicit_ascending_parses(catalogue):
    built = parser.plan("select id from facts order by amount asc", catalogue)
    keys = next(one for one in walk(built) if isinstance(one, Sort)).keys
    assert not keys[0].descending


def test_a_having_clause_filters_the_groups(catalogue):
    built = parser.plan(
        "select shop, count(*) as n from facts group by shop having n > 40", catalogue
    )
    produced = execute(built, catalogue)
    assert produced.rows >= 0 and "n" in produced.schema


def test_a_having_clause_over_a_missing_name_is_refused(catalogue):
    with pytest.raises(UnknownColumn):
        parser.plan(
            "select shop, count(*) as n from facts group by shop having m > 1", catalogue
        )


def test_every_aggregate_function_parses(catalogue):
    for function in ("sum", "min", "max", "avg"):
        built = parser.plan(
            f"select shop, {function}(amount) as v from facts group by shop", catalogue
        )
        assert execute(built, catalogue).rows > 0


def test_an_aggregate_gets_a_default_name():
    assert parser.parse("select sum(amount) from facts").items[0].alias == "sum_amount"


def test_a_count_of_a_star_gets_a_default_name():
    assert parser.parse("select count(*) from facts").items[0].alias == "count"


def test_a_select_item_describes_itself():
    assert parser.parse("select sum(amount) as t from facts").items[0].describe() == (
        "sum(amount) as t"
    )


def test_a_plain_item_describes_itself_without_an_alias():
    assert parser.parse("select id from facts").items[0].describe() == "id"


def test_a_parsed_query_summarises():
    summary = parser.parse("select id from facts limit 3").as_dict()
    assert summary["limit"] == 3 and summary["tables"] == ["facts"]


def test_distinct_is_recorded():
    assert parser.parse("select distinct label from facts").distinct


def test_a_from_entry_knows_whether_it_was_joined():
    query = parser.parse("select id from facts join shops on facts.shop = shops.shop")
    assert not query.tables[0].is_joined and query.tables[1].is_joined


def test_the_predicate_shapes_are_the_expression_types():
    predicate = parser.parse("select id from facts where a > 1 and b < 2").predicate
    assert isinstance(predicate, And) and all(
        isinstance(one, Compare) for one in predicate.parts
    )


def test_an_or_of_three_is_one_node():
    predicate = parser.parse("select id from facts where a = 1 or a = 2 or a = 3").predicate
    assert isinstance(predicate, Or) and len(predicate.parts) == 3


def test_a_double_negation_nests():
    predicate = parser.parse("select id from facts where not not a = 1").predicate
    assert isinstance(predicate, Not) and isinstance(predicate.part, Not)


def test_a_null_test_holds_a_column():
    predicate = parser.parse("select id from facts where a is null").predicate
    assert isinstance(predicate, IsNull)


def test_a_membership_test_holds_its_options():
    predicate = parser.parse("select id from facts where a in (1)").predicate
    assert isinstance(predicate, InList) and predicate.options == (1,)


def test_a_negative_literal_is_arithmetic():
    predicate = parser.parse("select id from facts where a > -1").predicate
    assert isinstance(predicate.right, Arithmetic)


def test_a_projection_of_two_columns_is_a_project(catalogue):
    built = parser.plan("select amount, id from facts", catalogue)
    assert isinstance(built, Project)


def test_the_plan_schema_matches_the_select_list(catalogue):
    built = parser.plan("select amount, id from facts", catalogue)
    assert list(built.schema().names) == ["amount", "id"]


def test_a_full_query_runs(catalogue):
    built = parser.plan(
        "select region, count(*) as n from facts join shops on facts.shop = shops.shop "
        "where amount > 90 group by region order by n desc limit 2",
        catalogue,
    )
    produced = execute(built, catalogue)
    assert produced.rows == 2 and list(produced.schema.names) == ["region", "n"]


def test_the_full_query_is_ordered_downwards(catalogue):
    built = parser.plan(
        "select region, count(*) as n from facts join shops on facts.shop = shops.shop "
        "group by region order by n desc",
        catalogue,
    )
    counts = execute(built, catalogue).column("n").values
    assert list(counts) == sorted(counts, reverse=True)
