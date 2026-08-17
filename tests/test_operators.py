from __future__ import annotations

import numpy as np
import pytest

from cqe.cost.meter import Meter
from cqe.errors import ConfigError, TypeMismatch, UnknownColumn
from cqe.exec import aggregate, sort
from cqe.exec import filter as filtering
from cqe.exec.batch import Batch
from cqe.exec.expr import (
    And,
    Arithmetic,
    Compare,
    InList,
    IsNull,
    Literal,
    Not,
    Or,
    all_of,
    column,
    conjuncts,
    describe,
    evaluate_to_mask,
    literal,
    to_callable,
)
from cqe.exec.join import hash as joins
from cqe.types.schema import BOOLEAN, FLOATING, INTEGER, STRING, Schema
from cqe.verify import reference


def sample() -> Batch:
    """The batch most expression tests use, with a null in each column."""
    return Batch.of(a=[1, 2, None, 4], g=["x", "y", "x", None], v=[1.5, 2.5, 3.5, 4.5])


class TestExpressions:
    def test_a_literal_broadcasts(self):
        assert literal(5).evaluate(sample()).to_list() == [5, 5, 5, 5]

    def test_a_literal_costs_nothing(self):
        meter = Meter()
        literal(5).evaluate(sample(), meter)
        assert meter.values_touched == 0

    def test_a_column_reference_reads_the_column(self):
        assert column("a").evaluate(sample()).to_list() == [1, 2, None, 4]

    def test_and_costs_one_value_per_row(self):
        meter = Meter()
        column("a").evaluate(sample(), meter)
        assert meter.values_touched == 4

    def test_an_unknown_column_is_refused(self):
        with pytest.raises(UnknownColumn, match="is not in"):
            column("z").evaluate(sample())

    def test_a_comparison_produces_booleans(self):
        result = Compare(">", column("a"), literal(1)).evaluate(sample())
        assert result.logical == BOOLEAN

    def test_a_comparison_against_null_is_null(self):
        result = Compare(">", column("a"), literal(1)).evaluate(sample())
        assert result.to_list()[2] is None

    def test_a_filter_drops_nulls(self):
        keep = evaluate_to_mask(Compare(">", column("a"), literal(1)), sample())
        assert list(keep) == [False, True, False, True]

    def test_a_negated_predicate_does_not_pick_nulls_back_up(self):
        predicate = Not(Compare("=", column("g"), literal("x")))
        assert list(evaluate_to_mask(predicate, sample())) == [False, True, False, False]

    def test_an_unknown_comparison_is_refused(self):
        with pytest.raises(ConfigError, match="not a comparison"):
            Compare("~", column("a"), literal(1))

    def test_a_string_equality_uses_the_dictionary(self):
        predicate = Compare("=", column("g"), literal("x"))
        assert list(evaluate_to_mask(predicate, sample())) == [True, False, True, False]

    def test_a_string_range_uses_the_dictionary_order(self):
        predicate = Compare("<", column("g"), literal("y"))
        assert list(evaluate_to_mask(predicate, sample())) == [True, False, True, False]

    def test_a_missing_literal_matches_nothing(self):
        predicate = Compare("=", column("g"), literal("zzz"))
        assert not any(evaluate_to_mask(predicate, sample()))

    def test_a_missing_literal_still_ranges(self):
        predicate = Compare("<", column("g"), literal("zzz"))
        assert list(evaluate_to_mask(predicate, sample())) == [True, True, True, False]

    def test_a_reversed_literal_comparison_works(self):
        predicate = Compare(">", literal("y"), column("g"))
        assert list(evaluate_to_mask(predicate, sample())) == [True, False, True, False]

    def test_comparing_a_string_to_a_number_is_refused(self):
        predicate = Compare("=", column("g"), literal(1))
        with pytest.raises(TypeMismatch):
            predicate.evaluate(sample())

    def test_arithmetic_combines_columns(self):
        result = Arithmetic("+", column("a"), literal(10)).evaluate(sample())
        assert result.to_list() == [11, 12, None, 14]

    def test_arithmetic_promotes_to_floating(self):
        result = Arithmetic("*", column("a"), column("v")).evaluate(sample())
        assert result.logical == FLOATING

    def test_an_unknown_operator_is_refused(self):
        with pytest.raises(ConfigError, match="not arithmetic"):
            Arithmetic("/", column("a"), literal(1))

    def test_arithmetic_on_a_string_is_refused(self):
        with pytest.raises(TypeMismatch, match="does not support"):
            Arithmetic("+", column("g"), literal(1)).type_of(sample().schema)


class TestThreeValuedLogic:
    def test_false_beats_null_in_an_and(self):
        """Row three has a of 4, so the first test is false, and a null g, so the second is
        null."""
        left = Compare("=", column("a"), literal(99))
        right = Compare("=", column("g"), literal("x"))
        result = And((left, right)).evaluate(sample())
        assert result.to_list()[3] is False

    def test_but_null_and_true_is_still_null(self):
        """Row two has a null, so the first test is null, and g of x, so the second is true."""
        left = Compare("=", column("a"), literal(99))
        right = Compare("=", column("g"), literal("x"))
        result = And((left, right)).evaluate(sample())
        assert result.to_list()[2] is None

    def test_true_beats_null_in_an_or(self):
        left = Compare("=", column("g"), literal("x"))
        right = Compare("=", column("a"), literal(1))
        result = Or((left, right)).evaluate(sample())
        assert result.to_list()[2] is True

    def test_null_survives_a_not(self):
        result = Not(Compare(">", column("a"), literal(1))).evaluate(sample())
        assert result.to_list()[2] is None

    def test_is_null_is_never_null(self):
        result = IsNull(column("a")).evaluate(sample())
        assert None not in result.to_list()

    def test_is_null_finds_the_missing(self):
        assert IsNull(column("a")).evaluate(sample()).to_list() == [
            False,
            False,
            True,
            False,
        ]

    def test_is_not_null_is_the_complement(self):
        assert IsNull(column("a"), negated=True).evaluate(sample()).to_list() == [
            True,
            True,
            False,
            True,
        ]

    def test_is_null_on_a_column_with_none_finds_none(self):
        batch = Batch.of(a=[1, 2, 3])
        assert not any(IsNull(column("a")).evaluate(batch).to_list())

    def test_an_in_list_tests_membership(self):
        result = InList(column("g"), ("x", "y")).evaluate(sample())
        assert result.to_list()[:2] == [True, True]

    def test_an_in_list_on_integers_works(self):
        result = InList(column("a"), (1, 4)).evaluate(sample())
        assert list(evaluate_to_mask(InList(column("a"), (1, 4)), sample())) == [
            True,
            False,
            False,
            True,
        ]
        assert result.logical == BOOLEAN

    def test_an_empty_in_list_is_refused(self):
        with pytest.raises(ConfigError, match="at least one option"):
            InList(column("a"), ())

    def test_an_empty_and_is_refused(self):
        with pytest.raises(ConfigError, match="at least one operand"):
            And(())

    def test_an_empty_or_is_refused(self):
        with pytest.raises(ConfigError, match="at least one operand"):
            Or(())

    def test_an_and_of_non_booleans_is_refused(self):
        with pytest.raises(TypeMismatch, match="takes booleans"):
            And((column("a"),)).type_of(sample().schema)

    def test_a_not_of_a_non_boolean_is_refused(self):
        with pytest.raises(TypeMismatch, match="takes a boolean"):
            Not(column("a")).type_of(sample().schema)

    def test_a_non_boolean_predicate_is_refused(self):
        with pytest.raises(TypeMismatch, match="predicate is boolean"):
            evaluate_to_mask(column("a"), sample())


class TestAgainstTheReference:
    @pytest.mark.parametrize(
        "expression",
        [
            Compare(">", column("a"), literal(1)),
            Compare("=", column("g"), literal("x")),
            Not(Compare("=", column("g"), literal("x"))),
            And((Compare(">", column("a"), literal(1)), IsNull(column("g")))),
            Or((Compare("=", column("a"), literal(1)), IsNull(column("g")))),
            IsNull(column("a")),
            IsNull(column("a"), negated=True),
            InList(column("g"), ("x",)),
        ],
    )
    def test_the_vectorised_form_matches_the_row_form(self, expression):
        batch = sample()
        table = reference.Rows.of(batch)
        fast = list(evaluate_to_mask(expression, batch))
        evaluator = to_callable(expression)
        slow = [
            reference.truth(evaluator(dict(zip(table.names, row, strict=True))))
            for row in table.rows
        ]
        assert fast == slow


class TestExpressionShape:
    def test_columns_used_finds_the_leaves(self):
        expression = And(
            (Compare(">", column("a"), literal(1)), Compare("=", column("g"), literal("x")))
        )
        assert expression.columns_used() == {"a", "g"}

    def test_a_literal_uses_nothing(self):
        assert literal(1).columns_used() == frozenset()

    def test_depth_grows_with_nesting(self):
        assert Compare(">", column("a"), literal(1)).depth() == 2

    def test_conjuncts_flatten_nested_ands(self):
        one = Compare(">", column("a"), literal(1))
        assert len(conjuncts(And((one, And((one, one)))))) == 3

    def test_a_non_conjunction_is_one_conjunct(self):
        assert len(conjuncts(Compare(">", column("a"), literal(1)))) == 1

    def test_all_of_rejoins(self):
        one = Compare(">", column("a"), literal(1))
        assert len(conjuncts(all_of([one, one]))) == 2

    def test_all_of_nothing_is_true(self):
        assert all_of([]) == Literal(True, BOOLEAN)

    def test_all_of_one_is_that_one(self):
        one = Compare(">", column("a"), literal(1))
        assert all_of([one]) is one

    def test_describe_renders_a_comparison(self):
        assert describe(Compare(">", column("a"), literal(1))) == "(a > 1)"

    def test_describe_renders_a_conjunction(self):
        one = Compare(">", column("a"), literal(1))
        assert " and " in describe(And((one, one)))

    def test_describe_renders_a_null_check(self):
        assert describe(IsNull(column("a"))) == "(a is null)"

    def test_a_type_check_reaches_the_schema(self):
        schema = Schema.of(("a", INTEGER), ("g", STRING))
        assert Compare(">", column("a"), literal(1)).type_of(schema) == BOOLEAN

    def test_an_expression_serialises(self):
        assert Compare(">", column("a"), literal(1)).as_dict()["op"] == ">"

    def test_a_bad_literal_type_is_refused(self):
        with pytest.raises(TypeMismatch, match="not a literal type"):
            literal([1, 2])


class TestFiltering:
    def test_the_representations_cross_at_a_quarter(self):
        rows = filtering.the_two_representations_cross_over_at_a_quarter()
        below = [row for row in rows if row["selectivity"] < 0.25]
        assert all(row["cheaper_form"] == "selection" for row in below)

    def test_and_the_mask_wins_above_it(self):
        rows = filtering.the_two_representations_cross_over_at_a_quarter()
        above = [row for row in rows if row["selectivity"] > 0.25]
        assert all(row["cheaper_form"] == "mask" for row in above)

    def test_chaining_gives_the_same_rows(self):
        assert filtering.chaining_conjuncts_saves_the_selectivity()["same_rows"]

    def test_and_touches_fewer_values(self):
        assert filtering.chaining_conjuncts_saves_the_selectivity()["chaining_wins"]

    def test_the_order_matters(self):
        assert filtering.the_order_of_the_conjuncts_matters_more()["the_order_matters"]

    def test_and_by_more_than_chaining_saves(self):
        ordered = filtering.the_order_of_the_conjuncts_matters_more()["ratio"]
        chained = filtering.chaining_conjuncts_saves_the_selectivity()["ratio"]
        assert ordered > 1.0 / chained

    def test_chaining_loses_when_nothing_is_selective(self):
        assert filtering.chaining_costs_something_when_nothing_is_selective()["chaining_loses"]

    def test_the_crossover_is_at_a_half(self):
        assert filtering.the_crossover_selectivity()["crossover"] == 0.5

    def test_a_wider_batch_makes_chaining_worse(self):
        ratios = [row["ratio"] for row in filtering.a_wider_batch_makes_chaining_worse()]
        assert ratios[-1] > ratios[0]

    def test_every_form_gives_the_same_rows(self):
        result = filtering.the_answer_never_changes()
        assert result["chained_matches"] and result["either_order_matches"]

    def test_nulls_are_dropped(self):
        assert filtering.the_answer_never_changes()["nulls_were_dropped"]

    def test_an_empty_selection_applies_cleanly(self):
        assert filtering.an_empty_selection_is_not_an_error()["applies_cleanly"]

    def test_a_full_selection_costs_the_gather(self):
        assert filtering.a_full_selection_costs_the_gather()["the_gather_is_the_cost"]

    def test_a_bad_ordering_is_refused(self):
        assert filtering.a_bad_ordering_is_refused()

    def test_a_mismatched_refinement_is_refused(self):
        assert filtering.a_mismatched_refinement_is_refused()

    def test_a_negative_row_count_is_refused(self):
        assert filtering.a_negative_row_count_is_refused()

    def test_everything_keeps_every_row(self):
        assert filtering.everything(10).kept == 10

    def test_nothing_keeps_none(self):
        assert filtering.nothing(10).kept == 0

    def test_a_selection_round_trips_through_a_mask(self):
        selection = filtering.from_mask(np.array([True, False, True]))
        assert list(selection.as_mask()) == [True, False, True]

    def test_a_selection_serialises(self):
        assert filtering.everything(4).as_dict()["kept"] == 4

    def test_apply_filters_a_batch(self):
        predicate = Compare(">", column("a"), literal(1))
        assert filtering.apply(predicate, sample()).rows == 2


class TestAggregation:
    def test_the_three_strategies_agree(self):
        result = aggregate.the_three_strategies_agree()
        assert result["hash_matches_counting"] and result["hash_matches_sorted"]

    def test_they_agree_with_the_reference(self):
        assert aggregate.they_agree_with_the_reference()["same"]

    def test_a_sum_over_only_nulls_is_null(self):
        assert aggregate.a_sum_over_only_nulls_is_null()["the_sum_is_null"]

    def test_but_the_count_is_zero(self):
        assert aggregate.a_sum_over_only_nulls_is_null()["the_count_is_zero"]

    def test_count_star_counts_rows(self):
        assert aggregate.count_star_and_count_disagree_on_nulls()["count_star_is_the_rows"]

    def test_and_the_difference_is_the_nulls(self):
        result = aggregate.count_star_and_count_disagree_on_nulls()
        assert result["the_difference_is_the_nulls"]

    def test_two_nulls_are_one_group(self):
        assert aggregate.two_nulls_are_one_group()["it_is_two_groups"]

    def test_a_correlated_key_adds_no_groups(self):
        assert aggregate.a_correlated_second_key_adds_no_groups()["no_extra_groups"]

    def test_but_costs_twice_as_much(self):
        assert aggregate.a_correlated_second_key_adds_no_groups()["cost_doubled"]

    def test_an_ungrouped_aggregate_is_one_group(self):
        assert aggregate.an_ungrouped_aggregate_is_one_group()["it_is_one_group"]

    def test_and_agrees_with_a_direct_sum(self):
        assert aggregate.an_ungrouped_aggregate_is_one_group()["agrees_with_the_direct_sum"]

    def test_the_cost_does_not_grow_with_the_groups(self):
        rows = aggregate.the_counting_form_does_no_work_per_group()
        assert len({row["hash_values"] for row in rows}) == 1

    def test_only_the_hash_form_probes(self):
        assert aggregate.the_counting_form_is_the_cheapest()["only_the_hash_form_probes"]

    def test_an_unsorted_input_is_refused(self):
        assert aggregate.an_unsorted_input_is_refused_by_the_sorted_form()

    def test_a_null_key_is_refused_by_the_counting_form(self):
        assert aggregate.a_null_key_is_refused_by_the_counting_form()

    def test_an_unknown_aggregate_is_refused(self):
        assert aggregate.an_unknown_aggregate_is_refused()

    def test_a_sourceless_aggregate_is_refused(self):
        assert aggregate.a_sourceless_aggregate_is_refused()

    def test_summing_a_string_is_refused(self):
        assert aggregate.summing_a_string_is_refused()

    def test_an_aggregate_serialises(self):
        assert aggregate.Aggregate("n", "count_star").as_dict()["function"] == "count_star"

    def test_a_grouping_serialises(self):
        batch = Batch.of(g=["x", "x", "y"], v=[1, 2, 3])
        result = aggregate.hash_aggregate(
            batch, ["g"], [aggregate.Aggregate("n", "count_star")]
        )
        assert result.as_dict()["groups"] == 2

    def test_an_empty_aggregate_list_is_refused(self):
        batch = Batch.of(g=["x"], v=[1])
        with pytest.raises(ConfigError, match="at least one aggregate"):
            aggregate.hash_aggregate(batch, ["g"], [])

    def test_a_mean_divides(self):
        batch = Batch.of(g=["x", "x"], v=[1, 3])
        result = aggregate.hash_aggregate(batch, ["g"], [aggregate.Aggregate("m", "mean", "v")])
        assert result.batch.to_rows()[0][1] == 2.0

    def test_min_and_max_bracket_the_group(self):
        batch = Batch.of(g=["x", "x", "x"], v=[5, 1, 9])
        result = aggregate.hash_aggregate(
            batch,
            ["g"],
            [aggregate.Aggregate("lo", "min", "v"), aggregate.Aggregate("hi", "max", "v")],
        )
        assert result.batch.to_rows()[0][1:] == [1, 9]


class TestSorting:
    def test_the_gather_grows_with_the_width(self):
        ratios = [row["ratio"] for row in sort.the_gather_costs_more_than_the_sort()]
        assert ratios == sorted(ratios)

    def test_a_top_k_saves_comparisons(self):
        rows = sort.a_top_k_avoids_almost_all_of_the_comparisons()
        assert all(row["ratio"] > 1.0 for row in rows)

    def test_but_only_by_the_log_of_the_row_count(self):
        rows = sort.a_top_k_avoids_almost_all_of_the_comparisons()
        assert max(row["ratio"] for row in rows) < 20

    def test_the_gather_saving_is_much_larger(self):
        assert sort.and_moves_almost_nothing()["ratio"] > 1000

    def test_and_is_the_row_count_over_the_limit(self):
        result = sort.and_moves_almost_nothing()
        assert result["the_saving_is_the_row_count_over_the_limit"]

    def test_the_top_k_answer_matches_a_full_sort(self):
        assert sort.the_top_k_answer_matches_a_full_sort()["same_rows"]

    def test_and_it_used_the_partition(self):
        assert sort.the_top_k_answer_matches_a_full_sort()["it_used_the_partition"]

    def test_a_multi_key_sort_costs_one_pass_per_key(self):
        rows = sort.a_multi_key_sort_costs_one_pass_per_key()
        assert len({row["per_key"] for row in rows}) == 1

    def test_a_descending_multi_key_sort_stays_stable(self):
        result = sort.a_descending_multi_key_sort_stays_stable()
        assert result["second_key_ascends_within_groups"]

    def test_and_the_first_key_really_descends(self):
        assert sort.a_descending_multi_key_sort_stays_stable()["first_key_descends"]

    def test_nulls_go_where_they_are_told(self):
        assert all(sort.nulls_go_where_they_are_told().values())

    def test_an_empty_batch_sorts_to_nothing(self):
        assert sort.an_empty_batch_sorts_to_nothing()["applies_cleanly"]

    def test_a_limit_past_the_end_is_a_full_sort(self):
        assert sort.a_limit_past_the_end_is_a_full_sort()["it_used_a_full_sort"]

    def test_a_top_k_with_nulls_falls_back(self):
        assert sort.a_top_k_with_nulls_falls_back()["it_fell_back"]

    def test_and_is_still_right(self):
        assert sort.a_top_k_with_nulls_falls_back()["the_answer_is_still_right"]

    def test_a_sort_with_no_keys_is_refused(self):
        assert sort.a_sort_with_no_keys_is_refused()

    def test_a_zero_limit_is_refused(self):
        assert sort.a_zero_limit_is_refused()

    def test_an_unknown_key_is_refused(self):
        assert sort.an_unknown_key_is_refused()

    def test_a_sort_key_serialises(self):
        assert sort.SortKey("a", descending=True).as_dict()["direction"] == "descending"

    def test_an_ordering_serialises(self):
        batch = Batch.of(a=[3, 1, 2])
        assert sort.order_by(batch, [sort.SortKey("a")]).as_dict()["rows"] == 3

    def test_sorting_orders_the_rows(self):
        batch = Batch.of(a=[3, 1, 2])
        assert sort.sort(batch, [sort.SortKey("a")]).column("a").to_list() == [1, 2, 3]

    def test_descending_reverses_it(self):
        batch = Batch.of(a=[3, 1, 2])
        ordered = sort.sort(batch, [sort.SortKey("a", descending=True)])
        assert ordered.column("a").to_list() == [3, 2, 1]


class TestJoining:
    def test_the_three_strategies_agree(self):
        result = joins.the_three_strategies_agree()
        assert result["hash_matches_merge"] and result["hash_matches_nested_loop"]

    def test_they_agree_with_the_reference(self):
        assert joins.they_agree_with_the_reference()["same"]

    def test_nulls_matched_nothing(self):
        assert joins.they_agree_with_the_reference()["nulls_matched_nothing"]

    def test_the_build_side_costs_the_same_in_total(self):
        assert joins.the_build_side_trades_probes_for_inserts()["the_total_operations_match"]

    def test_but_holds_less_from_the_small_side(self):
        result = joins.the_build_side_trades_probes_for_inserts()
        assert result["building_from_the_small_side_holds_less"]

    def test_and_gives_the_same_matches(self):
        assert joins.the_build_side_trades_probes_for_inserts()["same_matches"]

    def test_the_hash_join_pulls_away_from_the_nested_loop(self):
        rows = joins.the_hash_join_beats_the_nested_loop_by_the_build_size()
        ratios = [row["ratio"] for row in rows]
        assert ratios == sorted(ratios)

    def test_a_null_key_matches_nothing(self):
        assert joins.a_null_key_matches_nothing()["nothing_matched"]

    def test_a_composite_key_is_more_selective(self):
        assert joins.a_composite_key_joins_on_both_columns()["the_composite_is_more_selective"]

    def test_string_keys_join_across_dictionaries(self):
        assert joins.string_keys_join_across_dictionaries()["it_matched_on_text"]

    def test_and_the_dictionaries_really_differ(self):
        assert joins.string_keys_join_across_dictionaries()["the_dictionaries_differ"]

    def test_the_fanout_is_the_duplicate_count(self):
        rows = joins.the_fanout_is_what_a_cost_model_must_predict()
        assert all(row["fanout"] == float(row["duplicates"]) for row in rows)

    def test_an_empty_side_produces_nothing(self):
        assert joins.an_empty_side_produces_nothing()["it_produced_nothing"]

    def test_but_keeps_the_schema(self):
        assert joins.an_empty_side_produces_nothing()["the_schema_survived"]

    def test_mismatched_key_counts_are_refused(self):
        assert joins.mismatched_key_counts_are_refused()

    def test_mismatched_key_types_are_refused(self):
        assert joins.mismatched_key_types_are_refused()

    def test_a_join_with_no_keys_is_refused(self):
        assert joins.a_join_with_no_keys_is_refused()

    def test_an_oversized_nested_loop_is_refused(self):
        assert joins.an_oversized_nested_loop_is_refused()

    def test_an_unknown_key_is_refused(self):
        assert joins.an_unknown_key_is_refused()

    def test_a_joined_result_serialises(self):
        left = Batch.of(k=[1, 2], a=[1, 2])
        right = Batch.of(k=[1], b=[9])
        assert joins.hash_join(left, right, ["k"], ["k"]).as_dict()["matches"] == 1

    def test_a_bad_build_side_is_refused(self):
        left = Batch.of(k=[1], a=[1])
        right = Batch.of(k=[1], b=[1])
        with pytest.raises(ConfigError, match="not a side"):
            joins.hash_join(left, right, ["k"], ["k"], build_side="middle")

    def test_the_merge_join_emits_in_key_order(self):
        left = Batch.of(k=[3, 1, 2], a=[1, 2, 3])
        right = Batch.of(k=[1, 2, 3], b=[9, 8, 7])
        result = joins.merge_join(left, right, ["k"], ["k"])
        assert result.batch.column("k").to_list() == [1, 2, 3]


class TestSummaries:
    def test_every_operator_summarises(self):
        for module in (filtering, aggregate, sort, joins):
            assert isinstance(module.summarise(), dict)

    def test_the_filter_summary_names_the_crossover(self):
        assert filtering.summarise()["crossover"] == 0.5

    def test_the_aggregate_summary_confirms_agreement(self):
        assert aggregate.summarise()["agrees_with_the_reference"]

    def test_the_join_summary_confirms_agreement(self):
        assert joins.summarise()["agrees_with_the_reference"]
