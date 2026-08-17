from __future__ import annotations

import pytest

from cqe.errors import PlanError, UnknownColumn
from cqe.exec.aggregate import Aggregate
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.exec.sort import SortKey
from cqe.plan import logical
from cqe.plan.logical import Filter, Group, Scan, Sort, table
from cqe.plan.rules import ordering, pruning, pushdown


def sample() -> Batch:
    """The batch most plan tests build on."""
    return Batch.of(a=[1, 2, 3, 4], g=["x", "y", "x", "z"], v=[1.0, 2.0, 3.0, 4.0])


class TestLogicalPlan:
    def test_every_node_knows_its_schema(self):
        assert logical.a_plan_knows_its_schema()["the_project_narrowed"]

    def test_a_group_renames_its_outputs(self):
        assert logical.a_plan_knows_its_schema()["the_group_renamed"]

    def test_a_join_suffixes_repeated_names(self):
        assert logical.a_join_schema_disambiguates()["it_suffixed"]

    def test_and_keeps_the_left_names(self):
        assert logical.a_join_schema_disambiguates()["the_left_kept_its_names"]

    def test_a_transform_visits_children_first(self):
        assert logical.transform_rewrites_children_first()["the_scan_came_first"]

    def test_and_visits_every_node(self):
        assert logical.transform_rewrites_children_first()["every_node_was_visited"]

    def test_a_plan_renders_one_line_per_node(self):
        assert logical.a_plan_renders_as_a_tree()["it_is_one_line_per_node"]

    def test_and_shows_the_predicate(self):
        assert logical.a_plan_renders_as_a_tree()["the_predicate_is_shown"]

    def test_a_scan_keeps_pushed_predicates_separate(self):
        assert logical.a_scan_records_what_was_pushed_into_it()["they_are_separate"]

    def test_and_its_schema_is_unchanged(self):
        result = logical.a_scan_records_what_was_pushed_into_it()
        assert result["the_schema_is_unchanged"]

    def test_a_scan_narrows(self):
        assert logical.a_scan_narrows_its_projection()["it_narrowed"]

    def test_a_join_swap_changes_the_column_order(self):
        assert logical.a_join_swaps()["the_order_changed"]

    def test_but_keeps_the_same_columns(self):
        assert logical.a_join_swaps()["the_same_columns_are_present"]

    def test_columns_used_finds_what_a_node_reads(self):
        assert logical.columns_used_finds_what_a_node_reads()["the_filter_reads_one"]

    def test_and_the_group_reads_two(self):
        assert logical.columns_used_finds_what_a_node_reads()["the_group_reads_two"]

    def test_a_scan_reports_the_columns_its_predicates_read(self):
        batch = sample()
        scan = table("t", batch).with_predicate(Compare("<", column("a"), literal(3)))
        assert scan.columns_used() == {"a"}

    def test_a_scan_with_nothing_pushed_reads_nothing(self):
        assert table("t", sample()).columns_used() == frozenset()

    def test_a_plan_counts_its_shape(self):
        assert logical.a_plan_counts_its_own_shape()["they_agree"]

    def test_a_broken_rewrite_is_caught(self):
        assert logical.a_rewrite_that_drops_a_column_is_caught()["the_bad_plan_is_refused"]


class TestPlanRefusals:
    def test_a_missing_column_is_refused(self):
        assert logical.a_plan_referring_to_a_missing_column_is_refused()

    def test_a_missing_join_key_is_refused(self):
        assert logical.a_join_on_a_missing_key_is_refused()

    def test_mismatched_join_keys_are_refused(self):
        assert logical.mismatched_join_keys_are_refused()

    def test_an_empty_projection_is_refused(self):
        assert logical.an_empty_projection_is_refused()

    def test_a_group_with_no_aggregates_is_refused(self):
        assert logical.a_group_with_no_aggregates_is_refused()

    def test_a_negative_limit_is_refused(self):
        assert logical.a_negative_limit_is_refused()

    def test_a_nameless_scan_is_refused(self):
        assert logical.a_nameless_scan_is_refused()

    def test_rebuilding_a_scan_with_children_is_refused(self):
        assert logical.rebuilding_a_scan_with_children_is_refused()

    def test_an_impossible_plan_is_refused(self):
        assert logical.an_impossible_plan_is_refused()

    def test_a_group_on_a_missing_source_is_refused(self):
        with pytest.raises(UnknownColumn):
            Group(
                input=table("t", sample()),
                keys=("g",),
                aggregates=(Aggregate("total", "sum", "missing"),),
            )

    def test_a_sort_on_a_missing_key_is_refused(self):
        with pytest.raises(UnknownColumn):
            Sort(input=table("t", sample()), keys=(SortKey("missing"),))

    def test_a_sort_with_no_keys_is_refused(self):
        with pytest.raises(PlanError, match="at least one key"):
            Sort(input=table("t", sample()), keys=())

    def test_a_scan_projecting_a_missing_column_is_refused(self):
        with pytest.raises(UnknownColumn):
            Scan(name="t", table_schema=sample().schema, projected=("missing",))


class TestPushdown:
    def test_a_predicate_moves_below_a_join(self):
        assert pushdown.a_predicate_below_a_join_is_the_largest_win()["it_helped"]

    def test_and_the_answer_survives(self):
        assert pushdown.a_predicate_below_a_join_is_the_largest_win()["same_rows"]

    def test_a_conjunction_splits(self):
        assert pushdown.a_conjunction_splits_and_only_part_moves()["one_stayed_behind"]

    def test_and_the_answer_survives_that_too(self):
        assert pushdown.a_conjunction_splits_and_only_part_moves()["same_rows"]

    def test_a_predicate_on_a_grouping_key_moves(self):
        assert pushdown.a_predicate_on_an_aggregate_cannot_move()["the_key_one_moved"]

    def test_a_predicate_on_an_aggregate_does_not(self):
        assert pushdown.a_predicate_on_an_aggregate_cannot_move()["the_result_one_did_not"]

    def test_a_projection_narrows_every_scan(self):
        result = pushdown.a_projection_narrows_every_scan()
        assert result["columns_after"] < result["columns_before"]

    def test_and_the_projection_answer_survives(self):
        assert pushdown.a_projection_narrows_every_scan()["same_rows"]

    def test_the_rule_order_does_not_matter(self):
        result = pushdown.the_order_of_the_two_rules_does_not_matter()
        assert result["right_way_values"] == result["other_way_values"]

    def test_both_orders_push_the_predicate(self):
        result = pushdown.the_order_of_the_two_rules_does_not_matter()
        assert result["right_way_pushed_the_predicate"] and result["other_way_pushed_it"]

    def test_pushing_through_a_sort_is_safe(self):
        assert pushdown.pushing_through_a_sort_is_safe()["same_rows"]

    def test_and_helps(self):
        assert pushdown.pushing_through_a_sort_is_safe()["it_helped"]

    def test_pushing_through_a_limit_is_refused(self):
        assert pushdown.pushing_through_a_limit_is_refused()["nothing_moved"]

    def test_and_the_answers_match(self):
        assert pushdown.pushing_through_a_limit_is_refused()["the_answers_match"]

    def test_nothing_moves_when_nothing_can(self):
        assert pushdown.nothing_moves_when_nothing_can()["the_plan_is_identical"]

    def test_a_rewrite_keeps_the_schema(self):
        assert pushdown.a_rewrite_never_changes_the_schema()["they_match"]

    def test_every_rule_is_correct(self):
        rows = pushdown.compare_the_rules()
        assert all(row["correct"] for row in rows)

    def test_both_rules_together_are_cheapest(self):
        rows = pushdown.compare_the_rules()
        assert min(rows, key=lambda row: row["values"])["rule"] == "both"

    def test_a_broken_rewrite_is_refused(self):
        assert pushdown.a_rewrite_that_breaks_a_plan_is_refused()

    def test_an_unknown_node_cannot_be_evaluated(self):
        assert pushdown.an_unknown_node_cannot_be_evaluated()

    def test_an_impossible_table_is_refused(self):
        assert pushdown.an_impossible_table_is_refused()

    def test_rebuilding_a_join_as_a_single_input_is_refused(self):
        assert pushdown.rebuilding_a_join_as_a_single_input_is_refused()

    def test_a_rewrite_serialises(self):
        plan = Filter(
            input=table("t", sample()), predicate=Compare("<", column("a"), literal(3))
        )
        assert pushdown.push_predicates(plan).as_dict()["moved"] == 1


class TestOrdering:
    def test_the_order_changes_the_work(self):
        assert ordering.the_order_changes_the_work_by_a_lot()["ratio"] > 1.0

    def test_every_order_agrees(self):
        assert ordering.the_order_changes_the_work_by_a_lot()["every_order_agrees"]

    def test_six_orders_of_three_dimensions(self):
        assert ordering.the_order_changes_the_work_by_a_lot()["orders"] == 6

    def test_the_containment_estimate_chooses_badly(self):
        assert ordering.the_greedy_rule_finds_a_good_one()["share_of_the_saving"] < 0.5

    def test_the_corrected_estimate_chooses_better(self):
        result = ordering.correcting_the_estimate_fixes_the_choice()
        assert (
            result["corrected_share_of_the_saving"] > result["containment_share_of_the_saving"]
        )

    def test_and_reaches_most_of_the_saving(self):
        result = ordering.correcting_the_estimate_fixes_the_choice()
        assert result["corrected_share_of_the_saving"] > 0.5

    def test_the_correction_never_hurts_here(self):
        assert ordering.correcting_the_estimate_fixes_the_choice()["the_correction_helps"]

    def test_the_smallest_dimension_is_not_the_most_selective(self):
        assert not ordering.the_smallest_dimension_goes_first()["they_agree"]

    def test_a_sort_under_a_limit_is_found(self):
        assert ordering.a_sort_under_a_limit_becomes_a_top_k()["fused"] == 1

    def test_and_the_top_k_moves_far_less(self):
        assert ordering.a_sort_under_a_limit_becomes_a_top_k()["ratio"] > 2.0

    def test_a_useless_sort_is_dropped(self):
        assert ordering.drop_a_sort_nothing_reads()["the_sort_is_gone"]

    def test_and_the_answer_survives(self):
        assert ordering.drop_a_sort_nothing_reads()["same_rows"]

    def test_a_sorted_aggregate_needs_its_sort(self):
        result = ordering.a_sort_under_a_sorted_aggregate_stays()
        assert result["the_sorted_form_needs_the_sort"]

    def test_every_join_order_gives_the_same_rows(self):
        assert ordering.every_order_gives_the_same_answer()["all_agree"]

    def test_an_empty_dimension_list_is_refused(self):
        assert ordering.an_empty_dimension_list_is_refused()

    def test_an_impossible_star_is_refused(self):
        assert ordering.an_impossible_star_is_refused()

    def test_the_orders_sort_by_cost(self):
        values = [row["values"] for row in ordering.compare_the_orders()]
        assert values == sorted(values)


class TestPruning:
    def test_the_whole_chain_pays(self):
        assert pruning.the_whole_chain_pays_or_none_of_it_does()["ratio"] > 10

    def test_and_the_answer_survives(self):
        assert pruning.the_whole_chain_pays_or_none_of_it_does()["same_rows"]

    def test_a_predicate_that_did_not_push_prunes_nothing(self):
        result = pruning.a_predicate_that_did_not_push_prunes_nothing()
        assert result["nothing_was_skipped"]

    def test_a_shuffled_table_prunes_nothing(self):
        assert pruning.a_shuffled_table_prunes_nothing()["nothing_was_skipped"]

    def test_though_the_predicate_reached_the_reader(self):
        result = pruning.a_shuffled_table_prunes_nothing()
        assert result["the_predicate_did_reach_the_reader"]

    def test_a_failed_prune_costs_something(self):
        assert pruning.a_failed_prune_costs_the_statistics()["it_is_not_zero"]

    def test_but_not_much(self):
        assert pruning.a_failed_prune_costs_the_statistics()["it_is_small"]

    def test_the_saving_follows_the_selectivity(self):
        rows = pruning.the_selectivity_sets_the_saving()
        savings = [row["saving"] for row in rows]
        assert savings == sorted(savings, reverse=True)

    def test_pruning_and_projection_multiply(self):
        assert pruning.pruning_and_projection_multiply()["they_multiply"]

    def test_finer_groups_prune_more(self):
        rows = pruning.the_group_size_changes_what_prunes()
        assert rows[0]["skipped_share"] > rows[-1]["skipped_share"]

    def test_the_total_cost_has_an_interior_minimum(self):
        rows = pruning.the_group_size_changes_what_prunes()
        cheapest = min(rows, key=lambda row: row["total"])
        assert cheapest["group_size"] not in (rows[0]["group_size"], rows[-1]["group_size"])

    def test_the_answer_is_the_same_either_way(self):
        assert all(pruning.the_answer_is_the_same_either_way().values())

    def test_a_predicate_matching_nothing_skips_everything(self):
        result = pruning.a_predicate_matching_nothing_skips_everything()
        assert result["everything_was_skipped"]

    def test_and_reads_no_data_at_all(self):
        result = pruning.a_predicate_matching_nothing_skips_everything()
        assert result["only_statistics_were_read"]

    def test_conjuncts_prune_independently(self):
        assert pruning.conjuncts_prune_independently()["both_prune_more"]

    def test_and_the_conjunct_answer_survives(self):
        assert pruning.conjuncts_prune_independently()["same_rows"]

    def test_a_reader_without_predicates_reads_everything(self):
        assert pruning.a_reader_without_predicates_reads_everything()["it_read_everything"]

    def test_an_unknown_predicate_shape_is_safe(self):
        assert pruning.an_unknown_predicate_shape_is_safe()["nothing_skipped"]

    def test_and_gives_the_same_rows(self):
        assert pruning.an_unknown_predicate_shape_is_safe()["same_rows"]

    def test_a_two_column_predicate_reaches_the_scan(self):
        assert pruning.an_unpushable_predicate_stays_above()["it_reached_the_scan"]

    def test_but_prunes_nothing(self):
        assert pruning.an_unpushable_predicate_stays_above()["it_pruned_nothing"]

    def test_a_stored_table_with_no_groups_is_refused(self):
        assert pruning.a_stored_table_with_no_groups_is_refused()

    def test_mismatched_groups_and_statistics_are_refused(self):
        assert pruning.mismatched_groups_and_statistics_are_refused()

    def test_a_zero_group_size_is_refused(self):
        assert pruning.a_zero_group_size_is_refused()

    def test_an_impossible_table_is_refused(self):
        assert pruning.an_impossible_table_is_refused()

    def test_a_plan_without_a_scan_yields_no_predicates(self):
        assert pruning.a_plan_without_a_scan_has_nothing_to_prune()

    def test_a_clustered_layout_beats_a_shuffled_one(self):
        rows = {(row["layout"], row["share"]): row for row in pruning.compare_the_layouts()}
        assert rows[("clustered", 0.01)]["values"] < rows[("shuffled", 0.01)]["values"]

    def test_a_stored_table_serialises(self):
        stored = pruning.store("t", Batch.of(a=list(range(100))), group_size=25)
        assert stored.as_dict()["groups"] == 4


class TestSummaries:
    def test_every_plan_module_summarises(self):
        for module in (logical, pushdown, ordering, pruning):
            assert isinstance(module.summarise(), dict)

    def test_the_pushdown_summary_confirms_correctness(self):
        assert pushdown.summarise()["every_rule_is_correct"]

    def test_the_ordering_summary_confirms_the_correction(self):
        assert ordering.summarise()["the_correction_helps"]

    def test_the_pruning_summary_confirms_the_answer(self):
        assert pruning.summarise()["the_answer_survived"]

    def test_every_rule_list_is_callable(self):
        for module in (pushdown, ordering, pruning):
            assert all(callable(rule) for rule in module.rules())
