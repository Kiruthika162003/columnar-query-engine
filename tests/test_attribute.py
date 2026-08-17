from __future__ import annotations

from pathlib import Path

import pytest

from cqe.errors import ConfigError
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.plan import attribute as charging
from cqe.plan.attribute import (
    TOLERABLE,
    Attribution,
    NodeCost,
    attribute,
)
from cqe.plan.logical import Aggregate, Filter, Group, Limit, Project, Sort, SortKey, table


@pytest.fixture(scope="module")
def catalogue() -> dict[str, Batch]:
    """Two tables to attribute plans against."""
    return charging._tables(4_000)


def test_a_subtree_costs_the_same_twice():
    assert charging.a_subtree_costs_the_same_twice()["they_all_agree"]


def test_the_repeat_check_covers_every_node():
    assert charging.a_subtree_costs_the_same_twice()["nodes"] == 7


def test_a_limit_costs_a_negative_amount():
    assert charging.a_node_can_cost_less_than_nothing()["the_limit_is_negative"]


def test_the_negative_node_still_sums_to_the_plan():
    assert charging.a_node_can_cost_less_than_nothing()["and_they_still_sum_to_the_plan"]


def test_the_limit_is_not_separable():
    assert charging.a_node_can_cost_less_than_nothing()["the_limit_is_not_separable"]


def test_the_other_nodes_still_are():
    made = charging.a_node_can_cost_less_than_nothing()
    assert made["separable_nodes"] == made["of"] - 1


def test_the_total_is_out_by_about_two():
    assert charging.the_total_hides_the_per_node_error()["the_total_is_out_by_a_factor_of_two"]


def test_the_worst_node_is_out_by_thousands():
    assert charging.the_total_hides_the_per_node_error()[
        "while_the_worst_node_is_out_by_thousands"
    ]


def test_the_total_is_far_closer_than_the_worst_node():
    assert charging.the_total_hides_the_per_node_error()[
        "the_total_is_far_closer_than_the_worst_node"
    ]


def test_some_nodes_are_right():
    assert charging.the_total_hides_the_per_node_error()["right_nodes"]


def test_every_scan_is_charged_nothing():
    assert charging.the_model_and_the_meter_disagree_about_what_a_scan_is()[
        "every_scan_is_charged_nothing"
    ]


def test_the_model_charges_for_every_scan():
    assert charging.the_model_and_the_meter_disagree_about_what_a_scan_is()[
        "while_the_model_charges_for_all_of_them"
    ]


def test_the_worst_node_is_a_scan():
    assert charging.the_model_and_the_meter_disagree_about_what_a_scan_is()[
        "the_worst_node_is_a_scan"
    ]


def test_the_model_looks_better_without_the_scans():
    assert charging.the_model_and_the_meter_disagree_about_what_a_scan_is()[
        "the_model_looks_better_without_them"
    ]


def test_the_row_and_cost_errors_move_together():
    assert charging.the_row_error_arrives_before_the_cost_error()["they_move_together"]


def test_the_correlation_is_reported():
    assert charging.the_row_error_arrives_before_the_cost_error()["correlation"] > 0.5


def test_the_leaves_have_no_row_error():
    assert charging.the_error_compounds_up_the_tree()["the_leaves_are_exact"]


def test_the_row_error_grows_on_the_way_up():
    assert charging.the_error_compounds_up_the_tree()["it_grows_on_the_way_up"]


def test_the_root_is_a_limit():
    assert charging.the_error_compounds_up_the_tree()["the_root_is_a_limit"]


def test_the_limit_makes_the_root_exact_again():
    assert charging.the_error_compounds_up_the_tree()["and_the_root_is_exact_again"]


def test_the_root_hides_the_peak():
    assert charging.the_error_compounds_up_the_tree()["so_the_root_hides_it"]


def test_the_peak_is_not_at_the_root():
    made = charging.the_error_compounds_up_the_tree()
    assert made["peak_at_depth"] > 0


def test_the_model_usually_names_the_dominant_node():
    assert charging.the_model_names_the_wrong_dominant_node()["it_is_usually_right"]


def test_but_not_always():
    assert charging.the_model_names_the_wrong_dominant_node()["but_not_always"]


def test_the_dominant_comparison_skips_free_plans():
    made = charging.the_model_names_the_wrong_dominant_node()
    assert "project" not in [one["plan"] for one in made["plans"]]


def test_a_limit_makes_the_plan_cheaper():
    assert charging.the_limit_hole_is_a_pushdown_the_method_cannot_see()[
        "the_limit_makes_the_plan_cheaper"
    ]


def test_the_partial_sort_is_several_times_cheaper():
    assert charging.the_limit_hole_is_a_pushdown_the_method_cannot_see()["by_this_factor"] > 2


def test_the_limit_absorbs_the_difference():
    assert charging.the_limit_hole_is_a_pushdown_the_method_cannot_see()[
        "the_limit_absorbs_the_difference"
    ]


def test_the_pushdown_makes_the_limit_inseparable():
    assert charging.the_limit_hole_is_a_pushdown_the_method_cannot_see()[
        "which_makes_it_inseparable"
    ]


def test_a_projection_is_free_on_both_sides():
    assert charging.a_projection_costs_nothing_and_is_charged_nothing()["both_are_nothing"]


def test_the_scan_below_a_projection_is_free_too():
    assert charging.a_projection_costs_nothing_and_is_charged_nothing()[
        "the_scan_below_is_free_too"
    ]


def test_the_model_still_charges_for_that_scan():
    assert charging.a_projection_costs_nothing_and_is_charged_nothing()[
        "but_the_model_charges_for_it"
    ]


def test_a_projection_over_a_scan_runs_for_nothing():
    assert charging.a_projection_costs_nothing_and_is_charged_nothing()[
        "the_whole_plan_runs_for_nothing"
    ]


def test_a_subtree_cost_is_not_a_node_cost():
    assert charging.a_subtree_cost_is_not_a_node_cost()["the_children_are_a_large_share"]


def test_reporting_the_subtree_would_overstate():
    made = charging.a_subtree_cost_is_not_a_node_cost()
    assert made["reporting_the_subtree_would_overstate_by"] > 1.5


def test_attribution_costs_more_than_one_run():
    assert charging.attribution_costs_a_run_per_node()["it_costs_more_than_one_run"]


def test_the_extra_cost_is_a_factor_of_a_few():
    assert charging.attribution_costs_a_run_per_node()["by_this_factor"] > 2


def test_the_answer_is_still_one_run_of_work():
    assert charging.attribution_costs_a_run_per_node()[
        "and_the_answer_is_still_one_run_of_work"
    ]


def test_an_uncostable_plan_is_refused():
    assert charging.a_plan_the_model_cannot_cost_is_refused()


def test_an_empty_attribution_is_refused():
    assert charging.an_empty_attribution_is_refused()


def test_the_plan_table_covers_eight_plans():
    assert len(charging.compare_the_plans()) == 8


def test_every_plan_in_the_table_is_named():
    assert all(one["plan"] for one in charging.compare_the_plans())


def test_every_worst_node_is_a_scan_before_they_are_removed():
    assert all(one["worst"] == "Scan" for one in charging.compare_the_plans())


def test_the_worst_nodes_differ_once_the_scans_are_removed():
    names = {one["worst_without_scans"] for one in charging.compare_the_plans()}
    assert len(names) > 3


def test_the_root_looks_better_than_its_worst_node():
    assert charging.the_root_flatters_the_model()["the_root_looks_better"]


def test_one_plan_has_a_root_worse_than_its_parts():
    assert charging.the_root_flatters_the_model()["plans_where_the_root_is_worse"] == 1


def test_the_summary_reports_the_negative_node():
    assert charging.summarise()["a_limit_costs_a_negative_amount"]


def test_the_summary_reports_the_free_scan():
    assert charging.summarise()["a_scan_is_charged_nothing"]


def test_the_summary_reports_the_subtraction_is_repeatable():
    assert charging.summarise()["the_subtraction_is_repeatable"]


def test_attributing_gives_a_node_per_plan_node(catalogue):
    plan = charging._deep(catalogue)
    assert len(attribute(plan, catalogue).parts) == len(plan.children()) + 6


def test_the_root_is_first(catalogue):
    made = attribute(charging._deep(catalogue), catalogue)
    assert made.parts[0].depth == 0


def test_the_parts_are_ordered_by_depth(catalogue):
    made = attribute(charging._deep(catalogue), catalogue)
    depths = [one.depth for one in made.parts]
    assert depths == sorted(depths)


def test_an_attribution_totals_its_parts(catalogue):
    made = attribute(charging._deep(catalogue), catalogue)
    assert made.counted == sum(one.counted for one in made.parts)


def test_an_attribution_names_its_worst(catalogue):
    made = attribute(charging._deep(catalogue), catalogue)
    assert made.worst.node in {one.node for one in made.parts}


def test_an_attribution_names_its_dearest(catalogue):
    made = attribute(charging._deep(catalogue), catalogue)
    assert made.dearest.counted == max(one.counted for one in made.parts)


def test_an_attribution_names_the_predicted_dearest(catalogue):
    made = attribute(charging._deep(catalogue), catalogue)
    assert made.predicted_dearest.predicted == max(one.predicted for one in made.parts)


def test_an_attribution_finds_a_node_kind(catalogue):
    made = attribute(charging._deep(catalogue), catalogue)
    assert len(made.of("Scan")) == 2


def test_an_attribution_explains_itself(catalogue):
    made = attribute(charging._deep(catalogue), catalogue)
    assert len(made.explain().splitlines()) == len(made.parts)


def test_an_attribution_summarises(catalogue):
    made = attribute(charging._deep(catalogue), catalogue)
    assert made.as_dict()["nodes"] == len(made.parts)


def test_an_attribution_lists_its_separable_nodes(catalogue):
    ordered = Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="amount"),))
    made = attribute(Limit(input=ordered, count=10), catalogue)
    assert len(made.separable) == 2


def test_an_attribution_lists_its_dependent_nodes(catalogue):
    ordered = Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="amount"),))
    made = attribute(Limit(input=ordered, count=10), catalogue)
    assert [one.node for one in made.dependent] == ["Limit"]


def test_the_worst_ignores_a_dependent_node(catalogue):
    ordered = Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="amount"),))
    made = attribute(Limit(input=ordered, count=10), catalogue)
    assert made.worst.node != "Limit"


def test_a_single_node_plan_attributes(catalogue):
    made = attribute(table("facts", catalogue["facts"]), catalogue)
    assert len(made.parts) == 1


def test_a_filter_is_charged_what_it_touches(catalogue):
    plan = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(90.0)),
    )
    assert attribute(plan, catalogue).of("Filter")[0].counted > 0


def test_a_group_is_charged_what_it_probes(catalogue):
    plan = Group(
        input=table("facts", catalogue["facts"]),
        keys=("shop",),
        aggregates=(Aggregate(name="total", function="sum", source="amount"),),
    )
    assert attribute(plan, catalogue).of("Group")[0].counted > 0


def test_a_project_is_charged_nothing(catalogue):
    plan = Project(input=table("facts", catalogue["facts"]), names=("id",))
    assert attribute(plan, catalogue).of("Project")[0].counted == 0


def test_a_node_cost_reports_its_ratio():
    made = NodeCost(
        node="Filter", depth=0, predicted=200, counted=100, predicted_rows=10, counted_rows=10
    )
    assert made.ratio == 2.0


def test_a_node_cost_that_agrees_at_zero_has_a_ratio_of_one():
    made = NodeCost(
        node="Project", depth=0, predicted=0, counted=0, predicted_rows=10, counted_rows=10
    )
    assert made.ratio == 1.0


def test_a_node_cost_reports_its_row_ratio():
    made = NodeCost(
        node="Filter", depth=0, predicted=100, counted=100, predicted_rows=20, counted_rows=10
    )
    assert made.row_ratio == 2.0


def test_an_overestimate_and_an_underestimate_have_the_same_error():
    over = NodeCost(
        node="a", depth=0, predicted=400, counted=100, predicted_rows=1, counted_rows=1
    )
    under = NodeCost(
        node="a", depth=0, predicted=25, counted=100, predicted_rows=1, counted_rows=1
    )
    assert over.error == under.error


def test_a_node_within_the_tolerance_is_not_wrong():
    made = NodeCost(
        node="a", depth=0, predicted=150, counted=100, predicted_rows=1, counted_rows=1
    )
    assert not made.wrong


def test_a_node_outside_the_tolerance_is_wrong():
    made = NodeCost(
        node="a", depth=0, predicted=300, counted=100, predicted_rows=1, counted_rows=1
    )
    assert made.wrong


def test_a_negative_node_is_not_independent():
    made = NodeCost(
        node="a", depth=0, predicted=0, counted=-5, predicted_rows=1, counted_rows=1
    )
    assert not made.independent


def test_a_negative_node_has_an_infinite_error():
    made = NodeCost(
        node="a", depth=0, predicted=0, counted=-5, predicted_rows=1, counted_rows=1
    )
    assert made.error == float("inf")


def test_a_node_cost_describes_itself_with_its_depth():
    made = NodeCost(
        node="Sort", depth=2, predicted=10, counted=10, predicted_rows=1, counted_rows=1
    )
    assert made.describe().startswith("    Sort")


def test_a_node_cost_summarises():
    made = NodeCost(
        node="Sort", depth=2, predicted=10, counted=10, predicted_rows=1, counted_rows=1
    )
    assert made.as_dict()["node"] == "Sort"


def test_an_attribution_with_no_parts_is_refused():
    with pytest.raises(ConfigError):
        Attribution(parts=())


def test_the_tolerance_is_a_factor_of_two():
    assert TOLERABLE == 2.0


def test_the_module_does_not_write_files():
    before = set(Path().glob("*"))
    charging.summarise(2_000)
    assert set(Path().glob("*")) == before
