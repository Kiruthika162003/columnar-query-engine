from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost import model
from cqe.cost.model import Costing, Estimate, cheapest, compare, estimate
from cqe.errors import ConfigError, PlanError
from cqe.exec.aggregate import Aggregate
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.exec.sort import SortKey
from cqe.plan.logical import Filter, Group, Join, Limit, Plan, Project, Sort, table
from cqe.plan.rules.pushdown import push_everything
from cqe.stats.cardinality import collect


@pytest.fixture(scope="module")
def catalogue() -> dict[str, Batch]:
    """Two tables to cost plans against."""
    state = np.random.default_rng(41)
    rows = 1200
    facts = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 25, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("label", [f"kind{one % 6}" for one in range(rows)]),
        ]
    )
    shops = Batch.from_columns(
        [
            integer_column("shop", np.arange(25)),
            string_column("region", [f"region{one % 4}" for one in range(25)]),
        ]
    )
    return {"facts": facts, "shops": shops}


def test_a_scan_costs_its_values():
    assert model.a_scan_costs_rows_times_columns()["it_is_the_product"]


def test_a_projection_adds_nothing():
    assert model.a_projection_costs_nothing()["the_projection_added_nothing"]


def test_pushing_a_projection_halves_a_four_column_scan():
    assert model.a_narrower_scan_costs_less()["it_halved"]


def test_a_filter_is_charged_for_both_halves():
    assert model.a_filter_costs_what_it_reads_and_what_it_writes()["it_is_the_sum"]


def test_the_writing_half_is_the_larger_one():
    assert model.a_filter_costs_what_it_reads_and_what_it_writes()["the_writing_is_larger"]


def test_a_filter_expects_to_keep_a_fraction():
    assert model.a_filter_costs_what_it_reads_and_what_it_writes()["it_expects_a_fraction"]


def test_a_join_is_dominated_by_its_probe():
    assert model.a_join_costs_a_build_and_a_probe()["the_probe_dominates_the_build"]


def test_a_join_is_the_dominant_node_of_its_plan():
    assert model.a_join_costs_a_build_and_a_probe()["dominant"] == "Join"


def test_a_sort_grows_faster_than_its_input():
    assert model.a_sort_costs_more_than_linear()["it_is_more_than_two"]


def test_a_sort_does_not_grow_quadratically():
    assert model.a_sort_costs_more_than_linear()["and_less_than_three"]


def test_the_group_estimate_is_capped():
    assert model.a_group_is_capped_at_its_input()["it_is_capped"]


def test_the_uncapped_estimate_would_exceed_the_rows():
    measured = model.a_group_is_capped_at_its_input()
    assert measured["uncapped_would_be"] > measured["rows_in"]


def test_the_model_and_the_meter_agree_on_pushdown():
    assert model.the_model_orders_two_plans_the_way_the_meter_does()["they_agree"]


def test_both_prefer_the_pushed_plan():
    measured = model.the_model_orders_two_plans_the_way_the_meter_does()
    assert measured["the_model_prefers_the_pushed_one"] and measured["and_so_does_the_meter"]


def test_the_ranking_is_exact_with_statistics():
    assert model.the_model_orders_a_set_of_plans()["with_statistics"] == 1.0


def test_the_ranking_is_worse_without_them():
    measured = model.the_model_orders_a_set_of_plans()
    assert measured["without_statistics"] < measured["with_statistics"]


def test_the_statistics_are_what_make_the_difference():
    assert model.the_model_orders_a_set_of_plans()["the_statistics_are_what_order_it"]


def test_the_bare_scan_costs_the_meter_nothing():
    assert model.the_model_orders_a_set_of_plans()["the_bare_scan_costs_the_meter_nothing"]


def test_the_model_and_the_meter_pick_the_same_working_plan():
    measured = model.the_model_orders_a_set_of_plans()
    assert measured["the_model_picks"] == measured["the_meter_picks"]


def test_the_absolute_numbers_have_a_spread():
    assert model.the_model_is_wrong_about_the_absolute_number()["spread"] > 1


def test_the_group_plan_is_the_most_understated():
    assert model.the_model_is_wrong_about_the_absolute_number()[
        "the_extremes_are_the_group_and_the_sort"
    ]


def test_every_working_plan_has_a_ratio():
    assert model.the_model_is_wrong_about_the_absolute_number()["plans_with_a_ratio"] == 5


def test_the_limit_hole_is_real():
    assert model.the_limit_hole_is_visible()["model_says_the_same"]


def test_the_meter_sees_through_the_limit_hole():
    assert model.the_limit_hole_is_visible()["the_meter_disagrees"]


def test_a_costing_names_a_dominant_node():
    assert model.a_costing_names_its_dominant_node()["it_named_one"]


def test_an_explain_has_a_line_per_node():
    assert model.a_costing_names_its_dominant_node()["the_explain_has_a_line_per_node"]


def test_choosing_between_no_plans_is_refused():
    assert model.an_empty_plan_list_is_refused()


def test_costing_an_unknown_node_is_refused():
    assert model.a_plan_with_no_cost_is_refused()


def test_the_constants_are_all_positive():
    assert all(one["value"] > 0 for one in model.compare_the_constants())


def test_the_summary_reports_the_unit():
    assert model.summarise()["unit"] == "values touched"


def test_a_scan_of_a_table_costs_rows_times_columns(catalogue):
    built = table("facts", catalogue["facts"])
    assert estimate(built).total == catalogue["facts"].rows * 4


def test_a_narrower_projection_is_cheaper(catalogue):
    wide = Project(input=table("facts", catalogue["facts"]), names=("id", "shop", "amount"))
    narrow = Project(input=table("facts", catalogue["facts"]), names=("id",))
    assert (
        estimate(push_everything(narrow).after).total
        < estimate(push_everything(wide).after).total
    )


def test_a_selective_filter_is_cheaper_than_a_loose_one(catalogue):
    stats = {"facts": collect(catalogue["facts"])}
    tight = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(150.0)),
    )
    loose = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(50.0)),
    )
    assert estimate(tight, stats).total < estimate(loose, stats).total


def test_without_statistics_they_cost_the_same(catalogue):
    tight = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(150.0)),
    )
    loose = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(50.0)),
    )
    assert estimate(tight).total == estimate(loose).total


def test_a_join_costs_more_than_either_input(catalogue):
    left = table("facts", catalogue["facts"])
    right = table("shops", catalogue["shops"])
    joined = Join(left=left, right=right, left_keys=("shop",), right_keys=("shop",))
    assert estimate(joined).total > estimate(left).total + estimate(right).total


def test_a_sort_costs_more_than_its_input(catalogue):
    base = table("facts", catalogue["facts"])
    sorted_plan = Sort(input=base, keys=(SortKey(name="amount"),))
    assert estimate(sorted_plan).total > estimate(base).total


def test_a_limit_adds_nothing(catalogue):
    base = table("facts", catalogue["facts"])
    limited = Limit(input=base, count=10)
    assert estimate(limited).total == estimate(base).total


def test_a_limit_reduces_the_row_estimate(catalogue):
    base = table("facts", catalogue["facts"])
    assert estimate(Limit(input=base, count=10)).rows == 10


def test_a_limit_beyond_the_rows_keeps_them_all(catalogue):
    base = table("facts", catalogue["facts"])
    assert estimate(Limit(input=base, count=999999)).rows == catalogue["facts"].rows


def test_a_group_reduces_the_row_estimate(catalogue):
    built = Group(
        input=table("facts", catalogue["facts"]),
        keys=("shop",),
        aggregates=(Aggregate(name="n", function="count_star", source=""),),
    )
    assert estimate(built).rows < catalogue["facts"].rows


def test_a_group_with_no_keys_is_one_row(catalogue):
    built = Group(
        input=table("facts", catalogue["facts"]),
        keys=(),
        aggregates=(Aggregate(name="n", function="count_star", source=""),),
    )
    assert estimate(built).rows == 1


def test_a_costing_sums_its_parts(catalogue):
    built = Sort(
        input=Filter(
            input=table("facts", catalogue["facts"]),
            predicate=Compare(">", column("amount"), literal(100.0)),
        ),
        keys=(SortKey(name="amount"),),
    )
    costed = estimate(built)
    assert costed.total == sum(one.cost for one in costed.parts)


def test_a_costing_counts_its_nodes(catalogue):
    built = Project(input=table("facts", catalogue["facts"]), names=("id",))
    assert estimate(built).nodes == 2


def test_a_costing_summarises(catalogue):
    summary = estimate(table("facts", catalogue["facts"])).as_dict()
    assert summary["nodes"] == 1 and summary["dominant"] == "Scan"


def test_an_empty_costing_has_no_dominant_node():
    assert Costing(total=0.0, rows=0.0, parts=()).dominant() == ""


def test_asking_for_an_absent_node_is_zero(catalogue):
    assert estimate(table("facts", catalogue["facts"])).of("Join") == 0


def test_an_estimate_describes_itself():
    line = Estimate(node="Scan", rows=10.0, cost=40.0).describe()
    assert "Scan" in line and "40" in line


def test_an_estimate_summarises():
    assert Estimate(node="Scan", rows=10.0, cost=40.0).as_dict()["cost"] == 40


def test_cheapest_picks_the_lowest(catalogue):
    base = table("facts", catalogue["facts"])
    narrow = push_everything(Project(input=base, names=("id",))).after
    assert cheapest([base, narrow]) == 1


def test_cheapest_of_one_is_that_one(catalogue):
    assert cheapest([table("facts", catalogue["facts"])]) == 0


def test_cheapest_of_none_is_refused():
    with pytest.raises(ConfigError):
        cheapest([])


def test_comparing_a_plan_reports_both_numbers(catalogue):
    built = Project(
        input=Filter(
            input=table("facts", catalogue["facts"]),
            predicate=Compare(">", column("amount"), literal(100.0)),
        ),
        names=("id", "amount"),
    )
    measured = compare(built, catalogue)
    assert measured["predicted"] > 0 and measured["counted"] > 0


def test_comparing_a_plan_reports_the_row_counts(catalogue):
    built = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(100.0)),
    )
    measured = compare(built, catalogue)
    assert measured["rows_counted"] > 0


def test_costing_a_strange_node_is_refused():
    @dataclass(frozen=True)
    class Strange(Plan):
        def schema(self):
            return None

    with pytest.raises(PlanError):
        estimate(Strange())
