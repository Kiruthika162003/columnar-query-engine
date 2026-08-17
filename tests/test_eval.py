from __future__ import annotations

import pytest

from cqe.errors import ConfigError
from cqe.eval import workload
from cqe.eval.workload import QUERIES, Measurement, catalogue, measure, measure_all, named


def test_the_whole_set_runs():
    assert workload.the_whole_set_runs()["queries"] == len(QUERIES)


def test_two_queries_cost_the_meter_nothing():
    assert workload.the_whole_set_runs()["free_queries"] == 2


def test_the_working_queries_have_a_real_spread():
    assert workload.the_whole_set_runs()["spread"] > 2


def test_the_cheapest_query_is_the_aggregate():
    assert workload.the_whole_set_runs()["cheapest"] == "aggregate"


def test_the_dearest_query_joins_and_groups():
    assert workload.the_whole_set_runs()["dearest"] == "join and group"


def test_the_rewrite_helps_several_queries():
    assert workload.the_rewrite_helps_most_queries()["helped"] >= 3


def test_the_rewrite_never_hurts():
    assert workload.the_rewrite_helps_most_queries()["it_never_hurts"]


def test_the_conjunction_gains_most_from_the_rewrite():
    assert workload.the_rewrite_helps_most_queries()["best"] == "conjunction"


def test_every_query_agrees_with_the_reference():
    assert workload.every_query_agrees_with_the_reference()["they_all_agree"]


def test_seven_queries_were_checked():
    assert workload.every_query_agrees_with_the_reference()["checked"] == 7


def test_a_point_lookup_costs_about_what_a_range_does():
    assert workload.a_point_lookup_is_the_worst_case()["the_point_lookup_costs_the_same"]


def test_a_point_lookup_returns_far_fewer_rows():
    assert workload.a_point_lookup_is_the_worst_case()["but_returns_far_fewer_rows"]


def test_the_cost_per_row_is_far_worse_for_a_point_lookup():
    measured = workload.a_point_lookup_is_the_worst_case()
    assert measured["cost_per_row_point"] > measured["cost_per_row_range"] * 100


def test_a_layout_fixes_the_point_lookup():
    assert workload.a_point_lookup_on_disk_is_much_better()["the_layout_fixed_it"]


def test_the_sorted_file_reads_far_fewer_bytes():
    assert workload.a_point_lookup_on_disk_is_much_better()["ratio"] > 10


def test_the_model_ranks_most_pairs_correctly():
    assert workload.the_model_ranks_the_set()["share"] > 0.6


def test_the_model_ratios_have_a_spread():
    assert workload.the_model_ranks_the_set()["spread"] > 1


def test_the_costs_are_linear_in_the_rows():
    assert workload.the_costs_scale_with_the_rows()["most_grew_about_fourfold"]


def test_the_rows_really_grew_fourfold():
    assert workload.the_costs_scale_with_the_rows()["the_rows_grew_fourfold"]


def test_the_top_ten_query_is_linear_too():
    growth = workload.the_costs_scale_with_the_rows()["growth"]
    assert 3.5 < growth["top ten"] < 4.5


def test_a_file_reads_fewer_rows_than_memory_touches():
    assert workload.a_query_over_a_file_costs_less_than_over_memory()[
        "the_file_read_fewer_rows"
    ]


def test_the_file_and_the_batch_agree():
    assert workload.a_query_over_a_file_costs_less_than_over_memory()["they_agree"]


def test_the_set_reaches_a_join():
    assert workload.the_queries_reach_every_strategy()["it_reaches_a_join"]


def test_the_set_reaches_two_aggregate_strategies():
    assert workload.the_queries_reach_every_strategy()["and_two_aggregates"]


def test_the_set_reaches_a_partial_sort():
    assert workload.the_queries_reach_every_strategy()["and_a_partial_sort"]


def test_an_unknown_query_is_refused():
    assert workload.an_unknown_query_is_refused()


def test_the_query_table_is_sorted_by_cost():
    totals = [one["total"] for one in workload.compare_the_queries()]
    assert totals == sorted(totals)


def test_the_summary_says_they_all_agree():
    assert workload.summarise()["all_agree"]


def test_every_query_has_a_name():
    assert all(one.name for one in QUERIES)


def test_every_query_says_what_it_exercises():
    assert all(one.exercises for one in QUERIES)


def test_every_query_summarises():
    assert all(one.as_dict()["query"] for one in QUERIES)


def test_a_query_can_be_looked_up_by_name():
    assert named("point").name == "point"


def test_an_unknown_name_lists_the_others():
    with pytest.raises(ConfigError) as caught:
        named("nothing")
    assert "point" in str(caught.value)


def test_the_catalogue_has_two_tables():
    assert set(catalogue(1000)) == {"facts", "shops"}


def test_the_fact_table_has_five_columns():
    assert catalogue(1000)["facts"].width == 5


def test_the_dimension_has_the_join_key():
    assert "shop" in catalogue(1000)["shops"].schema


def test_measuring_one_query_returns_a_measurement():
    tables = catalogue(2000)
    assert isinstance(measure(named("range"), tables), Measurement)


def test_a_measurement_reports_its_rows():
    tables = catalogue(2000)
    assert measure(named("scan"), tables).rows == 2000


def test_a_measurement_reports_a_total():
    tables = catalogue(2000)
    assert measure(named("range"), tables).total > 0


def test_a_measurement_summarises():
    tables = catalogue(2000)
    assert measure(named("range"), tables).as_dict()["query"] == "range"


def test_a_measurement_reports_its_ratio():
    tables = catalogue(2000)
    assert measure(named("range"), tables).ratio > 0


def test_measuring_everything_covers_the_set():
    assert len(measure_all(2000)) == len(QUERIES)


def test_the_join_query_probes():
    tables = catalogue(2000)
    assert measure(named("join"), tables).hash_probes > 0


def test_the_range_query_probes_nothing():
    tables = catalogue(2000)
    assert measure(named("range"), tables).hash_probes == 0


def test_the_top_ten_query_returns_ten_rows():
    tables = catalogue(2000)
    assert measure(named("top ten"), tables).rows == 10


def test_the_point_query_returns_one_row():
    tables = catalogue(20000)
    assert measure(named("point"), tables).rows == 1


def test_the_aggregate_returns_its_groups():
    tables = catalogue(2000)
    assert measure(named("aggregate"), tables).rows == 12


def test_the_wide_aggregate_returns_more_groups():
    tables = catalogue(2000)
    assert measure(named("wide aggregate"), tables).rows == 50


def test_the_join_returns_a_row_per_fact():
    tables = catalogue(2000)
    assert measure(named("join"), tables).rows == 2000


def test_the_star_query_returns_every_column():
    tables = catalogue(2000)
    assert measure(named("everything"), tables).rows == 2000


def test_every_measurement_records_a_node_count():
    assert all(one.nodes > 0 for one in measure_all(2000))
