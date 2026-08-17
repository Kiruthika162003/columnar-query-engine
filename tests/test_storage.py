from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import ConfigError
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.storage import bloom, layout
from cqe.storage.bloom import Bloom, build, build_for, optimal_hashes, predicted_rate, prune
from cqe.storage.layout import (
    Layout,
    as_they_arrive,
    clustered_by,
    cut,
    interleaved,
    sorted_by,
)


@pytest.fixture(scope="module")
def batch() -> Batch:
    """A table with a rising column, a random one and a low cardinality string."""
    state = np.random.default_rng(29)
    rows = 8000
    return Batch.from_columns(
        [
            integer_column("stamp", np.arange(rows)),
            integer_column("shop", state.integers(0, 20, rows)),
            floating_column("amount", state.normal(100, 30, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 5, rows)]),
        ]
    )


def test_a_filter_never_denies_a_value_it_holds():
    assert bloom.a_filter_never_says_no_to_a_value_it_holds()["it_never_says_no"]


def test_the_false_positive_rate_is_near_the_prediction():
    measured = bloom.the_false_positive_rate_is_what_it_is()
    assert measured["measured"] < measured["predicted"] * 3


def test_the_hash_count_curve_has_a_minimum():
    assert bloom.more_hashes_are_better_up_to_a_point()["it_falls_then_rises"]


def test_the_measured_optimum_is_the_theoretical_one():
    assert bloom.more_hashes_are_better_up_to_a_point()["they_agree"]


def test_the_module_uses_the_optimum():
    assert bloom.more_hashes_are_better_up_to_a_point()["and_the_module_uses"] == bloom.HASHES


def test_two_hashes_would_have_been_much_worse():
    assert bloom.more_hashes_are_better_up_to_a_point()["two_would_cost_this_much_more"] > 5


def test_the_size_curve_falls_where_it_can_be_measured():
    assert bloom.more_bits_are_better_until_the_trial_runs_out()[
        "it_falls_while_it_can_be_measured"
    ]


def test_the_size_curve_is_not_monotone_at_the_floor():
    assert not bloom.more_bits_are_better_until_the_trial_runs_out()[
        "the_whole_curve_is_monotone"
    ]


def test_a_filter_prunes_most_groups_on_an_equality():
    assert bloom.a_filter_prunes_an_equality()["it_pruned_most"]


def test_a_filter_keeps_every_group_that_holds_the_value():
    assert bloom.a_filter_prunes_an_equality()["and_kept_every_group_that_has_it"]


def test_a_zone_map_keeps_nearly_everything_on_that_query():
    assert bloom.a_zone_map_barely_prunes_the_same_query()[
        "the_zone_map_kept_nearly_everything"
    ]


def test_a_zone_map_keeps_all_of_it_for_a_middle_key():
    assert bloom.a_zone_map_barely_prunes_the_same_query()["and_for_a_middle_key_it_keeps_all"]


def test_the_bloom_filter_prunes_that_query():
    assert bloom.a_zone_map_barely_prunes_the_same_query()["the_bloom_filter_pruned_most"]


def test_a_zone_map_prunes_a_range():
    assert bloom.a_bloom_filter_cannot_prune_a_range()["the_zone_map_pruned_some"]


def test_the_two_mechanisms_intersect():
    assert bloom.the_two_together_prune_more_than_either()["it_is_the_intersection"]


def test_together_they_prune_more_than_either():
    assert bloom.the_two_together_prune_more_than_either()["and_smaller_than_either"]


def test_building_from_a_dictionary_gives_the_same_filter():
    assert bloom.a_dictionary_column_builds_from_its_dictionary()["same_bits"]


def test_building_from_a_dictionary_saves_hashes():
    assert bloom.a_dictionary_column_builds_from_its_dictionary()["hashes_saved"] > 0


def test_the_hash_is_stable():
    assert bloom.the_hash_is_stable_across_runs()["stable"]


def test_a_string_and_a_number_hash_differently():
    assert bloom.the_hash_is_stable_across_runs()["a_string_and_a_number_differ"]


def test_the_hash_does_not_cluster():
    assert bloom.the_hash_spreads()["it_is_not_clustered"]


def test_every_type_can_be_filtered():
    assert all(bloom.integers_and_floats_and_strings_all_work().values())


def test_the_loose_filter_is_too_large_to_store():
    assert bloom.the_filter_is_not_free()["the_loose_form_is_not_storable"]


def test_the_packed_filter_is_not():
    assert bloom.the_filter_is_not_free()["the_packed_one_is"]


def test_packing_saves_eight_times():
    assert bloom.a_packed_filter_is_eight_times_smaller()["it_is_eight"]


def test_an_empty_filter_says_no():
    assert bloom.an_empty_set_still_answers()["it_says_no"]


def test_an_empty_filter_is_the_minimum_size():
    assert bloom.an_empty_set_still_answers()["it_is_the_minimum"]


def test_a_zero_bit_filter_is_refused():
    assert bloom.a_zero_size_filter_is_refused()


def test_a_zero_hash_filter_is_refused():
    assert bloom.a_zero_hash_filter_is_refused()


def test_a_short_truth_is_refused():
    assert bloom.a_truth_shorter_than_the_filters_is_refused()


def test_the_bloom_summary_reports_no_false_negatives():
    assert bloom.summarise()["no_false_negatives"]


def test_a_filter_finds_all_its_values():
    values = [f"one{index}" for index in range(200)]
    built = build(values)
    assert all(built.might_contain(one) for one in values)


def test_a_filter_rejects_most_others():
    built = build([f"one{index}" for index in range(200)])
    others = [f"two{index}" for index in range(200)]
    assert sum(built.might_contain(one) for one in others) < 20


def test_any_of_finds_a_present_value():
    built = build(["a", "b", "c"])
    assert built.any_of(["z", "b"])


def test_any_of_rejects_an_absent_set():
    built = build([f"one{index}" for index in range(200)])
    assert not built.any_of([f"missing{index}" for index in range(5)])


def test_a_filter_summarises():
    summary = build(["a", "b"]).as_dict()
    assert summary["entries"] == 2 and summary["hashes"] == bloom.HASHES


def test_a_filter_reports_its_occupancy():
    assert 0 < build([f"one{index}" for index in range(100)]).occupancy < 1


def test_a_bigger_filter_has_lower_occupancy():
    values = [f"one{index}" for index in range(100)]
    assert (
        build(values, bits_per_entry=40).occupancy < build(values, bits_per_entry=4).occupancy
    )


def test_the_optimal_hash_count_is_seven_at_ten_bits():
    assert optimal_hashes(10) == 7


def test_the_predicted_rate_falls_with_size():
    assert predicted_rate(20, 7) < predicted_rate(10, 7)


def test_pruning_reports_its_share():
    filters = [build(["a"]), build(["b"]), build(["c"])]
    assert prune(filters, "a").skipped >= 2


def test_pruning_with_a_truth_reports_false_positives():
    filters = [build(["a"]), build(["b"])]
    measured = prune(filters, "a", truth=[True, False])
    assert measured.false_positives == measured.kept - 1


def test_pruning_summarises():
    filters = [build(["a"]), build(["b"])]
    assert prune(filters, "a").as_dict()["groups"] == 2


def test_a_filter_over_a_column_of_integers_works():
    built = build_for(integer_column("v", np.arange(100)))
    assert built.might_contain(50) and not built.might_contain(100000)


def test_arrival_order_prunes_a_rising_column():
    assert layout.arrival_order_prunes_a_rising_column()["it_pruned_nearly_everything"]


def test_arrival_order_wastes_nothing_on_that_column():
    assert layout.arrival_order_prunes_a_rising_column()["and_wasted_almost_nothing"]


def test_arrival_order_prunes_nothing_on_an_unrelated_column():
    assert layout.arrival_order_prunes_nothing_else()["it_pruned_nothing"]


def test_arrival_order_wastes_nearly_everything_there():
    assert layout.arrival_order_prunes_nothing_else()["and_wasted_nearly_everything"]


def test_sorting_makes_the_zone_map_exact():
    assert layout.sorting_by_a_column_makes_its_zone_map_exact()[
        "the_waste_is_one_group_at_the_boundary"
    ]


def test_sorting_reads_far_fewer_rows():
    assert layout.sorting_by_a_column_makes_its_zone_map_exact()["ratio"] > 10


def test_sorting_by_one_column_costs_another():
    assert layout.sorting_by_one_column_costs_the_others()["the_sort_cost_the_stamp_column"]


def test_the_cost_to_the_other_column_is_large():
    assert (
        layout.sorting_by_one_column_costs_the_others()["it_now_reads_this_many_times_more"] > 5
    )


def test_clustering_prunes_an_equality():
    assert layout.clustering_by_a_low_cardinality_column_prunes_equalities()[
        "clustering_prunes"
    ]


def test_arrival_order_does_not_prune_that_equality():
    assert layout.clustering_by_a_low_cardinality_column_prunes_equalities()["arrival_does_not"]


def test_a_bloom_filter_adds_nothing_after_clustering():
    assert layout.a_bloom_filter_is_redundant_after_clustering()["they_agree"]


def test_clustering_helps_run_length_encoding():
    assert layout.clustering_helps_the_encodings_too()["it_helped"]


def test_clustering_helps_it_by_a_lot():
    assert layout.clustering_helps_the_encodings_too()["ratio"] > 10


def test_clustering_does_not_help_the_dictionary():
    assert layout.sorting_does_not_help_the_dictionary()["they_are_the_same"]


def test_a_sort_beats_an_interleave_on_its_own_column():
    assert layout.interleaving_favours_neither_column()["the_sort_is_better_on_its_own_column"]


def test_an_interleave_beats_a_sort_on_the_other_column():
    assert layout.interleaving_favours_neither_column()["the_interleave_is_better_on_the_other"]


def test_interleaving_ranks_beats_interleaving_values():
    assert layout.interleaving_ranks_rather_than_values()["ranks_prune_at_least_as_well"]


def test_smaller_groups_waste_fewer_rows():
    assert layout.a_smaller_group_prunes_more_and_costs_more()["smaller_groups_waste_less"]


def test_smaller_groups_are_more_numerous():
    assert layout.a_smaller_group_prunes_more_and_costs_more()["and_there_are_more_of_them"]


def test_an_arrival_layout_flattens_to_the_same_table():
    assert layout.a_layout_flattens_back_to_its_table()["arrival_is_identical"]


def test_a_sorted_layout_flattens_to_a_permutation():
    assert layout.a_layout_flattens_back_to_its_table()["sorted_is_a_permutation"]


def test_a_sorted_layout_is_ordered_by_its_key():
    assert layout.a_layout_flattens_back_to_its_table()["and_the_sort_key_is_ordered"]


def test_the_last_group_is_short():
    assert layout.the_last_group_is_short()["the_last_is_short"]


def test_the_groups_sum_to_the_table():
    assert layout.the_last_group_is_short()["they_sum_to_the_table"]


def test_no_layout_wins_every_predicate():
    assert layout.no_layout_wins_every_column()["they_are_not_all_the_same"]


def test_there_are_three_distinct_winners():
    assert layout.no_layout_wins_every_column()["distinct_winners"] == 3


def test_a_zero_group_size_is_refused():
    assert layout.a_zero_group_size_is_refused()


def test_sorting_by_a_missing_column_is_refused():
    assert layout.sorting_by_a_missing_column_is_refused()


def test_interleaving_no_columns_is_refused():
    assert layout.interleaving_nothing_is_refused()


def test_interleaving_five_columns_is_refused():
    assert layout.interleaving_five_columns_is_refused()


def test_flattening_an_empty_layout_is_refused():
    assert layout.flattening_nothing_is_refused()


def test_the_layout_summary_reports_the_group_size():
    assert layout.summarise()["group_size"] == layout.GROUP_SIZE


def test_cutting_a_table_covers_every_row(batch):
    assert sum(one.rows for one in cut(batch)) == batch.rows


def test_cutting_a_table_gives_the_expected_group_count(batch):
    assert len(cut(batch, 1000)) == 8


def test_an_arrival_layout_keeps_the_row_order(batch):
    produced = as_they_arrive(batch).flatten()
    assert np.array_equal(produced.column("stamp").values, batch.column("stamp").values)


def test_a_sorted_layout_orders_its_key(batch):
    produced = sorted_by(batch, "amount").flatten()
    assert np.all(np.diff(produced.column("amount").values) >= 0)


def test_a_clustered_layout_groups_equal_values(batch):
    produced = clustered_by(batch, "region").flatten().column("region").to_list()
    changes = sum(1 for one in range(1, len(produced)) if produced[one] != produced[one - 1])
    assert changes < 10


def test_an_interleaved_layout_keeps_every_row(batch):
    assert interleaved(batch, ["shop", "amount"]).rows == batch.rows


def test_a_layout_reports_its_order(batch):
    assert sorted_by(batch, "amount").order == "sorted"


def test_a_layout_reports_its_key(batch):
    assert clustered_by(batch, "region").key == "region"


def test_a_layout_summarises(batch):
    summary = as_they_arrive(batch).as_dict()
    assert summary["rows"] == batch.rows and summary["order"] == "arrival"


def test_a_layout_reports_its_size(batch):
    assert as_they_arrive(batch).nbytes > 0


def test_pruning_a_sorted_layout_reads_few_groups(batch):
    built = sorted_by(batch, "amount")
    predicate = Compare("<", column("amount"), literal(40.0))
    assert layout.prune(built, predicate).read < len(built.groups) // 4


def test_pruning_reports_the_rows_it_kept(batch):
    built = sorted_by(batch, "amount")
    predicate = Compare("<", column("amount"), literal(40.0))
    measured = layout.prune(built, predicate)
    assert 0 < measured.rows_kept <= measured.rows_read


def test_pruning_never_skips_a_group_with_matches(batch):
    from cqe.exec.filter import apply as apply_predicate

    built = sorted_by(batch, "amount")
    predicate = Compare("<", column("amount"), literal(40.0))
    total = apply_predicate(predicate, batch).rows
    assert layout.prune(built, predicate).rows_kept == total


def test_the_interleaved_key_is_refused_for_five_columns(batch):
    with pytest.raises(ConfigError):
        interleaved(batch, ["stamp", "shop", "amount", "region", "stamp"])


def test_an_empty_layout_has_no_rows():
    assert Layout(groups=(), order="arrival").rows == 0


def test_a_bloom_filter_of_a_slice_only_holds_its_own_values(batch):
    sliced = batch.slice(0, 50).column("region")
    present = set(sliced.to_list())
    built = build_for(sliced)
    absent = [one for one in ("region0", "region1", "region2") if one not in present]
    assert all(built.might_contain(one) for one in present)
    assert not absent or not all(built.might_contain(one) for one in absent)


def test_a_bloom_filter_is_smaller_than_the_column_it_covers(batch):
    sliced = batch.slice(0, 500).column("region")
    packed = np.packbits(build_for(sliced).bits).nbytes
    assert packed < sliced.values.nbytes


def test_a_bloom_object_can_be_built_directly():
    made = Bloom(bits=np.ones(64, dtype=bool), hashes=2, entries=1)
    assert made.size == 64 and made.might_contain("anything")
