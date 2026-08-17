from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.encode import bitpack, delta, dictionary, runlength
from cqe.errors import ConfigError, EncodingError


class TestDictionarySizes:
    def test_the_saving_falls_as_cardinality_rises(self):
        ratios = [row["ratio"] for row in dictionary.the_saving_depends_on_cardinality()]
        assert ratios == sorted(ratios)

    def test_a_low_cardinality_column_encodes_small(self):
        rows = dictionary.the_saving_depends_on_cardinality()
        assert rows[0]["ratio"] < 0.1

    def test_a_unique_column_encodes_larger_than_the_original(self):
        rows = dictionary.the_saving_depends_on_cardinality()
        assert rows[-1]["ratio"] > 1.0

    def test_the_crossover_is_late(self):
        assert dictionary.the_crossover_is_later_than_it_looks()["crossover"] == 0.9

    def test_it_still_pays_at_half_cardinality(self):
        assert dictionary.the_crossover_is_later_than_it_looks()["and_at_a_half"]

    def test_and_at_a_tenth(self):
        assert dictionary.the_crossover_is_later_than_it_looks()["it_pays_at_a_tenth"]

    def test_it_does_lose_eventually(self):
        assert dictionary.the_crossover_is_later_than_it_looks()["it_ever_loses"]

    def test_an_empty_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            dictionary.the_saving_depends_on_cardinality(fractions=())

    def test_the_code_width_steps_at_a_byte(self):
        assert dictionary.code_width(256) == 1 and dictionary.code_width(257) == 2

    def test_and_again_at_two_bytes(self):
        assert dictionary.code_width(65536) == 2 and dictionary.code_width(65537) == 4

    def test_a_negative_count_is_refused(self):
        with pytest.raises(ConfigError, match="not a count"):
            dictionary.code_width(-1)

    def test_the_sample_has_the_cardinality_it_was_asked_for(self):
        values = dictionary.sample(1000, 1000)
        assert len(set(values)) == 1000

    def test_a_cardinality_past_the_row_count_is_refused(self):
        with pytest.raises(ConfigError, match="do not fit"):
            dictionary.sample(10, 100)


class TestDictionaryQueries:
    def test_an_equality_filter_agrees_with_a_direct_scan(self):
        result = dictionary.an_equality_filter_is_one_integer_comparison()
        assert result["agrees_with_the_direct_scan"]

    def test_it_compares_fewer_bytes_than_the_text(self):
        result = dictionary.an_equality_filter_is_one_integer_comparison()
        assert result["compared_bytes_per_row"] < result["raw_bytes_per_row"]

    def test_a_range_filter_agrees_both_ways(self):
        assert dictionary.a_range_filter_needs_an_ordered_dictionary()["both_agree"]

    def test_the_ordered_form_does_less_per_row(self):
        result = dictionary.a_range_filter_needs_an_ordered_dictionary()
        assert result["ordered_work_per_row"] < result["unordered_work_per_row"]

    def test_an_unordered_dictionary_refuses_a_code_range(self):
        assert dictionary.an_unordered_dictionary_refuses_a_code_range()

    def test_ordering_costs_a_real_share_of_the_build(self):
        assert dictionary.ordering_the_dictionary_is_not_free()["it_is_not_negligible"]

    def test_and_scales_with_cardinality(self):
        result = dictionary.ordering_the_dictionary_is_not_free()
        assert result["it_scales_with_cardinality_not_height"]


class TestDictionaryOrdering:
    def test_first_seen_codes_are_monotone_across_runs(self):
        assert dictionary.sorting_the_dictionary_destroys_run_structure()[
            "first_seen_is_monotone"
        ]

    def test_sorted_codes_are_not(self):
        result = dictionary.sorting_the_dictionary_destroys_run_structure()
        assert not result["sorted_is_monotone"]

    def test_the_run_count_is_unchanged(self):
        result = dictionary.sorting_the_dictionary_destroys_run_structure()
        assert result["run_count_is_unchanged"]

    def test_skew_does_not_change_the_size(self):
        assert dictionary.skew_does_not_change_the_size()["they_are_close"]

    def test_and_the_cardinality_is_the_same(self):
        assert dictionary.skew_does_not_change_the_size()["distinct_matches"]

    def test_the_round_trip_is_exact_both_ways(self):
        result = dictionary.the_round_trip_is_exact()
        assert result["ordered_exact"] and result["unordered_exact"]

    def test_more_row_groups_duplicate_more_dictionary(self):
        assert dictionary.more_groups_means_more_duplicated_dictionary()["duplication_rises"]

    def test_and_by_a_lot(self):
        assert dictionary.more_groups_means_more_duplicated_dictionary()["and_a_lot"]


class TestDictionaryRefusals:
    def test_a_value_outside_the_dictionary_is_refused(self):
        assert dictionary.a_value_outside_the_dictionary_is_refused()

    def test_a_dictionary_with_repeats_is_refused(self):
        assert dictionary.a_dictionary_with_repeats_is_refused()

    def test_an_unsorted_ordered_dictionary_is_refused(self):
        assert dictionary.an_unsorted_ordered_dictionary_is_refused()

    def test_a_column_needs_an_ordered_dictionary(self):
        assert dictionary.a_column_needs_an_ordered_dictionary()

    def test_a_dictionary_reports_whether_it_holds_a_value(self):
        table = dictionary.Dictionary(("a", "b"), ordered=True)
        assert table.contains("a") and not table.contains("z")

    def test_a_dictionary_serialises(self):
        assert dictionary.Dictionary(("a",), ordered=True).as_dict()["entries"] == 1

    def test_the_summary_reports_the_crossover(self):
        assert dictionary.summarise()["crossover_fraction"] == 0.9


class TestRunLength:
    def test_clustering_beats_cardinality(self):
        result = runlength.it_is_clustering_not_cardinality()
        assert result["sorted_wins"] and result["scattered_loses"]

    def test_a_sorted_column_has_one_run_per_value(self):
        rows = runlength.a_sorted_column_is_the_best_case()
        assert all(row["runs_equal_cardinality"] for row in rows)

    def test_the_ratio_falls_with_the_run_length(self):
        ratios = [row["ratio"] for row in runlength.the_ratio_follows_the_run_length()]
        assert ratios == sorted(ratios, reverse=True)

    def test_a_run_of_one_expands_the_column(self):
        assert runlength.the_break_even_run_length()["a_run_of_one_expands_it"]

    def test_and_by_about_half(self):
        assert runlength.the_break_even_run_length()["and_by_half"]

    def test_the_threshold_is_low(self):
        assert runlength.the_break_even_run_length()["the_threshold_is_low"]

    def test_a_filter_on_runs_agrees_with_a_direct_scan(self):
        assert runlength.a_filter_runs_on_the_runs()["agrees_with_the_direct_scan"]

    def test_and_compares_far_fewer_values(self):
        result = runlength.a_filter_runs_on_the_runs()
        assert result["comparisons_on_runs"] < result["comparisons_on_rows"] / 100

    def test_two_columns_union_their_boundaries(self):
        result = runlength.it_composes_better_than_i_expected()
        assert result["the_union_is_bounded_by_the_sum"]

    def test_and_still_beat_the_row_count(self):
        assert runlength.it_composes_better_than_i_expected()["it_still_beats_the_row_count"]

    def test_the_cost_at_most_doubles(self):
        assert runlength.it_composes_better_than_i_expected()["the_cost_at_most_doubles"]

    def test_the_lengths_are_small_integers(self):
        result = runlength.the_run_lengths_are_small_integers()
        assert result["bits_needed"] < result["stored_bits"]

    def test_every_length_is_positive(self):
        assert runlength.the_run_lengths_are_small_integers()["every_length_is_positive"]

    def test_the_round_trip_is_exact_on_every_shape(self):
        assert all(runlength.the_round_trip_is_exact().values())

    def test_an_empty_column_encodes_to_nothing(self):
        result = runlength.an_empty_column_encodes_to_nothing()
        assert result["runs"] == 0 and result["round_trips"]

    def test_a_constant_column_is_one_run(self):
        assert runlength.a_single_run_is_the_extreme()["it_is_one_run"]

    def test_the_shapes_sort_by_ratio(self):
        ratios = [row["ratio"] for row in runlength.compare_the_shapes()]
        assert ratios == sorted(ratios)

    def test_sorted_is_the_best_shape(self):
        assert runlength.compare_the_shapes()[0]["shape"] == "sorted"

    def test_mismatched_run_arrays_are_refused(self):
        assert runlength.mismatched_run_arrays_are_refused()

    def test_a_zero_length_run_is_refused(self):
        assert runlength.a_zero_length_run_is_refused()

    def test_a_two_dimensional_column_is_refused(self):
        assert runlength.a_two_dimensional_column_is_refused()

    def test_an_impossible_generator_is_refused(self):
        assert runlength.an_impossible_generator_is_refused()

    def test_boundaries_start_at_zero(self):
        runs = runlength.encode(np.array([1, 1, 2, 2, 2, 3]))
        assert list(runs.boundaries()) == [0, 2, 5]

    def test_the_run_lengths_sum_to_the_rows(self):
        runs = runlength.encode(np.array([1, 1, 2, 3, 3]))
        assert runs.rows == 5


class TestBitPacking:
    def test_the_ratio_follows_the_width(self):
        ratios = [row["ratio"] for row in bitpack.the_saving_is_the_bit_width()]
        assert ratios == sorted(ratios)

    def test_one_outlier_widens_the_column(self):
        assert bitpack.one_outlier_sets_the_width()["the_width_more_than_quadrupled"]

    def test_and_triples_the_ratio(self):
        assert bitpack.one_outlier_sets_the_width()["the_ratio_more_than_tripled"]

    def test_a_reference_halves_the_width(self):
        result = bitpack.frame_of_reference_is_what_makes_it_work()
        assert result["it_more_than_halves_the_width"]

    def test_a_power_of_two_width_never_straddles(self):
        rows = bitpack.a_width_that_is_not_a_byte_straddles_words()
        assert all(row["straddle_share"] == 0.0 for row in rows if row["divides_sixty_four"])

    def test_and_everything_else_does(self):
        rows = bitpack.a_width_that_is_not_a_byte_straddles_words()
        assert all(row["straddle_share"] > 0.0 for row in rows if not row["divides_sixty_four"])

    def test_padding_removes_every_straddle(self):
        result = bitpack.rounding_up_to_a_byte_costs_little()
        assert result["the_padding_removes_every_straddle"]

    def test_and_costs_under_a_fifth(self):
        assert bitpack.rounding_up_to_a_byte_costs_little()["and_costs_under_a_fifth"]

    def test_the_validity_mask_is_over_a_tenth(self):
        assert bitpack.the_validity_mask_is_a_real_share()["it_is_over_a_tenth"]

    def test_the_round_trip_is_exact_at_every_width(self):
        assert all(bitpack.the_round_trip_is_exact(2_000).values())

    def test_negative_values_survive(self):
        result = bitpack.negative_values_survive_the_reference()
        assert result["round_trips"] and result["reference_is_negative"]

    def test_an_empty_column_packs_to_nothing(self):
        assert bitpack.an_empty_column_packs_to_nothing()["round_trips"]

    def test_a_constant_column_needs_one_bit(self):
        assert bitpack.a_constant_column_needs_one_bit()["it_is_one_bit"]

    def test_bits_needed_is_the_bit_length(self):
        assert bitpack.bits_needed(255) == 8 and bitpack.bits_needed(256) == 9

    def test_a_span_of_zero_still_needs_a_bit(self):
        assert bitpack.bits_needed(0) == 1

    def test_a_width_too_narrow_is_refused(self):
        assert bitpack.a_width_too_narrow_is_refused()

    def test_a_width_past_sixty_four_is_refused(self):
        assert bitpack.a_width_past_sixty_four_is_refused()

    def test_a_two_dimensional_column_is_refused(self):
        assert bitpack.a_two_dimensional_column_is_refused()

    def test_a_negative_span_is_refused(self):
        assert bitpack.a_negative_span_is_refused()

    def test_the_widths_compare_monotonically(self):
        ratios = [row["ratio"] for row in bitpack.compare_the_widths()]
        assert ratios == sorted(ratios)


class TestDelta:
    def test_a_sorted_column_deltas_small(self):
        assert delta.a_sorted_column_deltas_to_almost_nothing()["it_is_a_large_saving"]

    def test_and_its_differences_are_monotone(self):
        assert delta.a_sorted_column_deltas_to_almost_nothing()["monotone"]

    def test_shuffling_costs_exactly_one_bit(self):
        assert delta.shuffling_the_same_values_costs_a_bit()["it_costs_exactly_one_bit"]

    def test_the_order_is_the_whole_thing(self):
        result = delta.the_order_of_the_rows_is_the_whole_thing()
        assert result["the_gap_is_about_threefold"]

    def test_but_shuffled_still_beats_int64(self):
        assert delta.the_order_of_the_rows_is_the_whole_thing()["shuffled_still_beats_int64"]

    def test_a_regular_step_has_a_second_span_of_zero(self):
        assert delta.delta_of_delta_is_for_regular_steps()["the_second_span_is_zero"]

    def test_and_the_two_orders_are_level(self):
        assert delta.delta_of_delta_is_for_regular_steps()["they_are_level_here"]

    def test_a_jittered_step_costs_a_bit_per_order(self):
        result = delta.and_it_costs_a_bit_everywhere_else()
        assert result["each_order_costs_about_a_bit"]

    def test_and_the_first_order_wins(self):
        assert delta.and_it_costs_a_bit_everywhere_else()["the_first_order_wins"]

    def test_the_gap_size_sets_the_width(self):
        widths = [row["delta_bits"] for row in delta.the_gap_size_sets_the_width()]
        assert widths == sorted(widths)

    def test_one_jump_widens_everything(self):
        assert delta.one_jump_sets_the_width()["and_by_a_lot"]

    def test_the_round_trip_is_exact_on_every_shape(self):
        assert all(delta.the_round_trip_is_exact().values())

    def test_a_constant_column_deltas_to_one_bit(self):
        assert delta.a_constant_column_deltas_to_zero()["it_is_one_bit"]

    def test_a_descending_column_is_not_monotone(self):
        assert not delta.a_descending_column_is_not_monotone()["monotone"]

    def test_but_encodes_just_as_well(self):
        result = delta.a_descending_column_is_not_monotone()
        assert result["it_encodes_as_well_as_ascending"]

    def test_a_column_shorter_than_the_order_is_refused(self):
        assert delta.a_column_shorter_than_the_order_is_refused()

    def test_a_zero_order_is_refused(self):
        assert delta.a_zero_order_is_refused()

    def test_a_two_dimensional_column_is_refused(self):
        assert delta.a_two_dimensional_column_is_refused()

    def test_decoding_without_the_seeds_is_refused(self):
        assert delta.decoding_without_the_seeds_is_refused()

    def test_the_shapes_sort_by_ratio(self):
        ratios = [row["ratio"] for row in delta.compare_the_shapes()]
        assert ratios == sorted(ratios)

    def test_the_worst_shape_is_the_shuffled_one(self):
        assert delta.compare_the_shapes()[-1]["shape"] == "shuffled"

    def test_mismatched_seed_counts_are_refused(self):
        deltas = delta.encode(np.arange(10), order=2)
        with pytest.raises(EncodingError, match="seed values"):
            delta.decode(deltas)


class TestSummaries:
    def test_every_encoding_summarises(self):
        for module in (dictionary, runlength, bitpack, delta):
            assert isinstance(module.summarise(), dict)

    def test_the_run_length_summary_names_the_best_shape(self):
        assert runlength.summarise()["best_shape"] == "sorted"

    def test_the_delta_summary_names_the_worst(self):
        assert delta.summarise()["worst_shape"] == "shuffled"

    def test_the_bitpack_summary_reports_the_mask_share(self):
        assert bitpack.summarise()["mask_share"] > 0.1
