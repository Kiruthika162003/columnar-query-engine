from __future__ import annotations

import numpy as np
import pytest

from cqe.errors import ConfigError, UnknownColumn
from cqe.exec.batch import Batch
from cqe.stats import cardinality, histogram, sketch


class TestHistogramShapes:
    def test_both_shapes_agree_on_a_uniform_column(self):
        assert histogram.they_agree_on_a_uniform_column()["they_are_close"]

    def test_and_neither_is_bad(self):
        assert histogram.they_agree_on_a_uniform_column()["neither_is_bad"]

    def test_equi_depth_wins_on_skew(self):
        assert histogram.equi_depth_wins_on_a_skewed_column()["depth_wins"]

    def test_but_only_narrowly_on_the_median(self):
        assert histogram.equi_depth_wins_on_a_skewed_column()["ratio"] < 3.0

    def test_and_widely_on_the_worst_case(self):
        result = histogram.equi_depth_wins_on_a_skewed_column()
        assert result["width_worst"] > 10 * result["depth_worst"]

    def test_a_bimodal_column_wastes_equi_width_buckets(self):
        assert histogram.and_on_a_bimodal_one()["most_width_buckets_are_empty"]

    def test_and_equi_depth_wastes_none(self):
        assert histogram.and_on_a_bimodal_one()["empty_depth_buckets"] == 0

    def test_more_buckets_help_both_shapes(self):
        rows = histogram.more_buckets_help_both()
        assert rows[-1]["depth_median"] < rows[0]["depth_median"]

    def test_and_width_too(self):
        rows = histogram.more_buckets_help_both()
        assert rows[-1]["width_median"] < rows[0]["width_median"]

    def test_many_cheap_buckets_beat_a_few_good_ones(self):
        assert histogram.many_bad_buckets_beat_a_few_good_ones()["the_small_one_loses"]

    def test_though_the_few_are_smaller(self):
        assert histogram.many_bad_buckets_beat_a_few_good_ones()["but_is_smaller"]

    def test_an_empty_bucket_sweep_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to sweep"):
            histogram.more_buckets_help_both(counts=())


class TestHistogramEstimates:
    def test_the_distinct_count_improves_an_equality_estimate(self):
        result = histogram.an_equality_estimate_needs_the_distinct_count()
        assert result["the_distinct_count_helps"]

    def test_a_range_outside_the_column_estimates_zero(self):
        assert histogram.a_range_outside_the_column_estimates_nothing()["both_are_zero"]

    def test_and_the_full_range_estimates_everything(self):
        result = histogram.a_range_outside_the_column_estimates_nothing()
        assert result["the_full_range_is_everything"]

    def test_nulls_are_counted_separately(self):
        assert histogram.nulls_are_counted_and_never_estimated()["the_nulls_were_counted"]

    def test_and_the_buckets_hold_the_rest(self):
        result = histogram.nulls_are_counted_and_never_estimated()
        assert result["the_buckets_hold_the_rest"]

    def test_a_constant_column_is_one_equi_width_bucket(self):
        assert histogram.a_constant_column_is_one_bucket()["width_is_one"]

    def test_a_zero_width_bucket_records_a_common_value(self):
        result = histogram.a_zero_width_bucket_is_a_statement_about_skew()
        assert result["it_holds_the_common_value"]

    def test_and_estimates_it_as_common(self):
        result = histogram.a_zero_width_bucket_is_a_statement_about_skew()
        assert result["the_estimate_for_it_is_large"]

    def test_an_empty_column_estimates_nothing(self):
        assert histogram.an_empty_column_is_refused_nothing()["the_estimate_is_zero"]

    def test_a_zero_bucket_count_is_refused(self):
        assert histogram.a_zero_bucket_count_is_refused()

    def test_a_backwards_bucket_is_refused(self):
        assert histogram.a_backwards_bucket_is_refused()

    def test_a_negative_count_is_refused(self):
        assert histogram.a_negative_count_is_refused()

    def test_a_histogram_with_no_buckets_is_refused(self):
        assert histogram.a_histogram_with_no_buckets_is_refused()

    def test_a_two_dimensional_column_is_refused(self):
        assert histogram.a_two_dimensional_column_is_refused()

    def test_a_bucket_overlap_is_between_zero_and_one(self):
        bucket = histogram.Bucket(low=0.0, high=10.0, count=100)
        assert bucket.overlap(-5.0, 15.0) == 1.0 and bucket.overlap(20.0, 30.0) == 0.0

    def test_a_zero_width_bucket_is_all_or_nothing(self):
        bucket = histogram.Bucket(low=5.0, high=5.0, count=100)
        assert bucket.overlap(0.0, 10.0) == 1.0 and bucket.overlap(6.0, 10.0) == 0.0

    def test_a_backwards_range_estimates_nothing(self):
        built = histogram.equi_depth(histogram.uniform(1_000), 8)
        assert built.estimate_range(100.0, 50.0) == 0.0

    def test_a_histogram_serialises(self):
        built = histogram.equi_depth(histogram.uniform(1_000), 8)
        assert built.as_dict()["kind"] == "depth"

    def test_error_is_infinite_when_the_truth_is_zero(self):
        assert histogram.error(5.0, 0.0) == float("inf")

    def test_and_zero_when_both_are(self):
        assert histogram.error(0.0, 0.0) == 0.0


class TestSketches:
    def test_sampling_gets_one_of_them_badly_wrong(self):
        result = sketch.sampling_cannot_tell_the_two_apart()
        assert result["one_of_them_is_badly_wrong"]

    def test_a_bigger_sample_barely_helps(self):
        rows = sketch.a_bigger_sample_does_not_fix_it()
        assert rows[-1]["error"] > 0.5

    def test_a_linear_counter_is_accurate_until_it_fills(self):
        rows = sketch.a_linear_counter_is_exact_until_it_saturates()
        good = [row for row in rows if not row["saturated"]]
        assert all(row["error"] < 0.02 for row in good)

    def test_and_then_saturates_completely(self):
        rows = sketch.a_linear_counter_is_exact_until_it_saturates()
        assert rows[-1]["saturated"]

    def test_a_sketch_holds_across_the_whole_range(self):
        rows = sketch.hyperloglog_holds_across_the_whole_range()
        assert all(row["error"] < 0.1 for row in rows)

    def test_the_error_is_near_what_the_buckets_imply(self):
        rows = sketch.the_error_follows_the_bucket_count()
        assert all(row["error"] < 5 * row["expected_error"] for row in rows)

    def test_the_expected_error_falls_with_the_precision(self):
        rows = sketch.the_error_follows_the_bucket_count()
        expected = [row["expected_error"] for row in rows]
        assert expected == sorted(expected, reverse=True)

    def test_sketches_merge_exactly(self):
        assert sketch.sketches_merge_and_samples_do_not()["merging_matches_one_pass"]

    def test_and_the_registers_are_identical(self):
        assert sketch.merging_is_exact_not_approximate()["registers_match"]

    def test_summing_exact_counts_overcounts(self):
        assert sketch.sketches_merge_and_samples_do_not()["summing_overcounts"]

    def test_and_by_a_lot(self):
        assert sketch.sketches_merge_and_samples_do_not()["and_by_a_lot"]

    def test_the_small_range_correction_helps(self):
        assert sketch.the_small_range_correction_earns_its_place()["the_correction_helps"]

    def test_and_the_raw_estimator_is_badly_wrong(self):
        assert sketch.the_small_range_correction_earns_its_place()["raw_error"] > 10

    def test_adding_a_value_twice_changes_nothing(self):
        assert sketch.adding_a_value_twice_changes_nothing()["registers_match"]

    def test_strings_hash_too(self):
        assert sketch.string_columns_hash_too()["it_is_within_a_tenth"]

    def test_an_empty_sketch_estimates_zero(self):
        assert sketch.an_empty_column_estimates_nothing()["the_sketch_says_zero"]

    def test_a_bad_precision_is_refused(self):
        assert sketch.a_bad_precision_is_refused()

    def test_merging_different_precisions_is_refused(self):
        assert sketch.merging_different_precisions_is_refused()

    def test_a_tiny_counter_is_refused(self):
        assert sketch.a_tiny_counter_is_refused()

    def test_an_impossible_sample_is_refused(self):
        assert sketch.an_impossible_sample_is_refused()

    def test_a_two_dimensional_column_is_refused(self):
        assert sketch.a_two_dimensional_column_is_refused()

    def test_a_sketch_reports_its_size(self):
        assert sketch.HyperLogLog(precision=10).nbytes == 1024

    def test_a_sample_reports_its_rate(self):
        taken = sketch.Sample(seen=10, rows=1_000, sampled=100)
        assert taken.rate == 0.1 and taken.estimate() == 100.0

    def test_a_sample_larger_than_the_table_is_refused(self):
        with pytest.raises(ConfigError, match="sampled from"):
            sketch.Sample(seen=1, rows=10, sampled=100)

    def test_a_counter_serialises(self):
        assert sketch.LinearCounter(bits=1024).as_dict()["bits"] == 1024

    def test_the_hasher_spreads(self):
        values = sketch.hashed(np.arange(1_000))
        assert len(np.unique(values)) == 1_000


class TestCardinality:
    def test_a_single_predicate_is_estimated_well(self):
        assert cardinality.a_single_predicate_is_estimated_well()["it_is_accurate"]

    def test_independence_underestimates_correlated_columns(self):
        rows = cardinality.independence_underestimates_correlated_columns()
        assert rows[-1]["underestimates"]

    def test_and_not_independent_ones(self):
        rows = cardinality.independence_underestimates_correlated_columns()
        assert abs(rows[0]["ratio"] - 1.0) < 0.1

    def test_the_error_grows_with_the_conjuncts(self):
        rows = cardinality.the_error_compounds_with_the_conjuncts()
        ratios = [row["ratio"] for row in rows]
        assert ratios == sorted(ratios)

    def test_it_never_overestimates(self):
        assert cardinality.it_is_wrong_in_one_direction_only()["it_never_overestimates"]

    def test_a_disjunction_leans_the_other_way(self):
        assert cardinality.a_disjunction_is_estimated_the_other_way()["it_overestimates"]

    def test_null_checks_are_exact(self):
        assert cardinality.null_checks_are_exact()["it_is_exact"]

    def test_and_so_is_the_negation(self):
        assert cardinality.null_checks_are_exact()["the_negation_is_too"]

    def test_an_unestimable_predicate_uses_the_default(self):
        assert cardinality.an_unestimable_predicate_falls_back_to_a_third()["it_is_the_default"]

    def test_and_the_default_is_wrong_here(self):
        assert cardinality.an_unestimable_predicate_falls_back_to_a_third()["error"] > 0.2

    def test_a_group_count_is_capped(self):
        assert cardinality.a_group_count_is_capped_at_the_rows()["it_was_capped"]

    def test_because_the_product_is_absurd(self):
        assert cardinality.a_group_count_is_capped_at_the_rows()["the_product_was_absurd"]

    def test_a_correlated_group_count_is_overestimated(self):
        assert cardinality.a_correlated_group_count_is_overestimated()["it_overestimates"]

    def test_a_join_fanout_is_close_under_containment(self):
        assert cardinality.a_join_fanout_assumes_containment()["it_is_close"]

    def test_and_overestimates_when_containment_fails(self):
        assert cardinality.and_overestimates_when_containment_fails()["it_overestimates"]

    def test_by_about_the_match_rate(self):
        assert cardinality.and_overestimates_when_containment_fails()["by_about_two"]

    def test_a_string_equality_gives_one_estimate_for_every_value(self):
        result = cardinality.a_string_equality_uses_the_distinct_count()
        assert result["the_same_estimate_for_both"]

    def test_though_the_truths_differ(self):
        result = cardinality.a_string_equality_uses_the_distinct_count()
        assert result["the_truths_differ_by_a_lot"]

    def test_statistics_are_a_rounding_error(self):
        assert cardinality.statistics_cost_a_fixed_amount_per_column()["it_is_a_rounding_error"]

    def test_an_unknown_column_falls_back(self):
        assert cardinality.an_unknown_column_falls_back()["it_is_the_default"]

    def test_splitting_a_predicate_does_not_change_the_estimate(self):
        assert cardinality.conjuncts_and_selectivity_agree()["they_agree"]

    def test_an_impossible_correlation_is_refused(self):
        assert cardinality.an_impossible_correlation_is_refused()

    def test_an_unknown_column_lookup_is_refused(self):
        assert cardinality.an_unknown_column_lookup_is_refused()

    def test_too_few_trials_are_refused(self):
        assert cardinality.too_few_trials_are_refused()

    def test_statistics_serialise(self):
        stats = cardinality.collect(Batch.of(a=[1, 2, 3], b=["x", "y", "z"]))
        assert stats.as_dict()["columns"] == 2

    def test_a_column_reports_its_null_share(self):
        stats = cardinality.collect(Batch.of(a=[1, None, 3]))
        assert stats.column("a").null_share == pytest.approx(1 / 3)

    def test_an_unknown_column_is_refused_by_name(self):
        stats = cardinality.collect(Batch.of(a=[1]))
        with pytest.raises(UnknownColumn, match="is not in"):
            stats.column("z")


class TestSummaries:
    def test_every_stats_module_summarises(self):
        for module in (histogram, sketch, cardinality):
            assert isinstance(module.summarise(), dict)

    def test_the_histogram_summary_confirms_the_agreement(self):
        assert histogram.summarise()["they_agree_when_flat"]

    def test_the_sketch_summary_confirms_the_merge(self):
        assert sketch.summarise()["merging_matches_one_pass"]

    def test_the_cardinality_summary_confirms_the_direction(self):
        assert cardinality.summarise()["it_never_overestimates"]
