from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import Column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, SchemaError
from cqe.exec import distinct as dedupe
from cqe.exec.batch import Batch
from cqe.exec.distinct import (
    DICTIONARY,
    HASH,
    SKETCH_ABOVE,
    SORT,
    STRATEGIES,
    Distinct,
    by_dictionary,
    by_hash,
    by_sort,
    count_distinct,
    distinct,
    distinct_rows,
)


@pytest.fixture(scope="module")
def labels() -> Column:
    """A string column with a hundred distinct values."""
    state = np.random.default_rng(101)
    return string_column(
        "label", [f"kind{int(one):03d}" for one in state.integers(0, 100, 5000)]
    )


@pytest.fixture(scope="module")
def numbers() -> Column:
    """An integer column with fifty distinct values."""
    state = np.random.default_rng(103)
    return integer_column("v", state.integers(0, 50, 5000))


def test_every_strategy_finds_the_same_values():
    assert dedupe.every_strategy_finds_the_same_values()["they_all_agree"]


def test_every_strategy_finds_the_same_count():
    assert dedupe.every_strategy_finds_the_same_values()["and_the_counts_match"]


def test_the_hash_pass_is_not_ordered():
    assert dedupe.the_hash_pass_is_the_only_one_in_arrival_order()[
        "the_hash_one_is_not_ordered"
    ]


def test_the_other_two_are_ordered():
    assert dedupe.the_hash_pass_is_the_only_one_in_arrival_order()["the_other_two_are"]


def test_the_hash_output_is_arrival_order():
    assert dedupe.the_hash_pass_is_the_only_one_in_arrival_order()[
        "the_hash_output_is_arrival_order"
    ]


def test_the_sorted_output_is_not_arrival_order():
    assert dedupe.the_hash_pass_is_the_only_one_in_arrival_order()["and_the_sorted_one_is_not"]


def test_the_two_ordered_strategies_give_the_same_sequence():
    assert dedupe.the_hash_pass_is_the_only_one_in_arrival_order()[
        "the_two_ordered_ones_are_identical"
    ]


def test_a_whole_columns_dictionary_is_its_distinct_set():
    assert dedupe.a_dictionary_stops_being_the_distinct_set_after_a_filter()[
        "the_dictionary_is_the_distinct_set"
    ]


def test_a_filtered_dictionary_overstates():
    assert dedupe.a_dictionary_stops_being_the_distinct_set_after_a_filter()[
        "the_dictionary_now_overstates"
    ]


def test_the_filtered_dictionary_keeps_every_entry():
    made = dedupe.a_dictionary_stops_being_the_distinct_set_after_a_filter()
    assert made["dictionary_after_the_filter"] == made["dictionary_entries"]


def test_the_overstatement_is_large():
    assert (
        dedupe.a_dictionary_stops_being_the_distinct_set_after_a_filter()[
            "by_this_many_entries"
        ]
        > 100
    )


def test_a_sketch_is_smaller_than_the_exact_answer():
    assert dedupe.a_sketch_is_not_cheaper_to_compute()["the_sketch_is_smaller"]


def test_a_sketch_saves_two_orders_of_magnitude_of_memory():
    assert dedupe.a_sketch_is_not_cheaper_to_compute()["by_this_factor"] > 50


def test_the_sketch_error_exceeds_its_expected_error():
    assert dedupe.a_sketch_is_not_cheaper_to_compute()["the_error_is_above_the_expected_one"]


def test_the_sketch_error_is_within_three_deviations():
    assert dedupe.a_sketch_is_not_cheaper_to_compute()["but_within_three_deviations"]


def test_the_byte_crossover_is_five_hundred_and_twelve():
    assert dedupe.the_memory_crossover_is_lower_than_the_threshold()["byte_crossover"] == 512


def test_the_threshold_is_far_above_the_crossover():
    assert dedupe.the_memory_crossover_is_lower_than_the_threshold()[
        "the_threshold_is_far_above_the_crossover"
    ]


def test_exact_wins_at_a_hundred_distinct():
    assert dedupe.the_memory_crossover_is_lower_than_the_threshold()["exact_wins_at_a_hundred"]


def test_exact_loses_by_a_thousand_distinct():
    assert dedupe.the_memory_crossover_is_lower_than_the_threshold()["and_loses_by_a_thousand"]


def test_the_error_does_not_grow_with_the_cardinality():
    assert dedupe.the_memory_crossover_is_lower_than_the_threshold()[
        "the_error_does_not_grow_with_the_cardinality"
    ]


def test_a_correlated_pair_is_far_below_the_product():
    assert dedupe.distinct_rows_are_far_fewer_than_the_product()[
        "the_correlated_pair_is_far_below_the_product"
    ]


def test_an_independent_pair_is_near_the_product():
    assert dedupe.distinct_rows_are_far_fewer_than_the_product()[
        "the_independent_pair_is_near_it"
    ]


def test_the_product_only_holds_under_independence():
    assert dedupe.distinct_rows_are_far_fewer_than_the_product()[
        "the_product_is_only_right_when_they_are_independent"
    ]


def test_distinct_rows_are_capped_by_the_row_count():
    assert dedupe.distinct_rows_are_capped_by_the_row_count()["it_is_capped_at_the_rows"]


def test_the_five_column_product_is_absurd():
    assert dedupe.distinct_rows_are_capped_by_the_row_count()["the_product_is_absurd"]


def test_over_five_columns_every_row_is_distinct():
    assert dedupe.distinct_rows_are_capped_by_the_row_count()["every_row_is_distinct"]


def test_a_sorted_column_needs_fewer_comparisons():
    assert dedupe.a_column_that_is_already_sorted_deduplicates_without_sorting()[
        "the_ordered_one_is_cheaper"
    ]


def test_the_sorted_saving_is_the_log_factor():
    assert (
        dedupe.a_column_that_is_already_sorted_deduplicates_without_sorting()["by_this_factor"]
        > 5
    )


def test_the_adjacent_count_agrees_with_the_sort_pass():
    assert dedupe.a_column_that_is_already_sorted_deduplicates_without_sorting()[
        "and_the_sort_pass_agrees"
    ]


def test_one_null_comes_back():
    assert dedupe.two_nulls_are_one_distinct_value()["one_null_comes_back"]


def test_the_other_values_all_come_back():
    assert dedupe.two_nulls_are_one_distinct_value()["the_other_values_are_all_there"]


def test_the_null_count_is_the_values_plus_one():
    assert dedupe.two_nulls_are_one_distinct_value()["the_count_is_the_values_plus_one"]


def test_the_sort_pass_agrees_about_nulls():
    assert dedupe.two_nulls_are_one_distinct_value()["and_the_sort_pass_agrees"]


def test_the_reduction_falls_with_the_cardinality():
    assert dedupe.the_reduction_is_what_makes_it_worth_doing()[
        "the_reduction_falls_with_the_cardinality"
    ]


def test_ten_values_reduce_almost_everything():
    assert dedupe.the_reduction_is_what_makes_it_worth_doing()["at_ten_values"] > 0.99


def test_a_high_cardinality_column_barely_reduces():
    assert dedupe.the_reduction_is_what_makes_it_worth_doing()["the_last_one_barely_reduces"]


def test_an_empty_column_gives_nothing():
    assert dedupe.an_empty_column_has_no_distinct_values()["both_are_empty"]


def test_an_empty_column_has_no_reduction():
    assert dedupe.an_empty_column_has_no_distinct_values()["the_reduction_is_nothing"]


def test_an_empty_column_is_trivially_ordered():
    assert dedupe.an_empty_column_has_no_distinct_values()["and_it_is_trivially_ordered"]


def test_an_unknown_strategy_is_refused():
    assert dedupe.an_unknown_strategy_is_refused()


def test_the_dictionary_strategy_needs_a_dictionary():
    assert dedupe.a_dictionary_strategy_on_an_integer_is_refused()


def test_deduplicating_on_no_columns_is_refused():
    assert dedupe.deduplicating_on_no_columns_is_refused()


def test_deduplicating_on_a_missing_column_is_refused():
    assert dedupe.deduplicating_on_a_missing_column_is_refused()


def test_the_strategy_table_has_three_rows():
    assert len(dedupe.compare_the_strategies()) == 3


def test_every_strategy_in_the_table_is_named():
    assert [one["strategy"] for one in dedupe.compare_the_strategies()] == list(STRATEGIES)


def test_the_touches_are_identical():
    assert dedupe.the_strategies_differ_in_comparisons_not_in_touches()[
        "the_touches_are_identical"
    ]


def test_only_the_sort_pass_compares():
    assert dedupe.the_strategies_differ_in_comparisons_not_in_touches()[
        "only_the_sort_compares"
    ]


def test_they_all_find_the_same_count():
    assert dedupe.the_strategies_differ_in_comparisons_not_in_touches()[
        "and_they_all_find_the_same_count"
    ]


def test_the_summary_says_they_agree():
    assert dedupe.summarise()["they_all_agree"]


def test_the_summary_reports_the_filtered_dictionary():
    assert dedupe.summarise()["a_filtered_dictionary_overstates"]


def test_a_hash_pass_finds_every_value(numbers):
    assert by_hash(numbers).count == 50


def test_a_sort_pass_finds_every_value(numbers):
    assert by_sort(numbers).count == 50


def test_a_dictionary_pass_finds_every_value(labels):
    assert by_dictionary(labels).count == 100


def test_a_hash_pass_reports_its_strategy(numbers):
    assert by_hash(numbers).strategy == HASH


def test_a_sort_pass_reports_its_strategy(numbers):
    assert by_sort(numbers).strategy == SORT


def test_a_dictionary_pass_reports_its_strategy(labels):
    assert by_dictionary(labels).strategy == DICTIONARY


def test_a_result_reports_its_reduction(numbers):
    assert by_hash(numbers).reduction == 1 - 50 / len(numbers)


def test_a_result_reports_the_rows_it_read(numbers):
    assert by_hash(numbers).rows_in == len(numbers)


def test_a_result_summarises(numbers):
    assert by_hash(numbers).as_dict()["distinct"] == 50


def test_a_sorted_result_is_ordered(numbers):
    assert by_sort(numbers).ordered


def test_a_hashed_result_is_not(numbers):
    assert not by_hash(numbers).ordered


def test_an_empty_result_has_no_reduction():
    made = Distinct(column=integer_column("v", []), strategy=HASH, rows_in=0, values_touched=0)
    assert made.reduction == 0.0


def test_choosing_a_strategy_by_name(numbers):
    assert distinct(numbers, SORT).strategy == SORT


def test_an_unnamed_strategy_raises(numbers):
    with pytest.raises(ConfigError):
        distinct(numbers, "guess")


def test_the_dictionary_strategy_raises_on_an_integer(numbers):
    with pytest.raises(SchemaError):
        by_dictionary(numbers)


def test_a_meter_counts_the_touches(numbers):
    meter = Meter()
    by_hash(numbers, meter)
    assert meter.values_touched == len(numbers)


def test_a_meter_counts_the_materialised_rows(numbers):
    meter = Meter()
    by_hash(numbers, meter)
    assert meter.rows_materialised == 50


def test_the_hash_pass_makes_no_comparisons(numbers):
    meter = Meter()
    by_hash(numbers, meter)
    assert meter.comparisons == 0


def test_the_sort_pass_makes_comparisons(numbers):
    meter = Meter()
    by_sort(numbers, meter)
    assert meter.comparisons > len(numbers)


def test_distinct_rows_removes_duplicates():
    made = Batch.of(a=[1, 1, 2, 2], b=[1, 1, 2, 3])
    assert distinct_rows(made).rows == 3


def test_distinct_rows_keeps_arrival_order():
    made = Batch.of(a=[3, 1, 3, 2], b=[1, 1, 1, 1])
    assert list(distinct_rows(made).values("a")) == [3, 1, 2]


def test_distinct_rows_on_one_column_matches_the_column_pass():
    made = Batch.of(a=[1, 1, 2, 2, 3], b=[9, 8, 7, 6, 5])
    assert distinct_rows(made, ["a"]).rows == 3


def test_distinct_rows_defaults_to_every_column():
    made = Batch.of(a=[1, 1], b=[1, 2])
    assert distinct_rows(made).rows == 2


def test_distinct_rows_on_a_single_row_batch():
    assert distinct_rows(Batch.of(a=[1])).rows == 1


def test_distinct_rows_refuses_an_empty_name_list():
    with pytest.raises(ConfigError):
        distinct_rows(Batch.of(a=[1]), [])


def test_distinct_rows_refuses_a_missing_column():
    with pytest.raises(SchemaError):
        distinct_rows(Batch.of(a=[1]), ["b"])


def test_counting_exactly(numbers):
    assert count_distinct(numbers) == 50.0


def test_counting_approximately(numbers):
    assert 40 < count_distinct(numbers, exact=False) < 60


def test_a_nullable_column_counts_its_null_once():
    values = np.array([1, 2, 3, 4])
    valid = np.array([True, False, True, False])
    one = Column(field=integer_column("v", values).field, values=values, valid=valid)
    assert by_hash(one).count == 3


def test_a_nullable_column_returns_one_null():
    values = np.array([1, 2, 3, 4])
    valid = np.array([True, False, True, False])
    one = Column(field=integer_column("v", values).field, values=values, valid=valid)
    assert by_hash(one).column.to_list().count(None) == 1


def test_a_column_of_all_nulls_has_one_distinct_value():
    values = np.arange(5)
    one = Column(
        field=integer_column("v", values).field, values=values, valid=np.zeros(5, dtype=bool)
    )
    assert by_hash(one).count == 1


def test_the_sketch_threshold_is_a_hundred_thousand():
    assert SKETCH_ABOVE == 100_000


def test_there_are_three_strategies():
    assert len(STRATEGIES) == 3
