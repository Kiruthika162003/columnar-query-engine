from __future__ import annotations

import itertools

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, SchemaError, UnknownColumn
from cqe.exec import window as windows
from cqe.exec.batch import Batch
from cqe.exec.sort import SortKey
from cqe.exec.window import FUNCTIONS, Window, apply, evaluate
from cqe.types.schema import FLOATING, INTEGER, STRING


@pytest.fixture(scope="module")
def batch() -> Batch:
    """A table with a partition column, an ordering column and a value."""
    state = np.random.default_rng(61)
    rows = 1500
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 8, rows)),
            integer_column("day", state.integers(0, 20, rows)),
            floating_column("amount", state.normal(100, 20, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 3, rows)]),
        ]
    )


ORDER = (SortKey(name="day"), SortKey(name="id"))


def test_every_function_agrees_with_the_reference():
    assert windows.every_function_agrees_with_the_reference()["they_all_agree"]


def test_row_number_agrees():
    assert windows.every_function_agrees_with_the_reference()["row_number"]


def test_rank_agrees():
    assert windows.every_function_agrees_with_the_reference()["rank"]


def test_dense_rank_agrees():
    assert windows.every_function_agrees_with_the_reference()["dense_rank"]


def test_the_running_sum_agrees():
    assert windows.every_function_agrees_with_the_reference()["running_sum"]


def test_lag_and_lead_agree():
    measured = windows.every_function_agrees_with_the_reference()
    assert measured["lag"] and measured["lead"]


def test_row_number_never_ties():
    assert windows.the_three_rankings_differ_on_ties()["row_number_never_ties"]


def test_rank_ties_and_skips():
    assert windows.the_three_rankings_differ_on_ties()["rank_ties_and_skips"]


def test_dense_rank_ties_without_skipping():
    assert windows.the_three_rankings_differ_on_ties()["dense_rank_ties_without_skipping"]


def test_a_window_keeps_every_row():
    assert windows.a_window_does_not_collapse_its_input()["the_window_kept_every_row"]


def test_an_aggregate_does_not():
    assert windows.a_window_does_not_collapse_its_input()["the_aggregate_collapsed_them"]


def test_the_running_sum_ends_at_the_group_total():
    assert windows.the_running_sum_ends_at_the_group_total()["they_agree"]


def test_a_neighbour_stays_inside_its_partition():
    assert windows.a_neighbour_does_not_cross_a_partition()["one_null_per_partition"]


def test_lag_and_lead_have_the_same_null_count():
    assert windows.a_lead_is_a_lag_backwards()["they_have_the_same_null_count"]


def test_no_partition_means_one_partition():
    assert windows.a_window_with_no_partition_is_one_partition()["it_counts_the_whole_table"]


def test_the_running_total_ends_at_the_table_total():
    assert windows.a_window_with_no_partition_is_one_partition()["the_sum_ends_at_the_total"]


def test_the_result_comes_back_in_the_input_order():
    assert windows.the_result_comes_back_in_the_input_order()["the_input_did_not_move"]


def test_two_windows_can_be_added_at_once():
    assert windows.the_result_comes_back_in_the_input_order()["it_added_two"]


def test_the_vectorised_path_is_a_few_passes():
    assert windows.the_vectorised_path_beats_the_reference()["it_is_a_few_passes"]


def test_a_string_column_keeps_its_dictionary():
    assert windows.a_string_column_can_be_lagged()["it_kept_the_dictionary"]


def test_a_lagged_string_is_a_real_value():
    assert windows.a_string_column_can_be_lagged()["and_they_are_real_regions"]


def test_an_offset_of_two_nulls_twice_as_many():
    assert windows.an_offset_of_two_looks_two_rows_back()["it_is_twice_as_many"]


def test_a_descending_order_ranks_the_largest_first():
    assert windows.a_descending_order_reverses_the_ranking()["it_is_the_largest"]


def test_an_ascending_order_ranks_the_smallest_first():
    assert windows.a_descending_order_reverses_the_ranking()["it_is_the_smallest"]


def test_an_empty_batch_gives_an_empty_column():
    assert windows.an_empty_batch_produces_an_empty_column()["it_is_empty"]


def test_single_row_partitions_all_number_one():
    assert windows.a_partition_of_one_row_works()["every_number_is_one"]


def test_single_row_partitions_have_no_neighbours():
    assert windows.a_partition_of_one_row_works()["every_neighbour_is_null"]


def test_an_unknown_function_is_refused():
    assert windows.an_unknown_function_is_refused()


def test_a_running_sum_without_a_source_is_refused():
    assert windows.a_running_sum_without_a_source_is_refused()


def test_a_rank_without_an_order_is_refused():
    assert windows.a_rank_without_an_order_is_refused()


def test_a_zero_offset_is_refused():
    assert windows.a_zero_offset_is_refused()


def test_a_missing_source_is_refused():
    assert windows.a_missing_source_column_is_refused()


def test_a_missing_partition_is_refused():
    assert windows.a_missing_partition_column_is_refused()


def test_a_colliding_name_is_refused():
    assert windows.a_name_that_already_exists_is_refused()


def test_every_function_in_the_table_agrees():
    assert all(one["agrees"] for one in windows.compare_the_functions())


def test_only_the_neighbours_are_nullable():
    table = {one["function"]: one["nullable"] for one in windows.compare_the_functions()}
    assert table["lag"] and table["lead"] and not table["row_number"]


def test_the_summary_says_they_all_agree():
    assert windows.summarise()["all_agree"]


def test_a_row_number_starts_at_one(batch):
    made = evaluate(batch, Window("v", "row_number", partition=("shop",), order=ORDER))
    assert min(made.to_list()) == 1


def test_a_row_number_never_exceeds_its_partition(batch):
    made = evaluate(batch, Window("v", "row_number", partition=("shop",), order=ORDER))
    largest = max(made.to_list())
    counts = {}
    for one in batch.column("shop").to_list():
        counts[one] = counts.get(one, 0) + 1
    assert largest == max(counts.values())


def test_a_row_number_is_an_integer(batch):
    made = evaluate(batch, Window("v", "row_number", partition=("shop",), order=ORDER))
    assert made.field.logical == INTEGER


def test_a_running_sum_is_floating(batch):
    made = evaluate(batch, Window("v", "running_sum", "amount", ("shop",), ORDER))
    assert made.field.logical == FLOATING


def test_a_lagged_string_stays_a_string(batch):
    made = evaluate(batch, Window("v", "lag", "region", ("shop",), ORDER))
    assert made.field.logical == STRING


def test_a_running_sum_never_decreases_on_positive_values():
    positive = Batch.from_columns(
        [
            integer_column("shop", [0] * 100),
            integer_column("day", list(range(100))),
            floating_column("amount", [1.0] * 100),
        ]
    )
    made = evaluate(positive, Window("v", "running_sum", "amount", ("shop",), ORDER[:1]))
    values = made.to_list()
    assert values == sorted(values)


def test_a_running_max_never_decreases(batch):
    made = evaluate(batch, Window("v", "running_max", "amount", ("shop",), ORDER))
    rows = apply(batch, [Window("v", "running_max", "amount", ("shop",), ORDER)])
    ordered = sorted(
        zip(
            rows.column("shop").to_list(),
            rows.column("day").to_list(),
            rows.column("id").to_list(),
            rows.column("v").to_list(),
            strict=True,
        )
    )
    assert all(
        one[3] <= other[3] for one, other in itertools.pairwise(ordered) if one[0] == other[0]
    )
    assert len(made) == batch.rows


def test_a_running_max_ends_at_the_partition_maximum(batch):
    rows = apply(batch, [Window("v", "running_max", "amount", ("shop",), ORDER)])
    highest = {}
    for shop, amount in zip(
        rows.column("shop").to_list(), rows.column("amount").to_list(), strict=True
    ):
        highest[shop] = max(highest.get(shop, float("-inf")), amount)
    seen = {}
    for shop, value in zip(
        rows.column("shop").to_list(), rows.column("v").to_list(), strict=True
    ):
        seen[shop] = max(seen.get(shop, float("-inf")), value)
    assert all(abs(seen[one] - highest[one]) < 1e-9 for one in highest)


def test_applying_no_windows_returns_the_batch(batch):
    assert apply(batch, []).width == batch.width


def test_applying_two_windows_adds_two_columns(batch):
    made = apply(
        batch,
        [
            Window("a", "row_number", partition=("shop",), order=ORDER),
            Window("b", "running_sum", "amount", ("shop",), ORDER),
        ],
    )
    assert made.width == batch.width + 2


def test_a_window_describes_itself():
    made = Window("v", "running_sum", "amount", ("shop",), ORDER)
    assert made.describe().startswith("running_sum(amount)")


def test_a_window_summarises():
    made = Window("v", "running_sum", "amount", ("shop",), ORDER)
    assert made.as_dict()["partition"] == ["shop"]


def test_a_window_needs_an_order():
    assert Window("v", "row_number", order=ORDER).needs_order


def test_the_meter_counts_the_rows(batch):
    meter = Meter()
    evaluate(batch, Window("v", "row_number", partition=("shop",), order=ORDER), meter=meter)
    assert meter.values_touched >= batch.rows


def test_every_function_name_is_known():
    assert len(FUNCTIONS) == 7


def test_a_missing_order_column_is_refused(batch):
    with pytest.raises(UnknownColumn):
        evaluate(batch, Window("v", "row_number", order=(SortKey(name="nothing"),)))


def test_a_shadowing_name_is_refused(batch):
    with pytest.raises(SchemaError):
        evaluate(batch, Window("amount", "row_number", order=ORDER))


def test_a_negative_offset_is_refused():
    with pytest.raises(ConfigError):
        Window("v", "lag", "amount", order=ORDER, offset=-1)


def test_a_partition_by_a_string_works(batch):
    made = evaluate(batch, Window("v", "row_number", partition=("region",), order=ORDER))
    assert max(made.to_list()) < batch.rows


def test_two_partition_columns_work(batch):
    made = evaluate(batch, Window("v", "row_number", partition=("shop", "region"), order=ORDER))
    assert max(made.to_list()) < batch.rows


def test_a_window_over_one_row_is_one(batch):
    single = batch.slice(0, 1)
    made = evaluate(single, Window("v", "row_number", partition=("shop",), order=ORDER))
    assert made.to_list() == [1]
