from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import BudgetExceeded, ConfigError
from cqe.exec import spill
from cqe.exec.aggregate import Aggregate
from cqe.exec.batch import Batch
from cqe.exec.sort import SortKey
from cqe.exec.spill import (
    Bounded,
    Budget,
    Spilled,
    external_aggregate,
    external_sort,
    in_batches,
    merge_runs,
    spill_partitions,
    spill_sorted_runs,
)
from cqe.storage.file import read
from cqe.verify.reference import Rows, agree, group_by
from cqe.verify.reference import order_by as reference_order


@pytest.fixture(scope="module")
def batch() -> Batch:
    """A table large enough to spill at the run sizes used here."""
    state = np.random.default_rng(37)
    rows = 6000
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 120, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("label", [f"kind{one}" for one in state.integers(0, 8, rows)]),
        ]
    )


def test_an_external_sort_agrees_with_the_reference():
    assert spill.an_external_sort_gives_the_same_order()["they_agree"]


def test_the_external_sort_really_spilled():
    assert spill.an_external_sort_gives_the_same_order()["it_spilled"]


def test_a_run_never_holds_the_whole_table():
    assert spill.an_external_sort_holds_one_run_at_a_time()["it_never_held_the_table"]


def test_the_run_is_much_smaller_than_the_table():
    assert spill.an_external_sort_holds_one_run_at_a_time()["the_ratio"] > 5


def test_the_memory_curve_has_a_minimum_in_the_middle():
    assert spill.a_smaller_run_makes_more_runs()["the_smallest_total_is_in_the_middle"]


def test_the_minimum_is_at_the_square_root():
    assert spill.a_smaller_run_makes_more_runs()["and_it_is_at_the_square_root"]


def test_an_external_aggregate_agrees_with_the_reference():
    assert spill.an_external_aggregate_gives_the_same_groups()["they_agree"]


def test_the_external_aggregate_produced_every_group():
    measured = spill.an_external_aggregate_gives_the_same_groups()
    assert measured["groups"] == measured["expected"]


def test_a_group_lands_in_exactly_one_partition():
    assert spill.every_group_lands_in_one_partition()["it_holds"]


def test_the_partitions_are_roughly_even():
    assert spill.the_partitions_are_roughly_even()["it_is_within_a_third"]


def test_a_skewed_key_survives_the_hash():
    assert spill.a_skewed_key_makes_an_uneven_partition()["the_skew_survives_the_hash"]


def test_more_partitions_do_not_fix_skew():
    assert spill.more_partitions_do_not_fix_skew()["quadrupling_the_partitions_barely_helps"]


def test_the_largest_partition_cannot_beat_the_largest_group():
    assert spill.more_partitions_do_not_fix_skew()["it_never_goes_below_the_biggest_group"]


def test_a_budget_refuses_before_the_allocation():
    assert spill.a_budget_refuses_before_it_runs_out()["it_refused"]


def test_the_refusal_names_both_numbers():
    assert spill.a_budget_refuses_before_it_runs_out()["it_names_both_numbers"]


def test_a_bounded_sort_does_not_spill_when_it_fits():
    assert spill.a_bounded_sort_spills_only_when_it_has_to()["the_small_one_did_not_spill"]


def test_a_bounded_sort_spills_when_it_does_not():
    assert spill.a_bounded_sort_spills_only_when_it_has_to()["the_large_one_did"]


def test_both_bounded_sorts_are_sorted():
    assert spill.a_bounded_sort_spills_only_when_it_has_to()["both_are_sorted"]


def test_a_bounded_aggregate_does_not_spill_when_it_fits():
    assert spill.a_bounded_aggregate_spills_only_when_it_has_to()["the_small_one_did_not_spill"]


def test_a_bounded_aggregate_spills_when_it_does_not():
    assert spill.a_bounded_aggregate_spills_only_when_it_has_to()["the_large_one_did"]


def test_spilling_writes_about_the_data_size():
    assert 0.5 < spill.spilling_costs_bytes_written()["overhead"] < 2


def test_a_single_run_needs_no_merge():
    assert spill.a_merge_of_one_run_is_a_read()["it_is_one"]


def test_a_single_run_still_comes_back_sorted():
    assert spill.a_merge_of_one_run_is_a_read()["it_is_sorted"]


def test_a_descending_string_merge_is_refused():
    assert spill.a_descending_string_merge_is_refused()


def test_a_zero_run_size_is_refused():
    assert spill.a_zero_run_size_is_refused()


def test_merging_no_runs_is_refused():
    assert spill.merging_nothing_is_refused()


def test_partitioning_by_a_missing_column_is_refused():
    assert spill.partitioning_by_a_missing_column_is_refused()


def test_a_zero_batch_size_is_refused():
    assert spill.a_zero_batch_size_is_refused()


def test_the_two_spills_differ_on_skew():
    table = spill.compare_the_two_spills()
    assert table[0]["survives_skew"] and not table[1]["survives_skew"]


def test_the_summary_reports_both_agreements():
    summary = spill.summarise()
    assert summary["sort_agrees"] and summary["aggregate_agrees"]


def test_an_external_sort_returns_every_row(batch, tmp_path):
    produced = external_sort(batch, [SortKey(name="amount")], tmp_path, run_rows=500)
    assert produced.rows == batch.rows


def test_an_external_sort_is_ordered(batch, tmp_path):
    produced = external_sort(batch, [SortKey(name="amount")], tmp_path, run_rows=500)
    assert np.all(np.diff(produced.column("amount").values) >= 0)


def test_an_external_sort_matches_the_reference(batch, tmp_path):
    produced = external_sort(batch, [SortKey(name="amount")], tmp_path, run_rows=500)
    expected = reference_order(Rows.of(batch), ["amount"])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_an_external_sort_on_two_keys_matches(batch, tmp_path):
    keys = [SortKey(name="shop"), SortKey(name="amount")]
    produced = external_sort(batch, keys, tmp_path, run_rows=500)
    expected = reference_order(Rows.of(batch), ["shop", "amount"])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_a_descending_numeric_sort_matches(batch, tmp_path):
    keys = [SortKey(name="amount", descending=True)]
    produced = external_sort(batch, keys, tmp_path, run_rows=500)
    expected = reference_order(Rows.of(batch), ["amount"], descending=[True])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_the_run_count_follows_the_run_size(batch, tmp_path):
    spilled = spill_sorted_runs(batch, [SortKey(name="amount")], tmp_path, run_rows=1000)
    assert spilled.runs == 6


def test_every_run_is_sorted(batch, tmp_path):
    spilled = spill_sorted_runs(batch, [SortKey(name="amount")], tmp_path, run_rows=1000)
    assert all(np.all(np.diff(read(one).column("amount").values) >= 0) for one in spilled.paths)


def test_a_spilled_set_reports_its_rows(batch, tmp_path):
    spilled = spill_sorted_runs(batch, [SortKey(name="amount")], tmp_path, run_rows=1000)
    assert spilled.rows == batch.rows


def test_a_spilled_set_summarises(batch, tmp_path):
    spilled = spill_sorted_runs(batch, [SortKey(name="amount")], tmp_path, run_rows=1000)
    assert spilled.as_dict()["kind"] == "run"


def test_reading_every_run_back_gives_the_rows(batch, tmp_path):
    spilled = spill_sorted_runs(batch, [SortKey(name="amount")], tmp_path, run_rows=1000)
    assert sum(one.rows for one in spilled.read_all()) == batch.rows


def test_an_external_aggregate_matches_the_reference(batch, tmp_path):
    aggregates = [Aggregate(name="total", function="sum", source="amount")]
    produced = external_aggregate(batch, "shop", aggregates, tmp_path)
    expected = group_by(Rows.of(batch), ["shop"], [("total", "sum", "amount")])
    assert agree(Rows.of(produced), expected)


def test_an_external_count_matches_the_reference(batch, tmp_path):
    aggregates = [Aggregate(name="n", function="count_star", source="")]
    produced = external_aggregate(batch, "shop", aggregates, tmp_path)
    expected = group_by(Rows.of(batch), ["shop"], [("n", "count_star", "")])
    assert agree(Rows.of(produced), expected)


def test_an_external_aggregate_on_a_string_key_matches(batch, tmp_path):
    aggregates = [Aggregate(name="total", function="sum", source="amount")]
    produced = external_aggregate(batch, "label", aggregates, tmp_path)
    expected = group_by(Rows.of(batch), ["label"], [("total", "sum", "amount")])
    assert agree(Rows.of(produced), expected)


def test_the_partitions_cover_every_row(batch, tmp_path):
    spilled = spill_partitions(batch, "shop", tmp_path)
    assert sum(one.rows for one in spilled.read_all()) == batch.rows


def test_a_partitioned_set_reports_its_kind(batch, tmp_path):
    assert spill_partitions(batch, "shop", tmp_path).kind == "partition"


def test_one_partition_holds_everything(batch, tmp_path):
    spilled = spill_partitions(batch, "shop", tmp_path, partitions=1)
    assert spilled.runs == 1


def test_a_zero_partition_count_is_refused(batch, tmp_path):
    with pytest.raises(ConfigError):
        spill_partitions(batch, "shop", tmp_path, partitions=0)


def test_a_budget_tracks_its_peak():
    budget = Budget(rows=100)
    budget.take(40)
    budget.take(30)
    assert budget.peak == 70


def test_a_budget_releases():
    budget = Budget(rows=100)
    budget.take(80)
    budget.release(80)
    budget.take(80)
    assert budget.peak == 80


def test_a_budget_reports_that_it_fits():
    budget = Budget(rows=100)
    budget.take(50)
    assert budget.fits


def test_a_budget_refuses_a_single_oversized_take():
    with pytest.raises(BudgetExceeded):
        Budget(rows=10).take(11)


def test_a_budget_summarises():
    budget = Budget(rows=100)
    budget.take(20)
    assert budget.as_dict()["budget"] == 100


def test_batches_cover_every_row(batch):
    assert sum(one.rows for one in in_batches(batch, 700)) == batch.rows


def test_the_last_batch_is_short(batch):
    sizes = [one.rows for one in in_batches(batch, 700)]
    assert sizes[-1] < 700


def test_a_batch_size_larger_than_the_table_is_one_batch(batch):
    assert len(list(in_batches(batch, 100000))) == 1


def test_a_bounded_sort_of_a_small_table_is_exact(batch, tmp_path):
    bounded = Bounded(budget=Budget(rows=100000))
    produced = bounded.sort(batch, [SortKey(name="amount")], tmp_path)
    expected = reference_order(Rows.of(batch), ["amount"])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_a_bounded_sort_of_a_large_table_is_exact(batch, tmp_path):
    bounded = Bounded(budget=Budget(rows=800))
    produced = bounded.sort(batch, [SortKey(name="amount")], tmp_path)
    expected = reference_order(Rows.of(batch), ["amount"])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_a_bounded_aggregate_of_a_large_group_count_is_exact(batch, tmp_path):
    bounded = Bounded(budget=Budget(rows=10))
    aggregates = [Aggregate(name="total", function="sum", source="amount")]
    produced = bounded.aggregate(batch, "shop", aggregates, tmp_path)
    expected = group_by(Rows.of(batch), ["shop"], [("total", "sum", "amount")])
    assert agree(Rows.of(produced), expected)


def test_a_bounded_sort_records_its_spill(batch, tmp_path):
    bounded = Bounded(budget=Budget(rows=800))
    bounded.sort(batch, [SortKey(name="amount")], tmp_path)
    assert bounded.meter.spilled_bytes > 0


def test_merging_an_empty_spilled_set_is_refused():
    with pytest.raises(ConfigError):
        merge_runs(Spilled(paths=(), rows=0, bytes_written=0), [SortKey(name="a")])


def test_a_spilled_run_path_exists(batch, tmp_path):
    spilled = spill_sorted_runs(batch, [SortKey(name="amount")], tmp_path, run_rows=1000)
    assert all(Path(one).exists() for one in spilled.paths)
