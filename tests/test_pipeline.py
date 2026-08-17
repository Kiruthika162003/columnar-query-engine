from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError
from cqe.exec import pipeline
from cqe.exec.aggregate import Aggregate
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.exec.pipeline import (
    BATCH_OVERHEAD,
    BATCH_ROWS,
    aggregate_stage,
    filter_stage,
    in_batches,
    project_stage,
    run,
    run_whole,
    sort_stage,
)
from cqe.exec.sort import SortKey
from cqe.verify.reference import Rows, agree


@pytest.fixture(scope="module")
def batch() -> Batch:
    """A table large enough to slice several ways."""
    state = np.random.default_rng(109)
    rows = 20000
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 40, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 6, rows)]),
        ]
    )


def streaming() -> list:
    """A filter and a projection, which is the shape most of a plan has."""
    return [
        filter_stage(Compare(">", column("amount"), literal(90.0))),
        project_stage(["id", "amount"]),
    ]


def test_batching_does_not_change_the_answer():
    assert pipeline.batching_does_not_change_the_answer()["they_all_agree"]


def test_the_work_barely_changes_with_the_batch_size():
    assert pipeline.the_batch_size_barely_changes_the_work()["the_spread_is_small"]


def test_the_overhead_is_what_moves():
    assert pipeline.the_batch_size_barely_changes_the_work()["and_the_overhead_is_what_moves"]


def test_a_tiny_batch_is_dominated_by_overhead():
    assert pipeline.a_small_batch_is_dominated_by_its_overhead()["at_ten_rows"] > 0.1


def test_the_overhead_share_falls_with_the_batch_size():
    assert pipeline.a_small_batch_is_dominated_by_its_overhead()["the_share_falls"]


def test_a_thousand_rows_is_already_negligible():
    assert pipeline.a_small_batch_is_dominated_by_its_overhead()[
        "and_it_is_negligible_by_a_thousand"
    ]


def test_a_streaming_pipeline_holds_one_batch():
    assert pipeline.a_streaming_pipeline_holds_one_batch()["it_held_one_batch"]


def test_the_unbatched_form_holds_far_more():
    assert pipeline.a_streaming_pipeline_holds_one_batch()["ratio"] > 5


def test_the_batched_and_unbatched_answers_agree():
    assert pipeline.a_streaming_pipeline_holds_one_batch()["and_the_answers_agree"]


def test_a_blocking_stage_holds_the_whole_table():
    assert pipeline.a_blocking_stage_ends_the_streaming()["and_it_is_larger_than_a_batch"]


def test_the_blocking_stage_still_sorts():
    assert pipeline.a_blocking_stage_ends_the_streaming()["it_is_still_sorted"]


def test_an_accumulating_stage_returns_a_row_per_group():
    assert pipeline.an_accumulating_stage_holds_the_groups()["it_returned_a_row_per_group"]


def test_the_group_count_is_far_below_the_row_count():
    assert pipeline.an_accumulating_stage_holds_the_groups()[
        "the_group_count_is_tiny_against_the_rows"
    ]


def test_a_streaming_prefix_holds_a_batch():
    assert pipeline.the_streaming_prefix_is_found_automatically()["streaming_holds_a_batch"]


def test_a_sort_at_the_end_breaks_the_streaming():
    assert pipeline.the_streaming_prefix_is_found_automatically()["a_sort_at_the_end_does_not"]


def test_a_sort_at_the_front_breaks_it_too():
    assert pipeline.the_streaming_prefix_is_found_automatically()[
        "and_a_sort_at_the_front_does_not_either"
    ]


def test_a_selective_filter_shrinks_the_later_batches():
    assert pipeline.a_filter_makes_the_later_batches_smaller()["the_selective_one_holds_less"]


def test_an_empty_result_still_produces_a_batch():
    assert pipeline.an_empty_batch_passes_through()["it_is_empty"]


def test_an_empty_result_keeps_its_schema():
    assert pipeline.an_empty_batch_passes_through()["and_it_kept_its_schema"]


def test_a_large_batch_size_is_one_batch():
    assert pipeline.a_batch_larger_than_the_table_is_one_batch()["it_is_one"]


def test_a_large_batch_matches_the_unbatched_run():
    assert pipeline.a_batch_larger_than_the_table_is_one_batch()["it_matches_the_unbatched_run"]


def test_the_overhead_is_exactly_per_batch():
    assert pipeline.the_overhead_constant_is_what_it_claims()["it_is_per_batch_exactly"]


def test_a_pipeline_with_no_stages_is_refused():
    assert pipeline.a_pipeline_with_no_stages_is_refused()


def test_a_zero_batch_size_is_refused():
    assert pipeline.a_zero_batch_size_is_refused()


def test_there_are_three_kinds_of_stage():
    assert len(pipeline.compare_the_stage_kinds()) == 3


def test_only_one_kind_holds_every_row():
    table = {one["kind"]: one["holds"] for one in pipeline.compare_the_stage_kinds()}
    assert table["blocking"] == "every row" and table["streaming"] == "one batch"


def test_the_summary_says_batching_is_exact():
    assert pipeline.summarise()["batching_is_exact"]


def test_slicing_covers_every_row(batch):
    assert sum(one.rows for one in in_batches(batch, 3000)) == batch.rows


def test_slicing_gives_the_expected_count(batch):
    assert len(list(in_batches(batch, 5000))) == 4


def test_the_last_slice_is_short(batch):
    sizes = [one.rows for one in in_batches(batch, 3000)]
    assert sizes[-1] < 3000


def test_a_slice_larger_than_the_table_is_one(batch):
    assert len(list(in_batches(batch, 100000))) == 1


def test_a_zero_slice_is_refused(batch):
    with pytest.raises(ConfigError):
        list(in_batches(batch, 0))


def test_a_streaming_run_returns_the_filtered_rows(batch):
    made = run(batch, streaming(), batch_rows=2048)
    assert 0 < made.rows < batch.rows


def test_a_streaming_run_narrows_the_schema(batch):
    made = run(batch, streaming(), batch_rows=2048)
    assert list(made.batch.schema.names) == ["id", "amount"]


def test_a_streaming_run_matches_the_whole_run(batch):
    made = run(batch, streaming(), batch_rows=2048)
    whole = run_whole(batch, streaming())
    assert agree(Rows.of(made.batch), Rows.of(whole.batch), ordered=True)


def test_a_run_counts_its_batches(batch):
    assert run(batch, streaming(), batch_rows=5000).batches == 4


def test_a_run_reports_its_input_rows(batch):
    assert run(batch, streaming(), batch_rows=5000).rows_in == batch.rows


def test_a_run_summarises(batch):
    summary = run(batch, streaming(), batch_rows=5000).as_dict()
    assert summary["batches"] == 4 and summary["rows_in"] == batch.rows


def test_a_run_reports_its_overhead(batch):
    made = run(batch, streaming(), batch_rows=5000)
    assert made.overhead == made.batches * BATCH_OVERHEAD


def test_a_run_reports_its_peak(batch):
    assert run(batch, streaming(), batch_rows=2048).peak_rows <= 2048


def test_a_sort_stage_is_blocking():
    assert not sort_stage([SortKey(name="a")]).streaming


def test_a_filter_stage_is_streaming():
    assert filter_stage(Compare(">", column("a"), literal(1))).streaming


def test_a_projection_stage_is_streaming():
    assert project_stage(["a"]).streaming


def test_an_aggregate_stage_is_not_streaming():
    assert not aggregate_stage(["a"], [Aggregate("n", "count_star", "")]).streaming


def test_a_stage_summarises():
    assert project_stage(["a"]).as_dict()["kind"] == "streaming"


def test_a_sort_pipeline_returns_sorted_rows(batch):
    made = run(batch, [sort_stage([SortKey(name="amount")])], batch_rows=2048)
    assert np.all(np.diff(made.batch.column("amount").values) >= 0)


def test_an_aggregate_pipeline_returns_its_groups(batch):
    stages = [aggregate_stage(["shop"], [Aggregate("n", "count_star", "")])]
    made = run(batch, stages, batch_rows=4096)
    assert made.rows == len(set(batch.column("shop").to_list()))


def test_an_aggregate_after_a_filter_agrees_with_the_whole_run(batch):
    stages = [
        filter_stage(Compare(">", column("amount"), literal(100.0))),
        aggregate_stage(["shop"], [Aggregate("n", "count_star", "")]),
    ]
    made = run(batch, stages, batch_rows=1024)
    whole = run_whole(batch, stages)
    assert agree(Rows.of(made.batch), Rows.of(whole.batch))


def test_the_meter_is_shared_across_batches(batch):
    meter = Meter()
    run(batch, streaming(), batch_rows=2048, meter=meter)
    assert meter.values_touched > batch.rows


def test_the_default_batch_size_is_a_power_of_two():
    assert BATCH_ROWS == 4096


def test_a_pipeline_of_one_stage_works(batch):
    made = run(batch, [project_stage(["id"])], batch_rows=4096)
    assert made.rows == batch.rows and made.batch.width == 1


def test_two_filters_compose(batch):
    stages = [
        filter_stage(Compare(">", column("amount"), literal(90.0))),
        filter_stage(Compare("<", column("shop"), literal(20))),
    ]
    made = run(batch, stages, batch_rows=2048)
    whole = run_whole(batch, stages)
    assert made.rows == whole.rows


def test_an_empty_table_produces_an_empty_result(batch):
    made = run(batch.slice(0, 0), streaming(), batch_rows=1024)
    assert made.rows == 0
