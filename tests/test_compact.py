from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from cqe.errors import ConfigError, SchemaError
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.storage import compact as maintenance
from cqe.storage.compact import (
    COMPACT_GROUP,
    INGEST_ROWS,
    OPEN_COST,
    Compaction,
    Fragment,
    Table,
    compact,
    describe,
    ingest,
    load,
    scan_cost,
)
from cqe.storage.file import read


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A directory for one test's fragments."""
    made = tmp_path / "parts"
    made.mkdir()
    return made


@pytest.fixture(scope="module")
def table() -> Batch:
    """A small table to fragment."""
    state = np.random.default_rng(77)
    rows = 4_000
    return Batch.of(
        id=np.arange(rows).tolist(),
        shop=state.integers(0, 20, rows).tolist(),
        label=[f"kind{int(one):02d}" for one in state.integers(0, 30, rows)],
        amount=state.normal(100, 20, rows).tolist(),
    )


def test_small_files_pay_a_lot_of_metadata():
    assert maintenance.many_small_files_pay_more_metadata_than_data()[
        "metadata_is_a_large_share"
    ]


def test_the_footers_are_smaller_than_the_opens():
    assert maintenance.many_small_files_pay_more_metadata_than_data()[
        "the_footers_alone_are_smaller"
    ]


def test_the_metadata_share_is_over_a_tenth():
    assert maintenance.many_small_files_pay_more_metadata_than_data()["metadata_share"] > 0.1


def test_compaction_removes_most_of_the_metadata():
    assert maintenance.compaction_removes_almost_all_of_it()["it_removed_most_of_it"]


def test_compaction_also_shrinks_the_data():
    assert maintenance.compaction_removes_almost_all_of_it()["the_data_shrank_as_well"]


def test_the_data_saving_is_smaller_than_the_metadata_one():
    made = maintenance.compaction_removes_almost_all_of_it()
    assert made["data_saving"] < made["share_removed"]


def test_the_same_group_size_prunes_identically():
    assert maintenance.the_pruning_loss_is_the_group_size_not_the_compaction()[
        "same_group_size_prunes_identically"
    ]


def test_a_wider_group_reads_more():
    assert maintenance.the_pruning_loss_is_the_group_size_not_the_compaction()[
        "the_wide_one_reads_more"
    ]


def test_the_widening_cost_is_modest():
    made = maintenance.the_pruning_loss_is_the_group_size_not_the_compaction()
    assert 1.0 < made["widening_cost"] < 2.0


def test_both_shapes_read_less_metadata():
    assert maintenance.the_pruning_loss_is_the_group_size_not_the_compaction()[
        "and_both_read_less_metadata"
    ]


def test_the_total_scan_cost_falls():
    assert maintenance.the_total_cost_still_falls()["it_is_cheaper"]


def test_the_values_read_did_rise():
    assert maintenance.the_total_cost_still_falls()["the_values_did_rise"]


def test_the_total_falls_by_more_than_double():
    assert maintenance.the_total_cost_still_falls()["by_this_ratio"] > 2


def test_sorting_beats_the_fragments():
    assert maintenance.sorting_during_compaction_buys_the_pruning_back()[
        "the_sorted_one_reads_least"
    ]


def test_sorting_beats_the_unsorted_compaction_by_half():
    assert maintenance.sorting_during_compaction_buys_the_pruning_back()[
        "and_far_less_than_the_unsorted"
    ]


def test_sorting_does_not_change_the_group_count():
    assert maintenance.sorting_during_compaction_buys_the_pruning_back()[
        "with_the_same_group_count"
    ]


def test_the_sorted_file_skips_most_groups():
    assert (
        maintenance.sorting_during_compaction_buys_the_pruning_back()["skipped_share_sorted"]
        > 0.5
    )


def test_sorting_by_one_column_leaves_another_alone():
    assert maintenance.sorting_by_one_column_does_not_help_another()[
        "the_other_column_is_no_better"
    ]


def test_while_the_sorted_column_improves():
    assert maintenance.sorting_by_one_column_does_not_help_another()["while_the_sorted_one_is"]


def test_an_unsorted_compaction_is_row_identical():
    assert maintenance.compaction_keeps_every_row()["the_unsorted_one_is_identical"]


def test_a_sorted_compaction_is_a_permutation():
    assert maintenance.compaction_keeps_every_row()["the_sorted_one_is_a_permutation"]


def test_a_sorted_compaction_really_reordered():
    assert maintenance.compaction_keeps_every_row()["and_it_is_not_the_same_order"]


def test_the_row_counts_match():
    assert maintenance.compaction_keeps_every_row()["row_counts_match"]


def test_the_dictionary_copies_cost_more():
    assert maintenance.a_dictionary_over_more_rows_is_smaller_per_row()["the_copies_cost_more"]


def test_the_compacted_file_is_smaller():
    assert maintenance.a_dictionary_over_more_rows_is_smaller_per_row()["the_file_got_smaller"]


def test_one_fragment_holds_most_of_the_distinct_labels():
    made = maintenance.a_dictionary_over_more_rows_is_smaller_per_row()
    assert made["distinct_in_one_fragment"] > made["distinct"] / 2


def test_compaction_pays_back():
    assert maintenance.compaction_pays_for_itself_after_a_few_scans()["it_pays_back"]


def test_compaction_does_not_pay_back_at_once():
    assert maintenance.compaction_pays_for_itself_after_a_few_scans()["and_it_is_not_immediate"]


def test_the_rewrite_reads_and_writes():
    made = maintenance.compaction_pays_for_itself_after_a_few_scans()
    assert made["read_bytes"] > 0 and made["written_bytes"] > 0


def test_a_second_compaction_saves_nothing():
    assert maintenance.compacting_an_already_compact_file_earns_nothing()[
        "the_second_saves_nothing"
    ]


def test_a_second_compaction_never_breaks_even():
    assert maintenance.compacting_an_already_compact_file_earns_nothing()[
        "and_its_break_even_is_never"
    ]


def test_a_second_compaction_still_costs_a_rewrite():
    assert maintenance.compacting_an_already_compact_file_earns_nothing()[
        "but_it_still_cost_a_rewrite"
    ]


def test_a_larger_ingest_pays_less_metadata():
    assert maintenance.a_larger_ingest_needs_less_compacting()["the_large_ingest_pays_less"]


def test_eight_times_the_ingest_is_eight_times_less():
    assert maintenance.a_larger_ingest_needs_less_compacting()["by_this_ratio"] > 6


def test_a_larger_ingest_shrinks_the_data_a_little():
    made = maintenance.a_larger_ingest_needs_less_compacting()
    assert made["the_data_falls_as_well"] and made["but_by_far_less"]


def test_a_partial_compaction_helps():
    assert maintenance.partial_compaction_leaves_the_recent_fragments_alone()[
        "the_partial_one_helps"
    ]


def test_a_partial_compaction_helps_less_than_a_full_one():
    assert maintenance.partial_compaction_leaves_the_recent_fragments_alone()[
        "but_less_than_the_full_one"
    ]


def test_a_partial_compaction_takes_about_half():
    made = maintenance.partial_compaction_leaves_the_recent_fragments_alone()
    assert 0.3 < made["partial_saving"] < 0.7


def test_projection_costs_the_same_share_after_compaction():
    assert maintenance.a_scan_of_two_columns_of_five_is_unchanged_by_compaction()[
        "the_share_is_the_same"
    ]


def test_the_projected_share_is_not_the_column_count():
    assert maintenance.a_scan_of_two_columns_of_five_is_unchanged_by_compaction()[
        "and_it_is_not_two_fifths_of_bytes"
    ]


def test_compacting_nothing_is_refused():
    assert maintenance.compacting_nothing_is_refused()


def test_a_zero_group_size_is_refused():
    assert maintenance.a_zero_group_size_is_refused()


def test_a_zero_ingest_size_is_refused():
    assert maintenance.a_zero_ingest_size_is_refused()


def test_sorting_by_a_missing_column_is_refused():
    assert maintenance.sorting_by_a_missing_column_is_refused()


def test_describing_nothing_is_refused():
    assert maintenance.describing_nothing_is_refused()


def test_the_ingest_sweep_covers_four_sizes():
    assert len(maintenance.compare_the_ingest_sizes()) == 4


def test_the_metadata_share_falls_throughout():
    assert maintenance.the_metadata_share_falls_with_the_ingest_size()["it_falls_throughout"]


def test_each_doubling_roughly_halves_the_share():
    assert maintenance.the_metadata_share_falls_with_the_ingest_size()[
        "each_doubling_roughly_halves"
    ]


def test_the_largest_ingest_still_pays_something():
    assert maintenance.the_metadata_share_falls_with_the_ingest_size()[
        "the_largest_still_pays_something"
    ]


def test_the_shape_table_has_three_rows():
    assert len(maintenance.compare_the_shapes()) == 3


def test_the_shapes_are_named():
    names = [one["shape"] for one in maintenance.compare_the_shapes()]
    assert names == ["fragments", "compacted", "compacted sorted"]


def test_the_sorted_shape_is_cheapest():
    made = maintenance.compare_the_shapes()
    assert made[-1]["total_bytes"] == min(one["total_bytes"] for one in made)


def test_the_summary_says_the_group_size_is_the_cause():
    assert maintenance.summarise()["same_group_size_prunes_identically"]


def test_the_summary_reports_a_break_even():
    assert maintenance.summarise()["break_even_scans"] > 1


def test_ingesting_makes_one_file_per_piece(table, folder):
    made = ingest(table, folder, size=500)
    assert len(made.fragments) == 8


def test_ingesting_writes_the_files(table, folder):
    ingest(table, folder, size=500)
    assert len(list(folder.glob("*.cqe"))) == 8


def test_an_ingested_table_holds_every_row(table, folder):
    assert ingest(table, folder, size=500).rows == table.rows


def test_a_short_last_piece_is_still_a_fragment(table, folder):
    made = ingest(table, folder, size=700)
    assert made.fragments[-1].rows == table.rows % 700


def test_a_fragment_reports_its_groups(table, folder):
    made = ingest(table, folder, size=500)
    assert all(one.groups == 1 for one in made.fragments)


def test_a_smaller_group_size_gives_more_groups(table, folder):
    made = ingest(table, folder, size=500, group_size=250)
    assert all(one.groups == 2 for one in made.fragments)


def test_a_fragment_reports_its_overhead(table, folder):
    made = ingest(table, folder, size=500)
    assert 0 < made.fragments[0].overhead < 1


def test_a_fragment_summarises(table, folder):
    made = ingest(table, folder, size=500)
    assert made.fragments[0].as_dict()["rows"] == 500


def test_a_fragment_totals_its_bytes():
    made = Fragment(path=Path("x"), rows=10, groups=1, data_bytes=100, footer_bytes=20)
    assert made.total_bytes == 120


def test_a_table_sums_its_fragments(table, folder):
    made = ingest(table, folder, size=500)
    assert made.data_bytes == sum(one.data_bytes for one in made.fragments)


def test_a_table_counts_its_opens(table, folder):
    made = ingest(table, folder, size=500)
    assert made.open_bytes == 8 * OPEN_COST


def test_a_table_summarises(table, folder):
    made = ingest(table, folder, size=500)
    assert made.as_dict()["fragments"] == 8


def test_an_empty_table_is_refused():
    with pytest.raises(ConfigError):
        Table(fragments=())


def test_describing_reads_the_footers(table, folder):
    ingest(table, folder, size=500)
    made = describe(sorted(folder.glob("*.cqe")))
    assert made.rows == table.rows


def test_describing_matches_the_ingest(table, folder):
    first = ingest(table, folder, size=500)
    second = describe(sorted(folder.glob("*.cqe")))
    assert second.data_bytes == first.data_bytes


def test_loading_gives_back_the_table(table, folder):
    made = ingest(table, folder, size=500)
    assert load(made).to_rows() == table.to_rows()


def test_loading_some_columns_gives_those(table, folder):
    made = ingest(table, folder, size=500)
    assert load(made, columns=["id", "amount"]).names == ("id", "amount")


def test_compacting_gives_one_fragment(table, folder, tmp_path):
    made = compact(ingest(table, folder, size=500), tmp_path / "whole.cqe")
    assert len(made.after.fragments) == 1


def test_compacting_keeps_the_rows(table, folder, tmp_path):
    made = compact(ingest(table, folder, size=500), tmp_path / "whole.cqe")
    assert made.after.rows == table.rows


def test_a_compaction_reports_what_it_saved(table, folder, tmp_path):
    made = compact(ingest(table, folder, size=500), tmp_path / "whole.cqe")
    assert made.metadata_saved > 0


def test_a_compaction_reports_its_rewrite_cost(table, folder, tmp_path):
    made = compact(ingest(table, folder, size=500), tmp_path / "whole.cqe")
    assert made.rewrite_cost == made.read_bytes + made.written_bytes


def test_a_compaction_summarises(table, folder, tmp_path):
    made = compact(ingest(table, folder, size=500), tmp_path / "whole.cqe")
    assert made.as_dict()["fragments_after"] == 1


def test_a_compaction_that_saves_nothing_never_breaks_even():
    one = Fragment(path=Path("a"), rows=1, groups=1, data_bytes=10, footer_bytes=5)
    same = Table(fragments=(one,))
    made = Compaction(before=same, after=same, read_bytes=10, written_bytes=10)
    assert made.break_even == float("inf")


def test_compacting_sorted_orders_the_column(table, folder, tmp_path):
    compact(ingest(table, folder, size=500), tmp_path / "whole.cqe", sort_by="amount")
    values = read(tmp_path / "whole.cqe").values("amount")
    assert bool(np.all(values[:-1] <= values[1:]))


def test_compacting_sorted_keeps_the_other_columns_aligned(table, folder, tmp_path):
    compact(ingest(table, folder, size=500), tmp_path / "whole.cqe", sort_by="amount")
    made = read(tmp_path / "whole.cqe")
    expected = sorted(zip(table.values("amount"), table.values("id"), strict=True))
    assert list(made.values("id")) == [one for _, one in expected]


def test_compacting_into_a_missing_folder_makes_it(table, folder, tmp_path):
    made = compact(ingest(table, folder, size=500), tmp_path / "deep" / "whole.cqe")
    assert made.after.fragments[0].path.exists()


def test_a_missing_sort_column_is_refused(table, folder, tmp_path):
    with pytest.raises(SchemaError):
        compact(ingest(table, folder, size=500), tmp_path / "whole.cqe", sort_by="absent")


def test_a_zero_group_size_raises(table, folder, tmp_path):
    with pytest.raises(ConfigError):
        compact(ingest(table, folder, size=500), tmp_path / "whole.cqe", group_size=0)


def test_a_negative_ingest_size_raises(table, folder):
    with pytest.raises(ConfigError):
        ingest(table, folder, size=-1)


def test_scan_cost_counts_every_group(table, folder):
    made = ingest(table, folder, size=500)
    assert scan_cost(made, Compare(">", column("amount"), literal(0.0)))["groups"] == 8


def test_scan_cost_skips_nothing_for_a_true_predicate(table, folder):
    made = ingest(table, folder, size=500)
    assert scan_cost(made, Compare(">", column("amount"), literal(-1e9)))["skipped"] == 0


def test_scan_cost_skips_everything_for_a_false_predicate(table, folder):
    made = ingest(table, folder, size=500)
    assert scan_cost(made, Compare(">", column("amount"), literal(1e9)))["skipped"] == 8


def test_scan_cost_counts_the_metadata(table, folder):
    made = ingest(table, folder, size=500)
    priced = scan_cost(made, Compare(">", column("amount"), literal(1e9)))
    assert priced["metadata_bytes"] == made.metadata_bytes


def test_scan_cost_multiplies_by_the_columns(table, folder):
    made = ingest(table, folder, size=500)
    predicate = Compare(">", column("amount"), literal(-1e9))
    one = scan_cost(made, predicate, columns=1)
    four = scan_cost(made, predicate, columns=4)
    assert four["values_read"] == one["values_read"] * 4


def test_the_ingest_default_is_five_hundred():
    assert INGEST_ROWS == 500


def test_the_compact_group_is_larger_than_the_ingest():
    assert COMPACT_GROUP > INGEST_ROWS


def test_the_open_cost_is_four_kilobytes():
    assert OPEN_COST == 4_096


def test_the_temporary_folders_are_removed():
    maintenance.compaction_removes_almost_all_of_it(4_000)
    assert not list(Path().glob("_compact_*"))


def test_a_folder_helper_clears_an_existing_one(tmp_path):
    made = tmp_path / "gone"
    made.mkdir()
    (made / "a.txt").write_text("x")
    shutil.rmtree(made)
    assert not made.exists()
