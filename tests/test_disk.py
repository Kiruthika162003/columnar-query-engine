from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import (
    Column,
    floating_column,
    integer_column,
    string_column,
)
from cqe.errors import ConfigError, UnknownColumn
from cqe.exec.batch import Batch, stack
from cqe.exec.expr import Compare, column, literal
from cqe.exec.filter import apply as apply_predicate
from cqe.storage import disk
from cqe.storage.disk import batches, create, index, open_table, scan
from cqe.storage.file import peek, write
from cqe.storage.statistics import collect
from cqe.verify.reference import Rows, agree


@pytest.fixture()
def batch() -> Batch:
    """A table with columns of four shapes."""
    state = np.random.default_rng(53)
    rows = 8000
    return Batch.from_columns(
        [
            integer_column("stamp", np.arange(rows)),
            integer_column("shop", state.integers(0, 50, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 5, rows)]),
        ]
    )


def test_opening_a_table_reads_only_the_footer():
    assert disk.a_table_reads_its_footer_and_nothing_else()["it_knows_the_schema"]


def test_the_footer_is_a_small_share_of_the_file():
    assert disk.a_table_reads_its_footer_and_nothing_else()["footer_share"] < 0.05


def test_a_projection_reads_fewer_bytes():
    assert disk.a_projection_reads_fewer_bytes()["it_read_less"]


def test_a_projection_returns_the_same_rows():
    assert disk.a_projection_reads_fewer_bytes()["same_rows"]


def test_a_sorted_table_skips_most_groups():
    assert disk.a_predicate_on_the_sort_key_skips_most_groups()["the_sorted_one_skipped_most"]


def test_an_arrival_table_skips_nothing():
    assert disk.a_predicate_on_the_sort_key_skips_most_groups()["the_other_skipped_nothing"]


def test_the_sorted_table_reads_far_fewer_bytes():
    assert disk.a_predicate_on_the_sort_key_skips_most_groups()["bytes_ratio"] > 10


def test_every_layout_returns_the_same_rows():
    assert disk.a_scan_returns_the_same_rows_however_it_was_arranged()["they_all_agree"]


def test_a_bloom_index_prunes_a_high_cardinality_equality():
    assert disk.a_bloom_index_prunes_an_equality()["the_index_helped"]


def test_the_bloom_index_skips_most_groups():
    assert disk.a_bloom_index_prunes_an_equality()["it_skipped_most"]


def test_the_bloom_index_does_not_change_the_answer():
    assert disk.a_bloom_index_prunes_an_equality()["the_answer_is_the_same"]


def test_a_bloom_index_is_useless_on_a_spread_column():
    assert disk.a_bloom_index_is_useless_on_a_spread_column()["it_pruned_nothing"]


def test_the_narrowings_multiply():
    assert disk.the_narrowings_multiply()["within_a_tenth"]


def test_a_sorted_table_wastes_fewer_rows():
    assert disk.a_scan_reports_its_waste()["the_sorted_one_wastes_less"]


def test_streaming_agrees_with_reading():
    assert disk.streaming_gives_the_same_rows_as_reading()["they_agree"]


def test_streaming_never_holds_the_table():
    assert disk.streaming_holds_one_batch_at_a_time()["it_never_held_the_table"]


def test_a_full_scan_returns_the_table():
    assert disk.a_scan_of_everything_is_the_table()["and_gave_back_the_table"]


def test_an_empty_result_keeps_its_schema():
    assert disk.a_predicate_that_matches_nothing_returns_an_empty_batch()[
        "and_keeps_its_schema"
    ]


def test_a_predicate_column_is_read_even_when_not_selected():
    assert disk.a_predicate_column_is_read_even_when_not_selected()["it_read_more"]


def test_an_unknown_column_is_refused():
    assert disk.an_unknown_column_is_refused()


def test_an_unknown_group_is_refused():
    assert disk.an_unknown_group_is_refused()


def test_an_unknown_layout_is_refused():
    assert disk.an_unknown_layout_is_refused()


def test_indexing_a_missing_column_is_refused():
    assert disk.indexing_a_missing_column_is_refused()


def test_a_zero_batch_size_is_refused():
    assert disk.a_zero_batch_size_is_refused()


def test_the_narrowing_table_shows_the_product():
    table = disk.compare_the_narrowings()
    assert table[-1]["bytes_read"] < min(one["bytes_read"] for one in table[1:-1])


def test_the_summary_says_the_layouts_agree():
    assert disk.summarise()["layouts_agree"]


def test_creating_a_table_returns_its_row_count(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    assert table.rows == batch.rows


def test_creating_a_table_cuts_it_into_groups(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    assert table.groups == 16


def test_opening_a_table_gives_the_same_footer(batch, tmp_path):
    path = tmp_path / "one.cqe"
    created = create(path, batch, group_size=500)
    opened = open_table(path)
    assert opened.rows == created.rows and opened.groups == created.groups


def test_a_table_reports_its_size(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch)
    assert table.nbytes > 0


def test_a_table_summarises(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch)
    assert table.as_dict()["columns"] == 4


def test_reading_one_group_returns_its_rows(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    assert table.group(0).rows == 500


def test_reading_one_group_narrowed_returns_one_column(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    assert list(table.group(0, columns=["amount"]).schema.names) == ["amount"]


def test_a_scan_returns_every_row(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    produced, _ = scan(table)
    assert produced.rows == batch.rows


def test_a_scan_agrees_with_the_batch(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    produced, _ = scan(table)
    assert agree(Rows.of(produced), Rows.of(batch), ordered=True)


def test_a_scan_with_a_predicate_agrees_with_the_filter(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    predicate = Compare("<", column("amount"), literal(80.0))
    produced, _ = scan(table, predicate=predicate)
    expected = apply_predicate(predicate, batch)
    assert agree(Rows.of(produced), Rows.of(expected))


def test_a_scan_reports_the_groups_it_read(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    _, measured = scan(table)
    assert measured.groups_read == 16 and measured.groups_skipped == 0


def test_a_scan_summarises(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    _, measured = scan(table)
    assert measured.as_dict()["rows_read"] == batch.rows


def test_a_sorted_table_prunes_from_its_footer(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500, order="sorted", key="amount")
    predicate = Compare("<", column("amount"), literal(60.0))
    _, measured = scan(table, predicate=predicate)
    assert measured.groups_skipped > 10


def test_the_pruned_scan_still_finds_every_row(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500, order="sorted", key="amount")
    predicate = Compare("<", column("amount"), literal(60.0))
    produced, _ = scan(table, predicate=predicate)
    expected = apply_predicate(predicate, batch)
    assert produced.rows == expected.rows


def test_a_clustered_table_prunes_an_equality(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500, order="clustered", key="region")
    predicate = Compare("=", column("region"), literal("region2"))
    _, measured = scan(table, predicate=predicate)
    assert measured.groups_skipped > 5


def test_the_footer_carries_the_statistics(batch, tmp_path):
    path = tmp_path / "one.cqe"
    write(path, batch, group_size=500)
    stats = peek(path).groups[0].stats
    assert set(stats.columns) == set(batch.schema.names)


def test_the_stored_minimum_matches_the_data(batch, tmp_path):
    path = tmp_path / "one.cqe"
    write(path, batch, group_size=500)
    stats = peek(path).groups[0].stats
    assert stats.columns["stamp"].minimum == 0


def test_the_stored_maximum_matches_the_data(batch, tmp_path):
    path = tmp_path / "one.cqe"
    write(path, batch, group_size=500)
    stats = peek(path).groups[0].stats
    assert stats.columns["stamp"].maximum == 499


def test_a_stored_integer_statistic_is_an_integer(batch, tmp_path):
    path = tmp_path / "one.cqe"
    write(path, batch, group_size=500)
    stats = peek(path).groups[0].stats
    assert isinstance(stats.columns["stamp"].minimum, int)


def test_a_stored_string_statistic_is_a_string(batch, tmp_path):
    path = tmp_path / "one.cqe"
    write(path, batch, group_size=500)
    stats = peek(path).groups[0].stats
    assert isinstance(stats.columns["region"].minimum, str)


def test_a_stored_float_statistic_is_a_float(batch, tmp_path):
    path = tmp_path / "one.cqe"
    write(path, batch, group_size=500)
    stats = peek(path).groups[0].stats
    assert isinstance(stats.columns["amount"].minimum, float)


def test_the_stored_statistics_agree_with_a_fresh_collection(batch, tmp_path):
    path = tmp_path / "one.cqe"
    write(path, batch, group_size=500)
    stored = peek(path).groups[0].stats
    fresh = collect(batch.slice(0, 500))
    assert stored.columns["stamp"].maximum == fresh.columns["stamp"].maximum


def test_the_stored_null_count_survives(tmp_path):
    values = np.arange(100)
    made = integer_column("v", values)
    valid = np.array([one % 3 != 0 for one in range(100)])
    batch = Batch.from_columns([Column(field=made.field, values=values, valid=valid)])
    path = tmp_path / "one.cqe"
    write(path, batch, group_size=100)
    assert peek(path).groups[0].stats.columns["v"].nulls == int((~valid).sum())


def test_indexing_adds_filters(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    assert index(table, ["region"]).filters


def test_an_indexed_table_keeps_its_footer(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    assert index(table, ["region"]).rows == table.rows


def test_streaming_covers_every_row(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    assert sum(one.rows for one in batches(table, per_batch=4)) == batch.rows


def test_streaming_yields_the_expected_batch_count(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    assert len(list(batches(table, per_batch=4))) == 4


def test_streaming_a_narrowed_scan_returns_one_column(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    first = next(iter(batches(table, columns=["amount"], per_batch=2)))
    assert list(first.schema.names) == ["amount"]


def test_streaming_with_a_predicate_agrees(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch, group_size=500)
    predicate = Compare("<", column("amount"), literal(80.0))
    streamed = stack(list(batches(table, predicate=predicate, per_batch=4)))
    expected = apply_predicate(predicate, batch)
    assert agree(Rows.of(streamed), Rows.of(expected))


def test_scanning_a_missing_column_is_refused(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch)
    with pytest.raises(UnknownColumn):
        scan(table, columns=["nothing"])


def test_reading_a_group_past_the_end_is_refused(batch, tmp_path):
    table = create(tmp_path / "one.cqe", batch)
    with pytest.raises(ConfigError):
        table.group(9999)


def test_an_unknown_order_is_refused(batch, tmp_path):
    with pytest.raises(ConfigError):
        create(tmp_path / "one.cqe", batch, order="sideways")
