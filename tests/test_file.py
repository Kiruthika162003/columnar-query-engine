from __future__ import annotations

import struct

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import CorruptFile, DataError, SchemaError, UnknownColumn
from cqe.exec.batch import Batch
from cqe.storage import file as store
from cqe.storage.file import bytes_read, peek, read, write
from cqe.verify.reference import Rows, agree


@pytest.fixture()
def table(tmp_path):
    """A small table written to a temporary file, returned with its path."""
    state = np.random.default_rng(31)
    rows = 900
    batch = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 20, rows)),
            floating_column("amount", state.normal(50, 10, rows)),
            string_column("label", [f"kind{one % 5}" for one in range(rows)]),
        ]
    )
    path = tmp_path / "one.cqe"
    write(path, batch, group_size=100)
    return batch, path


def test_a_round_trip_keeps_every_row():
    assert store.the_round_trip_is_exact()["rows_match"]


def test_a_round_trip_keeps_the_schema():
    assert store.the_round_trip_is_exact()["schema_matches"]


def test_a_round_trip_keeps_the_nulls():
    assert store.the_round_trip_is_exact()["nulls_survived"]


def test_a_round_trip_keeps_the_dictionary():
    assert store.the_round_trip_is_exact()["the_dictionary_survived"]


def test_reading_two_columns_reads_fewer_bytes():
    assert store.reading_two_columns_of_six_costs_two()["only_two_columns_came_back"]


def test_the_projection_saving_is_the_column_ratio():
    assert store.reading_two_columns_of_six_costs_two()["ratio"] > 2


def test_reading_two_groups_reads_fewer_bytes():
    assert store.reading_two_groups_of_twenty_five_costs_two()["it_read_two_groups_worth"]


def test_the_group_saving_is_the_group_ratio():
    assert store.reading_two_groups_of_twenty_five_costs_two()["ratio"] > 10


def test_the_two_narrowings_multiply():
    assert store.both_narrowings_multiply()["they_multiply"]


def test_the_product_is_the_predicted_one():
    measured = store.both_narrowings_multiply()
    assert measured["both"] == measured["predicted"]


def test_the_footer_is_a_small_share_of_the_file():
    assert store.the_footer_is_read_without_the_data()["it_is_a_small_share"]


def test_the_footer_knows_the_row_count():
    assert store.the_footer_is_read_without_the_data()["it_knows_the_rows"]


def test_the_footer_knows_the_schema():
    assert store.the_footer_is_read_without_the_data()["and_the_schema"]


def test_smaller_groups_cost_more_metadata():
    shares = [one["overhead_share"] for one in store.the_group_size_sets_the_footer_size()]
    assert shares == sorted(shares, reverse=True)


def test_a_flipped_byte_is_caught():
    assert store.a_corrupted_payload_is_caught()["it_was_caught"]


def test_the_corruption_was_one_byte():
    assert store.a_corrupted_payload_is_caught()["one_byte_changed"]


def test_a_foreign_file_is_refused():
    assert store.a_wrong_magic_number_is_refused()["it_was_refused"]


def test_the_magic_refusal_names_the_format():
    assert store.a_wrong_magic_number_is_refused()["the_message_names_the_format"]


def test_a_future_version_is_refused():
    assert store.a_wrong_version_is_refused()["it_was_refused"]


def test_the_version_refusal_names_both_versions():
    assert store.a_wrong_version_is_refused()["the_message_names_both_versions"]


def test_a_truncated_file_is_refused():
    assert store.a_truncated_file_is_refused()["it_was_refused"]


def test_every_type_survives_a_round_trip():
    assert all(store.every_type_survives().values())


def test_an_empty_table_round_trips():
    assert store.an_empty_table_round_trips()["it_reads_back_empty"]


def test_an_empty_table_keeps_its_schema():
    assert store.an_empty_table_round_trips()["schema_survived"]


def test_an_empty_file_is_refused():
    assert store.an_empty_file_is_refused()


def test_an_unknown_column_is_refused():
    assert store.an_unknown_column_is_refused()


def test_an_unknown_group_is_refused():
    assert store.an_unknown_group_is_refused()


def test_a_zero_group_size_is_refused():
    assert store.a_zero_group_size_is_refused()


def test_the_summary_reports_the_magic_number():
    assert store.summarise()["magic"] == "CQE1"


def test_the_written_file_reads_back_equal(table):
    batch, path = table
    assert agree(Rows.of(read(path)), Rows.of(batch), ordered=True)


def test_the_written_file_has_the_expected_groups(table):
    _, path = table
    assert len(peek(path).groups) == 9


def test_the_footer_row_count_matches_the_data(table):
    batch, path = table
    assert peek(path).rows == batch.rows


def test_reading_one_column_returns_one_column(table):
    _, path = table
    assert list(read(path, columns=["amount"]).schema.names) == ["amount"]


def test_reading_one_column_reads_fewer_bytes(table):
    _, path = table
    footer = peek(path)
    every = list(footer.schema.names)
    groups = list(range(len(footer.groups)))
    assert bytes_read(footer, ["amount"], groups) < bytes_read(footer, every, groups)


def test_reading_one_group_returns_its_rows(table):
    _, path = table
    assert read(path, groups=[0]).rows == 100


def test_reading_two_groups_returns_both(table):
    _, path = table
    assert read(path, groups=[0, 1]).rows == 200


def test_the_groups_concatenate_in_order(table):
    _batch, path = table
    produced = read(path, groups=[0, 1])
    assert list(produced.column("id").values) == list(range(200))


def test_reading_every_group_is_the_whole_table(table):
    batch, path = table
    assert read(path, groups=list(range(9))).rows == batch.rows


def test_a_missing_column_is_refused(table):
    _, path = table
    with pytest.raises((UnknownColumn, SchemaError)):
        read(path, columns=["nothing"])


def test_a_missing_group_is_refused(table):
    _, path = table
    with pytest.raises(DataError):
        read(path, groups=[99])


def test_a_group_header_finds_its_chunk(table):
    _, path = table
    assert peek(path).groups[0].chunk("amount").rows == 100


def test_a_group_header_refuses_an_unknown_chunk(table):
    _, path = table
    with pytest.raises(SchemaError):
        peek(path).groups[0].chunk("nothing")


def test_a_group_header_sums_its_chunk_sizes(table):
    _, path = table
    group = peek(path).groups[0]
    assert group.nbytes == sum(one.nbytes for one in group.chunks)


def test_the_footer_sums_its_group_sizes(table):
    _, path = table
    footer = peek(path)
    assert footer.nbytes == sum(one.nbytes for one in footer.groups)


def test_a_chunk_header_summarises(table):
    _, path = table
    summary = peek(path).groups[0].chunks[0].as_dict()
    assert summary["rows"] == 100 and summary["bytes"] > 0


def test_a_group_header_summarises(table):
    _, path = table
    assert peek(path).groups[0].as_dict()["chunks"] == 4


def test_the_footer_summarises(table):
    _, path = table
    assert peek(path).as_dict()["columns"] == 4


def test_the_footer_records_the_version(table):
    _, path = table
    assert peek(path).version == store.VERSION


def test_the_file_starts_with_the_magic_number(table):
    _, path = table
    assert path.read_bytes()[:4] == store.MAGIC


def test_the_header_is_the_declared_size(table):
    _, _path = table
    assert store.HEADER.size == struct.calcsize("<4sIQ")


def test_the_chunk_record_knows_its_own_size():
    assert store.CHUNK_RECORD.size == 44


def test_the_group_record_knows_its_own_size():
    assert store.GROUP_RECORD.size == 16


def test_the_field_record_knows_its_own_size():
    assert store.FIELD_RECORD.size == 9


def test_the_column_lengths_record_knows_its_own_size():
    assert store.COLUMN_LENGTHS.size == 24


def test_a_flipped_byte_in_the_data_is_caught(tmp_path):
    batch = Batch.from_columns([integer_column("v", np.arange(300))])
    path = tmp_path / "flip.cqe"
    write(path, batch, group_size=100)
    raw = bytearray(path.read_bytes())
    raw[60] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(CorruptFile):
        read(path)


def test_a_file_of_one_row_round_trips(tmp_path):
    batch = Batch.from_columns([integer_column("v", [7])])
    path = tmp_path / "one_row.cqe"
    write(path, batch)
    assert list(read(path).column("v").values) == [7]


def test_a_group_size_larger_than_the_table_is_one_group(tmp_path):
    batch = Batch.from_columns([integer_column("v", np.arange(50))])
    path = tmp_path / "wide.cqe"
    write(path, batch, group_size=1000)
    assert len(peek(path).groups) == 1


def test_a_group_size_of_one_makes_a_group_per_row(tmp_path):
    batch = Batch.from_columns([integer_column("v", np.arange(20))])
    path = tmp_path / "narrow.cqe"
    write(path, batch, group_size=1)
    assert len(peek(path).groups) == 20


def test_many_groups_still_read_back_in_order(tmp_path):
    batch = Batch.from_columns([integer_column("v", np.arange(20))])
    path = tmp_path / "narrow.cqe"
    write(path, batch, group_size=1)
    assert list(read(path).column("v").values) == list(range(20))


def test_a_string_column_keeps_its_values(tmp_path):
    batch = Batch.from_columns([string_column("s", ["a", "b", "a", "c"])])
    path = tmp_path / "text.cqe"
    write(path, batch)
    assert read(path).column("s").to_list() == ["a", "b", "a", "c"]


def test_a_column_of_all_nulls_round_trips(tmp_path):
    values = np.arange(10)
    batch = Batch.from_columns([integer_column("v", values)])
    column = batch.column("v")
    nulled = Batch.from_columns(
        [
            type(column)(
                field=column.field,
                values=column.values,
                valid=np.zeros(10, dtype=bool),
            )
        ]
    )
    path = tmp_path / "nulls.cqe"
    write(path, nulled)
    assert read(path).column("v").valid.sum() == 0
