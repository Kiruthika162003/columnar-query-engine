from __future__ import annotations

import json

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import ConfigError, DataError, SchemaError
from cqe.exec.batch import Batch
from cqe.storage import catalogue as book
from cqe.storage.catalogue import (
    CATALOGUE,
    Catalogue,
    add,
    analyse,
    analyse_all,
    batches,
    load,
    open_all,
    refresh,
    save,
    table,
)
from cqe.storage.disk import create


def make(directory, tables: int = 3, rows: int = 2000) -> Catalogue:
    """A directory of tables, registered in a catalogue."""
    state = np.random.default_rng(137)
    made = Catalogue()
    for one in range(tables):
        batch = Batch.from_columns(
            [
                integer_column("id", np.arange(rows)),
                integer_column("shop", state.integers(0, 20, rows)),
                floating_column("amount", state.normal(100, 20, rows)),
                string_column(
                    "region", [f"region{value}" for value in state.integers(0, 4, rows)]
                ),
            ]
        )
        path = directory / f"table{one}.cqe"
        create(path, batch, group_size=500)
        add(made, f"table{one}", path)
    return made


def test_opening_a_catalogue_scans_nothing():
    assert book.opening_a_catalogue_reads_no_data()["no_scans"]


def test_opening_costs_one_footer_per_table():
    assert book.opening_a_catalogue_reads_no_data()["one_read_per_table"]


def test_opening_knows_every_schema():
    assert book.opening_a_catalogue_reads_no_data()["it_knows_every_schema"]


def test_the_footers_are_a_small_share():
    assert book.opening_a_catalogue_reads_no_data()["footer_share"] < 0.1


def test_analysing_scans_every_table():
    assert book.statistics_cost_a_scan()["analysing_scanned_every_table"]


def test_opening_did_not_scan():
    assert book.statistics_cost_a_scan()["opening_scanned_nothing"]


def test_analysing_gives_every_table_statistics():
    assert book.statistics_cost_a_scan()["and_every_table_has_statistics"]


def test_a_grown_table_goes_stale():
    assert book.statistics_go_stale_when_a_table_grows()["stale_after_growing"]


def test_a_freshly_analysed_table_is_not_stale():
    assert book.statistics_go_stale_when_a_table_grows()["fresh_after_analysing"]


def test_a_refresh_is_one_footer_read():
    assert book.statistics_go_stale_when_a_table_grows()["and_a_refresh_is_one_footer_read"]


def test_a_small_change_does_not_make_it_stale():
    assert book.a_small_change_does_not_make_it_stale()["it_is_still_fresh"]


def test_a_catalogue_round_trips():
    assert book.a_catalogue_round_trips()["names_match"]


def test_the_row_counts_survive_a_round_trip():
    assert book.a_catalogue_round_trips()["rows_match"]


def test_the_statistics_do_not_survive():
    assert book.a_catalogue_round_trips()["the_statistics_did_not_survive"]


def test_everything_is_stale_after_a_load():
    assert book.a_catalogue_round_trips()["and_everything_is_stale"]


def test_opening_a_directory_finds_every_table():
    assert book.opening_a_directory_finds_every_table()["it_found_them_all"]


def test_the_tables_are_named_after_their_files():
    assert book.opening_a_directory_finds_every_table()["named_after_the_files"]


def test_reading_costs_far_more_than_opening():
    assert book.reading_every_table_costs_far_more_than_opening()["the_ratio"] > 10


def test_a_schema_costs_no_further_reads():
    assert book.a_catalogue_answers_a_schema_without_touching_the_file()["it_read_nothing_more"]


def test_every_schema_comes_back():
    assert book.a_catalogue_answers_a_schema_without_touching_the_file()[
        "and_every_schema_came_back"
    ]


def test_a_missing_table_lists_the_others():
    assert book.a_missing_table_lists_the_others()["it_lists_the_others"]


def test_a_repeated_name_is_refused():
    assert book.a_repeated_name_is_refused()


def test_a_missing_file_is_refused():
    assert book.a_missing_file_is_refused()


def test_a_missing_directory_is_refused():
    assert book.a_missing_directory_is_refused()


def test_a_catalogue_of_the_wrong_format_is_refused():
    assert book.a_catalogue_of_the_wrong_format_is_refused()


def test_a_missing_catalogue_file_is_refused():
    assert book.a_missing_catalogue_file_is_refused()


def test_four_questions_cost_nothing():
    free = [one for one in book.compare_the_questions() if one["costs"] == "nothing"]
    assert len(free) == 4


def test_the_summary_says_opening_reads_no_data():
    assert book.summarise()["opening_reads_no_data"]


def test_a_catalogue_counts_its_tables(tmp_path):
    assert len(make(tmp_path)) == 3


def test_a_catalogue_lists_its_names(tmp_path):
    assert make(tmp_path).names == ("table0", "table1", "table2")


def test_a_catalogue_sums_its_rows(tmp_path):
    assert make(tmp_path, tables=3, rows=2000).rows == 6000


def test_a_catalogue_can_be_asked_whether_it_holds_a_table(tmp_path):
    made = make(tmp_path)
    assert "table0" in made and "nothing" not in made


def test_a_catalogue_iterates_its_names(tmp_path):
    assert sorted(make(tmp_path)) == ["table0", "table1", "table2"]


def test_a_catalogue_summarises(tmp_path):
    assert make(tmp_path).as_dict()["tables"] == 3


def test_an_entry_reports_its_groups(tmp_path):
    assert make(tmp_path, rows=2000).entry("table0").groups == 4


def test_an_entry_reports_its_schema(tmp_path):
    assert len(make(tmp_path).entry("table0").schema) == 4


def test_an_entry_summarises(tmp_path):
    assert make(tmp_path).entry("table0").as_dict()["name"] == "table0"


def test_an_unanalysed_entry_is_stale(tmp_path):
    assert make(tmp_path).entry("table0").stale


def test_an_analysed_entry_is_not(tmp_path):
    made = make(tmp_path)
    analyse(made, "table0")
    assert not made.entry("table0").stale


def test_analysing_gives_the_entry_statistics(tmp_path):
    made = make(tmp_path)
    analyse(made, "table0")
    assert made.entry("table0").has_stats


def test_analysing_everything_covers_every_table(tmp_path):
    made = analyse_all(make(tmp_path))
    assert all(one.has_stats for one in made.entries.values())


def test_the_statistics_know_the_columns(tmp_path):
    made = make(tmp_path)
    analyse(made, "table0")
    assert set(made.entry("table0").stats.columns) == {"id", "shop", "amount", "region"}


def test_asking_for_a_missing_table_is_refused(tmp_path):
    with pytest.raises(SchemaError):
        make(tmp_path).entry("nothing")


def test_asking_for_a_missing_schema_is_refused(tmp_path):
    with pytest.raises(SchemaError):
        make(tmp_path).schema("nothing")


def test_a_table_can_be_opened_for_scanning(tmp_path):
    made = make(tmp_path)
    assert table(made, "table0").rows == 2000


def test_reading_every_table_gives_a_batch_each(tmp_path):
    made = make(tmp_path)
    loaded = batches(made)
    assert set(loaded) == {"table0", "table1", "table2"}


def test_reading_every_table_counts_the_scans(tmp_path):
    made = make(tmp_path)
    batches(made)
    assert made.scans == 3


def test_refreshing_updates_the_row_count(tmp_path):
    made = make(tmp_path, tables=1, rows=1000)
    state = np.random.default_rng(139)
    larger = Batch.from_columns(
        [
            integer_column("id", np.arange(3000)),
            integer_column("shop", state.integers(0, 20, 3000)),
            floating_column("amount", state.normal(100, 20, 3000)),
            string_column("region", [f"region{one}" for one in state.integers(0, 4, 3000)]),
        ]
    )
    create(tmp_path / "table0.cqe", larger, group_size=500)
    assert refresh(made, "table0").rows == 3000


def test_refreshing_a_missing_table_is_refused(tmp_path):
    with pytest.raises(SchemaError):
        refresh(make(tmp_path), "nothing")


def test_saving_writes_readable_json(tmp_path):
    made = make(tmp_path)
    path = save(made, tmp_path / CATALOGUE)
    assert len(json.loads(path.read_text(encoding="utf-8"))["tables"]) == 3


def test_loading_gives_the_same_names(tmp_path):
    made = make(tmp_path)
    path = save(made, tmp_path / CATALOGUE)
    assert load(path).names == made.names


def test_loading_re_reads_the_footers(tmp_path):
    made = make(tmp_path)
    path = save(made, tmp_path / CATALOGUE)
    assert load(path).reads == 3


def test_loading_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ConfigError):
        load(tmp_path / "nothing.json")


def test_loading_a_future_format_is_refused(tmp_path):
    path = tmp_path / CATALOGUE
    path.write_text(json.dumps({"format": 9, "tables": []}), encoding="utf-8")
    with pytest.raises(DataError):
        load(path)


def test_opening_a_directory_of_no_tables_is_empty(tmp_path):
    assert len(open_all(tmp_path)) == 0


def test_adding_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ConfigError):
        add(Catalogue(), "one", tmp_path / "nothing.cqe")


def test_adding_a_repeated_name_is_refused(tmp_path):
    made = make(tmp_path, tables=1)
    with pytest.raises(ConfigError):
        add(made, "table0", tmp_path / "table0.cqe")


def test_an_empty_catalogue_has_no_rows():
    assert Catalogue().rows == 0
