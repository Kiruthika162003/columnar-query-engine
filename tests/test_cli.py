from __future__ import annotations

import json
from pathlib import Path

import pytest

from cqe.cli import main as cli
from cqe.cli.main import Result, _cell, _emit, _run, _table, build_parser, main
from cqe.storage.file import peek


def test_every_command_runs():
    assert cli.every_command_runs()["all_succeeded"]


def test_every_command_prints_something():
    assert cli.every_command_runs()["all_printed_something"]


def test_the_json_flag_parses():
    assert cli.the_json_flag_produces_json()["the_json_parses"]


def test_the_two_forms_hold_the_same_rows():
    assert cli.the_json_flag_produces_json()["they_hold_the_same_rows"]


def test_a_bad_query_prints_a_caret():
    assert cli.a_bad_query_prints_a_caret()["it_printed_a_caret"]


def test_a_bad_query_prints_no_traceback():
    assert cli.a_bad_query_prints_a_caret()["it_printed_no_traceback"]


def test_a_bad_query_exits_with_two():
    assert cli.a_bad_query_prints_a_caret()["status"] == 2


def test_a_missing_column_names_itself():
    assert cli.a_missing_column_prints_the_schema()["it_named_the_column"]


def test_a_missing_column_lists_the_real_ones():
    assert cli.a_missing_column_prints_the_schema()["it_listed_the_real_ones"]


def test_a_missing_table_lists_the_catalogue():
    assert cli.a_missing_table_prints_the_catalogue()["it_listed_the_tables"]


def test_the_plan_command_shows_both_trees():
    assert cli.the_plan_command_shows_both_trees()["it_showed_both"]


def test_the_plan_command_says_what_moved():
    assert cli.the_plan_command_shows_both_trees()["it_said_what_moved"]


def test_the_rewrite_removes_the_filter_node():
    measured = cli.the_plan_command_shows_both_trees()
    assert measured["the_before_has_a_filter"] and measured["the_after_does_not"]


def test_the_explain_command_names_a_strategy():
    assert cli.the_explain_command_names_a_strategy()["it_named_a_strategy"]


def test_the_query_command_prints_a_header():
    assert cli.the_query_command_returns_rows()["it_printed_a_header"]


def test_the_query_command_honours_its_limit():
    assert cli.the_query_command_returns_rows()["it_printed_five_rows"]


def test_an_empty_result_says_so():
    assert cli.a_query_returning_nothing_says_so()["it_said_so"]


def test_an_empty_result_still_succeeds():
    assert cli.a_query_returning_nothing_says_so()["it_succeeded"]


def test_the_write_command_writes_the_asked_for_rows():
    assert cli.the_write_command_writes_a_readable_file()["it_wrote_the_asked_for_rows"]


def test_the_write_command_writes_the_asked_for_groups():
    assert cli.the_write_command_writes_a_readable_file()["and_the_asked_for_groups"]


def test_the_written_schema_reads_back():
    assert cli.the_write_command_writes_a_readable_file()["the_schema_reads_back"]


def test_the_sort_flag_orders_the_file():
    assert cli.the_write_command_arranges_the_rows()["the_sorted_one_is_ordered"]


def test_without_the_flag_it_is_not_ordered():
    assert cli.the_write_command_arranges_the_rows()["the_plain_one_is_not"]


def test_the_verify_command_passes():
    assert cli.the_verify_command_reports_every_check()["every_check_passed"]


def test_the_verify_command_exits_zero():
    assert cli.the_verify_command_reports_every_check()["it_succeeded"]


def test_the_measure_command_covers_the_package():
    assert cli.the_measure_command_covers_every_module()["it_covers_the_package"]


def test_every_module_summarises():
    assert cli.the_measure_command_covers_every_module()["they_all_summarised"]


def test_the_only_flag_narrows():
    assert cli.the_only_flag_narrows_the_measurement()["it_narrowed"]


def test_the_only_flag_keeps_the_matching_modules():
    assert cli.the_only_flag_narrows_the_measurement()["and_kept_some"]


def test_an_unknown_command_is_refused():
    assert cli.an_unknown_command_is_refused()["it_refused"]


def test_no_command_is_refused():
    assert cli.no_command_at_all_is_refused()["it_refused"]


def test_the_command_table_covers_every_command():
    assert len(cli.compare_the_commands()) == 9


def test_the_summary_says_every_command_runs():
    assert cli.summarise()["all_run"]


def test_the_parser_knows_every_command():
    parser = build_parser()
    for one in ("schema", "stats", "plan", "explain", "cost", "query", "write", "verify"):
        assert parser.parse_args(
            [one, *(["x"] if one in ("plan", "explain", "cost", "query") else [])]
            + (["out"] if one == "write" else [])
        )


def test_the_schema_command_lists_four_columns():
    status, text = _run(["schema"])
    assert status == 0 and len(text.strip().split("\n")) == 6


def test_the_stats_command_reports_distinct_counts():
    status, text = _run(["stats"])
    assert status == 0 and "distinct" in text


def test_the_stats_command_reports_a_null_share():
    _, text = _run(["--json", "stats"])
    assert all("null_share" in one for one in json.loads(text))


def test_the_cost_command_reports_both_numbers():
    _, text = _run(["--json", "cost", "select id from facts where amount > 100"])
    parsed = json.loads(text)
    assert parsed["predicted"] > 0 and parsed["counted"] > 0


def test_the_cost_command_names_a_dominant_node():
    _, text = _run(["--json", "cost", "select id from facts where amount > 100"])
    assert json.loads(text)["dominant"]


def test_the_raw_flag_skips_the_rewrite():
    _, pushed = _run(["explain", "select id from facts where amount > 100"])
    _, raw = _run(["explain", "select id from facts where amount > 100", "--raw"])
    assert "Filter" in raw and "Filter" not in pushed


def test_a_group_query_returns_its_groups():
    _, text = _run(
        ["--json", "query", "select region, count(*) as n from facts group by region"]
    )
    assert len(json.loads(text)) == 6


def test_an_ordered_query_comes_back_ordered():
    _, text = _run(
        [
            "--json",
            "query",
            "select region, count(*) as n from facts group by region order by n desc",
        ]
    )
    counts = [one["n"] for one in json.loads(text)]
    assert counts == sorted(counts, reverse=True)


def test_a_query_over_a_written_file(tmp_path):
    path = str(tmp_path / "one.cqe")
    _run(["write", path, "--rows", "1000"])
    status, text = _run(
        ["--json", "query", "select id from facts where id < 10", "--file", path]
    )
    assert status == 0 and len(json.loads(text)) == 10


def test_the_schema_of_a_written_file_reads_from_the_footer(tmp_path):
    path = str(tmp_path / "one.cqe")
    _run(["write", path, "--rows", "1000"])
    _, text = _run(["--json", "schema", "--file", path])
    assert [one["column"] for one in json.loads(text)] == ["id", "shop", "amount", "region"]


def test_the_cluster_flag_groups_the_rows(tmp_path):
    path = str(tmp_path / "one.cqe")
    status, _ = _run(["write", path, "--rows", "2000", "--cluster", "region"])
    assert status == 0 and peek(Path(path)).rows == 2000


def test_a_table_of_no_rows_says_nothing_to_show():
    assert _table([]) == "nothing to show"


def test_a_table_has_a_header_and_a_rule():
    text = _table([{"a": 1, "b": 2}])
    assert len(text.split("\n")) == 3


def test_a_cell_rounds_a_float():
    assert _cell(1.23456).strip() == "1.235"


def test_a_cell_renders_a_boolean():
    assert _cell(True).strip() == "yes"


def test_a_cell_renders_a_list():
    assert _cell([1, 2]).strip() == "1,2"


def test_emitting_a_string_returns_it():
    assert _emit("hello", as_json=False) == "hello"


def test_emitting_a_mapping_makes_two_columns():
    text = _emit({"a": 1}, as_json=False)
    assert "name" in text and "value" in text


def test_emitting_as_json_parses():
    assert json.loads(_emit({"a": 1}, as_json=True)) == {"a": 1}


def test_a_result_reports_success():
    assert Result(0, "x").ok and not Result(1, "x").ok


def test_a_missing_file_is_refused(tmp_path):
    status, text = _run(["stats", "--file", str(tmp_path / "nothing.cqe")])
    assert status == 2 and "Traceback" not in text


def test_main_with_an_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["nothing"])
