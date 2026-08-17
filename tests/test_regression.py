from __future__ import annotations

import json

import pytest

from cqe.errors import ConfigError, DataError
from cqe.eval import regression
from cqe.eval.regression import (
    BASELINE,
    FORMAT,
    TOLERANCE,
    Baseline,
    Change,
    Comparison,
    Entry,
    check,
    compare,
    load,
    record,
    save,
)
from cqe.eval.workload import catalogue, measure, named


@pytest.fixture(scope="module")
def baseline() -> Baseline:
    """One recorded run, shared by the tests that only read it."""
    return record(2000)


def test_a_baseline_round_trips():
    assert regression.a_baseline_round_trips()["they_match"]


def test_the_baseline_file_is_not_empty():
    assert regression.a_baseline_round_trips()["the_file_is_readable"]


def test_a_run_against_itself_is_clean():
    assert regression.a_run_against_itself_is_clean()["it_is_clean"]


def test_every_ratio_is_exactly_one():
    assert regression.a_run_against_itself_is_clean()["every_ratio_is_exactly_one"]


def test_the_counts_are_identical_across_runs():
    assert regression.the_counts_are_reproducible_across_seeds()["they_are_identical"]


def test_a_regression_is_caught():
    assert regression.a_regression_is_caught()["it_was_caught"]


def test_the_regression_names_its_query():
    assert regression.a_regression_is_caught()["it_named_the_query"]


def test_the_regression_report_says_so():
    assert regression.a_regression_is_caught()["and_the_report_says_so"]


def test_an_improvement_is_reported():
    assert regression.an_improvement_is_not_a_regression()["it_was_reported"]


def test_an_improvement_does_not_fail_the_check():
    assert regression.an_improvement_is_not_a_regression()["and_the_check_is_clean"]


def test_a_changed_answer_is_caught():
    assert regression.a_changed_answer_is_reported_separately()["it_was_caught"]


def test_a_changed_answer_is_not_a_regression():
    assert regression.a_changed_answer_is_reported_separately()["it_is_not_called_a_regression"]


def test_a_changed_answer_fails_the_check():
    assert regression.a_changed_answer_is_reported_separately()["and_the_check_is_not_clean"]


def test_an_added_query_is_reported_as_added():
    assert regression.a_query_added_to_the_set_is_not_a_regression()["it_was_reported_as_added"]


def test_an_added_query_regresses_nothing():
    assert regression.a_query_added_to_the_set_is_not_a_regression()["and_nothing_regressed"]


def test_an_added_query_leaves_the_check_clean():
    assert regression.a_query_added_to_the_set_is_not_a_regression()["and_the_check_is_clean"]


def test_a_removed_query_is_reported():
    assert regression.a_query_removed_from_the_set_is_reported()["it_was_reported"]


def test_a_removed_query_fails_the_check():
    assert regression.a_query_removed_from_the_set_is_reported()["and_the_check_is_not_clean"]


def test_a_change_inside_the_tolerance_passes():
    assert regression.the_tolerance_is_what_decides()["the_small_change_passed"]


def test_a_change_outside_the_tolerance_does_not():
    assert regression.the_tolerance_is_what_decides()["and_the_large_one_did_not"]


def test_checking_against_a_file_is_clean():
    assert regression.checking_against_a_file_works()["it_is_clean"]


def test_a_clean_report_is_one_line():
    assert regression.checking_against_a_file_works()["the_report_is_one_line"]


def test_a_missing_baseline_is_refused():
    assert regression.a_missing_baseline_is_refused()


def test_a_baseline_of_the_wrong_format_is_refused():
    assert regression.a_baseline_of_the_wrong_format_is_refused()


def test_an_unknown_query_in_a_baseline_is_refused():
    assert regression.an_unknown_query_in_a_baseline_is_refused()


def test_every_outcome_says_whether_it_fails():
    assert all("fails" in one for one in regression.compare_the_outcomes())


def test_two_outcomes_do_not_fail():
    assert sum(1 for one in regression.compare_the_outcomes() if not one["fails"]) == 2


def test_the_summary_reports_the_tolerance():
    assert regression.summarise()["tolerance"] == TOLERANCE


def test_a_recorded_baseline_covers_every_query(baseline):
    assert len(baseline.entries) == 10


def test_a_recorded_baseline_reports_its_rows(baseline):
    assert baseline.rows == 2000


def test_a_baseline_finds_a_query_by_name(baseline):
    assert baseline.entry("range").query == "range"


def test_a_baseline_lists_its_queries(baseline):
    assert "join" in baseline.queries


def test_a_baseline_summarises(baseline):
    assert baseline.as_dict()["format"] == FORMAT


def test_an_entry_summarises(baseline):
    assert baseline.entry("range").as_dict()["query"] == "range"


def test_saving_and_loading_gives_the_same_entries(baseline, tmp_path):
    path = save(baseline, tmp_path / BASELINE)
    assert load(path).entries == baseline.entries


def test_a_saved_baseline_is_readable_json(baseline, tmp_path):
    path = save(baseline, tmp_path / BASELINE)
    assert json.loads(path.read_text(encoding="utf-8"))["rows"] == 2000


def test_loading_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ConfigError):
        load(tmp_path / "nothing.json")


def test_loading_a_future_format_is_refused(tmp_path):
    path = tmp_path / BASELINE
    path.write_text(json.dumps({"format": 2, "rows": 1, "entries": []}), encoding="utf-8")
    with pytest.raises(DataError):
        load(path)


def test_asking_for_an_unknown_query_is_refused(baseline):
    with pytest.raises(ConfigError):
        baseline.entry("nothing")


def test_comparing_a_baseline_with_itself_is_clean(baseline):
    assert compare(baseline, baseline).clean


def test_comparing_reports_one_change_per_query(baseline):
    assert len(compare(baseline, baseline).changes) == len(baseline.entries)


def test_a_comparison_summarises(baseline):
    assert compare(baseline, baseline).as_dict()["regressions"] == 0


def test_checking_a_saved_baseline_is_clean(baseline, tmp_path):
    path = save(baseline, tmp_path / BASELINE)
    assert check(path).clean


def test_checking_at_a_different_size_is_not_clean(baseline, tmp_path):
    path = save(baseline, tmp_path / BASELINE)
    assert not check(path, rows=4000).clean


def test_an_unchanged_change_has_a_ratio_of_one():
    one = Change(query="a", before=100, after=100, rows_before=1, rows_after=1)
    assert one.ratio == 1.0


def test_a_doubled_change_has_a_ratio_of_two():
    one = Change(query="a", before=100, after=200, rows_before=1, rows_after=1)
    assert one.ratio == 2.0 and one.regressed


def test_a_halved_change_is_an_improvement():
    one = Change(query="a", before=200, after=100, rows_before=1, rows_after=1)
    assert one.improved and not one.regressed


def test_a_change_from_zero_does_not_divide_by_zero():
    one = Change(query="a", before=0, after=10, rows_before=1, rows_after=1)
    assert one.ratio == 10.0


def test_zero_to_zero_is_unchanged():
    one = Change(query="a", before=0, after=0, rows_before=1, rows_after=1)
    assert one.ratio == 1.0 and not one.regressed and not one.improved


def test_a_change_summarises():
    one = Change(query="a", before=100, after=200, rows_before=1, rows_after=1)
    assert one.as_dict()["ratio"] == 2.0


def test_an_empty_comparison_is_clean():
    assert Comparison(changes=()).clean


def test_an_empty_comparison_reports_no_queries():
    assert "0 queries" in Comparison(changes=()).report()


def test_a_comparison_with_a_missing_query_is_not_clean():
    assert not Comparison(changes=(), missing=("a",)).clean


def test_a_comparison_with_an_added_query_is_clean():
    assert Comparison(changes=(), added=("a",)).clean


def test_an_entry_can_be_built_from_a_measurement():
    made = Entry.of(measure(named("range"), catalogue(1000)))
    assert made.query == "range" and made.total > 0
