from __future__ import annotations

import contextlib
import io
import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_example(name: str) -> str:
    """Run one example as a script and return what it printed.

    As a script rather than by importing and calling main, because an example is a file somebody
    runs and the failure this catches is the one where the file works when imported and not when
    run.
    """
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return out.getvalue()


@pytest.fixture(scope="module")
def file_output() -> str:
    """The file query example, run once for every test that reads it."""
    return run_example("query_a_file.py")


@pytest.fixture(scope="module")
def explain_output() -> str:
    """The explain example, run once."""
    return run_example("explain_a_query.py")


@pytest.fixture(scope="module")
def measure_output() -> str:
    """The measurement example, run once."""
    return run_example("measure_everything.py")


@pytest.fixture(scope="module")
def tidy_output() -> str:
    """The compaction example, run once."""
    return run_example("keep_a_table_tidy.py")


def test_every_example_exists():
    assert len(list(EXAMPLES.glob("*.py"))) == 4


def test_every_example_has_a_docstring():
    for one in EXAMPLES.glob("*.py"):
        assert one.read_text(encoding="utf-8").lstrip().startswith('"""')


def test_the_file_example_prints_something(file_output):
    assert file_output.strip()


def test_the_file_example_reports_its_rows(file_output):
    assert "40000 rows" in file_output


def test_the_file_example_shows_three_layouts(file_output):
    assert file_output.count("as they arrived") == 2


def test_the_sorted_layout_reads_fewer_groups(file_output):
    lines = [one for one in file_output.split("\n") if "sorted by total" in one]
    assert "groups   2 of 80" in lines[0]


def test_the_clustered_layout_reads_fewer_groups_on_its_own_column(file_output):
    lines = [one for one in file_output.split("\n") if "clustered by region" in one]
    assert "of 80" in lines[1] and "groups  80" not in lines[1]


def test_the_bloom_filter_prunes_the_lookup(file_output):
    lines = [one for one in file_output.split("\n") if "with a bloom filter" in one]
    assert "groups   6 of 80" in lines[0]


def test_narrowing_reads_fewer_bytes(file_output):
    lines = [
        one
        for one in file_output.split("\n")
        if one.strip().startswith("two columns") and "bytes" in one
    ]
    assert "640000" in lines[0]


def test_the_explain_example_prints_the_query(explain_output):
    assert "select region, count(*)" in explain_output


def test_the_explain_example_counts_its_tokens(explain_output):
    assert "tokens: 35" in explain_output


def test_the_explain_example_prints_a_plan(explain_output):
    assert "the plan, 7 nodes" in explain_output


def test_the_explain_example_shows_the_rewrite(explain_output):
    assert "2 predicates moved" in explain_output


def test_the_rewrite_pushes_the_predicate_into_the_scan(explain_output):
    after = explain_output.split("after the rewrite")[1]
    assert "Scan facts [shop, total] where" in after


def test_the_explain_example_names_its_strategies(explain_output):
    assert "Join: hash" in explain_output and "Group: counting" in explain_output


def test_the_explain_example_reports_a_cost(explain_output):
    assert "what it should cost" in explain_output


def test_the_explain_example_prints_an_answer(explain_output):
    assert "the answer" in explain_output and "region" in explain_output


def test_the_explain_example_returns_three_rows(explain_output):
    answer = explain_output.split("the answer")[1].strip().split("\n")
    assert len(answer) == 4


def test_the_tidy_example_ingests_in_pieces(tidy_output):
    assert "60000 rows arrived in 120 pieces of 500" in tidy_output


def test_the_tidy_example_reports_the_metadata_share(tidy_output):
    assert "metadata share" in tidy_output


def test_the_tidy_example_compacts_to_one_file(tidy_output):
    assert "120 to 1" in tidy_output


def test_the_tidy_example_reports_a_break_even(tidy_output):
    assert "break even" in tidy_output


def test_the_tidy_example_keeps_every_row(tidy_output):
    assert "holds the same 60000 rows: yes" in tidy_output


def test_the_tidy_example_admits_a_query_got_worse(tidy_output):
    assert "recent orders got worse" in tidy_output


def test_the_tidy_example_finds_four_statuses(tidy_output):
    line = next(one for one in tidy_output.split("\n") if one.startswith("status"))
    assert "4" in line


def test_the_tidy_example_leaves_no_files(tidy_output):
    assert tidy_output and not list(Path().glob("_tidy_*"))


def test_the_measure_example_covers_every_module(measure_output):
    assert "45 modules" in measure_output


def test_every_module_summarised(measure_output):
    assert "every module summarised" in measure_output


def test_no_module_failed_to_summarise(measure_output):
    assert "failed:" not in measure_output


def test_the_measure_example_reports_the_bloom_filter(measure_output):
    assert "storage/bloom" in measure_output and "no_false_negatives yes" in measure_output


def test_the_measure_example_reports_the_differential_harness(measure_output):
    assert "verify/differential" in measure_output and "all_pass yes" in measure_output


def test_the_measure_example_reports_one_line_per_module(measure_output):
    lines = [one for one in measure_output.split("\n") if one.strip() and "modules" not in one]
    assert len(lines) == 46
