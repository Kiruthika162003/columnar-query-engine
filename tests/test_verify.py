from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.errors import ConfigError
from cqe.exec.aggregate import Aggregate
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.exec.sort import SortKey, order_by
from cqe.plan.logical import Group, table
from cqe.plan.physical import COUNTING_GROUP, HASH_GROUP, execute, run
from cqe.verify import differential, fuzz
from cqe.verify.differential import CHECKS, Report, run_all, run_one
from cqe.verify.fuzz import Case, Generator, cases, shrink
from cqe.verify.reference import Rows, agree
from cqe.verify.reference import order_by as reference_order


def test_the_generator_is_reproducible():
    assert fuzz.the_generator_is_reproducible()["same_seed_matches"]


def test_a_different_seed_gives_different_cases():
    assert fuzz.the_generator_is_reproducible()["different_seed_differs"]


def test_the_generator_makes_null_columns_at_its_target_rate():
    assert fuzz.the_generator_makes_nulls()["the_column_rate_is_near_its_target"]


def test_the_generator_makes_null_values_at_its_target_rate():
    assert fuzz.the_generator_makes_nulls()["the_value_rate_is_near_its_target"]


def test_most_generated_predicates_keep_some_rows():
    assert fuzz.the_generator_makes_predicates_that_match()["most_are_partial"]


def test_the_generator_reaches_every_type():
    assert fuzz.the_generator_makes_every_type()["it_made_all_four"]


def test_generated_predicates_are_mostly_small():
    assert fuzz.the_predicates_are_mostly_small()["the_median_is_small"]


def test_generated_predicates_have_a_long_tail():
    assert fuzz.the_predicates_are_mostly_small()["and_the_tail_is_long"]


def test_shrinking_removes_most_of_a_case():
    assert fuzz.shrinking_removes_most_of_a_case()["it_removed_most_of_it"]


def test_shrinking_keeps_the_failure():
    assert fuzz.shrinking_keeps_the_failure()["it_still_fails"]


def test_shrinking_stops_at_the_boundary():
    assert fuzz.shrinking_keeps_the_failure()["and_it_is_at_the_boundary"]


def test_shrinking_a_passing_case_is_refused():
    assert fuzz.shrinking_a_passing_case_is_refused()


def test_a_sound_check_reports_nothing():
    assert fuzz.a_search_over_a_sound_check_finds_nothing()["it_found_nothing"]


def test_a_broken_check_is_found():
    assert fuzz.a_search_over_a_broken_check_finds_it()["it_found_some"]


def test_every_found_failure_was_shrunk():
    assert fuzz.a_search_over_a_broken_check_finds_it()["every_one_was_shrunk"]


def test_an_exception_is_reported_as_a_failure():
    assert fuzz.a_search_reports_an_exception_as_a_failure()["it_caught_them"]


def test_the_exception_message_names_the_error():
    assert fuzz.a_search_reports_an_exception_as_a_failure()["the_message_names_the_error"]


def test_a_zero_case_count_is_refused():
    assert fuzz.a_zero_case_count_is_refused()


def test_the_type_shares_are_roughly_even():
    shares = [one["share"] for one in fuzz.compare_the_generators()]
    assert max(shares) < 0.4


def test_the_fuzz_summary_reports_no_false_positives():
    assert fuzz.summarise()["no_false_positives"]


def test_a_generated_batch_has_the_asked_for_columns():
    made = Generator(seed=3).batch(columns=4, rows=20)
    assert made.width == 4 and made.rows == 20


def test_a_generated_batch_names_its_columns_in_order():
    made = Generator(seed=3).batch(columns=3, rows=10)
    assert list(made.schema.names) == ["c0", "c1", "c2"]


def test_a_generated_predicate_reads_a_real_column():
    maker = Generator(seed=5)
    made = maker.batch(columns=3, rows=20)
    predicate = maker.predicate(made)
    assert predicate.columns_used() <= set(made.schema.names)


def test_a_case_reports_its_size():
    maker = Generator(seed=7)
    made = maker.batch(columns=2, rows=10)
    case = Case(batch=made, predicate=Compare(">", column("c0"), literal(0)))
    assert case.size == 10 * 3


def test_a_case_describes_itself():
    maker = Generator(seed=7)
    made = maker.batch(columns=2, rows=10)
    case = Case(batch=made, predicate=Compare(">", column("c0"), literal(0)))
    assert "10 rows" in case.describe()


def test_a_case_summarises():
    maker = Generator(seed=7)
    made = maker.batch(columns=2, rows=10)
    case = Case(batch=made, predicate=Compare(">", column("c0"), literal(0)))
    assert case.as_dict()["columns"] == 2


def test_generating_no_cases_is_refused():
    with pytest.raises(ConfigError):
        cases(count=0)


def test_shrinking_reaches_a_single_row():
    maker = Generator(seed=11)
    made = maker.batch(columns=2, rows=64)
    case = Case(batch=made, predicate=Compare(">", column("c0"), literal(-100000)))
    smaller = shrink(case, lambda one: one.batch.rows >= 1)
    assert smaller.batch.rows == 1


def test_every_differential_check_passes():
    assert differential.every_check_passes(count=30)["they_all_passed"]


def test_the_differential_harness_runs_every_check():
    assert differential.every_check_passes(count=5)["checks"] == len(CHECKS)


def test_a_broken_filter_is_caught():
    assert differential.a_broken_filter_is_caught()["it_was_caught"]


def test_the_broken_filter_failure_is_small():
    assert differential.a_broken_filter_is_caught()["the_first_is_small"]


def test_a_broken_sort_is_caught():
    assert differential.a_broken_sort_is_caught()["it_was_caught"]


def test_a_broken_aggregate_is_caught():
    assert differential.a_broken_aggregate_is_caught()["it_was_caught"]


def test_the_broken_aggregate_was_caught_by_the_nulls():
    assert differential.a_broken_aggregate_is_caught()["the_nulls_are_what_it_found"]


def test_most_generated_cases_are_sortable():
    assert differential.the_checks_do_real_work()["most_are_sortable"]


def test_some_generated_cases_are_joinable():
    assert differential.the_checks_do_real_work()["joinable"] > 0


def test_many_generated_cases_have_nulls():
    assert differential.the_checks_do_real_work()["with_nulls"] > 10


def test_reported_failures_are_near_the_boundary():
    assert differential.a_failure_is_reported_small()["they_are_all_near_the_boundary"]


def test_an_unknown_check_is_refused():
    assert differential.an_unknown_check_is_refused()


def test_a_report_renders_a_failure():
    assert differential.a_report_renders_its_first_failure()["it_has_a_predicate"]


def test_a_passing_report_renders_nothing():
    assert differential.a_report_renders_its_first_failure()["a_passing_report_renders_nothing"]


def test_the_check_table_covers_every_check():
    assert len(differential.compare_the_checks(count=5)) == len(CHECKS)


def test_the_differential_summary_says_the_breaks_are_caught():
    summary = differential.summarise()
    assert summary["a_broken_filter_is_caught"] and summary["a_broken_sort_is_caught"]


def test_running_one_check_returns_a_report():
    assert isinstance(run_one("filter agrees", count=5), Report)


def test_running_one_check_reports_its_name():
    assert run_one("filter agrees", count=5).name == "filter agrees"


def test_running_every_check_returns_one_report_each():
    assert len(run_all(count=3)) == len(CHECKS)


def test_a_passing_report_has_no_failures():
    assert run_one("filter never grows", count=10).passed


def test_a_report_reports_its_rate():
    assert run_one("filter never grows", count=10).rate == 0


def test_a_report_summarises():
    assert run_one("filter agrees", count=5).as_dict()["cases"] == 5


def test_the_failures_helper_returns_nothing_when_all_pass():
    assert differential.failures(count=10) == []


def test_an_agreement_is_falsy_when_it_disagrees():
    left = Rows.of(Batch.from_columns([integer_column("v", [1, 2, 3])]))
    right = Rows.of(Batch.from_columns([integer_column("v", [1, 2])]))
    assert not agree(left, right)


def test_an_agreement_is_truthy_when_it_agrees():
    left = Rows.of(Batch.from_columns([integer_column("v", [1, 2, 3])]))
    right = Rows.of(Batch.from_columns([integer_column("v", [1, 2, 3])]))
    assert agree(left, right)


def test_an_agreement_on_different_values_is_falsy():
    left = Rows.of(Batch.from_columns([integer_column("v", [1, 2, 3])]))
    right = Rows.of(Batch.from_columns([integer_column("v", [1, 2, 4])]))
    assert not agree(left, right)


def test_an_agreement_on_different_names_is_falsy():
    left = Rows.of(Batch.from_columns([integer_column("v", [1])]))
    right = Rows.of(Batch.from_columns([integer_column("w", [1])]))
    assert not agree(left, right)


def test_nulls_keep_their_input_order_in_a_sort():
    values = np.array([5, 0, 0, 7, 0], dtype=np.int64)
    valid = np.array([True, False, False, True, False])
    keyed = Column(field=integer_column("k", values).field, values=values, valid=valid)
    marks = integer_column("mark", [0, 1, 2, 3, 4])
    batch = Batch.from_columns([keyed, marks])
    positions = order_by(batch, [SortKey(name="k")]).positions
    ordered = batch.take(positions).column("mark").to_list()
    assert ordered[-3:] == [1, 2, 4]


def test_the_null_order_matches_the_reference():
    state = np.random.default_rng(59)
    rows = 60
    values = state.integers(0, 5, rows)
    valid = state.random(rows) > 0.4
    keyed = Column(field=integer_column("k", values).field, values=values, valid=valid)
    marks = integer_column("mark", np.arange(rows))
    batch = Batch.from_columns([keyed, marks])
    positions = order_by(batch, [SortKey(name="k")]).positions
    produced = batch.take(positions)
    expected = reference_order(Rows.of(batch), ["k"])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_a_nullable_string_key_does_not_choose_the_counting_aggregate():
    entries = ["a", "b", "a", "c", "b"]
    made = string_column("k", entries)
    valid = np.array([True, False, True, True, False])
    keyed = Column(
        field=made.field, values=made.values, valid=valid, dictionary=made.dictionary
    )
    batch = Batch.from_columns([keyed, floating_column("v", [1.0, 2.0, 3.0, 4.0, 5.0])])
    built = Group(
        input=table("facts", batch),
        keys=("k",),
        aggregates=(Aggregate(name="n", function="count_star", source=""),),
    )
    executed = run(built, {"facts": batch})
    assert executed.strategy("Group") == HASH_GROUP


def test_a_string_key_without_nulls_still_counts():
    made = string_column("k", ["a", "b", "a", "c", "b"])
    batch = Batch.from_columns([made, floating_column("v", [1.0, 2.0, 3.0, 4.0, 5.0])])
    built = Group(
        input=table("facts", batch),
        keys=("k",),
        aggregates=(Aggregate(name="n", function="count_star", source=""),),
    )
    executed = run(built, {"facts": batch})
    assert executed.strategy("Group") == COUNTING_GROUP


def test_a_nullable_string_key_groups_without_raising():
    entries = ["a", "b", "a", "c", "b"]
    made = string_column("k", entries)
    valid = np.array([True, False, True, True, False])
    keyed = Column(
        field=made.field, values=made.values, valid=valid, dictionary=made.dictionary
    )
    batch = Batch.from_columns([keyed, floating_column("v", [1.0, 2.0, 3.0, 4.0, 5.0])])
    built = Group(
        input=table("facts", batch),
        keys=("k",),
        aggregates=(Aggregate(name="n", function="count_star", source=""),),
    )
    assert execute(built, {"facts": batch}).rows > 0
