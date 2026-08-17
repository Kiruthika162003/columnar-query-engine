from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.columns.encode import choose as chooser
from cqe.columns.encode.choose import (
    NAMES,
    RAW,
    SAMPLE_ROWS,
    WORTH_IT,
    Candidate,
    candidates,
    choose,
    choose_all,
)
from cqe.errors import ConfigError
from cqe.exec.batch import Batch


@pytest.fixture(scope="module")
def batch() -> Batch:
    """A table with columns of four different shapes."""
    state = np.random.default_rng(317)
    rows = 8000
    return Batch.from_columns(
        [
            integer_column("rising", np.arange(rows)),
            integer_column("narrow", state.integers(500, 560, rows)),
            floating_column("amount", state.normal(100, 20, rows)),
            string_column("label", [f"kind{one}" for one in state.integers(0, 5, rows)]),
        ]
    )


def test_different_shapes_get_different_encodings():
    assert chooser.every_shape_gets_a_different_encoding()["it_picked_several"]


def test_a_float_gets_no_encoding():
    assert chooser.every_shape_gets_a_different_encoding()["a_float_gets_nothing"]


def test_a_rising_column_picks_delta():
    assert chooser.a_rising_column_wants_delta()["it_picked_delta"]


def test_the_delta_saving_is_large():
    assert chooser.a_rising_column_wants_delta()["and_the_saving_is_large"]


def test_a_repeated_column_picks_run_length():
    assert chooser.a_repeated_column_wants_run_length()["it_picked_run_length"]


def test_the_run_length_saving_is_large():
    assert chooser.a_repeated_column_wants_run_length()["and_the_saving_is_large"]


def test_a_low_cardinality_string_picks_a_dictionary():
    assert chooser.a_low_cardinality_string_wants_a_dictionary()[
        "the_low_cardinality_one_wants_a_dictionary"
    ]


def test_a_high_cardinality_string_saves_less():
    assert chooser.a_low_cardinality_string_wants_a_dictionary()[
        "and_the_high_cardinality_one_saves_less"
    ]


def test_a_float_column_picks_raw():
    assert chooser.a_float_column_wants_nothing()["it_picked_raw"]


def test_nothing_was_worth_it_for_the_float():
    assert chooser.a_float_column_wants_nothing()["and_nothing_was_worth_it"]


def test_the_sample_agrees_with_the_whole_column():
    assert chooser.the_sample_agrees_with_the_whole_column()["they_all_agree"]


def test_no_natural_column_disagreed():
    assert chooser.the_sample_agrees_with_the_whole_column()["which_disagreed"] == []


def test_a_constructed_prefix_fools_the_sample():
    assert chooser.a_sample_can_be_wrong_about_a_sorted_prefix()["they_disagree"]


def test_the_sample_saw_runs_that_are_not_there():
    assert chooser.a_sample_can_be_wrong_about_a_sorted_prefix()["the_sample_saw_runs"]


def test_a_marginal_saving_is_refused():
    assert chooser.a_marginal_saving_is_refused()["so_it_picked_raw"]


def test_the_marginal_ratio_is_above_the_threshold():
    assert chooser.a_marginal_saving_is_refused()["it_was_below_the_threshold"]


def test_chaining_would_save_more():
    assert chooser.chaining_would_save_more()["the_chain_is_smaller"]


def test_the_chain_saving_is_worth_stating():
    assert chooser.chaining_would_save_more()["by_this_ratio"] > 1.5


def test_every_row_group_chooses_the_same():
    assert chooser.one_encoding_per_column_costs_little()["they_all_chose_the_same"]


def test_the_group_choice_matches_the_column():
    assert chooser.one_encoding_per_column_costs_little()["and_it_matches_the_whole_column"]


def test_choosing_reads_a_fraction_of_the_column():
    assert chooser.choosing_costs_far_less_than_encoding()["share_read"] < 0.2


def test_choosing_costs_less_than_a_pass():
    assert chooser.choosing_costs_far_less_than_encoding()["the_work_is_a_fraction_of_one_pass"]


def test_every_choice_decodes():
    assert chooser.every_choice_is_decodable()["they_all_decode"]


def test_an_empty_column_picks_raw():
    assert chooser.an_empty_column_picks_raw()["it_picked_raw"]


def test_a_zero_sample_is_refused():
    assert chooser.a_zero_sample_is_refused()


def test_the_column_table_covers_seven_shapes():
    assert len(chooser.compare_the_columns()) == 7


def test_every_row_of_the_table_names_a_choice():
    assert all(one["picked"] in NAMES for one in chooser.compare_the_columns())


def test_the_summary_says_every_choice_decodes():
    assert chooser.summarise()["every_choice_decodes"]


def test_choosing_returns_a_choice(batch):
    made = choose(batch.column("rising"))
    assert made.column == "rising" and made.picked in NAMES


def test_a_choice_reports_its_saving(batch):
    assert choose(batch.column("rising")).saving > 0.5


def test_a_choice_names_its_runner_up(batch):
    assert choose(batch.column("rising")).runner_up in NAMES


def test_a_choice_reports_what_it_sampled(batch):
    assert choose(batch.column("rising")).sampled == min(batch.rows, SAMPLE_ROWS)


def test_a_choice_summarises(batch):
    assert choose(batch.column("rising")).as_dict()["column"] == "rising"


def test_every_encoding_is_tried(batch):
    assert len(candidates(batch.column("rising"))) == len(NAMES)


def test_raw_is_always_usable(batch):
    made = candidates(batch.column("amount"))
    assert made[0].name == RAW and made[0].usable


def test_raw_has_a_ratio_of_one(batch):
    assert candidates(batch.column("amount"))[0].ratio == 1.0


def test_a_dictionary_is_unusable_on_a_number(batch):
    made = {one.name: one for one in candidates(batch.column("rising"))}
    assert not made["dictionary"].usable


def test_bit_packing_is_unusable_on_a_string(batch):
    made = {one.name: one for one in candidates(batch.column("label"))}
    assert not made["bit packing"].usable


def test_delta_is_unusable_on_a_string(batch):
    made = {one.name: one for one in candidates(batch.column("label"))}
    assert not made["delta"].usable


def test_an_unusable_candidate_gives_a_reason(batch):
    made = {one.name: one for one in candidates(batch.column("label"))}
    assert made["delta"].reason


def test_a_candidate_summarises(batch):
    made = candidates(batch.column("rising"))[0]
    assert made.as_dict()["encoding"] == RAW


def test_a_candidate_that_saves_nothing_is_not_worth_it():
    made = Candidate(name="x", raw_bytes=100, encoded_bytes=100)
    assert not made.worth_it


def test_a_candidate_that_halves_is_worth_it():
    made = Candidate(name="x", raw_bytes=100, encoded_bytes=50)
    assert made.worth_it


def test_a_candidate_just_inside_the_threshold_is_not():
    made = Candidate(name="x", raw_bytes=100, encoded_bytes=96)
    assert not made.worth_it


def test_an_unusable_candidate_is_never_worth_it():
    made = Candidate(name="x", raw_bytes=100, encoded_bytes=10, usable=False)
    assert not made.worth_it


def test_choosing_every_column_gives_one_each(batch):
    assert len(choose_all(batch)) == batch.width


def test_choosing_every_column_names_them(batch):
    assert [one.column for one in choose_all(batch)] == list(batch.schema.names)


def test_a_narrow_integer_picks_bit_packing(batch):
    assert choose(batch.column("narrow")).picked == "bit packing"


def test_a_label_column_picks_a_dictionary(batch):
    assert choose(batch.column("label")).picked == "dictionary"


def test_a_float_column_picks_raw_in_a_batch(batch):
    assert choose(batch.column("amount")).picked == RAW


def test_the_threshold_is_five_percent():
    assert WORTH_IT == 0.95


def test_the_sample_is_a_power_of_two():
    assert SAMPLE_ROWS == 4096


def test_a_negative_sample_is_refused(batch):
    with pytest.raises(ConfigError):
        choose(batch.column("rising"), sample_rows=-1)


def test_a_sample_larger_than_the_column_reads_the_column(batch):
    assert choose(batch.column("rising"), sample_rows=999999).sampled == batch.rows


def test_a_constant_column_encodes_well():
    made = choose(integer_column("v", [7] * 5000))
    assert made.saving > 0.9


def test_a_two_row_column_can_still_choose():
    assert choose(integer_column("v", [1, 2])).picked in NAMES
