from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import Column, boolean_column, floating_column, integer_column
from cqe.columns.array import string_column as make_string
from cqe.errors import DataError, TypeMismatch
from cqe.exec.batch import Batch
from cqe.types import casting
from cqe.types.casting import (
    EXACT_INTEGER,
    can_narrow,
    can_widen,
    cast,
    cast_batch,
    parse,
    what_it_loses,
)
from cqe.types.schema import BOOLEAN, DATE, FLOATING, INTEGER, PHYSICAL, STRING, Field


def test_a_widening_conversion_loses_nothing():
    assert casting.a_widening_conversion_loses_nothing()["and_it_lost_nothing"]


def test_a_widened_column_holds_the_same_values():
    assert casting.a_widening_conversion_loses_nothing()["the_values_match"]


def test_small_integers_survive_a_float():
    assert casting.a_large_integer_does_not_survive_a_float()["the_small_ones_are_exact"]


def test_large_integers_do_not():
    assert casting.a_large_integer_does_not_survive_a_float()["and_the_large_ones_are_not"]


def test_about_half_the_large_integers_change():
    measured = casting.a_large_integer_does_not_survive_a_float()
    assert 0.3 < measured["the_share_that_changed"] < 0.7


def test_a_narrowing_conversion_reports_its_loss():
    assert casting.a_narrowing_conversion_reports_what_it_lost()["it_lost_something"]


def test_a_whole_numbered_float_narrows_cleanly():
    assert casting.a_narrowing_conversion_reports_what_it_lost()[
        "and_a_whole_numbered_column_loses_nothing"
    ]


def test_a_round_trip_changes_the_total():
    assert casting.a_round_trip_through_an_integer_is_not_the_original()["the_totals_differ"]


def test_the_second_cast_reports_nothing_lost():
    assert casting.a_round_trip_through_an_integer_is_not_the_original()[
        "and_the_second_cast_reported_nothing"
    ]


def test_a_boolean_round_trips_through_an_integer():
    assert casting.a_boolean_widens_to_an_integer()["and_they_round_trip"]


def test_an_integer_narrows_to_a_boolean_badly():
    assert casting.an_integer_narrows_to_a_boolean_badly()["most_values_changed"]


def test_a_binary_column_narrows_cleanly():
    assert casting.an_integer_narrows_to_a_boolean_badly()["and_a_binary_column_loses_nothing"]


def test_a_string_is_never_cast():
    assert casting.a_string_is_never_cast()["it_was_refused"]


def test_the_refusal_says_to_parse():
    assert casting.a_string_is_never_cast()["it_says_to_parse"]


def test_parsing_the_same_column_works():
    assert casting.a_string_is_never_cast()["and_parsing_works"]


def test_parsing_refuses_by_default():
    assert casting.parsing_refuses_by_default()["it_was_refused"]


def test_the_parse_refusal_names_the_value():
    assert casting.parsing_refuses_by_default()["it_names_the_value"]


def test_the_null_policy_keeps_every_row():
    assert casting.parsing_can_null_the_failures()["it_kept_every_row"]


def test_the_null_policy_nulls_the_bad_one():
    assert casting.parsing_can_null_the_failures()["the_bad_one_is_null"]


def test_the_skip_policy_drops_the_bad_one():
    assert casting.parsing_can_skip_the_failures()["it_dropped_the_bad_one"]


def test_the_skip_policy_keeps_the_rest():
    assert casting.parsing_can_skip_the_failures()["and_kept_the_rest"]


def test_a_boolean_accepts_several_spellings():
    assert casting.parsing_a_boolean_accepts_several_spellings()["it_parsed_them_all"]


def test_an_unknown_spelling_is_refused():
    assert casting.parsing_a_boolean_accepts_several_spellings()["and_maybe_is_refused"]


def test_a_null_string_stays_null():
    assert casting.parsing_a_null_is_a_failure()["the_null_stayed_null"]


def test_a_nonsense_conversion_is_refused():
    assert casting.a_nonsense_conversion_is_refused()["it_was_refused"]


def test_the_nonsense_refusal_names_both_types():
    assert casting.a_nonsense_conversion_is_refused()["it_names_both_types"]


def test_a_conversion_keeps_the_null_mask():
    assert casting.a_conversion_keeps_the_nulls()["and_the_mask_is_the_same"]


def test_casting_a_batch_changes_the_named_columns():
    measured = casting.casting_a_batch_changes_several_columns()
    assert measured["the_widened_one_is_floating"] and measured["the_narrowed_one_is_integer"]


def test_casting_a_batch_leaves_the_others_alone():
    assert casting.casting_a_batch_changes_several_columns()["the_untouched_one_is_the_same"]


def test_an_unknown_type_is_refused():
    assert casting.an_unknown_type_is_refused()


def test_an_unknown_policy_is_refused():
    assert casting.an_unknown_policy_is_refused()


def test_parsing_a_number_is_refused():
    assert casting.parsing_a_number_column_is_refused()


def test_the_conversion_table_has_eight_entries():
    assert len(casting.compare_the_conversions()) == 8


def test_every_widening_loses_nothing():
    table = casting.compare_the_conversions()
    assert all(one["loses"] == "nothing" for one in table if one["kind"] == "widening")


def test_the_summary_says_text_is_parsed():
    assert casting.summarise()["text_is_parsed_not_cast"]


def test_an_integer_widens_to_a_float():
    assert can_widen(INTEGER, FLOATING)


def test_a_float_does_not_widen_to_an_integer():
    assert not can_widen(FLOATING, INTEGER)


def test_a_float_narrows_to_an_integer():
    assert can_narrow(FLOATING, INTEGER)


def test_a_string_neither_widens_nor_narrows():
    assert not can_widen(STRING, INTEGER) and not can_narrow(STRING, INTEGER)


def test_a_widening_loses_nothing_by_name():
    assert what_it_loses(INTEGER, FLOATING) == "nothing"


def test_a_narrowing_names_what_it_loses():
    assert what_it_loses(FLOATING, INTEGER) == "the fractional part"


def test_a_nonsense_conversion_says_so():
    assert what_it_loses(STRING, INTEGER) == "it is not a conversion"


def test_casting_to_the_same_type_is_free():
    column = integer_column("v", [1, 2, 3])
    made = cast(column, INTEGER)
    assert made.lossless and made.column is column


def test_a_conversion_reports_its_kind():
    assert cast(integer_column("v", [1]), FLOATING).kind == "widening"


def test_a_narrowing_conversion_reports_its_kind():
    assert cast(floating_column("v", [1.5]), INTEGER).kind == "narrowing"


def test_a_conversion_summarises():
    made = cast(integer_column("v", [1, 2]), FLOATING)
    assert made.as_dict()["to"] == FLOATING


def test_a_truncating_cast_rounds_towards_zero():
    made = cast(floating_column("v", [1.9, -1.9]), INTEGER)
    assert made.column.to_list() == [1, -1]


def test_a_boolean_becomes_one_and_zero():
    made = cast(boolean_column("v", [True, False]), INTEGER)
    assert made.column.to_list() == [1, 0]


def test_a_nonzero_integer_becomes_true():
    made = cast(integer_column("v", [0, 1, 7]), BOOLEAN)
    assert made.column.to_list() == [False, True, True]


def test_the_boundary_integer_is_exact():
    made = cast(integer_column("v", [EXACT_INTEGER]), FLOATING)
    assert made.lossless


def test_one_past_the_boundary_is_not():
    made = cast(integer_column("v", [EXACT_INTEGER + 1]), FLOATING)
    assert not made.lossless


def test_an_empty_column_casts_cleanly():
    made = cast(integer_column("v", []), FLOATING)
    assert made.lossless and len(made.column) == 0


def test_a_date_widens_to_an_integer():
    values = np.arange(5).astype(PHYSICAL[DATE])
    column = Column(field=Field(name="v", logical=DATE, nullable=False), values=values)
    assert cast(column, INTEGER).lossless


def test_parsing_an_integer_column_works():
    made = parse(make_string("v", ["1", "2", "3"]), INTEGER)
    assert made.column.to_list() == [1, 2, 3]


def test_parsing_a_float_column_works():
    made = parse(make_string("v", ["1.5", "2.5"]), FLOATING)
    assert made.column.to_list() == [1.5, 2.5]


def test_parsing_a_boolean_column_works():
    made = parse(make_string("v", ["true", "false"]), BOOLEAN)
    assert made.column.to_list() == [True, False]


def test_parsing_a_bad_float_is_refused():
    with pytest.raises(DataError):
        parse(make_string("v", ["1.5", "x"]), FLOATING)


def test_parsing_with_an_unknown_policy_is_refused():
    with pytest.raises(TypeMismatch):
        parse(make_string("v", ["1"]), INTEGER, on_error="guess")


def test_casting_to_an_unknown_type_is_refused():
    with pytest.raises(TypeMismatch):
        cast(integer_column("v", [1]), "decimal")


def test_a_batch_cast_returns_a_batch():
    batch = Batch.from_columns([integer_column("a", [1, 2]), floating_column("b", [1.5, 2.5])])
    made = cast_batch(batch, {"a": FLOATING})
    assert isinstance(made, Batch) and made.column("a").field.logical == FLOATING


def test_a_batch_cast_with_no_targets_is_the_batch():
    batch = Batch.from_columns([integer_column("a", [1, 2])])
    assert cast_batch(batch, {}).column("a") is batch.column("a")
