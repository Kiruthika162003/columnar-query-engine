from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import (
    Column,
    EncodingRequired,
    all_valid,
    boolean_column,
    column_from,
    combine_validity,
    concat,
    date_from_days,
    floating_column,
    integer_column,
    string_column,
)
from cqe.errors import ConfigError, DataError, SchemaError, TypeMismatch
from cqe.types.schema import BOOLEAN, DATE, FLOATING, INTEGER, STRING, Field


class TestBuilding:
    def test_an_integer_column_holds_its_values(self):
        assert column_from("a", [1, 2, 3]).to_list() == [1, 2, 3]

    def test_a_floating_column_holds_its_values(self):
        assert column_from("a", [1.5, 2.5]).to_list() == [1.5, 2.5]

    def test_a_boolean_column_holds_its_values(self):
        assert column_from("a", [True, False]).to_list() == [True, False]

    def test_a_string_column_holds_its_values(self):
        assert column_from("a", ["x", "y", "x"]).to_list() == ["x", "y", "x"]

    def test_nulls_survive_a_round_trip(self):
        assert column_from("a", [1, None, 3]).to_list() == [1, None, 3]

    def test_a_string_column_with_nulls_survives_too(self):
        assert column_from("a", ["x", None]).to_list() == ["x", None]

    def test_a_column_reports_its_length(self):
        assert len(column_from("a", [1, 2, 3])) == 3

    def test_a_column_reports_its_null_count(self):
        assert column_from("a", [1, None, None]).null_count == 2

    def test_a_column_with_no_nulls_carries_no_mask(self):
        assert column_from("a", [1, 2]).valid is None

    def test_and_reports_zero_nulls(self):
        assert column_from("a", [1, 2]).null_count == 0

    def test_has_nulls_is_false_without_a_mask(self):
        assert not column_from("a", [1, 2]).has_nulls

    def test_and_true_with_one(self):
        assert column_from("a", [1, None]).has_nulls

    def test_the_dictionary_of_a_string_column_is_sorted(self):
        assert column_from("a", ["c", "a", "b"]).dictionary == ("a", "b", "c")

    def test_and_holds_each_value_once(self):
        assert column_from("a", ["a", "a", "a"]).dictionary == ("a",)

    def test_the_codes_index_the_dictionary(self):
        column = column_from("a", ["c", "a", "b"])
        assert list(column.values) == [2, 0, 1]

    def test_the_codes_are_in_dictionary_order_not_first_seen_order(self):
        column = column_from("a", ["c", "a"])
        assert list(column.values) == [1, 0]

    def test_a_distinct_count_comes_from_the_dictionary(self):
        assert column_from("a", ["x", "y", "x"]).distinct_estimate == 2

    def test_and_from_the_values_otherwise(self):
        assert column_from("a", [1, 1, 2]).distinct_estimate == 2

    def test_a_column_reports_its_bytes(self):
        assert column_from("a", [1, 2]).nbytes == 16

    def test_a_mask_adds_to_the_bytes(self):
        assert column_from("a", [1, None]).nbytes > 16

    def test_a_column_serialises(self):
        assert column_from("a", [1, 2]).as_dict()["rows"] == 2

    def test_the_typed_constructors_agree_with_inference(self):
        assert integer_column("a", [1, 2]).logical == INTEGER
        assert floating_column("a", [1.0]).logical == FLOATING
        assert boolean_column("a", [True]).logical == BOOLEAN
        assert string_column("a", ["x"]).logical == STRING

    def test_a_date_column_is_days(self):
        assert date_from_days("d", [0, 1, 2]).logical == DATE

    def test_and_holds_integers(self):
        assert date_from_days("d", [5]).to_list() == [5]

    def test_a_column_can_be_forced_to_a_type(self):
        assert column_from("a", [1, 2], logical=FLOATING).logical == FLOATING


class TestValidation:
    def test_a_two_dimensional_array_is_refused(self):
        with pytest.raises(DataError, match="one dimensional"):
            Column(Field("a", INTEGER), np.zeros((2, 2), dtype=np.int64))

    def test_the_wrong_dtype_is_refused(self):
        with pytest.raises(TypeMismatch, match="needs"):
            Column(Field("a", INTEGER), np.zeros(2, dtype=np.float64))

    def test_a_non_boolean_mask_is_refused(self):
        with pytest.raises(DataError, match="validity mask is boolean"):
            Column(
                Field("a", INTEGER), np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)
            )

    def test_a_mask_of_the_wrong_length_is_refused(self):
        with pytest.raises(DataError, match="validity entries"):
            Column(Field("a", INTEGER), np.zeros(2, dtype=np.int64), np.zeros(3, dtype=bool))

    def test_a_string_column_without_a_dictionary_is_refused(self):
        with pytest.raises(EncodingRequired, match="no dictionary"):
            Column(Field("a", STRING), np.zeros(2, dtype=np.int32))

    def test_a_dictionary_on_a_non_string_column_is_refused(self):
        with pytest.raises(TypeMismatch, match="do not carry a dictionary"):
            Column(Field("a", INTEGER), np.zeros(2, dtype=np.int64), dictionary=("x",))

    def test_a_code_past_the_dictionary_is_refused(self):
        with pytest.raises(DataError, match="against a dictionary"):
            Column(Field("a", STRING), np.array([5], dtype=np.int32), dictionary=("x",))

    def test_an_encoding_required_is_a_schema_error(self):
        assert issubclass(EncodingRequired, SchemaError)


class TestTaking:
    def test_take_reorders(self):
        column = column_from("a", [10, 20, 30])
        assert column.take(np.array([2, 0])).to_list() == [30, 10]

    def test_take_can_repeat(self):
        column = column_from("a", [10, 20])
        assert column.take(np.array([0, 0, 1])).to_list() == [10, 10, 20]

    def test_take_carries_the_mask(self):
        column = column_from("a", [1, None, 3])
        assert column.take(np.array([1, 2])).to_list() == [None, 3]

    def test_take_carries_the_dictionary(self):
        column = column_from("a", ["x", "y"])
        assert column.take(np.array([1])).to_list() == ["y"]

    def test_a_float_position_array_is_refused(self):
        with pytest.raises(DataError, match="positions are integers"):
            column_from("a", [1]).take(np.array([0.0]))

    def test_a_position_past_the_end_is_refused(self):
        with pytest.raises(DataError, match="against a column"):
            column_from("a", [1, 2]).take(np.array([5]))

    def test_an_empty_take_gives_an_empty_column(self):
        assert len(column_from("a", [1, 2]).take(np.array([], dtype=np.int64))) == 0

    def test_mask_keeps_the_true_entries(self):
        column = column_from("a", [1, 2, 3])
        assert column.mask(np.array([True, False, True])).to_list() == [1, 3]

    def test_a_non_boolean_mask_is_refused(self):
        with pytest.raises(DataError, match="a mask is boolean"):
            column_from("a", [1]).mask(np.array([1]))

    def test_a_mask_of_the_wrong_length_is_refused(self):
        with pytest.raises(DataError, match="mask entries"):
            column_from("a", [1, 2]).mask(np.array([True]))

    def test_slice_takes_a_run(self):
        assert column_from("a", [1, 2, 3, 4]).slice(1, 3).to_list() == [2, 3]

    def test_slice_to_the_end_by_default(self):
        assert column_from("a", [1, 2, 3]).slice(1).to_list() == [2, 3]

    def test_a_slice_outside_the_column_is_refused(self):
        with pytest.raises(ConfigError, match="outside a column"):
            column_from("a", [1, 2]).slice(0, 5)

    def test_a_backwards_slice_is_refused(self):
        with pytest.raises(ConfigError, match="outside a column"):
            column_from("a", [1, 2]).slice(2, 1)


class TestTransforming:
    def test_renaming_keeps_the_values(self):
        assert column_from("a", [1, 2]).renamed("b").to_list() == [1, 2]

    def test_and_changes_the_name(self):
        assert column_from("a", [1]).renamed("b").name == "b"

    def test_filling_nulls_replaces_them(self):
        assert column_from("a", [1, None, 3]).fill_null(0).to_list() == [1, 0, 3]

    def test_and_drops_the_mask(self):
        assert column_from("a", [1, None]).fill_null(0).valid is None

    def test_filling_a_column_with_no_nulls_changes_nothing(self):
        column = column_from("a", [1, 2])
        assert column.fill_null(0) is column

    def test_a_column_reports_its_logical_type(self):
        assert column_from("a", [1]).logical == INTEGER


class TestConcatenating:
    def test_two_integer_columns_stack(self):
        left = column_from("a", [1, 2])
        right = column_from("a", [3])
        assert concat([left, right]).to_list() == [1, 2, 3]

    def test_two_string_columns_merge_their_dictionaries(self):
        left = column_from("a", ["x", "y"])
        right = column_from("a", ["z"])
        assert concat([left, right]).dictionary == ("x", "y", "z")

    def test_and_the_values_survive(self):
        left = column_from("a", ["b", "a"])
        right = column_from("a", ["c", "a"])
        assert concat([left, right]).to_list() == ["b", "a", "c", "a"]

    def test_a_shared_dictionary_is_not_duplicated(self):
        left = column_from("a", ["x"])
        right = column_from("a", ["x"])
        assert concat([left, right]).dictionary == ("x",)

    def test_nulls_from_one_side_survive(self):
        left = column_from("a", [1, None])
        right = column_from("a", [3])
        assert concat([left, right]).to_list() == [1, None, 3]

    def test_a_column_with_no_nulls_gains_no_mask(self):
        left = column_from("a", [1])
        right = column_from("a", [2])
        assert concat([left, right]).valid is None

    def test_mismatched_types_are_refused(self):
        with pytest.raises(TypeMismatch):
            concat([column_from("a", [1]), column_from("a", ["x"])])

    def test_mismatched_names_are_refused(self):
        with pytest.raises(SchemaError):
            concat([column_from("a", [1]), column_from("b", [2])])

    def test_concatenating_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to concatenate"):
            concat([])

    def test_concatenating_one_column_returns_its_values(self):
        assert concat([column_from("a", [1, 2])]).to_list() == [1, 2]


class TestValidityHelpers:
    def test_all_valid_is_all_true(self):
        assert bool(all_valid(4).all())

    def test_all_valid_has_the_right_length(self):
        assert all_valid(4).shape == (4,)

    def test_a_negative_length_is_refused(self):
        with pytest.raises(ConfigError, match="not a length"):
            all_valid(-1)

    def test_combining_two_absent_masks_gives_nothing(self):
        assert combine_validity(None, None) is None

    def test_combining_one_mask_gives_that_mask(self):
        mask = np.array([True, False])
        assert combine_validity(mask, None) is mask

    def test_in_either_order(self):
        mask = np.array([True, False])
        assert combine_validity(None, mask) is mask

    def test_combining_two_masks_takes_the_intersection(self):
        left = np.array([True, True, False])
        right = np.array([True, False, True])
        assert list(combine_validity(left, right)) == [True, False, False]

    def test_mismatched_masks_are_refused(self):
        with pytest.raises(DataError, match="validity entries"):
            combine_validity(np.array([True]), np.array([True, False]))
