from __future__ import annotations

import numpy as np
import pytest

from cqe.errors import ConfigError, SchemaError, TypeMismatch, UnknownColumn
from cqe.types.schema import (
    BOOLEAN,
    DATE,
    FLOATING,
    INTEGER,
    LOGICAL_TYPES,
    NUMERIC,
    ORDERED,
    PHYSICAL,
    STRING,
    Field,
    Schema,
    check_comparable,
    check_numeric,
    common_type,
    empty_array,
    infer_logical,
    schema_from_rows,
)


class TestField:
    def test_a_field_carries_its_type(self):
        assert Field("a", INTEGER).logical == INTEGER

    def test_and_its_physical_dtype(self):
        assert Field("a", INTEGER).physical == np.dtype(np.int64)

    def test_a_string_field_is_physically_an_integer(self):
        assert Field("a", STRING).physical == np.dtype(np.int32)

    def test_a_field_reports_its_width(self):
        assert Field("a", INTEGER).width == 8 and Field("a", STRING).width == 4

    def test_numeric_types_are_numeric(self):
        assert Field("a", INTEGER).is_numeric and Field("a", FLOATING).is_numeric

    def test_and_nothing_else_is(self):
        assert not Field("a", STRING).is_numeric
        assert not Field("a", BOOLEAN).is_numeric
        assert not Field("a", DATE).is_numeric

    def test_ordered_types_are_ordered(self):
        assert all(Field("a", one).is_ordered for one in ORDERED)

    def test_a_boolean_is_not_ordered(self):
        assert not Field("a", BOOLEAN).is_ordered

    def test_a_field_renames(self):
        assert Field("a", INTEGER).renamed("b").name == "b"

    def test_and_keeps_everything_else(self):
        original = Field("a", STRING, nullable=False)
        assert original.renamed("b").logical == STRING
        assert original.renamed("b").nullable is False

    def test_a_field_serialises(self):
        assert Field("a", INTEGER).as_dict()["type"] == INTEGER

    def test_a_field_prints_its_type(self):
        assert str(Field("a", INTEGER)) == "a integer"

    def test_a_not_null_field_says_so(self):
        assert str(Field("a", INTEGER, nullable=False)).endswith("not null")

    def test_a_nameless_field_is_refused(self):
        with pytest.raises(SchemaError, match="needs a name"):
            Field("", INTEGER)

    def test_an_unknown_type_is_refused(self):
        with pytest.raises(TypeMismatch, match="is not a type"):
            Field("a", "decimal")

    def test_the_refusal_lists_the_types(self):
        with pytest.raises(TypeMismatch, match="integer"):
            Field("a", "decimal")

    def test_a_field_is_hashable(self):
        assert len({Field("a", INTEGER), Field("a", INTEGER)}) == 1

    def test_every_type_has_a_physical_dtype(self):
        assert set(PHYSICAL) == set(LOGICAL_TYPES)


class TestSchema:
    def test_a_schema_lists_its_names(self):
        assert Schema.of(("a", INTEGER), ("b", STRING)).names == ("a", "b")

    def test_and_its_width(self):
        assert Schema.of(("a", INTEGER), ("b", STRING)).width == 2

    def test_a_schema_finds_a_column(self):
        assert Schema.of(("a", INTEGER), ("b", STRING)).index("b") == 1

    def test_and_its_type(self):
        assert Schema.of(("a", INTEGER), ("b", STRING)).logical("b") == STRING

    def test_an_unknown_column_is_refused(self):
        with pytest.raises(UnknownColumn, match="is not a column"):
            Schema.of(("a", INTEGER)).index("z")

    def test_the_refusal_lists_the_columns(self):
        with pytest.raises(UnknownColumn, match="'a'"):
            Schema.of(("a", INTEGER)).index("z")

    def test_repeated_names_are_refused(self):
        with pytest.raises(SchemaError, match="repeated column names"):
            Schema.of(("a", INTEGER), ("a", STRING))

    def test_the_refusal_names_the_repeat(self):
        with pytest.raises(SchemaError, match="'a'"):
            Schema.of(("a", INTEGER), ("a", STRING))

    def test_a_schema_selects_in_the_order_given(self):
        schema = Schema.of(("a", INTEGER), ("b", STRING), ("c", FLOATING))
        assert schema.select(["c", "a"]).names == ("c", "a")

    def test_selecting_an_unknown_column_is_refused(self):
        with pytest.raises(UnknownColumn):
            Schema.of(("a", INTEGER)).select(["z"])

    def test_a_schema_drops(self):
        schema = Schema.of(("a", INTEGER), ("b", STRING))
        assert schema.drop(["a"]).names == ("b",)

    def test_dropping_an_unknown_column_is_refused(self):
        with pytest.raises(UnknownColumn, match="cannot drop"):
            Schema.of(("a", INTEGER)).drop(["z"])

    def test_a_schema_adds(self):
        schema = Schema.of(("a", INTEGER))
        assert schema.add(Field("b", STRING)).names == ("a", "b")

    def test_a_schema_renames(self):
        schema = Schema.of(("a", INTEGER), ("b", STRING))
        assert schema.rename({"a": "x"}).names == ("x", "b")

    def test_renaming_an_unknown_column_is_refused(self):
        with pytest.raises(UnknownColumn, match="cannot rename"):
            Schema.of(("a", INTEGER)).rename({"z": "x"})

    def test_a_join_keeps_the_left_names(self):
        left = Schema.of(("a", INTEGER), ("k", STRING))
        right = Schema.of(("b", INTEGER), ("k", STRING))
        assert left.joined(right).names[:2] == ("a", "k")

    def test_and_suffixes_the_right_ones(self):
        left = Schema.of(
            ("k", STRING),
        )
        right = Schema.of(
            ("k", STRING),
        )
        assert left.joined(right).names == ("k", "k_right")

    def test_a_self_join_disambiguates_every_column(self):
        schema = Schema.of(("a", INTEGER), ("b", STRING))
        assert schema.joined(schema).names == ("a", "b", "a_right", "b_right")

    def test_a_repeated_suffix_keeps_going(self):
        left = Schema.of(("k", STRING), ("k_right", STRING))
        right = Schema.of(
            ("k", STRING),
        )
        assert left.joined(right).names[-1] == "k_right_right"

    def test_a_schema_reports_bytes_per_row(self):
        assert Schema.of(("a", INTEGER), ("b", STRING)).bytes_per_row == 12

    def test_a_schema_iterates(self):
        schema = Schema.of(("a", INTEGER), ("b", STRING))
        assert [one.name for one in schema] == ["a", "b"]

    def test_a_schema_has_a_length(self):
        assert len(Schema.of(("a", INTEGER), ("b", STRING))) == 2

    def test_a_schema_answers_contains(self):
        schema = Schema.of(
            ("a", INTEGER),
        )
        assert "a" in schema and "z" not in schema

    def test_a_schema_serialises(self):
        assert len(Schema.of(("a", INTEGER)).as_dict()["columns"]) == 1

    def test_a_schema_prints(self):
        assert str(Schema.of(("a", INTEGER), ("b", STRING))) == "a integer, b string"

    def test_an_empty_schema_is_allowed(self):
        assert Schema().width == 0


class TestPromotion:
    def test_a_type_combines_with_itself(self):
        assert common_type(INTEGER, INTEGER) == INTEGER

    def test_an_integer_promotes_to_a_float(self):
        assert common_type(INTEGER, FLOATING) == FLOATING

    def test_in_either_order(self):
        assert common_type(FLOATING, INTEGER) == FLOATING

    def test_a_string_and_an_integer_are_refused(self):
        with pytest.raises(TypeMismatch, match="cannot be combined"):
            common_type(STRING, INTEGER)

    def test_a_date_and_an_integer_are_refused(self):
        with pytest.raises(TypeMismatch):
            common_type(DATE, INTEGER)

    def test_a_boolean_and_an_integer_are_refused(self):
        with pytest.raises(TypeMismatch):
            common_type(BOOLEAN, INTEGER)

    def test_comparable_types_are_comparable(self):
        assert check_comparable(INTEGER, FLOATING) == FLOATING

    def test_booleans_compare_to_booleans(self):
        assert check_comparable(BOOLEAN, BOOLEAN) == BOOLEAN

    def test_strings_compare_to_strings(self):
        assert check_comparable(STRING, STRING) == STRING

    def test_numeric_checks_pass_for_numbers(self):
        assert check_numeric(INTEGER) == INTEGER

    def test_and_refuse_everything_else(self):
        with pytest.raises(TypeMismatch, match="does not support"):
            check_numeric(STRING)

    def test_the_refusal_names_the_operation(self):
        with pytest.raises(TypeMismatch, match="summing"):
            check_numeric(STRING, "summing")

    def test_every_numeric_type_is_ordered(self):
        assert set(NUMERIC) <= set(ORDERED)


class TestInference:
    def test_integers_infer_as_integer(self):
        assert infer_logical([1, 2, 3]) == INTEGER

    def test_floats_infer_as_floating(self):
        assert infer_logical([1.5, 2.5]) == FLOATING

    def test_a_mix_of_ints_and_floats_infers_as_floating(self):
        assert infer_logical([1, 2.5]) == FLOATING

    def test_strings_infer_as_string(self):
        assert infer_logical(["a", "b"]) == STRING

    def test_booleans_infer_as_boolean_not_integer(self):
        assert infer_logical([True, False]) == BOOLEAN

    def test_nulls_are_ignored_when_inferring(self):
        assert infer_logical([1, None, 3]) == INTEGER

    def test_nothing_cannot_be_inferred(self):
        with pytest.raises(SchemaError, match="from nothing"):
            infer_logical([])

    def test_nor_can_nulls_alone(self):
        with pytest.raises(SchemaError, match="from nulls alone"):
            infer_logical([None, None])

    def test_mixed_types_are_refused(self):
        with pytest.raises(SchemaError, match="mixed value types"):
            infer_logical([1, "a"])

    def test_the_refusal_names_the_types(self):
        with pytest.raises(SchemaError, match="int"):
            infer_logical([1, "a"])

    def test_a_schema_infers_from_rows(self):
        schema = schema_from_rows(["a", "b"], [[1, "x"], [2, "y"]])
        assert schema.logical("a") == INTEGER and schema.logical("b") == STRING

    def test_nullability_is_inferred_too(self):
        schema = schema_from_rows(["a"], [[1], [None]])
        assert schema.field("a").nullable

    def test_and_absence_of_nulls_is_noticed(self):
        schema = schema_from_rows(["a"], [[1], [2]])
        assert not schema.field("a").nullable

    def test_a_nameless_schema_is_refused(self):
        with pytest.raises(SchemaError, match="at least one column"):
            schema_from_rows([], [])

    def test_ragged_rows_are_refused(self):
        with pytest.raises(SchemaError, match="rows of widths"):
            schema_from_rows(["a", "b"], [[1, 2], [3]])


class TestArrays:
    def test_an_empty_array_has_the_right_dtype(self):
        assert empty_array(INTEGER, 4).dtype == np.dtype(np.int64)

    def test_and_the_right_length(self):
        assert empty_array(FLOATING, 4).shape == (4,)

    def test_a_zero_length_array_is_allowed(self):
        assert empty_array(STRING, 0).shape == (0,)

    def test_a_negative_length_is_refused(self):
        with pytest.raises(ConfigError, match="not a length"):
            empty_array(INTEGER, -1)

    def test_a_boolean_array_is_boolean(self):
        assert empty_array(BOOLEAN, 2).dtype == np.dtype(np.bool_)

    def test_a_date_array_is_a_small_integer(self):
        assert empty_array(DATE, 2).dtype == np.dtype(np.int32)
