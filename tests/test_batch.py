from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import column_from
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, DataError, SchemaError, UnknownColumn
from cqe.exec.batch import (
    Batch,
    from_rows,
    mask_to_selection,
    selection_to_mask,
    side_by_side,
    stack,
)
from cqe.types.schema import INTEGER, STRING, Schema


def sample() -> Batch:
    """The batch most tests here use."""
    return Batch.of(a=[1, 2, 3, 4], g=["x", "y", "x", "z"], v=[1.5, 2.5, 3.5, 4.5])


class TestBuilding:
    def test_a_batch_reports_its_rows(self):
        assert sample().rows == 4

    def test_and_its_width(self):
        assert sample().width == 3

    def test_and_its_names(self):
        assert sample().names == ("a", "g", "v")

    def test_a_batch_has_a_length(self):
        assert len(sample()) == 4

    def test_a_batch_iterates_over_columns(self):
        assert [column.name for column in sample()] == ["a", "g", "v"]

    def test_a_batch_answers_contains(self):
        assert "a" in sample() and "z" not in sample()

    def test_a_batch_reports_its_bytes(self):
        assert sample().nbytes > 0

    def test_a_batch_converts_to_rows(self):
        assert sample().to_rows()[0] == [1, "x", 1.5]

    def test_and_has_one_row_per_row(self):
        assert len(sample().to_rows()) == 4

    def test_a_batch_serialises(self):
        assert sample().as_dict()["rows"] == 4

    def test_a_batch_prints_its_shape(self):
        assert "4 rows" in str(sample())

    def test_a_batch_builds_from_rows(self):
        batch = from_rows(["a", "b"], [[1, "x"], [2, "y"]])
        assert batch.to_rows() == [[1, "x"], [2, "y"]]

    def test_ragged_rows_are_refused(self):
        with pytest.raises(DataError, match="rows of widths"):
            from_rows(["a", "b"], [[1, 2], [3]])

    def test_a_nameless_batch_is_refused(self):
        with pytest.raises(ConfigError, match="at least one column"):
            from_rows([], [])

    def test_a_batch_of_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="at least one column"):
            Batch.of()

    def test_columns_of_different_lengths_are_refused(self):
        with pytest.raises(DataError, match="columns of lengths"):
            Batch.from_columns([column_from("a", [1, 2]), column_from("b", [1])])

    def test_a_schema_that_does_not_match_is_refused(self):
        with pytest.raises(SchemaError, match="fields against"):
            Batch(schema=Schema.of(("a", INTEGER)), columns=())

    def test_a_name_mismatch_is_refused(self):
        with pytest.raises(SchemaError, match="a against b"):
            Batch(schema=Schema.of(("a", INTEGER)), columns=(column_from("b", [1]),))

    def test_a_type_mismatch_is_refused(self):
        with pytest.raises(SchemaError, match="against"):
            Batch(schema=Schema.of(("a", STRING)), columns=(column_from("a", [1]),))


class TestEmpty:
    def test_an_empty_batch_has_no_rows(self):
        assert Batch.empty(sample().schema).rows == 0

    def test_but_keeps_its_schema(self):
        assert Batch.empty(sample().schema).names == ("a", "g", "v")

    def test_and_its_width(self):
        assert Batch.empty(sample().schema).width == 3

    def test_an_empty_batch_converts_to_no_rows(self):
        assert Batch.empty(sample().schema).to_rows() == []

    def test_an_empty_batch_carries_an_empty_dictionary(self):
        empty = Batch.empty(sample().schema)
        assert empty.column("g").dictionary == ()


class TestSelecting:
    def test_select_keeps_the_named_columns(self):
        assert sample().select(["a", "v"]).names == ("a", "v")

    def test_select_reorders(self):
        assert sample().select(["v", "a"]).names == ("v", "a")

    def test_select_keeps_the_values(self):
        assert sample().select(["a"]).to_rows()[0] == [1]

    def test_an_unknown_column_is_refused(self):
        with pytest.raises(UnknownColumn, match="not in"):
            sample().select(["z"])

    def test_select_costs_no_values(self):
        meter = Meter()
        sample().select(["a"], meter=meter)
        assert meter.values_touched == 0

    def test_but_counts_a_batch(self):
        meter = Meter()
        sample().select(["a"], meter=meter)
        assert meter.batches == 1

    def test_drop_removes_the_named_columns(self):
        assert sample().drop(["g"]).names == ("a", "v")

    def test_dropping_nothing_changes_nothing(self):
        assert sample().drop([]).names == ("a", "g", "v")

    def test_a_column_is_reachable_by_name(self):
        assert sample().column("a").to_list() == [1, 2, 3, 4]

    def test_and_its_values_directly(self):
        assert list(sample().values("a")) == [1, 2, 3, 4]


class TestTaking:
    def test_take_reorders_rows(self):
        assert sample().take(np.array([3, 0])).to_rows()[0] == [4, "z", 4.5]

    def test_take_charges_per_value_moved(self):
        meter = Meter()
        sample().take(np.array([0, 1]), meter=meter)
        assert meter.values_touched == 6

    def test_and_counts_the_rows(self):
        meter = Meter()
        sample().take(np.array([0, 1]), meter=meter)
        assert meter.rows_materialised == 2

    def test_take_attributes_to_the_take_operator(self):
        meter = Meter()
        sample().take(np.array([0]), meter=meter)
        assert meter.by_operator == {"take": 3}

    def test_mask_keeps_the_true_rows(self):
        keep = np.array([True, False, True, False])
        assert sample().mask(keep).rows == 2

    def test_a_non_boolean_mask_is_refused(self):
        with pytest.raises(DataError, match="a mask is boolean"):
            sample().mask(np.array([1, 0, 1, 0]))

    def test_a_mask_of_the_wrong_length_is_refused(self):
        with pytest.raises(DataError, match="mask entries"):
            sample().mask(np.array([True]))

    def test_slice_takes_a_run(self):
        assert sample().slice(1, 3).to_rows() == [[2, "y", 2.5], [3, "x", 3.5]]

    def test_batches_cut_evenly(self):
        assert [batch.rows for batch in sample().batches(2)] == [2, 2]

    def test_the_last_batch_is_short_rather_than_padded(self):
        assert [batch.rows for batch in sample().batches(3)] == [3, 1]

    def test_a_batch_larger_than_the_data_gives_one_batch(self):
        assert [batch.rows for batch in sample().batches(100)] == [4]

    def test_an_empty_batch_yields_one_empty_batch(self):
        empty = Batch.empty(sample().schema)
        assert [batch.rows for batch in empty.batches(10)] == [0]

    def test_a_zero_batch_size_is_refused(self):
        with pytest.raises(ConfigError, match="not a batch size"):
            list(sample().batches(0))

    def test_every_row_survives_batching(self):
        rows = [row for batch in sample().batches(3) for row in batch.to_rows()]
        assert rows == sample().to_rows()


class TestChanging:
    def test_rename_changes_a_name(self):
        assert sample().rename({"a": "id"}).names == ("id", "g", "v")

    def test_and_keeps_the_values(self):
        assert sample().rename({"a": "id"}).column("id").to_list() == [1, 2, 3, 4]

    def test_renaming_nothing_changes_nothing(self):
        assert sample().rename({}).names == ("a", "g", "v")

    def test_with_column_appends(self):
        added = sample().with_column(column_from("w", [9, 9, 9, 9]))
        assert added.names == ("a", "g", "v", "w")

    def test_with_column_replaces_an_existing_name(self):
        replaced = sample().with_column(column_from("a", [9, 9, 9, 9]))
        assert replaced.width == 3 and replaced.column("a").to_list() == [9, 9, 9, 9]

    def test_a_column_of_the_wrong_length_is_refused(self):
        with pytest.raises(DataError, match="against"):
            sample().with_column(column_from("w", [1]))


class TestStacking:
    def test_two_batches_stack(self):
        other = Batch.of(a=[9], g=["w"], v=[9.5])
        assert stack([sample(), other]).rows == 5

    def test_and_keep_their_values(self):
        other = Batch.of(a=[9], g=["w"], v=[9.5])
        assert stack([sample(), other]).to_rows()[-1] == [9, "w", 9.5]

    def test_stacking_one_batch_returns_it(self):
        batch = sample()
        assert stack([batch]) is batch

    def test_stacking_nothing_is_refused(self):
        with pytest.raises(ConfigError, match="nothing to stack"):
            stack([])

    def test_mismatched_names_are_refused(self):
        with pytest.raises(SchemaError):
            stack([sample(), Batch.of(z=[1])])

    def test_string_dictionaries_merge_across_batches(self):
        other = Batch.of(a=[9], g=["w"], v=[9.5])
        merged = stack([sample(), other])
        assert merged.column("g").dictionary == ("w", "x", "y", "z")


class TestSideBySide:
    def test_two_batches_join_widthways(self):
        left = Batch.of(a=[1, 2])
        right = Batch.of(b=[3, 4])
        assert side_by_side(left, right).names == ("a", "b")

    def test_and_keep_their_values(self):
        left = Batch.of(a=[1, 2])
        right = Batch.of(b=[3, 4])
        assert side_by_side(left, right).to_rows() == [[1, 3], [2, 4]]

    def test_a_repeated_name_is_suffixed(self):
        left = Batch.of(a=[1])
        right = Batch.of(a=[2])
        assert side_by_side(left, right).names == ("a", "a_right")

    def test_different_heights_are_refused(self):
        with pytest.raises(DataError, match="rows against"):
            side_by_side(Batch.of(a=[1, 2]), Batch.of(b=[3]))


class TestSelectionVectors:
    def test_a_selection_becomes_a_mask(self):
        assert list(selection_to_mask(np.array([0, 2]), 4)) == [True, False, True, False]

    def test_an_empty_selection_gives_an_empty_mask(self):
        assert not selection_to_mask(np.array([], dtype=np.int64), 3).any()

    def test_a_position_past_the_length_is_refused(self):
        with pytest.raises(DataError, match="against a length"):
            selection_to_mask(np.array([5]), 3)

    def test_a_negative_length_is_refused(self):
        with pytest.raises(ConfigError, match="not a length"):
            selection_to_mask(np.array([0]), -1)

    def test_a_mask_becomes_a_selection(self):
        assert list(mask_to_selection(np.array([True, False, True]))) == [0, 2]

    def test_a_non_boolean_mask_is_refused(self):
        with pytest.raises(DataError, match="a mask is boolean"):
            mask_to_selection(np.array([1, 0]))

    def test_the_two_are_inverses(self):
        keep = np.array([True, False, True, True])
        assert list(selection_to_mask(mask_to_selection(keep), 4)) == list(keep)
