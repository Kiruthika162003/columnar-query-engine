from __future__ import annotations

import pytest

from cqe.errors import ConfigError, SchemaError, TypeMismatch, UnknownColumn
from cqe.exec.batch import Batch
from cqe.verify.reference import (
    Rows,
    agree,
    and_,
    anti_join,
    apply_aggregate,
    compare,
    distinct,
    distinct_from,
    equals,
    group_by,
    inner_join,
    left_join,
    limit,
    not_,
    or_,
    order_by,
    select,
    semi_join,
    truth,
    union_all,
    where,
)


def people() -> Rows:
    """The left table most tests here use."""
    return Rows(
        names=("id", "team", "score"),
        rows=[[1, "a", 10], [2, "b", 20], [3, "a", 30], [4, None, 40]],
    )


def teams() -> Rows:
    """The right table most tests here use."""
    return Rows(names=("team", "city"), rows=[["a", "york"], ["b", "leeds"], ["c", "hull"]])


class TestRows:
    def test_a_table_reports_its_width(self):
        assert people().width == 3

    def test_and_its_height(self):
        assert len(people()) == 4

    def test_a_table_finds_a_column(self):
        assert people().index("team") == 1

    def test_and_reads_it(self):
        assert people().column("score") == [10, 20, 30, 40]

    def test_an_unknown_column_is_refused(self):
        with pytest.raises(UnknownColumn, match="is not a column"):
            people().index("z")

    def test_a_nameless_table_is_refused(self):
        with pytest.raises(SchemaError, match="at least one column"):
            Rows(names=(), rows=[])

    def test_ragged_rows_are_refused(self):
        with pytest.raises(SchemaError, match="rows of widths"):
            Rows(names=("a", "b"), rows=[[1, 2], [3]])

    def test_a_table_serialises(self):
        assert people().as_dict()["rows"] == 4

    def test_a_batch_converts_to_rows_and_back(self):
        batch = Batch.of(a=[1, 2], g=["x", "y"])
        assert Rows.of(batch).to_batch().to_rows() == batch.to_rows()


class TestThreeValuedLogic:
    def test_a_comparison_with_null_is_null(self):
        assert compare(1, None) is None and compare(None, 1) is None

    def test_a_comparison_orders(self):
        assert compare(1, 2) == -1 and compare(2, 1) == 1 and compare(1, 1) == 0

    def test_a_string_and_a_number_cannot_be_compared(self):
        with pytest.raises(TypeMismatch):
            compare(1, "a")

    def test_a_boolean_and_a_number_cannot_be_compared(self):
        with pytest.raises(TypeMismatch):
            compare(True, 1)

    def test_truth_keeps_only_true(self):
        assert truth(True) and not truth(False) and not truth(None)

    def test_and_with_false_is_false_even_against_null(self):
        assert and_(False, None) is False and and_(None, False) is False

    def test_and_with_null_is_null(self):
        assert and_(True, None) is None

    def test_and_of_two_trues_is_true(self):
        assert and_(True, True) is True

    def test_or_with_true_is_true_even_against_null(self):
        assert or_(True, None) is True and or_(None, True) is True

    def test_or_with_null_is_null(self):
        assert or_(False, None) is None

    def test_or_of_two_falses_is_false(self):
        assert or_(False, False) is False

    def test_not_null_is_null(self):
        assert not_(None) is None

    def test_not_flips_the_others(self):
        assert not_(True) is False and not_(False) is True

    def test_equality_against_null_is_null(self):
        assert equals(1, None) is None

    def test_equality_is_true_for_equals(self):
        assert equals(1, 1) is True

    def test_two_nulls_are_not_distinct(self):
        assert distinct_from(None, None) is False

    def test_a_null_and_a_value_are_distinct(self):
        assert distinct_from(None, 1) is True

    def test_two_equal_values_are_not_distinct(self):
        assert distinct_from(1, 1) is False

    def test_the_two_equalities_disagree_on_nulls(self):
        assert equals(None, None) is None and distinct_from(None, None) is False


class TestSelection:
    def test_projection_keeps_the_named_columns(self):
        assert select(people(), ["id"]).names == ("id",)

    def test_and_reorders(self):
        assert select(people(), ["score", "id"]).names == ("score", "id")

    def test_and_keeps_the_values(self):
        assert select(people(), ["id"]).rows[0] == [1]

    def test_an_unknown_column_is_refused(self):
        with pytest.raises(UnknownColumn):
            select(people(), ["z"])

    def test_a_filter_keeps_true_rows(self):
        assert len(where(people(), lambda row: row["score"] > 15).rows) == 3

    def test_a_filter_drops_null_predicates(self):
        kept = where(people(), lambda row: equals(row["team"], "a"))
        assert [row[0] for row in kept.rows] == [1, 3]

    def test_a_negated_filter_does_not_pick_up_the_nulls(self):
        kept = where(people(), lambda row: not_(equals(row["team"], "a")))
        assert [row[0] for row in kept.rows] == [2]

    def test_a_filter_that_matches_nothing_gives_no_rows(self):
        assert where(people(), lambda _row: False).rows == []

    def test_and_keeps_the_schema(self):
        assert where(people(), lambda _row: False).names == people().names


class TestOrdering:
    def test_a_sort_orders_ascending(self):
        assert order_by(people(), ["score"]).column("score") == [10, 20, 30, 40]

    def test_and_descending(self):
        ordered = order_by(people(), ["score"], [True])
        assert ordered.column("score") == [40, 30, 20, 10]

    def test_nulls_sort_last_by_default(self):
        assert order_by(people(), ["team"]).column("team")[-1] is None

    def test_and_first_when_asked(self):
        ordered = order_by(people(), ["team"], nulls_first=True)
        assert ordered.column("team")[0] is None

    def test_nulls_first_does_not_flip_with_the_direction(self):
        ordered = order_by(people(), ["team"], [True], nulls_first=True)
        assert ordered.column("team")[0] is None

    def test_a_sort_on_two_keys_breaks_ties(self):
        table = Rows(names=("a", "b"), rows=[[1, 2], [1, 1], [0, 9]])
        assert order_by(table, ["a", "b"]).rows == [[0, 9], [1, 1], [1, 2]]

    def test_a_sort_on_strings_orders_them(self):
        assert order_by(teams(), ["city"]).column("city") == ["hull", "leeds", "york"]

    def test_a_sort_with_no_keys_is_refused(self):
        with pytest.raises(ConfigError, match="at least one key"):
            order_by(people(), [])

    def test_mismatched_directions_are_refused(self):
        with pytest.raises(ConfigError, match="directions against"):
            order_by(people(), ["id", "score"], [True])

    def test_a_sort_over_mixed_types_is_refused(self):
        table = Rows(names=("a",), rows=[[1], ["x"]])
        with pytest.raises(TypeMismatch, match="cannot sort mixed types"):
            order_by(table, ["a"])


class TestGrouping:
    def test_a_group_by_collects_keys(self):
        result = group_by(people(), ["team"], [("n", "count_star", "id")])
        assert sorted(str(row[0]) for row in result.rows) == ["None", "a", "b"]

    def test_nulls_form_their_own_group(self):
        result = group_by(people(), ["team"], [("n", "count_star", "id")])
        by_key = {str(row[0]): row[1] for row in result.rows}
        assert by_key["None"] == 1

    def test_count_star_counts_rows(self):
        result = group_by(people(), ["team"], [("n", "count_star", "score")])
        by_key = {str(row[0]): row[1] for row in result.rows}
        assert by_key["a"] == 2

    def test_a_sum_adds_the_group(self):
        result = group_by(people(), ["team"], [("s", "sum", "score")])
        by_key = {str(row[0]): row[1] for row in result.rows}
        assert by_key["a"] == 40

    def test_min_and_max_bracket_it(self):
        result = group_by(people(), ["team"], [("lo", "min", "score"), ("hi", "max", "score")])
        by_key = {str(row[0]): (row[1], row[2]) for row in result.rows}
        assert by_key["a"] == (10, 30)

    def test_a_mean_divides(self):
        result = group_by(people(), ["team"], [("m", "mean", "score")])
        by_key = {str(row[0]): row[1] for row in result.rows}
        assert by_key["a"] == 20.0

    def test_the_output_names_the_aggregates(self):
        result = group_by(people(), ["team"], [("n", "count_star", "id")])
        assert result.names == ("team", "n")

    def test_a_group_by_with_no_keys_gives_one_group(self):
        result = group_by(people(), [], [("n", "count_star", "id")])
        assert len(result.rows) == 1

    def test_count_skips_nulls(self):
        table = Rows(names=("g", "v"), rows=[["x", 1], ["x", None]])
        result = group_by(table, ["g"], [("n", "count", "v")])
        assert result.rows[0][1] == 1

    def test_but_count_star_does_not(self):
        table = Rows(names=("g", "v"), rows=[["x", 1], ["x", None]])
        result = group_by(table, ["g"], [("n", "count_star", "v")])
        assert result.rows[0][1] == 2

    def test_a_sum_over_only_nulls_is_null_not_zero(self):
        table = Rows(names=("g", "v"), rows=[["x", None]])
        result = group_by(table, ["g"], [("s", "sum", "v")])
        assert result.rows[0][1] is None

    def test_a_sum_ignoring_nulls_adds_the_rest(self):
        table = Rows(names=("g", "v"), rows=[["x", 1], ["x", None], ["x", 2]])
        result = group_by(table, ["g"], [("s", "sum", "v")])
        assert result.rows[0][1] == 3

    def test_any_and_all_reduce_booleans(self):
        table = Rows(names=("g", "v"), rows=[["x", True], ["x", False]])
        result = group_by(table, ["g"], [("a", "any", "v"), ("b", "all", "v")])
        assert result.rows[0][1] is True and result.rows[0][2] is False

    def test_an_unknown_aggregate_is_refused(self):
        with pytest.raises(ConfigError, match="is not an aggregate"):
            apply_aggregate("median", people(), people().rows, "score")


class TestJoining:
    def test_an_inner_join_matches(self):
        result = inner_join(people(), teams(), ["team"], ["team"])
        assert len(result.rows) == 3

    def test_and_carries_both_sides(self):
        result = inner_join(people(), teams(), ["team"], ["team"])
        assert result.names == ("id", "team", "score", "team_right", "city")

    def test_a_null_key_never_matches(self):
        result = inner_join(people(), teams(), ["team"], ["team"])
        assert all(row[1] is not None for row in result.rows)

    def test_an_unmatched_right_row_is_dropped(self):
        result = inner_join(people(), teams(), ["team"], ["team"])
        assert "hull" not in [row[-1] for row in result.rows]

    def test_a_left_join_keeps_unmatched_left_rows(self):
        result = left_join(people(), teams(), ["team"], ["team"])
        assert len(result.rows) == 4

    def test_and_fills_the_right_with_nulls(self):
        result = left_join(people(), teams(), ["team"], ["team"])
        unmatched = [row for row in result.rows if row[0] == 4]
        assert unmatched[0][-1] is None

    def test_mismatched_key_counts_are_refused(self):
        with pytest.raises(ConfigError, match="keys against"):
            inner_join(people(), teams(), ["team"], ["team", "city"])

    def test_a_join_with_no_keys_is_refused(self):
        with pytest.raises(ConfigError, match="at least one key"):
            inner_join(people(), teams(), [], [])

    def test_a_semi_join_keeps_each_left_row_once(self):
        result = semi_join(people(), teams(), ["team"], ["team"])
        assert len(result.rows) == 3

    def test_and_keeps_the_left_schema(self):
        result = semi_join(people(), teams(), ["team"], ["team"])
        assert result.names == people().names

    def test_an_anti_join_keeps_the_unmatched(self):
        result = anti_join(people(), teams(), ["team"], ["team"])
        assert [row[0] for row in result.rows] == [4]

    def test_a_null_key_lands_in_the_anti_join(self):
        result = anti_join(people(), teams(), ["team"], ["team"])
        assert result.rows[0][1] is None

    def test_the_two_partition_the_left_side(self):
        semi = semi_join(people(), teams(), ["team"], ["team"])
        anti = anti_join(people(), teams(), ["team"], ["team"])
        assert len(semi.rows) + len(anti.rows) == len(people().rows)


class TestOtherOperators:
    def test_distinct_removes_repeats(self):
        table = Rows(names=("a",), rows=[[1], [1], [2]])
        assert distinct(table).rows == [[1], [2]]

    def test_distinct_treats_two_nulls_as_one(self):
        table = Rows(names=("a",), rows=[[None], [None]])
        assert len(distinct(table).rows) == 1

    def test_limit_takes_the_first_rows(self):
        assert limit(people(), 2).column("id") == [1, 2]

    def test_limit_with_an_offset_skips(self):
        assert limit(people(), 2, offset=1).column("id") == [2, 3]

    def test_a_limit_past_the_end_gives_what_there_is(self):
        assert len(limit(people(), 100).rows) == 4

    def test_a_negative_limit_is_refused(self):
        with pytest.raises(ConfigError, match="not a window"):
            limit(people(), -1)

    def test_union_all_stacks(self):
        assert len(union_all(people(), people()).rows) == 8

    def test_and_keeps_duplicates(self):
        assert union_all(people(), people()).column("id") == [1, 2, 3, 4, 1, 2, 3, 4]

    def test_mismatched_names_are_refused(self):
        with pytest.raises(SchemaError):
            union_all(people(), teams())


class TestAgreement:
    def test_a_table_agrees_with_itself(self):
        assert agree(people(), people()).same

    def test_a_reordered_table_agrees_when_order_is_ignored(self):
        assert agree(people(), order_by(people(), ["score"], [True])).same

    def test_but_not_when_order_matters(self):
        flipped = order_by(people(), ["score"], [True])
        assert not agree(people(), flipped, ordered=True).same

    def test_a_different_row_count_is_reported(self):
        assert not agree(people(), limit(people(), 2)).same

    def test_and_the_difference_names_the_counts(self):
        result = agree(people(), limit(people(), 2))
        assert result.differences[0].kind == "count"

    def test_different_names_are_reported(self):
        assert not agree(people(), teams()).same

    def test_and_reported_first(self):
        assert agree(people(), teams()).differences[0].kind == "names"

    def test_a_value_difference_is_reported(self):
        changed = Rows(names=people().names, rows=[list(row) for row in people().rows])
        changed.rows[0][2] = 99
        assert not agree(people(), changed).same

    def test_floats_compare_within_a_tolerance(self):
        left = Rows(names=("a",), rows=[[1.0]])
        right = Rows(names=("a",), rows=[[1.0 + 1e-12]])
        assert agree(left, right).same

    def test_but_not_beyond_it(self):
        left = Rows(names=("a",), rows=[[1.0]])
        right = Rows(names=("a",), rows=[[1.1]])
        assert not agree(left, right).same

    def test_a_null_matches_only_a_null(self):
        left = Rows(names=("a",), rows=[[None]])
        right = Rows(names=("a",), rows=[[1]])
        assert not agree(left, right).same

    def test_two_nulls_match(self):
        left = Rows(names=("a",), rows=[[None]])
        assert agree(left, left).same

    def test_the_report_is_capped(self):
        left = Rows(names=("a",), rows=[[n] for n in range(20)])
        right = Rows(names=("a",), rows=[[n + 100] for n in range(20)])
        assert len(agree(left, right).differences) <= 5

    def test_an_agreement_serialises(self):
        assert agree(people(), people()).as_dict()["same"]

    def test_a_difference_serialises(self):
        result = agree(people(), teams())
        assert result.differences[0].as_dict()["kind"] == "names"
