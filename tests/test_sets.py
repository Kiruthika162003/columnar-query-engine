from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.errors import SchemaError
from cqe.exec import sets
from cqe.exec.batch import Batch
from cqe.exec.sets import difference, distinct, intersect, union, union_all
from cqe.verify.reference import Rows, agree
from cqe.verify.reference import distinct as reference_distinct


@pytest.fixture(scope="module")
def repeated() -> Batch:
    """A table of two thousand rows holding three hundred distinct ones."""
    state = np.random.default_rng(263)
    picks = state.integers(0, 300, 2000)
    return Batch.from_columns(
        [
            integer_column("shop", picks),
            floating_column("amount", (picks % 50).astype(np.float64)),
            string_column("region", [f"region{one % 4}" for one in picks]),
        ]
    )


def test_distinct_agrees_with_the_reference():
    assert sets.distinct_removes_the_repeats()["it_agrees_with_the_reference"]


def test_distinct_removes_the_repeats():
    assert sets.distinct_removes_the_repeats()["it_removed_the_repeats"]


def test_distinct_is_sorted():
    assert sets.distinct_is_sorted()["it_is_sorted"]


def test_a_shuffled_copy_gives_the_same_order():
    assert sets.distinct_is_sorted()["and_the_shuffled_copy_gives_the_same_order"]


def test_two_nulls_collapse_into_one_row():
    assert sets.two_nulls_are_the_same_row()["the_nulls_collapsed"]


def test_the_nulls_do_not_all_vanish():
    assert sets.two_nulls_are_the_same_row()["and_they_did_not_all_vanish"]


def test_the_null_rule_agrees_with_the_reference():
    assert sets.two_nulls_are_the_same_row()["it_agrees_with_the_reference"]


def test_a_join_drops_the_nulls_that_distinct_keeps():
    assert sets.a_null_matches_nothing_in_a_join()["the_nulls_were_dropped"]


def test_the_two_null_rules_differ():
    assert sets.a_null_matches_nothing_in_a_join()["the_two_rules_differ"]


def test_union_all_is_the_sum():
    assert sets.union_all_keeps_everything()["it_is_the_sum"]


def test_a_union_is_far_smaller_than_a_union_all():
    assert sets.union_all_keeps_everything()["and_the_union_is_far_smaller"]


def test_union_all_agrees_with_the_reference():
    assert sets.union_all_keeps_everything()["it_agrees_with_the_reference"]


def test_deduplicating_each_side_first_is_wrong():
    assert sets.a_union_is_a_distinct_over_a_concatenation()["the_naive_way_is_too_large"]


def test_the_difference_is_the_rows_in_both():
    assert sets.a_union_is_a_distinct_over_a_concatenation()["which_are_the_rows_in_both"]


def test_an_intersection_matches_a_python_set():
    assert sets.an_intersection_keeps_what_is_in_both()["it_matches_a_python_set"]


def test_an_intersection_is_smaller_than_either_side():
    assert sets.an_intersection_keeps_what_is_in_both()["it_is_smaller_than_either_side"]


def test_a_difference_matches_a_python_set():
    assert sets.a_difference_keeps_what_is_only_on_the_left()["it_matches_a_python_set"]


def test_the_three_operations_sum_to_the_union():
    assert sets.the_three_set_operations_partition()["they_sum_to_the_union"]


def test_a_union_with_itself_is_a_distinct():
    assert sets.a_set_operation_with_itself_is_a_distinct()["the_union_is_the_distinct"]


def test_an_intersection_with_itself_is_a_distinct():
    assert sets.a_set_operation_with_itself_is_a_distinct()["and_so_is_the_intersection"]


def test_a_difference_with_itself_is_empty():
    assert sets.a_set_operation_with_itself_is_a_distinct()["and_the_difference_is_empty"]


def test_a_union_with_nothing_is_a_distinct():
    assert sets.a_set_operation_with_nothing_is_the_table()["the_union_is_the_distinct"]


def test_an_intersection_with_nothing_is_empty():
    assert sets.a_set_operation_with_nothing_is_the_table()["the_intersection_is_empty"]


def test_a_difference_from_nothing_is_a_distinct():
    assert sets.a_set_operation_with_nothing_is_the_table()[
        "and_the_difference_is_the_distinct"
    ]


def test_the_saving_falls_with_the_distinct_count():
    assert sets.the_duplicate_share_decides_the_saving()["the_saving_falls"]


def test_the_saving_is_large_when_everything_repeats():
    assert sets.the_duplicate_share_decides_the_saving()["at_the_most_repeated"] > 0.9


def test_the_saving_is_smaller_when_little_repeats():
    assert sets.the_duplicate_share_decides_the_saving()["at_the_least"] < 0.6


def test_a_mismatched_schema_is_refused():
    assert sets.a_mismatched_schema_is_refused()["it_was_refused"]


def test_the_schema_refusal_names_both():
    assert sets.a_mismatched_schema_is_refused()["it_names_both"]


def test_a_mismatched_type_is_refused():
    assert sets.a_mismatched_type_is_refused()


def test_an_empty_table_distincts_to_nothing():
    assert sets.an_empty_table_distincts_to_nothing()["it_is_empty"]


def test_an_empty_distinct_keeps_its_schema():
    assert sets.an_empty_table_distincts_to_nothing()["and_it_kept_its_schema"]


def test_every_operation_agrees_with_a_python_set():
    assert sets.every_operation_agrees_with_a_python_set()["they_all_agree"]


def test_the_operation_table_covers_five():
    assert len(sets.compare_the_operations()) == 5


def test_only_union_all_keeps_every_row():
    table = {one["kind"]: one["rows"] for one in sets.compare_the_operations()}
    assert table["union all"] > table["union"]


def test_the_summary_says_python_sets_agree():
    assert sets.summarise()["python_sets_agree"]


def test_distinct_returns_fewer_rows(repeated):
    assert distinct(repeated).rows < repeated.rows


def test_distinct_returns_the_distinct_count(repeated):
    expected = len({tuple(one) for one in Rows.of(repeated).rows})
    assert distinct(repeated).rows == expected


def test_distinct_counts_its_duplicates(repeated):
    made = distinct(repeated)
    assert made.duplicates == repeated.rows - made.rows


def test_distinct_keeps_the_schema(repeated):
    assert list(distinct(repeated).batch.schema.names) == list(repeated.schema.names)


def test_distinct_matches_the_reference(repeated):
    assert agree(Rows.of(distinct(repeated).batch), reference_distinct(Rows.of(repeated)))


def test_distinct_of_a_distinct_is_itself(repeated):
    once = distinct(repeated).batch
    assert distinct(once).rows == once.rows


def test_a_produced_result_summarises(repeated):
    assert distinct(repeated).as_dict()["kind"] == "distinct"


def test_a_produced_result_reports_its_share(repeated):
    assert 0 < distinct(repeated).share < 1


def test_union_all_concatenates(repeated):
    assert union_all(repeated, repeated).rows == repeated.rows * 2


def test_a_union_of_a_table_with_itself_is_its_distinct(repeated):
    assert union(repeated, repeated).rows == distinct(repeated).rows


def test_an_intersection_of_a_table_with_itself_is_its_distinct(repeated):
    assert intersect(repeated, repeated).rows == distinct(repeated).rows


def test_a_difference_of_a_table_from_itself_is_empty(repeated):
    assert difference(repeated, repeated).rows == 0


def test_a_union_is_commutative(repeated):
    other = repeated.slice(0, 500)
    assert union(repeated, other).rows == union(other, repeated).rows


def test_an_intersection_is_commutative(repeated):
    other = repeated.slice(0, 500)
    assert intersect(repeated, other).rows == intersect(other, repeated).rows


def test_a_difference_is_not_commutative(repeated):
    other = repeated.slice(0, 200)
    assert difference(repeated, other).rows != difference(other, repeated).rows


def test_a_union_is_at_least_each_side(repeated):
    other = repeated.slice(0, 500)
    made = union(repeated, other).rows
    assert made >= distinct(repeated).rows and made >= distinct(other).rows


def test_an_intersection_is_at_most_each_side(repeated):
    other = repeated.slice(0, 500)
    made = intersect(repeated, other).rows
    assert made <= distinct(repeated).rows and made <= distinct(other).rows


def test_a_mismatched_column_list_is_refused(repeated):
    other = Batch.from_columns([integer_column("shop", [1, 2, 3])])
    with pytest.raises(SchemaError):
        union(repeated, other)


def test_a_mismatched_column_order_is_refused(repeated):
    other = Batch.from_columns(
        [
            floating_column("amount", [1.0]),
            integer_column("shop", [1]),
            string_column("region", ["region0"]),
        ]
    )
    with pytest.raises(SchemaError):
        union(repeated, other)


def test_two_null_rows_are_one_distinct_row():
    values = np.array([1, 1, 2], dtype=np.int64)
    made = integer_column("v", values)
    valid = np.array([False, False, True])
    batch = Batch.from_columns([Column(field=made.field, values=values, valid=valid)])
    assert distinct(batch).rows == 2


def test_a_null_row_and_a_zero_row_are_different():
    values = np.array([0, 0], dtype=np.int64)
    made = integer_column("v", values)
    valid = np.array([False, True])
    batch = Batch.from_columns([Column(field=made.field, values=values, valid=valid)])
    assert distinct(batch).rows == 2


def test_a_null_row_survives_a_union():
    values = np.array([1, 2], dtype=np.int64)
    made = integer_column("v", values)
    valid = np.array([False, True])
    batch = Batch.from_columns([Column(field=made.field, values=values, valid=valid)])
    assert union(batch, batch).rows == 2


def test_a_null_row_intersects_with_itself():
    values = np.array([1], dtype=np.int64)
    made = integer_column("v", values)
    valid = np.array([False])
    batch = Batch.from_columns([Column(field=made.field, values=values, valid=valid)])
    assert intersect(batch, batch).rows == 1


def test_an_empty_intersection_keeps_the_schema(repeated):
    empty = Batch.empty(repeated.schema)
    assert list(intersect(repeated, empty).batch.schema.names) == list(repeated.schema.names)
