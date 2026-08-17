from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.errors import SchemaError
from cqe.exec.batch import Batch
from cqe.exec.join import outer
from cqe.exec.join.hash import hash_join
from cqe.exec.join.outer import anti, full_outer, left_outer, semi
from cqe.verify.reference import Rows, agree, anti_join, left_join, semi_join


@pytest.fixture(scope="module")
def tables() -> tuple[Batch, Batch]:
    """A fact table and a dimension whose keys half overlap."""
    state = np.random.default_rng(67)
    rows = 800
    left = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 100, rows)),
            floating_column("amount", state.normal(50, 10, rows)),
        ]
    )
    right = Batch.from_columns(
        [
            integer_column("shop", np.arange(50)),
            string_column("region", [f"region{one % 4}" for one in range(50)]),
        ]
    )
    return left, right


def test_a_left_join_keeps_every_left_row():
    assert outer.a_left_join_keeps_every_left_row()["it_kept_every_left_row"]


def test_an_inner_join_does_not():
    assert outer.a_left_join_keeps_every_left_row()["the_inner_join_did_not"]


def test_a_left_join_agrees_with_the_reference():
    assert outer.a_left_join_keeps_every_left_row()["it_agrees_with_the_reference"]


def test_the_right_columns_become_nullable():
    assert outer.the_right_columns_become_nullable()["it_is_now"]


def test_the_left_columns_do_not():
    assert outer.the_right_columns_become_nullable()["the_left_columns_are_untouched"]


def test_a_semi_join_is_bounded_by_the_left_table():
    assert outer.a_semi_join_keeps_a_row_once()["the_semi_join_is_bounded"]


def test_an_inner_join_is_not_bounded():
    assert outer.a_semi_join_keeps_a_row_once()["the_inner_join_is_not"]


def test_a_semi_join_agrees_with_the_reference():
    assert outer.a_semi_join_keeps_a_row_once()["it_agrees_with_the_reference"]


def test_a_semi_join_keeps_the_left_schema():
    assert outer.a_semi_join_keeps_the_left_schema()["they_are_the_same"]


def test_semi_and_anti_partition_the_table():
    assert outer.an_anti_join_is_the_complement()["they_sum_to_the_table"]


def test_an_anti_join_agrees_with_the_reference():
    assert outer.an_anti_join_is_the_complement()["the_anti_join_agrees"]


def test_a_null_key_is_kept_by_an_anti_join():
    assert outer.a_null_key_is_kept_by_an_anti_join()["the_anti_join_kept_the_nulls"]


def test_a_null_key_is_dropped_by_a_semi_join():
    assert outer.a_null_key_is_kept_by_an_anti_join()["and_the_semi_join_kept_none"]


def test_the_null_handling_agrees_with_the_reference():
    assert outer.a_null_key_is_kept_by_an_anti_join()["and_it_agrees_with_the_reference"]


def test_a_null_key_gets_nulls_from_a_left_join():
    assert outer.a_null_key_gets_nulls_from_a_left_join()["the_right_side_is_nulled"]


def test_the_unmatched_counts_agree_across_joins():
    assert outer.a_null_key_gets_nulls_from_a_left_join()["they_agree"]


def test_the_fanout_equation_holds():
    assert outer.the_fanout_is_the_same_as_an_inner_join()["the_equation_holds"]


def test_the_semi_join_grows_with_the_overlap():
    assert outer.an_overlap_sweep_moves_every_count()["the_semi_join_grows"]


def test_the_anti_join_shrinks_with_the_overlap():
    assert outer.an_overlap_sweep_moves_every_count()["and_the_anti_join_shrinks"]


def test_they_always_sum_to_the_table():
    assert outer.an_overlap_sweep_moves_every_count()["they_always_sum"]


def test_a_full_join_keeps_every_left_row():
    assert outer.a_full_join_keeps_both_sides()["it_kept_every_left_row"]


def test_a_full_join_keeps_the_right_orphans():
    assert outer.a_full_join_keeps_both_sides()["right_orphans"] > 0


def test_the_left_and_semi_joins_agree_on_matches():
    assert outer.the_four_joins_are_one_implementation()["the_left_and_semi_agree"]


def test_the_semi_and_anti_joins_partition():
    assert outer.the_four_joins_are_one_implementation()["the_semi_and_anti_partition"]


def test_an_empty_right_side_gives_the_left_table():
    assert outer.an_empty_right_side_nulls_everything()["it_is_the_left_table"]


def test_an_empty_right_side_nulls_every_value():
    assert outer.an_empty_right_side_nulls_everything()["every_right_value_is_null"]


def test_an_empty_right_side_empties_the_semi_join():
    assert outer.an_empty_right_side_nulls_everything()["the_semi_join_is_empty"]


def test_a_join_with_no_keys_is_refused():
    assert outer.a_join_with_no_keys_is_refused()


def test_the_kind_table_covers_four_joins():
    assert len(outer.compare_the_kinds()) == 4


def test_only_the_left_join_widens_the_schema():
    table = {one["kind"]: one["columns"] for one in outer.compare_the_kinds()}
    assert table["left"] > table["semi"] and table["semi"] == table["anti"]


def test_the_summary_says_they_partition():
    assert outer.summarise()["they_partition"]


def test_a_left_join_returns_at_least_the_left_rows(tables):
    left, right = tables
    assert left_outer(left, right, ["shop"], ["shop"]).rows >= left.rows


def test_a_left_join_widens_the_schema(tables):
    left, right = tables
    produced = left_outer(left, right, ["shop"], ["shop"]).batch
    assert produced.width == left.width + right.width


def test_a_left_join_suffixes_a_repeated_name(tables):
    left, right = tables
    produced = left_outer(left, right, ["shop"], ["shop"]).batch
    assert "shop_right" in produced.schema


def test_a_left_join_matches_the_reference(tables):
    left, right = tables
    produced = left_outer(left, right, ["shop"], ["shop"]).batch
    expected = left_join(Rows.of(left), Rows.of(right), ["shop"], ["shop"])
    assert agree(Rows.of(produced), expected)


def test_a_semi_join_matches_the_reference(tables):
    left, right = tables
    produced = semi(left, right, ["shop"], ["shop"]).batch
    expected = semi_join(Rows.of(left), Rows.of(right), ["shop"], ["shop"])
    assert agree(Rows.of(produced), expected)


def test_an_anti_join_matches_the_reference(tables):
    left, right = tables
    produced = anti(left, right, ["shop"], ["shop"]).batch
    expected = anti_join(Rows.of(left), Rows.of(right), ["shop"], ["shop"])
    assert agree(Rows.of(produced), expected)


def test_a_semi_join_is_a_subset_of_the_left_table(tables):
    left, right = tables
    produced = semi(left, right, ["shop"], ["shop"]).batch
    assert set(produced.column("id").to_list()) <= set(left.column("id").to_list())


def test_an_anti_join_is_a_subset_of_the_left_table(tables):
    left, right = tables
    produced = anti(left, right, ["shop"], ["shop"]).batch
    assert set(produced.column("id").to_list()) <= set(left.column("id").to_list())


def test_the_two_subsets_are_disjoint(tables):
    left, right = tables
    kept = set(semi(left, right, ["shop"], ["shop"]).batch.column("id").to_list())
    dropped = set(anti(left, right, ["shop"], ["shop"]).batch.column("id").to_list())
    assert not (kept & dropped)


def test_the_two_subsets_cover_the_table(tables):
    left, right = tables
    kept = set(semi(left, right, ["shop"], ["shop"]).batch.column("id").to_list())
    dropped = set(anti(left, right, ["shop"], ["shop"]).batch.column("id").to_list())
    assert kept | dropped == set(left.column("id").to_list())


def test_a_left_join_of_a_full_overlap_is_an_inner_join(tables):
    left, _ = tables
    right = Batch.from_columns(
        [
            integer_column("shop", np.arange(100)),
            string_column("region", [f"region{one % 4}" for one in range(100)]),
        ]
    )
    inner = hash_join(left, right, ["shop"], ["shop"]).batch
    produced = left_outer(left, right, ["shop"], ["shop"])
    assert produced.rows == inner.rows and produced.unmatched == 0


def test_an_anti_join_of_a_full_overlap_is_empty(tables):
    left, _ = tables
    right = Batch.from_columns(
        [
            integer_column("shop", np.arange(100)),
            string_column("region", [f"region{one % 4}" for one in range(100)]),
        ]
    )
    assert anti(left, right, ["shop"], ["shop"]).rows == 0


def test_a_join_reports_its_fanout(tables):
    left, right = tables
    assert left_outer(left, right, ["shop"], ["shop"]).fanout >= 1


def test_a_join_summarises(tables):
    left, right = tables
    assert left_outer(left, right, ["shop"], ["shop"]).as_dict()["kind"] == "left"


def test_a_full_join_returns_more_than_a_left_join(tables):
    left, _right = tables
    smaller = Batch.from_columns(
        [
            integer_column("shop", np.arange(200)),
            string_column("region", [f"region{one % 4}" for one in range(200)]),
        ]
    )
    kept = left_outer(left, smaller, ["shop"], ["shop"])
    whole = full_outer(left, smaller, ["shop"], ["shop"])
    assert whole.rows > kept.rows


def test_a_null_key_on_the_right_matches_nothing(tables):
    left, _ = tables
    values = np.arange(50)
    made = integer_column("shop", values)
    valid = np.zeros(50, dtype=bool)
    right = Batch.from_columns(
        [
            Column(field=made.field, values=values, valid=valid),
            string_column("region", [f"region{one % 4}" for one in range(50)]),
        ]
    )
    assert semi(left, right, ["shop"], ["shop"]).rows == 0


def test_a_join_on_two_keys_works(tables):
    left, _ = tables
    right = Batch.from_columns(
        [
            integer_column("shop", np.arange(50)),
            integer_column("id", np.arange(50)),
            string_column("region", [f"region{one % 4}" for one in range(50)]),
        ]
    )
    produced = semi(left, right, ["shop", "id"], ["shop", "id"])
    assert produced.rows <= left.rows


def test_a_join_with_no_keys_raises(tables):
    left, right = tables
    with pytest.raises(SchemaError):
        semi(left, right, [], [])
