from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, UnknownColumn
from cqe.exec import project as projection
from cqe.exec.batch import Batch
from cqe.exec.expr import Arithmetic, Compare, column, literal
from cqe.exec.project import Computed, compute, drop, narrow, project, rename
from cqe.types.schema import BOOLEAN, FLOATING, INTEGER


@pytest.fixture(scope="module")
def batch() -> Batch:
    """A table with four columns of different types."""
    state = np.random.default_rng(43)
    rows = 2000
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 20, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 5, rows)]),
        ]
    )


def test_narrowing_shares_its_arrays():
    assert projection.narrowing_copies_nothing()["the_arrays_are_the_same_objects"]


def test_narrowing_shares_its_columns():
    assert projection.narrowing_copies_nothing()["and_the_columns_are_too"]


def test_narrowing_touches_nothing():
    assert projection.narrowing_costs_nothing_measurable()["it_touched_nothing"]


def test_narrowing_materialises_nothing():
    assert projection.narrowing_costs_nothing_measurable()["and_materialised_nothing"]


def test_computing_touches_every_row():
    assert projection.computing_a_column_is_not_free()["it_touched_the_rows"]


def test_computing_after_a_filter_is_cheaper():
    assert projection.computing_after_a_filter_is_cheaper()["later_is_cheaper"]


def test_narrowing_before_a_filter_is_cheaper():
    assert projection.narrowing_before_a_filter_is_cheaper()["narrowing_first_is_cheaper"]


def test_the_two_move_in_opposite_directions():
    assert projection.the_two_directions_are_opposite()["they_are_opposite"]


def test_every_computed_type_matches_its_declaration():
    assert projection.a_computed_column_gets_the_right_type()["they_all_agree"]


def test_an_integer_sum_stays_integer():
    assert projection.a_computed_column_gets_the_right_type()["an_integer_sum_stays_integer"]


def test_a_mixed_sum_becomes_floating():
    assert projection.a_computed_column_gets_the_right_type()["a_mixed_sum_becomes_floating"]


def test_a_comparison_is_boolean():
    assert projection.a_computed_column_gets_the_right_type()["a_comparison_is_boolean"]


def test_a_computed_column_cannot_read_another():
    assert projection.a_computed_column_cannot_read_another()["it_was_refused"]


def test_two_independent_computed_columns_work():
    assert projection.a_computed_column_cannot_read_another()["and_two_independent_ones_work"]


def test_a_computed_column_cannot_replace_one():
    assert projection.a_computed_column_cannot_replace_one()["it_was_refused"]


def test_renaming_first_makes_it_possible():
    assert projection.a_computed_column_cannot_replace_one()["and_renaming_first_works"]


def test_projecting_computes_before_narrowing():
    assert projection.projecting_computes_before_it_narrows()["it_is_only_the_computed_one"]


def test_the_computed_values_are_right():
    assert projection.projecting_computes_before_it_narrows()["and_the_values_are_right"]


def test_renaming_keeps_the_array():
    assert projection.renaming_keeps_the_arrays()["the_array_is_the_same_object"]


def test_renaming_leaves_the_others_alone():
    assert projection.renaming_keeps_the_arrays()["the_others_are_untouched"]


def test_dropping_and_keeping_agree():
    assert projection.dropping_is_narrowing_backwards()["they_agree"]


def test_a_missing_column_is_refused():
    assert projection.a_missing_column_is_refused()


def test_a_repeated_column_is_refused():
    assert projection.a_repeated_column_is_refused()


def test_dropping_everything_is_refused():
    assert projection.dropping_everything_is_refused()


def test_dropping_a_missing_column_is_refused():
    assert projection.dropping_a_missing_column_is_refused()


def test_a_colliding_rename_is_refused():
    assert projection.a_rename_that_collides_is_refused()


def test_renaming_a_missing_column_is_refused():
    assert projection.renaming_a_missing_column_is_refused()


def test_repeated_computed_names_are_refused():
    assert projection.repeated_computed_names_are_refused()


def test_only_the_computing_projection_costs_anything():
    table = projection.compare_the_projections()
    costs = {one["projection"]: one["values_touched"] for one in table}
    assert costs["compute"] > 0 and all(costs[one] == 0 for one in ("narrow", "drop", "rename"))


def test_the_summary_says_narrowing_is_free():
    assert projection.summarise()["narrowing_is_free"]


def test_narrowing_keeps_the_rows(batch):
    assert narrow(batch, ["id"]).rows == batch.rows


def test_narrowing_keeps_the_order_asked_for(batch):
    assert list(narrow(batch, ["amount", "id"]).schema.names) == ["amount", "id"]


def test_narrowing_to_every_column_is_a_reorder(batch):
    names = ["region", "amount", "shop", "id"]
    assert list(narrow(batch, names).schema.names) == names


def test_narrowing_to_one_column_gives_one(batch):
    assert narrow(batch, ["id"]).width == 1


def test_a_batch_of_no_columns_is_refused():
    with pytest.raises(ConfigError):
        Batch.from_columns([])


def test_narrowing_a_missing_column_names_it(batch):
    with pytest.raises(UnknownColumn):
        narrow(batch, ["nothing"])


def test_dropping_one_column_leaves_the_rest(batch):
    assert drop(batch, ["region"]).width == batch.width - 1


def test_dropping_two_columns_leaves_two(batch):
    assert drop(batch, ["region", "shop"]).width == 2


def test_renaming_changes_one_name(batch):
    renamed = rename(batch, {"id": "key"})
    assert "key" in renamed.schema and "id" not in renamed.schema


def test_renaming_two_columns_at_once(batch):
    renamed = rename(batch, {"id": "key", "amount": "value"})
    assert "key" in renamed.schema and "value" in renamed.schema


def test_renaming_to_the_same_name_is_a_no_operation(batch):
    assert list(rename(batch, {"id": "id"}).schema.names) == list(batch.schema.names)


def test_computing_adds_one_column(batch):
    made = Computed("doubled", Arithmetic("+", column("amount"), column("amount")))
    assert compute(batch, [made]).width == batch.width + 1


def test_computing_adds_two_columns(batch):
    first = Computed("doubled", Arithmetic("+", column("amount"), column("amount")))
    second = Computed("bumped", Arithmetic("+", column("id"), literal(1)))
    assert compute(batch, [first, second]).width == batch.width + 2


def test_computing_nothing_returns_the_batch(batch):
    assert compute(batch, []) is batch


def test_a_computed_sum_holds_the_right_values(batch):
    made = Computed("doubled", Arithmetic("+", column("amount"), column("amount")))
    produced = compute(batch, [made]).column("doubled").values
    assert np.allclose(produced, batch.column("amount").values * 2)


def test_a_computed_product_holds_the_right_values(batch):
    made = Computed("scaled", Arithmetic("*", column("id"), literal(3)))
    produced = compute(batch, [made]).column("scaled").values
    assert np.array_equal(produced, batch.column("id").values * 3)


def test_a_computed_comparison_is_boolean(batch):
    made = Computed("big", Compare(">", column("amount"), literal(100.0)))
    assert compute(batch, [made]).column("big").field.logical == BOOLEAN


def test_a_computed_integer_expression_stays_integer(batch):
    made = Computed("v", Arithmetic("+", column("id"), column("shop")))
    assert compute(batch, [made]).column("v").field.logical == INTEGER


def test_a_computed_mixed_expression_is_floating(batch):
    made = Computed("v", Arithmetic("+", column("id"), column("amount")))
    assert compute(batch, [made]).column("v").field.logical == FLOATING


def test_a_computed_column_describes_itself():
    made = Computed("v", Arithmetic("+", column("a"), column("b")))
    assert made.describe().endswith("as v")


def test_a_computed_column_summarises():
    made = Computed("v", Arithmetic("+", column("a"), column("b")))
    assert made.as_dict()["columns"] == ["a", "b"]


def test_a_computed_column_reports_its_type(batch):
    made = Computed("v", Arithmetic("+", column("amount"), column("amount")))
    assert made.type_of(batch.schema) == FLOATING


def test_projecting_with_no_names_keeps_everything(batch):
    made = Computed("v", Arithmetic("+", column("id"), literal(1)))
    assert project(batch, columns=[made]).width == batch.width + 1


def test_projecting_with_no_computed_columns_is_narrowing(batch):
    assert list(project(batch, names=["id"]).schema.names) == ["id"]


def test_projecting_neither_returns_the_batch(batch):
    assert project(batch) is batch


def test_projecting_keeps_a_source_column_when_asked(batch):
    made = Computed("v", Arithmetic("+", column("amount"), literal(1.0)))
    produced = project(batch, names=["amount", "v"], columns=[made])
    assert list(produced.schema.names) == ["amount", "v"]


def test_the_meter_counts_one_batch_for_a_narrowing(batch):
    meter = Meter()
    narrow(batch, ["id"], meter=meter)
    assert meter.batches == 1


def test_the_meter_counts_rows_for_a_computation(batch):
    meter = Meter()
    made = Computed("v", Arithmetic("+", column("amount"), column("amount")))
    compute(batch, [made], meter=meter)
    assert meter.rows_materialised == batch.rows


def test_narrowing_an_empty_batch_works(batch):
    empty = batch.slice(0, 0)
    assert narrow(empty, ["id"]).rows == 0


def test_computing_over_an_empty_batch_works(batch):
    empty = batch.slice(0, 0)
    made = Computed("v", Arithmetic("+", column("amount"), literal(1.0)))
    assert compute(empty, [made]).width == batch.width + 1
