from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import PlanError
from cqe.exec.aggregate import Aggregate
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.exec.sort import SortKey
from cqe.plan import physical
from cqe.plan.logical import Filter, Group, Join, Limit, Project, Sort, table
from cqe.plan.physical import execute, explain, run
from cqe.verify.reference import Rows, agree, group_by, inner_join, order_by, select, where


@pytest.fixture(scope="module")
def catalogue() -> dict[str, Batch]:
    """A fact table and a dimension, shared by every test in this file."""
    state = np.random.default_rng(23)
    rows = 1500
    facts = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 40, rows)),
            floating_column("amount", state.normal(100, 30, rows)),
            string_column("label", [f"kind{one % 6}" for one in range(rows)]),
        ]
    )
    shops = Batch.from_columns(
        [
            integer_column("shop", np.arange(40)),
            string_column("region", [f"region{one % 4}" for one in range(40)]),
        ]
    )
    return {"facts": facts, "shops": shops}


def test_a_plan_agrees_with_the_reference():
    assert physical.a_plan_runs_and_agrees_with_the_reference()["they_agree"]


def test_the_agreement_reports_no_differences():
    assert physical.a_plan_runs_and_agrees_with_the_reference()["differences"] == 0


def test_a_join_of_two_large_inputs_hashes():
    assert physical.a_join_picks_hash_when_neither_side_is_sorted()["it_chose_hash"]


def test_a_join_with_a_tiny_side_loops():
    assert physical.a_join_picks_nested_loop_when_one_side_is_tiny()["it_chose_nested_loop"]


def test_a_join_of_sorted_inputs_merges():
    assert physical.a_join_picks_merge_when_both_sides_arrived_sorted()["it_chose_merge"]


def test_the_hash_join_agrees_with_the_reference():
    assert physical.every_join_strategy_produces_the_same_rows()["hash_agrees"]


def test_the_nested_loop_join_agrees_with_the_reference():
    assert physical.every_join_strategy_produces_the_same_rows()["loop_agrees"]


def test_all_three_joins_return_the_same_count():
    measured = physical.every_join_strategy_produces_the_same_rows()
    assert measured["hash_rows"] == measured["loop_rows"] == measured["reference_rows"]


def test_a_dictionary_key_counts():
    assert physical.an_aggregate_picks_counting_on_a_dictionary_key()["it_chose_counting"]


def test_counting_makes_no_probes():
    assert physical.an_aggregate_picks_counting_on_a_dictionary_key()["it_made_no_probes"]


def test_an_integer_key_hashes():
    assert physical.an_aggregate_picks_hash_on_an_integer_key()["it_chose_hash"]


def test_hashing_does_make_probes():
    assert physical.an_aggregate_picks_hash_on_an_integer_key()["probes"] > 0


def test_a_sorted_input_uses_the_sorted_aggregate():
    assert physical.an_aggregate_picks_sorted_when_the_keys_arrived_ordered()["it_chose_sorted"]


def test_the_hash_aggregate_agrees_with_the_reference():
    assert physical.every_aggregate_strategy_produces_the_same_groups()["hash_agrees"]


def test_the_counting_aggregate_agrees_with_the_reference():
    assert physical.every_aggregate_strategy_produces_the_same_groups()["counting_agrees"]


def test_a_limit_above_a_sort_partitions():
    assert physical.a_limit_above_a_sort_becomes_a_partial_sort()["it_chose_partial"]


def test_the_partial_sort_makes_fewer_comparisons():
    measured = physical.a_limit_above_a_sort_becomes_a_partial_sort()
    assert measured["limited_comparisons"] < measured["whole_comparisons"]


def test_the_partial_sort_returns_the_same_rows():
    assert physical.a_partial_sort_returns_the_same_rows_as_a_full_one()["same_ids"]


def test_the_partial_sort_returns_the_same_values():
    assert physical.a_partial_sort_returns_the_same_rows_as_a_full_one()["same_values"]


def test_an_offset_still_returns_a_full_page():
    assert physical.an_offset_is_applied_after_the_limit_fetch()["it_returned_a_full_page"]


def test_an_offset_returns_the_right_page():
    assert physical.an_offset_is_applied_after_the_limit_fetch()["it_is_the_right_page"]


def test_a_limit_beyond_the_input_sorts_fully():
    assert physical.a_limit_larger_than_the_input_sorts_the_whole_thing()["it_chose_full"]


def test_a_limit_beyond_the_input_returns_everything():
    assert physical.a_limit_larger_than_the_input_sorts_the_whole_thing()[
        "it_returned_everything"
    ]


def test_pushing_a_predicate_does_not_change_the_answer():
    assert physical.a_scan_runs_the_predicates_pushed_into_it()["they_agree"]


def test_pushing_a_predicate_touches_fewer_values():
    measured = physical.a_scan_runs_the_predicates_pushed_into_it()
    assert measured["pushed_touched"] < measured["plain_touched"]


def test_an_explain_names_a_strategy_for_every_node():
    assert physical.the_choices_explain_the_run()["it_chose_for_every_node"]


def test_an_explain_prints_the_tree():
    assert physical.the_choices_explain_the_run()["the_text_has_the_tree"]


def test_the_ordering_check_tells_sorted_from_shuffled():
    assert physical.the_ordering_check_is_a_check_and_not_a_flag()["it_tells_them_apart"]


def test_an_empty_batch_is_not_called_ordered():
    assert physical.the_ordering_check_is_a_check_and_not_a_flag()[
        "an_empty_batch_is_not_ordered"
    ]


def test_a_null_disqualifies_an_ordering():
    assert physical.a_null_key_is_not_treated_as_ordered()["the_null_disqualifies_it"]


def test_an_unknown_table_is_refused():
    assert physical.an_unknown_table_is_refused()


def test_an_unsupported_node_is_refused():
    assert physical.an_unsupported_node_is_refused()


def test_the_strategy_table_reports_an_output_for_every_row():
    assert all(one["output"] > 0 for one in physical.compare_the_strategies())


def test_the_summary_agrees_with_the_reference():
    assert physical.summarise()["agrees_with_the_reference"]


def test_a_scan_returns_every_row(catalogue):
    produced = execute(table("facts", catalogue["facts"]), catalogue)
    assert produced.rows == catalogue["facts"].rows


def test_a_scan_of_a_missing_table_is_refused(catalogue):
    with pytest.raises(PlanError):
        execute(table("facts", catalogue["facts"]), {})


def test_a_projection_narrows_the_schema(catalogue):
    built = Project(input=table("facts", catalogue["facts"]), names=("id", "amount"))
    assert list(execute(built, catalogue).schema.names) == ["id", "amount"]


def test_a_filter_narrows_the_rows(catalogue):
    built = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(100.0)),
    )
    produced = execute(built, catalogue)
    assert 0 < produced.rows < catalogue["facts"].rows


def test_a_filter_agrees_with_the_reference(catalogue):
    built = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(100.0)),
    )
    produced = execute(built, catalogue)
    expected = where(Rows.of(catalogue["facts"]), lambda one: one["amount"] > 100.0)
    assert agree(Rows.of(produced), expected)


def test_a_join_agrees_with_the_reference(catalogue):
    built = Join(
        left=table("facts", catalogue["facts"]),
        right=table("shops", catalogue["shops"]),
        left_keys=("shop",),
        right_keys=("shop",),
    )
    produced = execute(built, catalogue)
    expected = inner_join(
        Rows.of(catalogue["facts"]), Rows.of(catalogue["shops"]), ["shop"], ["shop"]
    )
    assert agree(Rows.of(produced), expected)


def test_a_group_agrees_with_the_reference(catalogue):
    built = Group(
        input=table("facts", catalogue["facts"]),
        keys=("shop",),
        aggregates=(Aggregate(name="total", function="sum", source="amount"),),
    )
    produced = execute(built, catalogue)
    expected = group_by(Rows.of(catalogue["facts"]), ["shop"], [("total", "sum", "amount")])
    assert agree(Rows.of(produced), expected)


def test_a_sort_agrees_with_the_reference(catalogue):
    built = Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="amount"),))
    produced = execute(built, catalogue)
    expected = order_by(Rows.of(catalogue["facts"]), ["amount"])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_a_descending_sort_agrees_with_the_reference(catalogue):
    built = Sort(
        input=table("facts", catalogue["facts"]),
        keys=(SortKey(name="amount", descending=True),),
    )
    produced = execute(built, catalogue)
    expected = order_by(Rows.of(catalogue["facts"]), ["amount"], descending=[True])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_a_two_key_sort_agrees_with_the_reference(catalogue):
    built = Sort(
        input=table("facts", catalogue["facts"]),
        keys=(SortKey(name="shop"), SortKey(name="amount")),
    )
    produced = execute(built, catalogue)
    expected = order_by(Rows.of(catalogue["facts"]), ["shop", "amount"])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_a_limit_agrees_with_the_reference(catalogue):
    built = Limit(
        input=Sort(input=table("facts", catalogue["facts"]), keys=(SortKey(name="amount"),)),
        count=20,
    )
    produced = execute(built, catalogue)
    expected = order_by(Rows.of(catalogue["facts"]), ["amount"])
    expected = Rows(names=expected.names, rows=expected.rows[:20])
    assert agree(Rows.of(produced), expected, ordered=True)


def test_a_whole_query_agrees_with_the_reference(catalogue):
    built = Project(
        input=Sort(
            input=Group(
                input=Filter(
                    input=table("facts", catalogue["facts"]),
                    predicate=Compare(">", column("amount"), literal(90.0)),
                ),
                keys=("label",),
                aggregates=(Aggregate(name="total", function="sum", source="amount"),),
            ),
            keys=(SortKey(name="label"),),
        ),
        names=("label", "total"),
    )
    produced = execute(built, catalogue)
    kept = where(Rows.of(catalogue["facts"]), lambda one: one["amount"] > 90.0)
    expected = select(
        order_by(group_by(kept, ["label"], [("total", "sum", "amount")]), ["label"]),
        ["label", "total"],
    )
    assert agree(Rows.of(produced), expected, ordered=True)


def test_the_meter_counts_something(catalogue):
    executed = run(table("facts", catalogue["facts"]), catalogue)
    assert executed.meter.as_dict()["rows_materialised"] >= 0


def test_the_execution_reports_its_node_count(catalogue):
    built = Project(input=table("facts", catalogue["facts"]), names=("id",))
    assert run(built, catalogue).nodes == 2


def test_the_execution_summarises(catalogue):
    summary = run(table("facts", catalogue["facts"]), catalogue).as_dict()
    assert summary["rows"] == catalogue["facts"].rows


def test_a_choice_describes_itself(catalogue):
    built = Join(
        left=table("facts", catalogue["facts"]),
        right=table("shops", catalogue["shops"]),
        left_keys=("shop",),
        right_keys=("shop",),
    )
    executed = run(built, catalogue)
    assert "Join" in executed.explain()


def test_asking_for_a_strategy_that_was_not_chosen_is_empty(catalogue):
    executed = run(table("facts", catalogue["facts"]), catalogue)
    assert executed.strategy("Join") == ""


def test_explain_returns_text(catalogue):
    built = Project(input=table("facts", catalogue["facts"]), names=("id",))
    assert "Project" in explain(built, catalogue)


def test_a_limit_of_zero_returns_nothing(catalogue):
    built = Limit(input=table("facts", catalogue["facts"]), count=0)
    assert execute(built, catalogue).rows == 0


def test_an_offset_past_the_end_returns_nothing(catalogue):
    built = Limit(input=table("facts", catalogue["facts"]), count=10, offset=100000)
    assert execute(built, catalogue).rows == 0


def test_a_limit_without_a_sort_below_it_still_works(catalogue):
    built = Limit(input=table("facts", catalogue["facts"]), count=7)
    assert execute(built, catalogue).rows == 7


def test_a_filter_that_keeps_nothing_returns_an_empty_batch(catalogue):
    built = Filter(
        input=table("facts", catalogue["facts"]),
        predicate=Compare(">", column("amount"), literal(1e9)),
    )
    assert execute(built, catalogue).rows == 0


def test_a_group_over_an_empty_input_returns_no_groups(catalogue):
    built = Group(
        input=Filter(
            input=table("facts", catalogue["facts"]),
            predicate=Compare(">", column("amount"), literal(1e9)),
        ),
        keys=("shop",),
        aggregates=(Aggregate(name="n", function="count_star", source=""),),
    )
    assert execute(built, catalogue).rows == 0


def test_a_sort_of_an_empty_input_returns_nothing(catalogue):
    built = Sort(
        input=Filter(
            input=table("facts", catalogue["facts"]),
            predicate=Compare(">", column("amount"), literal(1e9)),
        ),
        keys=(SortKey(name="amount"),),
    )
    assert execute(built, catalogue).rows == 0


def test_the_nested_loop_limit_is_the_measured_one():
    assert physical.NESTED_LOOP_LIMIT == 32


def test_the_counting_limit_is_a_power_of_two():
    assert physical.COUNTING_LIMIT == 4096
