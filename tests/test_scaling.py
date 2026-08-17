from __future__ import annotations

import pytest

from cqe.errors import ConfigError
from cqe.eval import scaling
from cqe.eval.scaling import OPERATORS, SIZES, TOLERANCE, Growth, measure, measure_all


def test_a_filter_is_linear():
    assert scaling.a_filter_is_linear()["it_is_linear"]


def test_the_filter_fit_is_straight():
    assert scaling.a_filter_is_linear()["and_the_fit_is_straight"]


def test_a_projection_costs_nothing_at_every_size():
    assert scaling.a_projection_does_not_grow()["and_it_is_zero"]


def test_every_projection_cost_is_the_same():
    assert scaling.a_projection_does_not_grow()["every_cost_is_the_same"]


def test_a_sort_is_above_linear():
    assert scaling.a_sort_grows_faster_than_linear()["it_is_above_linear"]


def test_a_sort_is_below_quadratic():
    assert scaling.a_sort_grows_faster_than_linear()["and_below_quadratic"]


def test_the_sort_and_filter_exponents_differ():
    assert scaling.a_sort_grows_faster_than_linear()["the_gap"] > 0.05


def test_a_partial_sort_is_linear():
    assert scaling.a_partial_sort_is_linear()["it_is_linear"]


def test_a_partial_sort_beats_a_full_one_at_every_size():
    assert scaling.a_partial_sort_is_linear()["and_it_costs_less_at_every_size"]


def test_the_partial_sort_saving_is_large():
    assert scaling.a_partial_sort_is_linear()["the_ratio_at_the_largest"] > 3


def test_an_aggregate_is_linear_in_its_rows():
    assert scaling.an_aggregate_is_linear_in_its_rows()["it_is_linear"]


def test_the_per_row_work_does_not_move_with_the_groups():
    assert scaling.an_aggregate_grows_with_its_groups()["the_per_row_work_does_not_move"]


def test_the_per_group_work_does():
    assert scaling.an_aggregate_grows_with_its_groups()["the_per_group_work_does"]


def test_a_thousandfold_in_groups_costs_little():
    assert scaling.an_aggregate_grows_with_its_groups()["but_not_by_much"]


def test_a_join_is_linear_in_the_larger_side():
    assert scaling.a_join_is_linear_in_the_larger_side()["it_is_linear"]


def test_a_nested_loop_join_is_quadratic():
    assert scaling.a_nested_loop_join_is_quadratic()["and_the_method_can_see_a_two"]


def test_the_nested_loop_is_steeper_than_the_hash_join():
    assert scaling.a_nested_loop_join_is_quadratic()["the_nested_loop_is_steeper"]


def test_two_sizes_give_the_same_gap_as_sixteen():
    measured = scaling.two_sizes_cannot_tell_linear_from_n_log_n()
    assert abs(measured["the_narrow_gap"] - measured["the_wide_gap"]) < 0.05


def test_the_gap_between_a_sort_and_a_filter_is_small():
    assert scaling.two_sizes_cannot_tell_linear_from_n_log_n()["the_narrow_gap"] < 0.2


def test_every_operator_matches_its_expected_growth():
    assert scaling.every_operator_matches_its_expected_growth()["they_all_match"]


def test_no_operator_missed_its_expectation():
    assert scaling.every_operator_matches_its_expected_growth()["which_did_not"] == []


def test_every_fit_is_a_straight_line():
    assert scaling.every_fit_is_a_straight_line()["they_are_all_straight"]


def test_the_worst_fit_is_still_good():
    assert scaling.every_fit_is_a_straight_line()["the_worst"] > 0.95


def test_the_costs_are_reproducible():
    assert scaling.the_costs_are_reproducible()["they_are_identical"]


def test_the_exponents_are_reproducible():
    assert scaling.the_costs_are_reproducible()["the_exponents_match"]


def test_the_exponent_survives_a_different_range():
    assert scaling.the_exponent_survives_a_different_size_range()["they_agree"]


def test_the_range_difference_is_small():
    assert (
        scaling.the_exponent_survives_a_different_size_range()["the_largest_difference"] < 0.2
    )


def test_a_single_size_is_refused():
    assert scaling.a_single_size_is_refused()


def test_the_operator_table_covers_every_operator():
    assert len(scaling.compare_the_operators()) == len(OPERATORS)


def test_the_summary_says_they_all_match():
    assert scaling.summarise()["all_match"]


def test_the_summary_reports_the_range():
    assert scaling.summarise()["range"] == 16


def test_measuring_one_operator_returns_a_growth():
    made = measure("filter", scaling._filter, 1.0)
    assert isinstance(made, Growth) and made.name == "filter"


def test_a_growth_reports_its_sizes():
    assert measure("filter", scaling._filter, 1.0).sizes == SIZES


def test_a_growth_reports_one_cost_per_size():
    made = measure("filter", scaling._filter, 1.0)
    assert len(made.costs) == len(SIZES)


def test_a_growth_summarises():
    assert measure("filter", scaling._filter, 1.0).as_dict()["operator"] == "filter"


def test_a_growth_of_zero_costs_has_no_exponent():
    made = Growth(name="none", sizes=(1, 2), costs=(0, 0), expected=0.0)
    assert made.exponent == 0.0


def test_a_growth_of_one_point_has_no_exponent():
    made = Growth(name="none", sizes=(1,), costs=(10,), expected=1.0)
    assert made.exponent == 0.0


def test_a_perfect_power_law_fits_exactly():
    made = Growth(name="square", sizes=(1, 2, 4, 8), costs=(1, 4, 16, 64), expected=2.0)
    assert abs(made.exponent - 2.0) < 1e-9 and made.straight > 0.999


def test_a_linear_series_fits_at_one():
    made = Growth(name="line", sizes=(1, 2, 4, 8), costs=(1, 2, 4, 8), expected=1.0)
    assert abs(made.exponent - 1.0) < 1e-9


def test_a_growth_inside_the_tolerance_matches():
    made = Growth(
        name="near", sizes=(1, 2, 4, 8), costs=(1, 2, 4, 8), expected=1.0 + TOLERANCE / 2
    )
    assert made.matches


def test_a_growth_outside_the_tolerance_does_not():
    made = Growth(name="far", sizes=(1, 2, 4, 8), costs=(1, 4, 16, 64), expected=1.0)
    assert not made.matches


def test_measuring_every_operator_returns_one_each():
    assert len(measure_all()) == len(OPERATORS)


def test_measuring_at_two_sizes_works():
    assert len(measure("filter", scaling._filter, 1.0, sizes=(1000, 2000)).costs) == 2


def test_measuring_at_no_sizes_is_refused():
    with pytest.raises(ConfigError):
        measure("filter", scaling._filter, 1.0, sizes=())


def test_the_default_sizes_span_a_factor_of_sixteen():
    assert SIZES[-1] // SIZES[0] == 16


def test_every_operator_has_an_expected_exponent():
    assert all(expected >= 0 for _, _, expected in OPERATORS)


def test_the_costs_rise_with_the_sizes():
    made = measure("filter", scaling._filter, 1.0)
    assert list(made.costs) == sorted(made.costs)
