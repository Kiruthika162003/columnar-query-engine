from __future__ import annotations

import numpy as np
import pytest

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import ConfigError, UnknownColumn
from cqe.exec.batch import Batch
from cqe.exec.expr import And, Compare, column, literal
from cqe.stats import correlation
from cqe.stats.correlation import (
    BUCKETS,
    RELATED,
    corrected_selectivity,
    linear,
    mutual,
    relate,
    relate_all,
)


@pytest.fixture(scope="module")
def related() -> Batch:
    """Two columns where one is mostly the other."""
    state = np.random.default_rng(101)
    rows = 8000
    first = state.normal(100, 20, rows)
    return Batch.from_columns(
        [
            floating_column("first", first),
            floating_column("second", 0.9 * first + 0.1 * state.normal(0, 20, rows)),
            integer_column("shop", state.integers(0, 30, rows)),
        ]
    )


@pytest.fixture(scope="module")
def unrelated() -> Batch:
    """Two columns drawn separately."""
    state = np.random.default_rng(103)
    rows = 8000
    return Batch.from_columns(
        [
            floating_column("first", state.normal(100, 20, rows)),
            floating_column("second", state.normal(100, 20, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 5, rows)]),
        ]
    )


def test_independent_columns_measure_as_independent():
    assert correlation.independent_columns_measure_as_independent()["it_is_called_unrelated"]


def test_the_linear_measure_is_near_zero_on_independent_columns():
    assert correlation.independent_columns_measure_as_independent()[
        "the_linear_correlation_is_near_zero"
    ]


def test_correlated_columns_measure_as_correlated():
    assert correlation.correlated_columns_measure_as_correlated()["it_is_called_related"]


def test_the_linear_measure_is_high_on_correlated_columns():
    assert correlation.correlated_columns_measure_as_correlated()[
        "the_linear_correlation_is_high"
    ]


def test_a_curve_is_invisible_to_the_linear_measure():
    assert correlation.a_curved_relationship_is_invisible_to_the_linear_measure()[
        "the_line_sees_nothing"
    ]


def test_a_curve_is_visible_to_the_histogram():
    assert correlation.a_curved_relationship_is_invisible_to_the_linear_measure()[
        "the_histogram_sees_it"
    ]


def test_the_two_measures_differ_by_a_lot_on_a_curve():
    assert (
        correlation.a_curved_relationship_is_invisible_to_the_linear_measure()["the_ratio"] > 5
    )


def test_a_city_determines_its_country():
    assert correlation.a_categorical_relationship_needs_the_histogram()[
        "the_city_determines_the_country"
    ]


def test_an_unrelated_amount_is_unrelated():
    assert correlation.a_categorical_relationship_needs_the_histogram()[
        "and_the_amount_is_unrelated"
    ]


def test_the_linear_measure_refuses_text():
    assert correlation.a_categorical_relationship_needs_the_histogram()[
        "the_linear_measure_refuses_text"
    ]


def test_the_product_understates_a_correlated_conjunction():
    assert correlation.the_independence_assumption_understates_a_correlated_conjunction()[
        "the_estimate_is_too_small"
    ]


def test_the_understatement_is_a_real_factor():
    assert (
        correlation.the_independence_assumption_understates_a_correlated_conjunction()[
            "by_a_factor_of"
        ]
        > 1.5
    )


def test_the_product_is_right_on_independent_columns():
    assert correlation.the_assumption_is_right_when_it_should_be()["it_is_close"]


def test_the_correction_helps_where_it_should():
    assert correlation.the_correction_helps_on_correlated_columns()["the_correction_helped"]


def test_the_correction_closes_most_of_the_gap():
    assert (
        correlation.the_correction_helps_on_correlated_columns()[
            "it_closed_this_share_of_the_gap"
        ]
        > 0.5
    )


def test_the_correction_does_not_hurt_independent_columns():
    assert correlation.the_correction_does_not_hurt_independent_columns()["it_did_not_hurt"]


def test_the_mutual_information_rises_with_the_strength():
    assert correlation.the_correction_across_a_range_of_strengths()[
        "the_mutual_information_rises"
    ]


def test_the_independent_error_rises_with_it():
    assert correlation.the_correction_across_a_range_of_strengths()[
        "and_the_independent_error_rises_with_it"
    ]


def test_the_correction_never_hurts_across_the_sweep():
    assert correlation.the_correction_across_a_range_of_strengths()["hurt"] == 0


def test_a_column_is_perfectly_related_to_itself():
    assert correlation.a_column_is_perfectly_related_to_itself()["the_linear_measure_is_one"]


def test_the_mutual_information_of_a_column_with_itself_is_one():
    assert correlation.a_column_is_perfectly_related_to_itself()[
        "and_so_is_the_mutual_information"
    ]


def test_a_constant_column_relates_to_nothing():
    assert correlation.a_constant_column_relates_to_nothing()["neither_reports_a_relationship"]


def test_nulls_do_not_change_the_measure():
    assert correlation.nulls_are_left_out_of_both_measures()["they_agree"]


def test_the_measure_survives_every_bucket_count():
    assert correlation.the_bucket_count_changes_the_measure()["the_correlated_pair_stays_high"]


def test_the_independent_pair_stays_low_at_every_bucket_count():
    assert correlation.the_bucket_count_changes_the_measure()["the_independent_pair_stays_low"]


def test_the_gap_is_clear_at_every_bucket_count():
    assert correlation.the_bucket_count_changes_the_measure()[
        "and_the_gap_is_clear_at_every_size"
    ]


def test_every_pair_of_a_table_is_measured():
    assert correlation.every_pair_of_a_table_is_measured()["it_is_the_triangle"]


def test_the_related_pair_is_found():
    assert correlation.every_pair_of_a_table_is_measured()[
        "the_city_and_country_pair_is_related"
    ]


def test_a_missing_column_is_refused():
    assert correlation.a_missing_column_is_refused()


def test_a_text_column_is_refused_by_the_linear_measure():
    assert correlation.a_text_column_is_refused_by_the_linear_measure()


def test_a_single_bucket_is_refused():
    assert correlation.a_single_bucket_is_refused()


def test_the_measure_table_covers_four_shapes():
    assert len(correlation.compare_the_measures()) == 4


def test_only_the_independent_shape_is_unrelated():
    table = {one["shape"]: one["strength"] for one in correlation.compare_the_measures()}
    assert table["independent"] == "none" and table["curved"] == "strong"


def test_the_summary_says_the_line_misses_a_curve():
    assert correlation.summarise()["the_line_misses_a_curve"]


def test_a_linear_correlation_is_between_minus_one_and_one(related):
    assert -1 <= linear(related, "first", "second") <= 1


def test_a_correlated_pair_is_near_one(related):
    assert linear(related, "first", "second") > 0.9


def test_an_uncorrelated_pair_is_near_zero(unrelated):
    assert abs(linear(unrelated, "first", "second")) < 0.05


def test_a_correlation_is_symmetric(related):
    assert abs(linear(related, "first", "second") - linear(related, "second", "first")) < 1e-9


def test_a_mutual_information_is_between_zero_and_one(related):
    assert 0 <= mutual(related, "first", "second") <= 1


def test_a_mutual_information_is_symmetric(related):
    assert abs(mutual(related, "first", "second") - mutual(related, "second", "first")) < 1e-9


def test_a_related_pair_is_above_the_threshold(related):
    assert mutual(related, "first", "second") > RELATED


def test_an_unrelated_pair_is_below_it(unrelated):
    assert mutual(unrelated, "first", "second") < RELATED


def test_relating_reports_both_measures(related):
    made = relate(related, "first", "second")
    assert made.linear > 0.9 and made.mutual > RELATED


def test_a_relationship_names_its_strength(related):
    assert relate(related, "first", "second").strength in ("some", "strong")


def test_an_unrelated_pair_has_no_strength(unrelated):
    assert relate(unrelated, "first", "second").strength == "none"


def test_a_relationship_summarises(related):
    assert relate(related, "first", "second").as_dict()["left"] == "first"


def test_relating_a_text_column_skips_the_linear_measure(unrelated):
    assert relate(unrelated, "first", "region").linear == 0.0


def test_relating_every_pair_gives_the_triangle(related):
    assert len(relate_all(related)) == 3


def test_a_missing_column_in_the_linear_measure_is_refused(related):
    with pytest.raises(UnknownColumn):
        linear(related, "first", "nothing")


def test_a_missing_column_in_the_mutual_measure_is_refused(related):
    with pytest.raises(UnknownColumn):
        mutual(related, "first", "nothing")


def test_a_text_column_in_the_linear_measure_is_refused(unrelated):
    with pytest.raises(ConfigError):
        linear(unrelated, "first", "region")


def test_a_text_column_in_the_mutual_measure_is_allowed(unrelated):
    assert 0 <= mutual(unrelated, "first", "region") <= 1


def test_one_bucket_is_refused(related):
    with pytest.raises(ConfigError):
        mutual(related, "first", "second", buckets=1)


def test_a_correction_of_a_single_predicate_is_itself(related):
    predicate = Compare(">", column("first"), literal(100.0))
    independent, corrected = corrected_selectivity(related, predicate)
    assert independent == corrected


def test_a_correction_of_a_conjunction_differs(related):
    predicate = And(
        parts=(
            Compare(">", column("first"), literal(100.0)),
            Compare(">", column("second"), literal(100.0)),
        )
    )
    independent, corrected = corrected_selectivity(related, predicate)
    assert corrected > independent


def test_a_correction_never_exceeds_the_smaller_factor(related):
    predicate = And(
        parts=(
            Compare(">", column("first"), literal(100.0)),
            Compare(">", column("second"), literal(100.0)),
        )
    )
    _, corrected = corrected_selectivity(related, predicate)
    assert corrected <= 1.0


def test_a_conjunction_over_one_column_is_left_alone(related):
    predicate = And(
        parts=(
            Compare(">", column("first"), literal(100.0)),
            Compare("<", column("first"), literal(120.0)),
        )
    )
    independent, corrected = corrected_selectivity(related, predicate)
    assert independent == corrected


def test_the_bucket_default_is_a_power_of_two():
    assert BUCKETS == 16


def test_a_two_row_table_does_not_divide_by_zero():
    batch = Batch.from_columns(
        [floating_column("a", [1.0, 2.0]), floating_column("b", [1.0, 2.0])]
    )
    assert 0 <= mutual(batch, "a", "b") <= 1


def test_a_one_row_table_reports_nothing():
    batch = Batch.from_columns([floating_column("a", [1.0]), floating_column("b", [2.0])])
    assert mutual(batch, "a", "b") == 0.0 and linear(batch, "a", "b") == 0.0
