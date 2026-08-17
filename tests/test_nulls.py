from __future__ import annotations

import numpy as np
import pytest

from cqe.columns import nulls
from cqe.columns.array import Column, integer_column
from cqe.columns.nulls import (
    POSITIONAL_CROSSOVER,
    SENTINEL,
    cheapest,
    compare_at,
    masked,
    positional,
    sentinel,
)
from cqe.errors import ConfigError


def test_the_mask_costs_the_same_at_every_rate():
    assert nulls.the_mask_costs_the_same_whatever_the_null_rate()["the_cost_never_moves"]


def test_the_mask_is_one_bit_per_row():
    assert nulls.the_mask_costs_the_same_whatever_the_null_rate()["it_is_one_bit_per_row"]


def test_the_mask_overhead_is_small_on_a_wide_column():
    assert (
        nulls.the_mask_costs_the_same_whatever_the_null_rate()["the_overhead_at_eight_bytes"]
        < 0.02
    )


def test_the_mask_overhead_is_larger_on_a_narrow_one():
    assert nulls.the_mask_costs_the_same_whatever_the_null_rate()["and_at_one_byte"] == 0.125


def test_a_positional_list_wins_when_nulls_are_rare():
    assert nulls.the_positional_list_is_smaller_when_nulls_are_rare()["and_it_wins_somewhere"]


def test_it_only_wins_below_the_crossover():
    assert nulls.the_positional_list_is_smaller_when_nulls_are_rare()[
        "positional_wins_below_it"
    ]


def test_it_loses_when_nulls_are_common():
    assert nulls.the_positional_list_is_smaller_when_nulls_are_rare()["and_loses_above"]


def test_it_is_far_smaller_at_the_rarest():
    assert nulls.the_positional_list_is_smaller_when_nulls_are_rare()["at_the_rarest"] > 100


def test_a_safe_sentinel_round_trips():
    assert nulls.the_sentinel_costs_nothing_and_can_be_wrong()["the_safe_column_round_trips"]


def test_a_risky_sentinel_does_not():
    assert nulls.the_sentinel_costs_nothing_and_can_be_wrong()["the_risky_column_does_not"]


def test_the_sentinel_wrongly_nulls_real_rows():
    assert nulls.the_sentinel_costs_nothing_and_can_be_wrong()["rows_wrongly_called_null"] > 0


def test_a_float_column_holds_genuine_not_a_numbers():
    assert nulls.a_float_column_has_no_safe_sentinel()["and_they_are_real_values"]


def test_the_mask_and_the_positions_agree():
    assert nulls.the_mask_and_the_positions_agree()["they_agree"]


def test_a_null_is_not_a_zero():
    assert nulls.a_null_is_not_a_zero()["they_differ"]


def test_the_difference_is_the_hidden_values():
    assert nulls.a_null_is_not_a_zero()["and_the_difference_is_the_hidden_values"]


def test_the_overhead_falls_with_the_column_width():
    assert nulls.the_overhead_depends_on_the_column_width()["the_overhead_falls_with_the_width"]


def test_the_overhead_ratio_is_eight():
    assert nulls.the_overhead_depends_on_the_column_width()["the_ratio"] == 8.0


def test_the_nullable_filter_keeps_fewer_rows():
    assert nulls.a_filter_over_nulls_costs_one_extra_pass()["the_nullable_one_kept_fewer"]


def test_the_mask_pass_is_not_counted():
    assert nulls.a_filter_over_nulls_costs_one_extra_pass()["the_mask_pass_is_not_counted"]


def test_the_sentinel_always_wins_on_size():
    assert nulls.a_null_rate_sweep_names_the_winner()["the_sentinel_always_wins_on_size"]


def test_the_correct_winner_changes_with_the_rate():
    assert nulls.a_null_rate_sweep_names_the_winner()["the_correct_answer_changes"]


def test_positional_wins_when_nulls_are_rare():
    assert nulls.a_null_rate_sweep_names_the_winner()["positional_wins_when_rare"]


def test_the_mask_wins_when_nulls_are_common():
    assert nulls.a_null_rate_sweep_names_the_winner()["and_the_mask_wins_when_common"]


def test_a_column_of_all_nulls_keeps_its_values():
    assert nulls.a_column_of_all_nulls_still_holds_its_values()[
        "and_the_values_are_still_there"
    ]


def test_a_column_of_all_nulls_reads_as_nulls():
    assert nulls.a_column_of_all_nulls_still_holds_its_values()[
        "to_list_gives_nothing_but_nulls"
    ]


def test_a_clean_column_carries_no_mask():
    assert nulls.a_column_with_no_nulls_carries_no_mask()["the_clean_column_pays_nothing"]


def test_a_nullable_column_does():
    assert nulls.a_column_with_no_nulls_carries_no_mask()["and_the_nullable_one_pays"]


def test_more_nulls_than_rows_is_refused():
    assert nulls.more_nulls_than_rows_is_refused()


def test_a_mismatched_mask_is_refused():
    assert nulls.a_mask_of_the_wrong_length_is_refused()


def test_the_representation_table_covers_three():
    assert len(nulls.compare_the_representations()) == 3


def test_only_the_sentinel_has_a_risk():
    table = {one["representation"]: one["risks"] for one in nulls.compare_the_representations()}
    assert table["mask"] == "nothing" and table["sentinel"] != "nothing"


def test_the_summary_says_the_sentinel_can_be_wrong():
    assert nulls.summarise()["the_sentinel_can_be_wrong"]


def test_a_mask_costs_a_bit_per_row():
    assert masked(1000, 100).null_bytes == 125


def test_a_mask_of_a_partial_byte_rounds_up():
    assert masked(1001, 0).null_bytes == 126


def test_a_sentinel_costs_nothing():
    assert sentinel(1000, 100).null_bytes == 0


def test_a_positional_list_costs_four_bytes_a_null():
    assert positional(1000, 100).null_bytes == 400


def test_every_representation_holds_the_same_values():
    made = compare_at(1000, 100)
    assert len({one.value_bytes for one in made}) == 1


def test_a_representation_reports_its_rate():
    assert masked(1000, 250).rate == 0.25


def test_a_representation_reports_its_overhead():
    assert masked(1000, 0, width=8).overhead == 125 / 8000


def test_a_representation_summarises():
    assert masked(1000, 100).as_dict()["representation"] == "mask"


def test_the_cheapest_is_always_the_sentinel():
    assert cheapest(1000, 100) == "sentinel"


def test_comparing_at_an_impossible_rate_is_refused():
    with pytest.raises(ConfigError):
        compare_at(100, 500)


def test_the_crossover_is_one_in_thirty_two():
    assert POSITIONAL_CROSSOVER == 1 / 32


def test_at_the_crossover_the_two_are_equal():
    rows = 32000
    nulls_at = int(rows * POSITIONAL_CROSSOVER)
    assert masked(rows, nulls_at).null_bytes == positional(rows, nulls_at).null_bytes


def test_below_the_crossover_the_list_is_smaller():
    rows = 32000
    nulls_at = int(rows * POSITIONAL_CROSSOVER / 2)
    assert positional(rows, nulls_at).null_bytes < masked(rows, nulls_at).null_bytes


def test_above_the_crossover_the_mask_is_smaller():
    rows = 32000
    nulls_at = int(rows * POSITIONAL_CROSSOVER * 2)
    assert masked(rows, nulls_at).null_bytes < positional(rows, nulls_at).null_bytes


def test_the_sentinel_is_minus_one():
    assert SENTINEL == -1


def test_a_column_with_a_null_reports_it():
    values = np.arange(10)
    made = integer_column("v", values)
    valid = np.ones(10, dtype=bool)
    valid[3] = False
    column = Column(field=made.field, values=values, valid=valid)
    assert column.to_list()[3] is None


def test_a_column_with_a_null_keeps_the_other_values():
    values = np.arange(10)
    made = integer_column("v", values)
    valid = np.ones(10, dtype=bool)
    valid[3] = False
    column = Column(field=made.field, values=values, valid=valid)
    assert column.to_list()[4] == 4


def test_a_column_counts_its_nulls():
    values = np.arange(10)
    made = integer_column("v", values)
    valid = np.array([one % 2 == 0 for one in range(10)])
    column = Column(field=made.field, values=values, valid=valid)
    assert column.null_count == 5


def test_a_clean_column_counts_none():
    assert integer_column("v", np.arange(10)).null_count == 0
