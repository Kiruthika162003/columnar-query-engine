from __future__ import annotations

import pytest

from cqe.cost.meter import Budget, Meter, compare_meters, measured
from cqe.errors import BudgetExceeded, ConfigError


class TestCounting:
    def test_a_meter_starts_at_nothing(self):
        assert Meter().values_touched == 0

    def test_touching_values_counts_them(self):
        meter = Meter()
        meter.touch(100)
        assert meter.values_touched == 100

    def test_and_the_bytes_they_occupied(self):
        meter = Meter()
        meter.touch(100, width=4)
        assert meter.bytes_read == 400

    def test_the_default_width_is_eight_bytes(self):
        meter = Meter()
        meter.touch(10)
        assert meter.bytes_read == 80

    def test_touching_attributes_to_an_operator(self):
        meter = Meter()
        meter.touch(50, "scan")
        assert meter.by_operator == {"scan": 50}

    def test_and_accumulates_per_operator(self):
        meter = Meter()
        meter.touch(50, "scan")
        meter.touch(20, "scan")
        meter.touch(5, "filter")
        assert meter.by_operator == {"scan": 70, "filter": 5}

    def test_an_unattributed_touch_records_no_operator(self):
        meter = Meter()
        meter.touch(50)
        assert meter.by_operator == {}

    def test_a_negative_touch_is_refused(self):
        with pytest.raises(ConfigError, match="not a count"):
            Meter().touch(-1)

    def test_materialising_counts_rows(self):
        meter = Meter()
        meter.materialise(30)
        assert meter.rows_materialised == 30

    def test_a_negative_materialise_is_refused(self):
        with pytest.raises(ConfigError, match="not a count"):
            Meter().materialise(-1)

    def test_probes_are_counted(self):
        meter = Meter()
        meter.probe(7)
        assert meter.hash_probes == 7

    def test_a_probe_defaults_to_one(self):
        meter = Meter()
        meter.probe()
        assert meter.hash_probes == 1

    def test_comparisons_are_counted(self):
        meter = Meter()
        meter.compare(12)
        assert meter.comparisons == 12

    def test_spills_are_counted(self):
        meter = Meter()
        meter.spill(1024)
        assert meter.spilled_bytes == 1024

    def test_a_negative_spill_is_refused(self):
        with pytest.raises(ConfigError, match="not a size"):
            Meter().spill(-1)

    def test_batches_are_counted(self):
        meter = Meter()
        meter.batch()
        meter.batch(3)
        assert meter.batches == 4


class TestArithmetic:
    def test_merging_adds_every_field(self):
        left = Meter(values_touched=10, comparisons=2)
        right = Meter(values_touched=5, comparisons=3)
        left.merge(right)
        assert left.values_touched == 15 and left.comparisons == 5

    def test_merging_adds_per_operator_counts(self):
        left = Meter()
        left.touch(10, "scan")
        right = Meter()
        right.touch(5, "scan")
        right.touch(1, "join")
        left.merge(right)
        assert left.by_operator == {"scan": 15, "join": 1}

    def test_a_copy_is_independent(self):
        meter = Meter()
        meter.touch(10, "scan")
        snapshot = meter.copy()
        meter.touch(10, "scan")
        assert snapshot.values_touched == 10 and meter.values_touched == 20

    def test_and_its_operator_map_is_independent(self):
        meter = Meter()
        meter.touch(10, "scan")
        snapshot = meter.copy()
        meter.touch(10, "join")
        assert "join" not in snapshot.by_operator

    def test_since_reports_the_delta(self):
        meter = Meter()
        meter.touch(10)
        earlier = meter.copy()
        meter.touch(25)
        assert meter.since(earlier).values_touched == 25

    def test_and_only_the_operators_that_moved(self):
        meter = Meter()
        meter.touch(10, "scan")
        earlier = meter.copy()
        meter.touch(4, "join")
        assert meter.since(earlier).by_operator == {"join": 4}

    def test_the_dominant_operator_is_the_biggest(self):
        meter = Meter()
        meter.touch(10, "scan")
        meter.touch(50, "join")
        assert meter.dominant_operator == "join"

    def test_and_is_empty_when_nothing_was_attributed(self):
        assert Meter().dominant_operator == ""

    def test_the_share_sums_to_one(self):
        meter = Meter()
        meter.touch(30, "scan")
        meter.touch(70, "join")
        assert sum(meter.share.values()) == pytest.approx(1.0)

    def test_the_share_is_empty_when_nothing_was_attributed(self):
        assert Meter().share == {}

    def test_a_meter_serialises(self):
        meter = Meter()
        meter.touch(10)
        assert meter.as_dict()["values_touched"] == 10

    def test_a_meter_prints_its_three_headline_numbers(self):
        meter = Meter()
        meter.touch(10)
        meter.materialise(2)
        assert "10 values" in str(meter) and "2 rows" in str(meter)


class TestMeasuredRegion:
    def test_the_inner_meter_starts_clean(self):
        outer = Meter()
        outer.touch(100)
        with measured(outer) as inner:
            assert inner.values_touched == 0

    def test_and_folds_into_the_outer_one(self):
        outer = Meter()
        outer.touch(100)
        with measured(outer) as inner:
            inner.touch(50)
        assert outer.values_touched == 150

    def test_the_delta_is_readable_afterwards(self):
        outer = Meter()
        with measured(outer) as inner:
            inner.touch(50, "join")
        assert inner.by_operator == {"join": 50}

    def test_it_folds_in_even_after_a_failure(self):
        outer = Meter()
        with pytest.raises(ValueError, match="boom"), measured(outer) as inner:
            inner.touch(20)
            raise ValueError("boom")
        assert outer.values_touched == 20


class TestBudget:
    def test_an_empty_budget_limits_nothing(self):
        assert Budget().unlimited

    def test_a_set_budget_limits_something(self):
        assert not Budget(values=100).unlimited

    def test_a_budget_within_bounds_passes(self):
        meter = Meter()
        meter.touch(50)
        Budget(values=100).check(meter)

    def test_a_value_overrun_is_caught(self):
        meter = Meter()
        meter.touch(150)
        with pytest.raises(BudgetExceeded, match="values budget"):
            Budget(values=100).check(meter)

    def test_a_row_overrun_is_caught(self):
        meter = Meter()
        meter.materialise(150)
        with pytest.raises(BudgetExceeded, match="rows budget"):
            Budget(rows=100).check(meter)

    def test_a_spill_overrun_is_caught(self):
        meter = Meter()
        meter.spill(2048)
        with pytest.raises(BudgetExceeded, match="spill budget"):
            Budget(spill_bytes=1024).check(meter)

    def test_a_memory_overrun_is_caught(self):
        with pytest.raises(BudgetExceeded, match="memory budget"):
            Budget(memory_bytes=1024).check(Meter(), live_bytes=2048)

    def test_the_error_carries_the_limit_and_the_reach(self):
        meter = Meter()
        meter.touch(150)
        try:
            Budget(values=100).check(meter)
        except BudgetExceeded as overrun:
            assert overrun.limit == 100 and overrun.reached == 150

    def test_and_the_overrun_as_a_fraction(self):
        meter = Meter()
        meter.touch(150)
        try:
            Budget(values=100).check(meter)
        except BudgetExceeded as overrun:
            assert overrun.overrun == pytest.approx(0.5)

    def test_a_zero_budget_is_refused(self):
        with pytest.raises(ConfigError, match="is not a budget"):
            Budget(values=0)

    def test_a_negative_budget_is_refused(self):
        with pytest.raises(ConfigError, match="is not a budget"):
            Budget(rows=-1)

    def test_a_budget_answers_a_what_if(self):
        meter = Meter()
        meter.touch(90)
        assert Budget(values=100).would_exceed(meter, 20)

    def test_and_says_no_when_it_fits(self):
        meter = Meter()
        meter.touch(90)
        assert not Budget(values=100).would_exceed(meter, 5)

    def test_a_budget_serialises(self):
        assert Budget(values=100).as_dict()["values"] == 100


class TestComparison:
    def test_two_meters_compare_as_a_ratio(self):
        left = Meter(values_touched=100)
        right = Meter(values_touched=50)
        assert compare_meters(left, right)["values"] == 2.0

    def test_a_zero_denominator_gives_infinity(self):
        left = Meter(values_touched=100)
        assert compare_meters(left, Meter())["values"] == float("inf")

    def test_two_empty_meters_are_level(self):
        assert compare_meters(Meter(), Meter())["values"] == 1.0

    def test_the_comparison_carries_both_sides(self):
        left = Meter(values_touched=100)
        result = compare_meters(left, Meter(values_touched=50))
        assert result["left"]["values_touched"] == 100
        assert result["right"]["values_touched"] == 50

    def test_every_unit_is_compared(self):
        result = compare_meters(Meter(), Meter())
        assert {"values", "bytes", "rows", "comparisons"} <= set(result)
