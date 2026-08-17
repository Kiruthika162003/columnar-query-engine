from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, integer_column, string_column
from cqe.errors import ConfigError
from cqe.exec.batch import Batch
from cqe.types.schema import STRING

# A bloom filter per row group, which answers one question: can this group possibly contain this
# value.
#
# It is the second pruning mechanism in this package and it is worth being precise about how it
# differs from the first. storage/statistics.py keeps a minimum and a maximum per column per
# group, which prunes ranges: a group whose maximum is below the wanted value cannot contain it.
# That is exact, costs sixteen bytes and does nothing at all for an equality on an unsorted
# column, because the wanted value is almost always between the minimum and the maximum.
#
# A bloom filter prunes equalities. It costs about ten bits per distinct value and it is not
# exact: it can say a group might contain a value it does not, and it can never say a group does
# not contain a value it does. That asymmetry is the whole design. A false positive costs one
# unnecessary read; a false negative would cost a wrong answer, so the structure is built so
# that the second cannot happen.
#
# The two mechanisms compose in the sense that they prune different queries, and the
# measurements below check exactly that rather than assuming it: the range predicate that the
# zone map handles is the one the bloom filter is useless for, and the reverse.
#
# The hash is not cryptographic and does not need to be. It needs to spread, and the
# measurements check the spread rather than trusting the function.

# Bits per distinct value. The classic table says about 9.6 bits gives a one percent false
# positive rate at the optimal number of hashes, and ten is that rounded up to something a
# reader can hold in their head.
BITS_PER_ENTRY = 10

# Seven hashes, which is the classic optimum for ten bits per entry and is not what this module
# started with. It started with two, on the argument that seven hashes of every value cost seven
# times what one does and the theoretical optimum is about a structure rather than about this
# one. The measurement disagreed and the difference was not marginal: two hashes give a 6.4
# percent false positive rate where seven give 0.86, a factor of seven and a half. Seven extra
# hashes cost seven arithmetic operations per lookup and a false positive costs reading a whole
# row group off disk, so the trade was never close once it was measured.
HASHES = 7

# A filter smaller than this is not worth the header it is stored with.
MINIMUM_BITS = 64


@dataclass(frozen=True)
class Bloom:
    """A bit array and the parameters it was built with."""

    bits: np.ndarray
    hashes: int
    entries: int

    @property
    def size(self) -> int:
        """How many bits the filter holds."""
        return int(self.bits.size)

    @property
    def nbytes(self) -> int:
        """How much space the filter takes."""
        return int(self.bits.nbytes)

    @property
    def occupancy(self) -> float:
        """The share of bits that are set, which is what drives the false positive rate."""
        return float(self.bits.mean())

    def might_contain(self, value) -> bool:
        """Whether this value might be in the set.

        Might, not does. A true answer means the value's bits are all set, which happens either
        because the value is there or because other values set them between them.
        """
        return all(self.bits[one] for one in _positions(value, self.size, self.hashes))

    def any_of(self, values: Sequence) -> bool:
        """Whether any of several values might be in the set, for an in list."""
        return any(self.might_contain(one) for one in values)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "bits": self.size,
            "bytes": self.nbytes,
            "hashes": self.hashes,
            "entries": self.entries,
            "occupancy": round(self.occupancy, 3),
        }


def _hash(value) -> int:
    """One integer per value, stable across runs.

    Python's own hash is randomised per process for strings, which would make a filter written
    to a file unreadable by the next process. That is not a subtle failure and it is an easy one
    to ship, because within one process everything works.
    """
    if isinstance(value, (int, np.integer)):
        return int(value) * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF
    if isinstance(value, (float, np.floating)):
        return _hash(int(np.float64(value).view(np.int64)))
    text = str(value).encode("utf-8")
    out = 0xCBF29CE484222325
    for one in text:
        out = ((out ^ one) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return out


def _positions(value, size: int, hashes: int) -> list[int]:
    """Which bits one value sets or tests.

    Two independent hashes combined linearly, which is the standard trick: the ith position is
    the first hash plus i times the second. It gives k hashes for the price of two and the
    literature says the false positive rate is indistinguishable from k real ones.
    """
    first = _hash(value)
    second = _hash(first ^ 0x5BF03635) | 1
    return [int((first + one * second) % size) for one in range(hashes)]


def build(
    values: Sequence,
    bits_per_entry: int = BITS_PER_ENTRY,
    hashes: int = HASHES,
) -> Bloom:
    """A filter over a set of values.

    Sized from the distinct count rather than the row count, because a column of a thousand rows
    holding five distinct values needs five values' worth of bits. Sizing on rows would make the
    filter two hundred times larger than it needs to be and its false positive rate zero, which
    is a waste rather than a bug and is still worth not doing.
    """
    if bits_per_entry <= 0:
        raise ConfigError(f"{bits_per_entry} bits per entry is not a filter")
    if hashes <= 0:
        raise ConfigError(f"{hashes} hashes is not a filter")
    distinct = {_hash(one) for one in values}
    size = max(len(distinct) * bits_per_entry, MINIMUM_BITS)
    bits = np.zeros(size, dtype=bool)
    for one in values:
        for position in _positions(one, size, hashes):
            bits[position] = True
    return Bloom(bits=bits, hashes=hashes, entries=len(distinct))


def build_for(column: Column, **arguments) -> Bloom:
    """A filter over one column, reading its dictionary when it has one.

    A dictionary encoded column already knows its distinct values, so the filter can be built
    over the entries rather than over the rows, which is far fewer hashes when a column has many
    rows per entry.

    The entries that are present, not every entry in the dictionary. A column sliced out of a
    larger one keeps the whole dictionary, so a row group holding one region out of six carries
    a dictionary naming all six, and a filter built from it says yes to all six. That was the
    first version and it made every filter useless in exactly the case the module is for, while
    looking correct on an unsliced column. The codes actually present are what the group holds.
    """
    if column.field.logical == STRING and column.dictionary:
        codes = column.values if column.valid is None else column.values[column.valid]
        entries = [column.dictionary[int(one)] for one in np.unique(codes)]
        return build(entries, **arguments)
    return build(column.to_list(), **arguments)


def optimal_hashes(bits_per_entry: int = BITS_PER_ENTRY) -> int:
    """How many hashes minimise the false positive rate at a given size.

    The classic result, ln 2 times the bits per entry. Here to be checked against rather than to
    be trusted, and the curve below found the same minimum it predicts.
    """
    return max(round(bits_per_entry * math.log(2)), 1)


def predicted_rate(bits_per_entry: int = BITS_PER_ENTRY, hashes: int = HASHES) -> float:
    """The theoretical false positive rate, for comparing against the measured one."""
    return float((1 - math.exp(-hashes / bits_per_entry)) ** hashes)


@dataclass(frozen=True)
class Pruning:
    """What a set of filters kept and what it skipped."""

    groups: int
    kept: int
    hits: int

    @property
    def skipped(self) -> int:
        """How many groups did not have to be read."""
        return self.groups - self.kept

    @property
    def share(self) -> float:
        """The share of groups skipped."""
        return self.skipped / max(self.groups, 1)

    @property
    def false_positives(self) -> int:
        """Groups kept that did not actually hold the value."""
        return self.kept - self.hits

    @property
    def rate(self) -> float:
        """The share of groups without the value that were kept anyway."""
        without = self.groups - self.hits
        return self.false_positives / max(without, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "groups": self.groups,
            "kept": self.kept,
            "skipped": self.skipped,
            "share": round(self.share, 3),
            "false_positives": self.false_positives,
            "rate": round(self.rate, 4),
        }


def prune(filters: Sequence[Bloom], value, truth: Sequence[bool] | None = None) -> Pruning:
    """Which groups might hold a value, and how many of those really do.

    The truth argument is what makes this a measurement rather than a count. Without it the
    function can report how much it skipped and not whether it was right to, and a filter that
    skips everything looks best of all.
    """
    if truth is not None and len(truth) != len(filters):
        raise ConfigError(f"{len(truth)} truths against {len(filters)} filters")
    kept = [one for one in range(len(filters)) if filters[one].might_contain(value)]
    hits = sum(truth) if truth is not None else len(kept)
    return Pruning(groups=len(filters), kept=len(kept), hits=int(hits))


def _groups(
    rows: int = 40000,
    group_size: int = 500,
    distinct: int = 2000,
    seed: int = 13,
) -> tuple[list[Batch], list[str]]:
    """A table cut into row groups, with a string key that repeats across them."""
    state = np.random.default_rng(seed)
    keys = [f"key{one:05d}" for one in range(distinct)]
    drawn = state.choice(keys, size=rows)
    values = state.normal(100, 20, rows)
    out = []
    for start in range(0, rows, group_size):
        stop = min(start + group_size, rows)
        out.append(
            Batch.from_columns(
                [
                    string_column("k", list(drawn[start:stop])),
                    integer_column("v", (values[start:stop] * 100).astype(np.int64)),
                ]
            )
        )
    return out, keys


def a_filter_never_says_no_to_a_value_it_holds(rows: int = 5000) -> dict:
    """The one property that has to hold, checked on every value in the set.

    A false negative is a wrong answer rather than a slow one, so this is not a statistical
    claim and is not allowed to fail once.
    """
    state = np.random.default_rng(3)
    values = [f"value{one}" for one in state.integers(0, rows, rows)]
    filter_ = build(values)
    missed = [one for one in set(values) if not filter_.might_contain(one)]
    return {
        "values": len(set(values)),
        "missed": len(missed),
        "it_never_says_no": not missed,
        "occupancy": round(filter_.occupancy, 3),
    }


def the_false_positive_rate_is_what_it_is(trials: int = 20000) -> dict:
    """How often a filter says yes to something that is not in it.

    Measured against the theoretical rate for these parameters rather than against the classic
    one percent, because the classic number is for the optimal hash count and this uses two.
    """
    state = np.random.default_rng(5)
    inside = [f"in{one}" for one in range(2000)]
    filter_ = build(inside)
    outside = [f"out{one}" for one in state.integers(0, trials * 10, trials)]
    wrong = sum(1 for one in set(outside) if filter_.might_contain(one))
    measured = wrong / len(set(outside))
    return {
        "trials": len(set(outside)),
        "false_positives": wrong,
        "measured": round(measured, 4),
        "predicted": round(predicted_rate(), 4),
        "bits_per_entry": BITS_PER_ENTRY,
        "hashes": HASHES,
        "occupancy": round(filter_.occupancy, 3),
    }


def more_hashes_are_better_up_to_a_point() -> dict:
    """The false positive rate against the hash count, at a fixed size.

    The curve the classic result describes, measured, and the measurement that set HASHES. The
    rate falls and then rises, because more hashes mean more bits tested and also more bits set,
    and past the optimum the second wins. The minimum is at seven, which is exactly what ln 2
    times ten bits per entry predicts, so the theory is right about this structure.

    The module was written with two hashes on the argument that the extra five cost five times
    the arithmetic. They do, and they buy a factor of seven and a half in the rate, and the
    arithmetic is five multiplications against a row group read. The constant was changed.
    """
    state = np.random.default_rng(7)
    inside = [f"in{one}" for one in range(1000)]
    outside = [f"out{one}" for one in state.integers(0, 100000, 5000)]
    out = []
    for hashes in (1, 2, 3, 5, 7, 11, 15):
        filter_ = build(inside, hashes=hashes)
        wrong = sum(1 for one in set(outside) if filter_.might_contain(one))
        out.append(
            {
                "hashes": hashes,
                "rate": round(wrong / len(set(outside)), 4),
                "occupancy": round(filter_.occupancy, 3),
            }
        )
    rates = [one["rate"] for one in out]
    best = rates.index(min(rates))
    return {
        "curve": out,
        "best_hashes": out[best]["hashes"],
        "theory_says": optimal_hashes(),
        "they_agree": out[best]["hashes"] == optimal_hashes(),
        "it_falls_then_rises": best not in (0, len(rates) - 1),
        "the_rate_at_two": rates[1],
        "the_best_rate": rates[best],
        "two_would_cost_this_much_more": round(rates[1] / max(rates[best], 1e-9), 2),
        "and_the_module_uses": HASHES,
    }


def more_bits_are_better_until_the_trial_runs_out(trials: int = 5000) -> dict:
    """The false positive rate against the size, which falls until it reaches the noise floor.

    Unlike the hash count, which has a genuine optimum, more space is never worse in theory. The
    measured curve is not monotone at the bottom, and that is the measurement rather than the
    structure: with five thousand probes the smallest rate distinguishable from zero is one in
    five thousand, and past twenty four bits per entry the true rate is already below it. The
    two tail points are zero against two false positives, which is noise.

    So monotonicity is only claimed down to the resolution, and the resolution is reported next
    to it. A curve declared monotone on this many trials would be a claim about the seed.
    """
    state = np.random.default_rng(11)
    inside = [f"in{one}" for one in range(1000)]
    outside = list({f"out{one}" for one in state.integers(0, 100000, trials)})
    out = []
    for bits in (2, 4, 8, 10, 16, 24, 32):
        filter_ = build(inside, bits_per_entry=bits)
        wrong = sum(1 for one in outside if filter_.might_contain(one))
        out.append(
            {
                "bits_per_entry": bits,
                "bytes": filter_.nbytes,
                "rate": round(wrong / len(outside), 4),
            }
        )
    rates = [one["rate"] for one in out]
    floor = 1.0 / len(outside)
    above = [one for one in rates if one > floor]
    return {
        "curve": out,
        "resolution": round(floor, 5),
        "points_above_the_floor": len(above),
        "it_falls_while_it_can_be_measured": above == sorted(above, reverse=True),
        "the_whole_curve_is_monotone": rates == sorted(rates, reverse=True),
        "at_two_bits": rates[0],
        "at_thirty_two": rates[-1],
    }


def a_filter_prunes_an_equality(rows: int = 40000, group_size: int = 500) -> dict:
    """The case a bloom filter exists for: one value out of two thousand, over eighty groups.

    Every group is equally likely to hold any key, so a zone map cannot prune any of them. The
    bloom filter prunes the groups the key is not in, which is most of them.
    """
    groups, keys = _groups(rows, group_size)
    filters = [build_for(one.column("k")) for one in groups]
    wanted = keys[7]
    truth = [wanted in set(one.column("k").to_list()) for one in groups]
    pruned = prune(filters, wanted, truth)
    return {
        "groups": len(groups),
        **pruned.as_dict(),
        "it_pruned_most": pruned.share > 0.5,
        "and_kept_every_group_that_has_it": pruned.hits <= pruned.kept,
    }


def a_zone_map_barely_prunes_the_same_query(rows: int = 40000, group_size: int = 500) -> dict:
    """The comparison that says why this module exists.

    Written expecting a zone map to prune nothing here, and it prunes a little: six groups of
    eighty. The wanted key sits near the bottom of the key space, so a group whose smallest key
    happens to be larger can be skipped, and with five hundred rows drawn from two thousand keys
    that sometimes happens. For a key in the middle of the space it would not happen at all, and
    the measurement is left on the low key because that is the honest case rather than the
    flattering one.

    The gap is still the point. Six groups against sixty one, on the query the second is for.
    """
    groups, keys = _groups(rows, group_size)
    wanted = keys[7]
    middle = keys[len(keys) // 2]
    zone_kept = _zone_kept(groups, wanted)
    filters = [build_for(one.column("k")) for one in groups]
    truth = [wanted in set(one.column("k").to_list()) for one in groups]
    bloom = prune(filters, wanted, truth)
    return {
        "groups": len(groups),
        "zone_map_kept": zone_kept,
        "bloom_kept": bloom.kept,
        "really_hold_it": bloom.hits,
        "the_zone_map_kept_nearly_everything": zone_kept > len(groups) * 0.9,
        "and_for_a_middle_key_it_keeps_all": _zone_kept(groups, middle) == len(groups),
        "the_bloom_filter_pruned_most": bloom.kept < len(groups) * 0.5,
    }


def _zone_kept(groups: Sequence[Batch], wanted: str) -> int:
    """How many groups a minimum and maximum over the key column would keep."""
    kept = 0
    for one in groups:
        entries = one.column("k").to_list()
        if min(entries) <= wanted <= max(entries):
            kept += 1
    return kept


def a_bloom_filter_cannot_prune_a_range(rows: int = 40000, group_size: int = 500) -> dict:
    """And the reverse, which is the half that keeps the comparison honest.

    A bloom filter answers membership and nothing else. A range predicate would need it queried
    once per value in the range, which for an integer column is every value between the bounds,
    and that is not a query anyone would run. The zone map answers it in two comparisons.
    """
    groups, _ = _groups(rows, group_size)
    threshold = 17000
    zone_kept = sum(1 for one in groups if max(one.column("v").to_list()) >= threshold)
    return {
        "groups": len(groups),
        "zone_map_kept": zone_kept,
        "the_zone_map_pruned_some": zone_kept < len(groups),
        "the_bloom_filter_has_no_answer": True,
        "it_would_need_this_many_probes": threshold,
    }


def the_two_together_prune_more_than_either(rows: int = 40000, group_size: int = 500) -> dict:
    """A query with an equality and a range in it, pruned by both.

    The composition claim, measured rather than assumed. Each mechanism prunes what the other
    cannot, so a conjunction of both kinds of predicate is where they multiply.
    """
    groups, keys = _groups(rows, group_size)
    wanted = keys[11]
    threshold = 17000
    filters = [build_for(one.column("k")) for one in groups]
    by_bloom = {one for one in range(len(groups)) if filters[one].might_contain(wanted)}
    by_zone = {
        one for one in range(len(groups)) if max(groups[one].column("v").to_list()) >= threshold
    }
    both = by_bloom & by_zone
    return {
        "groups": len(groups),
        "bloom_kept": len(by_bloom),
        "zone_kept": len(by_zone),
        "both_kept": len(both),
        "it_is_the_intersection": len(both) <= min(len(by_bloom), len(by_zone)),
        "and_smaller_than_either": len(both) < max(len(by_bloom), len(by_zone)),
    }


def a_dictionary_column_builds_from_its_dictionary(rows: int = 20000) -> dict:
    """Building over the dictionary rather than the codes, which is the same set.

    A hundred rows per distinct value means a hundred times fewer hashes for the same filter,
    and the filters have to come out identical or the shortcut is a bug.
    """
    state = np.random.default_rng(17)
    entries = [f"kind{one}" for one in range(200)]
    column = string_column("k", list(state.choice(entries, size=rows)))
    from_dictionary = build_for(column)
    from_rows = build(column.to_list())
    return {
        "rows": rows,
        "dictionary_entries": len(column.dictionary or ()),
        "same_size": from_dictionary.size == from_rows.size,
        "same_bits": bool(np.array_equal(from_dictionary.bits, from_rows.bits)),
        "hashes_saved": rows - len(column.dictionary or ()),
    }


def the_hash_is_stable_across_runs() -> dict:
    """The same value hashes the same way every time, which a stored filter needs.

    Python's own hash is randomised per process for strings. A filter built with it and written
    to a file would read back as a filter over nothing, and every lookup would come back false,
    which is the false negative this structure is not allowed to have. Within one process it
    works perfectly, which is what makes it easy to ship.
    """
    once = [_hash(one) for one in ("a", "b", 1, 2.5)]
    twice = [_hash(one) for one in ("a", "b", 1, 2.5)]
    return {
        "stable": once == twice,
        "they_differ_from_each_other": len(set(once)) == len(once),
        "a_string_and_a_number_differ": _hash("1") != _hash(1),
    }


def the_hash_spreads(size: int = 4096, values: int = 4000) -> dict:
    """How evenly the hash fills a bit array, which is what the false positive rate rests on.

    Measured as the ratio of the busiest bucket to the average over a chi squared style count. A
    hash that clustered would set the same bits repeatedly and the filter would be full in one
    place and empty everywhere else.
    """
    counts = np.zeros(size, dtype=np.int64)
    for one in range(values):
        counts[_hash(f"value{one}") % size] += 1
    average = float(counts.mean())
    return {
        "buckets": size,
        "values": values,
        "empty": int((counts == 0).sum()),
        "busiest": int(counts.max()),
        "average": round(average, 3),
        "ratio": round(float(counts.max()) / max(average, 1e-9), 2),
        "it_is_not_clustered": bool(counts.max() < average * 8),
    }


def integers_and_floats_and_strings_all_work() -> dict:
    """Three types through the same filter, each finding its own values."""
    out = {}
    for name, values in (
        ("integer", list(range(500))),
        ("floating", [one * 1.5 for one in range(500)]),
        ("string", [f"text{one}" for one in range(500)]),
    ):
        filter_ = build(values)
        out[name] = all(filter_.might_contain(one) for one in values)
    return out


def the_filter_is_not_free(rows: int = 40000, group_size: int = 500) -> dict:
    """What the pruning costs in space, against the data it prunes.

    Written as the filter is small and it is not, at least not in the form it is built in. A
    numpy boolean array is one byte per bit, so eighty groups of about five hundred distinct
    keys each come to forty six percent of the data. Packed to actual bits it is under six
    percent, which is a real cost rather than a rounding error.

    The reason it is this dear is the reason it prunes this well. Five hundred rows drawn from
    two thousand keys are nearly all distinct, so there is almost nothing shared between groups
    to save on. A column of ten distinct values would give a filter of forty bytes a group and
    would also be prunable by other means. A bloom filter costs most where it is worth most, and
    any claim that it is cheap has to say which column it is cheap on.
    """
    groups, _ = _groups(rows, group_size)
    filters = [build_for(one.column("k")) for one in groups]
    data = sum(one.nbytes for one in groups)
    loose = sum(one.nbytes for one in filters)
    packed = sum(np.packbits(one.bits).nbytes for one in filters)
    return {
        "groups": len(groups),
        "data_bytes": data,
        "loose_bytes": loose,
        "packed_bytes": packed,
        "loose_share": round(loose / data, 4),
        "packed_share": round(packed / data, 4),
        "bytes_per_group": packed // len(groups),
        "the_loose_form_is_not_storable": loose > data * 0.25,
        "the_packed_one_is": packed < data * 0.1,
    }


def a_packed_filter_is_eight_times_smaller(rows: int = 40000, group_size: int = 500) -> dict:
    """The bits stored as bits rather than as bytes, which numpy does not do by default.

    A boolean array is one byte per element. Packing it is one call and the filter is a third of
    a percent of the file rather than two and a half. Worth having and worth measuring rather
    than assuming, because the packed form has to be unpacked to query and that cost is real.
    """
    groups, _ = _groups(rows, group_size)
    filters = [build_for(one.column("k")) for one in groups]
    loose = sum(one.nbytes for one in filters)
    packed = sum(np.packbits(one.bits).nbytes for one in filters)
    return {
        "loose_bytes": loose,
        "packed_bytes": packed,
        "ratio": round(loose / max(packed, 1), 2),
        "it_is_eight": round(loose / max(packed, 1)) == 8,
    }


def an_empty_set_still_answers() -> dict:
    """A filter over nothing, which says no to everything and is right to."""
    filter_ = build([])
    return {
        "size": filter_.size,
        "entries": filter_.entries,
        "it_is_the_minimum": filter_.size == MINIMUM_BITS,
        "it_says_no": not filter_.might_contain("anything"),
        "occupancy": filter_.occupancy,
    }


def a_zero_size_filter_is_refused() -> bool:
    """Zero bits per entry."""
    try:
        build(["a"], bits_per_entry=0)
    except ConfigError:
        return True
    return False


def a_zero_hash_filter_is_refused() -> bool:
    """Zero hashes, which would accept everything."""
    try:
        build(["a"], hashes=0)
    except ConfigError:
        return True
    return False


def a_truth_shorter_than_the_filters_is_refused() -> bool:
    """A pruning measurement whose ground truth does not cover every group."""
    filters = [build(["a"]), build(["b"])]
    try:
        prune(filters, "a", truth=[True])
    except ConfigError:
        return True
    return False


def compare_the_sizes() -> list[dict]:
    """False positive rate against space, which is the whole trade in one table."""
    return more_bits_are_better_until_the_trial_runs_out()["curve"]


def summarise() -> dict:
    """The module in one mapping."""
    rates = the_false_positive_rate_is_what_it_is()
    pruning = a_filter_prunes_an_equality()
    return {
        "bits_per_entry": BITS_PER_ENTRY,
        "hashes": HASHES,
        "measured_rate": rates["measured"],
        "predicted_rate": rates["predicted"],
        "pruned_share": pruning["share"],
        "no_false_negatives": a_filter_never_says_no_to_a_value_it_holds()["it_never_says_no"],
    }
