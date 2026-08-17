from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.encode.bitpack import bits_needed
from cqe.errors import ConfigError, EncodingError

# Delta encoding, which stores the difference between consecutive values instead of the values.
#
# It is the only encoding here that depends on the order of the rows rather than on their
# contents, and that is the whole of it. A sorted column has small non negative differences
# and delta encodes to almost nothing. The same values shuffled have differences spread over
# the whole range.
#
# The exact cost of shuffling is one bit per value and the arithmetic says why. Values in
# [0, 2^k) need k bits; differences between such values live in (-2^k, 2^k) and need k plus
# one. So delta encoding a column with no order structure is not neutral against bit packing,
# it is worse by exactly that bit, measured at a ratio of 1.0625 on a sixteen bit column.
#
# It is still better than storing int64, which is worth being clear about because I first
# wrote that an unordered column comes out larger than the original. It does not: at a span of
# 2^20 the shuffled form is 0.328 of int64 and the sorted form is 0.109, a factor of three
# between them. Delta never loses against unencoded storage. It loses against the encoding it
# would otherwise have been given.
#
# There is a second form, delta of delta, which stores the difference of the differences. It is
# for sequences with a roughly constant step: timestamps at a fixed interval, identifiers
# allocated in order, dates in a generated calendar. On those the first differences are all
# nearly equal and the second differences are nearly zero, so the column packs to one or two
# bits. On anything else it adds another bit for the same reason the first delta did, so it is
# strictly worse than delta unless the step really is regular.
#
# The engine applies delta only where the column is known to be sorted, which the writer knows
# because it sorted it. Guessing from a sample is possible and is not done here: a column that
# is sorted in the sample and not in the rest is a column whose encoded size doubles at write
# time, and columns/encode/choose.py measures how often a sample would be wrong.


@dataclass
class Deltas:
    """A delta encoded column: the first value and the differences after it."""

    first: int
    differences: np.ndarray
    order: int

    def __post_init__(self) -> None:
        if self.order < 1:
            raise EncodingError(f"{self.order} is not a delta order")
        if self.differences.ndim != 1:
            raise EncodingError(f"differences are one dimensional, not {self.differences.ndim}")

    @property
    def rows(self) -> int:
        """How many values the encoding covers."""
        return int(self.differences.shape[0]) + 1

    @property
    def span(self) -> int:
        """The range the differences cover, which sets the packed width."""
        if not len(self.differences):
            return 0
        return int(self.differences.max()) - int(self.differences.min())

    @property
    def bits(self) -> int:
        """Bits per difference once packed."""
        return bits_needed(self.span)

    @property
    def nbytes(self) -> int:
        """Bytes the packed differences occupy, plus the first value."""
        return (len(self.differences) * self.bits + 7) // 8 + 8

    def ratio(self, source_width: int = 8) -> float:
        """Encoded size over raw size, so below one is a saving."""
        if self.rows == 0:
            return 1.0
        return self.nbytes / (self.rows * source_width)

    @property
    def monotone(self) -> bool:
        """Whether every difference is non negative, which is what sorted means."""
        if not len(self.differences):
            return True
        return bool((self.differences >= 0).all())

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rows": self.rows,
            "order": self.order,
            "span": self.span,
            "bits": self.bits,
            "bytes": self.nbytes,
            "ratio": round(self.ratio(), 4),
            "monotone": self.monotone,
        }


def encode(values: Sequence | np.ndarray, order: int = 1) -> Deltas:
    """Take differences of the given order.

    Order one is delta, order two is delta of delta. Higher orders are allowed and are never
    useful on real data, which the module measures rather than asserting: a third difference is
    another bit again and there is no common sequence whose second differences are constant but
    whose first ones are not.
    """
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise EncodingError(f"a column is one dimensional, not {array.ndim}")
    if order < 1:
        raise EncodingError(f"{order} is not a delta order")
    if len(array) <= order:
        raise EncodingError(f"a column of {len(array)} cannot take {order} differences")
    working = array
    for _ in range(order):
        working = np.diff(working)
    return Deltas(first=int(array[0]), differences=working, order=order)


def decode(deltas: Deltas, tail: Sequence[int] = ()) -> np.ndarray:
    """Recover the original values.

    An order two encoding needs the first two values to rebuild, not one, which is what tail
    carries. Making the caller supply it rather than storing it inside the record is deliberate:
    it is the smallest possible reminder that a higher order encoding has more state, and
    forgetting that is how a decoder silently drifts by a constant.
    """
    working = deltas.differences
    seeds = [deltas.first, *list(tail)]
    if len(seeds) != deltas.order:
        raise EncodingError(f"order {deltas.order} needs {deltas.order} seed values")
    for seed in reversed(seeds):
        working = np.concatenate([[seed], np.cumsum(working) + seed])[: len(working) + 1]
    return working.astype(np.int64)


def sorted_ids(rows: int, gap: int = 3, seed: int = 0) -> np.ndarray:
    """Identifiers allocated in order with small irregular gaps, which is the common case."""
    if rows < 2 or gap < 1:
        raise ConfigError(f"{rows} rows with gaps of {gap} is not a column")
    steps = np.random.default_rng(seed).integers(1, gap + 1, size=rows - 1)
    return np.concatenate([[1_000_000], 1_000_000 + np.cumsum(steps)]).astype(np.int64)


def regular_timestamps(rows: int, step: int = 60, jitter: int = 0, seed: int = 0) -> np.ndarray:
    """Timestamps at a fixed interval, optionally jittered, which is the delta of delta case."""
    if rows < 3 or step < 1:
        raise ConfigError(f"{rows} rows at a step of {step} is not a series")
    base = np.arange(rows, dtype=np.int64) * step + 1_700_000_000
    if jitter:
        base = base + np.random.default_rng(seed).integers(-jitter, jitter + 1, size=rows)
    return base.astype(np.int64)


def shuffled(rows: int, span: int, seed: int = 0) -> np.ndarray:
    """Values over a span with no order structure, which is the case delta loses on."""
    if rows < 2 or span < 1:
        raise ConfigError(f"{rows} rows over a span of {span} is not a column")
    return np.random.default_rng(seed).integers(0, span, size=rows).astype(np.int64)


def a_sorted_column_deltas_to_almost_nothing(rows: int = 100_000) -> dict:
    """The case delta encoding is for.

    Identifiers allocated in order with gaps of one to three. The values span two hundred
    thousand and need eighteen bits; the differences span two and need two, for a ratio of
    0.031. The saving is the ratio of those widths and does not depend on the height at all.
    """
    values = sorted_ids(rows)
    deltas = encode(values)
    return {
        "value_span": int(values.max() - values.min()),
        "value_bits": bits_needed(int(values.max() - values.min())),
        "delta_span": deltas.span,
        "delta_bits": deltas.bits,
        "ratio": round(deltas.ratio(), 5),
        "monotone": deltas.monotone,
        "it_is_a_large_saving": deltas.ratio() < 0.1,
    }


def shuffling_the_same_values_costs_a_bit(rows: int = 100_000, span: int = 1 << 16) -> dict:
    """Delta encoding is not neutral on unordered data, it is one bit per value worse.

    Values in a range of n need k bits. Differences between two such values live in the range
    minus n to n, which needs k plus one. So a column with no order structure comes out larger
    than plain bit packing by exactly that bit, and the ratio against packing is 1 plus one over
    k.

    Worth measuring because delta encoding is often described as harmless when it does not help.
    It is not harmless, and on a sixteen bit column it costs six percent.
    """
    values = shuffled(rows, span)
    deltas = encode(values)
    packed_bits = bits_needed(int(values.max() - values.min()))
    return {
        "packed_bits": packed_bits,
        "delta_bits": deltas.bits,
        "extra_bits": deltas.bits - packed_bits,
        "it_costs_exactly_one_bit": deltas.bits - packed_bits == 1,
        "ratio_against_packing": round(deltas.bits / packed_bits, 4),
        "delta_ratio": round(deltas.ratio(), 4),
    }


def the_order_of_the_rows_is_the_whole_thing(rows: int = 100_000) -> dict:
    """The same values sorted and shuffled, which is the module's headline.

    Identical multiset, identical cardinality, identical everything a statistic would report.
    Sorted the column needs seven bits per difference and encodes to 0.109; shuffled it needs
    twenty one and encodes to 0.328. A factor of three between two columns no summary statistic
    can tell apart, which is why delta is applied from knowledge of the writer rather than from
    a look at the data.

    I expected the shuffled form to come out above one and it does not: even twenty one bits
    beats sixty four. The loss from shuffling shows up against bit packing rather than against
    unencoded storage, which is the previous function.
    """
    values = shuffled(rows, span=1 << 20)
    tidy = encode(np.sort(values))
    messy = encode(values)
    return {
        "sorted_bits": tidy.bits,
        "shuffled_bits": messy.bits,
        "sorted_ratio": round(tidy.ratio(), 5),
        "shuffled_ratio": round(messy.ratio(), 4),
        "sorted_wins": tidy.ratio() < 0.2,
        "shuffled_still_beats_int64": messy.ratio() < 1.0,
        "the_gap_is_about_threefold": messy.ratio() > 2.5 * tidy.ratio(),
        "same_values": True,
    }


def delta_of_delta_is_for_regular_steps(rows: int = 100_000) -> dict:
    """Timestamps at a fixed interval, where the second differences are all zero.

    The first differences are all sixty and the second differences are all zero, so both spans
    are zero, both forms need one bit, and both come to a ratio of 0.0156.

    Delta of delta buys nothing here, which is the opposite of how it is usually described. It
    only pays when the step itself drifts, and the next function measures what it costs when the
    step is merely noisy rather than drifting.
    """
    values = regular_timestamps(rows, step=60)
    first = encode(values, order=1)
    second = encode(values, order=2)
    return {
        "first_span": first.span,
        "second_span": second.span,
        "first_bits": first.bits,
        "second_bits": second.bits,
        "first_ratio": round(first.ratio(), 5),
        "second_ratio": round(second.ratio(), 5),
        "the_second_span_is_zero": second.span == 0,
        "they_are_level_here": abs(first.ratio() - second.ratio()) < 0.001,
    }


def and_it_costs_a_bit_everywhere_else(rows: int = 100_000) -> dict:
    """On an irregular sequence the second order form is strictly worse than the first.

    Jittered timestamps. The first differences vary by the jitter and need five bits, the second
    vary by twice it and need six, the third need seven. Every extra order adds a bit for the
    same reason the first one did.

    So delta of delta is a bet on the step being exactly regular rather than a general
    improvement, and a jitter of five seconds is enough to lose the bet.
    """
    values = regular_timestamps(rows, step=60, jitter=5)
    first = encode(values, order=1)
    second = encode(values, order=2)
    third = encode(values, order=3)
    return {
        "first_bits": first.bits,
        "second_bits": second.bits,
        "third_bits": third.bits,
        "each_order_costs_about_a_bit": second.bits > first.bits and third.bits > second.bits,
        "first_ratio": round(first.ratio(), 5),
        "second_ratio": round(second.ratio(), 5),
        "the_first_order_wins": first.ratio() < second.ratio(),
    }


def the_gap_size_sets_the_width(
    rows: int = 100_000,
    gaps: Sequence[int] = (1, 2, 8, 64, 1024),
) -> list[dict]:
    """How much the irregularity of a sorted sequence costs.

    Gaps of one or two delta to one bit, gaps up to sixty four to six, gaps up to a thousand to
    ten. The width is the span of the gaps and nothing else, so a sorted column with occasional
    large jumps pays for the largest jump on every row, exactly as bit packing pays for its
    outlier.
    """
    if not gaps:
        raise ConfigError("there is nothing to sweep")
    out = []
    for gap in gaps:
        deltas = encode(sorted_ids(rows, gap=gap))
        out.append(
            {
                "gap": gap,
                "delta_bits": deltas.bits,
                "ratio": round(deltas.ratio(), 5),
            }
        )
    return out


def one_jump_sets_the_width(rows: int = 100_000) -> dict:
    """The same outlier sensitivity bit packing has, arriving through a different door.

    A sorted column of identifiers with one gap of a million in it. Every other difference is
    one or two and needs a single bit; the width is set by the million and comes to twenty,
    taking the ratio from 0.016 to 0.313. The encoding has no way to charge that one row for
    it, which is the same failure bit packing has and arrives here through the differences
    rather than through the values.
    """
    values = sorted_ids(rows, gap=2)
    spoiled = values.copy()
    spoiled[rows // 2 :] += 1_000_000
    clean = encode(values)
    dirty = encode(spoiled)
    return {
        "clean_bits": clean.bits,
        "spoiled_bits": dirty.bits,
        "clean_ratio": round(clean.ratio(), 5),
        "spoiled_ratio": round(dirty.ratio(), 5),
        "one_jump_in": rows,
        "the_width_rose": dirty.bits > clean.bits,
        "and_by_a_lot": dirty.bits > 4 * clean.bits,
    }


def the_round_trip_is_exact(rows: int = 20_000) -> dict:
    """Decoding gives back exactly what was encoded, at order one.

    Order two and above are not round tripped here because they need the first two values and
    this module deliberately makes the caller carry them, so the check lives with the caller.
    """
    cases = {
        "sorted ids": sorted_ids(rows),
        "timestamps": regular_timestamps(rows),
        "shuffled": shuffled(rows, span=1 << 16),
    }
    return {
        name: bool(np.array_equal(decode(encode(column)), column))
        for name, column in cases.items()
    }


def a_constant_column_deltas_to_zero(rows: int = 10_000) -> dict:
    """A column of one repeated value, whose differences are all zero."""
    values = np.full(rows, 7, dtype=np.int64)
    deltas = encode(values)
    return {
        "span": deltas.span,
        "bits": deltas.bits,
        "it_is_one_bit": deltas.bits == 1,
        "monotone": deltas.monotone,
        "round_trips": bool(np.array_equal(decode(deltas), values)),
    }


def a_descending_column_is_not_monotone(rows: int = 1_000) -> dict:
    """Sorted descending, which deltas just as well and reports itself correctly.

    Worth a measurement because the size is the same and the monotone flag is not, and the flag
    is what the caller uses to decide whether a range predicate can be answered from the deltas.
    """
    values = np.sort(shuffled(rows, span=1 << 16))[::-1].copy()
    deltas = encode(values)
    return {
        "bits": deltas.bits,
        "monotone": deltas.monotone,
        "ratio": round(deltas.ratio(), 5),
        "it_encodes_as_well_as_ascending": deltas.ratio() < 0.5,
        "round_trips": bool(np.array_equal(decode(deltas), values)),
    }


def a_column_shorter_than_the_order_is_refused() -> bool:
    """Two values cannot take two differences."""
    try:
        encode(np.array([1, 2], dtype=np.int64), order=2)
    except EncodingError:
        return True
    return False


def a_zero_order_is_refused() -> bool:
    """A delta of order zero is the column itself and is not an encoding."""
    try:
        encode(np.arange(10), order=0)
    except EncodingError:
        return True
    return False


def a_two_dimensional_column_is_refused() -> bool:
    """The encoder takes a column, not a table."""
    try:
        encode(np.zeros((3, 3)))
    except EncodingError:
        return True
    return False


def decoding_without_the_seeds_is_refused() -> bool:
    """An order two encoding needs two seed values and says so rather than drifting."""
    deltas = encode(np.arange(10), order=2)
    try:
        decode(deltas)
    except EncodingError:
        return True
    return False


def compare_the_shapes(rows: int = 100_000) -> list[dict]:
    """Every column shape against delta at order one, as one table."""
    shapes = {
        "sorted ids gap 1": sorted_ids(rows, gap=1),
        "sorted ids gap 64": sorted_ids(rows, gap=64),
        "timestamps": regular_timestamps(rows),
        "sorted values": np.sort(shuffled(rows, span=1 << 20)),
        "shuffled": shuffled(rows, span=1 << 20),
    }
    out = []
    for name, column in shapes.items():
        row = encode(column).as_dict()
        row["shape"] = name
        out.append(row)
    return sorted(out, key=lambda row: row["ratio"])


def summarise(rows: int = 100_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    ordering = the_order_of_the_rows_is_the_whole_thing(rows=rows)
    shuffling = shuffling_the_same_values_costs_a_bit(rows=rows)
    regular = delta_of_delta_is_for_regular_steps(rows=rows)
    return {
        "sorted_ratio": ordering["sorted_ratio"],
        "shuffled_ratio": ordering["shuffled_ratio"],
        "extra_bits_when_unordered": shuffling["extra_bits"],
        "second_order_span_on_regular_steps": regular["second_span"],
        "best_shape": compare_the_shapes(rows=rows)[0]["shape"],
        "worst_shape": compare_the_shapes(rows=rows)[-1]["shape"],
    }
