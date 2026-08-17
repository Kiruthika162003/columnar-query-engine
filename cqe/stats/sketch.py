from __future__ import annotations

import statistics as pystats
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cqe.errors import ConfigError, DataError

# Counting distinct values without holding them, which is the other number a planner needs.
#
# A histogram says how many rows a range keeps. A distinct count says how many groups a group by
# will produce, how many buckets a hash table needs, and how far apart two join keys are.
# Getting it exactly means a hash set holding every distinct value, which is the memory the
# whole exercise is trying to avoid.
#
# Three estimators are here.
#
# A sample counts the distinct values in a fraction of the rows and scales up. It is the obvious
# thing to do and it is badly wrong in a way worth measuring: distinct count does not scale with
# the sample. A column with a million distinct values in a million rows has a thousand distinct
# values in a thousand row sample, and scaling that by a thousand gives a million by luck. A
# column with a thousand distinct values has a thousand in the sample too, and scaling gives a
# million again. The estimator cannot tell those apart.
#
# A linear counting sketch hashes each value into a bit array and estimates from how many bits
# stayed clear. It is exact for small cardinalities and saturates: once every bit is set it can
# only say the answer is large. That makes it right where sampling is wrong and useless where
# sampling is merely poor.
#
# A HyperLogLog keeps the maximum leading zero count per bucket, which is a statement about the
# largest hash it saw rather than about how many it saw, so it does not saturate. It costs one
# byte per bucket, it is accurate to about one over the root of the bucket count, and it merges:
# two sketches of two row groups combine into a sketch of both without reading either.
#
# That last property is the one that decides it. Everything else in this package can be computed
# per row group and summed. A distinct count cannot be summed, and a mergeable sketch is the
# only way a reader gets a table wide number without a pass over the table.

MASK = (1 << 64) - 1


@dataclass
class Sample:
    """A distinct count estimated by scaling up a sample."""

    seen: int
    rows: int
    sampled: int

    def __post_init__(self) -> None:
        if self.sampled < 0 or self.rows < 0:
            raise ConfigError(f"{self.sampled} of {self.rows} is not a sample")
        if self.sampled > self.rows:
            raise ConfigError(f"{self.sampled} sampled from {self.rows} rows")

    @property
    def rate(self) -> float:
        """The share of rows looked at."""
        if self.rows == 0:
            return 0.0
        return self.sampled / self.rows

    def estimate(self) -> float:
        """Scale the distinct count seen by the sampling rate.

        The naive estimator, and it is naive in a specific way: distinct count is not additive
        over rows, so scaling it is not a correction, it is a guess that happens to be right
        when every value appears exactly once and wrong otherwise.
        """
        if self.rate == 0:
            return 0.0
        return self.seen / self.rate

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kind": "sample",
            "seen": self.seen,
            "sampled": self.sampled,
            "rate": round(self.rate, 4),
            "estimate": round(self.estimate(), 1),
        }


@dataclass
class LinearCounter:
    """A bit array counted by how much of it stayed clear."""

    bits: int
    array: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __post_init__(self) -> None:
        if self.bits < 8:
            raise ConfigError(f"{self.bits} bits is not a counter")
        if not len(self.array):
            self.array = np.zeros(self.bits, dtype=bool)
        if len(self.array) != self.bits:
            raise DataError(f"{len(self.array)} bits against {self.bits}")

    @property
    def clear(self) -> int:
        """How many bits are still zero, which is what the estimate is built from."""
        return int((~self.array).sum())

    @property
    def saturated(self) -> bool:
        """Whether every bit is set, at which point the estimate is only a lower bound."""
        return self.clear == 0

    def add(self, hashes: np.ndarray) -> None:
        """Set the bit each hash lands on."""
        self.array[hashes % self.bits] = True

    def estimate(self) -> float:
        """Recover the cardinality from the share of bits left clear.

        The estimator is m times the negative log of the clear share, which inverts the
        probability that a bit stayed clear after n insertions. It is very accurate below about
        m over ten and it goes to infinity as the array fills, which is why saturation is
        reported rather than hidden.
        """
        if self.saturated:
            return float("inf")
        return -self.bits * float(np.log(self.clear / self.bits))

    def nbytes(self) -> int:
        """What the counter costs, at one bit per slot."""
        return self.bits // 8

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kind": "linear",
            "bits": self.bits,
            "clear": self.clear,
            "saturated": self.saturated,
            "estimate": None if self.saturated else round(self.estimate(), 1),
            "bytes": self.nbytes(),
        }


@dataclass
class HyperLogLog:
    """A mergeable distinct count sketch, one byte per bucket."""

    precision: int = 12
    registers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))

    def __post_init__(self) -> None:
        if not 4 <= self.precision <= 18:
            raise ConfigError(f"a precision of {self.precision} is outside four to eighteen")
        if not len(self.registers):
            self.registers = np.zeros(self.buckets, dtype=np.uint8)
        if len(self.registers) != self.buckets:
            raise DataError(f"{len(self.registers)} registers against {self.buckets}")

    @property
    def buckets(self) -> int:
        """How many registers there are, which is two to the precision."""
        return 1 << self.precision

    @property
    def nbytes(self) -> int:
        """One byte per register."""
        return self.buckets

    @property
    def expected_error(self) -> float:
        """The standard error the bucket count implies, which is 1.04 over its root."""
        return 1.04 / float(np.sqrt(self.buckets))

    def add(self, hashes: np.ndarray) -> None:
        """Fold a batch of hashes in.

        The top bits pick the register and the rest supply the leading zero count. Keeping the
        maximum rather than a sum is what makes the sketch idempotent: adding the same value
        twice changes nothing, which is the whole reason it counts distinct values rather than
        rows.
        """
        if not len(hashes):
            return
        values = np.asarray(hashes, dtype=np.uint64)
        index = (values >> np.uint64(64 - self.precision)).astype(np.int64)
        remainder = (values << np.uint64(self.precision)) | np.uint64(1 << (self.precision - 1))
        ranks = _leading_zeros(remainder) + 1
        np.maximum.at(self.registers, index, ranks.astype(np.uint8))

    def estimate(self) -> float:
        """The harmonic mean estimator, with the small range correction.

        The raw estimator is biased low for small cardinalities, badly enough that a sketch with
        four thousand buckets reports about seven hundred for an input of ten. The linear
        counting correction takes over below two and a half times the bucket count, which is the
        standard fix and is the only place this implementation is not the obvious thing.
        """
        counts = self.registers.astype(np.float64)
        harmonic = float((2.0 ** (-counts)).sum())
        alpha = 0.7213 / (1.0 + 1.079 / self.buckets)
        raw = alpha * self.buckets * self.buckets / harmonic
        empty = int((self.registers == 0).sum())
        if raw <= 2.5 * self.buckets and empty > 0:
            return self.buckets * float(np.log(self.buckets / empty))
        return raw

    def merge(self, other: HyperLogLog) -> HyperLogLog:
        """Combine two sketches into one covering both inputs.

        The register wise maximum, which is exact: the maximum leading zero count over a union
        is the maximum of the maxima. No other estimator here has this property and it is the
        reason the engine stores a sketch per row group rather than one per table.
        """
        if other.precision != self.precision:
            raise ConfigError(f"precision {self.precision} against {other.precision}")
        return HyperLogLog(
            precision=self.precision,
            registers=np.maximum(self.registers, other.registers),
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kind": "hyperloglog",
            "precision": self.precision,
            "buckets": self.buckets,
            "bytes": self.nbytes,
            "expected_error": round(self.expected_error, 4),
            "estimate": round(self.estimate(), 1),
        }


def _leading_zeros(values: np.ndarray) -> np.ndarray:
    """How many leading zero bits each sixty four bit value has."""
    out = np.zeros(len(values), dtype=np.int64)
    working = np.asarray(values, dtype=np.uint64).copy()
    for shift in (32, 16, 8, 4, 2, 1):
        step = np.uint64(shift)
        limit = np.uint64((1 << (64 - shift)) - 1)
        movable = working <= limit
        out[movable] += shift
        working[movable] = working[movable] << step
    return out


def hashed(values: Sequence | np.ndarray, seed: int = 0) -> np.ndarray:
    """Sixty four bit hashes of a column's values.

    A multiplicative hash over the integer form of the value, which is enough for the estimators
    here and is not a cryptographic claim. What matters is that it spreads: a hash that clusters
    makes a HyperLogLog report the cardinality of its own collisions.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise DataError(f"a column is one dimensional, not {array.ndim}")
    if array.dtype.kind in "US" or array.dtype == object:
        array = np.array([abs(hash(str(one))) for one in array], dtype=np.uint64)
    else:
        array = array.astype(np.int64).astype(np.uint64)
    working = (array + np.uint64(seed * 0x9E3779B97F4A7C15)) & np.uint64(MASK)
    working ^= working >> np.uint64(33)
    working = (working * np.uint64(0xFF51AFD7ED558CCD)) & np.uint64(MASK)
    working ^= working >> np.uint64(29)
    working = (working * np.uint64(0xC4CEB9FE1A85EC53)) & np.uint64(MASK)
    working ^= working >> np.uint64(32)
    return working


def sketch_of(values: Sequence | np.ndarray, precision: int = 12) -> HyperLogLog:
    """Build a sketch over a column in one pass."""
    out = HyperLogLog(precision=precision)
    out.add(hashed(values))
    return out


def linear_of(values: Sequence | np.ndarray, bits: int = 4096) -> LinearCounter:
    """Build a linear counter over a column in one pass."""
    out = LinearCounter(bits=bits)
    out.add(hashed(values))
    return out


def sample_of(values: Sequence | np.ndarray, rate: float = 0.01, seed: int = 0) -> Sample:
    """Take a sample of a column and count the distinct values in it."""
    if not 0.0 < rate <= 1.0:
        raise ConfigError(f"{rate} is not a sampling rate")
    array = np.asarray(values)
    count = max(1, int(len(array) * rate))
    picked = np.random.default_rng(seed).choice(len(array), size=count, replace=False)
    return Sample(seen=len(np.unique(array[picked])), rows=len(array), sampled=count)


def exact(values: Sequence | np.ndarray) -> int:
    """The true distinct count, which every estimate here is scored against."""
    return len(np.unique(np.asarray(values)))


def error(estimate: float, truth: float) -> float:
    """Relative error, as a ratio rather than a difference."""
    if truth == 0:
        return 0.0 if estimate == 0 else float("inf")
    if estimate == float("inf"):
        return float("inf")
    return abs(estimate - truth) / truth


def column(rows: int, distinct: int, seed: int = 0) -> np.ndarray:
    """A column of the given height with exactly the given number of distinct values."""
    if rows < 1 or distinct < 1 or distinct > rows:
        raise ConfigError(f"{distinct} distinct values do not fit in {rows} rows")
    generator = np.random.default_rng(seed)
    extra = generator.integers(0, distinct, size=rows - distinct)
    values = np.concatenate([np.arange(distinct), extra]).astype(np.int64)
    generator.shuffle(values)
    return values


def sampling_cannot_tell_the_two_apart(rows: int = 200_000) -> dict:
    """The measurement that rules out the obvious estimator.

    Two columns of the same height, one with every value distinct and one with a thousand. A one
    percent sample of the first holds two thousand distinct values and a one percent sample of
    the second holds close to a thousand, and scaling either by a hundred gives an answer that
    is wrong for at least one of them by a large factor.
    """
    unique = column(rows, rows)
    repeated = column(rows, 1_000)
    one = sample_of(unique, rate=0.01)
    other = sample_of(repeated, rate=0.01)
    return {
        "unique_truth": exact(unique),
        "unique_estimate": round(one.estimate(), 1),
        "unique_error": round(error(one.estimate(), exact(unique)), 4),
        "repeated_truth": exact(repeated),
        "repeated_estimate": round(other.estimate(), 1),
        "repeated_error": round(error(other.estimate(), exact(repeated)), 4),
        "one_of_them_is_badly_wrong": max(
            error(one.estimate(), exact(unique)), error(other.estimate(), exact(repeated))
        )
        > 1.0,
    }


def a_bigger_sample_does_not_fix_it(
    rows: int = 200_000,
    rates: Sequence[float] = (0.01, 0.05, 0.2, 0.5),
) -> list[dict]:
    """How the sampling error moves as the sample grows, on a repeated column.

    Slowly, and only because the sample starts to contain everything. A sample is a good
    estimator of a mean and a bad one of a maximum, and a distinct count is much closer to a
    maximum than to a mean.
    """
    if not rates:
        raise ConfigError("there is nothing to sweep")
    values = column(rows, 1_000)
    truth = exact(values)
    out = []
    for rate in rates:
        taken = sample_of(values, rate=rate)
        out.append(
            {
                "rate": rate,
                "seen": taken.seen,
                "estimate": round(taken.estimate(), 1),
                "error": round(error(taken.estimate(), truth), 4),
            }
        )
    return out


def a_linear_counter_is_exact_until_it_saturates(
    rows: int = 200_000,
    distincts: Sequence[int] = (10, 100, 1_000, 10_000, 100_000),
    bits: int = 4_096,
) -> list[dict]:
    """A bit array is very accurate until it fills, and useless the moment it does.

    Better than I expected on the accurate side. Four thousand bits stay within a percent up to
    ten thousand distinct values, which is two and a half times the bit count, not the tenth of
    it I assumed. The errors are 0.001, 0.008, 0.001 and 0.008 over four orders of magnitude.

    Then at a hundred thousand every bit is set, there is no information left in the array at
    all, and the estimate is infinite. There is no gentle degradation between those two states,
    which is the whole reason a linear counter cannot be the only estimator a writer keeps.
    """
    if not distincts:
        raise ConfigError("there is nothing to sweep")
    out = []
    for distinct in distincts:
        values = column(rows, distinct)
        counter = linear_of(values, bits=bits)
        truth = exact(values)
        out.append(
            {
                "distinct": distinct,
                "clear_bits": counter.clear,
                "saturated": counter.saturated,
                "estimate": None if counter.saturated else round(counter.estimate(), 1),
                "error": round(error(counter.estimate(), truth), 4),
            }
        )
    return out


def hyperloglog_holds_across_the_whole_range(
    rows: int = 200_000,
    distincts: Sequence[int] = (10, 100, 1_000, 10_000, 100_000, 200_000),
    precision: int = 12,
) -> list[dict]:
    """The sketch the engine actually uses, across five orders of magnitude of cardinality.

    The point of comparison with the two above. It does not saturate and it does not depend on
    the cardinality being small relative to anything, so one sketch size covers every column in
    a table without the writer having to know which is which.
    """
    if not distincts:
        raise ConfigError("there is nothing to sweep")
    out = []
    for distinct in distincts:
        values = column(rows, distinct)
        sketch = sketch_of(values, precision=precision)
        truth = exact(values)
        out.append(
            {
                "distinct": distinct,
                "estimate": round(sketch.estimate(), 1),
                "error": round(error(sketch.estimate(), truth), 4),
                "expected_error": round(sketch.expected_error, 4),
            }
        )
    return out


def the_error_follows_the_bucket_count(
    rows: int = 200_000,
    distinct: int = 50_000,
    precisions: Sequence[int] = (6, 8, 10, 12, 14),
) -> list[dict]:
    """The accuracy is one over the root of the bucket count, which sets the size.

    Doubling the precision quadruples the buckets and should halve the error. The measured
    errors are 0.123, 0.017, 0.053, 0.044 and 0.007 against expectations of 0.130, 0.065, 0.033,
    0.016 and 0.008, so the trend is there and the individual points are not on it.

    That is the measurement read correctly rather than a fault. A standard error describes the
    spread over many sketches and one sketch is a single draw from it, so precision eight came
    out four times better than expected and precision twelve two and a half times worse. Sizing
    a sketch on one trial is the mistake the expected error exists to prevent, and both columns
    are reported so the difference is visible.

    A byte per bucket at precision twelve is four kilobytes for
    about a two percent error, which is the size the engine writes per row group.
    """
    if not precisions:
        raise ConfigError("there is nothing to sweep")
    values = column(rows, distinct)
    truth = exact(values)
    out = []
    for precision in precisions:
        sketch = sketch_of(values, precision=precision)
        out.append(
            {
                "precision": precision,
                "buckets": sketch.buckets,
                "bytes": sketch.nbytes,
                "error": round(error(sketch.estimate(), truth), 4),
                "expected_error": round(sketch.expected_error, 4),
            }
        )
    return out


def sketches_merge_and_samples_do_not(
    rows: int = 200_000,
    groups: int = 8,
    distinct: int = 20_000,
) -> dict:
    """The property that decides which estimator a columnar file stores.

    A table is written as row groups and a reader wants one number for the table. Sketches merge
    exactly, so the reader combines eight per group sketches and gets what a single pass would
    have produced. Distinct counts do not add, so summing the per group counts overcounts every
    value that appears in more than one group.
    """
    values = column(rows, distinct)
    size = rows // groups
    pieces = [values[start : start + size] for start in range(0, rows, size)]
    merged = sketch_of(pieces[0])
    for piece in pieces[1:]:
        merged = merged.merge(sketch_of(piece))
    whole = sketch_of(values)
    summed = sum(exact(piece) for piece in pieces)
    truth = exact(values)
    return {
        "truth": truth,
        "merged": round(merged.estimate(), 1),
        "single_pass": round(whole.estimate(), 1),
        "summed_exact_counts": summed,
        "merging_matches_one_pass": abs(merged.estimate() - whole.estimate()) < 1e-6,
        "summing_overcounts": summed > truth,
        "and_by_a_lot": summed > truth * 2,
    }


def merging_is_exact_not_approximate(rows: int = 100_000, distinct: int = 10_000) -> dict:
    """The merged registers are identical to the single pass registers, not merely close.

    Worth checking as an equality rather than a tolerance, because the register wise maximum is
    exactly the maximum over the union and any difference would mean the hashing is not
    deterministic across calls.
    """
    values = column(rows, distinct)
    half = rows // 2
    merged = sketch_of(values[:half]).merge(sketch_of(values[half:]))
    whole = sketch_of(values)
    return {
        "registers_match": bool(np.array_equal(merged.registers, whole.registers)),
        "estimates_match": merged.estimate() == whole.estimate(),
        "buckets": merged.buckets,
    }


def the_small_range_correction_earns_its_place(rows: int = 10_000) -> dict:
    """What the raw estimator does on a tiny cardinality, and what the correction fixes.

    The harmonic mean estimator is biased low below about two and a half times the bucket count,
    and at a cardinality of ten with four thousand buckets it is out by a large factor. The
    linear counting fallback is exact there. Measured with the correction disabled so the size
    of the bias is visible rather than asserted.
    """
    values = column(rows, 10)
    sketch = sketch_of(values, precision=12)
    counts = sketch.registers.astype(np.float64)
    harmonic = float((2.0 ** (-counts)).sum())
    alpha = 0.7213 / (1.0 + 1.079 / sketch.buckets)
    raw = alpha * sketch.buckets * sketch.buckets / harmonic
    truth = exact(values)
    return {
        "truth": truth,
        "raw_estimate": round(raw, 1),
        "corrected_estimate": round(sketch.estimate(), 1),
        "raw_error": round(error(raw, truth), 4),
        "corrected_error": round(error(sketch.estimate(), truth), 4),
        "the_correction_helps": error(sketch.estimate(), truth) < error(raw, truth),
    }


def adding_a_value_twice_changes_nothing(rows: int = 10_000) -> dict:
    """The sketch counts distinct values and not rows, which is a property of taking a maximum.

    Doubling every row leaves the registers untouched, because the maximum leading zero count
    per bucket is already at its final value. That is what separates this from a counter and it
    is the reason a sketch over a row group is valid whatever the row group holds.
    """
    values = column(rows, 500)
    once = sketch_of(values)
    twice = sketch_of(np.concatenate([values, values]))
    return {
        "rows_doubled": True,
        "registers_match": bool(np.array_equal(once.registers, twice.registers)),
        "estimates_match": once.estimate() == twice.estimate(),
        "truth_unchanged": exact(values) == exact(np.concatenate([values, values])),
    }


def string_columns_hash_too(rows: int = 50_000, distinct: int = 5_000) -> dict:
    """The sketch works on text, which is where distinct counts are most often wanted.

    Strings go through Python's own hash rather than the integer path, which is slower per value
    and gives the same guarantees. A grouping column is usually text, so a sketch that only
    handled numbers would be a sketch for the wrong columns.
    """
    labels = np.array([f"k{int(value):06d}" for value in column(rows, distinct)], dtype=object)
    sketch = sketch_of(labels)
    truth = exact(labels)
    return {
        "truth": truth,
        "estimate": round(sketch.estimate(), 1),
        "error": round(error(sketch.estimate(), truth), 4),
        "it_is_within_a_tenth": error(sketch.estimate(), truth) < 0.1,
    }


def an_empty_column_estimates_nothing() -> dict:
    """A column with no rows, which a reader meets on an empty row group."""
    sketch = HyperLogLog(precision=8)
    counter = LinearCounter(bits=1024)
    return {
        "sketch_estimate": sketch.estimate(),
        "counter_estimate": counter.estimate(),
        "the_sketch_says_zero": sketch.estimate() == 0.0,
        "the_counter_says_zero": counter.estimate() == 0.0,
        "neither_saturated": not counter.saturated,
    }


def a_bad_precision_is_refused() -> bool:
    """A precision outside four to eighteen is a configuration mistake."""
    try:
        HyperLogLog(precision=2)
    except ConfigError:
        return True
    return False


def merging_different_precisions_is_refused() -> bool:
    """Two sketches of different sizes cannot be combined and say so."""
    try:
        HyperLogLog(precision=8).merge(HyperLogLog(precision=10))
    except ConfigError:
        return True
    return False


def a_tiny_counter_is_refused() -> bool:
    """A linear counter smaller than a byte estimates nothing."""
    try:
        LinearCounter(bits=4)
    except ConfigError:
        return True
    return False


def an_impossible_sample_is_refused() -> bool:
    """A sampling rate outside zero to one is not a rate."""
    try:
        sample_of(np.arange(100), rate=1.5)
    except ConfigError:
        return True
    return False


def a_two_dimensional_column_is_refused() -> bool:
    """The hasher takes a column, not a table."""
    try:
        hashed(np.zeros((3, 3)))
    except DataError:
        return True
    return False


def compare_the_estimators(
    rows: int = 200_000,
    distincts: Sequence[int] = (100, 10_000, 200_000),
) -> list[dict]:
    """All three across three cardinalities, which is the module in one table."""
    out = []
    for distinct in distincts:
        values = column(rows, distinct)
        truth = exact(values)
        for name, estimate, size in (
            ("sample", sample_of(values, rate=0.01).estimate(), 0),
            ("linear", linear_of(values).estimate(), 512),
            ("hyperloglog", sketch_of(values).estimate(), 4096),
        ):
            out.append(
                {
                    "distinct": distinct,
                    "estimator": name,
                    "estimate": None if estimate == float("inf") else round(estimate, 1),
                    "error": round(error(estimate, truth), 4),
                    "bytes": size,
                }
            )
    return out


def summarise(rows: int = 200_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    sampled = sampling_cannot_tell_the_two_apart(rows=rows)
    merged = sketches_merge_and_samples_do_not(rows=rows)
    across = hyperloglog_holds_across_the_whole_range(rows=rows)
    return {
        "sampling_is_wrong_somewhere": sampled["one_of_them_is_badly_wrong"],
        "merging_matches_one_pass": merged["merging_matches_one_pass"],
        "summing_overcounts": merged["summing_overcounts"],
        "worst_sketch_error": max(row["error"] for row in across),
        "median_sketch_error": round(pystats.median([row["error"] for row in across]), 4),
    }
