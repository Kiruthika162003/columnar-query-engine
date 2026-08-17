from __future__ import annotations

import statistics as pystats
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.errors import ConfigError, DataError

# Histograms, which exist to answer one question: how many rows will this predicate keep.
#
# Everything a planner decides rests on that number. Which side of a join to build from, whether
# to chain conjuncts, whether a sort can become a top k, how much memory to reserve. Get it
# wrong by an order of magnitude and every decision above it is made on the wrong information.
#
# Two shapes of histogram are here and they fail in different places.
#
# An equi width histogram cuts the value range into buckets of equal width and counts what lands
# in each. It is trivial to build in one pass, it is exact about where the boundaries are, and
# it is useless on a skewed column: a bucket holding ninety percent of the rows tells you almost
# nothing about a predicate falling inside it.
#
# An equi depth histogram cuts the range so each bucket holds the same number of rows, which
# makes the boundaries the quantiles. It costs a sort to build and it is uniform in the thing
# that matters, so a predicate falling inside one bucket is wrong by at most one bucket's worth
# of rows whatever the distribution.
#
# What that buys is not what I expected. On a skewed column the median errors are 0.0013 for
# equi width and 0.0008 for equi depth, a ratio of 1.62, which is almost nothing. The worst
# errors are 0.457 and 0.015, a ratio of 31. So the shapes differ in the tail and barely at
# all in the middle, and the case for equi depth is a case about the worst query rather than
# the typical one.
#
# That matters because a planner is not hurt by being wrong on average. It is hurt by being
# wrong by a factor on one predicate, since that is what makes it pick the wrong join order.
# A median error is the wrong statistic to compare estimators on and this module reports both.
#
# The bucket count turns out to dominate the shape. A hundred and twenty eight equi width
# buckets beat four equi depth ones by sixty times on the median, which is the opposite of
# the comparison I set out to make. Shape wins at a fixed budget; budget wins outright.


@dataclass
class Bucket:
    """One bucket: the range it covers and how many rows fell in it."""

    low: float
    high: float
    count: int
    distinct: int = 0

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ConfigError(f"a bucket from {self.low} to {self.high} is backwards")
        if self.count < 0:
            raise ConfigError(f"{self.count} is not a count")

    @property
    def width(self) -> float:
        """How much of the value range this bucket covers."""
        return self.high - self.low

    def overlap(self, low: float, high: float) -> float:
        """The share of this bucket a range covers, between zero and one.

        Linear interpolation inside the bucket, which is the uniformity assumption every
        histogram rests on. A bucket of zero width is either fully in or fully out, which is the
        case a column with repeated values produces and which the linear form divides by zero
        on.
        """
        if high < self.low or low > self.high:
            return 0.0
        if self.width == 0:
            return 1.0 if low <= self.low <= high else 0.0
        covered = min(high, self.high) - max(low, self.low)
        return max(0.0, min(1.0, covered / self.width))

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "count": self.count,
            "distinct": self.distinct,
        }


@dataclass
class Histogram:
    """A set of buckets covering a column, and what they were built from."""

    buckets: tuple[Bucket, ...]
    rows: int
    nulls: int
    kind: str

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ConfigError("a histogram needs at least one bucket")
        if self.rows < 0 or self.nulls < 0:
            raise ConfigError(f"{self.rows} rows with {self.nulls} nulls is not a column")

    @property
    def present(self) -> int:
        """How many rows hold a value, which is what the buckets cover."""
        return self.rows - self.nulls

    @property
    def low(self) -> float:
        """The smallest value the histogram saw."""
        return self.buckets[0].low

    @property
    def high(self) -> float:
        """The largest."""
        return self.buckets[-1].high

    @property
    def nbytes(self) -> int:
        """What the histogram costs to store, which is what a bucket count buys."""
        return len(self.buckets) * 24 + 16

    def estimate_range(self, low: float, high: float) -> float:
        """How many rows fall between two bounds.

        The sum over buckets of the count times the share of the bucket the range covers. Every
        error in a cost model traces back to this line, and the shape of the histogram decides
        how big it is.
        """
        if high < low:
            return 0.0
        return sum(bucket.count * bucket.overlap(low, high) for bucket in self.buckets)

    def estimate_less_than(self, value: float) -> float:
        """How many rows are below a bound."""
        return self.estimate_range(self.low - 1.0, value)

    def estimate_greater_than(self, value: float) -> float:
        """How many rows are above one."""
        return self.estimate_range(value, self.high + 1.0)

    def estimate_equal(self, value: float) -> float:
        """How many rows hold one value.

        The weakest estimate a histogram makes, because a bucket has no idea how its rows are
        spread over its distinct values. Uses the distinct count per bucket where the builder
        recorded one, and falls back to the bucket width otherwise.
        """
        for bucket in self.buckets:
            if bucket.low <= value <= bucket.high:
                if bucket.distinct > 0:
                    return bucket.count / bucket.distinct
                if bucket.width == 0:
                    return float(bucket.count)
                return bucket.count / max(bucket.width, 1.0)
        return 0.0

    def selectivity(self, low: float, high: float) -> float:
        """The estimated share of rows a range keeps."""
        if self.rows == 0:
            return 0.0
        return self.estimate_range(low, high) / self.rows

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kind": self.kind,
            "buckets": len(self.buckets),
            "rows": self.rows,
            "nulls": self.nulls,
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "bytes": self.nbytes,
        }


def equi_width(values: np.ndarray, buckets: int = 32) -> Histogram:
    """Cut the value range into equal width buckets and count what lands in each.

    One pass, no sort, and the boundaries are known before the data is read. That last part is
    what makes it attractive for a streaming writer and it is also what makes it fail on skew:
    the boundaries owe nothing to where the rows actually are.
    """
    array = _clean(values)
    if buckets < 1:
        raise ConfigError(f"{buckets} is not a bucket count")
    if not len(array):
        return Histogram(
            buckets=(Bucket(0.0, 0.0, 0),), rows=len(values), nulls=len(values), kind="width"
        )
    low = float(array.min())
    high = float(array.max())
    if low == high:
        return Histogram(
            buckets=(Bucket(low, high, len(array), 1),),
            rows=len(values),
            nulls=len(values) - len(array),
            kind="width",
        )
    edges = np.linspace(low, high, buckets + 1)
    counts, _ = np.histogram(array, bins=edges)
    out = []
    for position in range(buckets):
        inside = array[(array >= edges[position]) & (array <= edges[position + 1])]
        out.append(
            Bucket(
                low=float(edges[position]),
                high=float(edges[position + 1]),
                count=int(counts[position]),
                distinct=len(np.unique(inside)),
            )
        )
    return Histogram(
        buckets=tuple(out),
        rows=len(values),
        nulls=len(values) - len(array),
        kind="width",
    )


def equi_depth(values: np.ndarray, buckets: int = 32) -> Histogram:
    """Cut the range so every bucket holds the same number of rows.

    Costs a sort. The boundaries are the quantiles, so a bucket is narrow where the rows are
    dense and wide where they are sparse, which is exactly the correction skew needs.

    Buckets can collapse to zero width when a single value holds more rows than a bucket, and
    that is not a defect. A zero width bucket is a statement that one value is common, which is
    the most useful thing a histogram can say about a skewed column.
    """
    array = _clean(values)
    if buckets < 1:
        raise ConfigError(f"{buckets} is not a bucket count")
    if not len(array):
        return Histogram(
            buckets=(Bucket(0.0, 0.0, 0),), rows=len(values), nulls=len(values), kind="depth"
        )
    ordered = np.sort(array)
    edges = np.linspace(0, len(ordered), buckets + 1).astype(int)
    out = []
    for position in range(buckets):
        start, stop = edges[position], edges[position + 1]
        if stop <= start:
            continue
        piece = ordered[start:stop]
        out.append(
            Bucket(
                low=float(piece[0]),
                high=float(piece[-1]),
                count=len(piece),
                distinct=len(np.unique(piece)),
            )
        )
    if not out:
        out = [Bucket(float(ordered[0]), float(ordered[-1]), len(ordered), 1)]
    return Histogram(
        buckets=tuple(out),
        rows=len(values),
        nulls=len(values) - len(array),
        kind="depth",
    )


def _clean(values: np.ndarray) -> np.ndarray:
    """The values with nulls removed, as floats."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise DataError(f"a column is one dimensional, not {array.ndim}")
    return array[~np.isnan(array)]


def uniform(rows: int, span: int = 1_000_000, seed: int = 0) -> np.ndarray:
    """A column with no skew at all, which is the case histograms are not needed for."""
    if rows < 1:
        raise ConfigError(f"{rows} is not a row count")
    return np.random.default_rng(seed).integers(0, span, size=rows).astype(np.float64)


def skewed(rows: int, span: int = 1_000_000, power: float = 2.0, seed: int = 0) -> np.ndarray:
    """A column whose mass sits near the bottom of its range, which is what real data does.

    Raising a uniform draw to a power concentrates it towards zero while keeping the full range
    populated, so the maximum is still what it was and every equi width bucket still exists.
    That is the shape that breaks an equi width histogram, since the boundaries are set by a
    range the rows have almost abandoned.
    """
    if rows < 1 or power <= 0:
        raise ConfigError(f"{rows} rows at a power of {power} is not a column")
    draw = np.random.default_rng(seed).random(rows)
    return np.floor(np.power(draw, power) * span).astype(np.float64)


def bimodal(rows: int, span: int = 1_000_000, seed: int = 0) -> np.ndarray:
    """Two clusters with a gap between them, which is what a mixed population looks like."""
    if rows < 2:
        raise ConfigError(f"{rows} is not a row count")
    generator = np.random.default_rng(seed)
    half = rows // 2
    low = generator.integers(0, span // 20, size=half)
    high = generator.integers(19 * span // 20, span, size=rows - half)
    return np.concatenate([low, high]).astype(np.float64)


def exact_range(values: np.ndarray, low: float, high: float) -> int:
    """The true count, which every estimate here is scored against."""
    array = _clean(values)
    return int(((array >= low) & (array <= high)).sum())


def error(estimate: float, truth: float) -> float:
    """Relative error, with the convention that missing everything is total error.

    A ratio rather than a difference, because a planner cares whether an estimate is off by a
    factor and not by a count. An estimate of zero against a truth of zero is exact.
    """
    if truth == 0:
        return 0.0 if estimate == 0 else float("inf")
    return abs(estimate - truth) / truth


def _probes(values: np.ndarray, count: int = 40, seed: int = 0) -> list[tuple[float, float]]:
    """A set of range predicates spread over the column, for scoring an estimator."""
    array = _clean(values)
    generator = np.random.default_rng(seed)
    low, high = float(array.min()), float(array.max())
    out = []
    for _ in range(count):
        edges = sorted(generator.random(2) * (high - low) + low)
        out.append((float(edges[0]), float(edges[1])))
    return out


def score(
    histogram: Histogram, values: np.ndarray, probes: Sequence[tuple[float, float]]
) -> dict:
    """How wrong a histogram is over a set of range predicates."""
    if not probes:
        raise ConfigError("there is nothing to score")
    errors = []
    for low, high in probes:
        truth = exact_range(values, low, high)
        estimate = histogram.estimate_range(low, high)
        errors.append(error(estimate, float(truth)))
    finite = [one for one in errors if one != float("inf")]
    return {
        "kind": histogram.kind,
        "buckets": len(histogram.buckets),
        "probes": len(probes),
        "median_error": round(pystats.median(finite), 4) if finite else float("inf"),
        "mean_error": round(pystats.fmean(finite), 4) if finite else float("inf"),
        "worst_error": round(max(finite), 4) if finite else float("inf"),
        "misses": len(errors) - len(finite),
    }


def they_agree_on_a_uniform_column(rows: int = 200_000, buckets: int = 32) -> dict:
    """On a column with no skew the sort an equi depth histogram costs buys nothing.

    Both forms put their boundaries in the same places, because the quantiles of a uniform draw
    are evenly spaced. So the equi width form is exactly as good and is cheaper to build, which
    is worth knowing before deciding to always build the expensive one.
    """
    values = uniform(rows)
    probes = _probes(values)
    width = score(equi_width(values, buckets), values, probes)
    depth = score(equi_depth(values, buckets), values, probes)
    return {
        "width_median": width["median_error"],
        "depth_median": depth["median_error"],
        "they_are_close": abs(width["median_error"] - depth["median_error"]) < 0.02,
        "neither_is_bad": max(width["median_error"], depth["median_error"]) < 0.1,
    }


def equi_depth_wins_on_a_skewed_column(rows: int = 200_000, buckets: int = 32) -> dict:
    """The case histograms exist for, and the reason the engine pays for the sort.

    A skewed column puts most of its rows into the first few equi width buckets, so a predicate
    landing inside one of those is estimated by spreading a huge count uniformly over a range
    the rows do not occupy. The equi depth form has narrow buckets there.

    The medians are 0.0013 and 0.0008, a ratio of 1.62, which is far less than I expected. The
    worst errors are 0.457 and 0.015, a ratio of 31. The two shapes are nearly identical on a
    typical predicate and an order of magnitude apart on the worst one, so the median is the
    wrong number to compare them on and both are reported.
    """
    values = skewed(rows)
    probes = _probes(values)
    width = score(equi_width(values, buckets), values, probes)
    depth = score(equi_depth(values, buckets), values, probes)
    return {
        "width_median": width["median_error"],
        "depth_median": depth["median_error"],
        "width_worst": width["worst_error"],
        "depth_worst": depth["worst_error"],
        "depth_wins": depth["median_error"] < width["median_error"],
        "ratio": round(width["median_error"] / max(depth["median_error"], 1e-9), 2),
    }


def and_on_a_bimodal_one(rows: int = 200_000, buckets: int = 32) -> dict:
    """Two clusters with an empty gap, where equal width buckets are mostly empty.

    Nineteen twentieths of the range holds no rows, so twenty eight of thirty two equi width
    buckets are empty and the estimator has four buckets of resolution where the rows are.
    The equi depth form spends every bucket on a populated region and has none empty.

    Both medians come out at zero, which is the same lesson as the skewed case: an empty
    bucket is harmless on a predicate that misses it entirely, and the cost of the wasted
    buckets appears only on predicates that land in the dense regions.
    """
    values = bimodal(rows)
    probes = _probes(values)
    width_histogram = equi_width(values, buckets)
    depth_histogram = equi_depth(values, buckets)
    width = score(width_histogram, values, probes)
    depth = score(depth_histogram, values, probes)
    empty = sum(1 for bucket in width_histogram.buckets if bucket.count == 0)
    return {
        "width_median": width["median_error"],
        "depth_median": depth["median_error"],
        "empty_width_buckets": empty,
        "empty_depth_buckets": sum(
            1 for bucket in depth_histogram.buckets if bucket.count == 0
        ),
        "depth_wins": depth["median_error"] <= width["median_error"],
        "most_width_buckets_are_empty": empty > buckets // 2,
    }


def more_buckets_help_both(
    rows: int = 200_000,
    counts: Sequence[int] = (4, 8, 16, 32, 64, 128),
) -> list[dict]:
    """How the error falls as the bucket count rises, for both shapes.

    Roughly as one over the count for both, which is the expected behaviour and is not where the
    difference between the two lives. Worth measuring anyway, because it says the choice of
    shape matters more than the choice of size and a planner budget is better spent on the right
    shape than on more buckets of the wrong one.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    values = skewed(rows)
    probes = _probes(values)
    out = []
    for count in counts:
        width = score(equi_width(values, count), values, probes)
        depth = score(equi_depth(values, count), values, probes)
        out.append(
            {
                "buckets": count,
                "width_median": width["median_error"],
                "depth_median": depth["median_error"],
                "bytes": equi_depth(values, count).nbytes,
            }
        )
    return out


def many_bad_buckets_beat_a_few_good_ones(rows: int = 200_000) -> dict:
    """Four equi depth buckets against a hundred and twenty eight equi width ones.

    I wrote this expecting the four good buckets to win, on the grounds that shape beats size.
    They lose by sixty times, 0.0243 against 0.0004, for a twenty seventh of the storage.

    So the two are not substitutes. At a fixed bucket count the equi depth shape wins, and at a
    fixed byte budget more buckets of either shape wins. A planner with statistics to spend
    should spend them on resolution first and shape second, which is the reverse of the advice
    this module was written to give.
    """
    values = skewed(rows)
    probes = _probes(values)
    small = score(equi_depth(values, 4), values, probes)
    large = score(equi_width(values, 128), values, probes)
    return {
        "four_depth_buckets": small["median_error"],
        "hundred_and_twenty_eight_width_buckets": large["median_error"],
        "depth_bytes": equi_depth(values, 4).nbytes,
        "width_bytes": equi_width(values, 128).nbytes,
        "the_small_one_loses": small["median_error"] > large["median_error"],
        "but_is_smaller": equi_depth(values, 4).nbytes < equi_width(values, 128).nbytes,
        "ratio": round(small["median_error"] / max(large["median_error"], 1e-9), 1),
    }


def an_equality_estimate_needs_the_distinct_count(rows: int = 100_000) -> dict:
    """Estimating a single value, which is the weakest thing a histogram does.

    A bucket knows how many rows it holds and how wide it is, and without a distinct count it
    has to assume the rows are spread over every value in the range. Recording the distinct
    count per bucket costs four bytes and turns a guess into an average.

    Measured against the most common value in a skewed column, which is the one a planner is
    most likely to be asked about and the one a uniformity assumption is furthest from. The
    first version used the median, which on a continuous column is a value no row holds, so the
    true count was zero and both estimators scored perfectly.
    """
    values = skewed(rows)
    present, counts = np.unique(values, return_counts=True)
    wanted = float(present[int(np.argmax(counts))])
    truth = float(counts.max())
    with_counts = equi_depth(values, 32)
    without = Histogram(
        buckets=tuple(Bucket(one.low, one.high, one.count, 0) for one in with_counts.buckets),
        rows=with_counts.rows,
        nulls=with_counts.nulls,
        kind="depth",
    )
    return {
        "value": wanted,
        "truth": truth,
        "with_distinct": round(with_counts.estimate_equal(wanted), 2),
        "without_distinct": round(without.estimate_equal(wanted), 2),
        "with_error": round(error(with_counts.estimate_equal(wanted), truth), 4),
        "without_error": round(error(without.estimate_equal(wanted), truth), 4),
        "the_distinct_count_helps": (
            error(with_counts.estimate_equal(wanted), truth)
            < error(without.estimate_equal(wanted), truth)
        ),
    }


def a_range_outside_the_column_estimates_nothing(rows: int = 10_000) -> dict:
    """A predicate outside the observed range, which a planner asks for constantly.

    Every stale statistic produces this: the histogram was built last week and the query asks
    about today. The estimate is zero, which is right about the data the histogram saw and wrong
    about the table, and no histogram can do better. What it can do is be obviously zero rather
    than quietly small, so a planner can notice.
    """
    values = uniform(rows, span=1000)
    histogram = equi_depth(values, 16)
    return {
        "above": histogram.estimate_range(2000, 3000),
        "below": histogram.estimate_range(-2000, -1000),
        "both_are_zero": histogram.estimate_range(2000, 3000) == 0.0,
        "the_full_range_is_everything": round(
            histogram.estimate_range(histogram.low, histogram.high)
        )
        == rows,
    }


def nulls_are_counted_and_never_estimated(rows: int = 10_000) -> dict:
    """Nulls sit outside every bucket and are recorded separately.

    A histogram that folded nulls into a bucket would estimate them as matching a range, which
    no comparison ever does. Keeping the count separate is what lets a planner estimate is null
    exactly while estimating everything else approximately.
    """
    values = uniform(rows)
    values[: rows // 10] = np.nan
    histogram = equi_depth(values, 16)
    return {
        "rows": histogram.rows,
        "nulls": histogram.nulls,
        "present": histogram.present,
        "the_nulls_were_counted": histogram.nulls == rows // 10,
        "the_buckets_hold_the_rest": sum(one.count for one in histogram.buckets)
        == histogram.present,
    }


def a_constant_column_is_one_bucket(rows: int = 10_000) -> dict:
    """Every value the same, which collapses the range to a point."""
    values = np.full(rows, 7.0)
    width = equi_width(values, 32)
    depth = equi_depth(values, 32)
    return {
        "width_buckets": len(width.buckets),
        "depth_buckets": len(depth.buckets),
        "width_is_one": len(width.buckets) == 1,
        "the_estimate_is_everything": width.estimate_range(7, 7) == rows,
        "and_nothing_elsewhere": width.estimate_range(8, 9) == 0.0,
    }


def a_zero_width_bucket_is_a_statement_about_skew(rows: int = 100_000) -> dict:
    """A single value holding more rows than a bucket collapses that bucket to a point.

    Not a defect. A zero width bucket says one value is at least as common as a bucket's depth,
    which is the most useful thing a histogram can record about a skewed column, and it is
    exactly what an equi width histogram cannot express.
    """
    values = np.concatenate([np.full(rows // 2, 5000.0), uniform(rows // 2, span=1000)])
    histogram = equi_depth(values, 16)
    flat = [one for one in histogram.buckets if one.width == 0]
    return {
        "buckets": len(histogram.buckets),
        "zero_width_buckets": len(flat),
        "there_is_at_least_one": len(flat) >= 1,
        "it_holds_the_common_value": any(one.low == 5000.0 for one in flat),
        "the_estimate_for_it_is_large": histogram.estimate_equal(5000.0) > rows / 20,
    }


def an_empty_column_is_refused_nothing(rows: int = 0) -> dict:
    """A column of nothing, which a planner meets on an empty partition."""
    del rows
    values = np.array([], dtype=np.float64)
    histogram = equi_depth(values, 8)
    return {
        "buckets": len(histogram.buckets),
        "rows": histogram.rows,
        "the_estimate_is_zero": histogram.estimate_range(0, 100) == 0.0,
        "the_selectivity_is_zero": histogram.selectivity(0, 100) == 0.0,
    }


def a_zero_bucket_count_is_refused() -> bool:
    """A histogram with no buckets estimates nothing and is a configuration mistake."""
    try:
        equi_depth(uniform(100), 0)
    except ConfigError:
        return True
    return False


def a_backwards_bucket_is_refused() -> bool:
    """A bucket whose high is below its low is not a bucket."""
    try:
        Bucket(low=10.0, high=5.0, count=1)
    except ConfigError:
        return True
    return False


def a_negative_count_is_refused() -> bool:
    """Nor is one holding a negative number of rows."""
    try:
        Bucket(low=0.0, high=1.0, count=-1)
    except ConfigError:
        return True
    return False


def a_histogram_with_no_buckets_is_refused() -> bool:
    """And a histogram has to hold at least one."""
    try:
        Histogram(buckets=(), rows=10, nulls=0, kind="depth")
    except ConfigError:
        return True
    return False


def a_two_dimensional_column_is_refused() -> bool:
    """The builder takes a column, not a table."""
    try:
        equi_width(np.zeros((3, 3)))
    except DataError:
        return True
    return False


def compare_the_shapes(rows: int = 200_000, buckets: int = 32) -> list[dict]:
    """Both shapes against all three distributions, which is the module in one table."""
    out = []
    for name, values in (
        ("uniform", uniform(rows)),
        ("skewed", skewed(rows)),
        ("bimodal", bimodal(rows)),
    ):
        probes = _probes(values)
        for builder in (equi_width, equi_depth):
            row = score(builder(values, buckets), values, probes)
            row["distribution"] = name
            out.append(row)
    return out


def summarise(rows: int = 200_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    flat = they_agree_on_a_uniform_column(rows=rows)
    steep = equi_depth_wins_on_a_skewed_column(rows=rows)
    split = and_on_a_bimodal_one(rows=rows)
    budget = many_bad_buckets_beat_a_few_good_ones(rows=rows)
    return {
        "uniform_width": flat["width_median"],
        "uniform_depth": flat["depth_median"],
        "they_agree_when_flat": flat["they_are_close"],
        "skewed_width": steep["width_median"],
        "skewed_depth": steep["depth_median"],
        "skewed_median_ratio": steep["ratio"],
        "skewed_worst_width": steep["width_worst"],
        "skewed_worst_depth": steep["depth_worst"],
        "bimodal_depth_wins": split["depth_wins"],
        "buckets_beat_shape": budget["the_small_one_loses"],
    }
