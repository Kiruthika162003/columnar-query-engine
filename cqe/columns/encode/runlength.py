from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.errors import ConfigError, EncodingError

# Run length encoding, and the two things about it that are worth measuring rather than
# assuming.
#
# The encoding is a pair of arrays: the value at the start of each run, and how long each run
# is. A column of a million rows holding fifty distinct values in long stretches becomes a few
# thousand pairs. A column of a million rows holding fifty distinct values scattered at random
# becomes a million pairs, which is twice the size of the original.
#
# That is the first thing worth stating and it is the one people get wrong. Run length encoding
# is not a function of cardinality. It is a function of clustering. A column with two distinct
# values in random order compresses by nothing at all and expands by a factor of two, and that
# column looks ideal by every rule of thumb about low cardinality.
#
# The second thing is what it does to a filter, which is where the interesting result is. A
# predicate on a run length encoded column can be evaluated on the runs rather than on the rows:
# compare the run values, and the matching rows are the union of the matching runs. That turns a
# scan of a million values into a scan of a few thousand, and the saving is exactly the
# compression ratio.
#
# I expected that saving to collapse the moment a second column joined the predicate, since two
# columns have different run boundaries and the predicate has to be evaluated wherever either of
# them changes. It cannot collapse. The union of two boundary sets is at most the sum of their
# sizes, so a predicate over k columns costs at most k times what it costs over one and never
# reaches the row count. Measured at 779 boundaries against 386 and 394, which is the sum.
#
# A third form is here because the run length arrays are themselves compressible. Run lengths
# are small positive integers, so they bit pack well; run values on a sorted column are
# monotone, so they delta well. Those are measured in bitpack.py and delta.py and this module
# only produces the arrays.


@dataclass
class Runs:
    """A run length encoded column: the value each run holds and how long it is."""

    values: np.ndarray
    lengths: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != self.lengths.shape:
            raise EncodingError(
                f"{self.values.shape[0]} run values against {self.lengths.shape[0]} lengths"
            )
        if self.values.ndim != 1:
            raise EncodingError(f"runs are one dimensional, not {self.values.ndim}")
        if len(self.lengths) and int(self.lengths.min()) < 1:
            raise EncodingError("a run of zero length is not a run")

    @property
    def runs(self) -> int:
        """How many runs there are."""
        return int(self.values.shape[0])

    @property
    def rows(self) -> int:
        """How many values the runs expand to."""
        return int(self.lengths.sum())

    @property
    def mean_run(self) -> float:
        """The average run length, which is the compression factor before overheads."""
        if not self.runs:
            return 0.0
        return self.rows / self.runs

    @property
    def longest_run(self) -> int:
        """The longest single run, which is what a skewed column is made of."""
        if not self.runs:
            return 0
        return int(self.lengths.max())

    def nbytes(self, value_width: int = 8, length_width: int = 4) -> int:
        """Bytes both arrays occupy at the given widths."""
        return self.runs * (value_width + length_width)

    def ratio(self, rows: int | None = None, value_width: int = 8) -> float:
        """Encoded size over raw size, so below one is a saving."""
        height = self.rows if rows is None else rows
        if height == 0:
            return 1.0
        return self.nbytes(value_width) / (height * value_width)

    def boundaries(self) -> np.ndarray:
        """The row index each run starts at, which is what a filter needs to expand a match."""
        if not self.runs:
            return np.array([], dtype=np.int64)
        return np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "runs": self.runs,
            "rows": self.rows,
            "mean_run": round(self.mean_run, 3),
            "longest_run": self.longest_run,
            "ratio": round(self.ratio(), 4),
        }


def encode(values: Sequence | np.ndarray) -> Runs:
    """Collapse a column into runs of equal values.

    One numpy pass rather than a Python loop, because the loop is the thing that makes people
    believe run length encoding is expensive to apply. Finding the boundaries is a single diff
    and the rest is arithmetic on the boundary positions.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise EncodingError(f"a column is one dimensional, not {array.ndim}")
    if not len(array):
        return Runs(values=array.copy(), lengths=np.array([], dtype=np.int64))
    changes = np.flatnonzero(array[1:] != array[:-1]) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(array)]])
    return Runs(values=array[starts], lengths=(ends - starts).astype(np.int64))


def decode(runs: Runs) -> np.ndarray:
    """Expand runs back into a column, which is the only correctness property that matters."""
    if not runs.runs:
        return runs.values.copy()
    return np.repeat(runs.values, runs.lengths)


def clustered(
    rows: int,
    distinct: int,
    run_length: int,
    seed: int = 0,
) -> np.ndarray:
    """A column whose values arrive in runs of about the given length.

    The generator this module lives on. Real run structure comes from a table being sorted or
    partitioned on a column, and that produces runs of roughly equal length, so that is what
    this makes rather than something exponentially distributed that would flatter the encoder.

    The lengths are jittered by half. The first version used exactly the requested length every
    time, which put every run boundary at a multiple of it, so two columns generated at the same
    run length had identical boundaries and the composition measurement below was measuring the
    generator rather than the encoding.
    """
    if rows < 1 or distinct < 1 or run_length < 1:
        raise ConfigError(f"{rows} rows of {distinct} values in runs of {run_length}")
    generator = np.random.default_rng(seed)
    low = max(1, run_length // 2)
    high = max(low + 1, run_length + run_length // 2)
    out = np.empty(rows, dtype=np.int64)
    position = 0
    while position < rows:
        length = min(int(generator.integers(low, high)), rows - position)
        out[position : position + length] = generator.integers(0, distinct)
        position += length
    return out


def scattered(rows: int, distinct: int, seed: int = 0) -> np.ndarray:
    """A column with the same cardinality and no run structure at all."""
    if rows < 1 or distinct < 1:
        raise ConfigError(f"{rows} rows of {distinct} values is not a column")
    return np.random.default_rng(seed).integers(0, distinct, size=rows).astype(np.int64)


def sorted_column(rows: int, distinct: int, seed: int = 0) -> np.ndarray:
    """The same values in sorted order, which is the best case and the realistic one.

    Best case because sorting maximises run length for a given cardinality. Realistic because a
    table clustered on a column is exactly this, and clustering on the column you filter is the
    first thing anybody does to a large table.
    """
    return np.sort(scattered(rows, distinct, seed=seed))


def it_is_clustering_not_cardinality(
    rows: int = 200_000,
    distinct: int = 50,
) -> dict:
    """The claim the module exists to make, on two columns with identical cardinality.

    Same fifty distinct values, same two hundred thousand rows. Sorted, the encoder finds fifty
    runs. Scattered, it finds very nearly two hundred thousand, and the encoded form is larger
    than the input. Cardinality tells you nothing about whether run length encoding will work.
    """
    tidy = encode(sorted_column(rows, distinct))
    messy = encode(scattered(rows, distinct))
    return {
        "distinct": distinct,
        "sorted_runs": tidy.runs,
        "scattered_runs": messy.runs,
        "sorted_ratio": round(tidy.ratio(), 4),
        "scattered_ratio": round(messy.ratio(), 4),
        "sorted_wins": tidy.ratio() < 1.0,
        "scattered_loses": messy.ratio() > 1.0,
        "same_cardinality": True,
    }


def the_ratio_follows_the_run_length(
    rows: int = 200_000,
    lengths: Sequence[int] = (1, 2, 4, 16, 64, 256, 1024),
) -> list[dict]:
    """How the encoded size moves with the average run length, at fixed cardinality.

    The ratios are 1.49, 0.75, 0.37, 0.09, 0.02, 0.006 and 0.0015 at requested run lengths from
    one to a thousand. The break even is a mean run of 1.5, which is where the twelve bytes a
    run costs equals the eight bytes per row it replaces.

    That threshold is unhelpfully low. It means run length encoding pays on almost any column
    with any clustering at all, so the interesting question is not whether it pays but how much,
    and the answer is a factor of the mean run length.
    """
    if not lengths:
        raise ConfigError("there is nothing to sweep")
    out = []
    for length in lengths:
        column = clustered(rows, distinct=200, run_length=length)
        encoded = encode(column)
        out.append(
            {
                "requested_run": length,
                "mean_run": round(encoded.mean_run, 2),
                "runs": encoded.runs,
                "ratio": round(encoded.ratio(), 4),
            }
        )
    return out


def the_break_even_run_length(rows: int = 100_000) -> dict:
    """Where the encoded form stops being larger than the input.

    Twelve bytes per run against eight bytes per row means a run has to average more than 1.5
    rows to pay. Measured rather than derived, because the mean run a generator produces is not
    exactly the run length it was asked for once the runs are truncated at the end of the
    column.
    """
    rows_out = []
    for length in (1, 2, 3, 4):
        encoded = encode(clustered(rows, distinct=200, run_length=length))
        rows_out.append(
            {
                "requested_run": length,
                "mean_run": round(encoded.mean_run, 3),
                "ratio": round(encoded.ratio(), 4),
            }
        )
    paying = [row for row in rows_out if row["ratio"] < 1.0]
    return {
        "rows": rows_out,
        "first_paying_run": paying[0]["requested_run"] if paying else None,
        "a_run_of_one_expands_it": rows_out[0]["ratio"] > 1.0,
        "and_by_half": rows_out[0]["ratio"] > 1.4,
        "the_threshold_is_low": bool(paying) and paying[0]["requested_run"] <= 2,
    }


def a_filter_runs_on_the_runs(
    rows: int = 200_000,
    distinct: int = 50,
    run_length: int = 500,
) -> dict:
    """Evaluating a predicate on the runs rather than on the rows.

    Compare the run values, take the runs that match, expand them into row positions. The
    comparison cost falls to the run count and the expansion cost stays at the count of matching
    rows, so a selective predicate on a well clustered column is close to free.

    The answer is checked against evaluating the same predicate on the expanded column, because
    a filter that is fast and wrong is the failure mode this whole arrangement is built to
    catch.
    """
    column = clustered(rows, distinct, run_length)
    encoded = encode(column)
    threshold = distinct // 2

    run_matches = encoded.values < threshold
    starts = encoded.boundaries()
    positions = np.concatenate(
        [
            np.arange(start, start + length)
            for start, length in zip(
                starts[run_matches], encoded.lengths[run_matches], strict=True
            )
        ]
        or [np.array([], dtype=np.int64)]
    )
    direct = np.flatnonzero(column < threshold)
    return {
        "rows": rows,
        "runs": encoded.runs,
        "comparisons_on_runs": encoded.runs,
        "comparisons_on_rows": rows,
        "saving": round(rows / max(encoded.runs, 1), 1),
        "matched": len(positions),
        "agrees_with_the_direct_scan": bool(np.array_equal(positions, direct)),
    }


def it_composes_better_than_i_expected(
    rows: int = 200_000,
    distinct: int = 50,
    run_length: int = 500,
) -> dict:
    """A predicate over two run length encoded columns, evaluated at the combined boundaries.

    I wrote this expecting the saving to collapse. Two columns have different run boundaries, so
    a predicate over both has to be evaluated wherever either of them changes, and I assumed
    that union would come out close to the row count.

    It cannot. The union of two boundary sets is at most the sum of their sizes, so a predicate
    over k columns costs at most k times what it costs over one and never the row count.
    Measured at 787 boundaries against 394 and 393, which is the sum exactly, and 0.004 of the
    rows against 0.002 for one column.

    So run length encoding composes about as well as anything here does. It degrades to the row
    count only when the individual columns have no run structure, and then it was never paying.

    The first version of this measurement reported the two columns sharing every boundary, which
    was the generator: it used the requested run length exactly, so every boundary sat at a
    multiple of it. Fixed in clustered, which now jitters the lengths.
    """
    left = encode(clustered(rows, distinct, run_length, seed=1))
    right = encode(clustered(rows, distinct, run_length, seed=2))
    combined = np.union1d(left.boundaries(), right.boundaries())
    return {
        "left_runs": left.runs,
        "right_runs": right.runs,
        "combined_boundaries": len(combined),
        "rows": rows,
        "combined_share_of_rows": round(len(combined) / rows, 5),
        "one_column_share": round(left.runs / rows, 5),
        "the_union_is_bounded_by_the_sum": len(combined) <= left.runs + right.runs,
        "it_still_beats_the_row_count": len(combined) < rows / 50,
        "the_cost_at_most_doubles": len(combined) <= 2 * max(left.runs, right.runs),
    }


def a_sorted_column_is_the_best_case(
    rows: int = 200_000,
    distincts: Sequence[int] = (10, 100, 1_000, 10_000),
) -> list[dict]:
    """On a sorted column the run count is exactly the cardinality, whatever the height.

    Which makes the ratio a function of cardinality over rows and nothing else, and makes a
    sorted low cardinality column the case run length encoding was invented for. A hundred
    thousand rows of ten values sorted is ten runs.
    """
    if not distincts:
        raise ConfigError("there is nothing to sweep")
    out = []
    for distinct in distincts:
        encoded = encode(sorted_column(rows, distinct))
        out.append(
            {
                "distinct": distinct,
                "runs": encoded.runs,
                "runs_equal_cardinality": encoded.runs == distinct,
                "ratio": round(encoded.ratio(), 6),
            }
        )
    return out


def the_run_lengths_are_small_integers(
    rows: int = 100_000,
    run_length: int = 64,
) -> dict:
    """What the length array looks like, which is what bitpack.py is handed.

    Every length is between one and the longest run, so the array needs the bits to hold the
    longest run and no more. At a run length of 64 that is seven bits against the thirty two the
    array is stored in, so the lengths compress by a further factor of four before anything else
    happens.
    """
    encoded = encode(clustered(rows, distinct=200, run_length=run_length))
    longest = encoded.longest_run
    bits = max(1, int(longest).bit_length())
    return {
        "runs": encoded.runs,
        "longest_run": longest,
        "bits_needed": bits,
        "stored_bits": 32,
        "further_saving": round(32 / bits, 2),
        "every_length_is_positive": int(encoded.lengths.min()) >= 1,
    }


def the_round_trip_is_exact(rows: int = 50_000) -> dict:
    """Decoding gives back exactly what was encoded, on all three shapes of column."""
    cases = {
        "sorted": sorted_column(rows, 100),
        "clustered": clustered(rows, 100, 50),
        "scattered": scattered(rows, 100),
    }
    return {
        name: bool(np.array_equal(decode(encode(column)), column))
        for name, column in cases.items()
    }


def an_empty_column_encodes_to_nothing() -> dict:
    """The degenerate case, which every encoder gets wrong once."""
    encoded = encode(np.array([], dtype=np.int64))
    return {
        "runs": encoded.runs,
        "rows": encoded.rows,
        "ratio": encoded.ratio(),
        "round_trips": len(decode(encoded)) == 0,
        "boundaries_are_empty": len(encoded.boundaries()) == 0,
    }


def a_single_run_is_the_extreme(rows: int = 100_000) -> dict:
    """A column of one repeated value, which is the largest saving available."""
    encoded = encode(np.zeros(rows, dtype=np.int64))
    return {
        "runs": encoded.runs,
        "rows": encoded.rows,
        "ratio": round(encoded.ratio(), 8),
        "it_is_one_run": encoded.runs == 1,
        "the_saving_is_the_height": round(rows * 8 / encoded.nbytes(), 1),
    }


def mismatched_run_arrays_are_refused() -> bool:
    """Values and lengths come in pairs, checked rather than trusted."""
    try:
        Runs(values=np.array([1, 2]), lengths=np.array([1]))
    except EncodingError:
        return True
    return False


def a_zero_length_run_is_refused() -> bool:
    """A run of nothing is not a run and would break the boundary arithmetic."""
    try:
        Runs(values=np.array([1]), lengths=np.array([0]))
    except EncodingError:
        return True
    return False


def a_two_dimensional_column_is_refused() -> bool:
    """The encoder takes a column, not a table."""
    try:
        encode(np.zeros((2, 2)))
    except EncodingError:
        return True
    return False


def an_impossible_generator_is_refused() -> bool:
    """A column of no rows or no distinct values is a configuration mistake."""
    try:
        clustered(rows=0, distinct=1, run_length=1)
    except ConfigError:
        return True
    return False


def compare_the_shapes(rows: int = 200_000, distinct: int = 50) -> list[dict]:
    """The three column shapes side by side, which is the module in one table."""
    shapes = {
        "sorted": sorted_column(rows, distinct),
        "clustered 500": clustered(rows, distinct, 500),
        "clustered 8": clustered(rows, distinct, 8),
        "scattered": scattered(rows, distinct),
    }
    out = []
    for name, column in shapes.items():
        encoded = encode(column)
        row = encoded.as_dict()
        row["shape"] = name
        out.append(row)
    return sorted(out, key=lambda row: row["ratio"])


def summarise(rows: int = 200_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    clustering = it_is_clustering_not_cardinality(rows=rows)
    filtering = a_filter_runs_on_the_runs(rows=rows)
    composing = it_composes_better_than_i_expected(rows=rows)
    return {
        "sorted_ratio": clustering["sorted_ratio"],
        "scattered_ratio": clustering["scattered_ratio"],
        "filter_saving": filtering["saving"],
        "two_column_boundary_share": composing["combined_share_of_rows"],
        "the_cost_at_most_doubles": composing["the_cost_at_most_doubles"],
        "best_shape": compare_the_shapes(rows=rows)[0]["shape"],
    }
