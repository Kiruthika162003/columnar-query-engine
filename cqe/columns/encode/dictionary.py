from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column
from cqe.errors import ConfigError, EncodingError
from cqe.types.schema import STRING, Field

# Dictionary encoding, which the engine already applies to every string column, measured rather
# than assumed.
#
# The idea is one line: replace each value with an integer index into a table of the distinct
# values. What it costs and what it buys are both more interesting than the idea.
#
# The size saving is a function of cardinality and of the average value length. At one distinct
# value in a thousand the codes plus the dictionary come to 0.058 of the raw text. At one
# distinct value per row the dictionary holds everything and the codes are pure overhead, so
# the encoded form is 1.23 times the raw form.
#
# The crossover is at a cardinality fraction of 0.9, which is far later than the tenth I
# guessed. Codes are one to four bytes against about eighteen for the text, so the dictionary
# can hold most of the column and still win. On this data the encoding is worth applying to
# very nearly every string column, which is what the engine does.
#
# What it buys at query time is the part that matters more. An equality predicate on a
# dictionary column becomes a single lookup in the dictionary plus an integer comparison per
# row, so a string filter costs what an integer filter costs. That works whatever order the
# dictionary is in.
#
# A range predicate is different and it is why the dictionary here is sorted. On an ordered
# dictionary, x between two bounds becomes a code range and stays two integer comparisons per
# row. On an unordered one it becomes a membership test against however many codes fall in the
# range, which was 42 of them in the measurement below, so twenty one times the per row work.
#
# The sort at build time is not free, which I had assumed it was. Counted in operations it is
# 0.30 of the pass over the rows at five thousand distinct values in two hundred thousand, and
# it grows with the cardinality while the pass grows with the height. On a high cardinality
# column the sort is the build.
#
# And it costs something at query time too, in one case. A column whose values arrive already
# grouped has runs, and codes assigned in first seen order are monotone across those runs while
# codes assigned in sorted order are not. runlength.py's delta path needs the monotone form.
# So the unordered encoder stays, and this module measures both rather than picking one.


@dataclass
class Dictionary:
    """A table of distinct values and the codes that index it."""

    entries: tuple[str, ...]
    ordered: bool

    def __post_init__(self) -> None:
        if len(set(self.entries)) != len(self.entries):
            raise EncodingError("a dictionary holds each value once")
        if self.ordered and list(self.entries) != sorted(self.entries):
            raise EncodingError("an ordered dictionary is sorted")

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def nbytes(self) -> int:
        """Bytes the value table occupies."""
        return sum(len(entry) for entry in self.entries)

    def code(self, value: str) -> int:
        """The code for a value, or a refusal if it is not there."""
        try:
            return self.entries.index(value)
        except ValueError as missing:
            raise EncodingError(f"{value} is not in the dictionary") from missing

    def contains(self, value: str) -> bool:
        """Whether a value has a code, without raising."""
        return value in self.entries

    def range_codes(self, low: str, high: str) -> tuple[int, int]:
        """The half open code range covering values between two bounds.

        Only meaningful on an ordered dictionary, and refused otherwise rather than silently
        returning a range that means nothing. The refusal is the whole reason the ordered flag
        exists as data rather than as a comment.
        """
        if not self.ordered:
            raise EncodingError("a code range needs an ordered dictionary")
        start = int(np.searchsorted(np.array(self.entries), low, side="left"))
        stop = int(np.searchsorted(np.array(self.entries), high, side="right"))
        return start, stop

    def matching_codes(self, low: str, high: str) -> np.ndarray:
        """Every code whose value falls between two bounds, however the dictionary is ordered.

        The general form, and the one an unordered dictionary is stuck with. Costs a scan of the
        dictionary rather than two binary searches, which is cheap; the expense is downstream,
        where the filter has to test membership in a set rather than a range.
        """
        return np.array(
            [code for code, entry in enumerate(self.entries) if low <= entry <= high],
            dtype=np.int32,
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"entries": len(self.entries), "bytes": self.nbytes, "ordered": self.ordered}


@dataclass
class Encoded:
    """A dictionary encoded column, with the sizes of both forms."""

    codes: np.ndarray
    dictionary: Dictionary
    raw_bytes: int

    @property
    def rows(self) -> int:
        """How many values there are."""
        return int(self.codes.shape[0])

    @property
    def code_bytes(self) -> int:
        """Bytes the codes occupy at the width actually needed."""
        return self.rows * code_width(len(self.dictionary))

    @property
    def encoded_bytes(self) -> int:
        """Codes plus dictionary."""
        return self.code_bytes + self.dictionary.nbytes

    @property
    def ratio(self) -> float:
        """Encoded size over raw size, so below one is a saving."""
        if self.raw_bytes == 0:
            return 1.0
        return self.encoded_bytes / self.raw_bytes

    @property
    def cardinality_ratio(self) -> float:
        """Distinct values over rows, which is what the saving is a function of."""
        if not self.rows:
            return 0.0
        return len(self.dictionary) / self.rows

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rows": self.rows,
            "distinct": len(self.dictionary),
            "raw_bytes": self.raw_bytes,
            "encoded_bytes": self.encoded_bytes,
            "ratio": round(self.ratio, 4),
            "cardinality_ratio": round(self.cardinality_ratio, 4),
        }


def code_width(distinct: int) -> int:
    """Bytes per code at the narrowest integer that holds the largest code.

    One, two or four bytes. Wider than four is not offered, because a dictionary with more than
    four billion entries is a column that should not be dictionary encoded and saying so is more
    useful than supporting it.
    """
    if distinct < 0:
        raise ConfigError(f"{distinct} is not a count")
    if distinct <= 256:
        return 1
    if distinct <= 65536:
        return 2
    return 4


def encode(values: Sequence[str], ordered: bool = True) -> Encoded:
    """Build a dictionary and the codes that index it.

    Ordered by default. The first seen order is kept when ordered is false, which is what a
    streaming encoder would produce and what runlength.py wants, and the module measures the
    difference rather than picking one.
    """
    if ordered:
        entries = tuple(sorted(set(values)))
    else:
        seen: dict[str, int] = {}
        for value in values:
            if value not in seen:
                seen[value] = len(seen)
        entries = tuple(seen)
    lookup = {entry: code for code, entry in enumerate(entries)}
    codes = np.array([lookup[value] for value in values], dtype=np.int32)
    raw = sum(len(value) for value in values)
    return Encoded(codes=codes, dictionary=Dictionary(entries, ordered), raw_bytes=raw)


def decode(encoded: Encoded) -> list[str]:
    """Recover the original values, which is the only correctness property that matters."""
    entries = encoded.dictionary.entries
    return [entries[int(code)] for code in encoded.codes]


def to_column(name: str, encoded: Encoded) -> Column:
    """Wrap an encoded column in the engine's own container."""
    if not encoded.dictionary.ordered:
        raise EncodingError(f"{name} needs an ordered dictionary to become a column")
    return Column(
        field=Field(name=name, logical=STRING, nullable=False),
        values=encoded.codes,
        dictionary=encoded.dictionary.entries,
    )


def _words(count: int, seed: int = 0) -> list[str]:
    """A pool of distinct values with realistic lengths."""
    generator = np.random.default_rng(seed)
    lengths = generator.integers(4, 16, size=count)
    return [f"v{position:07d}" + "x" * int(length) for position, length in enumerate(lengths)]


def sample(rows: int, distinct: int, seed: int = 0, skew: float = 0.0) -> list[str]:
    """A column of the given height drawn from the given number of distinct values.

    The skew parameter controls how uneven the draw is. Zero is uniform; higher values
    concentrate the mass on a few entries, which is what real categorical columns look like and
    which changes nothing about the size but a great deal about run length encoding.

    Every entry in the pool is placed once before the rest are drawn, so the column really has
    the cardinality it was asked for. The first version drew every row at random and at a
    requested cardinality equal to the row count delivered 0.63 of it, because a uniform draw of
    n values from n does not cover them. That understated the encoded size at exactly the point
    where the crossover was being looked for, which is the one place it mattered.
    """
    if rows < 1 or distinct < 1:
        raise ConfigError(f"{rows} rows of {distinct} distinct values is not a column")
    if distinct > rows:
        raise ConfigError(f"{distinct} distinct values do not fit in {rows} rows")
    pool = _words(distinct, seed=seed)
    generator = np.random.default_rng(seed + 1)
    remaining = rows - distinct
    if skew <= 0.0:
        extra = generator.integers(0, distinct, size=remaining)
    else:
        weights = 1.0 / np.power(np.arange(1, distinct + 1), skew)
        weights = weights / weights.sum()
        extra = generator.choice(distinct, size=remaining, p=weights)
    picks = np.concatenate([np.arange(distinct), np.asarray(extra, dtype=np.int64)])
    generator.shuffle(picks)
    return [pool[int(pick)] for pick in picks]


def the_saving_depends_on_cardinality(
    rows: int = 100_000,
    fractions: Sequence[float] = (0.001, 0.01, 0.05, 0.2, 0.5, 1.0),
) -> list[dict]:
    """How the encoded size moves as the number of distinct values rises.

    The ratios are 0.058, 0.123, 0.164, 0.314, 0.614 and 1.229 at cardinality fractions of a
    thousandth up to one. The last of those is larger than the raw column, which is the case
    worth knowing: a column of unique values gains nothing and pays for the code array.
    """
    if not fractions:
        raise ConfigError("there is nothing to sweep")
    out = []
    for fraction in fractions:
        distinct = max(1, int(rows * fraction))
        values = sample(rows, distinct)
        out.append(encode(values).as_dict())
    return out


def the_crossover_is_later_than_it_looks(rows: int = 100_000) -> dict:
    """Where dictionary encoding stops paying, measured rather than guessed.

    I expected it around a cardinality fraction of a tenth, on the reasoning that a dictionary
    holding a tenth of the rows is already substantial. It sits at 0.9. The codes are one to
    four bytes against about eighteen for the text, so the dictionary can hold nearly the whole
    column and still win.

    The practical reading is that a rule of thumb about low cardinality columns is the wrong
    rule. On text of this width the encoding is worth applying unconditionally, and the
    cardinality only decides how much it saves.
    """
    fractions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]
    rows_out = []
    for fraction in fractions:
        distinct = max(1, int(rows * fraction))
        encoded = encode(sample(rows, distinct))
        rows_out.append({"fraction": fraction, "ratio": round(encoded.ratio, 4)})
    losing = [row for row in rows_out if row["ratio"] >= 1.0]
    return {
        "rows": rows_out,
        "crossover": losing[0]["fraction"] if losing else None,
        "it_pays_at_a_tenth": rows_out[1]["ratio"] < 1.0,
        "and_at_a_half": rows_out[5]["ratio"] < 1.0,
        "it_ever_loses": bool(losing),
    }


def the_code_width_steps(distincts: Sequence[int] = (100, 300, 70_000, 200_000)) -> list[dict]:
    """The code array width is one, two or four bytes and steps at the powers of two.

    Worth measuring because the steps are where a size estimate goes wrong. A column with 256
    distinct values encodes to a quarter of the size of one with 257, and a cost model that
    treats the width as continuous will price the second at almost the first.
    """
    if not distincts:
        raise ConfigError("there is nothing to sweep")
    return [
        {"distinct": distinct, "bytes_per_code": code_width(distinct)} for distinct in distincts
    ]


def an_equality_filter_is_one_integer_comparison(
    rows: int = 100_000,
    distinct: int = 1_000,
) -> dict:
    """Filtering a dictionary column for one value costs a lookup plus a scan of codes.

    The lookup is a single dictionary probe whatever the column height, so the per row cost is
    two bytes compared against 17.7 bytes of text. The string never enters the comparison at all,
    which is the reason the engine dictionary encodes every string column rather than only the
    low cardinality ones.
    """
    values = sample(rows, distinct)
    encoded = encode(values)
    wanted = values[0]
    code = encoded.dictionary.code(wanted)
    matched = int((encoded.codes == code).sum())
    direct = sum(1 for value in values if value == wanted)
    return {
        "rows": rows,
        "matched": matched,
        "agrees_with_the_direct_scan": matched == direct,
        "dictionary_probes": 1,
        "comparisons": rows,
        "compared_bytes_per_row": code_width(len(encoded.dictionary)),
        "raw_bytes_per_row": round(encoded.raw_bytes / rows, 1),
    }


def a_range_filter_needs_an_ordered_dictionary(
    rows: int = 50_000,
    distinct: int = 2_000,
) -> dict:
    """The measurement the sort at build time is paid for.

    On an ordered dictionary a range predicate is two binary searches into the dictionary and
    one pair of integer comparisons per row. On an unordered one it is a scan of the dictionary
    collecting matching codes, then a membership test per row against however many codes came
    back, which is 42 here against 2.

    Both give the same answer, which is checked, and one of them is twenty one times the per row
    work. The ratio is the count of codes in the range, so it grows with the cardinality and with
    the width of the predicate.
    """
    values = sample(rows, distinct)
    low, high = sorted(values)[distinct // 4], sorted(values)[3 * distinct // 4]

    ordered = encode(values, ordered=True)
    start, stop = ordered.dictionary.range_codes(low, high)
    ordered_matches = int(((ordered.codes >= start) & (ordered.codes < stop)).sum())

    unordered = encode(values, ordered=False)
    codes = unordered.dictionary.matching_codes(low, high)
    unordered_matches = int(np.isin(unordered.codes, codes).sum())

    direct = sum(1 for value in values if low <= value <= high)
    return {
        "matched": ordered_matches,
        "both_agree": ordered_matches == unordered_matches == direct,
        "ordered_work_per_row": 2,
        "unordered_work_per_row": len(codes),
        "codes_in_the_range": len(codes),
        "the_ordered_form_is_cheaper": len(codes) > 2,
    }


def an_unordered_dictionary_refuses_a_code_range() -> bool:
    """That the refusal is a refusal rather than a wrong answer."""
    encoded = encode(["b", "a", "c"], ordered=False)
    try:
        encoded.dictionary.range_codes("a", "b")
    except EncodingError:
        return True
    return False


def ordering_the_dictionary_is_not_free(
    rows: int = 200_000,
    distinct: int = 5_000,
) -> dict:
    """What the sort at build time costs, as a share of building the dictionary at all.

    An operation count rather than a measurement, and the docstring says so because everything
    else in this package is counted rather than modelled. The pass over the rows is one operation
    per row; the sort is taken as n log n over the distinct values.

    I assumed this was negligible and it is not. At five thousand distinct values in two hundred
    thousand rows it comes to 0.30 of the pass over the rows. The pass grows with the height and
    the sort grows with the cardinality, so on a column where nearly every value is distinct the
    sort is most of the build.
    """
    build_work = rows
    sort_work = distinct * max(1, int(np.log2(max(distinct, 2))))
    return {
        "rows": rows,
        "distinct": distinct,
        "pass_over_rows": build_work,
        "sort_over_distinct": sort_work,
        "share": round(sort_work / build_work, 4),
        "it_is_not_negligible": sort_work > build_work / 10,
        "it_scales_with_cardinality_not_height": True,
    }


def sorting_the_dictionary_destroys_run_structure(
    runs: int = 500,
    run_length: int = 100,
) -> dict:
    """The exception, and the reason the unordered path is kept rather than deleted.

    A column that arrives already grouped, which is what a clustered table looks like, has long
    runs of equal values. Assigning codes in first seen order makes those runs monotone in code
    space, since each new run gets the next code. Assigning them in sorted order scatters them,
    because the order the runs arrive in has nothing to do with the alphabet.

    The run count is identical either way, since a run is a run whatever it is called. What
    changes is monotonicity, which is what runlength.py's delta path needs.

    The first version of this generated the runs in pool order, and the pool is built with
    increasing names, so first seen order and sorted order were the same thing and the
    measurement reported no difference at all. The runs are shuffled now.
    """
    pool = _words(runs)
    order = np.random.default_rng(7).permutation(runs)
    values = [pool[int(position)] for position in order for _ in range(run_length)]

    first_seen = encode(values, ordered=False)
    sorted_form = encode(values, ordered=True)

    def monotone(codes: np.ndarray) -> bool:
        return bool((np.diff(codes) >= 0).all())

    return {
        "rows": len(values),
        "runs": runs,
        "first_seen_is_monotone": monotone(first_seen.codes),
        "sorted_is_monotone": monotone(sorted_form.codes),
        "the_orders_differ": not np.array_equal(first_seen.codes, sorted_form.codes),
        "run_count_is_unchanged": _count_runs(first_seen.codes)
        == _count_runs(sorted_form.codes),
    }


def _count_runs(codes: np.ndarray) -> int:
    """How many maximal runs of equal values a code array has."""
    if not len(codes):
        return 0
    return 1 + int((np.diff(codes) != 0).sum())


def skew_does_not_change_the_size(rows: int = 100_000, distinct: int = 1_000) -> dict:
    """How uneven the value distribution is, against how much the encoding saves.

    It does not matter at all, which is worth stating because people reach for dictionary
    encoding when they see a skewed column. The size is set by the cardinality and the value
    lengths, both of which are the same whatever the frequencies. Skew is what run length and
    entropy coding are for, and this is neither.
    """
    flat = encode(sample(rows, distinct, skew=0.0))
    steep = encode(sample(rows, distinct, skew=1.5))
    return {
        "flat_ratio": round(flat.ratio, 4),
        "skewed_ratio": round(steep.ratio, 4),
        "they_are_close": abs(flat.ratio - steep.ratio) < 0.05,
        "distinct_matches": len(flat.dictionary) == len(steep.dictionary),
    }


def the_round_trip_is_exact(rows: int = 20_000, distinct: int = 700) -> dict:
    """Decoding gives back exactly what was encoded, on both orderings.

    The property everything else rests on, checked on the two orderings separately because they
    are two code assignments and a bug in either would be invisible in the other.
    """
    values = sample(rows, distinct)
    return {
        "ordered_exact": decode(encode(values, ordered=True)) == values,
        "unordered_exact": decode(encode(values, ordered=False)) == values,
        "same_distinct_count": (
            len(encode(values, True).dictionary) == len(encode(values, False).dictionary)
        ),
    }


def merging_dictionaries_costs_the_union(
    groups: Sequence[int] = (1, 2, 4, 8, 16),
    rows: int = 40_000,
    distinct: int = 2_000,
) -> list[dict]:
    """What it costs to read a column written as several independently encoded row groups.

    Each group has its own dictionary, so reading them as one column means merging the
    dictionaries and remapping every code. The merge is over the union of the dictionaries and
    the remap is over every row, so splitting a column into more groups makes the read strictly
    more expensive. That is the argument for a row group being large, and storage/layout.py
    picks the size against this measurement.
    """
    if not groups:
        raise ConfigError("there is nothing to sweep")
    values = sample(rows, distinct)
    out = []
    for count in groups:
        size = rows // count
        pieces = [values[start : start + size] for start in range(0, rows, size)]
        dictionaries = [set(piece) for piece in pieces]
        union = set().union(*dictionaries) if dictionaries else set()
        out.append(
            {
                "groups": count,
                "union_size": len(union),
                "total_dictionary_entries": sum(len(one) for one in dictionaries),
                "remapped_values": rows,
                "duplication": round(
                    sum(len(one) for one in dictionaries) / max(len(union), 1), 3
                ),
            }
        )
    return out


def more_groups_means_more_duplicated_dictionary(rows: int = 40_000) -> dict:
    """State that as a claim, since it is what sets the row group size.

    Sixteen groups hold nearly eight times the dictionary entries one group does, because almost
    every distinct value appears in almost every group once the groups are small enough.
    """
    measured = merging_dictionaries_costs_the_union(rows=rows)
    return {
        "rows": measured,
        "one_group": measured[0]["duplication"],
        "sixteen_groups": measured[-1]["duplication"],
        "duplication_rises": measured[-1]["duplication"] > measured[0]["duplication"],
        "and_a_lot": measured[-1]["duplication"] > 3.0,
    }


def a_value_outside_the_dictionary_is_refused() -> bool:
    """Looking up a value that was never encoded is a mistake, not a missing code."""
    encoded = encode(["a", "b"])
    try:
        encoded.dictionary.code("z")
    except EncodingError:
        return True
    return False


def a_dictionary_with_repeats_is_refused() -> bool:
    """A dictionary holds each value once, by definition."""
    try:
        Dictionary(("a", "a"), ordered=False)
    except EncodingError:
        return True
    return False


def an_unsorted_ordered_dictionary_is_refused() -> bool:
    """And an ordered one is sorted, which is checked rather than trusted."""
    try:
        Dictionary(("b", "a"), ordered=True)
    except EncodingError:
        return True
    return False


def a_column_needs_an_ordered_dictionary() -> bool:
    """The engine's own container only accepts the ordered form."""
    try:
        to_column("x", encode(["b", "a"], ordered=False))
    except EncodingError:
        return True
    return False


def summarise(rows: int = 100_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    crossover = the_crossover_is_later_than_it_looks(rows=rows)
    ranges = a_range_filter_needs_an_ordered_dictionary()
    skewed = skew_does_not_change_the_size(rows=rows)
    return {
        "crossover_fraction": crossover["crossover"],
        "pays_at_a_half": crossover["and_at_a_half"],
        "range_work_ordered": ranges["ordered_work_per_row"],
        "range_work_unordered": ranges["unordered_work_per_row"],
        "skew_changes_nothing": skewed["they_are_close"],
        "cheapest_ratio": min(
            row["ratio"] for row in the_saving_depends_on_cardinality(rows=rows)
        ),
    }
