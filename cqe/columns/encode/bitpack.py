from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.errors import ConfigError, EncodingError

# Bit packing and frame of reference, which are the same idea applied twice.
#
# An integer column stored as int64 spends sixty four bits on every value whatever the values
# are. A column of ages needs seven. A column of order identifiers between ten million and ten
# million and a thousand needs ten, once the ten million is subtracted off and kept once. Bit
# packing is the first of those and frame of reference is the second, and they compose: subtract
# the minimum, then pack to the bits the range needs.
#
# The saving is exactly sixty four over the bit width, which makes this the most predictable
# encoding in the package and the one with the least to measure. What is worth measuring is
# everything around that number.
#
# The first thing is that the width is set by the single widest value and nothing else. One
# outlier in a million rows takes the column from seven bits to thirty two, so the encoding is
# not robust in the statistical sense at all. The measurement below puts a single value of two
# billion into a column of ages and the ratio goes from 0.11 to 0.5.
#
# The second is that packing to a width that is not a power of two costs something at read time,
# because a value then straddles a machine word and has to be assembled from two loads and a
# shift. This implementation packs to arbitrary widths and the measurement counts the straddling
# values. At seven bits 0.094 of them straddle a sixty four bit boundary and at eight bits none
# do, and the share rises with the width rather than staying flat: 0.125 at nine bits, 0.25 at
# seventeen. That is the argument for rounding a width up to a byte, and the measurement says it
# costs 0.143 of the size to do it.
#
# The third is the interaction with nulls. A packed column has no room for a null, so the
# validity mask has to be carried separately, and at one bit per value it is a real share of a
# seven bit column. The module measures it rather than ignoring it, because ignoring it is how a
# size estimate comes out a fifth too low.


@dataclass
class Packed:
    """A bit packed column and everything needed to unpack it."""

    words: np.ndarray
    bits: int
    rows: int
    reference: int

    def __post_init__(self) -> None:
        if self.bits < 1 or self.bits > 64:
            raise EncodingError(f"{self.bits} is not a bit width")
        if self.rows < 0:
            raise ConfigError(f"{self.rows} is not a row count")

    @property
    def nbytes(self) -> int:
        """Bytes the packed words occupy, plus the reference value."""
        return int(self.words.nbytes) + 8

    def ratio(self, source_width: int = 8) -> float:
        """Packed size over unpacked size, so below one is a saving."""
        if self.rows == 0:
            return 1.0
        return self.nbytes / (self.rows * source_width)

    @property
    def straddling(self) -> int:
        """How many values cross a sixty four bit word boundary.

        The cost bit packing hides. A value inside one word is one load and a mask; a value
        across two is two loads, two shifts and an or. Counting them is the only honest way to
        compare a seven bit packing against an eight bit one, since the sizes say the seven bit
        form wins and the loads say otherwise.
        """
        if self.rows == 0:
            return 0
        starts = np.arange(self.rows, dtype=np.int64) * self.bits
        return int(((starts % 64) + self.bits > 64).sum())

    @property
    def straddle_share(self) -> float:
        """The same as a fraction of the rows."""
        if self.rows == 0:
            return 0.0
        return self.straddling / self.rows

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rows": self.rows,
            "bits": self.bits,
            "reference": self.reference,
            "bytes": self.nbytes,
            "ratio": round(self.ratio(), 4),
            "straddle_share": round(self.straddle_share, 4),
        }


def bits_needed(span: int) -> int:
    """The bit width that holds every value in a range of the given span.

    A span of zero still needs one bit, because a column of one repeated value is a column and
    zero bits per value would make the row count unrecoverable from the packed form.
    """
    if span < 0:
        raise ConfigError(f"{span} is not a span")
    if span == 0:
        return 1
    return int(span).bit_length()


def plan(values: np.ndarray) -> tuple[int, int]:
    """The reference value and bit width a column would pack to."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise EncodingError(f"a column is one dimensional, not {array.ndim}")
    if not len(array):
        return 0, 1
    low = int(array.min())
    high = int(array.max())
    return low, bits_needed(high - low)


def pack(values: np.ndarray, bits: int | None = None) -> Packed:
    """Subtract the minimum and pack every value into the given number of bits.

    Written against numpy's own packbits rather than by hand, because a hand written bit
    shuffler is the kind of code that works on every test somebody thought of and fails on the
    one where the row count is not a multiple of eight.
    """
    array = np.asarray(values)
    reference, needed = plan(array)
    width = needed if bits is None else bits
    if width < needed:
        raise EncodingError(f"{width} bits cannot hold a span needing {needed}")
    if not len(array):
        return Packed(words=np.array([], dtype=np.uint8), bits=width, rows=0, reference=0)
    offsets = (array - reference).astype(np.uint64)
    spread = ((offsets[:, None] >> np.arange(width - 1, -1, -1, dtype=np.uint64)) & 1).astype(
        np.uint8
    )
    return Packed(
        words=np.packbits(spread.reshape(-1)),
        bits=width,
        rows=int(array.shape[0]),
        reference=reference,
    )


def unpack(packed: Packed) -> np.ndarray:
    """Recover the original values."""
    if packed.rows == 0:
        return np.array([], dtype=np.int64)
    flat = np.unpackbits(packed.words)[: packed.rows * packed.bits]
    spread = flat.reshape(packed.rows, packed.bits).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(packed.bits - 1, -1, -1, dtype=np.uint64)).astype(
        np.uint64
    )
    offsets = (spread * weights).sum(axis=1)
    return (offsets.astype(np.int64) + packed.reference).astype(np.int64)


def narrow(rows: int, span: int, low: int = 0, seed: int = 0) -> np.ndarray:
    """A column whose values sit in a narrow band, which is what this encoding is for."""
    if rows < 1 or span < 1:
        raise ConfigError(f"{rows} rows over a span of {span} is not a column")
    return np.random.default_rng(seed).integers(low, low + span, size=rows).astype(np.int64)


def the_saving_is_the_bit_width(
    rows: int = 100_000,
    spans: Sequence[int] = (2, 16, 256, 65_536, 2**31),
) -> list[dict]:
    """The ratio is sixty four over the width and there is nothing else in it.

    The most predictable encoding here, which is why the module spends its measurements on the
    things around it rather than on this. A span of 256 needs eight bits and gives a ratio of
    0.125; a span of two billion needs thirty one and gives 0.484.
    """
    if not spans:
        raise ConfigError("there is nothing to sweep")
    out = []
    for span in spans:
        packed = pack(narrow(rows, span))
        row = packed.as_dict()
        row["span"] = span
        out.append(row)
    return out


def one_outlier_sets_the_width(rows: int = 100_000) -> dict:
    """The width is the widest value and the encoding has no defence against a single one.

    A column of ages packs to seven bits. The same column with one value of two billion in it
    packs to thirty one, because the width is set by the range and the range is set by the
    extremes. The ratio goes from 0.11 to 0.49 for one row in a hundred thousand.

    Nothing here fixes that, and that is the point of measuring it: an encoder that chose bit
    packing on a sample would pick it, and the outlier would arrive later.
    """
    ages = narrow(rows, span=120, low=0)
    clean = pack(ages)
    spoiled = ages.copy()
    spoiled[rows // 2] = 2_000_000_000
    dirty = pack(spoiled)
    return {
        "clean_bits": clean.bits,
        "spoiled_bits": dirty.bits,
        "clean_ratio": round(clean.ratio(), 4),
        "spoiled_ratio": round(dirty.ratio(), 4),
        "one_row_in": rows,
        "the_width_more_than_quadrupled": dirty.bits > 4 * clean.bits,
        "the_ratio_more_than_tripled": dirty.ratio() > 3 * clean.ratio(),
    }


def frame_of_reference_is_what_makes_it_work(rows: int = 100_000) -> dict:
    """Subtracting the minimum, on a column where the values are large and the range is not.

    Order identifiers around ten million with a span of a thousand. Without the reference the
    width is set by the absolute magnitude and comes to twenty four bits. With it the width is
    set by the span and comes to ten. The reference itself costs eight bytes once.
    """
    values = narrow(rows, span=1_000, low=10_000_000)
    with_reference = pack(values)
    absolute_bits = bits_needed(int(values.max()))
    return {
        "with_reference_bits": with_reference.bits,
        "without_reference_bits": absolute_bits,
        "saving_in_bits": absolute_bits - with_reference.bits,
        "reference": with_reference.reference,
        "the_reference_costs_eight_bytes": True,
        "it_more_than_halves_the_width": with_reference.bits * 2 < absolute_bits,
    }


def a_width_that_is_not_a_byte_straddles_words(
    rows: int = 10_000,
    widths: Sequence[int] = (1, 2, 4, 7, 8, 9, 16, 17, 32),
) -> list[dict]:
    """How many values cross a sixty four bit boundary at each width.

    A power of two divides sixty four, so no value straddles. Anything else does, and the share
    is what a reader pays twice for: 0.094 at seven bits, 0.125 at nine, 0.25 at seventeen.

    The share is smaller than I expected and it grows with the width rather than staying roughly
    constant, because a wide value has more ways to land across a boundary. So the case for
    rounding up is weakest exactly where packing tightly saves least.
    """
    if not widths:
        raise ConfigError("there is nothing to sweep")
    out = []
    for width in widths:
        span = min(2 ** min(width, 40) - 1, 2**40)
        packed = pack(narrow(rows, span=max(span, 1)), bits=width)
        out.append(
            {
                "bits": width,
                "straddle_share": round(packed.straddle_share, 4),
                "divides_sixty_four": 64 % width == 0,
            }
        )
    return out


def rounding_up_to_a_byte_costs_little(rows: int = 100_000) -> dict:
    """What it costs to pack to eight bits instead of seven, against what it saves in loads.

    An eighth more space for no straddling values at all. Whether that is worth taking depends
    on whether the column is read more often than it is written, which it always is, so the
    engine rounds up. Recorded here rather than in the code so the number is checkable.
    """
    values = narrow(rows, span=120)
    tight = pack(values, bits=7)
    padded = pack(values, bits=8)
    return {
        "tight_bits": tight.bits,
        "padded_bits": padded.bits,
        "tight_bytes": tight.nbytes,
        "padded_bytes": padded.nbytes,
        "extra_size": round(padded.nbytes / tight.nbytes - 1, 4),
        "tight_straddle": round(tight.straddle_share, 4),
        "padded_straddle": round(padded.straddle_share, 4),
        "the_padding_removes_every_straddle": padded.straddle_share == 0.0,
        "and_costs_under_a_fifth": padded.nbytes / tight.nbytes < 1.2,
    }


def the_validity_mask_is_a_real_share(rows: int = 100_000) -> dict:
    """A packed column has no room for a null, so the mask is separate and it is not small.

    At seven bits per value a one bit mask is a seventh of the column again. Every size estimate
    that quotes a bit packing ratio without the mask is that much too low, and the ratio quoted
    for narrow columns is exactly where it matters most.
    """
    values = narrow(rows, span=120)
    packed = pack(values)
    mask_bits = rows
    return {
        "value_bits": packed.bits,
        "packed_bytes": packed.nbytes,
        "mask_bytes": mask_bits // 8,
        "mask_share": round((mask_bits // 8) / packed.nbytes, 4),
        "it_is_over_a_tenth": (mask_bits // 8) / packed.nbytes > 0.1,
        "with_mask_ratio": round((packed.nbytes + mask_bits // 8) / (rows * 8), 4),
        "without_mask_ratio": round(packed.ratio(), 4),
    }


def the_round_trip_is_exact(rows: int = 20_000) -> dict:
    """Unpacking gives back exactly what was packed, at every awkward width and length.

    The lengths are chosen to fall off the byte boundary, because a packer that works on
    multiples of eight and fails otherwise passes every casual test.
    """
    cases = {}
    for width in (1, 3, 7, 8, 13, 17, 32):
        for length in (rows, rows + 1, rows + 7):
            values = narrow(length, span=2**width - 1)
            packed = pack(values, bits=width)
            cases[f"{width} bits {length} rows"] = bool(np.array_equal(unpack(packed), values))
    return cases


def negative_values_survive_the_reference(rows: int = 10_000) -> dict:
    """A column spanning zero, which the reference has to handle and often does not.

    Values between minus five hundred and five hundred. The reference is negative, the offsets
    are unsigned, and the round trip has to come back through zero correctly. Worth a
    measurement because subtracting a negative is where an encoder written for counts breaks.
    """
    values = narrow(rows, span=1_000, low=-500)
    packed = pack(values)
    return {
        "reference": packed.reference,
        "bits": packed.bits,
        "reference_is_negative": packed.reference < 0,
        "round_trips": bool(np.array_equal(unpack(packed), values)),
        "spans_zero": bool(values.min() < 0 < values.max()),
    }


def an_empty_column_packs_to_nothing() -> dict:
    """The degenerate case."""
    packed = pack(np.array([], dtype=np.int64))
    return {
        "rows": packed.rows,
        "bytes": packed.nbytes,
        "ratio": packed.ratio(),
        "round_trips": len(unpack(packed)) == 0,
        "straddles_nothing": packed.straddling == 0,
    }


def a_constant_column_needs_one_bit(rows: int = 10_000) -> dict:
    """A column of one repeated value, which is the narrowest a span can be."""
    values = np.full(rows, 42, dtype=np.int64)
    packed = pack(values)
    return {
        "bits": packed.bits,
        "reference": packed.reference,
        "it_is_one_bit": packed.bits == 1,
        "round_trips": bool(np.array_equal(unpack(packed), values)),
        "ratio": round(packed.ratio(), 5),
    }


def a_width_too_narrow_is_refused() -> bool:
    """Packing into fewer bits than the span needs is a mistake, not a truncation."""
    try:
        pack(np.array([0, 1000], dtype=np.int64), bits=4)
    except EncodingError:
        return True
    return False


def a_width_past_sixty_four_is_refused() -> bool:
    """And there is no packing wider than the word it is packed into."""
    try:
        Packed(words=np.array([], dtype=np.uint8), bits=65, rows=0, reference=0)
    except EncodingError:
        return True
    return False


def a_two_dimensional_column_is_refused() -> bool:
    """The packer takes a column, not a table."""
    try:
        plan(np.zeros((2, 2)))
    except EncodingError:
        return True
    return False


def a_negative_span_is_refused() -> bool:
    """A span is a distance and distances are not negative."""
    try:
        bits_needed(-1)
    except ConfigError:
        return True
    return False


def compare_the_widths(rows: int = 100_000) -> list[dict]:
    """Every span from a bit to a word, as one table."""
    out = []
    for width in (1, 4, 8, 12, 16, 24, 32):
        packed = pack(narrow(rows, span=2**width - 1), bits=width)
        out.append(
            {
                "bits": width,
                "ratio": round(packed.ratio(), 4),
                "straddle_share": round(packed.straddle_share, 4),
                "bytes": packed.nbytes,
            }
        )
    return out


def summarise(rows: int = 100_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    outlier = one_outlier_sets_the_width(rows=rows)
    reference = frame_of_reference_is_what_makes_it_work(rows=rows)
    padding = rounding_up_to_a_byte_costs_little(rows=rows)
    mask = the_validity_mask_is_a_real_share(rows=rows)
    return {
        "clean_ratio": outlier["clean_ratio"],
        "spoiled_ratio": outlier["spoiled_ratio"],
        "reference_saves_bits": reference["saving_in_bits"],
        "padding_costs": padding["extra_size"],
        "mask_share": mask["mask_share"],
        "widest_useful_ratio": max(row["ratio"] for row in compare_the_widths(rows=rows)),
    }
