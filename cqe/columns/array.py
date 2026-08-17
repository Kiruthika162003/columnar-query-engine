from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.errors import ConfigError, DataError, SchemaError, TypeMismatch
from cqe.types.schema import (
    BOOLEAN,
    DATE,
    FLOATING,
    INTEGER,
    PHYSICAL,
    STRING,
    Field,
    infer_logical,
)

# One column of values, which is the only container the engine has.
#
# Three pieces: a physical numpy array, an optional validity mask saying which entries are real,
# and an optional dictionary for string columns. Everything else in the package is built on this
# and on nothing else, so the decisions here propagate.
#
# The validity mask is a boolean numpy array rather than a packed bitmap, and that is a
# deliberate loss. A packed bitmap is eight times smaller and every operation on it needs
# unpacking before numpy will use it, which costs more than it saves at these sizes. The packed
# form exists in storage/layout.py, where the size actually matters because it is written to
# disk. In memory the mask is a byte per value and the module says so rather than pretending
# otherwise.
#
# The mask is None when nothing is null, which is the common case, and every operation checks
# for None before touching it. That branch is the difference between a filter that runs one
# numpy comparison and one that runs a comparison plus an and, on every batch, forever.
#
# Strings are dictionary encoded at rest and at runtime, always, with no undictionaried path.
# The alternative is an array of Python objects, which makes every operator either slow or
# special cased. Carrying the dictionary means a string comparison is an int32 comparison and a
# string group by is an int32 group by, and the only operation that has to look at the actual
# text is printing.


class EncodingRequired(SchemaError):
    """A string column arrived without the dictionary it cannot work without."""


@dataclass
class Column:
    """A named, typed array of values with optional nulls."""

    field: Field
    values: np.ndarray
    valid: np.ndarray | None = None
    dictionary: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise DataError(f"a column is one dimensional, not {self.values.ndim}")
        expected = PHYSICAL[self.field.logical]
        if self.values.dtype != expected:
            raise TypeMismatch(
                f"{self.field.name} holds {self.values.dtype} where {self.field.logical} "
                f"needs {expected}"
            )
        if self.valid is not None:
            if self.valid.dtype != np.bool_:
                raise DataError(f"a validity mask is boolean, not {self.valid.dtype}")
            if self.valid.shape != self.values.shape:
                raise DataError(
                    f"{self.valid.shape[0]} validity entries against "
                    f"{self.values.shape[0]} values"
                )
        if self.field.logical == STRING and self.dictionary is None:
            raise EncodingRequired(f"{self.field.name} is a string column with no dictionary")
        if self.dictionary is not None and self.field.logical != STRING:
            raise TypeMismatch(f"{self.field.logical} columns do not carry a dictionary")
        if self.dictionary is not None and len(self.values):
            highest = int(self.values.max())
            if highest >= len(self.dictionary):
                raise DataError(
                    f"code {highest} against a dictionary of {len(self.dictionary)}"
                )

    @property
    def name(self) -> str:
        """The column's name."""
        return self.field.name

    @property
    def logical(self) -> str:
        """The column's logical type."""
        return self.field.logical

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def null_count(self) -> int:
        """How many entries are null."""
        if self.valid is None:
            return 0
        return int((~self.valid).sum())

    @property
    def has_nulls(self) -> bool:
        """Whether anything is null, checked without counting."""
        return self.valid is not None and not bool(self.valid.all())

    @property
    def nbytes(self) -> int:
        """Memory occupied, including the mask and the dictionary text."""
        total = int(self.values.nbytes)
        if self.valid is not None:
            total += int(self.valid.nbytes)
        if self.dictionary is not None:
            total += sum(len(entry) for entry in self.dictionary)
        return total

    @property
    def distinct_estimate(self) -> int:
        """An exact distinct count for dictionary columns, which is what the dictionary is.

        Exact rather than estimated only because the dictionary already holds the answer. For
        every other type this is a real counting problem and stats/sketch.py does it properly.
        """
        if self.dictionary is not None:
            return len(self.dictionary)
        return int(np.unique(self.values).shape[0])

    def take(self, positions: np.ndarray) -> Column:
        """The values at the given row positions, in the order given.

        The single operation the whole engine is built on. A filter is a take with a positions
        array from a comparison, a sort is a take with a positions array from an argsort, and a
        join is a take on each side with positions from a hash table.
        """
        if positions.dtype.kind not in "iu":
            raise DataError(f"positions are integers, not {positions.dtype}")
        if len(positions) and len(self):
            highest = int(positions.max())
            if highest >= len(self):
                raise DataError(f"position {highest} against a column of {len(self)}")
        return Column(
            field=self.field,
            values=self.values[positions],
            valid=None if self.valid is None else self.valid[positions],
            dictionary=self.dictionary,
        )

    def mask(self, keep: np.ndarray) -> Column:
        """The values where a boolean mask is true."""
        if keep.dtype != np.bool_:
            raise DataError(f"a mask is boolean, not {keep.dtype}")
        if keep.shape[0] != len(self):
            raise DataError(f"{keep.shape[0]} mask entries against {len(self)} values")
        return self.take(np.flatnonzero(keep))

    def slice(self, start: int, stop: int | None = None) -> Column:
        """A contiguous run, which is what a batch boundary produces."""
        end = len(self) if stop is None else stop
        if start < 0 or end > len(self) or start > end:
            raise ConfigError(f"slice {start} to {end} is outside a column of {len(self)}")
        return Column(
            field=self.field,
            values=self.values[start:end],
            valid=None if self.valid is None else self.valid[start:end],
            dictionary=self.dictionary,
        )

    def renamed(self, name: str) -> Column:
        """The same values under a different column name."""
        return Column(
            field=self.field.renamed(name),
            values=self.values,
            valid=self.valid,
            dictionary=self.dictionary,
        )

    def fill_null(self, value) -> Column:
        """Replace nulls with a value and drop the mask.

        Not a general purpose convenience. Aggregation needs it to make a sum of a column with
        nulls into a sum over zeros, and doing it here rather than in the aggregate keeps the
        rule that a null contributes nothing in one place.
        """
        if self.valid is None:
            return self
        filled = self.values.copy()
        filled[~self.valid] = value
        return Column(field=self.field, values=filled, dictionary=self.dictionary)

    def to_list(self) -> list:
        """Python values, with None for nulls and text for strings.

        The bridge to the reference interpreter in verify/reference.py, and the only place the
        engine converts out of its own representation. Slow by construction and used only where
        being obviously correct matters more than being fast.
        """
        out: list = []
        for position in range(len(self)):
            if self.valid is not None and not self.valid[position]:
                out.append(None)
            elif self.dictionary is not None:
                out.append(self.dictionary[int(self.values[position])])
            elif self.logical == BOOLEAN:
                out.append(bool(self.values[position]))
            elif self.logical == FLOATING:
                out.append(float(self.values[position]))
            else:
                out.append(int(self.values[position]))
        return out

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "type": self.logical,
            "rows": len(self),
            "nulls": self.null_count,
            "bytes": self.nbytes,
            "distinct": self.distinct_estimate,
        }


def column_from(name: str, values: Sequence, logical: str | None = None) -> Column:
    """Build a column from Python values, inferring the type when not told.

    The constructor every test uses. Nulls are written as None and become a validity mask;
    strings are collected into a sorted dictionary, because an ordered dictionary is what lets a
    comparison on codes mean a comparison on text and columns/encode/dictionary.py measures what
    the ordering costs.
    """
    kind = logical or infer_logical(values)
    present = [value for value in values if value is not None]
    nullable = len(present) != len(values)
    field = Field(name=name, logical=kind, nullable=nullable)

    if kind == STRING:
        dictionary = tuple(sorted({str(value) for value in present}))
        lookup = {entry: code for code, entry in enumerate(dictionary)}
        codes = np.array(
            [0 if value is None else lookup[str(value)] for value in values],
            dtype=np.int32,
        )
        valid = None if not nullable else np.array([v is not None for v in values], dtype=bool)
        return Column(field=field, values=codes, valid=valid, dictionary=dictionary)

    dtype = PHYSICAL[kind]
    blank = False if kind == BOOLEAN else 0
    raw = np.array(
        [blank if value is None else value for value in values],
        dtype=dtype,
    )
    valid = None if not nullable else np.array([v is not None for v in values], dtype=bool)
    return Column(field=field, values=raw, valid=valid)


def concat(columns: Sequence[Column]) -> Column:
    """Stack several columns of the same field into one.

    String columns are merged onto a shared dictionary rather than assumed to share one, because
    two batches read from two files have no reason to agree on their codes. That merge is the
    expensive part of concatenating string columns and is the reason storage/layout.py writes
    one dictionary per row group rather than one per page.
    """
    if not columns:
        raise ConfigError("there is nothing to concatenate")
    first = columns[0]
    for other in columns[1:]:
        if other.field.logical != first.field.logical:
            raise TypeMismatch(f"{first.field.logical} against {other.field.logical}")
        if other.name != first.name:
            raise SchemaError(f"{first.name} against {other.name}")

    if first.dictionary is not None:
        merged = tuple(sorted({entry for one in columns for entry in one.dictionary or ()}))
        lookup = {entry: code for code, entry in enumerate(merged)}
        pieces = []
        for one in columns:
            table = np.array(
                [lookup[entry] for entry in (one.dictionary or ())], dtype=np.int32
            )
            pieces.append(table[one.values] if len(one) else one.values)
        values = np.concatenate(pieces) if pieces else np.array([], dtype=np.int32)
        dictionary: tuple[str, ...] | None = merged
    else:
        values = np.concatenate([one.values for one in columns])
        dictionary = None

    if any(one.valid is not None for one in columns):
        valid = np.concatenate(
            [
                one.valid if one.valid is not None else np.ones(len(one), dtype=bool)
                for one in columns
            ]
        )
    else:
        valid = None
    nullable = valid is not None and not bool(valid.all())
    field = Field(name=first.name, logical=first.logical, nullable=nullable)
    return Column(field=field, values=values, valid=valid, dictionary=dictionary)


def all_valid(length: int) -> np.ndarray:
    """A mask saying everything is present, for the paths that need one explicitly."""
    if length < 0:
        raise ConfigError(f"{length} is not a length")
    return np.ones(length, dtype=bool)


def combine_validity(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
    """The validity of a value computed from two operands.

    Null propagates: anything combined with a null is null. Returning None when both sides are
    fully valid rather than an array of ones is what keeps the common path free of masks.
    """
    if left is None:
        return right
    if right is None:
        return left
    if left.shape != right.shape:
        raise DataError(f"{left.shape[0]} against {right.shape[0]} validity entries")
    return left & right


def date_from_days(name: str, days: Sequence[int]) -> Column:
    """A date column from days since an epoch, which is how dates are stored throughout.

    An integer count of days rather than a datetime, because every operation the engine performs
    on a date is a comparison or a range, and both work on the integer. Formatting is the
    printer's problem.
    """
    values = np.array(list(days), dtype=PHYSICAL[DATE])
    return Column(field=Field(name=name, logical=DATE, nullable=False), values=values)


def integer_column(name: str, values: Sequence[int]) -> Column:
    """An integer column with no nulls, which is what most generators produce."""
    raw = np.asarray(values, dtype=PHYSICAL[INTEGER])
    return Column(field=Field(name=name, logical=INTEGER, nullable=False), values=raw)


def floating_column(name: str, values: Sequence[float]) -> Column:
    """A floating column with no nulls."""
    raw = np.asarray(values, dtype=PHYSICAL[FLOATING])
    return Column(field=Field(name=name, logical=FLOATING, nullable=False), values=raw)


def boolean_column(name: str, values: Sequence[bool]) -> Column:
    """A boolean column with no nulls."""
    raw = np.asarray(values, dtype=PHYSICAL[BOOLEAN])
    return Column(field=Field(name=name, logical=BOOLEAN, nullable=False), values=raw)


def string_column(name: str, values: Sequence[str]) -> Column:
    """A string column with no nulls, dictionary encoded on a sorted dictionary."""
    return column_from(name, list(values), logical=STRING)
