from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, column_from, concat
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, DataError, SchemaError, UnknownColumn
from cqe.types.schema import STRING, Schema, empty_array

# A batch of rows held as columns, and the argument for why batches exist at all.
#
# Two obvious designs are wrong and this is the third. Processing a row at a time is correct and
# spends all of its time on interpretation overhead per value. Processing a whole table at a
# time is fast and needs the whole table in memory, which is exactly the case a query engine is
# for. A batch is the compromise: enough rows that the per call overhead amortises, few enough
# that the working set stays small.
#
# The batch size is a parameter and the module measures what it costs rather than asserting a
# default. The measurement is in eval/batching.py and the short version is that the curve is
# flat over two orders of magnitude and the only thing that matters is not being at either end.
#
# A Batch is immutable. Every operator returns a new one, sharing the underlying numpy arrays
# wherever it can, which is why take and mask are the only two primitives: both produce new
# arrays, and everything else in the engine is a projection or a rename that shares them.
#
# Batches and Tables are the same thing at different scales and are deliberately the same class.
# A table is a batch that happens to be large; an operator that works on one works on the other,
# and the split into batches is a scheduling decision made by the reader rather than a type.


@dataclass
class Batch:
    """A set of equal length columns with a schema."""

    schema: Schema
    columns: tuple[Column, ...]

    def __post_init__(self) -> None:
        if len(self.schema) != len(self.columns):
            raise SchemaError(f"{len(self.schema)} fields against {len(self.columns)} columns")
        for field, column in zip(self.schema.fields, self.columns, strict=True):
            if field.name != column.name:
                raise SchemaError(f"{field.name} against {column.name}")
            if field.logical != column.logical:
                raise SchemaError(f"{field.name} is {field.logical} against {column.logical}")
        lengths = {len(column) for column in self.columns}
        if len(lengths) > 1:
            raise DataError(f"columns of lengths {sorted(lengths)}")

    @classmethod
    def of(cls, **named: Sequence) -> Batch:
        """Build from keyword lists, which is what every test wants."""
        if not named:
            raise ConfigError("a batch needs at least one column")
        columns = tuple(column_from(name, list(values)) for name, values in named.items())
        return cls.from_columns(columns)

    @classmethod
    def from_columns(cls, columns: Sequence[Column]) -> Batch:
        """Build a batch and derive the schema from the columns."""
        if not columns:
            raise ConfigError("a batch needs at least one column")
        return cls(
            schema=Schema(tuple(column.field for column in columns)),
            columns=tuple(columns),
        )

    @classmethod
    def empty(cls, schema: Schema) -> Batch:
        """A batch with the right shape and no rows, which is what an empty result is.

        Distinct from no batch at all. A query that matched nothing still has a schema and a
        consumer still wants to know what the columns would have been.
        """
        columns = tuple(
            Column(
                field=one,
                values=empty_array(one.logical, 0),
                dictionary=() if one.logical == STRING else None,
            )
            for one in schema.fields
        )
        return cls(schema=schema, columns=columns)

    @property
    def rows(self) -> int:
        """How many rows there are."""
        return len(self.columns[0]) if self.columns else 0

    @property
    def width(self) -> int:
        """How many columns there are."""
        return len(self.columns)

    @property
    def nbytes(self) -> int:
        """Memory occupied by every column."""
        return sum(column.nbytes for column in self.columns)

    @property
    def names(self) -> tuple[str, ...]:
        """Column names in order."""
        return self.schema.names

    def __len__(self) -> int:
        return self.rows

    def __iter__(self) -> Iterator[Column]:
        return iter(self.columns)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def column(self, name: str) -> Column:
        """One column, by name."""
        return self.columns[self.schema.index(name)]

    def values(self, name: str) -> np.ndarray:
        """One column's physical array, by name, which is what operators want."""
        return self.column(name).values

    def select(self, names: Sequence[str], meter: Meter | None = None) -> Batch:
        """The named columns, in the order given.

        Free in values touched, which is the point of the whole storage layout: dropping a
        column from a projection costs nothing because the column was never read. The meter
        records nothing here for that reason, and storage/layout.py is where the saving actually
        appears.
        """
        missing = [name for name in names if name not in self.names]
        if missing:
            raise UnknownColumn(f"{missing} not in {list(self.names)}")
        columns = tuple(self.column(name) for name in names)
        if meter is not None:
            meter.batch()
        return Batch.from_columns(columns)

    def drop(self, names: Sequence[str]) -> Batch:
        """Everything except the named columns."""
        unwanted = set(names)
        return self.select([name for name in self.names if name not in unwanted])

    def take(self, positions: np.ndarray, meter: Meter | None = None) -> Batch:
        """The rows at the given positions, in the order given.

        Charged per value moved, across every column, because that is what it costs. A take on a
        ten column batch is ten times the take on a one column batch and a plan that takes early
        pays for columns it may later drop, which is the argument late materialisation is about.
        """
        columns = tuple(column.take(positions) for column in self.columns)
        if meter is not None:
            moved = len(positions) * self.width
            meter.touch(moved, "take")
            meter.materialise(len(positions))
        return Batch(schema=self.schema, columns=columns)

    def mask(self, keep: np.ndarray, meter: Meter | None = None) -> Batch:
        """The rows where a boolean mask is true."""
        if keep.dtype != np.bool_:
            raise DataError(f"a mask is boolean, not {keep.dtype}")
        if keep.shape[0] != self.rows:
            raise DataError(f"{keep.shape[0]} mask entries against {self.rows} rows")
        return self.take(np.flatnonzero(keep), meter=meter)

    def slice(self, start: int, stop: int | None = None) -> Batch:
        """A contiguous run of rows, which is how a table is cut into batches."""
        columns = tuple(column.slice(start, stop) for column in self.columns)
        return Batch(schema=self.schema, columns=columns)

    def batches(self, size: int) -> Iterator[Batch]:
        """Cut into batches of at most the given size.

        The last batch is short rather than padded. Padding would make every operator check a
        length it otherwise gets for free, to save a branch that numpy does not have.
        """
        if size < 1:
            raise ConfigError(f"{size} is not a batch size")
        for start in range(0, max(self.rows, 1), size):
            if start >= self.rows and self.rows:
                break
            yield self.slice(start, min(start + size, self.rows))
            if not self.rows:
                break

    def rename(self, mapping: dict[str, str]) -> Batch:
        """The same data under different column names."""
        columns = tuple(
            column.renamed(mapping.get(column.name, column.name)) for column in self.columns
        )
        return Batch.from_columns(columns)

    def with_column(self, column: Column) -> Batch:
        """The batch with one more column, or with one replaced.

        Replacement rather than a refusal when the name already exists, because that is what a
        projection computing a new value for an existing name means and refusing would make
        every caller drop first.
        """
        if len(self.columns) and len(column) != self.rows:
            raise DataError(f"{len(column)} values against {self.rows} rows")
        if column.name in self.names:
            columns = tuple(column if one.name == column.name else one for one in self.columns)
        else:
            columns = (*self.columns, column)
        return Batch.from_columns(columns)

    def to_rows(self) -> list[list]:
        """Python rows, for the reference interpreter and for printing.

        The only place the engine leaves its own representation, and slow by construction. Every
        differential check in verify/ runs through here on both sides, so it being obvious
        matters more than it being quick.
        """
        columns = [column.to_list() for column in self.columns]
        return [list(row) for row in zip(*columns, strict=True)] if columns else []

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rows": self.rows,
            "columns": self.width,
            "bytes": self.nbytes,
            "schema": [one.name for one in self.schema.fields],
        }

    def __str__(self) -> str:
        return f"{self.rows} rows of {', '.join(self.names)}"


def from_rows(names: Sequence[str], rows: Sequence[Sequence]) -> Batch:
    """Build a batch from Python rows, which is what tests and the loader produce."""
    if not names:
        raise ConfigError("a batch needs at least one column")
    widths = {len(row) for row in rows}
    if widths and widths != {len(names)}:
        raise DataError(f"rows of widths {sorted(widths)} against {len(names)} names")
    columns = []
    for position, name in enumerate(names):
        columns.append(column_from(name, [row[position] for row in rows]))
    return Batch.from_columns(tuple(columns))


def stack(batches: Sequence[Batch]) -> Batch:
    """Concatenate batches with the same schema into one.

    String columns are merged onto a shared dictionary by columns/array.py, which is the
    expensive part and the reason an operator that can avoid concatenating should.
    """
    if not batches:
        raise ConfigError("there is nothing to stack")
    first = batches[0]
    for other in batches[1:]:
        if other.names != first.names:
            raise SchemaError(f"{list(first.names)} against {list(other.names)}")
    if len(batches) == 1:
        return first
    columns = tuple(
        concat([batch.columns[position] for batch in batches])
        for position in range(first.width)
    )
    return Batch.from_columns(columns)


def side_by_side(left: Batch, right: Batch, suffix: str = "_right") -> Batch:
    """Two batches of the same height placed next to each other, which is what a join emits."""
    if left.rows != right.rows:
        raise DataError(f"{left.rows} rows against {right.rows}")
    schema = left.schema.joined(right.schema, suffix=suffix)
    names = schema.names[left.width :]
    columns = list(left.columns) + [
        column.renamed(name) for column, name in zip(right.columns, names, strict=True)
    ]
    return Batch.from_columns(tuple(columns))


def selection_to_mask(positions: np.ndarray, length: int) -> np.ndarray:
    """Turn a selection vector into a boolean mask.

    Both representations of the same fact and the engine uses both, because they are cheap in
    opposite regimes. exec/filter.py measures the crossover and it is not where I guessed.
    """
    if length < 0:
        raise ConfigError(f"{length} is not a length")
    out = np.zeros(length, dtype=bool)
    if len(positions):
        if int(positions.max()) >= length:
            raise DataError(f"position {int(positions.max())} against a length of {length}")
        out[positions] = True
    return out


def mask_to_selection(keep: np.ndarray) -> np.ndarray:
    """Turn a boolean mask into a selection vector."""
    if keep.dtype != np.bool_:
        raise DataError(f"a mask is boolean, not {keep.dtype}")
    return np.flatnonzero(keep)
