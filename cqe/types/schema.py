from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from cqe.errors import ConfigError, SchemaError, TypeMismatch, UnknownColumn

# The type system, which is small on purpose and has one decision in it worth explaining.
#
# Five logical types: boolean, integer, floating, string and date. Each maps to a physical numpy
# dtype, and the mapping is not one to one, which is the decision. A string column is stored as
# an integer array of dictionary codes plus a separate table of the distinct values, so its
# logical type is string and its physical type is int32. Every operator in the engine works on
# the physical array and consults the logical type only when the answer depends on it, which is
# comparison ordering and printing and almost nothing else.
#
# That separation is what makes dictionary encoding cheap rather than a special case threaded
# through every operator. A filter on a string column compares int32 codes; a group by on a
# string column hashes int32 codes; a sort on a string column sorts int32 codes, and gets the
# right answer only if the dictionary is ordered, which columns/encode/dictionary.py measures
# and which is not free.
#
# Nullability is carried separately, as a validity bitmap on the column rather than as a
# sentinel value in the data. Sentinels are cheaper and they are wrong: there is no integer that
# cannot appear in an integer column, and every engine that has tried has eventually shipped a
# query where a real value was silently dropped. The bitmap costs one bit per value and removes
# the whole class.

BOOLEAN = "boolean"
INTEGER = "integer"
FLOATING = "floating"
STRING = "string"
DATE = "date"

LOGICAL_TYPES = (BOOLEAN, INTEGER, FLOATING, STRING, DATE)

PHYSICAL = {
    BOOLEAN: np.dtype(np.bool_),
    INTEGER: np.dtype(np.int64),
    FLOATING: np.dtype(np.float64),
    STRING: np.dtype(np.int32),
    DATE: np.dtype(np.int32),
}

# Which types a comparison can be applied to, and which can be summed. Strings compare by
# dictionary code, which only means what a reader expects when the dictionary is ordered, so the
# ordering is a property of the column rather than of the type and is checked at the point of
# use.
ORDERED = (INTEGER, FLOATING, DATE, STRING)
NUMERIC = (INTEGER, FLOATING)


@dataclass(frozen=True)
class Field:
    """One column's name, logical type and whether it admits nulls."""

    name: str
    logical: str
    nullable: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaError("a field needs a name")
        if self.logical not in LOGICAL_TYPES:
            raise TypeMismatch(
                f"{self.logical} is not a type; try one of {list(LOGICAL_TYPES)}"
            )

    @property
    def physical(self) -> np.dtype:
        """The numpy dtype the values are actually stored in."""
        return PHYSICAL[self.logical]

    @property
    def is_ordered(self) -> bool:
        """Whether less than means anything for this type."""
        return self.logical in ORDERED

    @property
    def is_numeric(self) -> bool:
        """Whether arithmetic means anything for this type."""
        return self.logical in NUMERIC

    @property
    def width(self) -> int:
        """Bytes per value in the physical representation, ignoring the validity bitmap."""
        return int(self.physical.itemsize)

    def renamed(self, name: str) -> Field:
        """The same field under a different name."""
        return Field(name=name, logical=self.logical, nullable=self.nullable)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"name": self.name, "type": self.logical, "nullable": self.nullable}

    def __str__(self) -> str:
        return f"{self.name} {self.logical}" + ("" if self.nullable else " not null")


@dataclass(frozen=True)
class Schema:
    """An ordered set of fields with unique names."""

    fields: tuple[Field, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        names = [one.name for one in self.fields]
        if len(set(names)) != len(names):
            repeated = sorted({name for name in names if names.count(name) > 1})
            raise SchemaError(f"repeated column names: {repeated}")

    @classmethod
    def of(cls, *pairs: tuple[str, str]) -> Schema:
        """Build from name and type pairs, which is what every test wants."""
        return cls(tuple(Field(name=name, logical=logical) for name, logical in pairs))

    @property
    def names(self) -> tuple[str, ...]:
        """Column names in order."""
        return tuple(one.name for one in self.fields)

    @property
    def width(self) -> int:
        """How many columns there are."""
        return len(self.fields)

    @property
    def bytes_per_row(self) -> int:
        """Physical bytes for one row, ignoring nulls and encoding.

        The number a cost model starts from and never ends at, since every encoding in
        columns/encode moves it and the point of them is that it moves a long way.
        """
        return sum(one.width for one in self.fields)

    def __iter__(self) -> Iterator[Field]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def index(self, name: str) -> int:
        """Where a column sits, by name."""
        try:
            return self.names.index(name)
        except ValueError as missing:
            raise UnknownColumn(
                f"{name} is not a column; the schema has {list(self.names)}"
            ) from missing

    def field(self, name: str) -> Field:
        """One column's field, by name."""
        return self.fields[self.index(name)]

    def logical(self, name: str) -> str:
        """One column's logical type, by name."""
        return self.field(name).logical

    def select(self, names: Sequence[str]) -> Schema:
        """A schema holding only the named columns, in the order given.

        Order given rather than order held, because projection is allowed to reorder and a
        caller who wrote the names in an order expects them back that way.
        """
        return Schema(tuple(self.field(name) for name in names))

    def drop(self, names: Sequence[str]) -> Schema:
        """A schema without the named columns."""
        unwanted = set(names)
        missing = unwanted - set(self.names)
        if missing:
            raise UnknownColumn(f"cannot drop {sorted(missing)}; not in {list(self.names)}")
        return Schema(tuple(one for one in self.fields if one.name not in unwanted))

    def add(self, one: Field) -> Schema:
        """A schema with one more column on the end."""
        return Schema((*self.fields, one))

    def rename(self, mapping: dict[str, str]) -> Schema:
        """A schema with some columns renamed."""
        missing = set(mapping) - set(self.names)
        if missing:
            raise UnknownColumn(f"cannot rename {sorted(missing)}; not in {list(self.names)}")
        return Schema(
            tuple(one.renamed(mapping.get(one.name, one.name)) for one in self.fields)
        )

    def joined(self, other: Schema, suffix: str = "_right") -> Schema:
        """The schema of the two side by side, disambiguating repeated names.

        A join of a table with itself is the common case and it has every name twice, so a
        suffix rather than a refusal. The suffix is applied to the right side only, so the left
        side of a join keeps the names the query was written against.
        """
        taken = set(self.names)
        out = list(self.fields)
        for one in other.fields:
            name = one.name
            while name in taken:
                name = f"{name}{suffix}"
            taken.add(name)
            out.append(one.renamed(name))
        return Schema(tuple(out))

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"columns": [one.as_dict() for one in self.fields]}

    def __str__(self) -> str:
        return ", ".join(str(one) for one in self.fields)


def common_type(left: str, right: str) -> str:
    """The type two operands have to be brought to before they can be compared or combined.

    Only one promotion exists, integer to floating, and that is deliberate. Every other pair is
    a refusal. Engines that promote widely end up comparing a string to an integer by coercing
    one of them, which turns a query the user got wrong into a query that runs and returns
    something.
    """
    if left == right:
        return left
    if {left, right} == {INTEGER, FLOATING}:
        return FLOATING
    raise TypeMismatch(f"{left} and {right} cannot be combined")


def check_comparable(left: str, right: str) -> str:
    """The type a comparison between two operands is carried out in."""
    shared = common_type(left, right)
    if shared not in ORDERED and shared != BOOLEAN:
        raise TypeMismatch(f"{shared} cannot be ordered")
    return shared


def check_numeric(logical: str, what: str = "arithmetic") -> str:
    """Refuse a non numeric type where a number is required."""
    if logical not in NUMERIC:
        raise TypeMismatch(f"{logical} does not support {what}")
    return logical


def empty_array(logical: str, length: int) -> np.ndarray:
    """A zero filled physical array of the right dtype and length."""
    if length < 0:
        raise ConfigError(f"{length} is not a length")
    return np.zeros(length, dtype=PHYSICAL[logical])


def infer_logical(values: Sequence) -> str:
    """Guess a logical type from Python values, for tests and for the loader.

    Booleans are checked before integers because a bool is an int in Python and every inference
    routine that checks in the other order types a boolean column as integer. Worth the two
    lines.
    """
    if not len(values):
        raise SchemaError("cannot infer a type from nothing")
    present = [value for value in values if value is not None]
    if not present:
        raise SchemaError("cannot infer a type from nulls alone")
    if all(isinstance(value, bool) for value in present):
        return BOOLEAN
    if all(isinstance(value, str) for value in present):
        return STRING
    if all(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        for value in present
    ):
        return INTEGER
    if all(
        isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
        for value in present
    ):
        return FLOATING
    kinds = sorted({type(value).__name__ for value in present})
    raise SchemaError(f"mixed value types {kinds}")


def schema_from_rows(names: Sequence[str], rows: Sequence[Sequence]) -> Schema:
    """Infer a schema from a list of rows, for tests and for the loader."""
    if not names:
        raise SchemaError("a schema needs at least one column")
    widths = {len(row) for row in rows}
    if widths and widths != {len(names)}:
        raise SchemaError(f"rows of widths {sorted(widths)} against {len(names)} names")
    fields = []
    for position, name in enumerate(names):
        column = [row[position] for row in rows]
        fields.append(
            Field(
                name=name,
                logical=infer_logical(column),
                nullable=any(value is None for value in column),
            )
        )
    return Schema(tuple(fields))
