from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cqe.columns.array import Column, combine_validity
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, TypeMismatch, UnknownColumn
from cqe.exec.batch import Batch
from cqe.types.schema import (
    BOOLEAN,
    FLOATING,
    INTEGER,
    STRING,
    Field,
    Schema,
    check_comparable,
    check_numeric,
    common_type,
)

# Expressions, evaluated a column at a time.
#
# An expression is a tree: literals and column references at the leaves, comparisons and
# arithmetic and boolean connectives above them. Evaluating it produces a Column, and every
# operator in the engine is a thin shell around that.
#
# Two decisions carry the module.
#
# The first is that evaluation is vectorised all the way down, with no per row dispatch
# anywhere. Each node evaluates its children into whole columns and combines them with one numpy
# call. That is the entire reason a vectorised engine is faster than an interpreter: not that
# numpy is quick, but that the type dispatch happens once per column instead of once per value.
# The measurement in eval/batching.py puts the per call overhead at about the cost of forty
# values, which is why the batch size matters at the bottom and stops mattering above a few
# hundred.
#
# The second is nulls. A validity mask travels with every intermediate, and the rules are the
# ones verify/reference.py writes out in Python: a comparison against null is null, and is not
# false. That distinction survives here because the mask is a separate array from the values, so
# a comparison can produce false in the value array and invalid in the mask, and the difference
# is still readable at the top.
#
# The one place three valued logic costs something is and and or, which cannot simply combine
# their operands' masks. False and null is false, not null, so the mask of an and depends on the
# values of its children and not only on their masks. That is four numpy operations instead of
# two, and it is why is_null and is_not_null exist as their own nodes: they collapse to two
# valued logic immediately and let a plan avoid the expensive form when the writer knows there
# are no nulls.


@dataclass(frozen=True)
class Expr:
    """The base of the expression tree."""

    def type_of(self, schema: Schema) -> str:
        """The logical type this expression produces against a schema."""
        raise NotImplementedError

    def columns_used(self) -> frozenset[str]:
        """Every column name this expression reads, for pushdown and pruning."""
        raise NotImplementedError

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Produce a column of results over a batch."""
        raise NotImplementedError

    def depth(self) -> int:
        """How deep the tree is, which is what a plan printer wants."""
        return 1

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": type(self).__name__, "depth": self.depth()}


@dataclass(frozen=True)
class Literal(Expr):
    """A constant."""

    value: object
    logical: str

    def type_of(self, schema: Schema) -> str:  # noqa: ARG002
        """A literal's type does not depend on the schema."""
        return self.logical

    def columns_used(self) -> frozenset[str]:
        """A literal reads nothing."""
        return frozenset()

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Broadcast the constant to the batch height.

        Charged nothing, because a literal is one value however many rows read it. That is not a
        rounding error: a predicate comparing a column against a constant costs one column read
        and not two, and a cost model that charges both is out by a factor of two on the most
        common predicate there is.
        """
        del meter
        if self.logical == STRING:
            text = str(self.value)
            codes = np.zeros(batch.rows, dtype=np.int32)
            return Column(
                field=Field(name="literal", logical=STRING, nullable=False),
                values=codes,
                dictionary=(text,),
            )
        dtype = {BOOLEAN: np.bool_, INTEGER: np.int64, FLOATING: np.float64}.get(
            self.logical, np.int32
        )
        return Column(
            field=Field(name="literal", logical=self.logical, nullable=False),
            values=np.full(batch.rows, self.value, dtype=dtype),
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": "Literal", "value": self.value, "type": self.logical}


@dataclass(frozen=True)
class ColumnRef(Expr):
    """A reference to a column by name."""

    name: str

    def type_of(self, schema: Schema) -> str:
        """The referenced column's type."""
        return schema.logical(self.name)

    def columns_used(self) -> frozenset[str]:
        """Itself."""
        return frozenset({self.name})

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Read the column, charging one value per row."""
        if self.name not in batch.names:
            raise UnknownColumn(f"{self.name} is not in {list(batch.names)}")
        column = batch.column(self.name)
        if meter is not None:
            meter.touch(len(column), "read", width=column.field.width)
        return column

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": "ColumnRef", "name": self.name}


COMPARISONS = {
    "=": np.equal,
    "!=": np.not_equal,
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}

ARITHMETIC = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
}


@dataclass(frozen=True)
class Compare(Expr):
    """A comparison between two expressions."""

    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in COMPARISONS:
            raise ConfigError(
                f"{self.op} is not a comparison; try one of {sorted(COMPARISONS)}"
            )

    def type_of(self, schema: Schema) -> str:
        """A comparison is boolean, once its operands agree on a type."""
        check_comparable(self.left.type_of(schema), self.right.type_of(schema))
        return BOOLEAN

    def columns_used(self) -> frozenset[str]:
        """Both sides."""
        return self.left.columns_used() | self.right.columns_used()

    def depth(self) -> int:
        """One more than the deeper child."""
        return 1 + max(self.left.depth(), self.right.depth())

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Compare two columns, propagating nulls into the mask.

        String comparison goes through the dictionary codes, which is only correct when the two
        sides share a dictionary or one of them is a literal. The literal case is resolved by
        looking the text up in the column's own dictionary, which turns a string comparison into
        an integer one and is the whole payoff of dictionary encoding.
        """
        left = self.left.evaluate(batch, meter)
        right = self.right.evaluate(batch, meter)
        values, valid = _aligned(left, right, self.op)
        if meter is not None:
            meter.touch(len(values), "compare", width=1)
        return Column(
            field=Field(name="compare", logical=BOOLEAN, nullable=valid is not None),
            values=values,
            valid=valid,
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": "Compare", "op": self.op, "depth": self.depth()}


def _aligned(left: Column, right: Column, op: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Bring two columns onto comparable physical values and apply the comparison."""
    valid = combine_validity(left.valid, right.valid)
    if STRING in (left.logical, right.logical):
        if left.logical != right.logical:
            raise TypeMismatch(f"{left.logical} cannot be compared to {right.logical}")
        return _compare_strings(left, right, op), valid
    check_comparable(left.logical, right.logical)
    return COMPARISONS[op](left.values, right.values), valid


def _compare_strings(left: Column, right: Column, op: str) -> np.ndarray:
    """Compare two dictionary encoded columns.

    The interesting case is one side being a literal, which arrives as a one entry dictionary.
    Its text is located in the other side's dictionary by binary search, which is valid because
    columns/array.py builds every dictionary sorted, and the comparison then runs on codes.

    A value not present in the dictionary is not an error. For equality it means nothing
    matches; for a range it means the search position is still the right boundary, which is why
    searchsorted is used rather than a lookup that would have to fail.
    """
    left_entries = left.dictionary or ()
    right_entries = right.dictionary or ()
    if len(right_entries) == 1 and len(left_entries) != 1:
        position = int(np.searchsorted(np.array(left_entries), right_entries[0]))
        exact = position < len(left_entries) and left_entries[position] == right_entries[0]
        if op == "=":
            return np.equal(left.values, position) if exact else np.zeros(len(left), dtype=bool)
        if op == "!=":
            return (
                np.not_equal(left.values, position) if exact else np.ones(len(left), dtype=bool)
            )
        return COMPARISONS[op](left.values, position)
    if len(left_entries) == 1 and len(right_entries) != 1:
        flipped = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}[op]
        return _compare_strings(right, left, flipped)
    if left_entries != right_entries:
        merged = tuple(sorted(set(left_entries) | set(right_entries)))
        left_map = np.array([merged.index(entry) for entry in left_entries], dtype=np.int32)
        right_map = np.array([merged.index(entry) for entry in right_entries], dtype=np.int32)
        return COMPARISONS[op](left_map[left.values], right_map[right.values])
    return COMPARISONS[op](left.values, right.values)


@dataclass(frozen=True)
class Arithmetic(Expr):
    """Addition, subtraction or multiplication of two numeric expressions."""

    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in ARITHMETIC:
            raise ConfigError(f"{self.op} is not arithmetic; try one of {sorted(ARITHMETIC)}")

    def type_of(self, schema: Schema) -> str:
        """The promoted type of the two operands."""
        left = check_numeric(self.left.type_of(schema))
        right = check_numeric(self.right.type_of(schema))
        return common_type(left, right)

    def columns_used(self) -> frozenset[str]:
        """Both sides."""
        return self.left.columns_used() | self.right.columns_used()

    def depth(self) -> int:
        """One more than the deeper child."""
        return 1 + max(self.left.depth(), self.right.depth())

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Combine two numeric columns, propagating nulls."""
        left = self.left.evaluate(batch, meter)
        right = self.right.evaluate(batch, meter)
        logical = common_type(check_numeric(left.logical), check_numeric(right.logical))
        values = ARITHMETIC[self.op](left.values, right.values)
        if logical == FLOATING:
            values = values.astype(np.float64)
        else:
            values = values.astype(np.int64)
        valid = combine_validity(left.valid, right.valid)
        if meter is not None:
            meter.touch(len(values), "arithmetic")
        return Column(
            field=Field(name="arithmetic", logical=logical, nullable=valid is not None),
            values=values,
            valid=valid,
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": "Arithmetic", "op": self.op, "depth": self.depth()}


@dataclass(frozen=True)
class And(Expr):
    """Three valued conjunction over any number of operands."""

    parts: tuple[Expr, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.parts:
            raise ConfigError("an and needs at least one operand")

    def type_of(self, schema: Schema) -> str:
        """Boolean, once every operand is boolean."""
        for part in self.parts:
            if part.type_of(schema) != BOOLEAN:
                raise TypeMismatch(f"and takes booleans, not {part.type_of(schema)}")
        return BOOLEAN

    def columns_used(self) -> frozenset[str]:
        """Every operand's columns."""
        return frozenset().union(*(part.columns_used() for part in self.parts))

    def depth(self) -> int:
        """One more than the deepest operand."""
        return 1 + max(part.depth() for part in self.parts)

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Combine operands under the rule that false beats null.

        The mask is not the intersection of the operands' masks, which is what every other node
        here does. A row where one operand is false and another is null is false, so it is valid
        even though one input was not. Getting this wrong turns a filter that should drop rows
        into one that keeps them, and it is invisible on data with no nulls.
        """
        columns = [part.evaluate(batch, meter) for part in self.parts]
        values = np.ones(batch.rows, dtype=bool)
        known_false = np.zeros(batch.rows, dtype=bool)
        unknown = np.zeros(batch.rows, dtype=bool)
        for column in columns:
            present = np.ones(batch.rows, dtype=bool) if column.valid is None else column.valid
            known_false |= present & ~column.values
            unknown |= ~present
            values &= column.values | ~present
        valid = known_false | ~unknown
        values = values & ~known_false
        if meter is not None:
            meter.touch(batch.rows * len(columns), "and", width=1)
        return Column(
            field=Field(name="and", logical=BOOLEAN, nullable=True),
            values=values,
            valid=valid if not valid.all() else None,
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": "And", "operands": len(self.parts), "depth": self.depth()}


@dataclass(frozen=True)
class Or(Expr):
    """Three valued disjunction over any number of operands."""

    parts: tuple[Expr, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.parts:
            raise ConfigError("an or needs at least one operand")

    def type_of(self, schema: Schema) -> str:
        """Boolean, once every operand is boolean."""
        for part in self.parts:
            if part.type_of(schema) != BOOLEAN:
                raise TypeMismatch(f"or takes booleans, not {part.type_of(schema)}")
        return BOOLEAN

    def columns_used(self) -> frozenset[str]:
        """Every operand's columns."""
        return frozenset().union(*(part.columns_used() for part in self.parts))

    def depth(self) -> int:
        """One more than the deepest operand."""
        return 1 + max(part.depth() for part in self.parts)

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Combine operands under the rule that true beats null."""
        columns = [part.evaluate(batch, meter) for part in self.parts]
        known_true = np.zeros(batch.rows, dtype=bool)
        unknown = np.zeros(batch.rows, dtype=bool)
        for column in columns:
            present = np.ones(batch.rows, dtype=bool) if column.valid is None else column.valid
            known_true |= present & column.values
            unknown |= ~present
        valid = known_true | ~unknown
        if meter is not None:
            meter.touch(batch.rows * len(columns), "or", width=1)
        return Column(
            field=Field(name="or", logical=BOOLEAN, nullable=True),
            values=known_true,
            valid=valid if not valid.all() else None,
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": "Or", "operands": len(self.parts), "depth": self.depth()}


@dataclass(frozen=True)
class Not(Expr):
    """Three valued negation, where null stays null."""

    part: Expr

    def type_of(self, schema: Schema) -> str:
        """Boolean."""
        if self.part.type_of(schema) != BOOLEAN:
            raise TypeMismatch(f"not takes a boolean, not {self.part.type_of(schema)}")
        return BOOLEAN

    def columns_used(self) -> frozenset[str]:
        """Its operand's columns."""
        return self.part.columns_used()

    def depth(self) -> int:
        """One more than its operand."""
        return 1 + self.part.depth()

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Flip the values and carry the mask through unchanged."""
        column = self.part.evaluate(batch, meter)
        if meter is not None:
            meter.touch(len(column), "not", width=1)
        return Column(
            field=Field(name="not", logical=BOOLEAN, nullable=column.valid is not None),
            values=~column.values,
            valid=column.valid,
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": "Not", "depth": self.depth()}


@dataclass(frozen=True)
class IsNull(Expr):
    """Whether a value is missing, which is itself never missing."""

    part: Expr
    negated: bool = False

    def type_of(self, schema: Schema) -> str:
        """Boolean, whatever the operand's type is."""
        self.part.type_of(schema)
        return BOOLEAN

    def columns_used(self) -> frozenset[str]:
        """Its operand's columns."""
        return self.part.columns_used()

    def depth(self) -> int:
        """One more than its operand."""
        return 1 + self.part.depth()

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Read the mask rather than the values, and produce a column with no mask.

        The only node that reduces three valued logic to two, which is what it is for. A plan
        that ends in is_null has no null handling left to do above it, and And and Or can take
        their cheap paths.
        """
        column = self.part.evaluate(batch, meter)
        if column.valid is None:
            missing = np.zeros(batch.rows, dtype=bool)
        else:
            missing = ~column.valid
        if meter is not None:
            meter.touch(len(column), "is_null", width=1)
        return Column(
            field=Field(name="is_null", logical=BOOLEAN, nullable=False),
            values=~missing if self.negated else missing,
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": "IsNull", "negated": self.negated, "depth": self.depth()}


@dataclass(frozen=True)
class InList(Expr):
    """Membership in a fixed set of literals."""

    part: Expr
    options: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.options:
            raise ConfigError("an in list needs at least one option")

    def type_of(self, schema: Schema) -> str:
        """Boolean."""
        self.part.type_of(schema)
        return BOOLEAN

    def columns_used(self) -> frozenset[str]:
        """Its operand's columns."""
        return self.part.columns_used()

    def depth(self) -> int:
        """One more than its operand."""
        return 1 + self.part.depth()

    def evaluate(self, batch: Batch, meter: Meter | None = None) -> Column:
        """Test membership, which on a dictionary column is membership in a set of codes.

        The set of codes is computed once from the dictionary, so the per row cost is one
        isin lookup whatever the option list holds. That is the same trick the equality path
        uses and it extends to any number of options at no extra per row cost.
        """
        column = self.part.evaluate(batch, meter)
        if column.dictionary is not None:
            entries = column.dictionary
            codes = [entries.index(str(one)) for one in self.options if str(one) in entries]
            values = (
                np.isin(column.values, np.array(codes, dtype=np.int32))
                if codes
                else np.zeros(len(column), dtype=bool)
            )
        else:
            values = np.isin(column.values, np.array(list(self.options)))
        if meter is not None:
            meter.touch(len(column), "in_list", width=1)
        return Column(
            field=Field(name="in_list", logical=BOOLEAN, nullable=column.valid is not None),
            values=values,
            valid=column.valid,
        )

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": "InList", "options": len(self.options), "depth": self.depth()}


def column(name: str) -> ColumnRef:
    """A column reference, for building expressions readably."""
    return ColumnRef(name)


def literal(value) -> Literal:
    """A literal whose type is inferred from the Python value."""
    if isinstance(value, bool):
        return Literal(value, BOOLEAN)
    if isinstance(value, str):
        return Literal(value, STRING)
    if isinstance(value, (int, np.integer)):
        return Literal(int(value), INTEGER)
    if isinstance(value, (float, np.floating)):
        return Literal(float(value), FLOATING)
    raise TypeMismatch(f"{type(value).__name__} is not a literal type")


def evaluate_to_mask(expression: Expr, batch: Batch, meter: Meter | None = None) -> np.ndarray:
    """Evaluate a predicate and collapse three valued logic to a keep or drop decision.

    The one place the collapse is allowed. A row is kept when the predicate is true and dropped
    when it is false or null, which is the rule verify/reference.py writes out and the reason a
    negated predicate does not pick the nulls back up.
    """
    result = expression.evaluate(batch, meter)
    if result.logical != BOOLEAN:
        raise TypeMismatch(f"a predicate is boolean, not {result.logical}")
    if result.valid is None:
        return result.values
    return result.values & result.valid


def to_callable(expression: Expr):
    """A Python callable over a row mapping, for checking against verify/reference.py.

    Deliberately slow and deliberately separate from the vectorised path, so that the two
    implementations share nothing and a bug has to appear in both to go unnoticed.

    The import is local because verify/reference.py imports exec/batch.py, and a module level
    import here would close the cycle. That is the only such import in the package and it is
    here rather than in the reference, so the reference stays free of any knowledge of the fast
    path.
    """
    from cqe.verify import reference  # noqa: PLC0415

    def evaluate(row: dict):
        return _row_value(expression, row, reference)

    return evaluate


def _row_value(expression: Expr, row: dict, reference):
    """One expression against one row of Python values."""
    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, ColumnRef):
        if expression.name not in row:
            raise UnknownColumn(f"{expression.name} is not in {sorted(row)}")
        return row[expression.name]
    if isinstance(expression, Compare):
        left = _row_value(expression.left, row, reference)
        right = _row_value(expression.right, row, reference)
        order = reference.compare(left, right)
        if order is None:
            return None
        return {
            "=": order == 0,
            "!=": order != 0,
            "<": order < 0,
            "<=": order <= 0,
            ">": order > 0,
            ">=": order >= 0,
        }[expression.op]
    if isinstance(expression, Arithmetic):
        left = _row_value(expression.left, row, reference)
        right = _row_value(expression.right, row, reference)
        if left is None or right is None:
            return None
        return {"+": left + right, "-": left - right, "*": left * right}[expression.op]
    if isinstance(expression, And):
        result: bool | None = True
        for part in expression.parts:
            result = reference.and_(result, _row_value(part, row, reference))
        return result
    if isinstance(expression, Or):
        result = False
        for part in expression.parts:
            result = reference.or_(result, _row_value(part, row, reference))
        return result
    if isinstance(expression, Not):
        return reference.not_(_row_value(expression.part, row, reference))
    if isinstance(expression, IsNull):
        missing = _row_value(expression.part, row, reference) is None
        return not missing if expression.negated else missing
    if isinstance(expression, InList):
        value = _row_value(expression.part, row, reference)
        if value is None:
            return None
        return value in expression.options
    raise ConfigError(f"{type(expression).__name__} has no row form")


def describe(expression: Expr) -> str:
    """A readable rendering, for plan printing and for error messages."""
    if isinstance(expression, Literal):
        return repr(expression.value)
    if isinstance(expression, ColumnRef):
        return expression.name
    if isinstance(expression, Compare):
        return f"({describe(expression.left)} {expression.op} {describe(expression.right)})"
    if isinstance(expression, Arithmetic):
        return f"({describe(expression.left)} {expression.op} {describe(expression.right)})"
    if isinstance(expression, And):
        return "(" + " and ".join(describe(part) for part in expression.parts) + ")"
    if isinstance(expression, Or):
        return "(" + " or ".join(describe(part) for part in expression.parts) + ")"
    if isinstance(expression, Not):
        return f"(not {describe(expression.part)})"
    if isinstance(expression, IsNull):
        word = "is not null" if expression.negated else "is null"
        return f"({describe(expression.part)} {word})"
    if isinstance(expression, InList):
        return f"({describe(expression.part)} in {list(expression.options)})"
    raise ConfigError(f"{type(expression).__name__} has no rendering")


def conjuncts(expression: Expr) -> list[Expr]:
    """Split a predicate into the parts joined by and, flattening nested ands.

    The operation predicate pushdown is built on. A conjunction can be split and its parts
    distributed to wherever their columns come from; a disjunction cannot, which is why
    plan/rules/pushdown.py measures how much of a real workload is conjunctive.
    """
    if isinstance(expression, And):
        out: list[Expr] = []
        for part in expression.parts:
            out.extend(conjuncts(part))
        return out
    return [expression]


def all_of(parts: Sequence[Expr]) -> Expr:
    """Rejoin a list of conjuncts, collapsing the single and empty cases."""
    if not parts:
        return Literal(True, BOOLEAN)
    if len(parts) == 1:
        return parts[0]
    return And(tuple(parts))
