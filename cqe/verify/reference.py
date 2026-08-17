from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from cqe.errors import ConfigError, SchemaError, TypeMismatch, UnknownColumn
from cqe.exec.batch import Batch, from_rows

# The obviously correct engine, which every fast path in this package is checked against.
#
# It holds rows as lists of Python values, evaluates one row at a time, and does nothing clever.
# There are no encodings, no vectorisation, no hash tables, no statistics, no plan. A filter is
# a list comprehension. A join is two nested loops. A group by is a dictionary keyed on a tuple.
# Every one of those is the definition of what the operator means rather than an implementation
# of it, which is what makes this useful: when the vectorised engine and this one disagree, this
# one is right, and the disagreement is a bug in the fast path by construction.
#
# It is slow and that is a feature. Nobody is tempted to optimise it, so it stays readable, so
# it stays trustworthy. The moment somebody adds a dictionary to speed up a join here, the whole
# arrangement is worth nothing.
#
# Two rules keep it honest. It never imports from cqe.exec beyond the Batch container, so no
# operator can be shared between the two engines and quietly be wrong in both. And it defines
# null semantics itself, from scratch, rather than deferring: a comparison against null is null
# and not false, a null is excluded from an aggregate but counted by count star, and two nulls
# are not equal in a join but are equal in a group by. Those five rules are where every engine
# disagrees with every other engine, and having them written out once in Python is worth more
# than the rest of the module.


@dataclass
class Rows:
    """A table as a list of Python rows, with column names."""

    names: tuple[str, ...]
    rows: list[list]

    def __post_init__(self) -> None:
        if not self.names:
            raise SchemaError("a table needs at least one column")
        widths = {len(row) for row in self.rows}
        if widths and widths != {len(self.names)}:
            raise SchemaError(
                f"rows of widths {sorted(widths)} against {len(self.names)} names"
            )

    @classmethod
    def of(cls, batch: Batch) -> Rows:
        """Convert a columnar batch into rows."""
        return cls(names=batch.names, rows=batch.to_rows())

    def to_batch(self) -> Batch:
        """Convert back, for comparing against a columnar result."""
        return from_rows(list(self.names), self.rows)

    def index(self, name: str) -> int:
        """Where a column sits, by name."""
        if name not in self.names:
            raise UnknownColumn(f"{name} is not a column; the table has {list(self.names)}")
        return self.names.index(name)

    def column(self, name: str) -> list:
        """One column as a Python list."""
        position = self.index(name)
        return [row[position] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        """How many columns there are."""
        return len(self.names)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"rows": len(self.rows), "columns": self.width, "names": list(self.names)}


def compare(left, right) -> int | None:
    """Three way comparison with null semantics, returning None when either side is null.

    None rather than an exception and none rather than false, because a comparison with an
    unknown value has an unknown answer and every downstream rule here reads that None.
    Collapsing it to false at this point is the single most common way an engine gets nulls
    wrong, since false and false are indistinguishable once combined but null and false are not.
    """
    if left is None or right is None:
        return None
    if isinstance(left, bool) != isinstance(right, bool):
        raise TypeMismatch(f"{type(left).__name__} against {type(right).__name__}")
    if isinstance(left, str) != isinstance(right, str):
        raise TypeMismatch(f"{type(left).__name__} against {type(right).__name__}")
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def truth(value: bool | None) -> bool:
    """Collapse three valued logic to two, which only a filter is allowed to do.

    A row is kept when the predicate is true, and dropped when it is false or null. That is the
    rule and it is asymmetric on purpose: not (x = 1) does not keep the rows where x is null,
    and an engine that treats null as false everywhere gets that wrong in the other direction.
    """
    return value is True


def and_(left: bool | None, right: bool | None) -> bool | None:
    """Three valued and, where false wins over null."""
    if left is False or right is False:
        return False
    if left is None or right is None:
        return None
    return True


def or_(left: bool | None, right: bool | None) -> bool | None:
    """Three valued or, where true wins over null."""
    if left is True or right is True:
        return True
    if left is None or right is None:
        return None
    return False


def not_(value: bool | None) -> bool | None:
    """Three valued not, where null stays null."""
    if value is None:
        return None
    return not value


def equals(left, right) -> bool | None:
    """Equality with null semantics, which is not the same as identity of keys."""
    result = compare(left, right)
    return None if result is None else result == 0


def distinct_from(left, right) -> bool:
    """Equality where two nulls are equal, which is what a group by key needs.

    The other equality. A join matches on equals and drops null keys; a group by collects nulls
    into a group of their own. Both are correct and they are different functions, so they are
    two functions here rather than one with a flag.
    """
    if left is None and right is None:
        return False
    if left is None or right is None:
        return True
    return left != right


def select(table: Rows, names: Sequence[str]) -> Rows:
    """Projection: keep the named columns in the order given."""
    positions = [table.index(name) for name in names]
    return Rows(
        names=tuple(names),
        rows=[[row[position] for position in positions] for row in table.rows],
    )


def where(table: Rows, predicate: Callable[[dict], bool | None]) -> Rows:
    """Selection: keep the rows whose predicate is true, dropping false and null."""
    kept = []
    for row in table.rows:
        binding = dict(zip(table.names, row, strict=True))
        if truth(predicate(binding)):
            kept.append(list(row))
    return Rows(names=table.names, rows=kept)


def order_by(
    table: Rows,
    keys: Sequence[str],
    descending: Sequence[bool] | None = None,
    nulls_first: bool = False,
) -> Rows:
    """Sorting, with an explicit rule for where nulls go.

    Nulls last by default, which is what most engines do for ascending and what almost none do
    consistently for descending. Here the flag means what it says in both directions:
    nulls_first puts them first whatever the sort order, rather than flipping with it.
    """
    if not keys:
        raise ConfigError("a sort needs at least one key")
    order = list(descending) if descending is not None else [False] * len(keys)
    if len(order) != len(keys):
        raise ConfigError(f"{len(order)} directions against {len(keys)} keys")
    positions = [table.index(name) for name in keys]

    def rank(row: list) -> tuple:
        parts: list = []
        for position, falling in zip(positions, order, strict=True):
            value = row[position]
            missing = value is None
            flag = 0 if (missing == nulls_first) else 1
            parts.append(flag)
            if missing:
                parts.append(0)
            elif isinstance(value, bool):
                parts.append(-int(value) if falling else int(value))
            elif isinstance(value, str):
                parts.append(value)
            else:
                parts.append(-value if falling else value)
        return tuple(parts)

    try:
        ordered = sorted(table.rows, key=rank)
    except TypeError as mixed:
        raise TypeMismatch(f"cannot sort mixed types on {list(keys)}") from mixed
    return Rows(names=table.names, rows=[list(row) for row in ordered])


def group_by(
    table: Rows,
    keys: Sequence[str],
    aggregates: Sequence[tuple[str, str, str]],
) -> Rows:
    """Grouping, with each aggregate written out as a plain Python loop.

    Aggregates are (name, function, column) triples. Nulls are skipped by every aggregate except
    count star, which counts rows, and count column, which counts non nulls. A sum over no non
    null values is null and not zero, which is the rule everybody gets wrong in the first
    version and which exec/aggregate.py is checked against here.
    """
    positions = [table.index(name) for name in keys]
    buckets: dict[tuple, list[list]] = {}
    order: list[tuple] = []
    for row in table.rows:
        key = tuple(row[position] for position in positions)
        marker = tuple((value is None, "" if value is None else value) for value in key)
        if marker not in buckets:
            buckets[marker] = []
            order.append(marker)
        buckets[marker].append(list(row))

    names = tuple(list(keys) + [name for name, _, _ in aggregates])
    out: list[list] = []
    for marker in order:
        rows = buckets[marker]
        key = [None if missing else value for missing, value in marker]
        line = list(key)
        for _, function, source in aggregates:
            line.append(apply_aggregate(function, table, rows, source))
        out.append(line)
    return Rows(names=names, rows=out)


def apply_aggregate(function: str, table: Rows, rows: Sequence[list], source: str):
    """One aggregate over one group, written out rather than dispatched to numpy."""
    if function == "count_star":
        return len(rows)
    position = table.index(source)
    present = [row[position] for row in rows if row[position] is not None]
    if function == "count":
        return len(present)
    if not present:
        return None
    if function == "sum":
        return sum(present)
    if function == "min":
        return min(present)
    if function == "max":
        return max(present)
    if function == "mean":
        return sum(present) / len(present)
    if function == "any":
        return any(bool(value) for value in present)
    if function == "all":
        return all(bool(value) for value in present)
    raise ConfigError(f"{function} is not an aggregate")


def inner_join(
    left: Rows,
    right: Rows,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    suffix: str = "_right",
) -> Rows:
    """A join written as two nested loops, which is the definition of one.

    Null keys never match, including against another null, which is where this differs from the
    grouping rule above. Output order is left row then right row, which is what makes a
    comparison against a hash join meaningful only after both sides are sorted.
    """
    if len(left_keys) != len(right_keys):
        raise ConfigError(f"{len(left_keys)} keys against {len(right_keys)}")
    if not left_keys:
        raise ConfigError("a join needs at least one key")
    left_positions = [left.index(name) for name in left_keys]
    right_positions = [right.index(name) for name in right_keys]
    names = list(left.names) + _disambiguate(left.names, right.names, suffix)

    out: list[list] = []
    for left_row in left.rows:
        left_key = [left_row[position] for position in left_positions]
        if any(value is None for value in left_key):
            continue
        for right_row in right.rows:
            right_key = [right_row[position] for position in right_positions]
            if any(value is None for value in right_key):
                continue
            if all(equals(a, b) is True for a, b in zip(left_key, right_key, strict=True)):
                out.append(list(left_row) + list(right_row))
    return Rows(names=tuple(names), rows=out)


def left_join(
    left: Rows,
    right: Rows,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    suffix: str = "_right",
) -> Rows:
    """The same, keeping unmatched left rows with nulls on the right."""
    if len(left_keys) != len(right_keys):
        raise ConfigError(f"{len(left_keys)} keys against {len(right_keys)}")
    left_positions = [left.index(name) for name in left_keys]
    right_positions = [right.index(name) for name in right_keys]
    names = list(left.names) + _disambiguate(left.names, right.names, suffix)
    blank = [None] * right.width

    out: list[list] = []
    for left_row in left.rows:
        left_key = [left_row[position] for position in left_positions]
        matched = False
        if not any(value is None for value in left_key):
            for right_row in right.rows:
                right_key = [right_row[position] for position in right_positions]
                if any(value is None for value in right_key):
                    continue
                if all(equals(a, b) is True for a, b in zip(left_key, right_key, strict=True)):
                    out.append(list(left_row) + list(right_row))
                    matched = True
        if not matched:
            out.append(list(left_row) + list(blank))
    return Rows(names=tuple(names), rows=out)


def semi_join(
    left: Rows,
    right: Rows,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
) -> Rows:
    """Left rows that have at least one match, each appearing once."""
    matched = inner_join(left, right, left_keys, right_keys)
    seen = set()
    out: list[list] = []
    for row in left.rows:
        marker = tuple("" if value is None else value for value in row)
        if marker in seen:
            continue
        for candidate in matched.rows:
            if candidate[: left.width] == row:
                out.append(list(row))
                seen.add(marker)
                break
    return Rows(names=left.names, rows=out)


def anti_join(
    left: Rows,
    right: Rows,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
) -> Rows:
    """Left rows with no match, which is not the complement of a semi join under nulls.

    A left row whose key is null matches nothing, so it belongs in the anti join output, and it
    is also not in the semi join output. That much agrees. The disagreement is that a right side
    containing a null does not make every left row match, which is what the correlated subquery
    version of this operator does and is a different operator.
    """
    kept = semi_join(left, right, left_keys, right_keys)
    keepers = {tuple("" if value is None else value for value in row) for row in kept.rows}
    out = [
        list(row)
        for row in left.rows
        if tuple("" if value is None else value for value in row) not in keepers
    ]
    return Rows(names=left.names, rows=out)


def distinct(table: Rows) -> Rows:
    """Remove duplicate rows, treating two nulls as the same."""
    seen: set[tuple] = set()
    out: list[list] = []
    for row in table.rows:
        marker = tuple((value is None, "" if value is None else value) for value in row)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(list(row))
    return Rows(names=table.names, rows=out)


def limit(table: Rows, count: int, offset: int = 0) -> Rows:
    """The first rows after an offset."""
    if count < 0 or offset < 0:
        raise ConfigError(f"limit {count} offset {offset} is not a window")
    return Rows(
        names=table.names, rows=[list(row) for row in table.rows[offset : offset + count]]
    )


def union_all(left: Rows, right: Rows) -> Rows:
    """Stack two tables with the same column names."""
    if left.names != right.names:
        raise SchemaError(f"{list(left.names)} against {list(right.names)}")
    return Rows(names=left.names, rows=[list(row) for row in left.rows + right.rows])


def _disambiguate(taken: Sequence[str], names: Sequence[str], suffix: str) -> list[str]:
    """Rename right side columns that clash with left side ones."""
    used = set(taken)
    out = []
    for name in names:
        candidate = name
        while candidate in used:
            candidate = f"{candidate}{suffix}"
        used.add(candidate)
        out.append(candidate)
    return out


@dataclass
class Difference:
    """Where two results disagree, and how."""

    kind: str
    detail: str
    left_row: list | None = None
    right_row: list | None = None

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kind": self.kind,
            "detail": self.detail,
            "left": self.left_row,
            "right": self.right_row,
        }


@dataclass
class Agreement:
    """Whether two results are the same, and the first few places they are not."""

    same: bool
    differences: list[Difference] = field(default_factory=list)
    left_rows: int = 0
    right_rows: int = 0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "same": self.same,
            "differences": len(self.differences),
            "left_rows": self.left_rows,
            "right_rows": self.right_rows,
            "first": self.differences[0].as_dict() if self.differences else None,
        }


def agree(
    left: Rows,
    right: Rows,
    ordered: bool = False,
    tolerance: float = 1e-9,
    limit_reported: int = 5,
) -> Agreement:
    """Whether two results hold the same rows, with an option to ignore order.

    Unordered by default, because most operators here make no order guarantee and comparing them
    as ordered would report a difference on every hash join. The ordered comparison is for the
    operators that do promise an order, and using it where it is not promised is how a test ends
    up asserting an implementation detail.

    Floating values compare within a tolerance, because a sum computed in a different order is a
    different float and neither is wrong. Integers and strings compare exactly.
    """
    differences: list[Difference] = []
    if list(left.names) != list(right.names):
        differences.append(
            Difference("names", f"{list(left.names)} against {list(right.names)}")
        )
        return Agreement(False, differences, len(left), len(right))
    if len(left) != len(right):
        differences.append(Difference("count", f"{len(left)} rows against {len(right)}"))

    if ordered:
        pairs = list(zip(left.rows, right.rows, strict=False))
    else:
        pairs = list(
            zip(
                sorted(left.rows, key=_sortable),
                sorted(right.rows, key=_sortable),
                strict=False,
            )
        )
    for one, other in pairs:
        if not _rows_match(one, other, tolerance):
            differences.append(Difference("value", "rows differ", list(one), list(other)))
            if len(differences) >= limit_reported:
                break
    return Agreement(not differences, differences, len(left), len(right))


def _rows_match(left: Sequence, right: Sequence, tolerance: float) -> bool:
    """Whether two rows hold the same values, floats within a tolerance."""
    if len(left) != len(right):
        return False
    for one, other in zip(left, right, strict=True):
        if one is None or other is None:
            if one is not other:
                return False
            continue
        if isinstance(one, float) or isinstance(other, float):
            if abs(float(one) - float(other)) > tolerance * max(1.0, abs(float(one))):
                return False
        elif one != other:
            return False
    return True


def _sortable(row: Sequence) -> tuple:
    """A key that orders rows of mixed types without raising."""
    return tuple(
        (0, "", 0.0)
        if value is None
        else (1, value, 0.0)
        if isinstance(value, str)
        else (2, "", float(value))
        for value in row
    )
