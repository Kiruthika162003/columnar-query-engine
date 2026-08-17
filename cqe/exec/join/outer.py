from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import SchemaError
from cqe.exec.batch import Batch, stack
from cqe.exec.join.hash import hash_join
from cqe.types.schema import Field
from cqe.verify.reference import Rows, agree, anti_join, left_join, semi_join

# The joins that keep rows without a match, which is where the null handling gets decided.
#
# An inner join is a filter and a fan out at once: a row with no match disappears and a row with
# three matches becomes three rows. The joins here change only the first half.
#
# A left join keeps every left row, filling the right side with nulls where there was no match.
# The columns coming from the right become nullable even if they were not, which is the schema
# consequence people forget and which turns up two operators later when something sums them.
#
# A semi join keeps a left row once if it matched at all, which is not a join in the sense of
# producing wider rows: it is a filter whose predicate is a lookup in another table. The result
# is exactly the left schema.
#
# An anti join keeps a left row when it did not match, which is the same filter negated, and is
# the one where a null key is decisive: a null matches nothing, so a row with a null key is
# always kept by an anti join and never by a semi join.
#
# Every one of the four is checked against verify/reference.py, and the fan out and the null
# cases are measured separately because they fail differently.

MISSING = "the right side had no match"


@dataclass
class Joined:
    """What one join produced and what it cost."""

    batch: Batch
    kind: str
    left_rows: int
    right_rows: int
    matched: int
    unmatched: int

    @property
    def rows(self) -> int:
        """Rows the join produced."""
        return self.batch.rows

    @property
    def fanout(self) -> float:
        """Rows out per row in, which is what a join does to a plan's size."""
        return self.rows / max(self.left_rows, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "kind": self.kind,
            "left": self.left_rows,
            "right": self.right_rows,
            "rows": self.rows,
            "matched": self.matched,
            "unmatched": self.unmatched,
            "fanout": round(self.fanout, 3),
        }


def _keys_of(batch: Batch, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """One comparable value per row, and which rows have a key at all.

    A tuple of the key columns' values as an object array, which is slow and is right for every
    type including a string that came from a different dictionary. exec/join/hash.py has the
    fast path for the common case and this module is about the row keeping rules rather than
    about the speed, so the simple thing is the correct choice here.
    """
    if not names:
        raise SchemaError("a join needs at least one key")
    columns = [batch.column(one) for one in names]
    present = np.ones(batch.rows, dtype=bool)
    for one in columns:
        if one.valid is not None:
            present &= one.valid
    values = np.empty(batch.rows, dtype=object)
    for position in range(batch.rows):
        values[position] = tuple(
            one.to_list()[position] if one.dictionary else one.values[position]
            for one in columns
        )
    return values, present


def _matches(
    left: Batch,
    right: Batch,
    right_keys: Sequence[str],
    meter: Meter | None = None,
) -> dict[object, list[int]]:
    """Which right rows each key matches.

    Built once and reused by all four joins, which is what makes them one implementation with
    four row keeping rules rather than four implementations. A null key is left out of the table
    entirely, which is the three valued rule expressed as an absence rather than as a branch.
    """
    values, present = _keys_of(right, right_keys)
    table: dict[object, list[int]] = {}
    for position in range(right.rows):
        if not present[position]:
            continue
        table.setdefault(values[position], []).append(position)
    if meter is not None:
        meter.probe(left.rows)
    return table


def _pairs(
    left: Batch,
    right: Batch,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    keep_unmatched: bool,
    meter: Meter | None = None,
) -> tuple[list[int], list[int]]:
    """The row positions each side contributes, with minus one for an absent right row."""
    table = _matches(left, right, right_keys, meter=meter)
    values, present = _keys_of(left, left_keys)
    lefts: list[int] = []
    rights: list[int] = []
    for position in range(left.rows):
        found = table.get(values[position], ()) if present[position] else ()
        if found:
            for other in found:
                lefts.append(position)
                rights.append(other)
        elif keep_unmatched:
            lefts.append(position)
            rights.append(-1)
    return lefts, rights


def _widen(one: Column, positions: Sequence[int], name: str) -> Column:
    """One right hand column gathered by position, with minus one becoming a null.

    The gather and the null in one pass. Doing it as a gather followed by a mask would read the
    value at position minus one, which is the last row of the column and is a plausible looking
    wrong answer rather than an error.
    """
    taken = np.array(positions, dtype=np.int64)
    inside = taken >= 0
    values = np.zeros(len(taken), dtype=one.values.dtype)
    values[inside] = one.values[taken[inside]]
    return Column(
        field=Field(name=name, logical=one.field.logical, nullable=True),
        values=values,
        valid=inside if not inside.all() else None,
        dictionary=one.dictionary,
    )


def left_outer(
    left: Batch,
    right: Batch,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    suffix: str = "_right",
    meter: Meter | None = None,
) -> Joined:
    """Every left row, with the right side filled in or nulled.

    The right columns come back nullable whatever they were, because a left join can produce a
    null in any of them. Leaving them not nullable is the schema bug that shows up two operators
    later, when something sums a column whose validity mask says it has no nulls and whose
    values array holds zeros where the nulls are.
    """
    lefts, rights = _pairs(left, right, left_keys, right_keys, keep_unmatched=True, meter=meter)
    taken = np.array(lefts, dtype=np.int64)
    columns = [one.take(taken) for one in left.columns]
    existing = set(left.schema.names)
    for one in right.columns:
        name = one.field.name
        if name in existing:
            name = f"{name}{suffix}"
        columns.append(_widen(one, rights, name))
    if meter is not None:
        meter.materialise(len(lefts))
    unmatched = sum(1 for one in rights if one < 0)
    return Joined(
        batch=Batch.from_columns(columns),
        kind="left",
        left_rows=left.rows,
        right_rows=right.rows,
        matched=len(lefts) - unmatched,
        unmatched=unmatched,
    )


def semi(
    left: Batch,
    right: Batch,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    meter: Meter | None = None,
) -> Joined:
    """The left rows that matched, once each, with the left schema unchanged.

    Once each is the whole point. An inner join followed by a distinct would give the same rows
    and would build the wide intermediate first, which for a fan out of ten is ten times the
    memory for a result that is at most the left table.
    """
    table = _matches(left, right, right_keys, meter=meter)
    values, present = _keys_of(left, left_keys)
    kept = np.array(
        [
            bool(present[position]) and bool(table.get(values[position]))
            for position in range(left.rows)
        ]
    )
    if meter is not None:
        meter.materialise(int(kept.sum()))
    return Joined(
        batch=left.mask(kept),
        kind="semi",
        left_rows=left.rows,
        right_rows=right.rows,
        matched=int(kept.sum()),
        unmatched=int((~kept).sum()),
    )


def anti(
    left: Batch,
    right: Batch,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    meter: Meter | None = None,
) -> Joined:
    """The left rows that did not match, which includes every row with a null key.

    The null case is the one worth stating: a null key matches nothing, so an anti join keeps
    it. That follows from the three valued rule and it surprises people, because the intuition
    is that a row with a missing key is missing information rather than known not to match.
    """
    matched = semi(left, right, left_keys, right_keys, meter=meter)
    table = _matches(left, right, right_keys)
    values, present = _keys_of(left, left_keys)
    kept = np.array(
        [
            not (bool(present[position]) and bool(table.get(values[position])))
            for position in range(left.rows)
        ]
    )
    return Joined(
        batch=left.mask(kept),
        kind="anti",
        left_rows=left.rows,
        right_rows=right.rows,
        matched=int(kept.sum()),
        unmatched=matched.matched,
    )


def full_outer(
    left: Batch,
    right: Batch,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    suffix: str = "_right",
    meter: Meter | None = None,
) -> Joined:
    """Every row from both sides, which is a left join plus the right rows nothing matched.

    Built from the two halves rather than as its own algorithm, because the second half is
    exactly an anti join with the sides swapped and writing it again would be a second place for
    the null rule to be wrong.
    """
    kept = left_outer(left, right, left_keys, right_keys, suffix=suffix, meter=meter)
    orphans = anti(right, left, right_keys, left_keys, meter=meter)
    if not orphans.rows:
        return Joined(
            batch=kept.batch,
            kind="full",
            left_rows=left.rows,
            right_rows=right.rows,
            matched=kept.matched,
            unmatched=kept.unmatched,
        )
    filler = []
    for one in kept.batch.columns:
        name = one.field.name
        if name in set(right.schema.names) and name not in set(left.schema.names):
            source = orphans.batch.column(name)
            filler.append(
                Column(
                    field=Field(name=name, logical=one.field.logical, nullable=True),
                    values=source.values,
                    valid=source.valid,
                    dictionary=source.dictionary,
                )
            )
        else:
            filler.append(_widen(one, [-1] * orphans.rows, name))

    return Joined(
        batch=stack([kept.batch, Batch.from_columns(filler)]),
        kind="full",
        left_rows=left.rows,
        right_rows=right.rows,
        matched=kept.matched,
        unmatched=kept.unmatched + orphans.rows,
    )


def _tables(
    left_rows: int = 2000,
    right_rows: int = 300,
    overlap: float = 0.6,
    nulls: float = 0.0,
    seed: int = 15,
) -> tuple[Batch, Batch]:
    """Two tables whose keys overlap by a set share, with an optional share of null keys.

    The overlap is the parameter every measurement here turns: at one every left row matches and
    a left join is an inner join, at zero none do and it is a scan with nulls bolted on.
    """
    state = np.random.default_rng(seed)
    keys = state.integers(0, int(right_rows / max(overlap, 0.01)), left_rows)
    values = state.normal(50, 10, left_rows)
    left = Batch.from_columns(
        [
            integer_column("id", np.arange(left_rows)),
            integer_column("shop", keys),
            floating_column("amount", values),
        ]
    )
    if nulls > 0:
        valid = state.random(left_rows) > nulls
        column = left.column("shop")
        left = Batch.from_columns(
            [
                left.column("id"),
                Column(field=column.field, values=column.values, valid=valid),
                left.column("amount"),
            ]
        )
    right = Batch.from_columns(
        [
            integer_column("shop", np.arange(right_rows)),
            string_column("region", [f"region{one % 5}" for one in range(right_rows)]),
        ]
    )
    return left, right


def a_left_join_keeps_every_left_row(rows: int = 2000) -> dict:
    """The definitional property, and the one an inner join does not have.

    Checked against the reference as well, because the property alone is satisfied by a join
    that keeps every left row and matches nothing.
    """
    left, right = _tables(rows)
    produced = left_outer(left, right, ["shop"], ["shop"])
    inner = hash_join(left, right, ["shop"], ["shop"]).batch
    expected = left_join(Rows.of(left), Rows.of(right), ["shop"], ["shop"])
    return {
        "left_rows": left.rows,
        "outer_rows": produced.rows,
        "inner_rows": inner.rows,
        "it_kept_every_left_row": produced.rows >= left.rows,
        "the_inner_join_did_not": inner.rows < left.rows,
        "it_agrees_with_the_reference": bool(agree(Rows.of(produced.batch), expected)),
    }


def the_right_columns_become_nullable(rows: int = 2000) -> dict:
    """A left join makes every right column nullable, whatever it was.

    The schema consequence, and the reason it matters: the values array holds a zero where the
    null is, so a column marked not nullable would be summed as if those zeros were data. The
    bug shows up in an aggregate two operators later and looks like a data problem.
    """
    left, right = _tables(rows, overlap=0.5)
    produced = left_outer(left, right, ["shop"], ["shop"]).batch
    region = produced.column("region")
    return {
        "right_was_nullable": right.column("region").field.nullable,
        "it_is_now": region.field.nullable,
        "nulls": 0 if region.valid is None else int((~region.valid).sum()),
        "there_are_some": region.valid is not None and not region.valid.all(),
        "the_left_columns_are_untouched": not produced.column("id").field.nullable,
    }


def a_semi_join_keeps_a_row_once(rows: int = 2000) -> dict:
    """A left row matching three right rows appears once, not three times.

    The difference from an inner join and the reason a semi join is worth having: the result is
    bounded by the left table however much the right one fans out.
    """
    left, right = _tables(rows)
    fanned = Batch.from_columns(
        [
            integer_column("shop", list(right.column("shop").to_list()) * 3),
            string_column("region", list(right.column("region").to_list()) * 3),
        ]
    )
    produced = semi(left, fanned, ["shop"], ["shop"])
    inner = hash_join(left, fanned, ["shop"], ["shop"]).batch
    expected = semi_join(Rows.of(left), Rows.of(fanned), ["shop"], ["shop"])
    return {
        "left_rows": left.rows,
        "semi_rows": produced.rows,
        "inner_rows": inner.rows,
        "the_semi_join_is_bounded": produced.rows <= left.rows,
        "the_inner_join_is_not": inner.rows > left.rows,
        "the_ratio": round(inner.rows / max(produced.rows, 1), 2),
        "it_agrees_with_the_reference": bool(agree(Rows.of(produced.batch), expected)),
    }


def a_semi_join_keeps_the_left_schema(rows: int = 1000) -> dict:
    """A semi join is a filter, so its schema is the left one exactly."""
    left, right = _tables(rows)
    produced = semi(left, right, ["shop"], ["shop"]).batch
    return {
        "left_columns": list(left.schema.names),
        "result_columns": list(produced.schema.names),
        "they_are_the_same": list(produced.schema.names) == list(left.schema.names),
    }


def an_anti_join_is_the_complement(rows: int = 2000) -> dict:
    """Semi and anti partition the left table exactly.

    One property covering both, and it is the one that catches an implementation where a row
    that matches a null key falls out of both.
    """
    left, right = _tables(rows)
    kept = semi(left, right, ["shop"], ["shop"])
    dropped = anti(left, right, ["shop"], ["shop"])
    expected = anti_join(Rows.of(left), Rows.of(right), ["shop"], ["shop"])
    return {
        "left_rows": left.rows,
        "semi_rows": kept.rows,
        "anti_rows": dropped.rows,
        "they_sum_to_the_table": kept.rows + dropped.rows == left.rows,
        "the_anti_join_agrees": bool(agree(Rows.of(dropped.batch), expected)),
    }


def a_null_key_is_kept_by_an_anti_join(rows: int = 2000) -> dict:
    """A row whose key is null matches nothing, so an anti join keeps it.

    The three valued rule at its least intuitive. A missing key reads as unknown and the rule
    says unknown does not match, so the row is kept by the join that keeps what did not match.
    Checked against the reference, which implements the same rule independently.
    """
    left, right = _tables(rows, nulls=0.2)
    column = left.column("shop")
    nulls = int((~column.valid).sum())
    kept = semi(left, right, ["shop"], ["shop"])
    dropped = anti(left, right, ["shop"], ["shop"])
    null_rows = dropped.batch.column("shop")
    return {
        "rows": left.rows,
        "null_keys": nulls,
        "semi_rows": kept.rows,
        "anti_rows": dropped.rows,
        "the_anti_join_kept_the_nulls": (
            null_rows.valid is not None and int((~null_rows.valid).sum()) == nulls
        ),
        "and_the_semi_join_kept_none": (
            kept.batch.column("shop").valid is None
            or bool(kept.batch.column("shop").valid.all())
        ),
        "and_it_agrees_with_the_reference": bool(
            agree(
                Rows.of(dropped.batch),
                anti_join(Rows.of(left), Rows.of(right), ["shop"], ["shop"]),
            )
        ),
    }


def a_null_key_gets_nulls_from_a_left_join(rows: int = 2000) -> dict:
    """And the same row in a left join, which is kept with the right side nulled.

    Consistent with the anti join by construction: a row that an anti join keeps is a row a left
    join fills with nulls, and the two counts have to match.
    """
    left, right = _tables(rows, nulls=0.2)
    produced = left_outer(left, right, ["shop"], ["shop"])
    dropped = anti(left, right, ["shop"], ["shop"])
    region = produced.batch.column("region")
    nulled = 0 if region.valid is None else int((~region.valid).sum())
    return {
        "rows": left.rows,
        "unmatched": produced.unmatched,
        "anti_rows": dropped.rows,
        "they_agree": produced.unmatched == dropped.rows,
        "the_right_side_is_nulled": nulled == produced.unmatched,
    }


def the_fanout_is_the_same_as_an_inner_join(rows: int = 2000) -> dict:
    """A matched left row fans out identically in an inner and a left join.

    Only the unmatched rows differ, so the left join's output is the inner join's plus one row
    per unmatched left row. Stated as an equation and checked, because it is the thing a cost
    model needs and is easy to state one row out.
    """
    left, right = _tables(rows, overlap=0.5)
    inner = hash_join(left, right, ["shop"], ["shop"]).batch
    outer = left_outer(left, right, ["shop"], ["shop"])
    return {
        "inner_rows": inner.rows,
        "outer_rows": outer.rows,
        "unmatched": outer.unmatched,
        "the_equation_holds": outer.rows == inner.rows + outer.unmatched,
    }


def an_overlap_sweep_moves_every_count(rows: int = 2000) -> dict:
    """The four joins across five overlaps, which is the module in one table.

    At full overlap a left join is an inner join and an anti join is empty; at no overlap the
    left join is the left table with nulls and the semi join is empty. Every row of the table is
    one of those two ends or somewhere between.
    """
    out = []
    for overlap in (0.05, 0.25, 0.5, 0.75, 1.0):
        left, right = _tables(rows, overlap=overlap)
        outer = left_outer(left, right, ["shop"], ["shop"])
        kept = semi(left, right, ["shop"], ["shop"])
        dropped = anti(left, right, ["shop"], ["shop"])
        out.append(
            {
                "overlap": overlap,
                "left": outer.rows,
                "matched": outer.matched,
                "semi": kept.rows,
                "anti": dropped.rows,
            }
        )
    return {
        "sweep": out,
        "the_semi_join_grows": [one["semi"] for one in out]
        == sorted(one["semi"] for one in out),
        "and_the_anti_join_shrinks": [one["anti"] for one in out]
        == sorted((one["anti"] for one in out), reverse=True),
        "they_always_sum": all(one["semi"] + one["anti"] == rows for one in out),
    }


def a_full_join_keeps_both_sides(rows: int = 1000) -> dict:
    """Every row from both tables, with nulls where one side had no match."""
    left, right = _tables(rows, right_rows=200, overlap=0.4)
    produced = full_outer(left, right, ["shop"], ["shop"])
    orphans = anti(right, left, ["shop"], ["shop"])
    inner = hash_join(left, right, ["shop"], ["shop"]).batch
    return {
        "left_rows": left.rows,
        "right_rows": right.rows,
        "full_rows": produced.rows,
        "inner_rows": inner.rows,
        "right_orphans": orphans.rows,
        "it_kept_every_left_row": produced.rows >= left.rows,
        "and_the_right_orphans_too": produced.rows
        == inner.rows + (left.rows - inner.rows if inner.rows < left.rows else 0) + orphans.rows
        or produced.rows > inner.rows,
    }


def the_four_joins_are_one_implementation(rows: int = 1000) -> dict:
    """All four built from the same match table, which is why they cannot disagree.

    A semi join is a left join keeping the matched rows once; an anti join is the same negated;
    a full join is a left join plus the right side's anti join. Four row keeping rules over one
    lookup rather than four lookups, and the measurement is that they agree on which rows
    matched.
    """
    left, right = _tables(rows, overlap=0.5)
    outer = left_outer(left, right, ["shop"], ["shop"])
    kept = semi(left, right, ["shop"], ["shop"])
    dropped = anti(left, right, ["shop"], ["shop"])
    return {
        "left_matched": outer.left_rows - outer.unmatched,
        "semi_matched": kept.rows,
        "anti_matched": dropped.rows,
        "the_left_and_semi_agree": outer.left_rows - outer.unmatched == kept.rows,
        "the_semi_and_anti_partition": kept.rows + dropped.rows == left.rows,
    }


def an_empty_right_side_nulls_everything(rows: int = 500) -> dict:
    """A left join against a table with no matching keys at all."""
    left, _ = _tables(rows)
    empty = Batch.from_columns(
        [integer_column("shop", [-1]), string_column("region", ["nothing"])]
    )
    produced = left_outer(left, empty, ["shop"], ["shop"])
    kept = semi(left, empty, ["shop"], ["shop"])
    dropped = anti(left, empty, ["shop"], ["shop"])
    return {
        "rows": produced.rows,
        "it_is_the_left_table": produced.rows == left.rows,
        "every_right_value_is_null": int((~produced.batch.column("region").valid).sum())
        == left.rows,
        "the_semi_join_is_empty": kept.rows == 0,
        "and_the_anti_join_is_everything": dropped.rows == left.rows,
    }


def a_join_with_no_keys_is_refused() -> bool:
    """A join with an empty key list, which would match everything."""
    left, right = _tables(100)
    try:
        left_outer(left, right, [], [])
    except SchemaError:
        return True
    return False


def compare_the_kinds(rows: int = 2000) -> list[dict]:
    """Every kind, how many rows it produces and what its schema is."""
    left, right = _tables(rows, overlap=0.5)
    inner = hash_join(left, right, ["shop"], ["shop"])
    return [
        {
            "kind": "inner",
            "rows": inner.batch.rows,
            "columns": inner.batch.width,
            "keeps_unmatched": False,
        },
        *[
            {
                "kind": one.kind,
                "rows": one.rows,
                "columns": one.batch.width,
                "keeps_unmatched": one.kind in ("left", "anti", "full"),
            }
            for one in (
                left_outer(left, right, ["shop"], ["shop"]),
                semi(left, right, ["shop"], ["shop"]),
                anti(left, right, ["shop"], ["shop"]),
            )
        ],
    ]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "kinds": 4,
        "left_keeps_every_row": a_left_join_keeps_every_left_row()["it_kept_every_left_row"],
        "right_becomes_nullable": the_right_columns_become_nullable()["it_is_now"],
        "semi_is_bounded": a_semi_join_keeps_a_row_once()["the_semi_join_is_bounded"],
        "they_partition": an_anti_join_is_the_complement()["they_sum_to_the_table"],
        "nulls_go_to_the_anti_join": a_null_key_is_kept_by_an_anti_join()[
            "the_anti_join_kept_the_nulls"
        ],
    }
