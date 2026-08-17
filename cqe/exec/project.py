from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import SchemaError, TypeMismatch, UnknownColumn
from cqe.exec.batch import Batch
from cqe.exec.expr import Arithmetic, Compare, Expr, column, describe, literal
from cqe.exec.filter import apply as apply_predicate
from cqe.types.schema import BOOLEAN, FLOATING, INTEGER, Schema

# Projection, which in a columnar engine is two different operations that share a name.
#
# Narrowing is choosing which columns come out. It copies nothing: a batch is a tuple of columns
# and a narrowed batch is a shorter tuple holding the same column objects. That is the operation
# every measurement in this package calls free, and this module is where the claim is checked
# rather than repeated.
#
# Computing is adding a column that did not exist, which is an expression evaluated once per
# row. That is not free and its cost is the expression's, so the interesting question is not
# whether to do it but where: computing before a filter evaluates the expression on rows the
# filter will throw away, and computing after evaluates it on fewer rows and needs the inputs to
# have survived the projection.
#
# The two are separated here because they have opposite rules. Narrowing wants to happen as
# early as possible and computing wants to happen as late as possible, and a single operator
# that did both would have to choose one.

MAX_DEPTH = 8


@dataclass(frozen=True)
class Computed:
    """One derived column: a name and the expression that makes it."""

    name: str
    expression: Expr

    def type_of(self, schema: Schema) -> str:
        """What type the expression produces over a schema."""
        return self.expression.type_of(schema)

    def describe(self) -> str:
        """One line, as it would be written."""
        return f"{describe(self.expression)} as {self.name}"

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "expression": describe(self.expression),
            "columns": sorted(self.expression.columns_used()),
        }


def narrow(batch: Batch, names: Sequence[str], meter: Meter | None = None) -> Batch:
    """The same rows with fewer columns, sharing the arrays rather than copying them.

    Refuses a name that is not there rather than dropping it quietly, and refuses a repeat
    rather than producing two columns with one name. Both are the kind of thing a caller does by
    accident and neither has a sensible interpretation.
    """
    wanted = list(names)
    missing = [one for one in wanted if one not in batch.schema]
    if missing:
        raise UnknownColumn(f"{missing} not in {list(batch.schema.names)}")
    if len(set(wanted)) != len(wanted):
        repeated = sorted({one for one in wanted if wanted.count(one) > 1})
        raise SchemaError(f"{repeated} appear more than once in a projection")
    if meter is not None:
        meter.batch()
    return Batch.from_columns([batch.column(one) for one in wanted])


def drop(batch: Batch, names: Sequence[str], meter: Meter | None = None) -> Batch:
    """The same rows without the named columns, which is narrowing spelled the other way.

    Worth having as its own function rather than as a caller's list comprehension, because the
    comprehension silently accepts a name that is not there and this does not.
    """
    unwanted = set(names)
    missing = sorted(unwanted - set(batch.schema.names))
    if missing:
        raise UnknownColumn(f"{missing} not in {list(batch.schema.names)}")
    keep = [one for one in batch.schema.names if one not in unwanted]
    if not keep:
        raise SchemaError("dropping every column leaves nothing")
    return narrow(batch, keep, meter=meter)


def rename(batch: Batch, mapping: dict[str, str], meter: Meter | None = None) -> Batch:
    """The same columns under different names.

    Free in the same sense narrowing is: a field is a name, a type and a nullability, and only
    the first changes. The values are the same arrays.
    """
    missing = sorted(set(mapping) - set(batch.schema.names))
    if missing:
        raise UnknownColumn(f"{missing} not in {list(batch.schema.names)}")
    out = []
    for one in batch.columns:
        name = mapping.get(one.field.name, one.field.name)
        out.append(one.renamed(name) if name != one.field.name else one)
    names = [one.field.name for one in out]
    if len(set(names)) != len(names):
        raise SchemaError(f"the rename produces a duplicate in {names}")
    if meter is not None:
        meter.batch()
    return Batch.from_columns(out)


def compute(batch: Batch, columns: Sequence[Computed], meter: Meter | None = None) -> Batch:
    """The same rows with derived columns added.

    Every expression is evaluated against the original batch rather than against the growing
    one, so a computed column cannot refer to another computed column. That is a real
    restriction and it is deliberate: allowing it means the order of the list matters, and a
    list whose order matters is a program rather than a projection.
    """
    if not columns:
        return batch
    names = [one.name for one in columns]
    clashing = sorted(set(names) & set(batch.schema.names))
    if clashing:
        raise SchemaError(f"{clashing} already exist; a computed column cannot replace one")
    if len(set(names)) != len(names):
        raise SchemaError(f"the computed names repeat in {names}")
    made = list(batch.columns)
    for one in columns:
        produced = one.expression.evaluate(batch, meter=meter)
        made.append(produced.renamed(one.name))
    if meter is not None:
        meter.materialise(batch.rows)
    return Batch.from_columns(made)


def project(
    batch: Batch,
    names: Sequence[str] | None = None,
    columns: Sequence[Computed] = (),
    meter: Meter | None = None,
) -> Batch:
    """Compute, then narrow, which is the only order that works.

    Narrowing first would drop the columns the expressions read. That is not a preference, it is
    the reason a projection with a computed column in it cannot be pushed below anything that
    narrows, and plan/rules/pushdown.py has to know it.
    """
    produced = compute(batch, columns, meter=meter) if columns else batch
    return narrow(produced, names, meter=meter) if names is not None else produced


def _table(rows: int = 20000, seed: int = 5) -> Batch:
    """A table with four columns of different widths."""
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 40, rows)),
            floating_column("amount", state.normal(100, 30, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 6, rows)]),
        ]
    )


def narrowing_copies_nothing(rows: int = 20000) -> dict:
    """A narrowed batch holds the same array objects as the one it came from.

    The claim every other module rests on, checked on object identity rather than on equality. A
    projection that copied would still be correct and every cost measurement in this package
    would be wrong by the size of the copy.
    """
    batch = _table(rows)
    narrowed = narrow(batch, ["id", "amount"])
    return {
        "columns_before": batch.width,
        "columns_after": narrowed.width,
        "the_arrays_are_the_same_objects": all(
            narrowed.column(one).values is batch.column(one).values for one in ("id", "amount")
        ),
        "and_the_columns_are_too": all(
            narrowed.column(one) is batch.column(one) for one in ("id", "amount")
        ),
    }


def narrowing_costs_nothing_measurable(rows: int = 20000) -> dict:
    """The meter counts one batch and no values touched.

    A projection is bookkeeping. The one batch counted is the output batch, which every operator
    records, and there is nothing else to charge for.
    """
    batch = _table(rows)
    meter = Meter()
    narrow(batch, ["id", "amount"], meter=meter)
    return {
        "rows": rows,
        "values_touched": meter.values_touched,
        "rows_materialised": meter.rows_materialised,
        "batches": meter.batches,
        "it_touched_nothing": meter.values_touched == 0,
        "and_materialised_nothing": meter.rows_materialised == 0,
    }


def computing_a_column_is_not_free(rows: int = 20000) -> dict:
    """And the other half: an expression is evaluated once per row.

    The contrast with narrowing is the point. Both are called projection and one is free and one
    is linear in the rows, so a plan that treats them as the same node cannot cost either.
    """
    batch = _table(rows)
    meter = Meter()
    computed = Computed(
        name="doubled", expression=Arithmetic("+", column("amount"), column("amount"))
    )
    compute(batch, [computed], meter=meter)
    return {
        "rows": rows,
        "values_touched": meter.values_touched,
        "rows_materialised": meter.rows_materialised,
        "it_touched_the_rows": meter.values_touched >= rows,
        "against_narrowing": narrowing_costs_nothing_measurable(rows)["values_touched"],
    }


def computing_after_a_filter_is_cheaper(rows: int = 20000) -> dict:
    """The same expression evaluated before and after a filter that keeps a tenth.

    The reason a projection with a computed column in it is pushed up rather than down, which is
    the opposite direction from a narrowing projection. The saving is the filter's selectivity
    exactly, because the expression is linear in the rows.
    """
    batch = _table(rows)
    predicate = Compare(">", column("amount"), literal(140.0))
    computed = Computed(
        name="doubled", expression=Arithmetic("+", column("amount"), column("amount"))
    )
    early = Meter()
    apply_predicate(predicate, compute(batch, [computed], meter=early), meter=early)
    late = Meter()
    kept = apply_predicate(predicate, batch, meter=late)
    compute(kept, [computed], meter=late)
    return {
        "rows": rows,
        "kept": kept.rows,
        "early_touched": early.values_touched,
        "late_touched": late.values_touched,
        "ratio": round(early.values_touched / max(late.values_touched, 1), 2),
        "later_is_cheaper": late.values_touched < early.values_touched,
    }


def narrowing_before_a_filter_is_cheaper(rows: int = 20000) -> dict:
    """And the direction narrowing wants to go, which is down.

    A filter over two columns touches two columns' worth of values; over four it touches four.
    Narrowing first is what makes the difference and it is why plan/rules/pushdown.py pushes a
    projection below a filter whenever the filter's columns survive it.
    """
    batch = _table(rows)
    predicate = Compare(">", column("amount"), literal(100.0))
    wide = Meter()
    apply_predicate(predicate, batch, meter=wide)
    thin = Meter()
    apply_predicate(predicate, narrow(batch, ["amount"], meter=thin), meter=thin)
    return {
        "wide_touched": wide.values_touched,
        "narrow_touched": thin.values_touched,
        "ratio": round(wide.values_touched / max(thin.values_touched, 1), 2),
        "narrowing_first_is_cheaper": thin.values_touched < wide.values_touched,
    }


def the_two_directions_are_opposite(rows: int = 20000) -> dict:
    """Both measurements in one place, which is the module's whole argument.

    Narrowing wants to be as early as possible and computing wants to be as late as possible. A
    single projection node holding both cannot be placed correctly, which is why they are two
    functions here and two rules there.
    """
    narrowing = narrowing_before_a_filter_is_cheaper(rows)
    computing = computing_after_a_filter_is_cheaper(rows)
    return {
        "narrowing_wants_to_go_down": narrowing["narrowing_first_is_cheaper"],
        "computing_wants_to_go_up": computing["later_is_cheaper"],
        "narrowing_saves": narrowing["ratio"],
        "computing_saves": computing["ratio"],
        "they_are_opposite": narrowing["narrowing_first_is_cheaper"]
        and computing["later_is_cheaper"],
    }


def a_computed_column_gets_the_right_type(rows: int = 100) -> dict:
    """What each expression produces, checked against the schema it declares.

    Worth checking because the declared type is what a plan reasons about and the produced type
    is what a query returns, and a difference between them is a plan that is right about a table
    that does not exist.
    """
    batch = _table(rows)
    schema = batch.schema
    made = {
        "sum": Computed("v", Arithmetic("+", column("amount"), column("amount"))),
        "mixed": Computed("v", Arithmetic("+", column("id"), column("amount"))),
        "integer": Computed("v", Arithmetic("*", column("id"), column("shop"))),
        "comparison": Computed("v", Compare(">", column("amount"), literal(100.0))),
    }
    out = {}
    for name, one in made.items():
        declared = one.type_of(schema)
        produced = compute(batch, [one]).column("v").field.logical
        out[name] = {"declared": declared, "produced": produced, "agree": declared == produced}
    return {
        **out,
        "they_all_agree": all(one["agree"] for one in out.values()),
        "an_integer_sum_stays_integer": out["integer"]["produced"] == INTEGER,
        "a_mixed_sum_becomes_floating": out["mixed"]["produced"] == FLOATING,
        "a_comparison_is_boolean": out["comparison"]["produced"] == BOOLEAN,
    }


def a_computed_column_cannot_read_another(rows: int = 100) -> dict:
    """Two computed columns where the second names the first, which is refused.

    Allowing it would make the order of the list matter, and a projection whose order matters is
    a program. The refusal names the column, so the caller can see it is a restriction rather
    than a bug.
    """
    batch = _table(rows)
    first = Computed("doubled", Arithmetic("+", column("amount"), column("amount")))
    second = Computed("quadrupled", Arithmetic("+", column("doubled"), column("doubled")))
    caught = ""
    try:
        compute(batch, [first, second])
    except (UnknownColumn, SchemaError, TypeMismatch) as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_the_column": "doubled" in caught,
        "and_two_independent_ones_work": compute(
            batch,
            [first, Computed("halved", Arithmetic("*", column("amount"), literal(2)))],
        ).width
        == batch.width + 2,
    }


def a_computed_column_cannot_replace_one(rows: int = 100) -> dict:
    """A derived column named after an existing one, which is refused.

    Replacing would be useful and would make the schema depend on the order of operations, and
    the alternative is that a caller renames first, which is one more line and no ambiguity.
    """
    batch = _table(rows)
    caught = ""
    try:
        compute(batch, [Computed("amount", Arithmetic("+", column("id"), literal(1)))])
    except SchemaError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "and_renaming_first_works": compute(
            rename(batch, {"amount": "old"}),
            [Computed("amount", Arithmetic("+", column("id"), literal(1)))],
        ).width
        == batch.width + 1,
    }


def projecting_computes_before_it_narrows(rows: int = 100) -> dict:
    """A projection that keeps only the derived column, which needs its inputs first.

    The order the function documents, checked by asking for something that would fail under the
    other one: keep only doubled, which is computed from amount, which is not kept.
    """
    batch = _table(rows)
    computed = Computed("doubled", Arithmetic("+", column("amount"), column("amount")))
    produced = project(batch, names=["doubled"], columns=[computed])
    return {
        "columns": list(produced.schema.names),
        "it_is_only_the_computed_one": list(produced.schema.names) == ["doubled"],
        "rows": produced.rows,
        "and_the_values_are_right": bool(
            np.allclose(produced.column("doubled").values, batch.column("amount").values * 2)
        ),
    }


def renaming_keeps_the_arrays(rows: int = 20000) -> dict:
    """A rename is free in the same sense narrowing is."""
    batch = _table(rows)
    renamed = rename(batch, {"amount": "value"})
    return {
        "names": list(renamed.schema.names),
        "it_renamed": "value" in renamed.schema and "amount" not in renamed.schema,
        "the_array_is_the_same_object": renamed.column("value").values
        is batch.column("amount").values,
        "the_others_are_untouched": renamed.column("id") is batch.column("id"),
    }


def dropping_is_narrowing_backwards(rows: int = 100) -> dict:
    """Dropping two of four columns and keeping two are the same operation."""
    batch = _table(rows)
    dropped = drop(batch, ["shop", "region"])
    kept = narrow(batch, ["id", "amount"])
    return {
        "dropped": list(dropped.schema.names),
        "kept": list(kept.schema.names),
        "they_agree": list(dropped.schema.names) == list(kept.schema.names),
    }


def a_missing_column_is_refused() -> bool:
    """Narrowing to a column that is not there."""
    try:
        narrow(_table(10), ["nothing"])
    except UnknownColumn:
        return True
    return False


def a_repeated_column_is_refused() -> bool:
    """Narrowing to the same column twice, which would give two columns one name."""
    try:
        narrow(_table(10), ["id", "id"])
    except SchemaError:
        return True
    return False


def dropping_everything_is_refused() -> bool:
    """A projection that keeps no columns at all."""
    try:
        drop(_table(10), ["id", "shop", "amount", "region"])
    except SchemaError:
        return True
    return False


def dropping_a_missing_column_is_refused() -> bool:
    """Dropping something that is not there, which is usually a typo."""
    try:
        drop(_table(10), ["nothing"])
    except UnknownColumn:
        return True
    return False


def a_rename_that_collides_is_refused() -> bool:
    """Renaming one column onto another's name."""
    try:
        rename(_table(10), {"amount": "id"})
    except SchemaError:
        return True
    return False


def renaming_a_missing_column_is_refused() -> bool:
    """Renaming something that is not there."""
    try:
        rename(_table(10), {"nothing": "something"})
    except UnknownColumn:
        return True
    return False


def repeated_computed_names_are_refused() -> bool:
    """Two derived columns with the same name."""
    one = Computed("v", Arithmetic("+", column("id"), literal(1)))
    try:
        compute(_table(10), [one, one])
    except SchemaError:
        return True
    return False


def compare_the_projections(rows: int = 20000) -> list[dict]:
    """Every form of projection and what it costs, which is the module in one table."""
    batch = _table(rows)
    computed = Computed("doubled", Arithmetic("+", column("amount"), column("amount")))
    out = []
    for name, action in (
        ("narrow", lambda meter: narrow(batch, ["id", "amount"], meter=meter)),
        ("drop", lambda meter: drop(batch, ["region"], meter=meter)),
        ("rename", lambda meter: rename(batch, {"amount": "value"}, meter=meter)),
        ("compute", lambda meter: compute(batch, [computed], meter=meter)),
    ):
        meter = Meter()
        produced = action(meter)
        out.append(
            {
                "projection": name,
                "columns": produced.width,
                "values_touched": meter.values_touched,
                "rows_materialised": meter.rows_materialised,
            }
        )
    return out


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "narrowing_is_free": narrowing_costs_nothing_measurable()["it_touched_nothing"],
        "computing_is_not": computing_a_column_is_not_free()["it_touched_the_rows"],
        "they_move_in_opposite_directions": the_two_directions_are_opposite()[
            "they_are_opposite"
        ],
        "types_agree": a_computed_column_gets_the_right_type()["they_all_agree"],
    }
