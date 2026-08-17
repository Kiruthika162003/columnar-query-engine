from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, boolean_column, floating_column, integer_column
from cqe.columns.array import string_column as make_string
from cqe.errors import DataError, TypeMismatch
from cqe.exec.batch import Batch
from cqe.types.schema import (
    BOOLEAN,
    DATE,
    FLOATING,
    INTEGER,
    LOGICAL_TYPES,
    PHYSICAL,
    STRING,
    Field,
)

# Converting a column from one type to another, and refusing when that would lose something.
#
# Three rules, and the whole module is their consequences.
#
# A widening conversion is allowed silently. An integer fits in a float for values a query
# realistically holds, a boolean fits in an integer, and nothing is lost.
#
# A narrowing conversion is allowed and reports what it lost. A float to an integer truncates
# and a large integer to a small one wraps, and both are legitimate operations that a caller may
# want and neither should happen without the caller being told.
#
# A meaningless conversion is refused. A string to a float is not a narrowing, it is a parse,
# and it fails on data rather than on shape: the same conversion succeeds on one batch and
# raises on the next. That belongs behind an explicit call with an explicit failure policy
# rather than behind the same function that widens an integer.
#
# The interesting measurement here is the last one: how much a float loses when it goes through
# an integer and back, which is the thing a schema change does to a column and is invisible
# until somebody sums it.

# Which conversions lose nothing. Read as: from this type, to any of these, silently.
WIDENING = {
    BOOLEAN: (BOOLEAN, INTEGER, FLOATING),
    INTEGER: (INTEGER, FLOATING),
    FLOATING: (FLOATING,),
    STRING: (STRING,),
    DATE: (DATE, INTEGER),
}

# Which conversions lose something and are still allowed, with what they lose.
NARROWING = {
    (FLOATING, INTEGER): "the fractional part",
    (INTEGER, BOOLEAN): "every value except zero and one",
    (FLOATING, BOOLEAN): "every value except zero and one",
    (INTEGER, DATE): "nothing, unless the number is not a day",
}

# The largest integer that survives a round trip through a double, which is two to the fifty
# three. Above it, integers and floats stop agreeing and a conversion that looks lossless is
# not.
EXACT_INTEGER = 2**53


@dataclass(frozen=True)
class Conversion:
    """One cast: what it produced and what it cost."""

    column: Column
    source: str
    target: str
    changed: int
    nulls_added: int = 0

    @property
    def lossless(self) -> bool:
        """Whether every value came through unchanged."""
        return self.changed == 0 and self.nulls_added == 0

    @property
    def kind(self) -> str:
        """Widening, narrowing or refused, which is what the caller asked about."""
        if self.target in WIDENING.get(self.source, ()):
            return "widening"
        if (self.source, self.target) in NARROWING:
            return "narrowing"
        return "parsed"

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "from": self.source,
            "to": self.target,
            "kind": self.kind,
            "rows": len(self.column),
            "changed": self.changed,
            "nulls_added": self.nulls_added,
            "lossless": self.lossless,
        }


def can_widen(source: str, target: str) -> bool:
    """Whether a conversion loses nothing."""
    return target in WIDENING.get(source, ())


def can_narrow(source: str, target: str) -> bool:
    """Whether a conversion loses something and is still allowed."""
    return (source, target) in NARROWING


def what_it_loses(source: str, target: str) -> str:
    """What a narrowing conversion gives up, in words."""
    if can_widen(source, target):
        return "nothing"
    return NARROWING.get((source, target), "it is not a conversion")


def cast(column: Column, target: str, meter=None) -> Conversion:
    """One column converted, with the loss counted rather than hidden.

    The count is the point. A conversion that silently changed a tenth of the values would be
    indistinguishable from one that changed none, and the failure appears in an aggregate over
    the converted column, where it looks like the data was always like that.
    """
    if target not in LOGICAL_TYPES:
        raise TypeMismatch(f"{target} is not a type; try one of {sorted(LOGICAL_TYPES)}")
    source = column.field.logical
    if source == target:
        return Conversion(column=column, source=source, target=target, changed=0)
    if not can_widen(source, target) and not can_narrow(source, target):
        raise TypeMismatch(
            f"{source} does not convert to {target}; parse it instead if that is what you mean"
        )
    if STRING in (source, target):
        raise TypeMismatch(f"{source} to {target} is a parse rather than a conversion")
    values = column.values.astype(PHYSICAL[target])
    changed = _differences(column.values, values, source)
    if meter is not None:
        meter.touch(len(column), f"cast_{source}_{target}")
    return Conversion(
        column=Column(
            field=Field(name=column.field.name, logical=target, nullable=column.field.nullable),
            values=values,
            valid=column.valid,
        ),
        source=source,
        target=target,
        changed=changed,
    )


def _differences(before: np.ndarray, after: np.ndarray, source: str) -> int:
    """How many values the conversion changed.

    Compared by converting back rather than by reasoning about the types, which catches the case
    nobody thinks of: an integer above two to the fifty three does not survive a round trip
    through a double, so a widening conversion that is lossless by the table is not lossless in
    fact.

    Compared in the source type, exactly. The first version compared with numpy's isclose when
    either side was floating, which converts both arrays to double before comparing and so lost
    exactly the precision the check was looking for: two integers that differ above two to the
    fifty three become the same double and the difference reports zero. A check that destroys
    the difference it is measuring reports success every time.
    """
    if not len(before):
        return 0
    back = after.astype(PHYSICAL[source])
    return int(np.count_nonzero(back != before))


def parse(column: Column, target: str, on_error: str = "refuse", meter=None) -> Conversion:
    """A string column read as another type, which can fail on the data.

    Three policies, because there is no right answer. Refuse, which is what a schema check
    wants. Null, which is what a load from a messy file wants. Skip, which drops the rows and is
    what nobody wants but somebody always asks for, and is worth having spelled out so it is a
    choice rather than an accident.
    """
    if column.field.logical != STRING:
        raise TypeMismatch(f"{column.field.logical} is not text; use cast")
    if on_error not in ("refuse", "null", "skip"):
        raise TypeMismatch(f"{on_error} is not a policy; try refuse, null or skip")
    values = column.to_list()
    made: list = []
    failures = 0
    valid: list[bool] = []
    for one in values:
        parsed, ok = _parse_one(one, target)
        if not ok:
            failures += 1
            if on_error == "refuse":
                raise DataError(f"{one!r} is not a {target}")
        made.append(parsed if ok else _zero(target))
        valid.append(ok and one is not None)
    if meter is not None:
        meter.touch(len(column), f"parse_{target}")
    built = _column_of(column.field.name, target, made, valid)
    if on_error == "skip" and failures:
        keep = np.array(valid)
        built = built.mask(keep)
    return Conversion(
        column=built,
        source=STRING,
        target=target,
        changed=failures,
        nulls_added=failures if on_error == "null" else 0,
    )


def _parse_one(value, target: str) -> tuple[object, bool]:
    """One string as one value, and whether it worked."""
    if value is None:
        return _zero(target), False
    try:
        if target == INTEGER:
            return int(value), True
        if target == FLOATING:
            return float(value), True
        if target == BOOLEAN:
            if str(value).lower() in ("true", "1", "yes"):
                return True, True
            if str(value).lower() in ("false", "0", "no"):
                return False, True
            return False, False
        if target == DATE:
            return int(value), True
    except (TypeError, ValueError):
        return _zero(target), False
    return _zero(target), False


def _zero(target: str):
    """The value a failed parse leaves behind, which is never read."""
    return False if target == BOOLEAN else 0


def _column_of(name: str, target: str, values: Sequence, valid: Sequence[bool]) -> Column:
    """A column of the parsed values, with a mask where the parse failed."""
    mask = np.array(valid, dtype=bool)
    if target in (INTEGER, DATE):
        made = integer_column(name, [int(one) for one in values])
    elif target == FLOATING:
        made = floating_column(name, [float(one) for one in values])
    else:
        made = boolean_column(name, [bool(one) for one in values])
    return Column(
        field=Field(name=name, logical=target, nullable=not mask.all()),
        values=made.values,
        valid=None if mask.all() else mask,
    )


def cast_batch(batch: Batch, targets: dict[str, str], meter=None) -> Batch:
    """Several columns converted at once, which is what a schema change is."""
    made = []
    for one in batch.columns:
        target = targets.get(one.field.name)
        made.append(cast(one, target, meter=meter).column if target else one)
    return Batch.from_columns(made)


def a_widening_conversion_loses_nothing(rows: int = 10000) -> dict:
    """Integer to float, which is the conversion that happens most and costs least.

    Lossless for every value a query realistically holds, which is what the table says. The
    measurement below is about the values it does not hold.
    """
    state = np.random.default_rng(31)
    column = integer_column("v", state.integers(-1_000_000, 1_000_000, rows))
    made = cast(column, FLOATING)
    return {
        **made.as_dict(),
        "it_is_widening": made.kind == "widening",
        "and_it_lost_nothing": made.lossless,
        "the_values_match": bool(
            np.allclose(made.column.values, column.values.astype(np.float64))
        ),
    }


def a_large_integer_does_not_survive_a_float(rows: int = 1000) -> dict:
    """And the values it does not hold, which the widening table calls lossless and is not.

    Above two to the fifty three a double cannot represent every integer, so the conversion
    changes values while every rule in the table says it is safe. The count is what makes this
    visible: the same function reports zero changes on ordinary integers and a change on nearly
    every large one.
    """
    state = np.random.default_rng(37)
    small = integer_column("v", state.integers(0, 1_000_000, rows))
    large = integer_column(
        "v", EXACT_INTEGER + state.integers(1, 1_000_000, rows).astype(np.int64)
    )
    return {
        "boundary": EXACT_INTEGER,
        "small_changed": cast(small, FLOATING).changed,
        "large_changed": cast(large, FLOATING).changed,
        "the_small_ones_are_exact": cast(small, FLOATING).lossless,
        "and_the_large_ones_are_not": not cast(large, FLOATING).lossless,
        "the_share_that_changed": round(cast(large, FLOATING).changed / rows, 3),
    }


def a_narrowing_conversion_reports_what_it_lost(rows: int = 10000) -> dict:
    """Float to integer, which truncates, and says how many values it truncated.

    The number is the whole reason the function returns a Conversion rather than a Column. A
    cast that quietly changed nine values in ten would look exactly like one that changed none.
    """
    state = np.random.default_rng(41)
    column = floating_column("v", state.normal(100, 30, rows))
    made = cast(column, INTEGER)
    whole = floating_column("v", state.integers(0, 1000, rows).astype(np.float64))
    return {
        **made.as_dict(),
        "it_is_narrowing": made.kind == "narrowing",
        "it_lost_something": not made.lossless,
        "what_it_loses": what_it_loses(FLOATING, INTEGER),
        "nearly_every_value_changed": made.changed > rows * 0.99,
        "and_a_whole_numbered_column_loses_nothing": cast(whole, INTEGER).lossless,
    }


def a_round_trip_through_an_integer_is_not_the_original(rows: int = 10000) -> dict:
    """A float column cast to integer and back, which is what a schema change does to it.

    The measurement worth reading before anybody changes a column type. The values come back
    looking like data and the total is different, and nothing anywhere says so unless the counts
    are kept.
    """
    state = np.random.default_rng(43)
    column = floating_column("v", state.normal(100, 30, rows))
    there = cast(column, INTEGER)
    back = cast(there.column, FLOATING)
    original = float(np.sum(column.values))
    after = float(np.sum(back.column.values))
    return {
        "rows": rows,
        "changed_going": there.changed,
        "changed_coming_back": back.changed,
        "original_total": round(original, 2),
        "final_total": round(after, 2),
        "difference": round(abs(original - after), 2),
        "the_totals_differ": abs(original - after) > 1,
        "and_the_second_cast_reported_nothing": back.lossless,
    }


def a_boolean_widens_to_an_integer(rows: int = 1000) -> dict:
    """True and false as one and zero, which is lossless in both directions.

    The only conversion in the table that is lossless both ways, because the integers a boolean
    produces are exactly the integers a boolean accepts.
    """
    state = np.random.default_rng(47)
    column = boolean_column("v", state.random(rows) < 0.5)
    widened = cast(column, INTEGER)
    back = cast(widened.column, BOOLEAN)
    return {
        "widening_lost_nothing": widened.lossless,
        "and_so_did_coming_back": back.lossless,
        "the_values_are_zero_and_one": set(widened.column.to_list()) <= {0, 1},
        "and_they_round_trip": back.column.to_list() == column.to_list(),
    }


def an_integer_narrows_to_a_boolean_badly(rows: int = 1000) -> dict:
    """Every value except zero and one changes, which is what the table says it loses.

    A two becomes true and comes back as one, so the count is every row holding something other
    than zero or one. That is most of a real column and is why the conversion is in the
    narrowing table rather than the widening one.
    """
    state = np.random.default_rng(53)
    column = integer_column("v", state.integers(0, 10, rows))
    made = cast(column, BOOLEAN)
    binary = integer_column("v", state.integers(0, 2, rows))
    return {
        **made.as_dict(),
        "what_it_loses": what_it_loses(INTEGER, BOOLEAN),
        "most_values_changed": made.changed > rows * 0.5,
        "and_a_binary_column_loses_nothing": cast(binary, BOOLEAN).lossless,
    }


def a_string_is_never_cast() -> dict:
    """Text does not convert, it parses, and the refusal says so.

    The distinction the module is built around. A cast is a function of the type and a parse is
    a function of the data, so the same call succeeding on one batch and raising on the next
    means they cannot be the same function.
    """
    column = make_string("v", ["1", "2", "3"])
    caught = ""
    try:
        cast(column, INTEGER)
    except TypeMismatch as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_says_to_parse": "parse" in caught,
        "and_parsing_works": parse(column, INTEGER).column.to_list() == [1, 2, 3],
    }


def parsing_refuses_by_default() -> dict:
    """A column with one bad value, which stops the whole parse.

    The default because it is the only policy that cannot lose data quietly. The other two are
    available and have to be asked for.
    """
    column = make_string("v", ["1", "2", "not a number", "4"])
    caught = ""
    try:
        parse(column, INTEGER)
    except DataError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_the_value": "not a number" in caught,
    }


def parsing_can_null_the_failures() -> dict:
    """The same column with the null policy, which keeps the rows and marks them.

    What a load from a messy file wants: the good rows are usable and the bad ones are visibly
    missing rather than silently zero.
    """
    column = make_string("v", ["1", "2", "not a number", "4"])
    made = parse(column, INTEGER, on_error="null")
    return {
        **made.as_dict(),
        "rows_kept": len(made.column),
        "it_kept_every_row": len(made.column) == 4,
        "the_bad_one_is_null": made.column.to_list()[2] is None,
        "and_the_others_survived": [one for one in made.column.to_list() if one is not None]
        == [1, 2, 4],
    }


def parsing_can_skip_the_failures() -> dict:
    """And the third policy, which drops the rows.

    Worth having spelled out because it is the one that loses data, and a caller who asks for it
    by name has decided to rather than discovered it later.
    """
    column = make_string("v", ["1", "2", "not a number", "4"])
    made = parse(column, INTEGER, on_error="skip")
    return {
        **made.as_dict(),
        "rows_kept": len(made.column),
        "it_dropped_the_bad_one": len(made.column) == 3,
        "and_kept_the_rest": made.column.to_list() == [1, 2, 4],
    }


def parsing_a_boolean_accepts_several_spellings() -> dict:
    """True, yes and one all parse; anything else does not.

    A small set on purpose. Every spelling accepted is a spelling that stops being an error, and
    a parser that accepts everything cannot tell a boolean column from a broken one.
    """
    column = make_string("v", ["true", "TRUE", "yes", "1", "false", "no", "0"])
    made = parse(column, BOOLEAN)
    caught = ""
    try:
        parse(make_string("v", ["maybe"]), BOOLEAN)
    except DataError as problem:
        caught = str(problem)
    return {
        "values": made.column.to_list(),
        "it_parsed_them_all": made.lossless,
        "the_trues": sum(1 for one in made.column.to_list() if one),
        "and_maybe_is_refused": bool(caught),
    }


def parsing_a_null_is_a_failure() -> dict:
    """A null in a string column is not a parseable value, and the policy decides.

    A null string is missing text rather than the text of a missing number, so the parse cannot
    produce a value and the policy applies to it exactly as to a malformed one.
    """
    made = make_string("v", ["1", "2"])
    column = Column(
        field=made.field,
        values=made.values,
        valid=np.array([True, False]),
        dictionary=made.dictionary,
    )
    parsed = parse(column, INTEGER, on_error="null")
    return {
        "rows": len(column),
        "parsed": parsed.column.to_list(),
        "the_null_stayed_null": parsed.column.to_list()[1] is None,
        "and_the_value_survived": parsed.column.to_list()[0] == 1,
    }


def a_nonsense_conversion_is_refused() -> dict:
    """A date to a boolean, which is in neither table."""
    column = integer_column("v", [1, 2, 3])
    dated = Column(
        field=Field(name="v", logical=DATE, nullable=False),
        values=column.values.astype(PHYSICAL[DATE]),
    )
    caught = ""
    try:
        cast(dated, BOOLEAN)
    except TypeMismatch as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_both_types": DATE in caught and BOOLEAN in caught,
    }


def a_conversion_keeps_the_nulls(rows: int = 1000) -> dict:
    """A cast moves the validity mask across unchanged.

    Which sounds obvious and is the thing a conversion written on the values array alone gets
    wrong: the values come through and the mask does not, and every null becomes whatever the
    zero of the target type is.
    """
    state = np.random.default_rng(59)
    values = state.integers(0, 100, rows)
    made = integer_column("v", values)
    valid = state.random(rows) > 0.3
    column = Column(field=made.field, values=values, valid=valid)
    converted = cast(column, FLOATING)
    return {
        "nulls_before": int((~valid).sum()),
        "nulls_after": int((~converted.column.valid).sum()),
        "they_match": int((~valid).sum()) == int((~converted.column.valid).sum()),
        "and_the_mask_is_the_same": bool(np.array_equal(valid, converted.column.valid)),
    }


def casting_a_batch_changes_several_columns(rows: int = 1000) -> dict:
    """A schema change over a whole table, which is what this is for."""
    state = np.random.default_rng(61)
    batch = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 20, rows)),
            floating_column("amount", state.normal(100, 20, rows)),
        ]
    )
    made = cast_batch(batch, {"shop": FLOATING, "amount": INTEGER})
    return {
        "before": [one.logical for one in batch.schema.fields],
        "after": [one.logical for one in made.schema.fields],
        "the_untouched_one_is_the_same": made.column("id") is batch.column("id"),
        "the_widened_one_is_floating": made.column("shop").field.logical == FLOATING,
        "the_narrowed_one_is_integer": made.column("amount").field.logical == INTEGER,
    }


def an_unknown_type_is_refused() -> bool:
    """A target type that does not exist, with the list in the message."""
    try:
        cast(integer_column("v", [1]), "decimal")
    except TypeMismatch:
        return True
    return False


def an_unknown_policy_is_refused() -> bool:
    """A parse policy that does not exist."""
    try:
        parse(make_string("v", ["1"]), INTEGER, on_error="pretend")
    except TypeMismatch:
        return True
    return False


def parsing_a_number_column_is_refused() -> bool:
    """Parsing something that is not text, which is a cast."""
    try:
        parse(integer_column("v", [1]), FLOATING)
    except TypeMismatch:
        return True
    return False


def compare_the_conversions() -> list[dict]:
    """Every conversion in both tables, and what each one gives up."""
    out = []
    for source in LOGICAL_TYPES:
        for target in LOGICAL_TYPES:
            if source == target:
                continue
            if can_widen(source, target):
                kind = "widening"
            elif can_narrow(source, target):
                kind = "narrowing"
            else:
                continue
            out.append(
                {
                    "from": source,
                    "to": target,
                    "kind": kind,
                    "loses": what_it_loses(source, target),
                }
            )
    return out


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "conversions": len(compare_the_conversions()),
        "widening_is_lossless": a_widening_conversion_loses_nothing()["and_it_lost_nothing"],
        "large_integers_are_not": a_large_integer_does_not_survive_a_float()[
            "and_the_large_ones_are_not"
        ],
        "narrowing_reports_its_loss": a_narrowing_conversion_reports_what_it_lost()[
            "it_lost_something"
        ],
        "a_round_trip_changes_the_total": a_round_trip_through_an_integer_is_not_the_original()[
            "the_totals_differ"
        ],
        "text_is_parsed_not_cast": a_string_is_never_cast()["it_says_to_parse"],
    }
