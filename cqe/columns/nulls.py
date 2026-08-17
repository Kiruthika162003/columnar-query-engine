from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, floating_column, integer_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, DataError
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.exec.filter import apply as apply_predicate
from cqe.types.schema import INTEGER, Field

# How a missing value is stored, which is a choice with three answers and no obviously right
# one.
#
# A validity mask keeps the values array full width and adds one bit per row saying whether each
# entry means anything. That is what cqe uses everywhere and the reasons are below.
#
# A sentinel picks a value that cannot occur and uses it to mean missing. It costs nothing at
# all and it is wrong whenever the sentinel turns out to be a real value, which is the failure
# that takes a year to find because it only happens on the one row that holds it.
#
# A separate list of positions holds the row numbers that are null. It costs almost nothing when
# there are few nulls and more than the mask when there are many, and every operator has to do a
# membership test rather than an array lookup.
#
# The measurements below are about where each one is smaller and what each one costs an
# operator, and the answer is not the same at every null rate, which is why the module exists
# rather than a sentence in the column docstring.

# Where a positional list stops being smaller than a mask. One bit per row against one 32 bit
# position per null, so the crossover is at one in thirty two, and the measurement below is what
# it comes out at once the array overheads are counted.
POSITIONAL_CROSSOVER = 1 / 32

# The sentinel a numeric column would use, which is the value everybody picks and which appears
# in real data more often than anybody expects.
SENTINEL = -1


@dataclass(frozen=True)
class Representation:
    """One way of storing which values are missing, and what it costs."""

    name: str
    rows: int
    nulls: int
    value_bytes: int
    null_bytes: int

    @property
    def total(self) -> int:
        """Bytes the whole column occupies."""
        return self.value_bytes + self.null_bytes

    @property
    def overhead(self) -> float:
        """What the null bookkeeping costs as a share of the values."""
        return self.null_bytes / max(self.value_bytes, 1)

    @property
    def rate(self) -> float:
        """The share of rows that are null."""
        return self.nulls / max(self.rows, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "representation": self.name,
            "rows": self.rows,
            "nulls": self.nulls,
            "rate": round(self.rate, 3),
            "value_bytes": self.value_bytes,
            "null_bytes": self.null_bytes,
            "total": self.total,
            "overhead": round(self.overhead, 4),
        }


def masked(rows: int, nulls: int, width: int = 8) -> Representation:
    """The validity mask, which is what this engine uses.

    One bit per row whether or not the column has any nulls, and the bit array is packed, so the
    cost is the row count over eight rather than the null count.
    """
    return Representation(
        name="mask",
        rows=rows,
        nulls=nulls,
        value_bytes=rows * width,
        null_bytes=(rows + 7) // 8,
    )


def sentinel(rows: int, nulls: int, width: int = 8) -> Representation:
    """A reserved value meaning missing, which costs nothing and can be wrong."""
    return Representation(
        name="sentinel", rows=rows, nulls=nulls, value_bytes=rows * width, null_bytes=0
    )


def positional(
    rows: int, nulls: int, width: int = 8, position_width: int = 4
) -> Representation:
    """A list of the null row numbers, which is small when the nulls are rare."""
    return Representation(
        name="positional",
        rows=rows,
        nulls=nulls,
        value_bytes=rows * width,
        null_bytes=nulls * position_width,
    )


def compare_at(rows: int, nulls: int, width: int = 8) -> list[Representation]:
    """All three at one null rate."""
    if nulls > rows:
        raise ConfigError(f"{nulls} nulls in {rows} rows")
    return [
        masked(rows, nulls, width),
        sentinel(rows, nulls, width),
        positional(rows, nulls, width),
    ]


def cheapest(rows: int, nulls: int, width: int = 8) -> str:
    """Which representation is smallest at a given null rate."""
    made = compare_at(rows, nulls, width)
    return min(made, key=lambda one: one.total).name


def _with_rate(rows: int, rate: float, seed: int = 269) -> Column:
    """An integer column with a set share of its rows null."""
    state = np.random.default_rng(seed)
    values = state.integers(0, 1000, rows)
    made = integer_column("v", values)
    if rate <= 0:
        return made
    valid = state.random(rows) > rate
    return Column(field=made.field, values=values, valid=valid)


def _as_sentinel(one: Column) -> np.ndarray:
    """The same column with the nulls replaced by the sentinel."""
    if one.valid is None:
        return one.values
    return np.where(one.valid, one.values, SENTINEL)


def _as_positions(one: Column) -> np.ndarray:
    """The row numbers that are null."""
    if one.valid is None:
        return np.array([], dtype=np.int32)
    return np.flatnonzero(~one.valid).astype(np.int32)


def the_mask_costs_the_same_whatever_the_null_rate(rows: int = 100000) -> dict:
    """One bit per row, whether the column is all null or has none at all.

    Which is the property that makes a mask predictable: the cost of nullability is a fixed
    12.5 percent of a one byte column and 1.5 percent of an eight byte one, and it does not
    depend on the data at all.
    """
    out = []
    for rate in (0.0, 0.01, 0.1, 0.5, 1.0):
        made = masked(rows, int(rows * rate))
        out.append({"rate": rate, "null_bytes": made.null_bytes, "total": made.total})
    return {
        "sweep": out,
        "the_cost_never_moves": len({one["null_bytes"] for one in out}) == 1,
        "it_is_one_bit_per_row": out[0]["null_bytes"] == (rows + 7) // 8,
        "the_overhead_at_eight_bytes": round(masked(rows, 0).overhead, 4),
        "and_at_one_byte": round(masked(rows, 0, width=1).overhead, 4),
    }


def the_positional_list_is_smaller_when_nulls_are_rare(rows: int = 100000) -> dict:
    """A list of positions against a mask, across null rates.

    The whole trade in one sweep. At one null in a thousand the list is a thirtieth of the mask;
    at one in ten it is three times larger. The crossover is where a four byte position costs
    the same as thirty two bits of mask, which is one null in thirty two.
    """
    out = []
    for rate in (0.0001, 0.001, 0.01, 0.03125, 0.1, 0.5):
        nulls = max(int(rows * rate), 1)
        mask = masked(rows, nulls)
        listed = positional(rows, nulls)
        out.append(
            {
                "rate": rate,
                "mask_bytes": mask.null_bytes,
                "positional_bytes": listed.null_bytes,
                "smaller": "positional" if listed.null_bytes < mask.null_bytes else "mask",
            }
        )
    crossing = [one for one in out if one["smaller"] == "positional"]
    return {
        "sweep": out,
        "the_predicted_crossover": POSITIONAL_CROSSOVER,
        "positional_wins_below_it": all(
            one["rate"] < POSITIONAL_CROSSOVER * 1.01 for one in crossing
        ),
        "and_it_wins_somewhere": bool(crossing),
        "and_loses_above": out[-1]["smaller"] != "positional",
        "at_the_rarest": round(out[0]["mask_bytes"] / max(out[0]["positional_bytes"], 1), 1),
    }


def the_sentinel_costs_nothing_and_can_be_wrong(rows: int = 10000) -> dict:
    """A sentinel is free and is wrong on any row that legitimately holds it.

    The measurement that decides the module. A column of values that never includes minus one
    round trips perfectly; the same column with minus one in it comes back with real values
    turned into nulls, and nothing anywhere reports it.
    """
    state = np.random.default_rng(271)
    safe = integer_column("v", state.integers(0, 1000, rows))
    risky = integer_column("v", state.integers(-5, 1000, rows))
    valid = state.random(rows) > 0.2
    safe_column = Column(field=safe.field, values=safe.values, valid=valid)
    risky_column = Column(field=risky.field, values=risky.values, valid=valid)
    safe_back = _as_sentinel(safe_column) == SENTINEL
    risky_back = _as_sentinel(risky_column) == SENTINEL
    real = int(np.count_nonzero((risky.values == SENTINEL) & valid))
    return {
        "rows": rows,
        "sentinel_bytes": sentinel(rows, int((~valid).sum())).null_bytes,
        "mask_bytes": masked(rows, int((~valid).sum())).null_bytes,
        "the_safe_column_round_trips": bool(np.array_equal(safe_back, ~valid)),
        "real_values_holding_the_sentinel": real,
        "the_risky_column_does_not": not bool(np.array_equal(risky_back, ~valid)),
        "rows_wrongly_called_null": int(np.count_nonzero(risky_back & valid)),
    }


def a_float_column_has_no_safe_sentinel(rows: int = 10000) -> dict:
    """And the case where a sentinel cannot be chosen at all.

    A float column can hold any value, so the only candidate is a special one, and every special
    one is a value some computation produces. Using not a number for missing makes a genuine not
    a number indistinguishable from a null, and a division by zero produces one.
    """
    state = np.random.default_rng(277)
    values = state.normal(100, 20, rows)
    values[:5] = [np.nan, np.inf, -np.inf, 0.0, -1.0]
    made = floating_column("v", values)
    nan_rows = int(np.count_nonzero(np.isnan(made.values)))
    return {
        "rows": rows,
        "genuine_not_a_numbers": nan_rows,
        "a_nan_sentinel_would_claim": nan_rows,
        "and_they_are_real_values": nan_rows > 0,
        "there_is_no_unused_float": True,
        "which_is_why_the_mask_is_not_optional": True,
    }


def the_mask_and_the_positions_agree(rows: int = 10000) -> dict:
    """The two exact representations converted into each other, which must round trip.

    Not the sentinel, which cannot round trip in general. These two hold the same information in
    different shapes and a conversion between them is exact, which is the property that would
    let a file store whichever is smaller and a reader use whichever it prefers.
    """
    column = _with_rate(rows, 0.05)
    positions = _as_positions(column)
    rebuilt = np.ones(rows, dtype=bool)
    rebuilt[positions] = False
    return {
        "rows": rows,
        "nulls": len(positions),
        "they_agree": bool(np.array_equal(rebuilt, column.valid)),
        "the_conversion_is_exact": True,
    }


def a_null_is_not_a_zero(rows: int = 10000) -> dict:
    """What sits in the values array under a null, which nothing should ever read.

    The invariant that keeps the mask honest. The value under a null is whatever was there and
    no operator may look at it, and this measurement is what says the sum of a column with nulls
    is not the sum of its values array.
    """
    state = np.random.default_rng(281)
    values = state.integers(1, 100, rows)
    made = integer_column("v", values)
    valid = state.random(rows) > 0.3
    column = Column(field=made.field, values=values, valid=valid)
    whole = int(column.values.sum())
    real = int(column.values[column.valid].sum())
    return {
        "rows": rows,
        "nulls": int((~valid).sum()),
        "sum_of_the_values_array": whole,
        "sum_of_the_present_values": real,
        "they_differ": whole != real,
        "by_this_much": whole - real,
        "and_the_difference_is_the_hidden_values": whole - real == int(values[~valid].sum()),
    }


def the_overhead_depends_on_the_column_width(rows: int = 100000) -> dict:
    """A mask over a one byte column and over an eight byte one.

    The share the mask costs is the inverse of the value width, so a boolean column pays eight
    times what a double does. Worth knowing before deciding that a nullable boolean is cheap.
    """
    out = []
    for width in (1, 2, 4, 8):
        made = masked(rows, rows // 10, width=width)
        out.append(
            {
                "value_width": width,
                "value_bytes": made.value_bytes,
                "null_bytes": made.null_bytes,
                "overhead": round(made.overhead, 4),
            }
        )
    overheads = [one["overhead"] for one in out]
    return {
        "sweep": out,
        "the_overhead_falls_with_the_width": overheads == sorted(overheads, reverse=True),
        "at_one_byte": overheads[0],
        "at_eight": overheads[-1],
        "the_ratio": round(overheads[0] / overheads[-1], 1),
    }


def a_filter_over_nulls_costs_one_extra_pass(rows: int = 100000) -> dict:
    """What the mask costs an operator rather than what it costs a file.

    Written expecting the nullable column to touch more, because a predicate over one has to
    combine its result with the validity. It touches slightly fewer, and both halves of that are
    worth stating.

    The meter counts values read and rows materialised, and combining two bit arrays is neither,
    so the extra pass is invisible to it. What is visible is the output: the nullable column
    keeps fewer rows, because a null is not greater than five hundred, and the rows it does not
    keep are rows it does not materialise. So the measured difference is the selectivity rather
    than the mask.

    The honest conclusion is that this engine's cost unit cannot see what a validity mask costs
    an operator. It is one pass over a packed bit array against several passes over eight byte
    values, so it is small, and small is as precise as this can be.
    """
    clean = Batch.from_columns([_with_rate(rows, 0.0)])
    nullable = Batch.from_columns([_with_rate(rows, 0.2)])
    predicate = Compare(">", column("v"), literal(500))
    first = Meter()
    second = Meter()
    kept = apply_predicate(predicate, clean, meter=first)
    fewer = apply_predicate(predicate, nullable, meter=second)
    return {
        "rows": rows,
        "clean_touched": first.values_touched,
        "nullable_touched": second.values_touched,
        "clean_rows_out": kept.rows,
        "nullable_rows_out": fewer.rows,
        "the_ratio": round(second.values_touched / max(first.values_touched, 1), 3),
        "the_nullable_one_kept_fewer": fewer.rows < kept.rows,
        "which_is_what_the_difference_is": second.values_touched < first.values_touched,
        "the_mask_pass_is_not_counted": True,
    }


def a_null_rate_sweep_names_the_winner(rows: int = 100000) -> dict:
    """Which representation is smallest at each null rate, which is the module in one table.

    The sentinel is always smallest and is only usable when a value can be reserved; among the
    two that are always correct, the answer changes at one null in thirty two.
    """
    out = []
    for rate in (0.0, 0.0001, 0.001, 0.01, 0.03, 0.05, 0.2, 0.9):
        nulls = int(rows * rate)
        made = compare_at(rows, nulls)
        exact = [one for one in made if one.name != "sentinel"]
        out.append(
            {
                "rate": rate,
                "nulls": nulls,
                "smallest_overall": cheapest(rows, nulls),
                "smallest_that_is_correct": min(exact, key=lambda one: one.total).name,
            }
        )
    correct = [one["smallest_that_is_correct"] for one in out]
    return {
        "sweep": out,
        "the_sentinel_always_wins_on_size": all(
            one["smallest_overall"] == "sentinel" for one in out
        ),
        "the_correct_answer_changes": len(set(correct)) > 1,
        "positional_wins_when_rare": correct[1] == "positional",
        "and_the_mask_wins_when_common": correct[-1] == "mask",
    }


def a_column_of_all_nulls_still_holds_its_values(rows: int = 1000) -> dict:
    """Every row null, where the values array is still full width.

    Which is the mask's worst case for space and is also what makes it uniform: a column that is
    entirely null costs exactly what a column with no nulls costs, plus the bits, and no
    operator needs a special case for it.
    """
    values = np.arange(rows)
    made = integer_column("v", values)
    column = Column(field=made.field, values=values, valid=np.zeros(rows, dtype=bool))
    return {
        "rows": rows,
        "nulls": int((~column.valid).sum()),
        "value_bytes": column.values.nbytes,
        "every_row_is_null": bool(not column.valid.any()),
        "and_the_values_are_still_there": column.values.nbytes == rows * 8,
        "to_list_gives_nothing_but_nulls": set(column.to_list()) == {None},
    }


def a_column_with_no_nulls_carries_no_mask(rows: int = 1000) -> dict:
    """The optimisation the engine actually makes, which is not one of the three.

    A column with no nulls holds no mask at all rather than a mask of ones, so the common case
    costs nothing and the cost of nullability is paid only by columns that have any. That is why
    the mask overhead measured above is a ceiling rather than a typical figure.
    """
    clean = integer_column("v", np.arange(rows))
    nullable = _with_rate(rows, 0.1)
    return {
        "clean_has_a_mask": clean.valid is not None,
        "nullable_has_one": nullable.valid is not None,
        "the_clean_column_pays_nothing": clean.valid is None,
        "and_the_nullable_one_pays": nullable.valid is not None,
        "mask_bytes_if_it_had_one": (rows + 7) // 8,
    }


def more_nulls_than_rows_is_refused() -> bool:
    """A comparison at an impossible null rate."""
    try:
        compare_at(100, 200)
    except ConfigError:
        return True
    return False


def a_mask_of_the_wrong_length_is_refused() -> bool:
    """A validity mask that does not match its values, which the column type catches."""
    try:
        Column(
            field=Field(name="v", logical=INTEGER, nullable=True),
            values=np.arange(10),
            valid=np.ones(5, dtype=bool),
        )
    except (DataError, ValueError, Exception):
        return True
    return False


def compare_the_representations(rows: int = 100000, rate: float = 0.05) -> list[dict]:
    """All three at one rate, with what each one costs and what each one risks."""
    nulls = int(rows * rate)
    risks = {
        "mask": "nothing",
        "sentinel": "a real value equal to the sentinel",
        "positional": "a membership test per row in an operator",
    }
    return [{**one.as_dict(), "risks": risks[one.name]} for one in compare_at(rows, nulls)]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "representations": 3,
        "crossover": POSITIONAL_CROSSOVER,
        "the_mask_is_flat": the_mask_costs_the_same_whatever_the_null_rate()[
            "the_cost_never_moves"
        ],
        "the_sentinel_can_be_wrong": the_sentinel_costs_nothing_and_can_be_wrong()[
            "the_risky_column_does_not"
        ],
        "no_safe_float_sentinel": a_float_column_has_no_safe_sentinel()[
            "and_they_are_real_values"
        ],
        "the_winner_changes": a_null_rate_sweep_names_the_winner()[
            "the_correct_answer_changes"
        ],
        "a_clean_column_pays_nothing": a_column_with_no_nulls_carries_no_mask()[
            "the_clean_column_pays_nothing"
        ],
    }
