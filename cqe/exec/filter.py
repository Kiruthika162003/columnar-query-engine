from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.cost.meter import Meter
from cqe.errors import ConfigError
from cqe.exec.batch import Batch, mask_to_selection, selection_to_mask
from cqe.exec.expr import (
    And,
    Compare,
    Expr,
    all_of,
    column,
    conjuncts,
    evaluate_to_mask,
    literal,
)
from cqe.types.schema import BOOLEAN

# Filtering, and the two representations of a set of surviving rows.
#
# A boolean mask is one byte per row of the input. A selection vector is one integer per row of
# the output. Which is smaller depends entirely on the selectivity, and the crossover is where
# the selected fraction equals the ratio of the two widths, which for a byte mask against an
# int32 selection is one in four.
#
# That much is arithmetic. What is worth measuring is what each one costs the operator above it,
# because a filter almost never ends a plan. Taking rows by a selection vector is a gather,
# which numpy does in one call. Taking rows by a mask is also a gather, because numpy converts
# the mask to positions first, so the mask form pays the conversion on every downstream operator
# that uses it. The engine converts once, at the filter, and passes a selection vector onwards.
#
# The second subject is the order conjuncts are evaluated in. A predicate that is a and b can
# evaluate b on the rows a kept rather than on all of them, which is the vectorised form of
# short circuiting. The saving is the selectivity of a, and the cost is that b then runs on a
# gathered array rather than a contiguous one. The measurement below puts the crossover at a
# selectivity of 0.5: below it chaining wins, above it the gather costs more than the conjunct
# saves.
#
# The ordering matters more than the technique. The same two conjuncts with the selective one
# last cost 3.2 times what they cost with it first, against a best case saving from chaining at
# all of 0.43. Which is why stats/cardinality.py is the module that makes this one worth having.


@dataclass
class Selection:
    """A set of surviving rows, in whichever representation is cheaper."""

    positions: np.ndarray
    rows: int

    def __post_init__(self) -> None:
        if self.rows < 0:
            raise ConfigError(f"{self.rows} is not a row count")
        if self.positions.dtype.kind not in "iu":
            raise ConfigError(f"positions are integers, not {self.positions.dtype}")

    @property
    def kept(self) -> int:
        """How many rows survived."""
        return int(self.positions.shape[0])

    @property
    def selectivity(self) -> float:
        """The share of rows that survived."""
        if self.rows == 0:
            return 0.0
        return self.kept / self.rows

    @property
    def mask_bytes(self) -> int:
        """What the boolean mask form would occupy."""
        return self.rows

    @property
    def selection_bytes(self) -> int:
        """What the selection vector form occupies."""
        return self.kept * 4

    @property
    def cheaper_form(self) -> str:
        """Which representation is smaller for this selectivity."""
        return "selection" if self.selection_bytes < self.mask_bytes else "mask"

    def as_mask(self) -> np.ndarray:
        """The boolean form."""
        return selection_to_mask(self.positions, self.rows)

    def apply(self, batch: Batch, meter: Meter | None = None) -> Batch:
        """Gather the surviving rows out of a batch."""
        return batch.take(self.positions, meter=meter)

    def refine(self, keep: np.ndarray) -> Selection:
        """Narrow a selection by a mask over the rows it already holds.

        The operation conjunct chaining is built on. The mask is over the kept rows and not over
        the original ones, which is what makes the second conjunct cheaper and is also the one
        place an off by one turns into silently wrong rows rather than a crash.
        """
        if keep.dtype != np.bool_:
            raise ConfigError(f"a mask is boolean, not {keep.dtype}")
        if keep.shape[0] != self.kept:
            raise ConfigError(f"{keep.shape[0]} mask entries against {self.kept} kept rows")
        return Selection(positions=self.positions[keep], rows=self.rows)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rows": self.rows,
            "kept": self.kept,
            "selectivity": round(self.selectivity, 4),
            "cheaper_form": self.cheaper_form,
        }


def everything(rows: int) -> Selection:
    """A selection that keeps every row, which is what a plan starts with."""
    if rows < 0:
        raise ConfigError(f"{rows} is not a row count")
    return Selection(positions=np.arange(rows, dtype=np.int32), rows=rows)


def nothing(rows: int) -> Selection:
    """A selection that keeps no rows, which is what a pruned row group produces."""
    if rows < 0:
        raise ConfigError(f"{rows} is not a row count")
    return Selection(positions=np.array([], dtype=np.int32), rows=rows)


def from_mask(keep: np.ndarray) -> Selection:
    """Build a selection from a boolean mask."""
    return Selection(
        positions=mask_to_selection(keep).astype(np.int32),
        rows=int(keep.shape[0]),
    )


def evaluate(expression: Expr, batch: Batch, meter: Meter | None = None) -> Selection:
    """Evaluate a predicate over a whole batch and return the survivors.

    The straightforward form, which evaluates every conjunct on every row. Used as the reference
    the chained form below is measured against, and used in the plan whenever the selectivity is
    unknown or high.
    """
    keep = evaluate_to_mask(expression, batch, meter)
    return from_mask(keep)


def evaluate_chained(
    expression: Expr,
    batch: Batch,
    meter: Meter | None = None,
    order: Sequence[int] | None = None,
) -> Selection:
    """Evaluate conjuncts one at a time, each on the rows the previous ones kept.

    The vectorised form of short circuiting. Each conjunct after the first sees a gathered batch
    rather than the original, so it does less work when the conjuncts before it were selective
    and the same work when they were not.

    The order is a parameter rather than a decision, because choosing it is a cost model problem
    and belongs in plan/rules, and because leaving it here makes the measurement of how much the
    order matters possible at all.
    """
    parts = conjuncts(expression)
    if order is not None:
        if sorted(order) != list(range(len(parts))):
            raise ConfigError(f"{list(order)} is not an ordering of {len(parts)} conjuncts")
        parts = [parts[position] for position in order]
    current = everything(batch.rows)
    working = batch
    for position, part in enumerate(parts):
        keep = evaluate_to_mask(part, working, meter)
        current = current.refine(keep)
        if position + 1 < len(parts):
            working = current.apply(batch, meter)
    return current


def apply(expression: Expr, batch: Batch, meter: Meter | None = None) -> Batch:
    """Filter a batch, which is what an operator above actually wants."""
    return evaluate(expression, batch, meter).apply(batch, meter)


def _column(rows: int, span: int, seed: int) -> list[int]:
    """A column of integers over a span."""
    return np.random.default_rng(seed).integers(0, span, size=rows).tolist()


def _sample(rows: int = 100_000, columns: int = 4, seed: int = 0) -> Batch:
    """A batch wide enough that the cost of gathering it is visible."""
    if rows < 1 or columns < 1:
        raise ConfigError(f"{rows} rows of {columns} columns is not a batch")
    named = {
        f"c{position}": _column(rows, 1000, seed + position) for position in range(columns)
    }
    return Batch.of(**named)


def _threshold_for(selectivity: float) -> int:
    """The comparison bound that keeps roughly the given share of a uniform column."""
    if not 0.0 < selectivity <= 1.0:
        raise ConfigError(f"{selectivity} is not a selectivity")
    return int(1000 * selectivity)


def the_two_representations_cross_over_at_a_quarter(
    rows: int = 100_000,
    selectivities: Sequence[float] = (0.01, 0.1, 0.25, 0.5, 1.0),
) -> list[dict]:
    """Which of a mask and a selection vector is smaller, as the selectivity moves.

    A byte per input row against four bytes per output row, so the selection vector is smaller
    below a selectivity of a quarter and larger above it. Arithmetic rather than a discovery,
    and measured because the sizes are what a plan uses to decide which to carry between
    operators.
    """
    if not selectivities:
        raise ConfigError("there is nothing to sweep")
    batch = _sample(rows=rows, columns=1)
    out = []
    for share in selectivities:
        predicate = Compare("<", column("c0"), literal(_threshold_for(share)))
        selection = evaluate(predicate, batch)
        out.append(selection.as_dict())
    return out


def chaining_conjuncts_saves_the_selectivity(
    rows: int = 100_000,
    selectivity: float = 0.1,
) -> dict:
    """Evaluating the second conjunct only on the rows the first kept.

    The straightforward form evaluates both conjuncts on every row. The chained form evaluates
    the second on the survivors of the first, so the values touched fall by the selectivity of
    the first conjunct.

    Both are checked to produce the same rows, because a filter that is fast and wrong is the
    failure this arrangement exists to catch.
    """
    batch = _sample(rows=rows, columns=4)
    predicate = And(
        (
            Compare("<", column("c0"), literal(_threshold_for(selectivity))),
            Compare("<", column("c1"), literal(500)),
        )
    )
    plain_meter = Meter()
    plain = evaluate(predicate, batch, plain_meter)
    chained_meter = Meter()
    chained = evaluate_chained(predicate, batch, chained_meter)
    return {
        "rows": rows,
        "kept": plain.kept,
        "same_rows": bool(np.array_equal(plain.positions, chained.positions)),
        "plain_values": plain_meter.values_touched,
        "chained_values": chained_meter.values_touched,
        "ratio": round(chained_meter.values_touched / max(plain_meter.values_touched, 1), 4),
        "chaining_wins": chained_meter.values_touched < plain_meter.values_touched,
    }


def the_order_of_the_conjuncts_matters_more(
    rows: int = 100_000,
    selectivity: float = 0.05,
) -> dict:
    """The same two conjuncts in both orders, chained.

    Selective first, the second conjunct sees a twentieth of the rows. Selective last, it sees
    all of them and the first conjunct is the one that gets the saving, which is no saving at
    all because it is already cheap.

    This is the measurement that makes selectivity estimation worth doing. The technique buys
    something; getting the order right buys several times more.
    """
    batch = _sample(rows=rows, columns=4)
    selective = Compare("<", column("c0"), literal(_threshold_for(selectivity)))
    loose = Compare("<", column("c1"), literal(900))
    predicate = And((selective, loose))

    good_meter = Meter()
    good = evaluate_chained(predicate, batch, good_meter, order=(0, 1))
    bad_meter = Meter()
    bad = evaluate_chained(predicate, batch, bad_meter, order=(1, 0))
    return {
        "kept": good.kept,
        "same_rows": bool(np.array_equal(np.sort(good.positions), np.sort(bad.positions))),
        "selective_first": good_meter.values_touched,
        "selective_last": bad_meter.values_touched,
        "ratio": round(bad_meter.values_touched / max(good_meter.values_touched, 1), 3),
        "the_order_matters": bad_meter.values_touched > good_meter.values_touched,
    }


def chaining_costs_something_when_nothing_is_selective(rows: int = 100_000) -> dict:
    """Two conjuncts that each keep almost everything, where the gather is pure overhead.

    Chaining pays for a gather between the conjuncts and gets nothing back when the first one
    keeps nine rows in ten. This is the case a plan has to avoid, and it is why the chained form
    is a choice in this module rather than the only path.
    """
    batch = _sample(rows=rows, columns=4)
    predicate = And(
        (
            Compare("<", column("c0"), literal(900)),
            Compare("<", column("c1"), literal(900)),
        )
    )
    plain_meter = Meter()
    evaluate(predicate, batch, plain_meter)
    chained_meter = Meter()
    evaluate_chained(predicate, batch, chained_meter)
    return {
        "plain_values": plain_meter.values_touched,
        "chained_values": chained_meter.values_touched,
        "ratio": round(chained_meter.values_touched / max(plain_meter.values_touched, 1), 3),
        "chaining_loses": chained_meter.values_touched > plain_meter.values_touched,
    }


def the_crossover_selectivity(
    rows: int = 100_000,
    shares: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9),
) -> dict:
    """Where chaining stops paying, swept rather than argued.

    Below the crossover the second conjunct sees few enough rows that the gather is worth it;
    above it the gather costs more than the conjunct saves. The crossover is what a plan needs
    and the sweep is the only honest way to find it, since it depends on the batch width.
    """
    if not shares:
        raise ConfigError("there is nothing to sweep")
    batch = _sample(rows=rows, columns=4)
    out = []
    for share in shares:
        predicate = And(
            (
                Compare("<", column("c0"), literal(_threshold_for(share))),
                Compare("<", column("c1"), literal(500)),
            )
        )
        plain_meter = Meter()
        evaluate(predicate, batch, plain_meter)
        chained_meter = Meter()
        evaluate_chained(predicate, batch, chained_meter)
        out.append(
            {
                "selectivity": share,
                "plain": plain_meter.values_touched,
                "chained": chained_meter.values_touched,
                "chaining_wins": chained_meter.values_touched < plain_meter.values_touched,
            }
        )
    winning = [row for row in out if row["chaining_wins"]]
    return {
        "rows": out,
        "crossover": winning[-1]["selectivity"] if winning else None,
        "it_wins_somewhere": bool(winning),
        "it_loses_somewhere": len(winning) < len(out),
    }


def a_wider_batch_makes_chaining_worse(
    rows: int = 50_000,
    widths: Sequence[int] = (1, 2, 4, 8, 16),
) -> list[dict]:
    """How the batch width moves the crossover.

    The gather between conjuncts copies every column, so a wide batch pays more for it while the
    saving stays the same. A plan that chains on a sixteen column batch is copying fifteen
    columns it does not need for the next predicate, which is the argument for projecting before
    filtering rather than after.
    """
    if not widths:
        raise ConfigError("there is nothing to sweep")
    out = []
    for width in widths:
        batch = _sample(rows=rows, columns=max(width, 2))
        predicate = And(
            (
                Compare("<", column("c0"), literal(100)),
                Compare("<", column("c1"), literal(500)),
            )
        )
        plain_meter = Meter()
        evaluate(predicate, batch, plain_meter)
        chained_meter = Meter()
        evaluate_chained(predicate, batch, chained_meter)
        out.append(
            {
                "columns": max(width, 2),
                "plain": plain_meter.values_touched,
                "chained": chained_meter.values_touched,
                "ratio": round(
                    chained_meter.values_touched / max(plain_meter.values_touched, 1), 3
                ),
            }
        )
    return out


def the_answer_never_changes(rows: int = 20_000) -> dict:
    """Every form gives the same rows, on a batch with nulls in it.

    Three paths: whole batch, chained in the good order, chained in the bad order. Nulls are
    present because that is where the three valued logic in exec/expr.py has to survive being
    evaluated over a gathered array.
    """
    values = _column(rows, 1000, seed=3)
    holes = [None if position % 97 == 0 else value for position, value in enumerate(values)]
    batch = Batch.of(a=holes, b=_column(rows, 1000, seed=4))
    predicate = And(
        (
            Compare("<", column("a"), literal(200)),
            Compare("<", column("b"), literal(500)),
        )
    )
    plain = evaluate(predicate, batch)
    good = evaluate_chained(predicate, batch, order=(0, 1))
    bad = evaluate_chained(predicate, batch, order=(1, 0))
    return {
        "kept": plain.kept,
        "chained_matches": bool(np.array_equal(plain.positions, good.positions)),
        "either_order_matches": bool(
            np.array_equal(np.sort(plain.positions), np.sort(bad.positions))
        ),
        "nulls_were_present": batch.column("a").null_count > 0,
        "nulls_were_dropped": plain.kept < rows,
    }


def an_empty_selection_is_not_an_error(rows: int = 1_000) -> dict:
    """A predicate matching nothing, which every downstream operator has to survive."""
    batch = _sample(rows=rows, columns=2)
    selection = evaluate(Compare("<", column("c0"), literal(-1)), batch)
    return {
        "kept": selection.kept,
        "selectivity": selection.selectivity,
        "applies_cleanly": selection.apply(batch).rows == 0,
        "keeps_the_schema": selection.apply(batch).names == batch.names,
        "the_selection_form_is_cheaper": selection.cheaper_form == "selection",
    }


def a_full_selection_costs_the_gather(rows: int = 10_000) -> dict:
    """A predicate matching everything, where the filter is pure overhead.

    Worth measuring because a plan that cannot estimate selectivity will filter anyway, and the
    cost of a filter that keeps everything is one full gather of every column. That is the price
    of guessing wrong in the other direction from the chaining case.
    """
    batch = _sample(rows=rows, columns=4)
    meter = Meter()
    selection = evaluate(Compare(">=", column("c0"), literal(0)), batch, meter)
    applied = Meter()
    selection.apply(batch, applied)
    return {
        "kept": selection.kept,
        "everything_survived": selection.kept == rows,
        "predicate_values": meter.values_touched,
        "gather_values": applied.values_touched,
        "the_gather_is_the_cost": applied.values_touched > meter.values_touched,
        "the_mask_form_is_cheaper": selection.cheaper_form == "mask",
    }


def a_bad_ordering_is_refused() -> bool:
    """An order that is not a permutation of the conjuncts is a mistake."""
    batch = _sample(rows=100, columns=2)
    predicate = And(
        (
            Compare("<", column("c0"), literal(5)),
            Compare("<", column("c1"), literal(5)),
        )
    )
    try:
        evaluate_chained(predicate, batch, order=(0, 0))
    except ConfigError:
        return True
    return False


def a_non_boolean_predicate_is_refused() -> bool:
    """A filter takes a predicate, not a value."""
    batch = _sample(rows=100, columns=2)
    try:
        evaluate(column("c0"), batch)
    except Exception as problem:
        return "boolean" in str(problem)
    return False


def a_mismatched_refinement_is_refused() -> bool:
    """A refining mask is over the kept rows, and a wrong length says so."""
    try:
        everything(10).refine(np.ones(3, dtype=bool))
    except ConfigError:
        return True
    return False


def a_negative_row_count_is_refused() -> bool:
    """A selection over a negative number of rows is a configuration mistake."""
    try:
        everything(-1)
    except ConfigError:
        return True
    return False


def summarise(rows: int = 100_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    chained = chaining_conjuncts_saves_the_selectivity(rows=rows)
    ordered = the_order_of_the_conjuncts_matters_more(rows=rows)
    crossover = the_crossover_selectivity(rows=rows)
    return {
        "chaining_ratio": chained["ratio"],
        "ordering_ratio": ordered["ratio"],
        "crossover": crossover["crossover"],
        "chaining_wins_somewhere": crossover["it_wins_somewhere"],
        "and_loses_somewhere": crossover["it_loses_somewhere"],
        "predicate_type": BOOLEAN,
    }


def rebuild(parts: Sequence[Expr]) -> Expr:
    """Rejoin conjuncts, which plan/rules/pushdown.py needs after redistributing them."""
    return all_of(list(parts))
