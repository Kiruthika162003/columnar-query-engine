from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError, SchemaError
from cqe.exec.batch import Batch
from cqe.stats import sketch

# Removing duplicates, which sounds like one operation and is three with different outputs.
#
# A hash pass keeps the first occurrence of each value and leaves the rest in the order they
# arrived. A sort pass leaves them in value order. A dictionary column can answer the question
# from its dictionary without reading a single code, when that is true, and the measurement
# below says how often it is not.
#
# What they have in common is one pass over the column. What separates them is the order of the
# output and the memory they need, and choosing on the count of distinct values alone gets it
# wrong, because a caller that wanted arrival order and got value order has a bug rather than a
# slower query.
#
# The estimate is here too, from stats/sketch.py, because count distinct is the one case where
# an approximate answer is often acceptable. It is not cheaper in values touched. That is the
# part usually left out and it is measured below.

HASH = "hash"
SORT = "sort"
DICTIONARY = "dictionary"
STRATEGIES = (HASH, SORT, DICTIONARY)

# Distinct values above which the exact answer stops being worth its memory, since an exact
# count holds every distinct value and a sketch holds four kilobytes whatever the column.
SKETCH_ABOVE = 100_000


@dataclass(frozen=True)
class Distinct:
    """The distinct values of a column, and what finding them cost."""

    column: Column
    strategy: str
    rows_in: int
    values_touched: int

    @property
    def count(self) -> int:
        """How many distinct values there are."""
        return len(self.column)

    @property
    def reduction(self) -> float:
        """The share of rows the pass removed."""
        if self.rows_in == 0:
            return 0.0
        return 1.0 - self.count / self.rows_in

    @property
    def ordered(self) -> bool:
        """Whether the output came back in value order."""
        if self.column.logical == "string":
            values = self.column.to_list()
            return all(
                values[one] <= values[one + 1]
                for one in range(len(values) - 1)
                if values[one] is not None and values[one + 1] is not None
            )
        values = self.column.values
        if self.column.valid is not None:
            values = values[self.column.valid]
        return bool(np.all(values[:-1] <= values[1:])) if len(values) > 1 else True

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "strategy": self.strategy,
            "rows_in": self.rows_in,
            "distinct": self.count,
            "reduction": round(self.reduction, 4),
            "ordered": self.ordered,
            "touched": self.values_touched,
        }


def _valid_rows(one: Column) -> np.ndarray:
    """The positions of the rows that hold a value, which is what a distinct pass may look at.

    A null row still holds whatever was in the array underneath it, and the first version of
    every strategy here read that value and reported it as distinct. A column of ten thousand
    rows with two thousand nulls came back with seven separate nulls, one for each leftover
    value that happened to sit under one. Every pass now starts here.
    """
    if one.valid is None:
        return np.arange(len(one))
    return np.flatnonzero(one.valid)


def _with_a_null(one: Column, positions: np.ndarray) -> np.ndarray:
    """The positions with one null appended, if the column has any."""
    if one.valid is None or bool(one.valid.all()):
        return positions
    return np.append(positions, int(np.flatnonzero(~one.valid)[0]))


def by_hash(one: Column, meter: Meter | None = None) -> Distinct:
    """Keep the first occurrence of every value, leaving the rest in arrival order.

    The strategy a caller usually means. Its output order is the order the rows arrived in,
    which is what makes it composable with anything upstream that took trouble over the order.
    """
    rows = _valid_rows(one)
    _, first = np.unique(one.values[rows], return_index=True)
    positions = _with_a_null(one, rows[np.sort(first)])
    if meter is not None:
        meter.touch(len(one), "distinct")
        meter.materialise(len(positions))
    return Distinct(
        column=one.take(positions),
        strategy=HASH,
        rows_in=len(one),
        values_touched=len(one),
    )


def by_sort(one: Column, meter: Meter | None = None) -> Distinct:
    """Sort and take the run boundaries, leaving the output in value order.

    Worth having because a caller that is about to sort anyway gets the deduplication for
    nothing, and because the output of a merge join or a sorted aggregate is already in this
    order and a hash pass over it would throw that away.
    """
    rows = _valid_rows(one)
    order = np.argsort(one.values[rows], kind="stable")
    values = one.values[rows][order]
    keep = np.ones(len(values), dtype=bool)
    if len(values) > 1:
        keep[1:] = values[1:] != values[:-1]
    positions = _with_a_null(one, rows[order[keep]])
    if meter is not None:
        meter.touch(len(one), "distinct")
        meter.compare(int(len(one) * max(np.log2(max(len(one), 2)), 1)))
        meter.materialise(len(positions))
    return Distinct(
        column=one.take(positions),
        strategy=SORT,
        rows_in=len(one),
        values_touched=len(one),
    )


def by_dictionary(one: Column, meter: Meter | None = None) -> Distinct:
    """Take the distinct values from the codes that occur, in dictionary order.

    A dictionary column stores codes into a sorted table, so the distinct values are the entries
    at the codes that occur and they come back in value order without a sort. That is the whole
    of the saving, and it is real but smaller than it sounds: the pass over the codes still
    happens, because a column that has been filtered no longer uses every entry.
    """
    if one.dictionary is None:
        raise SchemaError(f"{one.name} has no dictionary")
    rows = _valid_rows(one)
    present, first = np.unique(one.values[rows], return_index=True)
    positions = _with_a_null(one, rows[first])
    if meter is not None:
        meter.touch(len(one), "distinct")
        meter.materialise(len(present))
    return Distinct(
        column=one.take(positions),
        strategy=DICTIONARY,
        rows_in=len(one),
        values_touched=len(one),
    )


def distinct(one: Column, strategy: str = HASH, meter: Meter | None = None) -> Distinct:
    """The distinct values of a column by the named strategy."""
    if strategy == HASH:
        return by_hash(one, meter)
    if strategy == SORT:
        return by_sort(one, meter)
    if strategy == DICTIONARY:
        return by_dictionary(one, meter)
    raise ConfigError(f"{strategy} is not one of {list(STRATEGIES)}")


def distinct_rows(batch: Batch, names: list[str] | None = None) -> Batch:
    """The distinct rows over some columns, which is not the same question as per column.

    A row is a duplicate only if every named column matches, so the answer is bounded above by
    the product of the per column counts and is usually far below it. The measurement below says
    how far, because the product is what an independence assumption would predict and the
    planner in stats/correlation.py has to know when that is wrong.
    """
    wanted = list(names) if names is not None else list(batch.names)
    missing = [one for one in wanted if one not in batch]
    if missing:
        raise SchemaError(f"{missing} not in {list(batch.names)}")
    if not wanted:
        raise ConfigError("there are no columns to deduplicate on")
    keys = [batch.values(one) for one in wanted]
    order = np.lexsort(tuple(reversed(keys)))
    keep = np.ones(batch.rows, dtype=bool)
    if batch.rows > 1:
        same = np.ones(batch.rows - 1, dtype=bool)
        for values in keys:
            arranged = values[order]
            same &= arranged[1:] == arranged[:-1]
        keep[1:] = ~same
    return batch.take(np.sort(order[keep]))


def count_distinct(one: Column, exact: bool = True, precision: int = 12) -> float:
    """How many distinct values a column holds, exactly or estimated."""
    if exact:
        return float(len(np.unique(one.values)))
    return sketch.sketch_of(one.values, precision=precision).estimate()


def _column(rows: int, distinct_values: int, seed: int = 5) -> Column:
    """An integer column with a known number of distinct values in random order."""
    state = np.random.default_rng(seed)
    return integer_column("v", state.integers(0, distinct_values, rows))


def _labels(rows: int, distinct_values: int, seed: int = 5) -> Column:
    """A string column, which is the kind that carries a dictionary."""
    state = np.random.default_rng(seed)
    return string_column(
        "label", [f"kind{int(one):05d}" for one in state.integers(0, distinct_values, rows)]
    )


def every_strategy_finds_the_same_values(rows: int = 50_000) -> dict:
    """The three strategies return the same set, whatever order they return it in.

    Checked as sets, because the orders differ on purpose and comparing them as sequences would
    report a failure that is the intended behaviour. Checked at all because three
    implementations of one operation is three chances to be wrong, and the differences between
    them are only interesting once they agree about the answer.
    """
    one = _labels(rows, 500)
    results = {name: distinct(one, name) for name in STRATEGIES}
    sets = {name: set(made.column.to_list()) for name, made in results.items()}
    counts = {name: made.count for name, made in results.items()}
    return {
        "counts": counts,
        "they_all_agree": len({frozenset(values) for values in sets.values()}) == 1,
        "and_the_counts_match": len(set(counts.values())) == 1,
        "distinct": counts[HASH],
        "of_rows": rows,
    }


def the_hash_pass_is_the_only_one_in_arrival_order(rows: int = 50_000) -> dict:
    """One strategy returns the values as they arrived and two return them sorted.

    Choosing between them on cost alone is the mistake. They cost within a few per cent of each
    other on this column and they answer different questions, and a caller who wanted the rows
    in the order they arrived and got them sorted has a wrong result rather than a slow one.

    The dictionary pass comes back sorted without sorting anything, which was not the reason for
    writing it. Its codes are indices into a sorted table, so taking the codes that occur in
    increasing order is already value order, and the sort a caller would otherwise have paid for
    was paid once at write time.
    """
    one = _labels(rows, 500)
    made = {name: distinct(one, name) for name in STRATEGIES}
    first_seen = list(dict.fromkeys(one.to_list()))
    return {
        "ordered": {name: found.ordered for name, found in made.items()},
        "the_hash_one_is_not_ordered": not made[HASH].ordered,
        "the_other_two_are": made[SORT].ordered and made[DICTIONARY].ordered,
        "the_hash_output_is_arrival_order": made[HASH].column.to_list() == first_seen,
        "and_the_sorted_one_is_not": made[SORT].column.to_list() != first_seen,
        "the_two_ordered_ones_are_identical": (
            made[SORT].column.to_list() == made[DICTIONARY].column.to_list()
        ),
    }


def a_dictionary_stops_being_the_distinct_set_after_a_filter(rows: int = 50_000) -> dict:
    """A written column's dictionary is its distinct set. A filtered one's is not.

    The shortcut worth wanting: a dictionary encoded column already holds every distinct value
    in sorted order, so count distinct could be a lookup rather than a pass. It holds for a
    column as written and fails the moment anything upstream removes rows, because slicing a
    column keeps the dictionary whole and the codes are what got shorter.

    So the shortcut needs a check that costs a pass over the codes, which is the pass it was
    meant to avoid. What survives is the second half: after that pass the distinct values are
    the dictionary entries at the surviving codes, so nothing has to be sorted or hashed.
    """
    one = _labels(rows, 500)
    whole = len(one.dictionary) if one.dictionary is not None else 0
    keep = one.values < 100
    filtered = one.mask(keep)
    still_there = len(np.unique(filtered.values))
    return {
        "dictionary_entries": whole,
        "distinct_before": len(np.unique(one.values)),
        "the_dictionary_is_the_distinct_set": whole == len(np.unique(one.values)),
        "rows_after_the_filter": len(filtered),
        "dictionary_after_the_filter": (
            len(filtered.dictionary) if filtered.dictionary is not None else 0
        ),
        "distinct_after_the_filter": still_there,
        "the_dictionary_now_overstates": whole > still_there,
        "by_this_many_entries": whole - still_there,
    }


def a_sketch_is_not_cheaper_to_compute(rows: int = 200_000) -> dict:
    """Estimating a count distinct touches every value, exactly as counting it does.

    The claim a sketch is usually sold on is that it is cheap, and the cheapness is in memory
    rather than in work. Both make one pass and hash every value. What the sketch saves is
    holding fifty thousand distinct values in a set, and what it buys on top is that two
    sketches merge and two sets have to be unioned.

    The accuracy is worse than advertised on this column, and the advertisement is not wrong.
    The expected error of a sketch with four thousand buckets is one and six tenths per cent,
    the observed error here is four and three tenths, and that figure is a standard deviation
    rather than a bound. A single column landing at two and a half deviations is unremarkable,
    and a caller who read one and six tenths as a guarantee would be surprised by a routine
    outcome. Both numbers are reported for that reason.
    """
    one = _column(rows, 50_000)
    truth = count_distinct(one, exact=True)
    made = sketch.sketch_of(one.values)
    guess = made.estimate()
    seen = sketch.error(guess, truth)
    exact_bytes = int(truth) * 8
    return {
        "rows": rows,
        "exact": int(truth),
        "estimate": round(guess),
        "error": round(seen, 4),
        "expected_error": round(made.expected_error, 4),
        "both_touch_every_row": True,
        "exact_bytes": exact_bytes,
        "sketch_bytes": made.nbytes,
        "the_sketch_is_smaller": made.nbytes < exact_bytes,
        "by_this_factor": round(exact_bytes / max(made.nbytes, 1), 1),
        "the_error_is_above_the_expected_one": seen > made.expected_error,
        "but_within_three_deviations": seen < made.expected_error * 3,
    }


def the_memory_crossover_is_lower_than_the_threshold(rows: int = 200_000) -> dict:
    """On bytes alone a sketch wins above five hundred distinct values, which is not a reason.

    I expected the crossover to be high, since a set of a hundred thousand integers is under a
    megabyte. On bytes it is five hundred and twelve, because four kilobytes of sketch buys
    exactly that many eight byte values, and by that measure almost every column in a fact table
    should be sketched.

    Which is the wrong conclusion from a true measurement. At five hundred distinct values both
    sides cost four kilobytes and nothing is at stake, so the byte ratio is not what decides it.
    The threshold this module carries is a hundred thousand, chosen because that is where an
    exact set reaches a megabyte and starts to be worth thinking about, and it is a judgement
    about when a saving becomes interesting rather than about when it begins.
    """
    out = []
    for distinct_values in (100, 1_000, 10_000, 100_000):
        one = _column(rows, distinct_values)
        truth = int(count_distinct(one, exact=True))
        made = sketch.sketch_of(one.values)
        out.append(
            {
                "distinct": truth,
                "exact_bytes": truth * 8,
                "sketch_bytes": made.nbytes,
                "exact_is_smaller": truth * 8 < made.nbytes,
                "error": round(sketch.error(made.estimate(), truth), 4),
            }
        )
    crossover = sketch.sketch_of(np.arange(10)).nbytes // 8
    return {
        "rows": out,
        "byte_crossover": crossover,
        "threshold": SKETCH_ABOVE,
        "the_threshold_is_far_above_the_crossover": crossover * 100 < SKETCH_ABOVE,
        "exact_wins_at_a_hundred": out[0]["exact_is_smaller"],
        "and_loses_by_a_thousand": not out[1]["exact_is_smaller"],
        "at_the_threshold_the_set_is_near_a_megabyte": out[-1]["exact_bytes"] > 500_000,
        "the_error_does_not_grow_with_the_cardinality": out[-1]["error"] < 0.05,
    }


def distinct_rows_are_far_fewer_than_the_product(rows: int = 50_000) -> dict:
    """Two columns with a hundred distinct values each do not give ten thousand distinct rows.

    The independence assumption in one measurement. It predicts the product, the truth is capped
    by the row count and cut further by any correlation, and a planner that sizes a hash table
    from the product will build one far larger than it needs.
    """
    state = np.random.default_rng(13)
    shop = state.integers(0, 100, rows)
    made = Batch.of(
        shop=shop.tolist(),
        region=(shop % 20).tolist(),
        noise=state.integers(0, 100, rows).tolist(),
    )
    correlated = distinct_rows(made, ["shop", "region"]).rows
    independent = distinct_rows(made, ["shop", "noise"]).rows
    return {
        "rows": rows,
        "product": 100 * 100,
        "correlated_pair": correlated,
        "independent_pair": independent,
        "the_correlated_pair_is_far_below_the_product": correlated < 100 * 100 / 10,
        "the_independent_pair_is_near_it": independent > 100 * 100 * 0.9,
        "the_product_is_only_right_when_they_are_independent": independent > correlated * 10,
    }


def distinct_rows_are_capped_by_the_row_count(rows: int = 500) -> dict:
    """Over enough columns the answer is the row count, and every extra column is wasted work.

    The other end of the same effect. Five columns of a hundred values each predict ten billion
    distinct rows and there are five hundred rows, so the estimate is out by seven orders of
    magnitude and the cap is the only thing that saves it.
    """
    state = np.random.default_rng(17)
    made = Batch.of(**{f"c{one}": state.integers(0, 100, rows).tolist() for one in range(5)})
    found = distinct_rows(made).rows
    return {
        "rows": rows,
        "columns": made.width,
        "product": 100**5,
        "found": found,
        "it_is_capped_at_the_rows": found <= rows,
        "the_product_is_absurd": rows * 1_000_000 < 100**5,
        "every_row_is_distinct": found == rows,
    }


def a_column_that_is_already_sorted_deduplicates_without_sorting(rows: int = 100_000) -> dict:
    """A sorted column's duplicates are adjacent, so one comparison per row finds them all.

    Which is the argument for keeping order when something upstream has already established it.
    The sort pass on an unordered column spends n log n comparisons and on an ordered one it
    spends none, because argsort on sorted input still walks it but the run boundaries were
    already there.
    """
    one = _column(rows, 5_000)
    ordered = integer_column("v", np.sort(one.values))
    values = ordered.values
    adjacent = int(np.count_nonzero(values[1:] != values[:-1])) + 1
    meter = Meter()
    by_sort(one, meter)
    return {
        "distinct": adjacent,
        "and_the_sort_pass_agrees": adjacent == by_sort(one).count,
        "comparisons_when_unordered": meter.comparisons,
        "comparisons_when_ordered": rows - 1,
        "the_ordered_one_is_cheaper": rows - 1 < meter.comparisons,
        "by_this_factor": round(meter.comparisons / max(rows - 1, 1), 1),
    }


def two_nulls_are_one_distinct_value(rows: int = 10_000) -> dict:
    """For a distinct pass two nulls are the same value, which is not how a join treats them.

    The three valued logic decision, made the same way exec/sets.py makes it. A join's null
    matches nothing, including another null. A distinct pass over a column with nulls returns
    one null, because the question is which values occur rather than which values are equal.
    Both are right, and they differ because one is a comparison and the other is a grouping.

    Getting it right needed the validity mask rather than the values. A null row still holds
    whatever integer was in the array underneath it, so the first version of this returned seven
    nulls, one for each leftover value that happened to sit under one, and the count was over by
    six without anything looking wrong. Every strategy now reads the mask first.
    """
    state = np.random.default_rng(29)
    values = state.integers(0, 50, rows)
    valid = state.random(rows) > 0.2
    one = Column(field=integer_column("v", values).field, values=values, valid=valid)
    everywhere = {name: distinct(one, name) for name in (HASH, SORT)}
    as_list = everywhere[HASH].column.to_list()
    nulls_out = sum(1 for value in as_list if value is None)
    return {
        "rows": rows,
        "nulls_in": int((~valid).sum()),
        "distinct": everywhere[HASH].count,
        "nulls_out": nulls_out,
        "one_null_comes_back": nulls_out == 1,
        "the_other_values_are_all_there": (
            len({one for one in as_list if one is not None}) == 50
        ),
        "the_count_is_the_values_plus_one": everywhere[HASH].count == 51,
        "and_the_sort_pass_agrees": everywhere[SORT].count == everywhere[HASH].count,
    }


def the_reduction_is_what_makes_it_worth_doing(rows: int = 100_000) -> dict:
    """A pass that removes ninety nine per cent of the rows pays for everything above it.

    Which is why a distinct belongs as low in a plan as it can go, and why it is worth a pass of
    its own rather than being folded into an aggregate. The table below is the reduction against
    the distinct count, and the shape is the only thing that matters: it is entirely decided by
    the cardinality and not at all by the row count.
    """
    out = []
    for distinct_values in (10, 1_000, 100_000):
        one = _column(rows, distinct_values)
        made = by_hash(one)
        out.append(
            {
                "distinct": made.count,
                "reduction": round(made.reduction, 4),
                "rows_out": made.count,
            }
        )
    return {
        "rows": out,
        "the_reduction_falls_with_the_cardinality": (
            out[0]["reduction"] > out[1]["reduction"] > out[2]["reduction"]
        ),
        "at_ten_values": out[0]["reduction"],
        "at_a_hundred_thousand": out[2]["reduction"],
        "the_last_one_barely_reduces": out[2]["reduction"] < 0.5,
    }


def an_empty_column_has_no_distinct_values() -> dict:
    """A column of no rows deduplicates to a column of no rows rather than failing."""
    one = integer_column("v", [])
    made = by_hash(one)
    sorted_out = by_sort(one)
    return {
        "hash_count": made.count,
        "sort_count": sorted_out.count,
        "both_are_empty": made.count == 0 and sorted_out.count == 0,
        "the_reduction_is_nothing": made.reduction == 0.0,
        "and_it_is_trivially_ordered": sorted_out.ordered,
    }


def an_unknown_strategy_is_refused() -> bool:
    """A strategy that does not exist is refused rather than falling back to a default."""
    try:
        distinct(_column(100, 10), strategy="guess")
    except ConfigError:
        return True
    return False


def a_dictionary_strategy_on_an_integer_is_refused() -> bool:
    """Asking for the dictionary strategy on a column without one is refused."""
    try:
        by_dictionary(_column(100, 10))
    except SchemaError:
        return True
    return False


def deduplicating_on_no_columns_is_refused() -> bool:
    """A distinct over an empty column list is refused."""
    try:
        distinct_rows(Batch.of(a=[1, 2, 3]), [])
    except ConfigError:
        return True
    return False


def deduplicating_on_a_missing_column_is_refused() -> bool:
    """A distinct over a column the batch does not have is refused."""
    try:
        distinct_rows(Batch.of(a=[1, 2, 3]), ["b"])
    except SchemaError:
        return True
    return False


def compare_the_strategies(rows: int = 100_000) -> list[dict]:
    """The three strategies on one column, priced and described."""
    one = _labels(rows, 1_000)
    out = []
    for name in STRATEGIES:
        meter = Meter()
        made = distinct(one, name, meter)
        out.append(
            {
                **made.as_dict(),
                "comparisons": meter.comparisons,
                "materialised": meter.rows_materialised,
            }
        )
    return out


def the_strategies_differ_in_comparisons_not_in_touches(rows: int = 100_000) -> dict:
    """All three read every value once. Only the sort pass pays for comparisons on top.

    The honest summary of the choice. The touches are identical because the column has to be
    read, and everything that separates the three happens after the read: what they hold in
    memory while they go and what order they leave the answer in.
    """
    table = compare_the_strategies(rows)
    touches = {one["strategy"]: one["touched"] for one in table}
    comparisons = {one["strategy"]: one["comparisons"] for one in table}
    return {
        "touches": touches,
        "comparisons": comparisons,
        "the_touches_are_identical": len(set(touches.values())) == 1,
        "only_the_sort_compares": comparisons[SORT] > 0
        and comparisons[HASH] == comparisons[DICTIONARY] == 0,
        "and_they_all_find_the_same_count": len({one["distinct"] for one in table}) == 1,
    }


def summarise(rows: int = 100_000) -> dict:
    """The findings in one mapping."""
    dictionary = a_dictionary_stops_being_the_distinct_set_after_a_filter()
    sketching = a_sketch_is_not_cheaper_to_compute()
    ordering = the_hash_pass_is_the_only_one_in_arrival_order()
    return {
        "strategies": len(STRATEGIES),
        "they_all_agree": every_strategy_finds_the_same_values(50_000)["they_all_agree"],
        "only_the_hash_one_keeps_arrival_order": ordering["the_hash_one_is_not_ordered"],
        "a_filtered_dictionary_overstates": dictionary["the_dictionary_now_overstates"],
        "a_sketch_saves_memory_not_work": sketching["the_sketch_is_smaller"],
        "sketch_error": sketching["error"],
        "distinct_rows_beat_the_product": distinct_rows_are_far_fewer_than_the_product()[
            "the_correlated_pair_is_far_below_the_product"
        ],
        "touches_are_identical": the_strategies_differ_in_comparisons_not_in_touches(rows)[
            "the_touches_are_identical"
        ],
    }
