from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from cqe.cost.meter import Meter
from cqe.errors import ConfigError, UnknownColumn
from cqe.exec.batch import Batch
from cqe.types.schema import FLOATING, INTEGER

# Sorting, and the two questions worth asking about it in a columnar engine.
#
# The first is what to move. A sort produces an ordering; applying it to the data is a separate
# gather, and the two costs are unrelated. Sorting a ten column batch on one key costs a
# comparison sort over one column and a gather over ten. So a plan that sorts early and projects
# late pays nine columns of gather it did not need, and the module measures that ratio rather
# than asserting it.
#
# The second is that a query with a limit does not need a sort. Taking the smallest k of n needs
# a partial selection, which numpy does with argpartition, and the work is linear in n rather
# than n log n.
#
# The comparison saving is smaller than it sounds. Under the standard models, n log2 n against n
# plus k log2 k, a hundred thousand rows with a limit of ten is a factor of 16.6 and not the
# thousands the asymptotics suggest, because log2 of a hundred thousand is only 17. Those counts
# are a model and the module labels them so, since everything else here is counted.
#
# The saving that is counted is the gather, and it is enormous: a full sort moves every row of
# every column and a top k moves ten, which at these sizes is a factor of ten thousand. That is
# the number a plan should be choosing on, and it is the one that does not need a model.
#
# Multi key sorts are done as a sequence of stable single key sorts from the last key to the
# first, which is the standard trick and is correct exactly because numpy's mergesort is stable.
# It costs one pass per key rather than one comparison over a composite, and it means a two key
# sort costs twice a one key sort and not more.

DIRECTIONS = ("ascending", "descending")


@dataclass(frozen=True)
class SortKey:
    """One column to order by, and how."""

    name: str
    descending: bool = False
    nulls_first: bool = False

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "name": self.name,
            "direction": DIRECTIONS[1] if self.descending else DIRECTIONS[0],
            "nulls_first": self.nulls_first,
        }


@dataclass
class Ordering:
    """A permutation of row positions, and what producing it cost."""

    positions: np.ndarray
    rows: int
    keys: tuple[SortKey, ...]
    strategy: str

    def __post_init__(self) -> None:
        if self.positions.dtype.kind not in "iu":
            raise ConfigError(f"positions are integers, not {self.positions.dtype}")

    @property
    def kept(self) -> int:
        """How many positions the ordering carries, which a top k truncates."""
        return int(self.positions.shape[0])

    def apply(self, batch: Batch, meter: Meter | None = None) -> Batch:
        """Gather the rows into order."""
        return batch.take(self.positions, meter=meter)

    def comparison_model(self) -> int:
        """The comparisons a sort of this shape would make, under the standard model.

        A model and not a measurement. numpy does the comparisons inside compiled code where
        this package cannot count them, so the number here is n log2 n or n plus k log2 k
        depending on the strategy. Every other cost in this package is counted; this one is
        derived, and it is labelled so nobody reads it as the same kind of number.
        """
        if self.rows <= 1:
            return 0
        per_key = len(self.keys) or 1
        if self.strategy == "partition":
            k = max(self.kept, 1)
            return per_key * (self.rows + int(k * max(np.log2(max(k, 2)), 1)))
        return per_key * int(self.rows * np.log2(self.rows))

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "rows": self.rows,
            "kept": self.kept,
            "keys": [key.name for key in self.keys],
            "strategy": self.strategy,
            "modelled_comparisons": self.comparison_model(),
        }


def _rank_array(batch: Batch, key: SortKey) -> tuple[np.ndarray, np.ndarray | None]:
    """The values to order by, and the validity mask that decides where nulls go."""
    if key.name not in batch.names:
        raise UnknownColumn(f"{key.name} is not in {list(batch.names)}")
    column = batch.column(key.name)
    return column.values, column.valid


def order_by(
    batch: Batch,
    keys: Sequence[SortKey],
    meter: Meter | None = None,
) -> Ordering:
    """A full ordering on any number of keys.

    Stable sorts applied from the last key to the first. Stability is what makes that correct:
    the sort on key one leaves rows equal on key one in the order the sort on key two left them,
    so the composite ordering falls out without ever comparing a composite.
    """
    if not keys:
        raise ConfigError("a sort needs at least one key")
    positions = np.arange(batch.rows, dtype=np.int64)
    for key in reversed(list(keys)):
        values, valid = _rank_array(batch, key)
        working = _flattened(values, valid)[positions]
        order = np.argsort(working, kind="stable")
        if key.descending:
            order = order[::-1]
            order = _restabilise(working, order)
        positions = positions[order]
        if valid is not None:
            positions = _move_nulls(positions, valid, key.nulls_first)
        if meter is not None:
            meter.touch(batch.rows, "sort_key", width=batch.column(key.name).field.width)
    if meter is not None:
        meter.compare(int(batch.rows * max(np.log2(max(batch.rows, 2)), 1)) * len(keys))
    return Ordering(positions=positions, rows=batch.rows, keys=tuple(keys), strategy="full")


def _flattened(values: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """The key values with every null row set to the same thing.

    A null row still holds something in the values array, and whatever it holds took part in the
    ordering. Two null rows then came out ordered by their leftover values rather than by their
    input order, so the sort was not stable across nulls: verify/differential.py generated a
    column with three nulls and the fast path returned them in a different order from the
    reference, which does keep them in input order.

    The fix is to make every null equal before the comparison rather than to reorder them after
    it. Ordering them afterwards would be wrong for a multi key sort, because the nulls of one
    key must stay in the order the less significant keys left them in, and that order is only
    knowable at this point.
    """
    if valid is None or bool(valid.all()):
        return values
    out = values.copy()
    out[~valid] = values[valid][0] if bool(valid.any()) else values[0]
    return out


def _restabilise(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Restore stability after reversing an ascending order for a descending sort.

    Reversing an ascending stable order gives a descending order that is anti stable: rows equal
    on the key come back in the opposite order from the input. That is invisible on a single key
    sort and produces the wrong answer on a multi key one, because the later key's ordering has
    been reversed inside each tie. Fixing it costs one more stable sort over the tie groups.
    """
    ranked = values[order]
    boundaries = np.flatnonzero(np.concatenate([[True], ranked[1:] != ranked[:-1]]))
    if len(boundaries) == len(order):
        return order
    out = order.copy()
    edges = np.append(boundaries, len(order))
    for start, stop in pairwise(edges):
        if stop - start > 1:
            out[start:stop] = out[start:stop][::-1]
    return out


def _move_nulls(positions: np.ndarray, valid: np.ndarray, first: bool) -> np.ndarray:
    """Move the null rows to one end, keeping the order of everything else.

    Nulls last by default and first when asked, in both directions, which is the rule
    verify/reference.py writes down. Most engines flip this with the sort direction and the
    inconsistency is invisible until a query mixes directions across keys.
    """
    present = valid[positions]
    if bool(present.all()):
        return positions
    kept = positions[present]
    missing = positions[~present]
    return np.concatenate([missing, kept]) if first else np.concatenate([kept, missing])


def top_k(
    batch: Batch,
    keys: Sequence[SortKey],
    k: int,
    meter: Meter | None = None,
) -> Ordering:
    """The first k rows in order, without ordering the rest.

    Partition to find the k smallest, then sort only those. On a single key this is exact and
    linear in the row count. On several keys the partition would have to be on a composite,
    which this does not build, so the multi key path falls back to a full sort and says so in
    the strategy rather than pretending.
    """
    if k < 1:
        raise ConfigError(f"{k} is not a limit")
    if not keys:
        raise ConfigError("a top k needs at least one key")
    if k >= batch.rows:
        ordering = order_by(batch, keys, meter)
        return Ordering(
            positions=ordering.positions, rows=batch.rows, keys=tuple(keys), strategy="full"
        )
    if len(keys) > 1 or batch.column(keys[0].name).has_nulls:
        ordering = order_by(batch, keys, meter)
        return Ordering(
            positions=ordering.positions[:k],
            rows=batch.rows,
            keys=tuple(keys),
            strategy="full",
        )
    key = keys[0]
    values, _ = _rank_array(batch, key)
    working = -values if key.descending else values
    candidates = np.argpartition(working, k)[:k]
    order = np.argsort(working[candidates], kind="stable")
    positions = candidates[order].astype(np.int64)
    if meter is not None:
        meter.touch(batch.rows, "top_k_key", width=batch.column(key.name).field.width)
        meter.compare(batch.rows + int(k * max(np.log2(max(k, 2)), 1)))
    return Ordering(
        positions=positions, rows=batch.rows, keys=tuple(keys), strategy="partition"
    )


def sort(batch: Batch, keys: Sequence[SortKey], meter: Meter | None = None) -> Batch:
    """Order a batch, which is what an operator above wants."""
    return order_by(batch, keys, meter).apply(batch, meter)


def _sample(rows: int = 100_000, columns: int = 8, seed: int = 0) -> Batch:
    """A batch wide enough that the gather cost is visible against the sort cost."""
    if rows < 1 or columns < 2:
        raise ConfigError(f"{rows} rows of {columns} columns is not a batch")
    generator = np.random.default_rng(seed)
    named = {
        f"c{position}": generator.integers(0, 1_000_000, size=rows).tolist()
        for position in range(columns)
    }
    return Batch.of(**named)


def the_gather_costs_more_than_the_sort(
    rows: int = 100_000,
    widths: Sequence[int] = (2, 4, 8, 16, 32),
) -> list[dict]:
    """How the values moved by a sort grow with the batch width, at a fixed key count.

    The sort reads one column per key whatever the batch holds. The gather moves every column.
    So the values touched by the whole operation are dominated by the width, and the ratio
    between them is the width over the key count.
    """
    if not widths:
        raise ConfigError("there is nothing to sweep")
    out = []
    for width in widths:
        batch = _sample(rows=rows, columns=width)
        keys = [SortKey("c0")]
        sort_meter = Meter()
        ordering = order_by(batch, keys, sort_meter)
        gather_meter = Meter()
        ordering.apply(batch, gather_meter)
        out.append(
            {
                "columns": width,
                "sort_values": sort_meter.values_touched,
                "gather_values": gather_meter.values_touched,
                "ratio": round(
                    gather_meter.values_touched / max(sort_meter.values_touched, 1), 2
                ),
            }
        )
    return out


def a_top_k_avoids_almost_all_of_the_comparisons(
    rows: int = 100_000,
    limits: Sequence[int] = (1, 10, 100, 1_000, 10_000),
) -> list[dict]:
    """Comparisons under the model, for a full sort against a partition, as the limit rises.

    The full sort is n log2 n whatever the limit. The partition is n plus k log2 k, so it starts
    near n and approaches the full sort as k approaches n. The ratios are 16.6, 16.6, 16.5, 15.1
    and 7.1 at limits from one to ten thousand.

    Two things to read off that. The saving is flat over four orders of magnitude of limit, so
    there is no limit small enough to be worth special casing. And the best it ever gets is
    16.6, which is log2 of the row count, because that is all a full sort ever costs over a
    linear scan. The asymptotic argument oversells this badly at realistic sizes.
    """
    if not limits:
        raise ConfigError("there is nothing to sweep")
    batch = _sample(rows=rows, columns=4)
    full = order_by(batch, [SortKey("c0")])
    out = []
    for limit in limits:
        partial = top_k(batch, [SortKey("c0")], limit)
        out.append(
            {
                "limit": limit,
                "strategy": partial.strategy,
                "full_comparisons": full.comparison_model(),
                "partial_comparisons": partial.comparison_model(),
                "ratio": round(full.comparison_model() / max(partial.comparison_model(), 1), 1),
            }
        )
    return out


def and_moves_almost_nothing(rows: int = 100_000, limit: int = 10) -> dict:
    """The gather is where the top k saving is actually counted rather than modelled.

    A full sort gathers every row of every column. A top k gathers k rows of every column. That
    is a saving of the row count over the limit, and unlike the comparison model it is a number
    this package counts directly.
    """
    batch = _sample(rows=rows, columns=8)
    full = order_by(batch, [SortKey("c0")])
    full_meter = Meter()
    full.apply(batch, full_meter)
    partial = top_k(batch, [SortKey("c0")], limit)
    partial_meter = Meter()
    partial.apply(batch, partial_meter)
    return {
        "rows": rows,
        "limit": limit,
        "full_gather": full_meter.values_touched,
        "top_k_gather": partial_meter.values_touched,
        "ratio": round(full_meter.values_touched / max(partial_meter.values_touched, 1), 1),
        "the_saving_is_the_row_count_over_the_limit": (
            abs(full_meter.values_touched / max(partial_meter.values_touched, 1) - rows / limit)
            < 1
        ),
    }


def the_top_k_answer_matches_a_full_sort(rows: int = 20_000, limit: int = 50) -> dict:
    """The partition path and the sort path return the same rows in the same order.

    Checked because argpartition returns the k smallest in no particular order, and a
    disagreement here means the sort of the partition was applied to the wrong indices, which is
    the one bug this path invites.
    """
    batch = _sample(rows=rows, columns=3)
    keys = [SortKey("c0")]
    full = order_by(batch, keys).apply(batch)
    partial = top_k(batch, keys, limit).apply(batch)
    return {
        "limit": limit,
        "same_rows": full.slice(0, limit).to_rows() == partial.to_rows(),
        "strategy": top_k(batch, keys, limit).strategy,
        "it_used_the_partition": top_k(batch, keys, limit).strategy == "partition",
    }


def a_multi_key_sort_costs_one_pass_per_key(
    rows: int = 100_000,
    counts: Sequence[int] = (1, 2, 3, 4),
) -> list[dict]:
    """Sorting on more keys costs linearly in the key count, not worse.

    Because the implementation is a sequence of stable single key sorts rather than a comparison
    over a composite key. That is the whole reason for doing it that way: a composite comparator
    would cost the same in the best case and would need a materialised composite key in memory.
    """
    if not counts:
        raise ConfigError("there is nothing to sweep")
    batch = _sample(rows=rows, columns=8)
    out = []
    for count in counts:
        keys = [SortKey(f"c{position}") for position in range(count)]
        meter = Meter()
        order_by(batch, keys, meter)
        out.append(
            {
                "keys": count,
                "values": meter.values_touched,
                "per_key": meter.values_touched // count,
            }
        )
    return out


def a_descending_multi_key_sort_stays_stable(rows: int = 5_000) -> dict:
    """The bug reversing an ascending order introduces, and the check that catches it.

    Reversing an ascending stable order gives a descending order in which rows equal on the key
    come back in the opposite order from the input. On a single key that is invisible. On two
    keys it reverses the second key's ordering inside each tie of the first, which is wrong.

    Measured by sorting descending on a low cardinality first key and ascending on a second, and
    checking the second key really is ascending inside every group of the first.
    """
    generator = np.random.default_rng(11)
    batch = Batch.of(
        a=generator.integers(0, 5, size=rows).tolist(),
        b=generator.integers(0, 1000, size=rows).tolist(),
    )
    keys = [SortKey("a", descending=True), SortKey("b")]
    ordered = sort(batch, keys)
    first = np.asarray(ordered.column("a").to_list())
    second = np.asarray(ordered.column("b").to_list())
    within = True
    start = 0
    for position in range(1, len(first) + 1):
        if position == len(first) or first[position] != first[start]:
            block = second[start:position]
            within = within and bool((np.diff(block) >= 0).all())
            start = position
    return {
        "rows": rows,
        "first_key_descends": bool((np.diff(first) <= 0).all()),
        "second_key_ascends_within_groups": within,
        "groups": len(np.unique(first)),
    }


def nulls_go_where_they_are_told(rows: int = 1_000) -> dict:
    """Nulls last by default and first when asked, in both directions.

    The rule verify/reference.py writes down, and the one most engines get inconsistent by
    flipping it with the sort direction. Checked in all four combinations here because that is
    where the inconsistency would show.
    """
    values = [None if position % 7 == 0 else position for position in range(rows)]
    batch = Batch.of(a=values)
    cases = {}
    for descending in (False, True):
        for first in (False, True):
            key = SortKey("a", descending=descending, nulls_first=first)
            ordered = sort(batch, [key]).column("a").to_list()
            label = f"{'desc' if descending else 'asc'} nulls {'first' if first else 'last'}"
            cases[label] = (ordered[0] is None) == first
    return cases


def an_empty_batch_sorts_to_nothing() -> dict:
    """The degenerate case every operator has to survive."""
    batch = Batch.empty(_sample(rows=2, columns=2).schema)
    ordering = order_by(batch, [SortKey("c0")])
    return {
        "rows": ordering.rows,
        "kept": ordering.kept,
        "applies_cleanly": ordering.apply(batch).rows == 0,
        "no_comparisons": ordering.comparison_model() == 0,
    }


def a_limit_past_the_end_is_a_full_sort(rows: int = 1_000) -> dict:
    """Asking for more rows than there are, which falls back rather than refusing."""
    batch = _sample(rows=rows, columns=2)
    ordering = top_k(batch, [SortKey("c0")], rows * 2)
    return {
        "kept": ordering.kept,
        "strategy": ordering.strategy,
        "it_kept_everything": ordering.kept == rows,
        "it_used_a_full_sort": ordering.strategy == "full",
    }


def a_top_k_with_nulls_falls_back(rows: int = 1_000) -> dict:
    """A key with nulls cannot be partitioned, so the path says so rather than guessing.

    argpartition has no notion of where a null belongs, and the sentinel that would give it one
    is exactly the trick columns/array.py refuses for the storage itself. So this falls back to
    a full sort and records that it did, which is what a plan reading the strategy needs.
    """
    values = [None if position % 11 == 0 else position for position in range(rows)]
    batch = Batch.of(a=values, b=list(range(rows)))
    ordering = top_k(batch, [SortKey("a")], 10)
    return {
        "strategy": ordering.strategy,
        "it_fell_back": ordering.strategy == "full",
        "kept": ordering.kept,
        "the_answer_is_still_right": ordering.apply(batch).column("a").to_list()
        == sorted(v for v in values if v is not None)[:10],
    }


def a_sort_with_no_keys_is_refused() -> bool:
    """An ordering has to be on something."""
    try:
        order_by(_sample(rows=10, columns=2), [])
    except ConfigError:
        return True
    return False


def a_zero_limit_is_refused() -> bool:
    """A top nothing is a configuration mistake, not an empty result."""
    try:
        top_k(_sample(rows=10, columns=2), [SortKey("c0")], 0)
    except ConfigError:
        return True
    return False


def an_unknown_key_is_refused() -> bool:
    """Sorting on a column that is not there says which columns are."""
    try:
        order_by(_sample(rows=10, columns=2), [SortKey("z")])
    except UnknownColumn:
        return True
    return False


def compare_the_strategies(rows: int = 100_000, limit: int = 10) -> list[dict]:
    """A full sort against a top k, counted and modelled, as one table."""
    batch = _sample(rows=rows, columns=8)
    out = []
    for name in ("full", "top k"):
        meter = Meter()
        if name == "full":
            ordering = order_by(batch, [SortKey("c0")], meter)
        else:
            ordering = top_k(batch, [SortKey("c0")], limit, meter)
        gather = Meter()
        ordering.apply(batch, gather)
        row = ordering.as_dict()
        row["name"] = name
        row["gather_values"] = gather.values_touched
        out.append(row)
    return out


def summarise(rows: int = 100_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    gathers = the_gather_costs_more_than_the_sort(rows=rows)
    limits = a_top_k_avoids_almost_all_of_the_comparisons(rows=rows)
    moved = and_moves_almost_nothing(rows=rows)
    return {
        "widest_gather_ratio": gathers[-1]["ratio"],
        "narrowest_gather_ratio": gathers[0]["ratio"],
        "top_one_comparison_ratio": limits[0]["ratio"],
        "top_k_gather_ratio": moved["ratio"],
        "numeric_types": (INTEGER, FLOATING),
    }
