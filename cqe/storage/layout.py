from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.columns.encode import dictionary, runlength
from cqe.errors import ConfigError
from cqe.exec.batch import Batch, stack
from cqe.exec.expr import Compare, Expr, column, literal
from cqe.exec.filter import apply as apply_predicate
from cqe.storage.bloom import build_for
from cqe.storage.statistics import can_skip, collect

# How rows are arranged into groups before they are written, which is the one decision in this
# package that a reader cannot undo.
#
# Everything else about a file is a representation: an encoding can be changed on rewrite, a
# statistic can be recomputed, a bloom filter can be added. The order of the rows and where the
# group boundaries fall are baked in, and every pruning structure downstream is only as good as
# that order made it.
#
# Three arrangements are measured here and they are the three that come up.
#
# As they arrive, which is the honest default. Whatever order the data was produced in, which
# for an event stream is usually time and for anything else is nothing in particular.
#
# Sorted by one column, which makes that column's zone map exact: each group holds a contiguous
# range and a predicate on it prunes everything outside. It costs a sort and it costs every
# other column's locality, which is the part that gets forgotten.
#
# Clustered by a low cardinality column, which is sorting by something with few distinct values.
# It gives most of the pruning of a sort on that column and leaves the rest of the order alone.
#
# The measurements are about what each one buys and what it costs the columns it did not favour,
# because a layout decision is always a trade between columns and a claim about one column alone
# is half the story.

# The row group size storage/statistics.py measured as the total cost minimum, repeated here
# because the layout functions default to it and a reader should not have to go and look.
GROUP_SIZE = 500


@dataclass(frozen=True)
class Layout:
    """A table cut into row groups, and what it was arranged by."""

    groups: tuple[Batch, ...]
    order: str
    key: str = ""

    @property
    def rows(self) -> int:
        """Rows across every group."""
        return sum(one.rows for one in self.groups)

    @property
    def nbytes(self) -> int:
        """Bytes the data occupies."""
        return sum(one.nbytes for one in self.groups)

    def flatten(self) -> Batch:
        """Every group concatenated, which is the table the layout was built from."""
        if not self.groups:
            raise ConfigError("an empty layout has no rows")
        return stack(list(self.groups))

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "order": self.order,
            "key": self.key,
            "groups": len(self.groups),
            "rows": self.rows,
            "bytes": self.nbytes,
        }


def cut(batch: Batch, group_size: int = GROUP_SIZE) -> list[Batch]:
    """A table sliced into row groups of a fixed size, the last one short."""
    if group_size <= 0:
        raise ConfigError(f"{group_size} rows is not a group")
    return [
        batch.slice(start, min(start + group_size, batch.rows))
        for start in range(0, max(batch.rows, 1), group_size)
    ]


def as_they_arrive(batch: Batch, group_size: int = GROUP_SIZE) -> Layout:
    """The rows in the order they were produced, cut into groups."""
    return Layout(groups=tuple(cut(batch, group_size)), order="arrival")


def sorted_by(batch: Batch, name: str, group_size: int = GROUP_SIZE) -> Layout:
    """The rows ordered by one column.

    Stable, so that a second sort by a second column would leave the first one's order inside
    each of its groups. That costs nothing here and is the property that makes a layout sorted
    by two columns mean what a reader expects.
    """
    if name not in batch.schema:
        raise ConfigError(f"{name} is not a column of {list(batch.schema.names)}")
    order = np.argsort(batch.column(name).values, kind="stable")
    return Layout(groups=tuple(cut(batch.take(order), group_size)), order="sorted", key=name)


def clustered_by(batch: Batch, name: str, group_size: int = GROUP_SIZE) -> Layout:
    """The rows gathered so that equal values of one column sit together.

    The same as sorting by that column when the column has few distinct values, and named
    separately because the intent is different: a sort promises an order and this only promises
    that a group holds few distinct values, which is all the pruning needs.
    """
    if name not in batch.schema:
        raise ConfigError(f"{name} is not a column of {list(batch.schema.names)}")
    keys = batch.column(name)
    values = keys.values
    order = np.argsort(values, kind="stable")
    return Layout(groups=tuple(cut(batch.take(order), group_size)), order="clustered", key=name)


def interleaved(batch: Batch, names: Sequence[str], group_size: int = GROUP_SIZE) -> Layout:
    """The rows ordered by several columns at once, each contributing equally.

    A z order, built by interleaving the bits of the columns' ranks rather than of their values,
    so a column of large integers does not dominate a column of small ones. It is the
    arrangement for a table queried by several columns with no one of them favoured, and the
    measurement below is whether it delivers that or just does two things badly.
    """
    if not names:
        raise ConfigError("interleaving needs at least one column")
    missing = [one for one in names if one not in batch.schema]
    if missing:
        raise ConfigError(f"{missing} are not columns of {list(batch.schema.names)}")
    ranks = [_ranked(batch.column(one)) for one in names]
    keys = _interleave(ranks)
    order = np.argsort(keys, kind="stable")
    return Layout(
        groups=tuple(cut(batch.take(order), group_size)),
        order="interleaved",
        key=",".join(names),
    )


def _ranked(one: Column) -> np.ndarray:
    """A column's values as their positions in sorted order.

    Ranks rather than values, because interleaving the bits of the values themselves gives every
    bit of a large column priority over every bit of a small one, and the result is a z order
    that is really an order by the largest column.
    """
    order = np.argsort(one.values, kind="stable")
    out = np.empty(len(order), dtype=np.int64)
    out[order] = np.arange(len(order))
    return out


def _interleave(ranks: Sequence[np.ndarray]) -> np.ndarray:
    """One integer per row, with the bits of each rank interleaved.

    Sixteen bits per column, which is enough for sixty five thousand rows and keeps the result
    inside an int64 for up to four columns. Beyond that the low bits fall off the end, which
    degrades the ordering rather than breaking it, and the guard says so.
    """
    if len(ranks) > 4:
        raise ConfigError(f"{len(ranks)} columns will not fit in an interleaved key")
    bits = 16
    out = np.zeros(len(ranks[0]), dtype=np.int64)
    for position in range(bits):
        for index, one in enumerate(ranks):
            out |= ((one >> position) & 1) << (position * len(ranks) + index)
    return out


@dataclass(frozen=True)
class Pruning:
    """What a layout let a predicate skip."""

    groups: int
    read: int
    rows_read: int
    rows_kept: int

    @property
    def skipped(self) -> int:
        """Groups the predicate could not be true in."""
        return self.groups - self.read

    @property
    def share(self) -> float:
        """The share of groups skipped."""
        return self.skipped / max(self.groups, 1)

    @property
    def waste(self) -> float:
        """Rows read that the predicate then rejected, as a share of what was read."""
        return (self.rows_read - self.rows_kept) / max(self.rows_read, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "groups": self.groups,
            "read": self.read,
            "skipped": self.skipped,
            "share": round(self.share, 3),
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "waste": round(self.waste, 3),
        }


def prune(layout: Layout, predicate: Expr) -> Pruning:
    """How much of a layout a predicate has to read.

    Uses the same zone maps storage/statistics.py builds, so this measures the layout rather
    than a second pruning implementation. A group is read when its statistics do not rule the
    predicate out, and the rows kept are counted by actually evaluating it, which is what makes
    the waste number real rather than estimated.
    """
    read = 0
    rows_read = 0
    rows_kept = 0
    for one in layout.groups:
        stats = collect(one)
        if not can_skip(stats, predicate):
            read += 1
            rows_read += one.rows
            rows_kept += apply_predicate(predicate, one).rows
    return Pruning(
        groups=len(layout.groups), read=read, rows_read=rows_read, rows_kept=rows_kept
    )


def _table(rows: int = 40000, seed: int = 19) -> Batch:
    """A table with columns of three different shapes, to arrange three ways.

    One column that rises with the row number, one that is random, and one low cardinality
    string. That is enough for a layout to favour one and hurt another, which is the thing being
    measured.
    """
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            integer_column("stamp", np.arange(rows)),
            integer_column("shop", state.integers(0, 40, rows)),
            floating_column("amount", state.normal(100, 30, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 6, rows)]),
        ]
    )


def arrival_order_prunes_a_rising_column(rows: int = 40000) -> dict:
    """A column that rises with the row number is already sorted, so arrival order prunes it.

    The case worth stating first because it is the one that catches people out: a timestamp in
    an event stream needs no arrangement at all. Sorting by it would produce the same file.
    """
    batch = _table(rows)
    layout = as_they_arrive(batch)
    predicate = Compare("<", column("stamp"), literal(rows // 20))
    pruned = prune(layout, predicate)
    return {
        "groups": len(layout.groups),
        **pruned.as_dict(),
        "it_pruned_nearly_everything": pruned.share > 0.9,
        "and_wasted_almost_nothing": pruned.waste < 0.05,
    }


def arrival_order_prunes_nothing_else(rows: int = 40000) -> dict:
    """And the same layout prunes nothing at all on a column that is not correlated with it.

    Every group's minimum and maximum bracket the same range, so no group can be ruled out. This
    is the failure a layout exists to fix.
    """
    batch = _table(rows)
    layout = as_they_arrive(batch)
    predicate = Compare("<", column("amount"), literal(40.0))
    pruned = prune(layout, predicate)
    return {
        "groups": len(layout.groups),
        **pruned.as_dict(),
        "it_pruned_nothing": pruned.share == 0,
        "and_wasted_nearly_everything": pruned.waste > 0.9,
    }


def sorting_by_a_column_makes_its_zone_map_exact(rows: int = 40000) -> dict:
    """Sorted by amount, a predicate on amount reads only the groups that can hold it.

    Exact in the sense that a group is read only if the range it holds overlaps the predicate,
    and after sorting the ranges do not overlap each other, so the number of groups read is the
    smallest any zone map could achieve.
    """
    batch = _table(rows)
    layout = sorted_by(batch, "amount")
    predicate = Compare("<", column("amount"), literal(40.0))
    pruned = prune(layout, predicate)
    arrival = prune(as_they_arrive(batch), predicate)
    return {
        "groups": len(layout.groups),
        **pruned.as_dict(),
        "arrival_read": arrival.read,
        "sorted_read": pruned.read,
        "ratio": round(arrival.rows_read / max(pruned.rows_read, 1), 2),
        "the_waste_is_one_group_at_the_boundary": pruned.waste < 2 / max(pruned.read, 1),
    }


def sorting_by_one_column_costs_the_others(rows: int = 40000) -> dict:
    """The half of the trade that gets left out of the sales pitch.

    Sorting by amount destroys the arrival order, so the timestamp column that pruned perfectly
    before now prunes nothing. There is no arrangement that is good for every column, and this
    measures what the good one costs the rest.
    """
    batch = _table(rows)
    stamps = Compare("<", column("stamp"), literal(rows // 20))
    arrival = prune(as_they_arrive(batch), stamps)
    ordered = prune(sorted_by(batch, "amount"), stamps)
    return {
        "arrival_skipped": arrival.skipped,
        "sorted_skipped": ordered.skipped,
        "arrival_rows_read": arrival.rows_read,
        "sorted_rows_read": ordered.rows_read,
        "the_sort_cost_the_stamp_column": ordered.rows_read > arrival.rows_read,
        "it_now_reads_this_many_times_more": round(
            ordered.rows_read / max(arrival.rows_read, 1), 1
        ),
    }


def clustering_by_a_low_cardinality_column_prunes_equalities(rows: int = 40000) -> dict:
    """Six regions over eighty groups, clustered, so each region occupies its own groups.

    The arrangement for a column with few distinct values, and the one that makes a bloom filter
    unnecessary on that column: after clustering, the zone map alone answers the equality.
    """
    batch = _table(rows)
    layout = clustered_by(batch, "region")
    predicate = Compare("=", column("region"), literal("region3"))
    clustered = prune(layout, predicate)
    arrival = prune(as_they_arrive(batch), predicate)
    return {
        "groups": len(layout.groups),
        "clustered_read": clustered.read,
        "arrival_read": arrival.read,
        "clustered_waste": round(clustered.waste, 3),
        "arrival_waste": round(arrival.waste, 3),
        "clustering_prunes": clustered.read < arrival.read,
        "arrival_does_not": arrival.read == len(layout.groups),
    }


def a_bloom_filter_is_redundant_after_clustering(rows: int = 40000) -> dict:
    """The same equality answered by a zone map on a clustered column and by a bloom filter.

    Worth measuring because the two structures are usually presented as alternatives and here
    one of them makes the other pointless. After clustering, the zone map skips exactly the
    groups the bloom filter would, and costs sixteen bytes against several hundred.
    """
    batch = _table(rows)
    layout = clustered_by(batch, "region")
    predicate = Compare("=", column("region"), literal("region3"))
    by_zone = prune(layout, predicate)
    filters = [build_for(one.column("region")) for one in layout.groups]
    by_bloom = sum(1 for one in filters if one.might_contain("region3"))
    return {
        "groups": len(layout.groups),
        "zone_read": by_zone.read,
        "bloom_read": by_bloom,
        "they_agree": abs(by_zone.read - by_bloom) <= 1,
        "bloom_bytes": sum(np.packbits(one.bits).nbytes for one in filters),
        "zone_bytes": len(layout.groups) * 16,
    }


def clustering_helps_the_encodings_too(rows: int = 40000) -> dict:
    """A clustered column runs, so run length encoding suddenly works on it.

    A second effect of the same arrangement, and the one that shows up on the file size rather
    than on the read. Six regions in random order have runs of length one; clustered they have
    runs of thousands.
    """
    batch = _table(rows)
    arrival = as_they_arrive(batch).flatten().column("region")
    clustered = clustered_by(batch, "region").flatten().column("region")
    loose = runlength.encode(arrival.values)
    tight = runlength.encode(clustered.values)
    return {
        "arrival_runs": len(loose.values),
        "clustered_runs": len(tight.values),
        "arrival_bytes": loose.nbytes(),
        "clustered_bytes": tight.nbytes(),
        "ratio": round(loose.nbytes() / max(tight.nbytes(), 1), 1),
        "it_helped": tight.nbytes() < loose.nbytes(),
    }


def sorting_does_not_help_the_dictionary(rows: int = 40000) -> dict:
    """And a third effect that does not happen, which is worth recording as well.

    A dictionary encoded column stores one code per row and one copy of each entry. Reordering
    the rows changes neither count, so the dictionary is exactly the same size whatever the
    layout. Sorting helps run length and bit packing and does nothing here.
    """
    batch = _table(rows)
    arrival = as_they_arrive(batch).flatten().column("region")
    clustered = clustered_by(batch, "region").flatten().column("region")
    loose = dictionary.encode(arrival.to_list())
    tight = dictionary.encode(clustered.to_list())
    return {
        "arrival_entries": len(loose.dictionary),
        "clustered_entries": len(tight.dictionary),
        "arrival_bytes": loose.encoded_bytes,
        "clustered_bytes": tight.encoded_bytes,
        "they_are_the_same": loose.encoded_bytes == tight.encoded_bytes,
    }


def interleaving_favours_neither_column(rows: int = 40000) -> dict:
    """A z order over two columns, against sorting by one of them.

    The claim is that interleaving is worse than a sort for the sorted column and better for the
    other one, so a table queried on both is better off interleaved. Measured on both predicates
    rather than on the flattering one.
    """
    batch = _table(rows)
    on_shop = Compare("<", column("shop"), literal(4))
    on_amount = Compare("<", column("amount"), literal(40.0))
    by_shop = sorted_by(batch, "shop")
    both = interleaved(batch, ["shop", "amount"])
    return {
        "sorted_by_shop": {
            "shop_read": prune(by_shop, on_shop).read,
            "amount_read": prune(by_shop, on_amount).read,
        },
        "interleaved": {
            "shop_read": prune(both, on_shop).read,
            "amount_read": prune(both, on_amount).read,
        },
        "the_sort_is_better_on_its_own_column": prune(by_shop, on_shop).read
        <= prune(both, on_shop).read,
        "the_interleave_is_better_on_the_other": prune(both, on_amount).read
        < prune(by_shop, on_amount).read,
    }


def interleaving_ranks_rather_than_values(rows: int = 20000) -> dict:
    """Why the interleave is over ranks and not over the values themselves.

    Interleaving raw bits gives every bit of a wide column priority over every bit of a narrow
    one, so a z order over a timestamp and a small integer is really an order by the timestamp.
    Ranks put both columns on the same scale by construction.
    """
    batch = _table(rows)
    wide = batch.column("stamp").values
    narrow = batch.column("shop").values
    raw = np.zeros(rows, dtype=np.int64)
    for position in range(16):
        raw |= ((wide >> position) & 1) << (position * 2)
        raw |= ((narrow >> position) & 1) << (position * 2 + 1)
    on_shop = Compare("<", column("shop"), literal(4))
    ranked = interleaved(batch, ["stamp", "shop"])
    naive = Layout(groups=tuple(cut(batch.take(np.argsort(raw, kind="stable")))), order="raw")
    return {
        "ranked_read": prune(ranked, on_shop).read,
        "raw_read": prune(naive, on_shop).read,
        "groups": len(ranked.groups),
        "ranks_prune_at_least_as_well": prune(ranked, on_shop).read
        <= prune(naive, on_shop).read,
    }


def a_smaller_group_prunes_more_and_costs_more(rows: int = 40000) -> dict:
    """The group size sweep, from the layout's side rather than the file format's.

    Smaller groups prune finer and cost more metadata. storage/file.py measured the metadata and
    this measures the pruning, and the two together are why five hundred is the default.
    """
    batch = _table(rows)
    predicate = Compare("<", column("amount"), literal(40.0))
    out = []
    for size in (100, 500, 2000, 10000):
        layout = sorted_by(batch, "amount", group_size=size)
        pruned = prune(layout, predicate)
        out.append(
            {
                "group_size": size,
                "groups": len(layout.groups),
                "read": pruned.read,
                "rows_read": pruned.rows_read,
                "waste": round(pruned.waste, 3),
            }
        )
    wastes = [one["waste"] for one in out]
    return {
        "sweep": out,
        "smaller_groups_waste_less": wastes == sorted(wastes),
        "and_there_are_more_of_them": out[0]["groups"] > out[-1]["groups"],
    }


def a_layout_flattens_back_to_its_table(rows: int = 4000) -> dict:
    """Cutting a table into groups and concatenating them gives the rows back.

    For the arrival layout it gives the same table; for a sorted one it gives a permutation, and
    the check is on the multiset rather than on the order.
    """
    batch = _table(rows)
    arrival = as_they_arrive(batch).flatten()
    ordered = sorted_by(batch, "amount").flatten()
    return {
        "rows": arrival.rows,
        "arrival_is_identical": bool(
            np.array_equal(arrival.column("stamp").values, batch.column("stamp").values)
        ),
        "sorted_is_a_permutation": sorted(ordered.column("stamp").values.tolist())
        == sorted(batch.column("stamp").values.tolist()),
        "and_the_sort_key_is_ordered": bool(
            np.all(np.diff(ordered.column("amount").values) >= 0)
        ),
    }


def the_last_group_is_short(rows: int = 4050, group_size: int = 500) -> dict:
    """A table that does not divide evenly, which is most of them."""
    layout = as_they_arrive(_table(rows), group_size)
    sizes = [one.rows for one in layout.groups]
    return {
        "groups": len(sizes),
        "sizes": sizes[-3:],
        "the_last_is_short": sizes[-1] < group_size,
        "the_rest_are_full": all(one == group_size for one in sizes[:-1]),
        "they_sum_to_the_table": sum(sizes) == rows,
    }


def a_zero_group_size_is_refused() -> bool:
    """A group of no rows."""
    try:
        cut(_table(100), group_size=0)
    except ConfigError:
        return True
    return False


def sorting_by_a_missing_column_is_refused() -> bool:
    """A layout key that is not a column."""
    try:
        sorted_by(_table(100), "nothing")
    except ConfigError:
        return True
    return False


def interleaving_nothing_is_refused() -> bool:
    """A z order over no columns."""
    try:
        interleaved(_table(100), [])
    except ConfigError:
        return True
    return False


def interleaving_five_columns_is_refused() -> bool:
    """More columns than fit in the key, refused rather than silently degraded."""
    batch = _table(100)
    try:
        interleaved(batch, ["stamp", "shop", "amount", "region", "stamp"])
    except ConfigError:
        return True
    return False


def flattening_nothing_is_refused() -> bool:
    """A layout with no groups in it."""
    try:
        Layout(groups=(), order="arrival").flatten()
    except ConfigError:
        return True
    return False


def compare_the_layouts(rows: int = 40000) -> list[dict]:
    """Every arrangement against every predicate, which is the module in one table.

    The table nobody can read off a single measurement: each row is a layout, each column a
    predicate, and the point is that no layout wins every column.
    """
    batch = _table(rows)
    predicates = {
        "stamp": Compare("<", column("stamp"), literal(rows // 20)),
        "amount": Compare("<", column("amount"), literal(40.0)),
        "region": Compare("=", column("region"), literal("region3")),
    }
    layouts = {
        "arrival": as_they_arrive(batch),
        "sorted by amount": sorted_by(batch, "amount"),
        "clustered by region": clustered_by(batch, "region"),
        "interleaved": interleaved(batch, ["shop", "amount"]),
    }
    return [
        {
            "layout": name,
            **{
                f"{which}_read": prune(layout, predicate).read
                for which, predicate in predicates.items()
            },
        }
        for name, layout in layouts.items()
    ]


def no_layout_wins_every_column(rows: int = 40000) -> dict:
    """The point of the table above, as one claim.

    If some arrangement read the fewest groups for every predicate there would be nothing to
    decide and this module would be one function. There is not one, and the measurement says so
    rather than the prose.
    """
    table = compare_the_layouts(rows)
    columns = [one for one in table[0] if one.endswith("_read")]
    winners = {one: min(table, key=lambda row, key=one: row[key])["layout"] for one in columns}
    return {
        "table": table,
        "winners": winners,
        "they_are_not_all_the_same": len(set(winners.values())) > 1,
        "distinct_winners": len(set(winners.values())),
    }


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "layouts": 4,
        "group_size": GROUP_SIZE,
        "arrival_prunes_the_rising_column": arrival_order_prunes_a_rising_column()[
            "it_pruned_nearly_everything"
        ],
        "and_nothing_else": arrival_order_prunes_nothing_else()["it_pruned_nothing"],
        "no_layout_wins_everything": no_layout_wins_every_column()["they_are_not_all_the_same"],
    }
