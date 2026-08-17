from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cqe.cost.meter import Meter
from cqe.errors import ConfigError, DataError, UnknownColumn
from cqe.exec.batch import Batch, side_by_side
from cqe.types.schema import INTEGER

# Joins, in three implementations that have to agree.
#
# A nested loop join compares every left row against every right row. It is the definition of a
# join, it costs the product of the two sizes, and it is here as the thing the other two are
# checked against rather than as something to run.
#
# A hash join builds a table from one side and probes it with the other. It costs the sum of the
# two sizes plus one probe per probing row, which is the whole of why joins are practical.
#
# A sort merge join sorts both sides and walks them together. It costs two sorts and one pass,
# and it is here because it needs no hash table, because it produces its output in key order for
# free, and because it is what a spilling join degrades into when the build side does not fit.
#
# The decision that matters in a hash join is which side to build from. The usual rule is to
# build from the smaller side and the usual reason given is memory, and the measurement below
# says memory really is the whole reason. Building from the small side does 2000 inserts and
# 20000 probes; building from the large side does 20000 inserts and 2000 probes. Twenty two
# thousand operations either way, and twenty thousand rows resident against two thousand.
#
# The meter reports the two at 104000 values against 122000, which looks like a difference in
# work and is not: an insert is charged as a value touched and a probe is counted separately, so
# the totals are not comparable. Worth writing down, because a cost model reading values touched
# alone would reach the right answer for a reason that does not hold.
#
# Null keys never match anything, including other null keys. That is the opposite of the group
# by rule in exec/aggregate.py, where two nulls are the same group. Both are correct because
# they answer different questions, and having the two rules in two modules with two names is the
# only way anybody remembers which is which.


@dataclass
class Joined:
    """The result of a join and what producing it cost."""

    batch: Batch
    strategy: str
    build_rows: int
    probe_rows: int
    probes: int
    matches: int

    @property
    def fanout(self) -> float:
        """Output rows per probe row, which is what a cost model has to predict."""
        if self.probe_rows == 0:
            return 0.0
        return self.matches / self.probe_rows

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "strategy": self.strategy,
            "build_rows": self.build_rows,
            "probe_rows": self.probe_rows,
            "probes": self.probes,
            "matches": self.matches,
            "fanout": round(self.fanout, 4),
            "rows": self.batch.rows,
        }


def _key_values(batch: Batch, keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """A single integer key per row, and a mask saying which rows have a usable key.

    Composite keys are folded into one integer by ranking each column and combining, which is
    the same trick exec/aggregate.py uses. A row with a null in any key column is unusable, and
    the mask carries that so the caller can drop it before matching rather than after.
    """
    if not keys:
        raise ConfigError("a join needs at least one key")
    missing = [name for name in keys if name not in batch.names]
    if missing:
        raise UnknownColumn(f"{missing} not in {list(batch.names)}")
    usable = np.ones(batch.rows, dtype=bool)
    combined = np.zeros(batch.rows, dtype=np.int64)
    for name in keys:
        column = batch.column(name)
        if column.valid is not None:
            usable &= column.valid
        unique, positions = np.unique(column.values, return_inverse=True)
        combined = combined * max(len(unique), 1) + positions
    return combined, usable


def _shared_codes(
    left: Batch,
    left_keys: Sequence[str],
    right: Batch,
    right_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Key codes for both sides drawn from one shared ranking.

    Ranking the two sides separately would give the same value different codes on each side,
    which is the mistake that makes a join silently return nothing. The columns are stacked,
    ranked once, and split again.
    """
    if not left_keys:
        raise ConfigError("a join needs at least one key")
    if len(left_keys) != len(right_keys):
        raise ConfigError(f"{len(left_keys)} keys against {len(right_keys)}")
    left_usable = np.ones(left.rows, dtype=bool)
    right_usable = np.ones(right.rows, dtype=bool)
    left_code = np.zeros(left.rows, dtype=np.int64)
    right_code = np.zeros(right.rows, dtype=np.int64)
    for left_name, right_name in zip(left_keys, right_keys, strict=True):
        left_column = left.column(left_name)
        right_column = right.column(right_name)
        if left_column.logical != right_column.logical:
            raise DataError(
                f"{left_name} is {left_column.logical} against "
                f"{right_name} as {right_column.logical}"
            )
        if left_column.valid is not None:
            left_usable &= left_column.valid
        if right_column.valid is not None:
            right_usable &= right_column.valid
        left_side = _comparable(left_column)
        right_side = _comparable(right_column)
        stacked = np.concatenate([left_side, right_side])
        unique, positions = np.unique(stacked, return_inverse=True)
        width = max(len(unique), 1)
        left_code = left_code * width + positions[: len(left_side)]
        right_code = right_code * width + positions[len(left_side) :]
    return left_code, left_usable, right_code, right_usable


def _comparable(column) -> np.ndarray:
    """The values to match on, with a string column's codes replaced by its text.

    Two string columns from different batches have different dictionaries, so their codes are
    not comparable. Matching on the text is correct and slower, and doing it here rather than at
    every call site is what keeps the mistake from spreading.
    """
    if column.dictionary is not None:
        entries = np.array(column.dictionary, dtype=object)
        return entries[column.values] if len(column) else np.array([], dtype=object)
    return column.values


def hash_join(
    left: Batch,
    right: Batch,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    meter: Meter | None = None,
    build_side: str = "right",
) -> Joined:
    """Build a table from one side and probe it with the other.

    The build side is a parameter rather than a decision, so the module can measure what
    choosing it wrongly costs instead of asserting a rule about it.
    """
    if build_side not in ("left", "right"):
        raise ConfigError(f"{build_side} is not a side")
    left_code, left_ok, right_code, right_ok = _shared_codes(left, left_keys, right, right_keys)
    if meter is not None:
        meter.touch((left.rows + right.rows) * len(left_keys), "join_key")

    build_codes, build_ok = (
        (right_code, right_ok) if build_side == "right" else (left_code, left_ok)
    )
    probe_codes, probe_ok = (
        (left_code, left_ok) if build_side == "right" else (right_code, right_ok)
    )
    table: dict[int, list[int]] = {}
    for position in np.flatnonzero(build_ok):
        table.setdefault(int(build_codes[position]), []).append(int(position))
    if meter is not None:
        meter.touch(int(build_ok.sum()), "join_build")
        meter.probe(int(probe_ok.sum()))

    probe_rows: list[int] = []
    build_rows: list[int] = []
    for position in np.flatnonzero(probe_ok):
        for other in table.get(int(probe_codes[position]), ()):
            probe_rows.append(int(position))
            build_rows.append(other)

    if build_side == "right":
        left_positions, right_positions = probe_rows, build_rows
    else:
        left_positions, right_positions = build_rows, probe_rows
    return _assemble(
        left,
        right,
        left_positions,
        right_positions,
        strategy=f"hash build {build_side}",
        build_rows=int(build_ok.sum()),
        probe_rows=int(probe_ok.sum()),
        probes=int(probe_ok.sum()),
        meter=meter,
    )


def merge_join(
    left: Batch,
    right: Batch,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    meter: Meter | None = None,
) -> Joined:
    """Sort both sides on the key and walk them together.

    No hash table, and the output comes out in key order, which is worth something when the
    query sorts afterwards. The cost counted is the two sorts plus one pass, and the sorts are
    counted as values touched rather than modelled comparisons, so this number is comparable
    with the hash join's directly.
    """
    left_code, left_ok, right_code, right_ok = _shared_codes(left, left_keys, right, right_keys)
    if meter is not None:
        meter.touch((left.rows + right.rows) * len(left_keys), "join_key")
        meter.touch(left.rows + right.rows, "join_sort")

    left_positions_all = np.flatnonzero(left_ok)
    right_positions_all = np.flatnonzero(right_ok)
    left_order = left_positions_all[np.argsort(left_code[left_positions_all], kind="stable")]
    right_order = right_positions_all[
        np.argsort(right_code[right_positions_all], kind="stable")
    ]

    left_positions: list[int] = []
    right_positions: list[int] = []
    one = 0
    other = 0
    while one < len(left_order) and other < len(right_order):
        left_key = left_code[left_order[one]]
        right_key = right_code[right_order[other]]
        if left_key < right_key:
            one += 1
        elif left_key > right_key:
            other += 1
        else:
            one_end = one
            while one_end < len(left_order) and left_code[left_order[one_end]] == left_key:
                one_end += 1
            other_end = other
            while (
                other_end < len(right_order) and right_code[right_order[other_end]] == right_key
            ):
                other_end += 1
            for a in left_order[one:one_end]:
                for b in right_order[other:other_end]:
                    left_positions.append(int(a))
                    right_positions.append(int(b))
            one = one_end
            other = other_end
    return _assemble(
        left,
        right,
        left_positions,
        right_positions,
        strategy="merge",
        build_rows=int(right_ok.sum()),
        probe_rows=int(left_ok.sum()),
        probes=0,
        meter=meter,
    )


def nested_loop_join(
    left: Batch,
    right: Batch,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    meter: Meter | None = None,
) -> Joined:
    """Compare every left row against every right row.

    The definition of an equi join and the reference the other two are checked against. Costs
    the product of the two sizes and is never the right choice above a few thousand rows on each
    side, which is why the module refuses to run it on anything larger.
    """
    if left.rows * right.rows > 4_000_000:
        raise ConfigError(
            f"a nested loop over {left.rows} by {right.rows} is not worth running"
        )
    left_code, left_ok, right_code, right_ok = _shared_codes(left, left_keys, right, right_keys)
    left_positions: list[int] = []
    right_positions: list[int] = []
    for one in range(left.rows):
        if not left_ok[one]:
            continue
        for other in range(right.rows):
            if not right_ok[other]:
                continue
            if left_code[one] == right_code[other]:
                left_positions.append(one)
                right_positions.append(other)
    if meter is not None:
        meter.touch(left.rows * right.rows, "nested_loop")
        meter.compare(left.rows * right.rows)
    return _assemble(
        left,
        right,
        left_positions,
        right_positions,
        strategy="nested loop",
        build_rows=right.rows,
        probe_rows=left.rows,
        probes=0,
        meter=meter,
    )


def _assemble(
    left: Batch,
    right: Batch,
    left_positions: Sequence[int],
    right_positions: Sequence[int],
    strategy: str,
    build_rows: int,
    probe_rows: int,
    probes: int,
    meter: Meter | None,
) -> Joined:
    """Gather both sides at the matched positions and place them side by side."""
    left_taken = left.take(np.array(left_positions, dtype=np.int64), meter=meter)
    right_taken = right.take(np.array(right_positions, dtype=np.int64), meter=meter)
    return Joined(
        batch=side_by_side(left_taken, right_taken),
        strategy=strategy,
        build_rows=build_rows,
        probe_rows=probe_rows,
        probes=probes,
        matches=len(left_positions),
    )


def _pair(
    left_rows: int = 20_000,
    right_rows: int = 2_000,
    keys: int = 2_000,
    seed: int = 0,
) -> tuple[Batch, Batch]:
    """A large fact side and a small dimension side, which is the shape of most joins."""
    if left_rows < 1 or right_rows < 1 or keys < 1:
        raise ConfigError("that is not a pair of tables")
    generator = np.random.default_rng(seed)
    fact = Batch.of(
        k=generator.integers(0, keys, size=left_rows).tolist(),
        v=generator.integers(0, 1000, size=left_rows).tolist(),
    )
    dimension = Batch.of(
        k=list(range(min(right_rows, keys))),
        label=[f"d{position:05d}" for position in range(min(right_rows, keys))],
    )
    return fact, dimension


def the_three_strategies_agree(left_rows: int = 2_000, right_rows: int = 500) -> dict:
    """All three joins return the same rows, which is what lets them be compared at all.

    Compared as sets rather than in order, because only the merge join promises an order and
    asserting one on the others would be asserting an implementation detail.
    """
    fact, dimension = _pair(left_rows=left_rows, right_rows=right_rows, keys=right_rows)
    hashed = hash_join(fact, dimension, ["k"], ["k"])
    merged = merge_join(fact, dimension, ["k"], ["k"])
    looped = nested_loop_join(fact, dimension, ["k"], ["k"])

    def rows(result: Joined) -> list:
        return sorted(result.batch.to_rows(), key=str)

    return {
        "matches": hashed.matches,
        "hash_matches_merge": rows(hashed) == rows(merged),
        "hash_matches_nested_loop": rows(hashed) == rows(looped),
        "all_found_the_same_count": hashed.matches == merged.matches == looped.matches,
    }


def they_agree_with_the_reference(left_rows: int = 800, right_rows: int = 200) -> dict:
    """The vectorised joins against the row at a time interpreter, with null keys present.

    The null rule is the point of this one. A null key matches nothing, including another null,
    and the reference implements that separately from the fast path so a shared misunderstanding
    cannot hide.
    """
    from cqe.verify import reference  # noqa: PLC0415

    fact, dimension = _pair(left_rows=left_rows, right_rows=right_rows, keys=right_rows)
    holes = [
        None if position % 17 == 0 else value
        for position, value in enumerate(fact.column("k").to_list())
    ]
    from cqe.columns.array import column_from  # noqa: PLC0415

    fact = fact.with_column(column_from("k", holes))
    fast = hash_join(fact, dimension, ["k"], ["k"])
    slow = reference.inner_join(
        reference.Rows.of(fact), reference.Rows.of(dimension), ["k"], ["k"]
    )
    agreement = reference.agree(reference.Rows.of(fast.batch), slow)
    return {
        "matches": fast.matches,
        "null_keys": fact.column("k").null_count,
        "same": agreement.same,
        "differences": len(agreement.differences),
        "nulls_matched_nothing": fast.matches < fact.rows,
    }


def the_build_side_trades_probes_for_inserts(
    left_rows: int = 20_000,
    right_rows: int = 2_000,
) -> dict:
    """Which side the table is built from, measured rather than argued.

    I expected the two to be identical in total work and they are, once the units are added up
    properly. Building from the small side does 2000 inserts and 20000 probes; building from the
    large side does 20000 inserts and 2000 probes. Twenty two thousand operations either way.

    The meter makes them look different, 104000 values touched against 122000, because an insert
    is charged as a value touched and a probe is counted separately. That is an artefact of the
    accounting rather than a fact about the join, and it is worth writing down because a cost
    model built on values touched alone would prefer the small build side for the wrong reason.

    The right reason is the last two numbers: building from the large side holds twenty thousand
    rows resident against two thousand, and the probe side streams. Memory is the whole
    argument.
    """
    fact, dimension = _pair(left_rows=left_rows, right_rows=right_rows, keys=right_rows)
    right_meter = Meter()
    from_right = hash_join(fact, dimension, ["k"], ["k"], right_meter, build_side="right")
    left_meter = Meter()
    from_left = hash_join(fact, dimension, ["k"], ["k"], left_meter, build_side="left")
    return {
        "build_right_values": right_meter.values_touched,
        "build_left_values": left_meter.values_touched,
        "build_right_probes": right_meter.hash_probes,
        "build_left_probes": left_meter.hash_probes,
        "same_matches": from_right.matches == from_left.matches,
        "right_operations": from_right.build_rows + right_meter.hash_probes,
        "left_operations": from_left.build_rows + left_meter.hash_probes,
        "the_total_operations_match": (
            from_right.build_rows + right_meter.hash_probes
            == from_left.build_rows + left_meter.hash_probes
        ),
        "resident_rows_right": from_right.build_rows,
        "resident_rows_left": from_left.build_rows,
        "building_from_the_small_side_holds_less": (
            from_right.build_rows < from_left.build_rows
        ),
    }


def the_hash_join_beats_the_nested_loop_by_the_build_size(
    left_rows: int = 2_000,
    right_sizes: Sequence[int] = (50, 200, 800, 2_000),
) -> list[dict]:
    """How the two costs separate as the build side grows.

    The nested loop is the product of the sizes and the hash join is the sum, so the ratio is
    roughly the smaller side. That is the entire argument for hash joins stated as a measurement
    rather than as an asymptotic.
    """
    if not right_sizes:
        raise ConfigError("there is nothing to sweep")
    out = []
    for size in right_sizes:
        fact, dimension = _pair(left_rows=left_rows, right_rows=size, keys=size)
        hashed_meter = Meter()
        hash_join(fact, dimension, ["k"], ["k"], hashed_meter)
        looped_meter = Meter()
        nested_loop_join(fact, dimension, ["k"], ["k"], looped_meter)
        out.append(
            {
                "right_rows": size,
                "hash_values": hashed_meter.values_touched,
                "nested_values": looped_meter.values_touched,
                "ratio": round(
                    looped_meter.values_touched / max(hashed_meter.values_touched, 1), 1
                ),
            }
        )
    return out


def a_null_key_matches_nothing(rows: int = 1_000) -> dict:
    """Including another null, which is where this differs from a group by.

    Two tables whose key columns are entirely null. A group by would put every row in one group.
    A join produces no rows at all, and the module measures it rather than asserting it because
    the two rules are one line apart in the code and easy to swap.
    """
    from cqe.columns.array import column_from  # noqa: PLC0415

    blanks = column_from("k", [None] * rows, logical=INTEGER)
    left = Batch.of(a=list(range(rows))).with_column(blanks)
    right = Batch.of(b=list(range(rows))).with_column(blanks)
    result = hash_join(left, right, ["k"], ["k"])
    return {
        "rows_each_side": rows,
        "matches": result.matches,
        "nothing_matched": result.matches == 0,
        "a_group_by_would_have_made_one_group": True,
    }


def a_composite_key_joins_on_both_columns(rows: int = 2_000) -> dict:
    """Two key columns, ranked together so a value means the same on both sides.

    The failure this guards is ranking each side separately, which gives the same value
    different codes and produces a join that returns nothing at all. Checked against the single
    column join on a key built so that the composite is strictly more selective.
    """
    generator = np.random.default_rng(5)
    left = Batch.of(
        a=generator.integers(0, 10, size=rows).tolist(),
        b=generator.integers(0, 10, size=rows).tolist(),
        v=list(range(rows)),
    )
    right = Batch.of(
        a=[position // 10 for position in range(100)],
        b=[position % 10 for position in range(100)],
        label=[f"p{position:03d}" for position in range(100)],
    )
    both = hash_join(left, right, ["a", "b"], ["a", "b"])
    one = hash_join(left, right, ["a"], ["a"])
    return {
        "composite_matches": both.matches,
        "single_matches": one.matches,
        "the_composite_is_more_selective": both.matches < one.matches,
        "every_left_row_matched_once": both.matches == rows,
    }


def string_keys_join_across_dictionaries(rows: int = 2_000) -> dict:
    """Two string columns from different batches, whose codes are not comparable.

    Each batch built its own dictionary, so the code for a value on one side has nothing to do
    with the code on the other. Matching on codes would return nothing; matching on the text is
    correct. The check is that the join finds the rows a value based comparison finds.
    """
    generator = np.random.default_rng(6)
    labels = [f"k{int(value):03d}" for value in generator.integers(0, 100, size=rows)]
    left = Batch.of(k=labels, v=list(range(rows)))
    right = Batch.of(k=[f"k{position:03d}" for position in range(0, 100, 2)], w=list(range(50)))
    result = hash_join(left, right, ["k"], ["k"])
    expected = sum(1 for label in labels if int(label[1:]) % 2 == 0)
    return {
        "matches": result.matches,
        "expected": expected,
        "it_matched_on_text": result.matches == expected,
        "the_dictionaries_differ": left.column("k").dictionary != right.column("k").dictionary,
    }


def the_fanout_is_what_a_cost_model_must_predict(
    left_rows: int = 10_000,
    duplicates: Sequence[int] = (1, 2, 5, 10),
) -> list[dict]:
    """Output rows per probe row, as the right side gains duplicate keys.

    A join to a unique key produces one row in, one row out. A join to a key repeated five times
    produces five, and the output is five times the input. Estimating that badly is the single
    largest source of a plan going wrong, because everything above the join is sized on it.
    """
    if not duplicates:
        raise ConfigError("there is nothing to sweep")
    generator = np.random.default_rng(7)
    out = []
    for repeat in duplicates:
        keys = 200
        left = Batch.of(
            k=generator.integers(0, keys, size=left_rows).tolist(),
            v=list(range(left_rows)),
        )
        right = Batch.of(
            k=[position % keys for position in range(keys * repeat)],
            w=list(range(keys * repeat)),
        )
        result = hash_join(left, right, ["k"], ["k"])
        out.append(
            {
                "duplicates": repeat,
                "matches": result.matches,
                "fanout": round(result.fanout, 3),
                "output_over_input": round(result.batch.rows / left_rows, 3),
            }
        )
    return out


def an_empty_side_produces_nothing(rows: int = 100) -> dict:
    """A join against no rows, which every operator above has to survive."""
    left = Batch.of(k=list(range(rows)), v=list(range(rows)))
    right = Batch.empty(Batch.of(k=[1], w=[2]).schema)
    result = hash_join(left, right, ["k"], ["k"])
    return {
        "matches": result.matches,
        "rows": result.batch.rows,
        "it_produced_nothing": result.batch.rows == 0,
        "the_schema_survived": "w" in result.batch.names,
    }


def mismatched_key_counts_are_refused() -> bool:
    """Two keys on one side and one on the other is a mistake."""
    left, right = _pair(left_rows=10, right_rows=10, keys=10)
    try:
        hash_join(left, right, ["k", "v"], ["k"])
    except ConfigError:
        return True
    return False


def mismatched_key_types_are_refused() -> bool:
    """Joining an integer to a string says so rather than matching nothing."""
    left = Batch.of(k=[1, 2], v=[1, 2])
    right = Batch.of(k=["1", "2"], w=[1, 2])
    try:
        hash_join(left, right, ["k"], ["k"])
    except DataError:
        return True
    return False


def a_join_with_no_keys_is_refused() -> bool:
    """A cross join is a different operator and is not this one."""
    left, right = _pair(left_rows=10, right_rows=10, keys=10)
    try:
        hash_join(left, right, [], [])
    except ConfigError:
        return True
    return False


def an_oversized_nested_loop_is_refused() -> bool:
    """The reference implementation refuses to run where it would take forever."""
    left, right = _pair(left_rows=4_000, right_rows=2_000, keys=2_000)
    try:
        nested_loop_join(left, right, ["k"], ["k"])
    except ConfigError:
        return True
    return False


def an_unknown_key_is_refused() -> bool:
    """Joining on a column that is not there names the columns that are."""
    left, right = _pair(left_rows=10, right_rows=10, keys=10)
    try:
        hash_join(left, right, ["z"], ["k"])
    except UnknownColumn:
        return True
    return False


def compare_the_strategies(left_rows: int = 2_000, right_rows: int = 500) -> list[dict]:
    """The three joins side by side, which is the module in one table."""
    fact, dimension = _pair(left_rows=left_rows, right_rows=right_rows, keys=right_rows)
    out = []
    for name in ("hash", "merge", "nested loop"):
        meter = Meter()
        if name == "hash":
            result = hash_join(fact, dimension, ["k"], ["k"], meter)
        elif name == "merge":
            result = merge_join(fact, dimension, ["k"], ["k"], meter)
        else:
            result = nested_loop_join(fact, dimension, ["k"], ["k"], meter)
        row = result.as_dict()
        row["values_touched"] = meter.values_touched
        out.append(row)
    return sorted(out, key=lambda row: row["values_touched"])


def summarise(left_rows: int = 20_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    agreement = the_three_strategies_agree()
    against = they_agree_with_the_reference()
    sides = the_build_side_trades_probes_for_inserts(left_rows=left_rows)
    scaling = the_hash_join_beats_the_nested_loop_by_the_build_size()
    return {
        "strategies_agree": agreement["hash_matches_merge"]
        and agreement["hash_matches_nested_loop"],
        "agrees_with_the_reference": against["same"],
        "the_total_operations_match": sides["the_total_operations_match"],
        "widest_nested_loop_ratio": scaling[-1]["ratio"],
        "cheapest": compare_the_strategies()[0]["strategy"],
        "key_type": INTEGER,
    }
