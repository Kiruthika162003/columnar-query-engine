from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from cqe.errors import BudgetExceeded, ConfigError

# The cost unit for the whole engine, and the reason there is only one.
#
# Nothing here is timed. A wall clock measurement on a laptop with a browser open is not a
# property of the query plan, and every conclusion drawn from one has to be redrawn on the next
# machine. Instead the engine counts what it does: values touched, bytes read, rows
# materialised, hash probes, comparisons. Those are properties of the plan and the data, they
# are identical on every machine, and they are what a cost model is trying to predict anyway.
#
# The primary unit is a value touched, meaning one column value that some operator actually
# looked at. A scan of a million row table projecting two of twenty columns touches two million
# values and not twenty million, which is the entire argument for columnar storage stated as an
# arithmetic fact rather than as a claim.
#
# The secondary units exist because values touched alone hides two real effects. Bytes read
# separates a scan of an int64 column from an equivalent scan of a bit packed one, which values
# touched cannot see. Rows materialised separates a plan that carries a wide row through five
# operators from one that carries a row identifier and fetches the columns at the end, which is
# the whole of late materialisation and is invisible in every other unit.
#
# A meter is passed down a plan rather than kept globally, so two plans can be measured in the
# same process and so a test can assert on a subtree. That costs one argument on every operator
# and it is worth it: a global counter makes concurrent measurement impossible and makes every
# test order dependent.


@dataclass
class Meter:
    """What a run did, counted rather than timed."""

    values_touched: int = 0
    bytes_read: int = 0
    rows_materialised: int = 0
    hash_probes: int = 0
    comparisons: int = 0
    spilled_bytes: int = 0
    batches: int = 0
    by_operator: dict[str, int] = field(default_factory=dict)

    def touch(self, values: int, operator: str = "", width: int = 8) -> None:
        """Record values looked at, and the bytes they occupied."""
        if values < 0:
            raise ConfigError(f"{values} is not a count")
        self.values_touched += values
        self.bytes_read += values * width
        if operator:
            self.by_operator[operator] = self.by_operator.get(operator, 0) + values

    def materialise(self, rows: int) -> None:
        """Record rows assembled into an output batch."""
        if rows < 0:
            raise ConfigError(f"{rows} is not a count")
        self.rows_materialised += rows

    def probe(self, count: int = 1) -> None:
        """Record hash table lookups."""
        self.hash_probes += count

    def compare(self, count: int = 1) -> None:
        """Record value comparisons, which is what a sort spends everything on."""
        self.comparisons += count

    def spill(self, nbytes: int) -> None:
        """Record bytes written to disk because memory ran out."""
        if nbytes < 0:
            raise ConfigError(f"{nbytes} is not a size")
        self.spilled_bytes += nbytes

    def batch(self, count: int = 1) -> None:
        """Record output batches produced, which sets the per batch overhead."""
        self.batches += count

    def merge(self, other: Meter) -> None:
        """Fold another meter into this one, for a subtree measured separately."""
        self.values_touched += other.values_touched
        self.bytes_read += other.bytes_read
        self.rows_materialised += other.rows_materialised
        self.hash_probes += other.hash_probes
        self.comparisons += other.comparisons
        self.spilled_bytes += other.spilled_bytes
        self.batches += other.batches
        for name, count in other.by_operator.items():
            self.by_operator[name] = self.by_operator.get(name, 0) + count

    def copy(self) -> Meter:
        """An independent snapshot."""
        return Meter(
            values_touched=self.values_touched,
            bytes_read=self.bytes_read,
            rows_materialised=self.rows_materialised,
            hash_probes=self.hash_probes,
            comparisons=self.comparisons,
            spilled_bytes=self.spilled_bytes,
            batches=self.batches,
            by_operator=dict(self.by_operator),
        )

    def since(self, earlier: Meter) -> Meter:
        """What happened between an earlier snapshot and now."""
        return Meter(
            values_touched=self.values_touched - earlier.values_touched,
            bytes_read=self.bytes_read - earlier.bytes_read,
            rows_materialised=self.rows_materialised - earlier.rows_materialised,
            hash_probes=self.hash_probes - earlier.hash_probes,
            comparisons=self.comparisons - earlier.comparisons,
            spilled_bytes=self.spilled_bytes - earlier.spilled_bytes,
            batches=self.batches - earlier.batches,
            by_operator={
                name: count - earlier.by_operator.get(name, 0)
                for name, count in self.by_operator.items()
                if count - earlier.by_operator.get(name, 0)
            },
        )

    @property
    def dominant_operator(self) -> str:
        """Which operator touched the most values, or an empty string if nothing did."""
        if not self.by_operator:
            return ""
        return max(self.by_operator, key=lambda name: self.by_operator[name])

    @property
    def share(self) -> dict[str, float]:
        """Each operator's share of the values touched."""
        total = sum(self.by_operator.values())
        if not total:
            return {}
        return {name: count / total for name, count in self.by_operator.items()}

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "values_touched": self.values_touched,
            "bytes_read": self.bytes_read,
            "rows_materialised": self.rows_materialised,
            "hash_probes": self.hash_probes,
            "comparisons": self.comparisons,
            "spilled_bytes": self.spilled_bytes,
            "batches": self.batches,
        }

    def __str__(self) -> str:
        return (
            f"{self.values_touched} values, {self.bytes_read} bytes, "
            f"{self.rows_materialised} rows"
        )


@dataclass
class Budget:
    """A ceiling on what a run may spend, checked as it spends it.

    Separate from the meter because a budget is a policy and a meter is a fact, and a run with
    no budget still wants the fact. Every limit defaults to no limit, so a Budget with nothing
    set is the same as no budget at all and the operators do not need two code paths.
    """

    values: float = float("inf")
    memory_bytes: float = float("inf")
    spill_bytes: float = float("inf")
    rows: float = float("inf")

    def __post_init__(self) -> None:
        for name in ("values", "memory_bytes", "spill_bytes", "rows"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} budget of {getattr(self, name)} is not a budget")

    @property
    def unlimited(self) -> bool:
        """Whether this budget constrains anything at all."""
        return all(
            getattr(self, name) == float("inf")
            for name in ("values", "memory_bytes", "spill_bytes", "rows")
        )

    def check(self, meter: Meter, live_bytes: int = 0) -> None:
        """Raise if the run has gone past any limit."""
        if meter.values_touched > self.values:
            raise BudgetExceeded("values", self.values, meter.values_touched)
        if meter.rows_materialised > self.rows:
            raise BudgetExceeded("rows", self.rows, meter.rows_materialised)
        if meter.spilled_bytes > self.spill_bytes:
            raise BudgetExceeded("spill", self.spill_bytes, meter.spilled_bytes)
        if live_bytes > self.memory_bytes:
            raise BudgetExceeded("memory", self.memory_bytes, live_bytes)

    def would_exceed(self, meter: Meter, extra_values: int) -> bool:
        """Whether spending a bit more would go past the value limit.

        Used by operators that can choose a cheaper strategy before committing, which is the
        only useful thing a budget does beyond stopping a run: exec/spill.py switches algorithms
        on this rather than waiting to be killed.
        """
        return meter.values_touched + extra_values > self.values

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "values": self.values,
            "memory_bytes": self.memory_bytes,
            "spill_bytes": self.spill_bytes,
            "rows": self.rows,
        }


@contextmanager
def measured(meter: Meter) -> Iterator[Meter]:
    """Measure one region of a plan without disturbing the running total.

    Yields a fresh meter, folds it into the outer one on exit, and hands back the delta. Used
    everywhere a module wants to say what one operator cost inside a plan that is also counting.
    """
    inner = Meter()
    try:
        yield inner
    finally:
        meter.merge(inner)


def compare_meters(left: Meter, right: Meter) -> dict:
    """Two runs side by side, as ratios rather than differences.

    Ratios because the useful question is almost always how many times cheaper one plan is, and
    a difference of nine hundred thousand values means nothing without the total beside it.
    """

    def ratio(a: int, b: int) -> float:
        if b == 0:
            return float("inf") if a else 1.0
        return a / b

    return {
        "values": ratio(left.values_touched, right.values_touched),
        "bytes": ratio(left.bytes_read, right.bytes_read),
        "rows": ratio(left.rows_materialised, right.rows_materialised),
        "comparisons": ratio(left.comparisons, right.comparisons),
        "left": left.as_dict(),
        "right": right.as_dict(),
    }
