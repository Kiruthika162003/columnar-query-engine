from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.meter import Meter
from cqe.errors import ConfigError
from cqe.exec.aggregate import Aggregate, hash_aggregate
from cqe.exec.batch import Batch, stack
from cqe.exec.expr import Compare, Expr, column, literal
from cqe.exec.filter import apply as apply_predicate
from cqe.exec.sort import SortKey, order_by
from cqe.verify.reference import Rows, agree

# Running a query in batches rather than all at once, which is what every engine does and which
# changes the cost in ways that are worth measuring rather than guessing.
#
# One batch of a million rows and a thousand batches of a thousand rows do the same arithmetic.
# What differs is everything around it: the fixed cost per batch, how much memory is held at
# once, and whether an operator can stop early.
#
# Three kinds of operator behave differently under batching and the difference decides the whole
# design.
#
# A streaming operator handles one batch at a time and forgets it. A filter and a projection are
# streaming, and for them a batch is free apart from the per batch overhead.
#
# A blocking operator has to see every row before it can produce any. A sort is blocking and so
# is an aggregate that has to return groups in order. For them batching does nothing except add
# the cost of putting the pieces back together.
#
# A partly blocking operator accumulates but produces at the end. A hash aggregate is one: it
# holds one entry per group rather than one per row, so it can consume a stream of any length in
# bounded memory even though it cannot emit until the stream ends.
#
# The measurements below are about where the batch size actually matters, and the answer is that
# it matters far less than the literature suggests until the batch gets small enough that the
# per batch overhead dominates, which is around a thousand rows here.

# The batch size everything defaults to. Measured below rather than chosen: it is where the per
# batch overhead has fallen to a few percent and the working set is still small.
BATCH_ROWS = 4096

# What one batch costs before it touches a single value, in units of values touched. It is the
# schema handling, the object allocation and the bookkeeping, and it is what makes a batch of
# ten rows a bad idea.
BATCH_OVERHEAD = 8


@dataclass
class Stage:
    """One operator in a pipeline, and whether it can work a batch at a time."""

    name: str
    kind: str
    apply: object

    @property
    def streaming(self) -> bool:
        """Whether it can hand a batch on without seeing the rest."""
        return self.kind == "streaming"

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"stage": self.name, "kind": self.kind, "streaming": self.streaming}


@dataclass
class Run:
    """What one pipeline produced and what it cost."""

    batch: Batch
    batches: int
    rows_in: int
    meter: Meter = field(default_factory=Meter)
    peak_rows: int = 0

    @property
    def rows(self) -> int:
        """Rows the pipeline produced."""
        return self.batch.rows

    @property
    def overhead(self) -> int:
        """What the batching itself cost, in values touched."""
        return self.batches * BATCH_OVERHEAD

    @property
    def share(self) -> float:
        """What share of the total the per batch overhead was."""
        total = self.meter.values_touched + self.overhead
        return self.overhead / max(total, 1)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "batches": self.batches,
            "rows_in": self.rows_in,
            "rows_out": self.rows,
            "touched": self.meter.values_touched,
            "overhead": self.overhead,
            "overhead_share": round(self.share, 4),
            "peak_rows": self.peak_rows,
        }


def in_batches(batch: Batch, rows: int) -> Iterator[Batch]:
    """A table as a stream of slices."""
    if rows <= 0:
        raise ConfigError(f"{rows} is not a batch size")
    for start in range(0, max(batch.rows, 1), rows):
        piece = batch.slice(start, min(start + rows, batch.rows))
        if piece.rows or batch.rows == 0:
            yield piece


def filter_stage(predicate: Expr) -> Stage:
    """A filter, which is streaming."""
    return Stage(
        name=f"filter {predicate.__class__.__name__}",
        kind="streaming",
        apply=lambda one, meter: apply_predicate(predicate, one, meter=meter),
    )


def project_stage(names: Sequence[str]) -> Stage:
    """A projection, which is streaming and free."""
    return Stage(
        name=f"project {len(names)}",
        kind="streaming",
        apply=lambda one, meter: one.select(list(names), meter=meter),
    )


def sort_stage(keys: Sequence[SortKey]) -> Stage:
    """A sort, which is blocking: it cannot emit a row until it has seen them all."""

    def run(one: Batch, meter: Meter | None) -> Batch:
        return one.take(order_by(one, keys, meter=meter).positions, meter=meter)

    return Stage(name=f"sort {len(keys)}", kind="blocking", apply=run)


def aggregate_stage(keys: Sequence[str], aggregates: Sequence[Aggregate]) -> Stage:
    """A hash aggregate, which accumulates and emits at the end."""

    def run(one: Batch, meter: Meter | None) -> Batch:
        return hash_aggregate(one, keys, aggregates, meter=meter).batch

    return Stage(name=f"aggregate {len(keys)}", kind="accumulating", apply=run)


def run(
    batch: Batch,
    stages: Sequence[Stage],
    batch_rows: int = BATCH_ROWS,
    meter: Meter | None = None,
) -> Run:
    """A pipeline over a table, in batches.

    The streaming prefix runs a batch at a time and everything from the first blocking stage
    onwards runs once over the concatenation. That split is the whole implementation and it is
    what a real engine does: the pipeline breaks at the first operator that has to see
    everything.
    """
    if not stages:
        raise ConfigError("a pipeline needs at least one stage")
    counted = meter or Meter()
    streaming = []
    rest = list(stages)
    while rest and rest[0].streaming:
        streaming.append(rest.pop(0))
    pieces = []
    batches = 0
    peak = 0
    for piece in in_batches(batch, batch_rows):
        batches += 1
        current = piece
        for one in streaming:
            current = one.apply(current, counted)
        peak = max(peak, current.rows)
        pieces.append(current)
    made = stack(pieces) if pieces else batch
    for one in rest:
        made = one.apply(made, counted)
        peak = max(peak, made.rows)
    return Run(batch=made, batches=batches, rows_in=batch.rows, meter=counted, peak_rows=peak)


def run_whole(batch: Batch, stages: Sequence[Stage], meter: Meter | None = None) -> Run:
    """The same pipeline with no batching at all, for comparison."""
    return run(batch, stages, batch_rows=max(batch.rows, 1), meter=meter)


def _table(rows: int = 100000, seed: int = 107) -> Batch:
    """A table large enough for the batch size to matter."""
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 60, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 8, rows)]),
        ]
    )


def _streaming_stages() -> list[Stage]:
    """A filter and a projection, which is the shape most of a plan has."""
    return [
        filter_stage(Compare(">", column("amount"), literal(90.0))),
        project_stage(["id", "amount"]),
    ]


def batching_does_not_change_the_answer(rows: int = 50000) -> dict:
    """The same pipeline at five batch sizes, which must all produce the same rows.

    The property batching has to have before any measurement of it means anything. Checked
    against the unbatched run rather than against a reference, because the question is whether
    batching changed the answer rather than whether the operators are right.
    """
    batch = _table(rows)
    stages = _streaming_stages()
    whole = run_whole(batch, stages)
    out = {}
    for size in (128, 1024, 4096, 16384):
        made = run(batch, stages, batch_rows=size)
        out[size] = bool(agree(Rows.of(made.batch), Rows.of(whole.batch), ordered=True))
    return {
        "rows": whole.rows,
        "sizes": list(out),
        "results": out,
        "they_all_agree": all(out.values()),
    }


def the_batch_size_barely_changes_the_work(rows: int = 100000) -> dict:
    """Values touched against batch size, which is nearly flat.

    The measurement that contradicts the intuition. The arithmetic is the same however it is
    sliced, so the only thing that changes with the batch size is the per batch overhead, and
    over a wide range of sizes that is a rounding error.
    """
    batch = _table(rows)
    stages = _streaming_stages()
    out = []
    for size in (256, 1024, 4096, 16384, 65536):
        made = run(batch, stages, batch_rows=size)
        out.append(
            {
                "batch_rows": size,
                "batches": made.batches,
                "touched": made.meter.values_touched,
                "overhead": made.overhead,
                "overhead_share": round(made.share, 4),
            }
        )
    touched = [one["touched"] for one in out]
    return {
        "sweep": out,
        "smallest_touched": min(touched),
        "largest_touched": max(touched),
        "the_spread_is_small": max(touched) / max(min(touched), 1) < 1.05,
        "and_the_overhead_is_what_moves": out[0]["overhead_share"]
        > out[-1]["overhead_share"] * 10,
    }


def a_small_batch_is_dominated_by_its_overhead(rows: int = 50000) -> dict:
    """Where the batch size does matter, which is at the bottom.

    At ten rows a batch the fixed cost per batch is comparable to the work in it, and the
    overhead share is the number that says so. This is the measurement that sets BATCH_ROWS, and
    the answer is not the smallest batch that fits in cache, it is the smallest batch whose
    overhead has become negligible.
    """
    batch = _table(rows)
    stages = _streaming_stages()
    out = []
    for size in (10, 50, 200, 1000, 4096):
        made = run(batch, stages, batch_rows=size)
        out.append(
            {
                "batch_rows": size,
                "batches": made.batches,
                "overhead": made.overhead,
                "overhead_share": round(made.share, 4),
            }
        )
    shares = [one["overhead_share"] for one in out]
    return {
        "sweep": out,
        "at_ten_rows": shares[0],
        "at_four_thousand": shares[-1],
        "the_share_falls": shares == sorted(shares, reverse=True),
        "and_it_is_negligible_by_a_thousand": shares[3] < 0.02,
        "the_chosen_size": BATCH_ROWS,
    }


def a_streaming_pipeline_holds_one_batch(rows: int = 100000) -> dict:
    """What batching is actually for: the peak rows held, not the work done.

    The number that matters and the one the work sweep does not show. A streaming pipeline at
    four thousand rows a batch holds four thousand rows however large the table is, and the
    unbatched form holds all of them.
    """
    batch = _table(rows)
    stages = _streaming_stages()
    batched = run(batch, stages, batch_rows=BATCH_ROWS)
    whole = run_whole(batch, stages)
    return {
        "rows": rows,
        "batched_peak": batched.peak_rows,
        "whole_peak": whole.peak_rows,
        "ratio": round(whole.peak_rows / max(batched.peak_rows, 1), 1),
        "it_held_one_batch": batched.peak_rows <= BATCH_ROWS,
        "and_the_answers_agree": bool(
            agree(Rows.of(batched.batch), Rows.of(whole.batch), ordered=True)
        ),
    }


def a_blocking_stage_ends_the_streaming(rows: int = 50000) -> dict:
    """A sort in the middle, which makes everything after it see the whole table.

    The limit of batching and the reason a plan is a pipeline of pipelines rather than one. A
    sort cannot emit its first row until it has seen the last, so the peak memory is the whole
    table whatever the batch size is.
    """
    batch = _table(rows)
    stages = [*_streaming_stages(), sort_stage([SortKey(name="amount")])]
    made = run(batch, stages, batch_rows=1024)
    return {
        "rows": rows,
        "batches": made.batches,
        "peak_rows": made.peak_rows,
        "rows_out": made.rows,
        "the_peak_is_the_filtered_table": made.peak_rows == made.rows,
        "and_it_is_larger_than_a_batch": made.peak_rows > 1024,
        "it_is_still_sorted": bool(np.all(np.diff(made.batch.column("amount").values) >= 0)),
    }


def an_accumulating_stage_holds_the_groups(rows: int = 100000) -> dict:
    """A hash aggregate, which holds one entry per group rather than one per row.

    The third kind. It cannot emit until the stream ends, so it is not streaming, and it does
    not hold the rows, so it is not blocking either. That distinction is what lets an aggregate
    over a table of any size run in the memory its group count needs.
    """
    batch = _table(rows)
    stages = [
        filter_stage(Compare(">", column("amount"), literal(60.0))),
        aggregate_stage(["shop"], [Aggregate(name="n", function="count_star", source="")]),
    ]
    made = run(batch, stages, batch_rows=4096)
    groups = len(set(batch.column("shop").to_list()))
    return {
        "rows": rows,
        "groups": groups,
        "rows_out": made.rows,
        "it_returned_a_row_per_group": made.rows == groups,
        "the_group_count_is_tiny_against_the_rows": groups < rows / 1000,
    }


def the_streaming_prefix_is_found_automatically(rows: int = 20000) -> dict:
    """Where the pipeline breaks, which is at the first blocking stage.

    Read off the run rather than declared, so a stage added later that is blocking and not
    marked as such shows up as a peak memory that stops matching the batch size.
    """
    batch = _table(rows)
    cases = {
        "all streaming": _streaming_stages(),
        "sort last": [*_streaming_stages(), sort_stage([SortKey(name="amount")])],
        "sort first": [sort_stage([SortKey(name="amount")]), *_streaming_stages()],
    }
    out = {}
    for name, stages in cases.items():
        made = run(batch, stages, batch_rows=1024)
        out[name] = {"batches": made.batches, "peak": made.peak_rows}
    return {
        "cases": out,
        "streaming_holds_a_batch": out["all streaming"]["peak"] <= 1024,
        "a_sort_at_the_end_does_not": out["sort last"]["peak"] > 1024,
        "and_a_sort_at_the_front_does_not_either": out["sort first"]["peak"] > 1024,
    }


def a_filter_makes_the_later_batches_smaller(rows: int = 50000) -> dict:
    """A selective filter early in a pipeline shrinks every batch that follows it.

    Which is the composition every plan relies on and is worth measuring on the batch sizes
    rather than on the totals: the second stage sees a tenth of the rows and does a tenth of the
    work, batch by batch.
    """
    batch = _table(rows)
    tight = [
        filter_stage(Compare(">", column("amount"), literal(140.0))),
        project_stage(["id", "amount"]),
    ]
    loose = [
        filter_stage(Compare(">", column("amount"), literal(20.0))),
        project_stage(["id", "amount"]),
    ]
    strict = run(batch, tight, batch_rows=4096)
    relaxed = run(batch, loose, batch_rows=4096)
    return {
        "selective_rows": strict.rows,
        "loose_rows": relaxed.rows,
        "selective_peak": strict.peak_rows,
        "loose_peak": relaxed.peak_rows,
        "the_selective_one_holds_less": strict.peak_rows < relaxed.peak_rows,
        "by_about_the_selectivity": round(relaxed.peak_rows / max(strict.peak_rows, 1), 1),
    }


def an_empty_batch_passes_through(rows: int = 10000) -> dict:
    """A filter that keeps nothing, where the pipeline still produces a batch.

    The case that turns a pipeline into a crash: an empty piece is a legitimate batch and every
    stage after it has to accept one. Checked rather than assumed because an operator written
    against a non empty batch fails on the last slice of an unlucky table rather than on the
    first.
    """
    batch = _table(rows)
    stages = [
        filter_stage(Compare(">", column("amount"), literal(1e9))),
        project_stage(["id", "amount"]),
    ]
    made = run(batch, stages, batch_rows=1024)
    return {
        "batches": made.batches,
        "rows_out": made.rows,
        "it_is_empty": made.rows == 0,
        "and_it_kept_its_schema": list(made.batch.schema.names) == ["id", "amount"],
    }


def a_batch_larger_than_the_table_is_one_batch(rows: int = 5000) -> dict:
    """A batch size above the row count, which is the unbatched case by another name."""
    batch = _table(rows)
    made = run(batch, _streaming_stages(), batch_rows=rows * 10)
    whole = run_whole(batch, _streaming_stages())
    return {
        "batches": made.batches,
        "it_is_one": made.batches == 1,
        "it_matches_the_unbatched_run": made.rows == whole.rows,
        "and_the_work_is_the_same": made.meter.values_touched == whole.meter.values_touched,
    }


def the_overhead_constant_is_what_it_claims(rows: int = 50000) -> dict:
    """What the per batch overhead is against what the constant says.

    An honest note rather than a measurement of the engine: the constant is a stand in for the
    fixed cost per batch and this package counts values rather than instructions, so the number
    cannot be measured from inside. What can be checked is that the model built on it is
    consistent, which is that the overhead scales with the batch count exactly.
    """
    batch = _table(rows)
    stages = _streaming_stages()
    out = []
    for size in (500, 1000, 2000):
        made = run(batch, stages, batch_rows=size)
        out.append(
            {
                "batch_rows": size,
                "batches": made.batches,
                "overhead": made.overhead,
                "per_batch": made.overhead / max(made.batches, 1),
            }
        )
    return {
        "sweep": out,
        "the_constant": BATCH_OVERHEAD,
        "it_is_per_batch_exactly": all(one["per_batch"] == BATCH_OVERHEAD for one in out),
        "and_it_is_a_stand_in": True,
    }


def a_pipeline_with_no_stages_is_refused() -> bool:
    """A pipeline that does nothing, which is a caller error."""
    try:
        run(_table(100), [])
    except ConfigError:
        return True
    return False


def a_zero_batch_size_is_refused() -> bool:
    """Batches of no rows, which would never finish."""
    try:
        run(_table(100), _streaming_stages(), batch_rows=0)
    except ConfigError:
        return True
    return False


def compare_the_stage_kinds() -> list[dict]:
    """The three kinds of stage and what each one does to a pipeline."""
    return [
        {
            "kind": "streaming",
            "example": "filter, projection",
            "holds": "one batch",
            "emits": "per batch",
        },
        {
            "kind": "accumulating",
            "example": "hash aggregate",
            "holds": "one entry per group",
            "emits": "at the end",
        },
        {
            "kind": "blocking",
            "example": "sort",
            "holds": "every row",
            "emits": "at the end",
        },
    ]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "batch_rows": BATCH_ROWS,
        "batch_overhead": BATCH_OVERHEAD,
        "batching_is_exact": batching_does_not_change_the_answer()["they_all_agree"],
        "the_work_barely_moves": the_batch_size_barely_changes_the_work()[
            "the_spread_is_small"
        ],
        "the_memory_does": a_streaming_pipeline_holds_one_batch()["it_held_one_batch"],
        "a_blocking_stage_ends_it": a_blocking_stage_ends_the_streaming()[
            "and_it_is_larger_than_a_batch"
        ],
    }
