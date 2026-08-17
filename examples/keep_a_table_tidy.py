"""Ingest a table in small pieces, query it, compact it, and query it again.

Run it with python examples/keep_a_table_tidy.py. It is the maintenance story end to end: a
writer that cannot buffer produces one file per batch, queries get steadily worse at the
metadata, and a compaction pass fixes it at the cost of one rewrite. Every number printed comes
from the same functions the tests assert on, so nothing here is a separate implementation.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from cqe.exec.batch import Batch
from cqe.exec.distinct import by_hash, count_distinct
from cqe.exec.expr import Compare, column, literal
from cqe.storage.compact import compact, ingest, load, scan_cost
from cqe.storage.file import read

ROWS = 60_000
INGEST = 500
FOLDER = Path("_tidy_parts")
COMPACTED = Path("_tidy_whole") / "table.cqe"


def orders(rows: int = ROWS, seed: int = 23) -> Batch:
    """A table of orders, arriving in identifier order the way an appended one does."""
    state = np.random.default_rng(seed)
    return Batch.of(
        id=np.arange(rows).tolist(),
        shop=state.integers(0, 60, rows).tolist(),
        customer=state.integers(0, 8_000, rows).tolist(),
        status=[
            ["placed", "paid", "shipped", "returned"][int(one)]
            for one in state.integers(0, 4, rows)
        ],
        amount=(state.gamma(2.0, 40.0, rows)).tolist(),
    )


def show_table(title: str, rows: list[dict]) -> None:
    """A list of mappings as an aligned table."""
    if not rows:
        return
    names = list(rows[0])
    widths = {name: max(len(name), *(len(str(one[name])) for one in rows)) for name in names}
    print(f"\n{title}")
    print("  ".join(name.ljust(widths[name]) for name in names))
    for one in rows:
        print("  ".join(str(one[name]).ljust(widths[name]) for name in names))


def clear() -> None:
    """Remove anything a previous run left behind."""
    for one in (FOLDER, COMPACTED.parent):
        if one.exists():
            shutil.rmtree(one, ignore_errors=True)


def the_ingest(batch: Batch) -> None:
    """Write the table as many small files and say what that costs."""
    table = ingest(batch, FOLDER, size=INGEST)
    print(f"{batch.rows} rows arrived in {len(table.fragments)} pieces of {INGEST}")
    show_table(
        "what the files hold",
        [
            {
                "fragments": len(table.fragments),
                "rows": table.rows,
                "data bytes": table.data_bytes,
                "footer bytes": table.footer_bytes,
                "open cost": table.open_bytes,
                "metadata share": round(table.metadata_bytes / table.data_bytes, 3),
            }
        ],
    )


def the_queries(title: str, table) -> list[dict]:
    """Three predicates priced against one shape of the table."""
    questions = {
        "recent orders": Compare("<", column("id"), literal(int(ROWS * 0.2))),
        "large amounts": Compare(">", column("amount"), literal(300.0)),
        "one shop": Compare("=", column("shop"), literal(7)),
    }
    rows = []
    for name, predicate in questions.items():
        priced = scan_cost(table, predicate, columns=2)
        rows.append(
            {
                "query": name,
                "groups": priced["groups"],
                "skipped": priced["skipped"],
                "rows read": priced["rows_read"],
                "metadata": priced["metadata_bytes"],
                "total bytes": priced["total_bytes"],
            }
        )
    show_table(title, rows)
    return rows


def the_compaction(table):
    """Merge the fragments and report the trade."""
    made = compact(table, COMPACTED, group_size=4_000, sort_by="amount")
    show_table(
        "what the compaction did",
        [
            {
                "fragments": f"{len(made.before.fragments)} to {len(made.after.fragments)}",
                "groups": f"{made.before.groups} to {made.after.groups}",
                "metadata": f"{made.before.metadata_bytes} to {made.after.metadata_bytes}",
                "saved per scan": made.metadata_saved,
                "rewrite cost": made.rewrite_cost,
                "break even": round(made.break_even, 1),
            }
        ],
    )
    return made


def the_contents(before: Batch) -> None:
    """Check that the compacted file still holds the table, and describe what is in it."""
    after = read(COMPACTED)
    same = sorted(tuple(one) for one in before.to_rows()) == sorted(
        tuple(one) for one in after.to_rows()
    )
    print(f"\nthe compacted file holds the same {after.rows} rows: {'yes' if same else 'no'}")
    show_table(
        "what is in it",
        [
            {
                "column": name,
                "type": after.column(name).logical,
                "distinct": (
                    by_hash(after.column(name)).count
                    if after.column(name).logical == "string"
                    else int(count_distinct(after.column(name)))
                ),
                "nulls": after.column(name).null_count,
            }
            for name in after.names
        ],
    )


def main() -> None:
    """The whole story, in the order it happens."""
    clear()
    try:
        batch = orders()
        the_ingest(batch)
        table = ingest(batch, FOLDER, size=INGEST)
        before = the_queries("queries against the fragments", table)
        made = the_compaction(table)
        after = the_queries("the same queries against the compacted file", made.after)
        gains = [
            {
                "query": one["query"],
                "before": one["total bytes"],
                "after": other["total bytes"],
                "ratio": round(one["total bytes"] / max(other["total bytes"], 1), 2),
            }
            for one, other in zip(before, after, strict=True)
        ]
        show_table("what each query gained", gains)
        worse = [one["query"] for one in gains if one["ratio"] < 1]
        if worse:
            print(
                f"\n{', '.join(worse)} got worse, because the compaction sorted by amount and "
                "a file has one order to spend"
            )
        the_contents(load(table))
    finally:
        clear()


if __name__ == "__main__":
    main()
