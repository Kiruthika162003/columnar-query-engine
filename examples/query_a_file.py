"""Write a table to a file and query it, which is the engine end to end.

Run it with python examples/query_a_file.py. It writes a temporary file, runs four queries
against it, and prints what each one read. The numbers are the point: the same query against
three different arrangements of the same rows reads wildly different amounts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.exec.batch import Batch
from cqe.exec.expr import Compare, column, literal
from cqe.storage.disk import create, index, scan

ROWS = 40000


def build() -> Batch:
    """A table of orders, with the columns a report would group and filter on."""
    state = np.random.default_rng(2)
    return Batch.from_columns(
        [
            integer_column("order", np.arange(ROWS)),
            integer_column("shop", state.integers(0, 120, ROWS)),
            floating_column("total", state.normal(60, 25, ROWS)),
            string_column("region", [f"region{one}" for one in state.integers(0, 6, ROWS)]),
            string_column(
                "customer", [f"c{one:05d}" for one in state.integers(0, ROWS // 10, ROWS)]
            ),
        ]
    )


def show(name: str, measured) -> None:
    """One scan's accounting, on one line."""
    print(
        f"  {name:<22} groups {measured.groups_read:>3} of {measured.groups:<3} "
        f"rows {measured.rows_kept:>6} of {measured.rows_read:<6} "
        f"bytes {measured.bytes_read:>9}"
    )


def main() -> None:
    """Write the table three ways and query each."""
    batch = build()
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        arrival = create(where / "arrival.cqe", batch, group_size=500)
        by_total = create(
            where / "total.cqe", batch, group_size=500, order="sorted", key="total"
        )
        by_region = create(
            where / "region.cqe", batch, group_size=500, order="clustered", key="region"
        )

        print(f"{ROWS} rows, {arrival.groups} row groups of 500, {arrival.nbytes} bytes")

        print("\na range on total")
        predicate = Compare("<", column("total"), literal(10.0))
        for name, table in (
            ("as they arrived", arrival),
            ("sorted by total", by_total),
            ("clustered by region", by_region),
        ):
            _, measured = scan(table, predicate=predicate)
            show(name, measured)

        print("\nan equality on region")
        predicate = Compare("=", column("region"), literal("region3"))
        for name, table in (
            ("as they arrived", arrival),
            ("sorted by total", by_total),
            ("clustered by region", by_region),
        ):
            _, measured = scan(table, predicate=predicate)
            show(name, measured)

        print("\na lookup of one customer")
        predicate = Compare("=", column("customer"), literal("c00042"))
        _, plain = scan(arrival, predicate=predicate)
        show("no index", plain)
        _, indexed = scan(index(arrival, ["customer"]), predicate=predicate)
        show("with a bloom filter", indexed)

        print("\ntwo columns of five, no predicate")
        _, whole = scan(arrival)
        show("every column", whole)
        _, narrow = scan(arrival, columns=["order", "total"])
        show("two columns", narrow)


if __name__ == "__main__":
    main()
