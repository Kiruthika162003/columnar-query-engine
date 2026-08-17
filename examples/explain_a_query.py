"""Take a query apart: tokens, plan, rewrite, strategies, cost, answer.

Run it with python examples/explain_a_query.py. It prints one query at every stage it passes
through, which is the whole engine in one page and is the fastest way to see what each module in
this package is for.
"""

from __future__ import annotations

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.cost.model import estimate
from cqe.exec.batch import Batch
from cqe.plan.logical import render, walk
from cqe.plan.physical import run
from cqe.plan.rules.pushdown import push_everything
from cqe.sql.parse import parse
from cqe.sql.parse import plan as plan_query
from cqe.sql.tokenise import tokenise
from cqe.stats.cardinality import collect

QUERY = (
    "select region, count(*) as orders from facts "
    "join shops on facts.shop = shops.shop "
    "where total > 80 group by region order by orders desc limit 3"
)


def tables(rows: int = 20000) -> dict[str, Batch]:
    """A fact table and a dimension to run the query against."""
    state = np.random.default_rng(4)
    facts = Batch.from_columns(
        [
            integer_column("order", np.arange(rows)),
            integer_column("shop", state.integers(0, 60, rows)),
            floating_column("total", state.normal(70, 30, rows)),
        ]
    )
    shops = Batch.from_columns(
        [
            integer_column("shop", np.arange(60)),
            string_column("region", [f"region{one % 5}" for one in range(60)]),
        ]
    )
    return {"facts": facts, "shops": shops}


def main() -> None:
    """Every stage, in the order the engine runs them."""
    catalogue = tables()

    print("the query")
    print(f"  {QUERY}")

    tokens = tokenise(QUERY)
    print(f"\ntokens: {len(tokens)}")
    print("  " + " ".join(one.value for one in tokens[:14]) + " ...")

    parsed = parse(QUERY)
    print("\nwhat it asked for")
    for name, value in parsed.as_dict().items():
        print(f"  {name:<16} {value}")

    built = plan_query(QUERY, catalogue)
    print(f"\nthe plan, {len(walk(built))} nodes")
    for line in render(built).split("\n"):
        print(f"  {line}")

    rewritten = push_everything(built)
    print(f"\nafter the rewrite, {rewritten.moved} predicates moved")
    for line in render(rewritten.after).split("\n"):
        print(f"  {line}")

    stats = {name: collect(one) for name, one in catalogue.items()}
    costed = estimate(rewritten.after, stats)
    print(f"\nwhat it should cost: {costed.total:.0f}, dominated by the {costed.dominant()}")
    for line in costed.explain().split("\n"):
        print(f"  {line}")

    executed = run(rewritten.after, catalogue)
    print("\nhow it ran")
    for one in executed.choices:
        print(f"  {one.describe()}")
    counted = executed.meter.values_touched + executed.meter.hash_probes
    print(f"  counted {counted}, predicted {costed.total:.0f}")

    print("\nthe answer")
    names = list(executed.batch.schema.names)
    print("  " + "  ".join(f"{one:>12}" for one in names))
    for row in executed.batch.to_rows():
        print("  " + "  ".join(f"{one!s:>12}" for one in row))


if __name__ == "__main__":
    main()
