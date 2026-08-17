"""Run every module's summary and print them as one table.

Run it with python examples/measure_everything.py. It is the package's own claim about itself:
each module states a handful of numbers it stands behind, and this prints all of them together
so that a change anywhere shows up in one place.
"""

from __future__ import annotations

from cqe.columns import nulls
from cqe.columns.encode import bitpack, choose, delta, dictionary, runlength
from cqe.cost import model
from cqe.eval import regression, scaling, workload
from cqe.exec import aggregate, distinct, pipeline, project, sets, sort, spill, window
from cqe.exec import filter as filters
from cqe.exec.join import hash as joins
from cqe.exec.join import outer
from cqe.plan import attribute, logical, physical
from cqe.plan.rules import ordering, pruning, pushdown, simplify
from cqe.sql import parse, render, tokenise
from cqe.stats import cardinality, correlation, histogram, sketch
from cqe.storage import bloom, catalogue, compact, disk, file, layout, statistics
from cqe.types import casting
from cqe.verify import differential, fuzz

MODULES = {
    "columns/nulls": nulls,
    "columns/encode/dictionary": dictionary,
    "columns/encode/runlength": runlength,
    "columns/encode/bitpack": bitpack,
    "columns/encode/delta": delta,
    "columns/encode/choose": choose,
    "types/casting": casting,
    "exec/filter": filters,
    "exec/project": project,
    "exec/aggregate": aggregate,
    "exec/distinct": distinct,
    "exec/sort": sort,
    "exec/window": window,
    "exec/sets": sets,
    "exec/join/hash": joins,
    "exec/join/outer": outer,
    "exec/pipeline": pipeline,
    "exec/spill": spill,
    "storage/statistics": statistics,
    "storage/file": file,
    "storage/bloom": bloom,
    "storage/layout": layout,
    "storage/disk": disk,
    "storage/catalogue": catalogue,
    "storage/compact": compact,
    "stats/histogram": histogram,
    "stats/sketch": sketch,
    "stats/cardinality": cardinality,
    "stats/correlation": correlation,
    "plan/logical": logical,
    "plan/physical": physical,
    "plan/attribute": attribute,
    "plan/rules/pushdown": pushdown,
    "plan/rules/ordering": ordering,
    "plan/rules/pruning": pruning,
    "plan/rules/simplify": simplify,
    "cost/model": model,
    "sql/tokenise": tokenise,
    "sql/parse": parse,
    "sql/render": render,
    "verify/fuzz": fuzz,
    "verify/differential": differential,
    "eval/workload": workload,
    "eval/regression": regression,
    "eval/scaling": scaling,
}


def render(value) -> str:
    """One value, short enough for a table."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    text = str(value)
    return text if len(text) <= 22 else text[:19] + "..."


def main() -> None:
    """Every module, its summary, and how long the whole thing took to state."""
    print(f"{len(MODULES)} modules\n")
    failures = []
    for name, module in MODULES.items():
        try:
            summary = module.summarise()
        except Exception as problem:
            failures.append((name, str(problem)))
            print(f"{name:<28} failed: {problem}")
            continue
        pieces = ", ".join(f"{key} {render(value)}" for key, value in summary.items())
        print(f"{name:<28} {pieces}")
    print()
    if failures:
        print(f"{len(failures)} modules could not summarise")
    else:
        print("every module summarised")


if __name__ == "__main__":
    main()
