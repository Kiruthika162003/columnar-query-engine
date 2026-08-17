"""Run every module's summary and print them as one table.

Run it with python examples/measure_everything.py. It is the package's own claim about itself:
each module states a handful of numbers it stands behind, and this prints all of them together
so that a change anywhere shows up in one place.
"""

from __future__ import annotations

from cqe.columns.encode import bitpack, delta, dictionary, runlength
from cqe.cost import model
from cqe.eval import regression, workload
from cqe.exec import aggregate, project, sort, spill
from cqe.exec import filter as filters
from cqe.exec.join import hash as joins
from cqe.plan import logical, physical
from cqe.plan.rules import ordering, pruning, pushdown
from cqe.sql import parse, tokenise
from cqe.stats import cardinality, histogram, sketch
from cqe.storage import bloom, disk, file, layout, statistics
from cqe.verify import differential, fuzz

MODULES = {
    "columns/encode/dictionary": dictionary,
    "columns/encode/runlength": runlength,
    "columns/encode/bitpack": bitpack,
    "columns/encode/delta": delta,
    "exec/filter": filters,
    "exec/project": project,
    "exec/aggregate": aggregate,
    "exec/sort": sort,
    "exec/join/hash": joins,
    "exec/spill": spill,
    "storage/statistics": statistics,
    "storage/file": file,
    "storage/bloom": bloom,
    "storage/layout": layout,
    "storage/disk": disk,
    "stats/histogram": histogram,
    "stats/sketch": sketch,
    "stats/cardinality": cardinality,
    "plan/logical": logical,
    "plan/physical": physical,
    "plan/rules/pushdown": pushdown,
    "plan/rules/ordering": ordering,
    "plan/rules/pruning": pruning,
    "cost/model": model,
    "sql/tokenise": tokenise,
    "sql/parse": parse,
    "verify/fuzz": fuzz,
    "verify/differential": differential,
    "eval/workload": workload,
    "eval/regression": regression,
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
