from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.columns.encode import bitpack, delta, dictionary, runlength
from cqe.cost import model
from cqe.cost.model import estimate
from cqe.errors import ParseError, QueryEngineError
from cqe.eval import regression as eval_regression
from cqe.eval import workload as eval_workload
from cqe.exec import aggregate, project, sort, spill
from cqe.exec import filter as filters
from cqe.exec.batch import Batch
from cqe.exec.join import hash as joins
from cqe.plan import logical, physical
from cqe.plan.logical import render
from cqe.plan.physical import explain as explain_plan
from cqe.plan.physical import run
from cqe.plan.rules import ordering, pruning, pushdown
from cqe.plan.rules.pushdown import push_everything
from cqe.sql import parse, tokenise
from cqe.sql.parse import plan as plan_query
from cqe.stats import cardinality, histogram, sketch
from cqe.stats.cardinality import collect
from cqe.storage import bloom, disk, file, layout, statistics
from cqe.storage.file import peek, read, write
from cqe.storage.layout import as_they_arrive, clustered_by, sorted_by
from cqe.verify import differential, fuzz
from cqe.verify.differential import run_all

# The command line, which exists so that every measurement in this package can be run without
# writing any Python.
#
# The design rule is that a subcommand does one thing and prints one table. Nothing here
# composes subcommands, nothing takes a configuration file, and nothing has a mode flag that
# changes what it means. A command line that needs a manual has failed, and the way they get
# there is by growing options that interact.
#
# Everything prints as either a readable table or as JSON, chosen by one flag, because the two
# audiences are a person reading and a script parsing and neither is served by the other's
# format.
#
# Errors print the message and nothing else. A traceback is for a bug in this package; a query
# with a typo in it is not a bug and should not look like one.

WIDTH = 18


@dataclass
class Result:
    """What one subcommand produced: a status and something to print."""

    status: int
    payload: object

    @property
    def ok(self) -> bool:
        """Whether the command succeeded."""
        return self.status == 0


def _sample(rows: int = 5000, seed: int = 3) -> Batch:
    """The table the commands work on when no file is given.

    A generated table rather than a required argument, so that every command can be run and read
    with nothing set up. The columns are the ones the measurements elsewhere use, so what the
    command line prints matches what the module docstrings say.
    """
    state = np.random.default_rng(seed)
    return Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 40, rows)),
            floating_column("amount", state.normal(100, 30, rows)),
            string_column("region", [f"region{one}" for one in state.integers(0, 6, rows)]),
        ]
    )


def _load(path: str | None) -> Batch:
    """The table to work on, from a file or generated."""
    return _sample() if not path else read(Path(path))


def _catalogue(path: str | None) -> dict[str, Batch]:
    """The tables a query can name."""
    return {"facts": _load(path)}


def _table(rows: Sequence[dict], columns: Sequence[str] | None = None) -> str:
    """A list of mappings as an aligned table.

    Aligned rather than comma separated, because the output is read by a person by default and
    the JSON flag is there for everything else. Numbers are rounded to three places, which is
    enough for every ratio this package produces and short enough to line up.
    """
    if not rows:
        return "nothing to show"
    names = list(columns) if columns else list(rows[0])
    header = " ".join(one.rjust(WIDTH)[:WIDTH] for one in names)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" ".join(_cell(row.get(one, "")) for one in names))
    return "\n".join(lines)


def _cell(value) -> str:
    """One value, formatted to a fixed width."""
    if isinstance(value, float):
        text = f"{value:.3f}"
    elif isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, (list, tuple)):
        text = ",".join(str(one) for one in value)
    else:
        text = str(value)
    return text.rjust(WIDTH)[:WIDTH]


def _emit(payload: object, as_json: bool) -> str:
    """One command's output, in whichever form was asked for."""
    if as_json:
        return json.dumps(payload, indent=2, default=str)
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        return _table(payload)
    if isinstance(payload, dict):
        return _table([{"name": name, "value": value} for name, value in payload.items()])
    return str(payload)


def command_schema(arguments: argparse.Namespace) -> Result:
    """What a table holds, without reading any of it.

    For a file this is the footer alone, which is a few kilobytes of a file that may be
    gigabytes. That is the whole point of putting the schema in a footer.
    """
    if arguments.file:
        footer = peek(Path(arguments.file))
        return Result(
            0,
            [
                {"column": one.name, "type": one.logical, "nullable": one.nullable}
                for one in footer.schema.fields
            ],
        )
    batch = _sample()
    return Result(
        0,
        [
            {"column": one.name, "type": one.logical, "nullable": one.nullable}
            for one in batch.schema.fields
        ],
    )


def command_stats(arguments: argparse.Namespace) -> Result:
    """The statistics a planner would use, per column."""
    batch = _load(arguments.file)
    stats = collect(batch)
    return Result(
        0,
        [
            {
                "column": name,
                "rows": one.rows,
                "distinct": round(one.distinct, 1),
                "nulls": one.nulls,
                "null_share": round(one.null_share, 3),
                "bytes": one.nbytes,
            }
            for name, one in stats.columns.items()
        ],
    )


def command_plan(arguments: argparse.Namespace) -> Result:
    """A query as a logical plan, before and after the rewrite.

    Both, because the rewrite is the interesting part and a command that printed only the final
    plan would hide it. The two trees side by side are what makes pushdown legible.
    """
    catalogue = _catalogue(arguments.file)
    built = plan_query(arguments.query, catalogue)
    rewritten = push_everything(built)
    return Result(
        0,
        "before the rewrite\n"
        + render(built)
        + "\n\nafter the rewrite\n"
        + render(rewritten.after)
        + f"\n\nmoved {rewritten.moved} predicates",
    )


def command_explain(arguments: argparse.Namespace) -> Result:
    """A query as a physical plan, with the strategy chosen for every node."""
    catalogue = _catalogue(arguments.file)
    built = plan_query(arguments.query, catalogue)
    if not arguments.raw:
        built = push_everything(built).after
    return Result(0, explain_plan(built, catalogue))


def command_cost(arguments: argparse.Namespace) -> Result:
    """What the model predicts a query costs, against what the meter counts."""
    catalogue = _catalogue(arguments.file)
    built = plan_query(arguments.query, catalogue)
    if not arguments.raw:
        built = push_everything(built).after
    stats = {name: collect(one) for name, one in catalogue.items()}
    predicted = estimate(built, stats)
    executed = run(built, catalogue)
    counted = executed.meter.values_touched + executed.meter.hash_probes
    return Result(
        0,
        {
            "predicted": round(predicted.total),
            "counted": counted,
            "ratio": round(predicted.total / max(counted, 1), 2),
            "dominant": predicted.dominant(),
            "rows": executed.rows,
            "choices": [one.strategy for one in executed.choices],
        },
    )


def command_query(arguments: argparse.Namespace) -> Result:
    """Run a query and print its rows."""
    catalogue = _catalogue(arguments.file)
    built = plan_query(arguments.query, catalogue)
    if not arguments.raw:
        built = push_everything(built).after
    produced = run(built, catalogue).batch
    limit = min(produced.rows, arguments.limit)
    return Result(
        0,
        [
            {name: row[position] for position, name in enumerate(produced.schema.names)}
            for row in produced.slice(0, limit).to_rows()
        ],
    )


def command_write(arguments: argparse.Namespace) -> Result:
    """Write a table to a file in the columnar format, arranged as asked."""
    batch = _sample(rows=arguments.rows)
    if arguments.sort:
        layout = sorted_by(batch, arguments.sort, group_size=arguments.group_size)
    elif arguments.cluster:
        layout = clustered_by(batch, arguments.cluster, group_size=arguments.group_size)
    else:
        layout = as_they_arrive(batch, group_size=arguments.group_size)
    path = Path(arguments.out)
    write(path, layout.flatten(), group_size=arguments.group_size)
    footer = peek(path)
    return Result(
        0,
        {
            "path": str(path),
            "rows": footer.rows,
            "groups": len(footer.groups),
            "bytes": path.stat().st_size,
            "layout": layout.order,
            "key": layout.key,
        },
    )


def command_verify(arguments: argparse.Namespace) -> Result:
    """Run the differential harness and report every check.

    The command that says whether this engine is right, which is a different question from
    whether the tests pass: the tests run fixed cases and this runs generated ones, so a run
    here covers inputs no test names.
    """
    reports = run_all(count=arguments.cases, seed=arguments.seed)
    rows = [one.as_dict() for one in reports]
    failed = [one for one in reports if not one.passed]
    return Result(1 if failed else 0, rows)


def command_measure(arguments: argparse.Namespace) -> Result:
    """Every module's summary in one table, which is the package in one screen."""
    modules = {
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
        "eval/workload": eval_workload,
        "eval/regression": eval_regression,
    }
    wanted = [one for one in modules if arguments.only in one] if arguments.only else modules
    out = []
    for name in wanted:
        summary = modules[name].summarise()
        out.append({"module": name, **{key: summary[key] for key in list(summary)[:4]}})
    return Result(0, out)


def build_parser() -> argparse.ArgumentParser:
    """Every subcommand, its arguments and its help."""
    parser = argparse.ArgumentParser(
        prog="cqe", description="a columnar query engine that measures itself"
    )
    parser.add_argument("--json", action="store_true", help="print JSON rather than a table")
    subcommands = parser.add_subparsers(dest="command", required=True)

    schema = subcommands.add_parser("schema", help="what a table holds")
    schema.add_argument("--file", help="a file to read the footer of")
    schema.set_defaults(run=command_schema)

    stats = subcommands.add_parser("stats", help="the statistics a planner would use")
    stats.add_argument("--file", help="a file to read")
    stats.set_defaults(run=command_stats)

    plan = subcommands.add_parser("plan", help="a query as a logical plan")
    plan.add_argument("query", help="the query")
    plan.add_argument("--file", help="a file to query")
    plan.set_defaults(run=command_plan)

    explain = subcommands.add_parser("explain", help="a query as a physical plan")
    explain.add_argument("query", help="the query")
    explain.add_argument("--file", help="a file to query")
    explain.add_argument("--raw", action="store_true", help="skip the rewrite")
    explain.set_defaults(run=command_explain)

    cost = subcommands.add_parser("cost", help="predicted against counted cost")
    cost.add_argument("query", help="the query")
    cost.add_argument("--file", help="a file to query")
    cost.add_argument("--raw", action="store_true", help="skip the rewrite")
    cost.set_defaults(run=command_cost)

    query = subcommands.add_parser("query", help="run a query")
    query.add_argument("query", help="the query")
    query.add_argument("--file", help="a file to query")
    query.add_argument("--raw", action="store_true", help="skip the rewrite")
    query.add_argument("--limit", type=int, default=20, help="how many rows to print")
    query.set_defaults(run=command_query)

    writer = subcommands.add_parser("write", help="write a table to a file")
    writer.add_argument("out", help="where to write it")
    writer.add_argument("--rows", type=int, default=5000, help="how many rows")
    writer.add_argument("--group-size", type=int, default=500, help="rows per row group")
    writer.add_argument("--sort", help="a column to sort by")
    writer.add_argument("--cluster", help="a column to cluster by")
    writer.set_defaults(run=command_write)

    verify = subcommands.add_parser("verify", help="run the differential harness")
    verify.add_argument("--cases", type=int, default=40, help="cases per check")
    verify.add_argument("--seed", type=int, default=0, help="the generator seed")
    verify.set_defaults(run=command_verify)

    measure = subcommands.add_parser("measure", help="every module's summary")
    measure.add_argument("--only", help="a substring of the module names to show")
    measure.set_defaults(run=command_measure)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments, run the command, print the result.

    Every engine error is caught and printed as a message, because a bad query is not a crash. A
    ParseError prints with its caret, which is the whole reason the tokeniser records positions.
    """
    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = arguments.run(arguments)
    except ParseError as problem:
        print(problem.marked(), file=sys.stderr)
        return 2
    except QueryEngineError as problem:
        print(f"{type(problem).__name__}: {problem}", file=sys.stderr)
        return 2
    except OSError as problem:
        print(f"{problem.strerror or problem}: {problem.filename or ''}", file=sys.stderr)
        return 2
    print(_emit(result.payload, arguments.json))
    return result.status


def _run(argv: Sequence[str]) -> tuple[int, str]:
    """One command, with its output captured, for the measurements below."""
    out = io.StringIO()
    errors = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(errors):
        status = main(argv)
    return status, out.getvalue() + errors.getvalue()


def every_command_runs() -> dict:
    """Each subcommand once, on the generated table, checking only that it worked.

    A smoke test rather than a measurement, and the one that catches the commonest command line
    failure: an argument renamed in one place and not the other, which no unit test sees because
    every unit test calls the function directly.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "table.cqe")
        commands = [
            ["schema"],
            ["stats"],
            ["plan", "select id, amount from facts where amount > 100"],
            ["explain", "select shop, count(*) as n from facts group by shop"],
            ["cost", "select id from facts where amount > 100"],
            ["query", "select id, amount from facts where amount > 130", "--limit", "5"],
            ["write", path, "--rows", "500"],
            ["verify", "--cases", "5"],
            ["measure", "--only", "sql"],
        ]
        results = [(one[0], *_run(one)) for one in commands]
    return {
        "commands": len(results),
        "all_succeeded": all(status == 0 for _, status, _ in results),
        "all_printed_something": all(bool(text.strip()) for _, _, text in results),
        "failed": [name for name, status, _ in results if status != 0],
    }


def the_json_flag_produces_json() -> dict:
    """The same command twice, once as a table and once as JSON."""
    status, table = _run(["stats"])
    _, text = _run(["--json", "stats"])
    parsed = json.loads(text)
    return {
        "status": status,
        "the_table_has_a_header": "column" in table,
        "the_json_parses": isinstance(parsed, list),
        "they_hold_the_same_rows": len(parsed) == len(table.strip().split("\n")) - 2,
    }


def a_bad_query_prints_a_caret() -> dict:
    """A query with a stray character, which comes back with a caret under it.

    The measurement that says the position bookkeeping in the tokeniser was worth doing, since
    this is the only place a user ever sees it.
    """
    status, text = _run(["plan", "select id from facts where a # 1"])
    lines = [one for one in text.split("\n") if one.strip()]
    return {
        "status": status,
        "it_failed": status == 2,
        "it_printed_a_caret": any("^" in one for one in lines),
        "it_printed_no_traceback": "Traceback" not in text,
        "lines": len(lines),
    }


def a_missing_column_prints_the_schema() -> dict:
    """A query naming a column that does not exist, which comes back with the ones that do."""
    status, text = _run(["plan", "select nothing from facts"])
    return {
        "status": status,
        "it_failed": status == 2,
        "it_named_the_column": "nothing" in text,
        "it_listed_the_real_ones": "amount" in text,
        "it_printed_no_traceback": "Traceback" not in text,
    }


def a_missing_table_prints_the_catalogue() -> dict:
    """A query against a table that is not there."""
    status, text = _run(["plan", "select id from nowhere"])
    return {
        "status": status,
        "it_failed": status == 2,
        "it_listed_the_tables": "facts" in text,
    }


def the_plan_command_shows_both_trees() -> dict:
    """The plan before and after the rewrite, which is what makes pushdown visible."""
    status, text = _run(["plan", "select id, amount from facts where amount > 100"])
    return {
        "status": status,
        "it_showed_both": "before the rewrite" in text and "after the rewrite" in text,
        "it_said_what_moved": "moved" in text,
        "the_before_has_a_filter": text.split("after the rewrite")[0].count("Filter") == 1,
        "the_after_does_not": "Filter" not in text.split("after the rewrite")[1],
    }


def the_explain_command_names_a_strategy() -> dict:
    """An explain over a query with a group in it."""
    status, text = _run(["explain", "select shop, count(*) as n from facts group by shop"])
    return {
        "status": status,
        "it_showed_the_tree": "Group" in text,
        "it_named_a_strategy": "hash" in text or "counting" in text or "sorted" in text,
    }


def the_query_command_returns_rows() -> dict:
    """A query that returns a few rows, printed as a table."""
    status, text = _run(
        ["query", "select id, amount from facts where amount > 150", "--limit", "5"]
    )
    lines = [one for one in text.split("\n") if one.strip()]
    return {
        "status": status,
        "lines": len(lines),
        "it_printed_a_header": "id" in lines[0] and "amount" in lines[0],
        "it_printed_five_rows": len(lines) == 7,
    }


def a_query_returning_nothing_says_so() -> dict:
    """A predicate that keeps no rows, which prints a message rather than a bare header."""
    status, text = _run(["query", "select id from facts where amount > 100000"])
    return {
        "status": status,
        "it_succeeded": status == 0,
        "it_said_so": "nothing" in text,
    }


def the_write_command_writes_a_readable_file() -> dict:
    """Write a file and read its footer back, the round trip through the command line."""
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "one.cqe")
        status, _ = _run(["write", path, "--rows", "2000", "--group-size", "250"])
        footer = peek(Path(path))
        _, schema = _run(["schema", "--file", path])
    return {
        "status": status,
        "rows": footer.rows,
        "groups": len(footer.groups),
        "it_wrote_the_asked_for_rows": footer.rows == 2000,
        "and_the_asked_for_groups": len(footer.groups) == 8,
        "the_schema_reads_back": "amount" in schema,
    }


def the_write_command_arranges_the_rows() -> dict:
    """Writing sorted and writing as they arrive, which produce different files."""
    with tempfile.TemporaryDirectory() as directory:
        plain = str(Path(directory) / "plain.cqe")
        ordered = str(Path(directory) / "sorted.cqe")
        _run(["write", plain, "--rows", "2000"])
        _run(["write", ordered, "--rows", "2000", "--sort", "amount"])
        first = read(Path(ordered)).column("amount").values
        second = read(Path(plain)).column("amount").values
        return {
            "the_sorted_one_is_ordered": bool(np.all(np.diff(first) >= 0)),
            "the_plain_one_is_not": not bool(np.all(np.diff(second) >= 0)),
        }


def the_verify_command_reports_every_check() -> dict:
    """The differential harness through the command line."""
    status, text = _run(["--json", "verify", "--cases", "5"])
    parsed = json.loads(text)
    return {
        "status": status,
        "checks": len(parsed),
        "it_succeeded": status == 0,
        "every_check_passed": all(one["passed"] for one in parsed),
    }


def the_measure_command_covers_every_module() -> dict:
    """Every module's summary, which is the package in one table."""
    status, text = _run(["--json", "measure"])
    parsed = json.loads(text)
    return {
        "status": status,
        "modules": len(parsed),
        "they_all_summarised": all("module" in one for one in parsed),
        "it_covers_the_package": len(parsed) >= 25,
    }


def the_only_flag_narrows_the_measurement() -> dict:
    """Filtering the measure command to one part of the package."""
    _, everything = _run(["--json", "measure"])
    _, narrow = _run(["--json", "measure", "--only", "encode"])
    return {
        "all_modules": len(json.loads(everything)),
        "narrowed": len(json.loads(narrow)),
        "it_narrowed": len(json.loads(narrow)) < len(json.loads(everything)),
        "and_kept_some": len(json.loads(narrow)) == 4,
    }


def an_unknown_command_is_refused() -> dict:
    """A subcommand that does not exist, refused by argparse with its own message."""
    return {"status": _exit_code(["nothing"]), "it_refused": _exit_code(["nothing"]) != 0}


def no_command_at_all_is_refused() -> dict:
    """Running with no subcommand, which prints the usage."""
    return {"status": _exit_code([]), "it_refused": _exit_code([]) != 0}


def _exit_code(argv: Sequence[str]) -> int:
    """What argparse exits with, without letting its usage text onto the terminal.

    Argparse writes to stderr and raises SystemExit rather than returning, so a measurement that
    called main directly would print a usage message every time the suite ran. Catching both is
    two lines and keeps the output to what was asked for.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            return main(argv)
        except SystemExit as problem:
            return int(problem.code or 0)


def compare_the_commands() -> list[dict]:
    """Every subcommand and what it prints, which is the module in one table."""
    return [
        {"command": "schema", "prints": "columns and types", "reads": "the footer only"},
        {"command": "stats", "prints": "per column statistics", "reads": "the whole table"},
        {"command": "plan", "prints": "two trees", "reads": "the schema only"},
        {"command": "explain", "prints": "a tree and its strategies", "reads": "the table"},
        {"command": "cost", "prints": "predicted against counted", "reads": "the table"},
        {"command": "query", "prints": "rows", "reads": "the table"},
        {"command": "write", "prints": "what it wrote", "reads": "nothing"},
        {"command": "verify", "prints": "every check", "reads": "generated tables"},
        {"command": "measure", "prints": "every module", "reads": "generated tables"},
    ]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "commands": len(compare_the_commands()),
        "all_run": every_command_runs()["all_succeeded"],
        "errors_have_no_traceback": a_bad_query_prints_a_caret()["it_printed_no_traceback"],
        "json_parses": the_json_flag_produces_json()["the_json_parses"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
