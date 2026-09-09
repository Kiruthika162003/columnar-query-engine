# Running columnar-query-engine

A columnar store with a cost based query planner, written in Python. This file covers how to build
the project, run its tests, and execute it. Every command below was run from a
clean checkout of this repository before being written down.

## Requirements

Python 3.11 or newer, plus `numpy>=1.26`.

## Setup

Install the package and its dependencies from the repository root:

```bash
pip install -e .
```

## Run the tests

```bash
python -m pytest -q
```

The suite is the primary check. It runs from the repository root with no
arguments and no configuration, and it prints the number of tests it ran. A
non-zero exit status means something is wrong. Read the printed summary rather
than a shell pipeline, because piping the output through another command
replaces the real exit code with that of the last command in the pipe.

## Lint

```bash
python -m ruff check .
```

The lint configuration lives in `pyproject.toml`. It passes with no findings.

## Run the command line tool

```bash
python -m cqe.cli.main --help
```

The subcommands are `schema`, `stats`, `plan`, `explain`, `cost`, `query`, `write`, `verify`, `measure`.

For example:

```bash
python -m cqe.cli.main schema
```

## Run a worked example

There are 4 runnable examples in `examples/`. Each exposes `run()`, which
returns the lines it would print, and `main()`, which prints them:

```bash
python -c "from examples.explain_a_query import main; main()"
```

The full list is `explain_a_query`, `keep_a_table_tidy`, `measure_everything`, `query_a_file`.

## Layout

- `cqe/` the package itself
- `cqe/cost/` the verification organ, a set of measured claims about the
  package as a whole rather than unit tests of one function
- `tests/` the test suite
- `examples/` runnable end to end scenarios

## Notes

The examples are pinned line by line in the test suite, so an example whose
output drifts fails the build rather than quietly changing. Where a guess about
behaviour was refuted by measurement, the wrong guess is kept in the source
beside the measured value rather than deleted.
