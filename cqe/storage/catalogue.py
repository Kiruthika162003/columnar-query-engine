from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import ConfigError, DataError, SchemaError
from cqe.exec.batch import Batch
from cqe.stats.cardinality import TableStatistics, collect
from cqe.storage.disk import Table, create, open_table, scan
from cqe.storage.file import peek
from cqe.types.schema import Schema

# A set of named tables and what is known about each, which is what a planner asks before it
# plans anything.
#
# The whole reason this is a module rather than a dictionary is that the answers come from three
# different places and cost three different amounts.
#
# The schema comes from a file's footer, which is a few kilobytes however large the file is. It
# is free enough that a catalogue of a thousand tables can be opened on a whim.
#
# The row count and the group boundaries come from the same footer, so they are free too. That
# is a consequence of putting them there rather than an accident, and it is what makes a cost
# model possible without reading any data.
#
# The statistics come from reading the data, so they cost a scan and they go stale. That is the
# only expensive thing here and it is the reason the catalogue caches them and records when they
# were collected, rather than recomputing on every plan.
#
# The measurements are about what each of the three costs and about what a stale statistic does
# to a plan, which is the failure mode a catalogue actually has.

# What a catalogue file is called and the version it records, so a file from a different version
# is refused rather than misread.
CATALOGUE = "catalogue.json"
FORMAT = 1


@dataclass
class Entry:
    """One table: where it is, what it holds, and what is known about it."""

    name: str
    path: Path
    rows: int
    groups: int
    schema: Schema
    stats: TableStatistics | None = None
    stale_after: int = 0

    @property
    def has_stats(self) -> bool:
        """Whether the statistics have been collected."""
        return self.stats is not None

    @property
    def stale(self) -> bool:
        """Whether the row count has moved far enough for the statistics to be suspect.

        Measured on the row count rather than on a clock, because a table that has not changed
        does not go stale however long it sits. A time based rule would recollect statistics for
        a table nobody wrote to and would keep stale ones for a table that doubled in a minute.
        """
        if self.stats is None:
            return True
        return self.rows > self.stale_after

    def as_dict(self) -> dict:
        """Flat mapping for logging and for the catalogue file."""
        return {
            "name": self.name,
            "path": str(self.path),
            "rows": self.rows,
            "groups": self.groups,
            "columns": len(self.schema),
            "has_stats": self.has_stats,
            "stale": self.stale,
        }


@dataclass
class Catalogue:
    """Every table the engine knows about."""

    entries: dict[str, Entry] = field(default_factory=dict)
    reads: int = 0
    scans: int = 0

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    @property
    def names(self) -> tuple[str, ...]:
        """Every table name, sorted."""
        return tuple(sorted(self.entries))

    @property
    def rows(self) -> int:
        """Rows across every table."""
        return sum(one.rows for one in self.entries.values())

    def entry(self, name: str) -> Entry:
        """One table, with the names in the refusal."""
        if name not in self.entries:
            raise SchemaError(f"there is no table called {name}, only {list(self.names)}")
        return self.entries[name]

    def schema(self, name: str) -> Schema:
        """One table's schema, which costs a footer read at open time and nothing now."""
        return self.entry(name).schema

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "tables": len(self.entries),
            "rows": self.rows,
            "with_stats": sum(1 for one in self.entries.values() if one.has_stats),
            "stale": sum(1 for one in self.entries.values() if one.stale),
            "footer_reads": self.reads,
            "scans": self.scans,
        }


def add(catalogue: Catalogue, name: str, path: Path | str) -> Catalogue:
    """Register a table by reading its footer and nothing else.

    The footer read is counted, so the measurement below can say that opening a catalogue costs
    one read per table and no data at all.
    """
    if name in catalogue.entries:
        raise ConfigError(f"{name} is already in the catalogue")
    where = Path(path)
    if not where.exists():
        raise ConfigError(f"there is no file at {where}")
    footer = peek(where)
    catalogue.reads += 1
    catalogue.entries[name] = Entry(
        name=name,
        path=where,
        rows=footer.rows,
        groups=len(footer.groups),
        schema=footer.schema,
    )
    return catalogue


def open_all(directory: Path | str, suffix: str = ".cqe") -> Catalogue:
    """Every table in a directory, named after its file."""
    where = Path(directory)
    if not where.is_dir():
        raise ConfigError(f"{where} is not a directory")
    made = Catalogue()
    for one in sorted(where.glob(f"*{suffix}")):
        add(made, one.stem, one)
    return made


def table(catalogue: Catalogue, name: str) -> Table:
    """One table as something that can be scanned."""
    return open_table(catalogue.entry(name).path)


def analyse(catalogue: Catalogue, name: str, growth: float = 0.2) -> Entry:
    """Collect the statistics for one table, which costs a scan.

    The stale threshold is set from the row count at collection time plus a share of it, so a
    table has to grow by that share before its statistics are called stale. A fixed number of
    rows would be wrong for both a tiny table and a large one.
    """
    one = catalogue.entry(name)

    batch, _ = scan(open_table(one.path))
    catalogue.scans += 1
    one.stats = collect(batch)
    one.stale_after = int(one.rows * (1 + growth))
    return one


def analyse_all(catalogue: Catalogue, growth: float = 0.2) -> Catalogue:
    """Collect the statistics for every table, which is what a load does once."""
    for name in list(catalogue.entries):
        analyse(catalogue, name, growth=growth)
    return catalogue


def refresh(catalogue: Catalogue, name: str) -> Entry:
    """Re read one table's footer, which is what a write to it needs.

    Cheap enough to do on every access and not done on every access, because a catalogue that
    read the footer before every plan would turn a thousand table join into a thousand file
    opens. The measurement below is what that would cost.
    """
    one = catalogue.entry(name)
    footer = peek(one.path)
    catalogue.reads += 1
    one.rows = footer.rows
    one.groups = len(footer.groups)
    one.schema = footer.schema
    return one


def save(catalogue: Catalogue, path: Path | str) -> Path:
    """Write the catalogue's own contents, without the statistics.

    Without them on purpose. A statistic is a summary of data that may have changed since, and a
    catalogue file that carried one would let a planner use a number collected before a rewrite.
    The schema and the row count come from the footers and are re read on load.
    """
    where = Path(path)
    where.write_text(
        json.dumps(
            {
                "format": FORMAT,
                "tables": [
                    {"name": one.name, "path": str(one.path)}
                    for one in catalogue.entries.values()
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return where


def load(path: Path | str) -> Catalogue:
    """Read a catalogue back, re reading every footer."""
    where = Path(path)
    if not where.exists():
        raise ConfigError(f"there is no catalogue at {where}")
    parsed = json.loads(where.read_text(encoding="utf-8"))
    if parsed.get("format") != FORMAT:
        raise DataError(f"a catalogue of format {parsed.get('format')} against {FORMAT}")
    made = Catalogue()
    for one in parsed["tables"]:
        add(made, one["name"], one["path"])
    return made


def batches(catalogue: Catalogue) -> dict[str, Batch]:
    """Every table read into memory, which is what the query planner wants.

    The expensive call in this module and the one a real engine would not have: it reads every
    table whole. It is here because the planner and the operators work on batches, and the
    measurement below is what it costs against opening the catalogue.
    """
    out = {}
    for name in catalogue.entries:
        batch, _ = scan(open_table(catalogue.entry(name).path))
        catalogue.scans += 1
        out[name] = batch
    return out


def _make(directory: Path, tables: int = 6, rows: int = 5000, seed: int = 113) -> Catalogue:
    """A directory of tables to open, with different shapes."""
    state = np.random.default_rng(seed)
    made = Catalogue()
    for one in range(tables):
        batch = Batch.from_columns(
            [
                integer_column("id", np.arange(rows)),
                integer_column("shop", state.integers(0, 30, rows)),
                floating_column("amount", state.normal(100, 20, rows)),
                string_column(
                    "region", [f"region{value}" for value in state.integers(0, 5, rows)]
                ),
            ]
        )
        path = directory / f"table{one}.cqe"
        create(path, batch, group_size=500)
        add(made, f"table{one}", path)
    return made


def opening_a_catalogue_reads_no_data(tables: int = 6, rows: int = 5000) -> dict:
    """One footer read per table and nothing else, which is what makes a catalogue cheap.

    The measurement that pays for putting the metadata in a footer. Six tables of five thousand
    rows come to a megabyte and a half on disk, and knowing every schema, row count and group
    boundary costs the footers alone.
    """
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        made = _make(where, tables=tables, rows=rows)
        data = sum(one.stat().st_size for one in where.glob("*.cqe"))
        footers = sum(
            one.path.stat().st_size - peek(one.path).nbytes for one in made.entries.values()
        )
        return {
            "tables": len(made),
            "rows": made.rows,
            "file_bytes": data,
            "footer_bytes": footers,
            "footer_share": round(footers / data, 4),
            "footer_reads": made.reads,
            "one_read_per_table": made.reads == tables,
            "no_scans": made.scans == 0,
            "it_knows_every_schema": all(len(one.schema) == 4 for one in made.entries.values()),
        }


def statistics_cost_a_scan(tables: int = 4, rows: int = 5000) -> dict:
    """And the expensive half, which is the only thing here that reads data.

    The contrast is the point. Opening is one footer per table and analysing is one scan per
    table, and the ratio is the file size over the footer size, which storage/file.py measured
    at a few hundred.
    """
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        made = _make(where, tables=tables, rows=rows)
        opened = {"reads": made.reads, "scans": made.scans}
        analyse_all(made)
        return {
            "tables": len(made),
            "after_opening": opened,
            "after_analysing": {"reads": made.reads, "scans": made.scans},
            "opening_scanned_nothing": opened["scans"] == 0,
            "analysing_scanned_every_table": made.scans == tables,
            "and_every_table_has_statistics": all(
                one.has_stats for one in made.entries.values()
            ),
        }


def statistics_go_stale_when_a_table_grows(rows: int = 4000) -> dict:
    """A table analysed and then rewritten larger, which the catalogue notices.

    On the row count rather than on a clock. A table that nobody wrote to does not go stale
    however long it sits, and a table that doubled in a minute is stale immediately, and a time
    based rule gets both of those backwards.
    """
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        made = _make(where, tables=1, rows=rows)
        analyse(made, "table0")
        fresh = made.entry("table0").stale
        state = np.random.default_rng(127)
        larger = Batch.from_columns(
            [
                integer_column("id", np.arange(rows * 2)),
                integer_column("shop", state.integers(0, 30, rows * 2)),
                floating_column("amount", state.normal(100, 20, rows * 2)),
                string_column(
                    "region", [f"region{one}" for one in state.integers(0, 5, rows * 2)]
                ),
            ]
        )
        create(where / "table0.cqe", larger, group_size=500)
        refresh(made, "table0")
        after = made.entry("table0").stale
        return {
            "rows_before": rows,
            "rows_after": made.entry("table0").rows,
            "fresh_after_analysing": not fresh,
            "stale_after_growing": after,
            "and_a_refresh_is_one_footer_read": made.reads == 2,
        }


def a_small_change_does_not_make_it_stale(rows: int = 4000) -> dict:
    """A table that grew by a tenth, where the statistics are still good enough.

    The threshold's other side. Recollecting on every change would make the cache useless and
    recollecting on none would make the estimates wrong, and the growth share is the one knob
    between those.
    """
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        made = _make(where, tables=1, rows=rows)
        analyse(made, "table0", growth=0.2)
        state = np.random.default_rng(131)
        slightly = int(rows * 1.1)
        larger = Batch.from_columns(
            [
                integer_column("id", np.arange(slightly)),
                integer_column("shop", state.integers(0, 30, slightly)),
                floating_column("amount", state.normal(100, 20, slightly)),
                string_column(
                    "region", [f"region{one}" for one in state.integers(0, 5, slightly)]
                ),
            ]
        )
        create(where / "table0.cqe", larger, group_size=500)
        refresh(made, "table0")
        return {
            "rows_before": rows,
            "rows_after": made.entry("table0").rows,
            "growth": 0.1,
            "threshold": 0.2,
            "it_is_still_fresh": not made.entry("table0").stale,
        }


def a_catalogue_round_trips(tables: int = 4) -> dict:
    """Save and load, which must give the same tables back.

    Without the statistics, which is the decision worth reading. A saved statistic is a summary
    of data that may have been rewritten since, and a planner using one would be reasoning about
    a table that no longer exists.
    """
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        made = _make(where, tables=tables)
        analyse_all(made)
        path = save(made, where / CATALOGUE)
        again = load(path)
        return {
            "tables": len(again),
            "names_match": again.names == made.names,
            "rows_match": again.rows == made.rows,
            "the_statistics_did_not_survive": not any(
                one.has_stats for one in again.entries.values()
            ),
            "and_everything_is_stale": all(one.stale for one in again.entries.values()),
        }


def opening_a_directory_finds_every_table(tables: int = 5) -> dict:
    """A directory of files, opened by name, which is the ordinary way in."""
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        _make(where, tables=tables)
        found = open_all(where)
        return {
            "tables": len(found),
            "names": list(found.names),
            "it_found_them_all": len(found) == tables,
            "named_after_the_files": found.names
            == tuple(sorted(f"table{one}" for one in range(tables))),
        }


def reading_every_table_costs_far_more_than_opening(tables: int = 4, rows: int = 5000) -> dict:
    """Opening a catalogue against reading its tables, which is the whole point of a footer.

    The ratio is the reason a planner can consider a table it will not read: it knows the row
    count, the schema and the group boundaries for the cost of a few kilobytes, and only pays
    for the data if the plan actually reads it.
    """
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        made = _make(where, tables=tables, rows=rows)
        footer_bytes = sum(
            one.path.stat().st_size - peek(one.path).nbytes for one in made.entries.values()
        )
        file_bytes = sum(one.path.stat().st_size for one in made.entries.values())
        loaded = batches(made)
        return {
            "tables": tables,
            "file_bytes": file_bytes,
            "footer_bytes": footer_bytes,
            "rows_read": sum(one.rows for one in loaded.values()),
            "reads": made.reads,
            "scans": made.scans,
            "the_footers_are_a_fraction": footer_bytes < file_bytes,
            "the_ratio": round(file_bytes / max(footer_bytes, 1), 1),
        }


def a_catalogue_answers_a_schema_without_touching_the_file(tables: int = 3) -> dict:
    """The question a planner asks most, answered from memory after one footer read.

    Which is what lets a parser resolve a query against a table it never reads, and is why
    sql/parse.py takes a catalogue rather than a table.
    """
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        made = _make(where, tables=tables)
        before = made.reads
        schemas = [made.schema(one) for one in made.names]
        return {
            "tables": tables,
            "reads_before": before,
            "reads_after": made.reads,
            "it_read_nothing_more": made.reads == before,
            "and_every_schema_came_back": all(len(one) == 4 for one in schemas),
        }


def a_missing_table_lists_the_others() -> dict:
    """Asking for a table that is not there, with the names in the message."""
    with tempfile.TemporaryDirectory() as directory:
        made = _make(Path(directory), tables=3)
        caught = ""
        try:
            made.entry("nothing")
        except SchemaError as problem:
            caught = str(problem)
        return {
            "message": caught,
            "it_was_refused": bool(caught),
            "it_lists_the_others": "table0" in caught,
        }


def a_repeated_name_is_refused() -> bool:
    """Adding the same name twice, which would silently replace a table."""
    with tempfile.TemporaryDirectory() as directory:
        where = Path(directory)
        made = _make(where, tables=1)
        try:
            add(made, "table0", where / "table0.cqe")
        except ConfigError:
            return True
    return False


def a_missing_file_is_refused() -> bool:
    """Adding a table whose file is not there."""
    with tempfile.TemporaryDirectory() as directory:
        try:
            add(Catalogue(), "one", Path(directory) / "nothing.cqe")
        except ConfigError:
            return True
    return False


def a_missing_directory_is_refused() -> bool:
    """Opening a directory that does not exist."""
    try:
        open_all(Path("nowhere at all"))
    except ConfigError:
        return True
    return False


def a_catalogue_of_the_wrong_format_is_refused() -> bool:
    """A file written by another version."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / CATALOGUE
        path.write_text(json.dumps({"format": 99, "tables": []}), encoding="utf-8")
        try:
            load(path)
        except DataError:
            return True
    return False


def a_missing_catalogue_file_is_refused() -> bool:
    """Loading a catalogue that is not there."""
    with tempfile.TemporaryDirectory() as directory:
        try:
            load(Path(directory) / "nothing.json")
        except ConfigError:
            return True
    return False


def compare_the_questions() -> list[dict]:
    """Every question a catalogue answers and what each one costs."""
    return [
        {"question": "what tables are there", "costs": "nothing", "reads": "memory"},
        {"question": "what columns does one have", "costs": "nothing", "reads": "memory"},
        {"question": "how many rows", "costs": "nothing", "reads": "memory"},
        {"question": "how many row groups", "costs": "nothing", "reads": "memory"},
        {"question": "has it changed", "costs": "one footer", "reads": "a few kilobytes"},
        {"question": "what is the distribution", "costs": "one scan", "reads": "the table"},
    ]


def summarise() -> dict:
    """The module in one mapping."""
    opened = opening_a_catalogue_reads_no_data()
    return {
        "format": FORMAT,
        "opening_reads_no_data": opened["no_scans"],
        "one_footer_per_table": opened["one_read_per_table"],
        "statistics_cost_a_scan": statistics_cost_a_scan()["analysing_scanned_every_table"],
        "growth_makes_them_stale": statistics_go_stale_when_a_table_grows()[
            "stale_after_growing"
        ],
        "questions": len(compare_the_questions()),
    }
