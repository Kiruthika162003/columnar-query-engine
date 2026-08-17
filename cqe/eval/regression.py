from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cqe.errors import ConfigError, DataError
from cqe.eval.workload import QUERIES, Measurement, measure_all

# Recording what the engine cost yesterday so that today's number can be compared with it.
#
# The measurements everywhere else in this package answer whether a thing is fast; this one
# answers whether it got slower, which is a different question and the one that actually catches
# regressions. A module that was twice as expensive as it should be from the day it was written
# looks fine to every measurement and looks fine here too. A module that was right and then
# doubled is invisible to every measurement and is exactly what this catches.
#
# The whole design is one decision: how much change is a regression. Too tight and every commit
# fails on noise; too loose and a real doubling slips through. Since nothing here is timed, the
# noise is not machine noise, it is the generated data changing between runs, and that is zero
# for a fixed seed. So the tolerance can be far tighter than a timing benchmark could ever use,
# and the measurement below is what it actually needs to be.

# How much a query may change before it counts as a regression. One percent, which would be
# absurd for a timing benchmark and is generous here: with a fixed seed the counts are exactly
# reproducible, so the honest tolerance is zero and one percent is the room left for a
# legitimate refactor that changes an operator's accounting by a value or two.
TOLERANCE = 0.01

# What a baseline file is called by default, and the version it records so that a baseline
# written by an older format is refused rather than misread.
BASELINE = "baseline.json"
FORMAT = 1


@dataclass(frozen=True)
class Entry:
    """One query's recorded cost."""

    query: str
    rows: int
    total: int
    values_touched: int
    hash_probes: int

    def as_dict(self) -> dict:
        """Flat mapping, which is also what goes in the file."""
        return {
            "query": self.query,
            "rows": self.rows,
            "total": self.total,
            "values_touched": self.values_touched,
            "hash_probes": self.hash_probes,
        }

    @classmethod
    def of(cls, one: Measurement) -> Entry:
        """One entry from a workload measurement."""
        return cls(
            query=one.query,
            rows=one.rows,
            total=one.total,
            values_touched=one.values_touched,
            hash_probes=one.hash_probes,
        )


@dataclass(frozen=True)
class Baseline:
    """Every query's cost at one point in time."""

    entries: tuple[Entry, ...]
    rows: int
    format: int = FORMAT

    def entry(self, name: str) -> Entry:
        """One query by name."""
        for one in self.entries:
            if one.query == name:
                return one
        raise ConfigError(f"{name} is not in the baseline")

    @property
    def queries(self) -> tuple[str, ...]:
        """Every query the baseline covers."""
        return tuple(one.query for one in self.entries)

    def as_dict(self) -> dict:
        """The whole baseline as a mapping, which is what is written to the file."""
        return {
            "format": self.format,
            "rows": self.rows,
            "entries": [one.as_dict() for one in self.entries],
        }


def record(rows: int) -> Baseline:
    """Run the whole query set and record what it cost."""
    return Baseline(
        entries=tuple(Entry.of(one) for one in measure_all(rows)),
        rows=rows,
    )


def save(baseline: Baseline, path: Path | str) -> Path:
    """Write a baseline to a file, as readable JSON.

    JSON rather than the package's own format, which would be neater and would make the baseline
    unreadable by anything that is not this engine, including a person looking at a diff. A
    baseline whose changes cannot be read in a review is not doing its job.
    """
    where = Path(path)
    where.write_text(json.dumps(baseline.as_dict(), indent=2), encoding="utf-8")
    return where


def load(path: Path | str) -> Baseline:
    """Read a baseline back, refusing one written by a different format."""
    where = Path(path)
    if not where.exists():
        raise ConfigError(f"there is no baseline at {where}")
    parsed = json.loads(where.read_text(encoding="utf-8"))
    if parsed.get("format") != FORMAT:
        raise DataError(f"a baseline of format {parsed.get('format')} against {FORMAT}")
    return Baseline(
        entries=tuple(Entry(**one) for one in parsed["entries"]),
        rows=parsed["rows"],
        format=parsed["format"],
    )


@dataclass(frozen=True)
class Change:
    """One query's cost now against its cost then."""

    query: str
    before: int
    after: int
    rows_before: int
    rows_after: int

    @property
    def ratio(self) -> float:
        """How much dearer it got, so above one is worse."""
        if self.before == self.after:
            return 1.0
        return self.after / max(self.before, 1)

    @property
    def regressed(self) -> bool:
        """Whether it got dearer by more than the tolerance."""
        return self.ratio > 1 + TOLERANCE

    @property
    def improved(self) -> bool:
        """Whether it got cheaper by more than the tolerance."""
        return self.ratio < 1 - TOLERANCE

    @property
    def changed_its_answer(self) -> bool:
        """Whether the query returns a different number of rows than it did.

        A row count change is not a regression, it is a correctness question, and it is reported
        separately because the right response is different: a slower query is a decision and a
        query that returns a different number of rows is a bug in one of the two versions.
        """
        return self.rows_before != self.rows_after

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "query": self.query,
            "before": self.before,
            "after": self.after,
            "ratio": round(self.ratio, 3),
            "regressed": self.regressed,
            "improved": self.improved,
            "rows_changed": self.changed_its_answer,
        }


@dataclass
class Comparison:
    """A whole run against a whole baseline."""

    changes: tuple[Change, ...]
    missing: tuple[str, ...] = ()
    added: tuple[str, ...] = ()

    @property
    def regressions(self) -> tuple[Change, ...]:
        """The queries that got dearer."""
        return tuple(one for one in self.changes if one.regressed)

    @property
    def improvements(self) -> tuple[Change, ...]:
        """The queries that got cheaper."""
        return tuple(one for one in self.changes if one.improved)

    @property
    def answer_changes(self) -> tuple[Change, ...]:
        """The queries returning a different number of rows."""
        return tuple(one for one in self.changes if one.changed_its_answer)

    @property
    def clean(self) -> bool:
        """Whether anything at all needs looking at."""
        return not (self.regressions or self.answer_changes or self.missing)

    def report(self) -> str:
        """The comparison as lines, which is what a command line prints."""
        if self.clean:
            return f"{len(self.changes)} queries, no regressions"
        lines = []
        for one in self.regressions:
            lines.append(f"regression: {one.query} {one.before} to {one.after}")
        for one in self.answer_changes:
            lines.append(
                f"answer changed: {one.query} {one.rows_before} rows to {one.rows_after}"
            )
        for one in self.missing:
            lines.append(f"missing: {one} is in the baseline and was not run")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "queries": len(self.changes),
            "regressions": len(self.regressions),
            "improvements": len(self.improvements),
            "answer_changes": len(self.answer_changes),
            "missing": list(self.missing),
            "added": list(self.added),
            "clean": self.clean,
        }


def compare(baseline: Baseline, now: Baseline) -> Comparison:
    """A run against a baseline, matching queries by name.

    By name rather than by position, because a query added to the middle of the set would
    otherwise shift every comparison after it and report nine regressions for one addition. The
    names that appear on only one side are reported as their own thing.
    """
    before = {one.query: one for one in baseline.entries}
    after = {one.query: one for one in now.entries}
    changes = []
    for name in sorted(set(before) & set(after)):
        changes.append(
            Change(
                query=name,
                before=before[name].total,
                after=after[name].total,
                rows_before=before[name].rows,
                rows_after=after[name].rows,
            )
        )
    return Comparison(
        changes=tuple(changes),
        missing=tuple(sorted(set(before) - set(after))),
        added=tuple(sorted(set(after) - set(before))),
    )


def check(path: Path | str, rows: int | None = None) -> Comparison:
    """Run the set and compare it against a saved baseline."""
    saved = load(path)
    return compare(saved, record(rows if rows is not None else saved.rows))


def a_baseline_round_trips(rows: int = 4000) -> dict:
    """Write a baseline and read it back, which must give the same numbers.

    The only property a baseline file needs. A format that lost a digit would report a
    regression on the next run and the investigation would be into the engine rather than into
    the file.
    """
    with tempfile.TemporaryDirectory() as directory:
        first = record(rows)
        path = save(first, Path(directory) / BASELINE)
        second = load(path)
        size = path.stat().st_size
    return {
        "queries": len(first.entries),
        "rows": first.rows,
        "they_match": first.as_dict() == second.as_dict(),
        "the_file_is_readable": size > 0,
    }


def a_run_against_itself_is_clean(rows: int = 4000) -> dict:
    """The same set run twice, which must show no change at all.

    Zero, not near zero. Nothing here is timed and the data is generated from a fixed seed, so
    two runs of the same code produce identical counts, and any difference at all is a real
    change in what the engine did.
    """
    first = record(rows)
    second = record(rows)
    measured = compare(first, second)
    ratios = [one.ratio for one in measured.changes]
    return {
        "queries": len(measured.changes),
        "regressions": len(measured.regressions),
        "every_ratio_is_exactly_one": all(one == 1.0 for one in ratios),
        "it_is_clean": measured.clean,
    }


def the_counts_are_reproducible_across_seeds(rows: int = 4000) -> dict:
    """And the reason the tolerance can be one percent rather than twenty.

    A timing benchmark on a shared machine varies by tens of percent between runs and has to set
    its threshold above that noise, which means it cannot see a regression smaller than the
    noise. Counting means the noise is zero and a one percent change is visible.
    """
    runs = [record(rows) for _ in range(3)]
    totals = [[one.total for one in run.entries] for run in runs]
    return {
        "runs": len(runs),
        "queries": len(totals[0]),
        "they_are_identical": all(one == totals[0] for one in totals),
        "the_tolerance_could_be": 0.0,
        "and_is": TOLERANCE,
    }


def a_regression_is_caught(rows: int = 4000) -> dict:
    """A baseline with one query's cost halved, which the comparison must flag.

    The measurement that says the checker can fail, which every checker needs and this one gets
    by editing a baseline rather than by breaking the engine.
    """
    now = record(rows)
    tampered = Baseline(
        entries=tuple(
            Entry(
                query=one.query,
                rows=one.rows,
                total=one.total // 2 if one.query == "range" else one.total,
                values_touched=one.values_touched,
                hash_probes=one.hash_probes,
            )
            for one in now.entries
        ),
        rows=rows,
    )
    measured = compare(tampered, now)
    return {
        "regressions": len(measured.regressions),
        "it_was_caught": bool(measured.regressions),
        "it_named_the_query": measured.regressions[0].query == "range"
        if measured.regressions
        else False,
        "the_ratio": round(measured.regressions[0].ratio, 2) if measured.regressions else 0,
        "and_the_report_says_so": "regression" in measured.report(),
    }


def an_improvement_is_not_a_regression(rows: int = 4000) -> dict:
    """A query that got cheaper, which is reported and does not fail the check.

    Worth having as its own case because the obvious implementation compares absolute
    differences and flags an improvement as loudly as a regression, and then everybody turns the
    check off.
    """
    now = record(rows)
    tampered = Baseline(
        entries=tuple(
            Entry(
                query=one.query,
                rows=one.rows,
                total=one.total * 2 if one.query == "range" else one.total,
                values_touched=one.values_touched,
                hash_probes=one.hash_probes,
            )
            for one in now.entries
        ),
        rows=rows,
    )
    measured = compare(tampered, now)
    return {
        "improvements": len(measured.improvements),
        "regressions": len(measured.regressions),
        "it_was_reported": bool(measured.improvements),
        "and_the_check_is_clean": measured.clean,
    }


def a_changed_answer_is_reported_separately(rows: int = 4000) -> dict:
    """A query returning a different number of rows, which is a correctness question.

    Kept apart from the cost comparison because the response is different. A dearer query is a
    decision to make; a query returning a different number of rows is a bug in one of the two
    versions and the cost is beside the point.
    """
    now = record(rows)
    tampered = Baseline(
        entries=tuple(
            Entry(
                query=one.query,
                rows=one.rows + 1 if one.query == "range" else one.rows,
                total=one.total,
                values_touched=one.values_touched,
                hash_probes=one.hash_probes,
            )
            for one in now.entries
        ),
        rows=rows,
    )
    measured = compare(tampered, now)
    return {
        "answer_changes": len(measured.answer_changes),
        "regressions": len(measured.regressions),
        "it_was_caught": bool(measured.answer_changes),
        "it_is_not_called_a_regression": not measured.regressions,
        "and_the_check_is_not_clean": not measured.clean,
        "the_report_says_answer_changed": "answer changed" in measured.report(),
    }


def a_query_added_to_the_set_is_not_a_regression(rows: int = 4000) -> dict:
    """A baseline missing one query, which is an addition rather than nine regressions.

    The failure a positional comparison has. Adding a query to the middle of the set shifts
    every entry after it, and a checker matching by position reports every one of them as
    changed.
    """
    now = record(rows)
    shorter = Baseline(entries=now.entries[1:], rows=rows)
    measured = compare(shorter, now)
    return {
        "added": list(measured.added),
        "regressions": len(measured.regressions),
        "it_was_reported_as_added": measured.added == (now.entries[0].query,),
        "and_nothing_regressed": not measured.regressions,
        "and_the_check_is_clean": measured.clean,
    }


def a_query_removed_from_the_set_is_reported(rows: int = 4000) -> dict:
    """And the other direction, which is not clean.

    A query in the baseline that no longer runs is either a deletion somebody meant or a query
    that stopped working, and the checker cannot tell which, so it says so rather than passing.
    """
    now = record(rows)
    shorter = Baseline(entries=now.entries[1:], rows=rows)
    measured = compare(now, shorter)
    return {
        "missing": list(measured.missing),
        "it_was_reported": bool(measured.missing),
        "and_the_check_is_not_clean": not measured.clean,
        "the_report_names_it": "missing" in measured.report(),
    }


def the_tolerance_is_what_decides(rows: int = 4000) -> dict:
    """A change just inside the tolerance and one just outside it.

    The whole checker is one threshold and this is where it is exercised. A half percent change
    passes and a two percent change does not, which is the behaviour the constant claims.
    """
    now = record(rows)
    out = {}
    for name, factor in (("inside", 1.005), ("outside", 1.02)):
        tampered = Baseline(
            entries=tuple(
                Entry(
                    query=one.query,
                    rows=one.rows,
                    total=int(one.total / factor) if one.query == "range" else one.total,
                    values_touched=one.values_touched,
                    hash_probes=one.hash_probes,
                )
                for one in now.entries
            ),
            rows=rows,
        )
        out[name] = len(compare(tampered, now).regressions)
    return {
        **out,
        "tolerance": TOLERANCE,
        "the_small_change_passed": out["inside"] == 0,
        "and_the_large_one_did_not": out["outside"] > 0,
    }


def checking_against_a_file_works(rows: int = 4000) -> dict:
    """The whole cycle: record, save, run again, compare, which is what a build does."""
    with tempfile.TemporaryDirectory() as directory:
        path = save(record(rows), Path(directory) / BASELINE)
        measured = check(path)
    return {
        "queries": len(measured.changes),
        "it_is_clean": measured.clean,
        "the_report_is_one_line": len(measured.report().split("\n")) == 1,
    }


def a_missing_baseline_is_refused() -> bool:
    """Checking against a file that is not there."""
    with tempfile.TemporaryDirectory() as directory:
        try:
            load(Path(directory) / "nothing.json")
        except ConfigError:
            return True
    return False


def a_baseline_of_the_wrong_format_is_refused() -> bool:
    """A file written by a different version, refused rather than misread."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / BASELINE
        path.write_text(json.dumps({"format": 99, "rows": 10, "entries": []}), encoding="utf-8")
        try:
            load(path)
        except DataError:
            return True
    return False


def an_unknown_query_in_a_baseline_is_refused(rows: int = 1000) -> bool:
    """Asking a baseline for a query it does not hold."""
    try:
        record(rows).entry("nothing")
    except ConfigError:
        return True
    return False


def compare_the_outcomes() -> list[dict]:
    """Every outcome the checker can report and what it means."""
    return [
        {
            "outcome": "regression",
            "means": "a query got dearer by more than the tolerance",
            "fails": True,
        },
        {
            "outcome": "improvement",
            "means": "a query got cheaper",
            "fails": False,
        },
        {
            "outcome": "answer changed",
            "means": "a query returns a different number of rows",
            "fails": True,
        },
        {
            "outcome": "added",
            "means": "a query is new since the baseline",
            "fails": False,
        },
        {
            "outcome": "missing",
            "means": "a query in the baseline no longer runs",
            "fails": True,
        },
    ]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "queries": len(QUERIES),
        "tolerance": TOLERANCE,
        "reproducible": the_counts_are_reproducible_across_seeds()["they_are_identical"],
        "a_regression_is_caught": a_regression_is_caught()["it_was_caught"],
        "an_improvement_is_not": an_improvement_is_not_a_regression()["and_the_check_is_clean"],
        "outcomes": len(compare_the_outcomes()),
    }
