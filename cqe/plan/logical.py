from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from cqe.errors import ConfigError, PlanError, SchemaError, UnknownColumn
from cqe.exec.aggregate import Aggregate
from cqe.exec.batch import Batch
from cqe.exec.expr import Expr, all_of, conjuncts, describe
from cqe.exec.sort import SortKey
from cqe.types.schema import INTEGER, Field, Schema

# The logical plan: what a query asks for, with nothing said about how.
#
# Seven node types, each holding its inputs and its own parameters and nothing else. A scan
# names a table. A filter holds a predicate. A project holds a list of names. A join holds two
# inputs and a pair of key lists. Aggregate, sort and limit hold what they are called after.
#
# The separation from the physical plan is the point of having two. A logical join says two
# tables are matched on a key; a physical join says which side builds the hash table, or that
# there is no hash table because both sides arrive sorted. Rewriting a logical plan cannot
# change the answer, only the work, and every rule in plan/rules is a function from logical plan
# to logical plan for exactly that reason.
#
# Two properties are enforced here rather than left to the rules.
#
# The schema of every node is computable from its inputs, so a rule that produces a plan
# referring to a column nobody produces fails at the point it is built rather than at the point
# it is run. That is the single most useful thing a plan representation can do, because a rule
# that drops a column is easy to write and hard to notice.
#
# And every node is immutable. A rule returns a new plan rather than editing one, so the
# original survives for comparison. plan/rules measures every rewrite against the plan it
# replaced, and that is only possible because the plan it replaced still exists.


@dataclass(frozen=True)
class Plan:
    """The base of the logical plan tree."""

    def schema(self) -> Schema:
        """The columns this node produces."""
        raise NotImplementedError

    def children(self) -> tuple[Plan, ...]:
        """The inputs this node reads."""
        return ()

    def rows(self) -> float:
        """An estimate of the rows this node produces, before any statistics are available.

        A structural guess used only where no statistics exist. Every rule that cares about row
        counts takes them from stats/cardinality.py instead, and this exists so a plan can be
        printed and compared before statistics have been collected.
        """
        return max((child.rows() for child in self.children()), default=0.0)

    def height(self) -> int:
        """How deep the tree is."""
        return 1 + max((child.height() for child in self.children()), default=0)

    def nodes(self) -> int:
        """How many nodes the tree holds."""
        return 1 + sum(child.nodes() for child in self.children())

    def columns_used(self) -> frozenset[str]:
        """Every column this node reads from its inputs."""
        return frozenset()

    def describe(self) -> str:
        """One line for this node, without its children."""
        return type(self).__name__

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "node": type(self).__name__,
            "height": self.height(),
            "nodes": self.nodes(),
        }


@dataclass(frozen=True)
class Scan(Plan):
    """Read a table, optionally only some of its columns."""

    name: str
    table_schema: Schema
    projected: tuple[str, ...] | None = None
    row_count: int = 0
    pushed: tuple[Expr, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise PlanError("a scan needs a table name")
        if self.projected is not None:
            missing = [one for one in self.projected if one not in self.table_schema]
            if missing:
                raise UnknownColumn(f"{missing} not in {list(self.table_schema.names)}")

    def schema(self) -> Schema:
        """The projected columns, or all of them."""
        if self.projected is None:
            return self.table_schema
        return self.table_schema.select(list(self.projected))

    def rows(self) -> float:
        """What the catalogue said, unadjusted for anything pushed into it."""
        return float(self.row_count)

    def columns_read(self) -> tuple[str, ...]:
        """The columns a reader would actually fetch, which is what a scan costs."""
        return self.projected if self.projected is not None else self.table_schema.names

    def columns_used(self) -> frozenset[str]:
        """The columns any pushed predicate reads.

        Not the projected columns, which are an output rather than a requirement. This is the
        input side, and it exists because projection pushdown asks every node what it reads and
        a scan with a predicate pushed into it reads more than it produces.

        Leaving it empty was a bug the measurements caught. Running predicate pushdown and then
        projection pushdown narrowed a scan to the query's output columns and dropped the column
        the pushed predicate needed, so the plan built cleanly and failed at the point the
        predicate was evaluated.
        """
        if not self.pushed:
            return frozenset()
        return frozenset().union(*(one.columns_used() for one in self.pushed))

    def with_projection(self, names: Sequence[str]) -> Scan:
        """The same scan reading only the named columns."""
        return Scan(
            name=self.name,
            table_schema=self.table_schema,
            projected=tuple(names),
            row_count=self.row_count,
            pushed=self.pushed,
        )

    def with_predicate(self, predicate: Expr) -> Scan:
        """The same scan with one more predicate pushed into it.

        Kept as a tuple rather than one combined expression, because the reader uses them
        separately: each conjunct is tested against the row group statistics on its own, and a
        combined expression would have to be split again to do that.
        """
        return Scan(
            name=self.name,
            table_schema=self.table_schema,
            projected=self.projected,
            row_count=self.row_count,
            pushed=(*self.pushed, predicate),
        )

    def describe(self) -> str:
        """One line naming the table and what was pushed into it."""
        columns = "*" if self.projected is None else ", ".join(self.projected)
        pushed = (
            ""
            if not self.pushed
            else " where " + " and ".join(describe(one) for one in self.pushed)
        )
        return f"Scan {self.name} [{columns}]{pushed}"


@dataclass(frozen=True)
class Filter(Plan):
    """Keep the rows a predicate accepts."""

    input: Plan
    predicate: Expr

    def schema(self) -> Schema:
        """A filter does not change the columns."""
        return self.input.schema()

    def children(self) -> tuple[Plan, ...]:
        """Its one input."""
        return (self.input,)

    def columns_used(self) -> frozenset[str]:
        """The predicate's columns."""
        return self.predicate.columns_used()

    def describe(self) -> str:
        """One line showing the predicate."""
        return f"Filter {describe(self.predicate)}"


@dataclass(frozen=True)
class Project(Plan):
    """Keep some columns, in the order given."""

    input: Plan
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise PlanError("a projection needs at least one column")
        missing = [one for one in self.names if one not in self.input.schema()]
        if missing:
            raise UnknownColumn(f"{missing} not in {list(self.input.schema().names)}")

    def schema(self) -> Schema:
        """The named columns."""
        return self.input.schema().select(list(self.names))

    def children(self) -> tuple[Plan, ...]:
        """Its one input."""
        return (self.input,)

    def columns_used(self) -> frozenset[str]:
        """The names it keeps."""
        return frozenset(self.names)

    def describe(self) -> str:
        """One line listing the columns."""
        return f"Project [{', '.join(self.names)}]"


@dataclass(frozen=True)
class Join(Plan):
    """Match two inputs on a pair of key lists."""

    left: Plan
    right: Plan
    left_keys: tuple[str, ...]
    right_keys: tuple[str, ...]
    suffix: str = "_right"

    def __post_init__(self) -> None:
        if not self.left_keys:
            raise PlanError("a join needs at least one key")
        if len(self.left_keys) != len(self.right_keys):
            raise PlanError(
                f"{len(self.left_keys)} left keys against {len(self.right_keys)} right"
            )
        missing = [one for one in self.left_keys if one not in self.left.schema()]
        if missing:
            raise UnknownColumn(f"{missing} not in the left input")
        missing = [one for one in self.right_keys if one not in self.right.schema()]
        if missing:
            raise UnknownColumn(f"{missing} not in the right input")

    def schema(self) -> Schema:
        """Both inputs side by side, with repeated names suffixed."""
        return self.left.schema().joined(self.right.schema(), suffix=self.suffix)

    def children(self) -> tuple[Plan, ...]:
        """Both inputs."""
        return (self.left, self.right)

    def columns_used(self) -> frozenset[str]:
        """The key columns from both sides."""
        return frozenset(self.left_keys) | frozenset(self.right_keys)

    def rows(self) -> float:
        """The product, which is the only structural bound without statistics."""
        return self.left.rows() * self.right.rows()

    def swapped(self) -> Join:
        """The same join with the inputs exchanged.

        The rewrite join ordering is built on. It changes the output column order, so it is not
        a pure no operation, and plan/rules/ordering.py measures what that costs against what
        the reordering saves.
        """
        return Join(
            left=self.right,
            right=self.left,
            left_keys=self.right_keys,
            right_keys=self.left_keys,
            suffix=self.suffix,
        )

    def describe(self) -> str:
        """One line showing the keys."""
        pairs = ", ".join(
            f"{one} = {other}"
            for one, other in zip(self.left_keys, self.right_keys, strict=True)
        )
        return f"Join on {pairs}"


@dataclass(frozen=True)
class Group(Plan):
    """Collect rows by key and reduce each group."""

    input: Plan
    keys: tuple[str, ...]
    aggregates: tuple[Aggregate, ...]

    def __post_init__(self) -> None:
        if not self.aggregates:
            raise PlanError("a group by needs at least one aggregate")
        missing = [one for one in self.keys if one not in self.input.schema()]
        if missing:
            raise UnknownColumn(f"{missing} not in the input")
        for one in self.aggregates:
            if one.source and one.source not in self.input.schema():
                raise UnknownColumn(f"{one.source} not in the input")

    def schema(self) -> Schema:
        """The key columns followed by one column per aggregate."""
        fields = [self.input.schema().field(name) for name in self.keys]
        for one in self.aggregates:
            source = self.input.schema().logical(one.source) if one.source else INTEGER
            fields.append(Field(name=one.name, logical=one.result_type(source)))
        return Schema(tuple(fields))

    def children(self) -> tuple[Plan, ...]:
        """Its one input."""
        return (self.input,)

    def columns_used(self) -> frozenset[str]:
        """The keys and every aggregate's source."""
        sources = {one.source for one in self.aggregates if one.source}
        return frozenset(self.keys) | frozenset(sources)

    def rows(self) -> float:
        """One row per group, bounded by the input without statistics to say more."""
        return self.input.rows()

    def describe(self) -> str:
        """One line showing the keys and the aggregates."""
        names = ", ".join(one.name for one in self.aggregates)
        return f"Group by [{', '.join(self.keys)}] producing [{names}]"


@dataclass(frozen=True)
class Sort(Plan):
    """Order the rows."""

    input: Plan
    keys: tuple[SortKey, ...]

    def __post_init__(self) -> None:
        if not self.keys:
            raise PlanError("a sort needs at least one key")
        missing = [one.name for one in self.keys if one.name not in self.input.schema()]
        if missing:
            raise UnknownColumn(f"{missing} not in the input")

    def schema(self) -> Schema:
        """A sort does not change the columns."""
        return self.input.schema()

    def children(self) -> tuple[Plan, ...]:
        """Its one input."""
        return (self.input,)

    def columns_used(self) -> frozenset[str]:
        """The key columns."""
        return frozenset(one.name for one in self.keys)

    def describe(self) -> str:
        """One line showing the keys and their directions."""
        parts = ", ".join(f"{one.name}{' desc' if one.descending else ''}" for one in self.keys)
        return f"Sort by [{parts}]"


@dataclass(frozen=True)
class Limit(Plan):
    """Keep the first rows."""

    input: Plan
    count: int
    offset: int = 0

    def __post_init__(self) -> None:
        if self.count < 0 or self.offset < 0:
            raise PlanError(f"limit {self.count} offset {self.offset} is not a window")

    def schema(self) -> Schema:
        """A limit does not change the columns."""
        return self.input.schema()

    def children(self) -> tuple[Plan, ...]:
        """Its one input."""
        return (self.input,)

    def rows(self) -> float:
        """At most the count."""
        return min(float(self.count), self.input.rows())

    def describe(self) -> str:
        """One line showing the window."""
        skip = f" offset {self.offset}" if self.offset else ""
        return f"Limit {self.count}{skip}"


def walk(plan: Plan) -> list[Plan]:
    """Every node in the tree, parents before children."""
    out = [plan]
    for child in plan.children():
        out.extend(walk(child))
    return out


def scans(plan: Plan) -> list[Scan]:
    """Every scan in a plan, which is where pushdown ends up."""
    return [node for node in walk(plan) if isinstance(node, Scan)]


def render(plan: Plan, indent: int = 0) -> str:
    """The plan as an indented tree, which is what the command line prints."""
    lines = [" " * indent + plan.describe()]
    for child in plan.children():
        lines.append(render(child, indent + 2))
    return "\n".join(lines)


def rebuild(plan: Plan, children: Sequence[Plan]) -> Plan:
    """A copy of a node with different inputs.

    The primitive every rewrite rule is built from. Written as one function with a branch per
    node type rather than as a method, because a rule wants to transform children generically
    and a method would put that knowledge in seven places.
    """
    if isinstance(plan, Scan):
        if children:
            raise PlanError("a scan has no children")
        return plan
    if isinstance(plan, Filter):
        return Filter(input=children[0], predicate=plan.predicate)
    if isinstance(plan, Project):
        return Project(input=children[0], names=plan.names)
    if isinstance(plan, Join):
        return Join(
            left=children[0],
            right=children[1],
            left_keys=plan.left_keys,
            right_keys=plan.right_keys,
            suffix=plan.suffix,
        )
    if isinstance(plan, Group):
        return Group(input=children[0], keys=plan.keys, aggregates=plan.aggregates)
    if isinstance(plan, Sort):
        return Sort(input=children[0], keys=plan.keys)
    if isinstance(plan, Limit):
        return Limit(input=children[0], count=plan.count, offset=plan.offset)
    raise PlanError(f"{type(plan).__name__} cannot be rebuilt")


def transform(plan: Plan, rule) -> Plan:
    """Apply a rule to every node, children first.

    Children first so a rule sees a subtree that has already been rewritten, which is what makes
    a sequence of local rewrites compose into a global one. The alternative, parents first,
    means a rule has to reason about what its children will become.
    """
    children = [transform(child, rule) for child in plan.children()]
    rebuilt = rebuild(plan, children) if children else plan
    return rule(rebuilt)


def table(name: str, batch: Batch, rows: int | None = None) -> Scan:
    """A scan over a batch, which is how every test builds a plan."""
    return Scan(
        name=name,
        table_schema=batch.schema,
        row_count=batch.rows if rows is None else rows,
    )


def a_plan_knows_its_schema() -> dict:
    """Every node computes its columns from its inputs, which is what catches a dropped one."""
    batch = Batch.of(a=[1, 2], g=["x", "y"], v=[1.0, 2.0])
    scan = table("t", batch)
    project = Project(input=scan, names=("a", "v"))
    grouped = Group(input=scan, keys=("g",), aggregates=(Aggregate("n", "count_star"),))
    return {
        "scan": list(scan.schema().names),
        "project": list(project.schema().names),
        "group": list(grouped.schema().names),
        "the_project_narrowed": len(project.schema()) < len(scan.schema()),
        "the_group_renamed": "n" in grouped.schema(),
    }


def a_join_schema_disambiguates(rows: int = 10) -> dict:
    """Two inputs with the same column names come out with the right side suffixed."""
    batch = Batch.of(k=list(range(rows)), v=list(range(rows)))
    joined = Join(
        left=table("l", batch),
        right=table("r", batch),
        left_keys=("k",),
        right_keys=("k",),
    )
    return {
        "names": list(joined.schema().names),
        "it_suffixed": "k_right" in joined.schema(),
        "the_left_kept_its_names": joined.schema().names[:2] == ("k", "v"),
    }


def a_plan_referring_to_a_missing_column_is_refused() -> bool:
    """The property that makes a plan representation worth having.

    A rule that drops a column and leaves something above it reading that column produces a plan
    that cannot be built, rather than one that fails halfway through a run.
    """
    batch = Batch.of(a=[1], b=[2])
    scan = table("t", batch)
    narrowed = Scan(name="t", table_schema=batch.schema, projected=("a",))
    del scan
    try:
        Project(input=narrowed, names=("b",))
    except UnknownColumn:
        return True
    return False


def a_join_on_a_missing_key_is_refused() -> bool:
    """And the same for a join key."""
    batch = Batch.of(k=[1], v=[2])
    try:
        Join(
            left=table("l", batch),
            right=table("r", batch),
            left_keys=("z",),
            right_keys=("k",),
        )
    except UnknownColumn:
        return True
    return False


def mismatched_join_keys_are_refused() -> bool:
    """Two keys on one side and one on the other cannot be matched."""
    batch = Batch.of(k=[1], v=[2])
    try:
        Join(
            left=table("l", batch),
            right=table("r", batch),
            left_keys=("k", "v"),
            right_keys=("k",),
        )
    except PlanError:
        return True
    return False


def an_empty_projection_is_refused() -> bool:
    """A projection of nothing produces no columns and is a mistake."""
    try:
        Project(input=table("t", Batch.of(a=[1])), names=())
    except PlanError:
        return True
    return False


def a_group_with_no_aggregates_is_refused() -> bool:
    """A group by has to compute something."""
    try:
        Group(input=table("t", Batch.of(a=[1])), keys=("a",), aggregates=())
    except PlanError:
        return True
    return False


def a_negative_limit_is_refused() -> bool:
    """A window has to be a window."""
    try:
        Limit(input=table("t", Batch.of(a=[1])), count=-1)
    except PlanError:
        return True
    return False


def a_nameless_scan_is_refused() -> bool:
    """A scan names a table."""
    try:
        Scan(name="", table_schema=Batch.of(a=[1]).schema)
    except PlanError:
        return True
    return False


def transform_rewrites_children_first(rows: int = 10) -> dict:
    """The order a rule sees the tree in, which decides whether local rewrites compose.

    A rule that replaces a filter with its input is applied to the inner filter first, so by the
    time it reaches the outer one the tree below has already changed. Applying parents first
    would make a rule reason about what its children are going to become.
    """
    batch = Batch.of(a=list(range(rows)))
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    inner = Filter(input=table("t", batch), predicate=Compare(">", column("a"), literal(1)))
    outer = Filter(input=inner, predicate=Compare("<", column("a"), literal(9)))
    seen: list[str] = []

    def record(node: Plan) -> Plan:
        seen.append(type(node).__name__)
        return node

    transform(outer, record)
    return {
        "order": seen,
        "the_scan_came_first": seen[0] == "Scan",
        "the_outer_filter_came_last": seen[-1] == "Filter",
        "every_node_was_visited": len(seen) == outer.nodes(),
    }


def a_plan_renders_as_a_tree(rows: int = 10) -> dict:
    """The printed form, which is what a reader compares two plans with."""
    batch = Batch.of(a=list(range(rows)), g=["x"] * rows)
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    plan = Limit(
        input=Sort(
            input=Group(
                input=Filter(
                    input=table("t", batch),
                    predicate=Compare(">", column("a"), literal(3)),
                ),
                keys=("g",),
                aggregates=(Aggregate("n", "count_star"),),
            ),
            keys=(SortKey("n", descending=True),),
        ),
        count=5,
    )
    text = render(plan)
    return {
        "lines": len(text.split("\n")),
        "height": plan.height(),
        "nodes": plan.nodes(),
        "it_is_one_line_per_node": len(text.split("\n")) == plan.nodes(),
        "the_predicate_is_shown": "(a > 3)" in text,
        "the_sort_direction_is_shown": "desc" in text,
    }


def a_scan_records_what_was_pushed_into_it(rows: int = 100) -> dict:
    """Predicates pushed into a scan are kept separately rather than combined.

    Because the reader tests each conjunct against the row group statistics on its own, and a
    combined expression would have to be split again to do that. storage/statistics.py is the
    consumer and it takes conjuncts.
    """
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    batch = Batch.of(a=list(range(rows)), b=list(range(rows)))
    scan = table("t", batch)
    pushed = scan.with_predicate(Compare(">", column("a"), literal(5))).with_predicate(
        Compare("<", column("b"), literal(50))
    )
    return {
        "pushed": len(pushed.pushed),
        "they_are_separate": len(pushed.pushed) == 2,
        "the_description_shows_both": pushed.describe().count(" and ") == 1,
        "the_schema_is_unchanged": pushed.schema().names == scan.schema().names,
    }


def a_scan_narrows_its_projection(rows: int = 100) -> dict:
    """A scan reading two columns of five is what projection pushdown produces."""
    batch = Batch.of(a=[1] * rows, b=[1] * rows, c=[1] * rows, d=[1] * rows, e=[1] * rows)
    scan = table("t", batch)
    narrowed = scan.with_projection(["a", "c"])
    return {
        "before": len(scan.columns_read()),
        "after": len(narrowed.columns_read()),
        "it_narrowed": len(narrowed.columns_read()) < len(scan.columns_read()),
        "the_schema_followed": list(narrowed.schema().names) == ["a", "c"],
    }


def a_join_swaps(rows: int = 10) -> dict:
    """Exchanging the inputs, which is what join ordering does.

    Not a no operation: the output column order changes, so anything above the join that reads
    by position rather than by name sees something different. Every consumer here reads by name,
    which is what makes the rewrite legal, and it is worth stating because it is the assumption
    that would break first.
    """
    left = Batch.of(k=list(range(rows)), a=list(range(rows)))
    right = Batch.of(k=list(range(rows)), b=list(range(rows)))
    joined = Join(
        left=table("l", left),
        right=table("r", right),
        left_keys=("k",),
        right_keys=("k",),
    )
    swapped = joined.swapped()
    return {
        "before": list(joined.schema().names),
        "after": list(swapped.schema().names),
        "the_order_changed": joined.schema().names != swapped.schema().names,
        "the_same_columns_are_present": {
            name.replace("_right", "") for name in joined.schema().names
        }
        == {name.replace("_right", "") for name in swapped.schema().names},
    }


def columns_used_finds_what_a_node_reads(rows: int = 10) -> dict:
    """What each node needs from below, which is what projection pushdown works from."""
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    batch = Batch.of(a=list(range(rows)), b=list(range(rows)), c=list(range(rows)))
    filtered = Filter(input=table("t", batch), predicate=Compare(">", column("a"), literal(3)))
    grouped = Group(input=filtered, keys=("b",), aggregates=(Aggregate("total", "sum", "c"),))
    return {
        "filter": sorted(filtered.columns_used()),
        "group": sorted(grouped.columns_used()),
        "the_filter_reads_one": filtered.columns_used() == {"a"},
        "the_group_reads_two": grouped.columns_used() == {"b", "c"},
        "the_union_is_everything": (filtered.columns_used() | grouped.columns_used())
        == {"a", "b", "c"},
    }


def a_plan_counts_its_own_shape(rows: int = 10) -> dict:
    """Height and node count, which is how a rewrite is checked for having done anything."""
    from cqe.exec.expr import Compare, column, literal  # noqa: PLC0415

    batch = Batch.of(a=list(range(rows)))
    plan = Limit(
        input=Filter(input=table("t", batch), predicate=Compare(">", column("a"), literal(1))),
        count=5,
    )
    return {
        "height": plan.height(),
        "nodes": plan.nodes(),
        "walk_length": len(walk(plan)),
        "they_agree": len(walk(plan)) == plan.nodes(),
        "one_scan": len(scans(plan)) == 1,
    }


def rebuilding_a_scan_with_children_is_refused() -> bool:
    """A scan is a leaf and giving it inputs is a mistake in a rule."""
    batch = Batch.of(a=[1])
    try:
        rebuild(table("t", batch), [table("u", batch)])
    except PlanError:
        return True
    return False


def summarise() -> dict:
    """The module in one mapping, for the command line and for logging."""
    shape = a_plan_counts_its_own_shape()
    printed = a_plan_renders_as_a_tree()
    order = transform_rewrites_children_first()
    return {
        "nodes": shape["nodes"],
        "height": shape["height"],
        "render_lines": printed["lines"],
        "children_first": order["the_scan_came_first"],
        "node_types": 7,
    }


def rejoin(parts: Sequence[Expr]) -> Expr:
    """Rebuild a predicate from its conjuncts, which pushdown needs after redistributing."""
    return all_of(list(parts))


def split(predicate: Expr) -> list[Expr]:
    """Split a predicate into conjuncts, which pushdown needs before redistributing."""
    return conjuncts(predicate)


def check_schema(plan: Plan) -> Schema:
    """Compute a plan's schema, raising if any node reads something nobody produces.

    Called by every rule after it rewrites, so a rule that drops a column fails at the rewrite
    rather than at the run. Cheap, since the schema is computed from the tree and not the data.
    """
    for node in walk(plan):
        try:
            node.schema()
        except (SchemaError, UnknownColumn) as problem:
            raise PlanError(
                f"{node.describe()} produces nothing usable: {problem}"
            ) from problem
    return plan.schema()


def a_rewrite_that_drops_a_column_is_caught(rows: int = 10) -> dict:
    """The check that makes rewriting safe, demonstrated on a rewrite that breaks a plan."""
    batch = Batch.of(a=list(range(rows)), b=list(range(rows)))
    good = Project(input=table("t", batch), names=("a", "b"))
    check_schema(good)
    caught = False
    try:
        Project(
            input=Scan(name="t", table_schema=batch.schema, projected=("a",)),
            names=("a", "b"),
        )
    except UnknownColumn:
        caught = True
    return {
        "the_good_plan_checks": list(check_schema(good).names) == ["a", "b"],
        "the_bad_plan_is_refused": caught,
    }


def an_impossible_plan_is_refused() -> bool:
    """A node that cannot compute a schema is reported by name rather than by traceback."""
    batch = Batch.of(a=[1])
    try:
        check_schema(Limit(input=table("t", batch), count=-1))
    except (PlanError, ConfigError):
        return True
    return False
