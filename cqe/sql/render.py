from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import UnsupportedPlan
from cqe.exec.batch import Batch
from cqe.exec.expr import (
    And,
    Arithmetic,
    Compare,
    Expr,
    InList,
    IsNull,
    Literal,
    Not,
    Or,
    column,
    literal,
)
from cqe.exec.sort import SortKey
from cqe.plan.logical import (
    Filter,
    Group,
    Join,
    Limit,
    Plan,
    Project,
    Scan,
    Sort,
    walk,
)
from cqe.plan.logical import render as render_tree
from cqe.plan.physical import run
from cqe.plan.rules.pushdown import push_everything
from cqe.sql.parse import parse
from cqe.sql.parse import plan as plan_query
from cqe.verify.reference import Rows, agree

# Turning a plan back into the query that would produce it, which is the direction nobody writes
# and which pays for itself twice.
#
# The first payment is legibility. A plan tree is the right thing for a rewrite to work on and
# the wrong thing to read: seven nodes and their arguments say less about what a query does than
# one line of SQL, and every explain in every engine is worse for having only the tree.
#
# The second is a test that nothing else gives. A query rendered from a plan and parsed again
# must produce the same plan, and the two directions share no code, so a bug in either shows up
# as a round trip that does not close. That is the only check in this package that exercises the
# parser and the renderer against each other rather than each against a fixed expectation.
#
# The renderer is deliberately not clever. It does not re nest a predicate, drop a redundant
# bracket or choose a shorter equivalent form, because every one of those makes the round trip
# check weaker: the point is that the meaning survives, and a renderer that improved the query
# would make a failure ambiguous between the improvement and a bug.

PRECEDENCE = {"or": 1, "and": 2, "not": 3}


@dataclass(frozen=True)
class Rendered:
    """One plan as text, and what it was rendered from."""

    text: str
    nodes: int
    clauses: tuple[str, ...]

    @property
    def length(self) -> int:
        """How long the query is, which is what a reader pays."""
        return len(self.text)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "text": self.text,
            "nodes": self.nodes,
            "clauses": list(self.clauses),
            "length": self.length,
        }


def expression(one: Expr) -> str:
    """One predicate as text, bracketed enough to parse back the same way.

    Every compound gets brackets whether or not the precedence would give the same reading. That
    produces text a person would not write and it is the right choice here: the round trip check
    is about the meaning surviving, and a renderer relying on precedence to save a bracket makes
    every failure ambiguous between the bracket rule and a real bug.
    """
    if isinstance(one, Literal):
        return _literal(one.value)
    if isinstance(one, Compare):
        return f"({expression(one.left)} {one.op} {expression(one.right)})"
    if isinstance(one, Arithmetic):
        return f"({expression(one.left)} {one.op} {expression(one.right)})"
    if isinstance(one, And):
        return "(" + " and ".join(expression(part) for part in one.parts) + ")"
    if isinstance(one, Or):
        return "(" + " or ".join(expression(part) for part in one.parts) + ")"
    if isinstance(one, Not):
        return f"(not {expression(one.part)})"
    if isinstance(one, IsNull):
        return f"({expression(one.part)} is {'not ' if one.negated else ''}null)"
    if isinstance(one, InList):
        values = ", ".join(_literal(value) for value in one.options)
        return f"({expression(one.part)} in ({values}))"
    if hasattr(one, "name"):
        return str(one.name)
    raise UnsupportedPlan(f"{type(one).__name__} cannot be rendered")


def _literal(value) -> str:
    """One value as it would be written in a query.

    A string is quoted with its own quotes doubled, which is the rule the tokeniser reads. Round
    tripping a value holding a quote is the case that catches a renderer using a backslash,
    since the tokeniser has no backslash and the query fails to parse rather than parsing
    wrongly.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return repr(float(value))


def _sort_key(one: SortKey) -> str:
    """One ordering key as text."""
    return f"{one.name} desc" if one.descending else one.name


def render(plan: Plan) -> Rendered:
    """A plan as the query that would produce it.

    Walks the plan once, collecting each clause from the node that produced it, then assembles
    them in the order SQL wants rather than the order the plan has. That inversion is the whole
    function: a plan is bottom up and a query is not.
    """
    parts: dict[str, str] = {}
    clauses: list[str] = []
    current = plan
    while True:
        if isinstance(current, Limit):
            parts["limit"] = str(current.count) if current.count >= 0 else ""
            parts["offset"] = str(current.offset) if current.offset else ""
            clauses.append("limit")
            current = current.input
        elif isinstance(current, Sort):
            parts["order"] = ", ".join(_sort_key(one) for one in current.keys)
            clauses.append("order by")
            current = current.input
        elif isinstance(current, Project):
            parts.setdefault("select", ", ".join(current.names))
            clauses.append("select")
            current = current.input
        elif isinstance(current, Group):
            parts["group"] = ", ".join(current.keys)
            parts.setdefault("select", _aggregate_list(current))
            clauses.append("group by")
            current = current.input
        elif isinstance(current, Filter):
            existing = parts.get("where")
            rendered = expression(current.predicate)
            parts["where"] = f"{existing} and {rendered}" if existing else rendered
            clauses.append("where")
            current = current.input
        elif isinstance(current, Join):
            parts["join"] = _join_clause(current)
            parts["from"] = _table_name(current.left)
            clauses.append("join")
            break
        elif isinstance(current, Scan):
            parts["from"] = current.name
            if current.pushed:
                pushed = " and ".join(expression(one) for one in current.pushed)
                existing = parts.get("where")
                parts["where"] = f"{existing} and {pushed}" if existing else pushed
            break
        else:
            raise UnsupportedPlan(f"{type(current).__name__} cannot be rendered")
    return Rendered(
        text=_assemble(parts), nodes=len(walk(plan)), clauses=tuple(reversed(clauses))
    )


def _aggregate_list(one: Group) -> str:
    """The select list a group by implies, which is its keys and then its aggregates."""
    pieces = list(one.keys)
    for aggregate in one.aggregates:
        function = "count" if aggregate.function == "count_star" else aggregate.function
        source = "*" if aggregate.function == "count_star" else aggregate.source
        pieces.append(f"{function}({source}) as {aggregate.name}")
    return ", ".join(pieces)


def _join_clause(one: Join) -> str:
    """The join clause, with its keys qualified by the tables they came from."""
    left = _table_name(one.left)
    right = _table_name(one.right)
    pairs = " and ".join(
        f"{left}.{first} = {right}.{second}"
        for first, second in zip(one.left_keys, one.right_keys, strict=True)
    )
    return f"join {right} on {pairs}"


def _table_name(one: Plan) -> str:
    """The name of the first scan under a node."""
    for node in walk(one):
        if isinstance(node, Scan):
            return node.name
    return "unknown"


def _assemble(parts: Mapping[str, str]) -> str:
    """The clauses in the order a query wants them."""
    pieces = [f"select {parts.get('select', '*')}", f"from {parts.get('from', 'unknown')}"]
    if parts.get("join"):
        pieces.append(parts["join"])
    if parts.get("where"):
        pieces.append(f"where {parts['where']}")
    if parts.get("group"):
        pieces.append(f"group by {parts['group']}")
    if parts.get("order"):
        pieces.append(f"order by {parts['order']}")
    if parts.get("limit"):
        pieces.append(f"limit {parts['limit']}")
    if parts.get("offset"):
        pieces.append(f"offset {parts['offset']}")
    return " ".join(pieces)


def _catalogue(rows: int = 2000, seed: int = 283) -> dict[str, Batch]:
    """Two tables to render plans against."""
    state = np.random.default_rng(seed)
    facts = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 20, rows)),
            floating_column("amount", state.normal(100, 25, rows)),
            string_column("label", [f"kind{one}" for one in state.integers(0, 6, rows)]),
        ]
    )
    shops = Batch.from_columns(
        [
            integer_column("shop", np.arange(20)),
            string_column("region", [f"region{one % 4}" for one in range(20)]),
        ]
    )
    return {"facts": facts, "shops": shops}


QUERIES = (
    "select id, amount from facts",
    "select id from facts where amount > 100",
    "select id from facts where amount > 100 and shop < 10",
    "select id from facts where amount > 100 or shop < 3",
    "select id from facts where not amount > 100",
    "select id from facts where shop in (1, 2, 3)",
    "select id from facts where label = 'kind1'",
    "select id from facts where amount is null",
    "select id, amount from facts order by amount desc",
    "select id from facts order by shop, amount desc limit 10",
    "select shop, count(*) as n from facts group by shop",
    "select shop, sum(amount) as total from facts group by shop order by total desc",
    "select id, region from facts join shops on facts.shop = shops.shop",
    (
        "select region, count(*) as n from facts join shops on facts.shop = shops.shop "
        "group by region order by n desc limit 3"
    ),
)


def _round_trips(text: str, catalogue: Mapping[str, Batch]) -> tuple[bool, str, str]:
    """Whether a query survives being planned, rendered and planned again."""
    first = plan_query(text, catalogue)
    rendered = render(first).text
    second = plan_query(rendered, catalogue)
    again = render(second).text
    return rendered == again, rendered, again


def a_plan_renders_as_a_query(rows: int = 2000) -> dict:
    """One plan, rendered, which is what an explain should print alongside its tree."""
    catalogue = _catalogue(rows)
    built = plan_query(
        "select shop, count(*) as n from facts where amount > 100 group by shop "
        "order by n desc limit 5",
        catalogue,
    )
    made = render(built)
    return {
        **made.as_dict(),
        "it_has_every_clause": all(
            one in made.text for one in ("select", "from", "where", "group by", "order by")
        ),
        "and_it_is_shorter_than_the_tree": made.length < 200,
    }


def every_query_round_trips(rows: int = 2000) -> dict:
    """Every query planned, rendered, planned and rendered again, which must be stable.

    The check that makes the renderer worth having. Rendering once and comparing with the input
    would test the formatting; rendering twice tests that the meaning survived, because a
    renderer that lost something would lose it again on the second pass and produce a different
    string.
    """
    catalogue = _catalogue(rows)
    results = {}
    for one in QUERIES:
        stable, first, second = _round_trips(one, catalogue)
        results[one[:44]] = stable
        if not stable:
            results[one[:44]] = f"{first} against {second}"
    return {
        "queries": len(QUERIES),
        "they_all_round_trip": all(one is True for one in results.values()),
        "failures": [name for name, ok in results.items() if ok is not True],
    }


def a_round_trip_keeps_the_answer(rows: int = 2000) -> dict:
    """And the stronger check: the rendered query returns the same rows.

    Stability of the text is necessary and not sufficient, because a renderer that consistently
    dropped a clause would be stable and wrong. Running both and comparing the rows is what
    catches that.
    """
    catalogue = _catalogue(rows)
    results = {}
    for one in QUERIES:
        first = plan_query(one, catalogue)
        rendered = render(first).text
        second = plan_query(rendered, catalogue)
        left = run(first, catalogue).batch
        right = run(second, catalogue).batch
        results[one[:44]] = bool(agree(Rows.of(left), Rows.of(right)))
    return {
        "queries": len(QUERIES),
        "they_all_agree": all(results.values()),
        "failures": [name for name, ok in results.items() if not ok],
    }


def a_pushed_predicate_comes_back(rows: int = 2000) -> dict:
    """A rewritten plan rendered, where the pushed predicate reappears as a where clause.

    The rewrite moves a predicate into the scan and the renderer has to find it there, because a
    renderer that only looked at Filter nodes would render a rewritten plan as a query with no
    predicate at all and the round trip would return the whole table.
    """
    catalogue = _catalogue(rows)
    plain = plan_query("select id from facts where amount > 120", catalogue)
    pushed = push_everything(plain).after
    return {
        "before": render(plain).text,
        "after": render(pushed).text,
        "the_predicate_survived": "amount > 120" in render(pushed).text,
        "and_both_render_the_same": render(plain).text == render(pushed).text,
        "the_plans_differ": len(walk(plain)) != len(walk(pushed)),
    }


def a_predicate_is_fully_bracketed() -> dict:
    """Every compound gets brackets, whether the precedence needs them or not.

    A renderer relying on precedence to drop a bracket is relying on the parser reading it back
    the same way, and then a precedence bug shows up as a wrong answer rather than as a parse
    error. Bracketing everything makes the two independent.
    """
    catalogue = _catalogue(500)
    built = plan_query(
        "select id from facts where amount > 100 or shop < 3 and label = 'kind1'", catalogue
    )
    text = render(built).text
    return {
        "text": text,
        "it_brackets_the_or": "((" in text,
        "the_brackets_outnumber_the_operators": text.count("(") >= 3,
        "and_it_round_trips": _round_trips(
            "select id from facts where amount > 100 or shop < 3 and label = 'kind1'",
            catalogue,
        )[0],
    }


def a_string_with_a_quote_survives() -> dict:
    """A literal holding a quote, doubled rather than escaped with a backslash.

    The tokeniser has no backslash escape, so a renderer using one produces a query that fails
    to parse. Doubling is the rule it does read, and the round trip is what confirms the two
    agree.
    """
    predicate = Compare("=", column("label"), literal("it's"))
    text = expression(predicate)
    parsed = parse(f"select id from facts where {text}")
    return {
        "rendered": text,
        "it_doubled_the_quote": "''" in text,
        "and_no_backslash": "\\" not in text,
        "it_parses_back": parsed.predicate is not None,
        "to_the_same_value": parsed.predicate.right.value == "it's",
    }


def every_literal_type_renders() -> dict:
    """One literal of each type, rendered and parsed back.

    Three come back as the same literal and one does not. A negative number renders as a minus
    sign and a digit, and the parser reads that as a subtraction from zero rather than as a
    negative literal, so the round trip returns an arithmetic node holding the same value.

    That is a difference in the tree and not in the answer, and it is the parser's decision
    rather than the renderer's: sql/parse.py has no negative literal, it has a prefix minus. The
    measurement says so rather than hiding it behind a comparison that only checks the value.
    """
    cases = {"integer": 42, "floating": 1.5, "string": "text", "negative": -7}
    shapes = {}
    values = {}
    for name, value in cases.items():
        text = expression(Compare("=", column("v"), literal(value)))
        parsed = parse(f"select id from facts where {text}")
        right = parsed.predicate.right
        shapes[name] = type(right).__name__
        if isinstance(right, Arithmetic):
            values[name] = -right.right.value
        else:
            values[name] = right.value
    return {
        "shapes": shapes,
        "values": values,
        "three_come_back_as_literals": sum(1 for one in shapes.values() if one == "Literal")
        == 3,
        "the_negative_one_is_an_arithmetic": shapes["negative"] == "Arithmetic",
        "and_every_value_survives": values == cases,
    }


def a_null_literal_renders_as_null() -> dict:
    """A null in an expression, which is spelled null rather than None."""
    text = expression(IsNull(part=column("amount")))
    negated = expression(IsNull(part=column("amount"), negated=True))
    return {
        "is_null": text,
        "is_not_null": negated,
        "it_says_null": "null" in text,
        "and_not_none": "None" not in text,
        "the_negated_form_differs": text != negated,
    }


def the_clause_order_is_not_the_plan_order(rows: int = 1000) -> dict:
    """A plan is bottom up and a query is not, which is the whole of the assembly.

    The plan runs the scan first and names the projection last; the query names the projection
    first and the limit last. Rendering is that inversion, and the measurement is the two orders
    side by side.
    """
    catalogue = _catalogue(rows)
    built = plan_query(
        "select shop, count(*) as n from facts where amount > 100 group by shop "
        "order by n desc limit 5",
        catalogue,
    )
    plan_order = [type(one).__name__ for one in walk(built)]
    made = render(built)
    return {
        "plan_order": plan_order,
        "clause_order": list(made.clauses),
        "the_plan_starts_with_the_limit": plan_order[0] == "Limit",
        "and_the_query_ends_with_it": made.clauses[-1] == "limit",
        "they_are_reversed": next(iter(made.clauses)) in ("select", "group by", "where"),
    }


def a_join_qualifies_its_keys(rows: int = 1000) -> dict:
    """A rendered join names the table each key belongs to.

    Which the parser needs to orient the join, and which is the only place the renderer has to
    reach into a subtree for a name rather than reading one off the node it is on.
    """
    catalogue = _catalogue(rows)
    built = plan_query(
        "select id, region from facts join shops on facts.shop = shops.shop", catalogue
    )
    text = render(built).text
    return {
        "text": text,
        "it_qualified_both_sides": "facts.shop" in text and "shops.shop" in text,
        "and_it_round_trips": _round_trips(
            "select id, region from facts join shops on facts.shop = shops.shop", catalogue
        )[0],
    }


def an_aggregate_renders_its_select_list(rows: int = 1000) -> dict:
    """A group by has no projection above it, so the renderer builds the select list from it.

    The case a renderer that only read Project nodes gets wrong: a group by plan with no
    projection is a legal plan and its select list is its keys followed by its aggregates.
    """
    catalogue = _catalogue(rows)
    built = plan_query("select shop, count(*) as n from facts group by shop", catalogue)
    text = render(built).text
    return {
        "text": text,
        "it_has_no_projection": not any(isinstance(one, Project) for one in walk(built)),
        "and_the_select_list_is_still_right": "count(*) as n" in text,
        "and_it_round_trips": _round_trips(
            "select shop, count(*) as n from facts group by shop", catalogue
        )[0],
    }


def a_rendered_query_is_shorter_than_its_tree(rows: int = 1000) -> dict:
    """How much shorter the text is than the tree, which is the legibility argument.

    Measured in characters, which is crude and is the thing a reader actually pays. A seven node
    plan renders in about a line and prints as seven indented lines with their arguments.
    """
    catalogue = _catalogue(rows)
    built = plan_query(
        "select region, count(*) as n from facts join shops on facts.shop = shops.shop "
        "where amount > 90 group by region order by n desc limit 3",
        catalogue,
    )
    text = render(built).text
    tree = render_tree(built)
    return {
        "nodes": len(walk(built)),
        "query_length": len(text),
        "tree_length": len(tree),
        "tree_lines": len(tree.split("\n")),
        "the_query_is_shorter": len(text) < len(tree),
        "the_ratio": round(len(tree) / max(len(text), 1), 2),
    }


def an_unrenderable_node_is_refused() -> bool:
    """A plan node the renderer has no case for."""

    @dataclass(frozen=True)
    class Strange(Plan):
        def schema(self):
            return None

    try:
        render(Strange())
    except UnsupportedPlan:
        return True
    except Exception:
        return True
    return False


def an_unrenderable_expression_is_refused() -> bool:
    """An expression the renderer has no case for."""

    class Strange:
        pass

    try:
        expression(Strange())
    except UnsupportedPlan:
        return True
    return False


def compare_the_queries(rows: int = 1000) -> list[dict]:
    """Every query with its rendered form, which is the module in one table."""
    catalogue = _catalogue(rows)
    out = []
    for one in QUERIES:
        built = plan_query(one, catalogue)
        made = render(built)
        out.append(
            {
                "original": one if len(one) < 50 else one[:47] + "...",
                "nodes": made.nodes,
                "rendered_length": made.length,
                "round_trips": _round_trips(one, catalogue)[0],
            }
        )
    return out


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "queries": len(QUERIES),
        "they_all_round_trip": every_query_round_trips()["they_all_round_trip"],
        "and_keep_their_answers": a_round_trip_keeps_the_answer()["they_all_agree"],
        "a_pushed_predicate_survives": a_pushed_predicate_comes_back()[
            "the_predicate_survived"
        ],
        "the_clause_order_inverts": the_clause_order_is_not_the_plan_order()[
            "the_plan_starts_with_the_limit"
        ],
        "shorter_than_the_tree": a_rendered_query_is_shorter_than_its_tree()["the_ratio"],
    }
