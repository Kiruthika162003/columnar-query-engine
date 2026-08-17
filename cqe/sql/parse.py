from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from cqe.columns.array import floating_column, integer_column, string_column
from cqe.errors import ParseError, SchemaError, UnknownColumn
from cqe.exec.aggregate import Aggregate
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
    describe,
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
    Sort,
    table,
    walk,
)
from cqe.plan.physical import execute
from cqe.sql.tokenise import END, NUMBER, SYMBOL, TEXT, WORD, Stream, stream
from cqe.verify.reference import Rows, agree, order_by, select, where

# A parser for the subset of SQL this engine can answer, which is one SELECT with joins,
# a predicate, grouping, ordering and a limit.
#
# The parse is in two halves and the split is the only interesting decision in here.
#
# The first half reads text into a Select, which is a record of what the query said. It knows
# nothing about the tables: a column that does not exist parses cleanly, an aggregate over a
# string parses cleanly, a group by naming a column that is not selected parses cleanly. The
# grammar is the only thing it enforces.
#
# The second half resolves a Select against a catalogue into a Plan. Every error that needs to
# know what the data looks like happens here, and every one of them can name the column and the
# table it looked in.
#
# Splitting them costs one extra type. It buys a parser that can be tested without any data at
# all, and a resolver whose errors are about the schema rather than about the syntax. The first
# version of this did both at once and its error for a misspelled column was a parse error at
# the character after the name, which is where the parser noticed rather than where the problem
# was.
#
# Precedence is the standard one and is expressed as a chain of functions rather than a table:
# or, then and, then not, then comparison, then addition, then multiplication, then a primary.
# A table would be shorter. A chain puts each level's rule next to its own name, which is what a
# reader checking whether not binds tighter than and actually wants.

# The symbols a query can be written with, mapped to the operators exec/expr.py evaluates. Two
# of them differ: SQL spells inequality both as an exclamation mark and as a pair of angle
# brackets, and the engine has one name for it.
COMPARISONS = {"=": "=", "!=": "!=", "<>": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}
AGGREGATE_NAMES = ("count", "sum", "min", "max", "avg")

# What each of them is called inside the engine. Two names differ. SQL says avg where
# exec/aggregate.py says mean, and counting rows is a different aggregate from counting the
# values of a column, because the second skips nulls and the first cannot: count(*) is the size
# of the group and count(x) is how many rows of it have an x.
FUNCTIONS = {"count": "count", "sum": "sum", "min": "min", "max": "max", "avg": "mean"}


@dataclass(frozen=True)
class Item:
    """One entry in a select list."""

    source: str
    alias: str
    function: str = ""

    @property
    def is_aggregate(self) -> bool:
        """Whether this item needs a group by to compute."""
        return bool(self.function)

    def describe(self) -> str:
        """One line, as it would be written."""
        body = f"{self.function}({self.source})" if self.function else self.source
        return body if body == self.alias else f"{body} as {self.alias}"


@dataclass(frozen=True)
class From:
    """One table in a from clause, and the join that brought it in."""

    name: str
    left_key: str = ""
    right_key: str = ""

    @property
    def is_joined(self) -> bool:
        """Whether this table arrived through a join rather than first."""
        return bool(self.left_key)


@dataclass(frozen=True)
class Select:
    """A parsed query, before anything has been resolved against a schema."""

    items: tuple[Item, ...]
    tables: tuple[From, ...]
    predicate: Expr | None = None
    group_keys: tuple[str, ...] = ()
    having: Expr | None = None
    order_keys: tuple[SortKey, ...] = ()
    limit: int = -1
    offset: int = 0
    distinct: bool = False
    star: bool = False

    @property
    def aggregates(self) -> tuple[Item, ...]:
        """The items that need a group by."""
        return tuple(one for one in self.items if one.is_aggregate)

    @property
    def plain(self) -> tuple[Item, ...]:
        """The items that are columns rather than aggregates."""
        return tuple(one for one in self.items if not one.is_aggregate)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "items": [one.describe() for one in self.items],
            "tables": [one.name for one in self.tables],
            "has_predicate": self.predicate is not None,
            "group_keys": list(self.group_keys),
            "order_keys": [one.name for one in self.order_keys],
            "limit": self.limit,
            "offset": self.offset,
            "distinct": self.distinct,
        }


@dataclass
class Parser:
    """The recursive descent parser, holding only its cursor."""

    tokens: Stream
    qualified: dict[str, str] = field(default_factory=dict)

    def parse(self) -> Select:
        """One complete select statement, and nothing after it."""
        self.tokens.expect(WORD, "select")
        distinct = self.tokens.accept(WORD, "distinct") is not None
        items, star = self._select_list()
        tables = self._from_clause()
        predicate = self._where_clause()
        group_keys = self._group_clause()
        having = self._having_clause()
        order_keys = self._order_clause()
        limit, offset = self._limit_clause()
        if not self.tokens.done:
            raise ParseError(
                f"the query ended and then continued with {self.tokens.current.value!r}",
                self.tokens.current.position,
                self.tokens.text,
            )
        return Select(
            items=items,
            tables=tables,
            predicate=predicate,
            group_keys=group_keys,
            having=having,
            order_keys=order_keys,
            limit=limit,
            offset=offset,
            distinct=distinct,
            star=star,
        )

    def _select_list(self) -> tuple[tuple[Item, ...], bool]:
        """The items between select and from, or a star."""
        if self.tokens.accept(SYMBOL, "*") is not None:
            return (), True
        out: list[Item] = []
        while True:
            out.append(self._item())
            if self.tokens.accept(SYMBOL, ",") is None:
                break
        return tuple(out), False

    def _item(self) -> Item:
        """One select item: a column, or an aggregate, either optionally aliased."""
        if self.tokens.looking_at(*AGGREGATE_NAMES) and self.tokens.peek(1).matches(
            SYMBOL, "("
        ):
            function = self.tokens.take().value.lower()
            self.tokens.expect(SYMBOL, "(")
            if self.tokens.accept(SYMBOL, "*") is not None:
                source = "*"
                if function != "count":
                    raise ParseError(
                        f"{function} of a star is not a thing",
                        self.tokens.current.position,
                        self.tokens.text,
                    )
            else:
                source = self._name()
            self.tokens.expect(SYMBOL, ")")
            default = f"{function}_{source}" if source != "*" else "count"
            return Item(source=source, alias=self._alias(default), function=function)
        source = self._name()
        return Item(source=source, alias=self._alias(source))

    def _alias(self, default: str) -> str:
        """An explicit alias, or the name the item would have anyway.

        The as keyword is required. Making it optional is legal SQL and turns every missing
        comma in a select list into a silently renamed column, which is the worst class of bug
        this parser could have: the query runs and answers a different question.
        """
        if self.tokens.accept(WORD, "as") is None:
            return default
        token = self.tokens.expect(WORD)
        if token.is_keyword:
            raise ParseError(
                f"{token.value!r} is a keyword and cannot be an alias",
                token.position,
                self.tokens.text,
            )
        return token.value

    def _name(self) -> str:
        """A column name, possibly qualified by a table.

        A qualified name is recorded so the resolver can tell which side of a join it meant, and
        returned unqualified because that is what the column is called once it is in a batch.
        """
        token = self.tokens.expect(WORD)
        if token.is_keyword:
            raise ParseError(
                f"{token.value!r} is a keyword, not a column",
                token.position,
                self.tokens.text,
            )
        if self.tokens.accept(SYMBOL, ".") is None:
            return token.value
        second = self.tokens.expect(WORD)
        self.qualified[second.value] = token.value
        return second.value

    def _from_clause(self) -> tuple[From, ...]:
        """The table, and any joined tables with their keys."""
        self.tokens.expect(WORD, "from")
        first = self.tokens.expect(WORD)
        out = [From(name=first.value)]
        while self.tokens.accept(WORD, "join") is not None:
            joined = self.tokens.expect(WORD)
            self.tokens.expect(WORD, "on")
            left = self._name()
            self.tokens.expect(SYMBOL, "=")
            right = self._name()
            left, right = self._orient(left, right, joined.value)
            out.append(From(name=joined.value, left_key=left, right_key=right))
        return tuple(out)

    def _orient(self, left: str, right: str, joined: str) -> tuple[str, str]:
        """Put the joined table's key on the right, whichever side it was written on.

        On a.id = b.id and on b.id = a.id are the same join and a user writing the second should
        not get a different plan. The qualifier says which table each name came from, so this is
        decidable whenever the names are qualified, and when they are not the order as written
        is the only information available and is kept.
        """
        if self.qualified.get(left) == joined and self.qualified.get(right) != joined:
            return right, left
        return left, right

    def _where_clause(self) -> Expr | None:
        """The predicate, if there is one."""
        if self.tokens.accept(WORD, "where") is None:
            return None
        return self._or()

    def _group_clause(self) -> tuple[str, ...]:
        """The group by keys, if there are any."""
        if not self.tokens.looking_at("group"):
            return ()
        self.tokens.expect(WORD, "group")
        self.tokens.expect(WORD, "by")
        out = [self._name()]
        while self.tokens.accept(SYMBOL, ",") is not None:
            out.append(self._name())
        return tuple(out)

    def _having_clause(self) -> Expr | None:
        """A predicate over the aggregated output, if there is one."""
        if self.tokens.accept(WORD, "having") is None:
            return None
        return self._or()

    def _order_clause(self) -> tuple[SortKey, ...]:
        """The sort keys, with their directions."""
        if not self.tokens.looking_at("order"):
            return ()
        self.tokens.expect(WORD, "order")
        self.tokens.expect(WORD, "by")
        out: list[SortKey] = []
        while True:
            name = self._name()
            descending = False
            if self.tokens.accept(WORD, "desc") is not None:
                descending = True
            else:
                self.tokens.accept(WORD, "asc")
            out.append(SortKey(name=name, descending=descending))
            if self.tokens.accept(SYMBOL, ",") is None:
                break
        return tuple(out)

    def _limit_clause(self) -> tuple[int, int]:
        """The row limit and the offset, both optional."""
        limit = -1
        offset = 0
        if self.tokens.accept(WORD, "limit") is not None:
            limit = self._integer("a limit")
        if self.tokens.accept(WORD, "offset") is not None:
            offset = self._integer("an offset")
        return limit, offset

    def _integer(self, what: str) -> int:
        """A whole number, refused if it is a decimal or negative."""
        token = self.tokens.expect(NUMBER)
        if "." in token.value:
            raise ParseError(f"{what} must be whole", token.position, self.tokens.text)
        return int(token.value)

    def _or(self) -> Expr:
        """The loosest binding level."""
        parts = [self._and()]
        while self.tokens.accept(WORD, "or") is not None:
            parts.append(self._and())
        return parts[0] if len(parts) == 1 else Or(parts=tuple(parts))

    def _and(self) -> Expr:
        """Tighter than or, looser than not."""
        parts = [self._not()]
        while self.tokens.accept(WORD, "and") is not None:
            parts.append(self._not())
        return parts[0] if len(parts) == 1 else And(parts=tuple(parts))

    def _not(self) -> Expr:
        """Prefix negation, which binds tighter than and."""
        if self.tokens.accept(WORD, "not") is not None:
            return Not(part=self._not())
        return self._comparison()

    def _comparison(self) -> Expr:
        """A comparison, an is null, or an in list.

        Not chained: a < b < c is refused rather than read as it would be in Python or as it
        would be in C. Both readings are defensible and a user cannot tell which they got, so
        neither is offered.
        """
        left = self._sum()
        if self.tokens.accept(WORD, "is") is not None:
            negated = self.tokens.accept(WORD, "not") is not None
            self.tokens.expect(WORD, "null")
            return IsNull(part=left, negated=negated)
        if self.tokens.accept(WORD, "in") is not None:
            return InList(part=left, options=self._list())
        token = self.tokens.current
        if token.kind == SYMBOL and token.value in COMPARISONS:
            self.tokens.take()
            right = self._sum()
            if self.tokens.current.kind == SYMBOL and self.tokens.current.value in COMPARISONS:
                raise ParseError(
                    "comparisons do not chain, use and",
                    self.tokens.current.position,
                    self.tokens.text,
                )
            return Compare(op=COMPARISONS[token.value], left=left, right=right)
        return left

    def _list(self) -> tuple:
        """The parenthesised values of an in clause."""
        self.tokens.expect(SYMBOL, "(")
        out = [self._constant()]
        while self.tokens.accept(SYMBOL, ",") is not None:
            out.append(self._constant())
        self.tokens.expect(SYMBOL, ")")
        return tuple(out)

    def _constant(self) -> object:
        """One literal value, as a Python object rather than an expression."""
        token = self.tokens.current
        if token.kind == NUMBER:
            self.tokens.take()
            return float(token.value) if "." in token.value else int(token.value)
        if token.kind == TEXT:
            self.tokens.take()
            return token.value
        raise ParseError(
            f"expected a value and found {token.value!r}", token.position, self.tokens.text
        )

    def _sum(self) -> Expr:
        """Addition and subtraction."""
        left = self._product()
        while True:
            token = self.tokens.current
            if not (token.kind == SYMBOL and token.value in ("+", "-")):
                return left
            self.tokens.take()
            left = Arithmetic(op=token.value, left=left, right=self._product())

    def _product(self) -> Expr:
        """Multiplication, which binds tighter than addition."""
        left = self._primary()
        while self.tokens.current.matches(SYMBOL, "*"):
            self.tokens.take()
            left = Arithmetic(op="*", left=left, right=self._primary())
        return left

    def _primary(self) -> Expr:
        """A literal, a column, or a parenthesised expression."""
        token = self.tokens.current
        if token.kind in (NUMBER, TEXT):
            return literal(self._constant())
        if token.matches(SYMBOL, "("):
            self.tokens.take()
            inner = self._or()
            self.tokens.expect(SYMBOL, ")")
            return inner
        if token.matches(SYMBOL, "-"):
            self.tokens.take()
            return Arithmetic(op="-", left=literal(0), right=self._primary())
        if token.matches(WORD, "null"):
            self.tokens.take()
            return Literal(value=None, logical="integer")
        if token.kind == WORD:
            return column(self._name())
        raise ParseError(
            f"expected a value or a column and found {token.value!r}",
            token.position,
            self.tokens.text,
        )


def parse(text: str) -> Select:
    """Text into a Select, without looking at any data."""
    if not text.strip():
        raise ParseError("there is no query here", 0, text)
    cursor = stream(text)
    if cursor.current.kind == END:
        raise ParseError("there is no query here", 0, text)
    return Parser(tokens=cursor).parse()


def build(query: Select, catalogue: Mapping[str, Batch]) -> Plan:
    """A parsed query against a set of named tables, as a logical plan.

    The order is the order a reader expects and is also the only correct one: scan, join,
    filter, group, having, sort, limit, project. Projection last rather than first because a
    query can order by a column it does not select, and putting the project above the sort would
    drop the column before the sort could use it. plan/rules/pushdown.py then moves it back down
    as far as it can go, which is exactly the point of having a rewrite rather than a clever
    builder.
    """
    plan = _scan(query.tables[0].name, catalogue)
    for one in query.tables[1:]:
        plan = _join(plan, one, catalogue)
    if query.predicate is not None:
        _check_columns(query.predicate, plan, "the where clause")
        plan = Filter(input=plan, predicate=query.predicate)
    if query.group_keys or query.aggregates:
        plan = _group(query, plan)
    if query.having is not None:
        _check_columns(query.having, plan, "the having clause")
        plan = Filter(input=plan, predicate=query.having)
    if query.order_keys:
        _check_order(query, plan)
        plan = Sort(input=plan, keys=query.order_keys)
    if query.limit >= 0 or query.offset:
        plan = Limit(input=plan, count=query.limit, offset=query.offset)
    return _project(query, plan)


def _scan(name: str, catalogue: Mapping[str, Batch]) -> Plan:
    """One named table from the catalogue, or a refusal listing what there is."""
    if name not in catalogue:
        known = sorted(catalogue)
        raise SchemaError(f"there is no table called {name}, only {known}")
    return table(name, catalogue[name])


def _join(left: Plan, right: From, catalogue: Mapping[str, Batch]) -> Plan:
    """One join, with the keys the query gave."""
    other = _scan(right.name, catalogue)
    if right.left_key not in left.schema():
        raise UnknownColumn(f"{right.left_key} is not in the left side of the join")
    if right.right_key not in other.schema():
        raise UnknownColumn(f"{right.right_key} is not in {right.name}")
    return Join(
        left=left,
        right=other,
        left_keys=(right.left_key,),
        right_keys=(right.right_key,),
    )


def _group(query: Select, plan: Plan) -> Plan:
    """The group by, including the case where there are aggregates and no keys.

    Aggregating with no group by is one group over everything, which is what the standard says
    and is also the only reading that makes select count(*) from t mean anything.
    """
    if not query.aggregates:
        raise ParseError("a group by with nothing to aggregate is a distinct", -1, "")
    schema = plan.schema()
    for one in query.group_keys:
        if one not in schema:
            raise UnknownColumn(f"{one} is grouped by and is not a column")
    for one in query.aggregates:
        if one.source != "*" and one.source not in schema:
            raise UnknownColumn(f"{one.source} is aggregated and is not a column")
    for one in query.plain:
        if one.source not in query.group_keys:
            raise SchemaError(f"{one.source} is selected and is neither grouped nor aggregated")
    aggregates = tuple(
        Aggregate(
            name=one.alias,
            function="count_star" if one.source == "*" else FUNCTIONS[one.function],
            source="" if one.source == "*" else one.source,
        )
        for one in query.aggregates
    )
    return Group(input=plan, keys=query.group_keys, aggregates=aggregates)


def _project(query: Select, plan: Plan) -> Plan:
    """The final column list, or nothing at all for a star.

    A star is not a projection over every column, it is the absence of one. The difference shows
    up in the plan: a projection naming every column is a node the rewrite has to reason about
    and eventually delete, and no node at all is nothing to reason about.
    """
    if query.star:
        return plan
    names = tuple(one.alias for one in query.items)
    schema = plan.schema()
    missing = [one for one in names if one not in schema]
    if missing:
        raise UnknownColumn(f"{missing} are selected and are not columns of the result")
    if names == tuple(schema.names):
        return plan
    return Project(input=plan, names=names)


def _check_columns(predicate: Expr, plan: Plan, where: str) -> None:
    """Every column a predicate reads exists, named against the clause it was in."""
    schema = plan.schema()
    missing = sorted(one for one in predicate.columns_used() if one not in schema)
    if missing:
        raise UnknownColumn(f"{missing} in {where} are not columns of {sorted(schema.names)}")


def _check_order(query: Select, plan: Plan) -> None:
    """Every sort key exists, checked against the plan below the projection.

    Which is why the projection is built last. Ordering by a column that is not selected is
    legal and common, and a builder projecting first would reject it.
    """
    schema = plan.schema()
    aliases = {one.alias for one in query.items}
    missing = [
        one.name
        for one in query.order_keys
        if one.name not in schema and one.name not in aliases
    ]
    if missing:
        raise UnknownColumn(f"{missing} are ordered by and are not columns")


def plan(text: str, catalogue: Mapping[str, Batch]) -> Plan:
    """Text straight to a logical plan."""
    return build(parse(text), catalogue)


def render(text: str) -> str:
    """A parsed query as one readable line, for a command line and for logging."""
    query = parse(text)
    pieces = ["select " + (", ".join(one.describe() for one in query.items) or "*")]
    pieces.append("from " + query.tables[0].name)
    for one in query.tables[1:]:
        pieces.append(f"join {one.name} on {one.left_key} = {one.right_key}")
    if query.predicate is not None:
        pieces.append("where " + describe(query.predicate))
    if query.group_keys:
        pieces.append("group by " + ", ".join(query.group_keys))
    if query.order_keys:
        pieces.append(
            "order by "
            + ", ".join(
                f"{one.name} desc" if one.descending else one.name for one in query.order_keys
            )
        )
    if query.limit >= 0:
        pieces.append(f"limit {query.limit}")
    return " ".join(pieces)


def _catalogue(rows: int = 200) -> dict[str, Batch]:
    """Two tables to parse against, small and deterministic."""
    state = np.random.default_rng(11)
    facts = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 10, rows)),
            floating_column("amount", state.normal(100, 20, rows)),
            string_column("label", [f"item{one % 7}" for one in range(rows)]),
        ]
    )
    shops = Batch.from_columns(
        [
            integer_column("shop", np.arange(10)),
            string_column("region", [f"region{one % 3}" for one in range(10)]),
        ]
    )
    return {"facts": facts, "shops": shops}


def the_simplest_query_parses() -> dict:
    """Select two columns from a table, which is the shape everything else extends."""
    query = parse("select id, amount from facts")
    return {
        "items": [one.alias for one in query.items],
        "table": query.tables[0].name,
        "no_predicate": query.predicate is None,
        "no_limit": query.limit == -1,
        "it_is_not_a_star": not query.star,
    }


def a_star_is_not_a_projection() -> dict:
    """Select star builds a plan with no project node in it at all.

    A projection naming every column would be equivalent and would leave the rewrite a node to
    recognise and delete. Not building it is one branch here against a rule there.
    """
    catalogue = _catalogue()
    star = plan("select * from facts", catalogue)
    named = plan("select id, amount from facts", catalogue)
    return {
        "star_nodes": len(walk(star)),
        "named_nodes": len(walk(named)),
        "the_star_has_no_project": not any(isinstance(one, Project) for one in walk(star)),
        "the_named_one_does": any(isinstance(one, Project) for one in walk(named)),
        "they_have_the_same_columns_underneath": len(star.schema()) == 4,
    }


def selecting_every_column_by_name_is_also_not_a_projection() -> dict:
    """Naming all four columns in order is a star written out, and builds the same plan.

    The check is on the names in order, not as a set, because a reordering is a real projection.
    """
    catalogue = _catalogue()
    everything = plan("select id, shop, amount, label from facts", catalogue)
    reordered = plan("select amount, id, shop, label from facts", catalogue)
    return {
        "in_order_nodes": len(walk(everything)),
        "reordered_nodes": len(walk(reordered)),
        "the_ordered_one_is_bare": not any(
            isinstance(one, Project) for one in walk(everything)
        ),
        "the_reordered_one_projects": any(isinstance(one, Project) for one in walk(reordered)),
    }


def precedence_is_the_standard_one() -> dict:
    """Not binds tighter than and, which binds tighter than or.

    Read off the shape of the tree rather than off the rendered text, because the rendering
    parenthesises and would make any grouping look right.
    """
    both = parse("select id from facts where id = 1 or id = 2 and id = 3").predicate
    negated = parse("select id from facts where not id = 1 and id = 2").predicate
    return {
        "top_of_or_and": type(both).__name__,
        "and_binds_tighter": isinstance(both, Or)
        and isinstance(both.parts[1], And)
        and len(both.parts) == 2,
        "top_of_not_and": type(negated).__name__,
        "not_binds_tighter": isinstance(negated, And) and isinstance(negated.parts[0], Not),
    }


def parentheses_override_precedence() -> dict:
    """The same predicate with brackets is a different tree, and a different answer."""
    without = parse("select id from facts where id = 1 or id = 2 and id = 3").predicate
    within = parse("select id from facts where (id = 1 or id = 2) and id = 3").predicate
    return {
        "without": type(without).__name__,
        "within": type(within).__name__,
        "they_differ": type(without) is not type(within),
        "the_bracketed_one_is_an_and": isinstance(within, And),
        "and_holds_an_or": isinstance(within, And) and isinstance(within.parts[0], Or),
    }


def arithmetic_binds_tighter_than_comparison() -> dict:
    """A plus B less than C is a comparison of a sum, not a sum of a comparison."""
    parsed = parse("select id from facts where id + 1 < 10").predicate
    return {
        "top": type(parsed).__name__,
        "it_is_a_comparison": isinstance(parsed, Compare),
        "its_left_is_arithmetic": isinstance(parsed, Compare)
        and isinstance(parsed.left, Arithmetic),
    }


def multiplication_binds_tighter_than_addition() -> dict:
    """One plus two times three is one plus a product, evaluated to seven."""
    parsed = parse("select id from facts where id < 1 + 2 * 3").predicate
    right = parsed.right if isinstance(parsed, Compare) else None
    single = _catalogue()["facts"].slice(0, 1)
    evaluated = right.evaluate(single) if right is not None else None
    return {
        "shape": type(right).__name__,
        "it_is_a_sum": isinstance(right, Arithmetic) and right.op == "+",
        "its_right_is_a_product": isinstance(right, Arithmetic)
        and isinstance(right.right, Arithmetic),
        "value": None if evaluated is None else int(evaluated.values[0]),
        "it_is_seven": evaluated is not None and int(evaluated.values[0]) == 7,
    }


def comparisons_do_not_chain() -> dict:
    """A less than B less than C is refused rather than given one of two readings.

    Python reads it as a conjunction and C reads it as a comparison against a boolean. A user
    cannot tell from the query which they got, so this parser gives neither.
    """
    caught = ""
    try:
        parse("select id from facts where 1 < id < 10")
    except ParseError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_says_what_to_write_instead": "and" in caught,
        "and_the_spelled_out_form_works": parse(
            "select id from facts where 1 < id and id < 10"
        ).predicate
        is not None,
    }


def a_missing_as_is_refused() -> dict:
    """An alias needs the keyword, so a missing comma is an error rather than a rename.

    Legal SQL allows the bare form. It also turns select a b from t into a column called b, and
    that is a query that runs and answers a different question, which is the failure this parser
    will not have.
    """
    caught = ""
    try:
        parse("select id amount from facts")
    except ParseError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "and_the_explicit_form_works": parse("select id as amount from facts").items[0].alias
        == "amount",
        "and_the_comma_form_works": len(parse("select id, amount from facts").items) == 2,
    }


def a_keyword_cannot_be_an_alias() -> dict:
    """Aliasing to a reserved word is refused, naming the word."""
    caught = ""
    try:
        parse("select id as from facts")
    except ParseError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_the_word": "'from'" in caught,
    }


def a_join_orients_itself() -> dict:
    """On a.k = b.k and on b.k = a.k build the same plan.

    The qualifier says which table each name belongs to, so the side a user wrote a key on is
    not information the plan needs. Without the qualifiers the order as written is all there is,
    and is kept.
    """
    catalogue = _catalogue()
    forward = plan("select id from facts join shops on facts.shop = shops.shop", catalogue)
    backward = plan("select id from facts join shops on shops.shop = facts.shop", catalogue)
    return {
        "forward": _find_join(forward),
        "backward": _find_join(backward),
        "they_agree": _find_join(forward) == _find_join(backward),
    }


def _find_join(one: Plan) -> tuple:
    """The keys of the first join in a plan, for comparing two builds."""
    for node in walk(one):
        if isinstance(node, Join):
            return (node.left_keys, node.right_keys)
    return ((), ())


def the_build_order_is_scan_join_filter_group_sort_limit() -> dict:
    """A query using every clause, as a plan, read from the top down.

    Written expecting a projection on top and there is not one, which is the rule about a
    projection naming exactly the columns underneath doing its work here rather than in a case
    invented to show it off. A group by produces its keys and then its aggregates, in that
    order, and a select list naming the same things in the same order is the identity. The name
    of this measurement lost a word to the finding.
    """
    catalogue = _catalogue()
    built = plan(
        "select region, count(*) as n from facts join shops on facts.shop = shops.shop "
        "where amount > 90 group by region order by n desc limit 3",
        catalogue,
    )
    reordered = plan(
        "select count(*) as n, region from facts join shops on facts.shop = shops.shop "
        "where amount > 90 group by region order by n desc limit 3",
        catalogue,
    )
    order = [type(one).__name__ for one in walk(built)]
    return {
        "order": order,
        "there_is_no_projection": "Project" not in order,
        "it_starts_at_the_limit": order[0] == "Limit",
        "then_the_sort": order[1] == "Sort",
        "then_the_group": order[2] == "Group",
        "then_the_filter": order[3] == "Filter",
        "then_the_join": order[4] == "Join",
        "and_swapping_the_select_list_puts_one_back": isinstance(reordered, Project),
    }


def ordering_by_an_unselected_column_works() -> dict:
    """Order by a column the query does not return, which is why project is built last.

    A builder projecting before sorting would have dropped the column and would refuse this, and
    the refusal would be about its own construction order rather than about the query.
    """
    catalogue = _catalogue()
    built = plan("select id from facts order by amount desc limit 5", catalogue)
    return {
        "columns": list(built.schema().names),
        "it_returns_one_column": len(built.schema()) == 1,
        "and_sorted_by_another": isinstance(built, Project)
        and any(one.name == "amount" for one in _find_sort(built)),
    }


def _find_sort(one: Plan) -> tuple[SortKey, ...]:
    """The keys of the first sort in a plan."""
    for node in walk(one):
        if isinstance(node, Sort):
            return node.keys
    return ()


def an_aggregate_with_no_group_by_is_one_group() -> dict:
    """Count of a table, which the standard makes one group over everything."""
    catalogue = _catalogue()
    built = plan("select count(*) as n from facts", catalogue)
    return {
        "columns": list(built.schema().names),
        "it_has_one_column": len(built.schema()) == 1,
        "it_is_a_group": any(
            isinstance(one, Group)
            for one in __import__("cqe.plan.logical", fromlist=["walk"]).walk(built)
        ),
    }


def selecting_an_ungrouped_column_is_refused() -> dict:
    """A column that is neither grouped nor aggregated has no single value per group.

    Some engines pick an arbitrary row. That is a query returning a value the user did not ask
    for and cannot predict, which is worse than a refusal naming the column.
    """
    caught = ""
    try:
        plan("select label, count(*) as n from facts group by shop", _catalogue())
    except SchemaError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_the_column": "label" in caught,
        "and_grouping_by_it_works": plan(
            "select label, count(*) as n from facts group by label", _catalogue()
        )
        is not None,
    }


def a_misspelled_column_names_the_alternatives() -> dict:
    """The resolver refuses with the schema, which is the error the split buys.

    A parser doing both halves at once would say only that something went wrong at a character
    position, because at parse time it has no idea what the columns are called.
    """
    caught = ""
    try:
        plan("select id from facts where amont > 5", _catalogue())
    except UnknownColumn as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_the_column": "amont" in caught,
        "and_lists_the_real_ones": "amount" in caught,
        "the_parse_itself_was_fine": parse("select id from facts where amont > 5") is not None,
    }


def a_missing_table_lists_the_catalogue() -> dict:
    """The same split, one level up."""
    caught = ""
    try:
        plan("select id from nowhere", _catalogue())
    except SchemaError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_lists_what_there_is": "facts" in caught and "shops" in caught,
    }


def trailing_text_is_refused() -> dict:
    """A query that ends and then continues, which is usually a typo in a clause name."""
    caught = ""
    try:
        parse("select id from facts wehre id > 1")
    except ParseError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_the_leftover": "wehre" in caught,
    }


def an_in_list_parses() -> dict:
    """A membership test, which is a shorthand for a chain of ors and is measured as one."""
    parsed = parse("select id from facts where shop in (1, 2, 3)").predicate
    return {
        "shape": type(parsed).__name__,
        "it_is_a_list": isinstance(parsed, InList),
        "options": list(parsed.options) if isinstance(parsed, InList) else [],
        "strings_work": list(
            parse("select id from facts where label in ('a', 'b')").predicate.options
        ),
    }


def is_null_parses_both_ways() -> dict:
    """Is null and is not null, which are the only way to test for a null."""
    plain = parse("select id from facts where amount is null").predicate
    negated = parse("select id from facts where amount is not null").predicate
    return {
        "plain": type(plain).__name__,
        "both_are_null_tests": isinstance(plain, IsNull) and isinstance(negated, IsNull),
        "one_is_negated": negated.negated and not plain.negated,
    }


def a_query_round_trips_through_its_own_description() -> dict:
    """Describe a parsed query, parse the description, and compare.

    Not a proof that the description is faithful, but it catches the clause the renderer forgot,
    which is the failure a describe function actually has.
    """
    original = (
        "select region, count(*) as n from facts join shops on facts.shop = shops.shop "
        "where amount > 90 group by region order by n desc limit 3"
    )
    once = render(original)
    twice = render(once)
    return {
        "once": once,
        "it_is_stable": once == twice,
        "it_kept_the_join": "join shops" in once,
        "the_group": "group by region" in once,
        "the_order": "order by n desc" in once,
        "the_limit": "limit 3" in once,
    }


def a_parsed_query_evaluates_to_the_same_rows_as_the_reference(rows: int = 500) -> dict:
    """The whole point: a query parsed, planned, run, and checked against the reference.

    The reference is verify/reference.py, a row at a time interpreter that shares no code with
    any of this. It is given the same filter and the same ordering written directly in Python.
    """
    state = np.random.default_rng(5)
    batch = Batch.from_columns(
        [
            integer_column("id", np.arange(rows)),
            integer_column("shop", state.integers(0, 10, rows)),
            floating_column("amount", state.normal(100, 20, rows)),
        ]
    )
    built = plan(
        "select id, amount from facts where amount > 100 order by amount desc", {"facts": batch}
    )
    produced = execute(built, {"facts": batch})
    kept = where(Rows.of(batch), lambda one: one["amount"] > 100)
    expected = select(order_by(kept, ["amount"], descending=[True]), ["id", "amount"])
    result = agree(Rows.of(produced), expected, ordered=True)
    return {
        "rows": produced.rows,
        "expected": len(expected.rows),
        "they_agree": bool(result),
        "and_in_the_same_order": bool(result),
    }


def compare_the_clauses() -> list[dict]:
    """Plan sizes for queries of rising complexity, which is the module in one table."""
    catalogue = _catalogue()
    queries = (
        "select * from facts",
        "select id, amount from facts",
        "select id, amount from facts where amount > 100",
        "select shop, count(*) as n from facts group by shop",
        "select shop, count(*) as n from facts group by shop order by n desc limit 5",
        "select region, count(*) as n from facts join shops on facts.shop = shops.shop "
        "group by region",
    )
    return [
        {
            "query": one if len(one) < 60 else one[:57] + "...",
            "nodes": len(walk(plan(one, catalogue))),
            "columns": len(plan(one, catalogue).schema()),
        }
        for one in queries
    ]


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "aggregates": len(AGGREGATE_NAMES),
        "comparisons": len(COMPARISONS),
        "precedence_holds": precedence_is_the_standard_one()["and_binds_tighter"],
        "chaining_refused": comparisons_do_not_chain()["it_was_refused"],
        "star_has_no_project": a_star_is_not_a_projection()["the_star_has_no_project"],
    }
