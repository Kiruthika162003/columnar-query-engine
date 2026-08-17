from __future__ import annotations

# One exception hierarchy for the whole engine, so a caller can catch everything this package
# raises without catching everything Python raises.
#
# The split is by who is at fault and what they can do about it, not by which module noticed.
# A schema mistake is the caller's and is fixable by changing the query. A corrupt file is the
# data's and is fixable by rewriting the file. A budget overrun is neither, and the caller has
# to decide whether to raise the budget or accept a partial answer. Those are three different
# reactions, so they are three different types.
#
# Every message names the thing that was wrong and the thing that was expected, in that order,
# because an error that says "invalid type" and stops has thrown away the only two facts the
# reader needed.


class QueryEngineError(Exception):
    """Anything this package raises deliberately."""


class SchemaError(QueryEngineError):
    """A column, type or name that does not line up with the data it is used against."""


class TypeMismatch(SchemaError):
    """Two types that cannot be combined, or one that cannot do what was asked of it."""


class UnknownColumn(SchemaError):
    """A name that is not in the schema it was looked up in."""


class ParseError(QueryEngineError):
    """Text that is not a query this engine understands."""

    def __init__(self, message: str, position: int = -1, text: str = "") -> None:
        super().__init__(message)
        self.position = position
        self.text = text

    def marked(self) -> str:
        """The offending text with a caret under the position, for a command line.

        Only useful when the position was recorded, and the parser records it everywhere, so a
        ParseError without one came from somewhere that should be fixed rather than from a case
        this cannot show.
        """
        if self.position < 0 or not self.text:
            return str(self)
        return f"{self.text}\n{' ' * self.position}^\n{self}"


class DataError(QueryEngineError):
    """Data that does not have the shape or the contents it claims to."""


class CorruptFile(DataError):
    """A stored file whose header, footer or checksum does not agree with its contents."""


class EncodingError(DataError):
    """A column that cannot be encoded the way it was asked to be, or decoded back."""


class PlanError(QueryEngineError):
    """A plan that cannot be built or cannot be run as written."""


class UnsupportedPlan(PlanError):
    """A shape of query the engine does not implement, as opposed to one that is wrong."""


class BudgetExceeded(QueryEngineError):
    """A limit on memory, values touched or spill volume that the run went past.

    Carries the limit and the amount reached, because the caller's next decision is whether to
    raise the limit and by how much, and a message alone makes them guess.
    """

    def __init__(self, kind: str, limit: float, reached: float) -> None:
        super().__init__(f"{kind} budget of {limit} exceeded at {reached}")
        self.kind = kind
        self.limit = limit
        self.reached = reached

    @property
    def overrun(self) -> float:
        """How far past the limit it went, as a fraction of the limit."""
        if self.limit <= 0:
            return float("inf")
        return (self.reached - self.limit) / self.limit


class ConfigError(QueryEngineError):
    """A setting that is not a setting: a negative width, an empty sweep, a zero batch."""
