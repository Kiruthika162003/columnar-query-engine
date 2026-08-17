from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from cqe.errors import ParseError

# The tokeniser, which is small and has three decisions in it worth the space.
#
# It records a position for every token. That is the only reason a parse error can point at the
# character it failed on, and pointing at the character is the difference between an error a
# user can fix and one they have to guess at. It costs one integer per token.
#
# It does not fold keywords into a separate token kind. A keyword is a word, and whether a
# particular word is a keyword depends on where it appears: a column called count is legal and a
# tokeniser that decided otherwise would make it unusable. The parser asks whether the word it
# is looking at is the keyword it wants, which is where that knowledge belongs.
#
# It refuses an unterminated string rather than running to the end of the input. That is the one
# error a tokeniser can diagnose better than a parser, because the parser sees a token stream
# that ended and the tokeniser knows where the quote was.

KEYWORDS = frozenset(
    {
        "select",
        "from",
        "where",
        "group",
        "by",
        "order",
        "limit",
        "offset",
        "join",
        "on",
        "as",
        "and",
        "or",
        "not",
        "is",
        "null",
        "in",
        "asc",
        "desc",
        "count",
        "sum",
        "min",
        "max",
        "avg",
        "having",
        "distinct",
    }
)

SYMBOLS = ("<=", ">=", "!=", "<>", "=", "<", ">", "(", ")", ",", "*", "+", "-", ".")

WORD = "word"
NUMBER = "number"
TEXT = "text"
SYMBOL = "symbol"
END = "end"


@dataclass(frozen=True)
class Token:
    """One token: what kind, what it says, and where it was."""

    kind: str
    value: str
    position: int

    @property
    def is_keyword(self) -> bool:
        """Whether this word happens to be one of the reserved ones."""
        return self.kind == WORD and self.value.lower() in KEYWORDS

    def matches(self, kind: str, value: str | None = None) -> bool:
        """Whether this token is of a kind and optionally a particular value.

        Case insensitive on words, because SQL is, and exact on everything else, because a
        string literal that changed case would be a different literal.
        """
        if self.kind != kind:
            return False
        if value is None:
            return True
        if kind == WORD:
            return self.value.lower() == value.lower()
        return self.value == value

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {"kind": self.kind, "value": self.value, "position": self.position}

    def __str__(self) -> str:
        return f"{self.kind} {self.value!r} at {self.position}"


def tokenise(text: str) -> list[Token]:
    """Cut a query into tokens, recording where each one started.

    One pass, no regular expressions. A regular expression would be shorter and would make the
    position bookkeeping implicit, and the position bookkeeping is the whole reason this
    function exists rather than a call to split.
    """
    out: list[Token] = []
    position = 0
    length = len(text)
    while position < length:
        character = text[position]
        if character.isspace():
            position += 1
            continue
        if character == "-" and text.startswith("--", position):
            while position < length and text[position] != "\n":
                position += 1
            continue
        if character == "'":
            token, position = _string(text, position)
            out.append(token)
            continue
        if character.isdigit() or (
            character == "." and position + 1 < length and text[position + 1].isdigit()
        ):
            token, position = _number(text, position)
            out.append(token)
            continue
        if character.isalpha() or character == "_":
            token, position = _word(text, position)
            out.append(token)
            continue
        symbol = _symbol(text, position)
        if symbol is None:
            raise ParseError(f"{character!r} is not part of a query", position, text)
        out.append(Token(kind=SYMBOL, value=symbol, position=position))
        position += len(symbol)
    out.append(Token(kind=END, value="", position=length))
    return out


def _string(text: str, start: int) -> tuple[Token, int]:
    """A quoted string, with two quotes meaning one."""
    position = start + 1
    pieces: list[str] = []
    while position < len(text):
        character = text[position]
        if character == "'":
            if position + 1 < len(text) and text[position + 1] == "'":
                pieces.append("'")
                position += 2
                continue
            return Token(kind=TEXT, value="".join(pieces), position=start), position + 1
        pieces.append(character)
        position += 1
    raise ParseError("a string was opened and never closed", start, text)


def _number(text: str, start: int) -> tuple[Token, int]:
    """An integer or a decimal, without an exponent.

    No exponent because nothing in this engine produces one and supporting it would mean
    deciding whether 1e5 is a number or a number followed by a column called e5. That ambiguity
    is real and the cheapest way to avoid it is to not have the syntax.
    """
    position = start
    seen_point = False
    while position < len(text):
        character = text[position]
        if character.isdigit():
            position += 1
            continue
        if character == "." and not seen_point:
            seen_point = True
            position += 1
            continue
        break
    value = text[start:position]
    if value in (".", ""):
        raise ParseError("that is not a number", start, text)
    return Token(kind=NUMBER, value=value, position=start), position


def _word(text: str, start: int) -> tuple[Token, int]:
    """An identifier or a keyword, which are the same thing here."""
    position = start
    while position < len(text) and (text[position].isalnum() or text[position] == "_"):
        position += 1
    return Token(kind=WORD, value=text[start:position], position=start), position


def _symbol(text: str, position: int) -> str | None:
    """The longest symbol starting here, or nothing.

    Longest first, so that a less than or equal sign is one token and not two. Getting that
    wrong produces a parser that accepts a query and means something different by it, which is
    worse than one that rejects it.
    """
    for candidate in SYMBOLS:
        if text.startswith(candidate, position):
            return candidate
    return None


@dataclass
class Stream:
    """A cursor over a token list, which is what the parser holds."""

    tokens: tuple[Token, ...]
    text: str
    position: int = 0

    @property
    def current(self) -> Token:
        """The token the cursor is on."""
        return self.tokens[min(self.position, len(self.tokens) - 1)]

    @property
    def done(self) -> bool:
        """Whether the stream has reached its end token."""
        return self.current.kind == END

    def peek(self, ahead: int = 0) -> Token:
        """A token without moving the cursor."""
        return self.tokens[min(self.position + ahead, len(self.tokens) - 1)]

    def take(self) -> Token:
        """The current token, moving past it."""
        token = self.current
        self.position += 1
        return token

    def accept(self, kind: str, value: str | None = None) -> Token | None:
        """Take the current token if it matches, or leave the cursor alone."""
        if self.current.matches(kind, value):
            return self.take()
        return None

    def expect(self, kind: str, value: str | None = None) -> Token:
        """Take the current token if it matches, or refuse with a position.

        The refusal names what was wanted and what was found, in that order, and carries the
        position so the caller can point at it. A parser error that says only what was found
        makes the reader work out what was expected, which is the parser's job.
        """
        token = self.accept(kind, value)
        if token is None:
            wanted = f"{value!r}" if value else kind
            raise ParseError(
                f"expected {wanted} and found {self.current.value!r}",
                self.current.position,
                self.text,
            )
        return token

    def expect_keyword(self, *words: str) -> Token:
        """Take one of several keywords, or refuse listing all of them."""
        for word in words:
            token = self.accept(WORD, word)
            if token is not None:
                return token
        raise ParseError(
            f"expected one of {list(words)} and found {self.current.value!r}",
            self.current.position,
            self.text,
        )

    def looking_at(self, *words: str) -> bool:
        """Whether the current token is one of these keywords, without taking it."""
        return any(self.current.matches(WORD, word) for word in words)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)


def stream(text: str) -> Stream:
    """A cursor over a tokenised query."""
    return Stream(tokens=tuple(tokenise(text)), text=text)


def kinds(text: str) -> list[str]:
    """The token kinds a query produces, for measurements and for tests."""
    return [token.kind for token in tokenise(text) if token.kind != END]


def values(text: str) -> list[str]:
    """The token values a query produces."""
    return [token.value for token in tokenise(text) if token.kind != END]


def a_query_tokenises(text: str = "select a, b from t where a < 10") -> dict:
    """The ordinary case, which is most of what a tokeniser does."""
    tokens = tokenise(text)
    return {
        "tokens": len(tokens),
        "kinds": kinds(text),
        "values": values(text),
        "it_ends_with_an_end_token": tokens[-1].kind == END,
        "every_token_has_a_position": all(token.position >= 0 for token in tokens),
    }


def positions_point_at_the_source(text: str = "select a from t") -> dict:
    """Every token's position indexes the original text, which is what makes errors readable.

    Checked by slicing the input at each position and confirming it starts with the token. That
    is a stronger property than the positions merely increasing, and it is the one an error
    message depends on.
    """
    tokens = [token for token in tokenise(text) if token.kind != END]
    return {
        "tokens": len(tokens),
        "every_position_lands_on_its_token": all(
            text[token.position :].lower().startswith(token.value.lower())
            for token in tokens
            if token.kind in (WORD, NUMBER, SYMBOL)
        ),
        "they_increase": [token.position for token in tokens]
        == sorted(token.position for token in tokens),
    }


def the_longest_symbol_wins() -> dict:
    """Less than or equal is one token, which a shortest first scanner gets wrong.

    The failure is not a refused query, it is an accepted query that means something else: a
    tokeniser splitting it into two tokens leaves the parser reading a less than followed by an
    equals, and a parser tolerant enough to continue would compare against the wrong thing.
    """
    return {
        "less_or_equal": values("a <= 1"),
        "greater_or_equal": values("a >= 1"),
        "not_equal": values("a != 1"),
        "angle_not_equal": values("a <> 1"),
        "they_are_single_tokens": all(
            len(values(f"a {symbol} 1")) == 3 for symbol in ("<=", ">=", "!=", "<>")
        ),
        "and_a_bare_one_still_works": values("a < 1") == ["a", "<", "1"],
    }


def a_keyword_is_just_a_word() -> dict:
    """Keywords are not a token kind, so a column can be called count.

    A tokeniser that classified keywords separately would make every reserved word unusable as
    an identifier, and the list of reserved words in this engine includes count, min, max and
    sum, which are exactly the names a table of counts would use.
    """
    tokens = tokenise("select count from count")
    words = [token for token in tokens if token.kind == WORD]
    return {
        "words": [token.value for token in words],
        "they_are_all_words": all(token.kind == WORD for token in words),
        "count_is_a_keyword": any(
            token.is_keyword for token in words if token.value == "count"
        ),
        "and_still_usable_as_a_name": len(words) == 4,
    }


def matching_is_case_insensitive_on_words_only() -> dict:
    """SELECT and select are the same token; 'A' and 'a' are different strings.

    The asymmetry is deliberate and is the one every tokeniser gets asked about. A keyword is
    syntax and syntax is case insensitive; a string literal is data and data is not.
    """
    keyword = tokenise("SELECT")[0]
    text = tokenise("'A'")[0]
    return {
        "select_matches_lowercase": keyword.matches(WORD, "select"),
        "and_uppercase": keyword.matches(WORD, "SELECT"),
        "a_string_is_exact": text.matches(TEXT, "A") and not text.matches(TEXT, "a"),
    }


def a_string_with_a_quote_in_it_survives() -> dict:
    """Two quotes mean one, which is the SQL rule and the only escaping this supports."""
    return {
        "plain": values("'hello'"),
        "escaped": values("'it''s'"),
        "it_became_one_quote": values("'it''s'") == ["it's"],
        "an_empty_string_is_a_token": values("''") == [""],
    }


def a_comment_is_skipped() -> dict:
    """Two dashes to the end of the line, which is the only comment syntax here."""
    text = "select a -- this is ignored\nfrom t"
    return {
        "values": values(text),
        "the_comment_is_gone": "this" not in values(text),
        "the_query_survived": values(text) == ["select", "a", "from", "t"],
    }


def a_decimal_is_one_token() -> dict:
    """A number with a point in it, and a point that is not part of a number."""
    return {
        "integer": values("42"),
        "decimal": values("1.5"),
        "leading_point": values(".5"),
        "a_qualified_name_is_three_tokens": values("t.a") == ["t", ".", "a"],
        "the_decimal_is_one_token": values("1.5") == ["1.5"],
    }


def an_unterminated_string_is_refused() -> dict:
    """The one error a tokeniser diagnoses better than a parser.

    The parser would see a token stream that ended early and could only say the query was
    incomplete. The tokeniser knows where the quote was opened and says so.
    """
    caught = ""
    position = -1
    try:
        tokenise("select a from t where g = 'unclosed")
    except ParseError as problem:
        caught = str(problem)
        position = problem.position
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_the_problem": "never closed" in caught,
        "it_points_at_the_quote": position == 26,
    }


def an_unknown_character_is_refused() -> dict:
    """A character that cannot start a token, reported with its position."""
    caught = ""
    position = -1
    try:
        tokenise("select a from t where a # 1")
    except ParseError as problem:
        caught = str(problem)
        position = problem.position
    return {
        "message": caught,
        "it_was_refused": bool(caught),
        "it_names_the_character": "'#'" in caught,
        "position": position,
    }


def an_error_marks_the_position() -> dict:
    """The rendered form of a parse error, which is what a command line prints.

    A caret under the offending character. Worth having as a method on the error rather than in
    the command line, so every consumer gets the same rendering and the position is never
    formatted by hand.
    """
    marked = ""
    try:
        tokenise("select a where a # 1")
    except ParseError as problem:
        marked = problem.marked()
    lines = marked.split("\n")
    return {
        "lines": len(lines),
        "it_has_three_lines": len(lines) == 3,
        "the_caret_is_under_the_character": lines[1].index("^") == 17,
        "the_first_line_is_the_query": lines[0].startswith("select"),
    }


def an_empty_query_is_one_end_token() -> dict:
    """Nothing at all, which the parser then refuses with a better message."""
    tokens = tokenise("")
    return {
        "tokens": len(tokens),
        "it_is_one_token": len(tokens) == 1,
        "and_it_is_the_end": tokens[0].kind == END,
    }


def a_stream_walks_and_refuses() -> dict:
    """The cursor the parser holds, and what it does when the query is wrong."""
    walker = stream("select a from t")
    first = walker.expect(WORD, "select")
    second = walker.take()
    caught = ""
    try:
        walker.expect(WORD, "where")
    except ParseError as problem:
        caught = str(problem)
    return {
        "first": first.value,
        "second": second.value,
        "the_refusal_names_both": "where" in caught and "from" in caught,
        "peeking_does_not_move": walker.peek().value == walker.current.value,
        "accept_returns_nothing_on_a_miss": walker.accept(WORD, "limit") is None,
    }


def a_stream_ends_cleanly() -> dict:
    """Reading past the end returns the end token rather than raising.

    Which lets the parser check for the end the same way it checks for anything else, instead of
    guarding every read. The alternative is an index error somewhere deep in an expression
    parser, reported at a position that means nothing.
    """
    walker = stream("select")
    walker.take()
    return {
        "done": walker.done,
        "reading_past_the_end_is_safe": walker.take().kind == END,
        "and_again": walker.take().kind == END,
    }


def expect_keyword_lists_the_options() -> dict:
    """A refusal naming every keyword that would have been accepted."""
    walker = stream("select a")
    walker.take()
    caught = ""
    try:
        walker.expect_keyword("from", "where", "limit")
    except ParseError as problem:
        caught = str(problem)
    return {
        "message": caught,
        "it_lists_them": all(word in caught for word in ("from", "where", "limit")),
        "and_names_what_was_found": "'a'" in caught,
    }


def compare_the_shapes(
    queries: Sequence[str] = (
        "select a from t",
        "select a, b from t where a < 10",
        "select g, count(*) from t group by g order by g desc limit 5",
    ),
) -> list[dict]:
    """Token counts for queries of rising complexity, which is the module in one table."""
    if not queries:
        raise ParseError("there is nothing to tokenise", -1, "")
    return [
        {
            "query": one,
            "tokens": len(kinds(one)),
            "words": kinds(one).count(WORD),
            "symbols": kinds(one).count(SYMBOL),
        }
        for one in queries
    ]


def summarise() -> dict:
    """The module in one mapping, for the command line and for logging."""
    return {
        "keywords": len(KEYWORDS),
        "symbols": len(SYMBOLS),
        "longest_symbol_wins": the_longest_symbol_wins()["they_are_single_tokens"],
        "positions_land": positions_point_at_the_source()["every_position_lands_on_its_token"],
        "keywords_are_words": a_keyword_is_just_a_word()["they_are_all_words"],
    }
