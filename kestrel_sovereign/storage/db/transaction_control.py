"""SQL may not replace a native adapter's owning transaction boundary.

This is a lexical transaction-command refusal, not SQL validation or an
authorization parser. The database still parses and validates each statement.
SQLite statement boundaries use its own completeness parser (including trigger
bodies). PostgreSQL boundaries respect nested comments, dollar-quoted bodies,
quoted identifiers and both ordinary-string backslash settings. Refuse if
either possible string setting exposes a transaction command; caller SQL must
not change the session setting and thereby change the guard's interpretation.
"""

from __future__ import annotations

import re
import sqlite3

from .interface import QueryError

_CONTROL = frozenset(
    {
        "BEGIN",
        "START",
        "COMMIT",
        "END",
        "ROLLBACK",
        "ABORT",
        "SAVEPOINT",
        "RELEASE",
        "PREPARE",
    }
)
_CANDIDATE = re.compile(
    r"\b(?:BEGIN|START|COMMIT|END|ROLLBACK|ABORT|SAVEPOINT|RELEASE|PREPARE)\b",
    re.IGNORECASE,
)


def sqlite_statements(script: str) -> list[str]:
    """Split native SQLite statements without splitting trigger bodies."""
    statements = []
    start = 0
    for position, character in enumerate(script):
        if character == ";" and sqlite3.complete_statement(
            script[start : position + 1]
        ):
            statements.append(script[start : position + 1])
            start = position + 1
    if script[start:].strip():
        statements.append(script[start:])
    return statements


def _identifier_character(character: str) -> bool:
    return character.isalnum() or character in "_$" or ord(character) >= 128


def _quoted_end(sql: str, position: int, *, escapes: bool, postgres: bool) -> int:
    """Consume a quote token, retaining escape mode across PG continuations.

    PostgreSQL treats newline-separated single-quoted segments as one token:
    E appears only on its first segment. A separate ordinary string later in
    the statement does not inherit that mode. Comments are whitespace too.
    """
    quote, length = sql[position], len(sql)
    while True:
        position += 1
        while position < length:
            if sql[position] == quote:
                position += 1
                if position < length and sql[position] == quote:
                    position += 1
                    continue
                break
            if quote == "'" and escapes and sql[position] == "\\":
                position += 2
            else:
                position += 1
        if not postgres or quote != "'":
            return position
        lookahead, newline = position, False
        while lookahead < length:
            if sql[lookahead].isspace():
                newline |= sql[lookahead] in "\r\n"
                lookahead += 1
            elif sql.startswith("--", lookahead):
                lookahead += 2
                while lookahead < length and sql[lookahead] not in "\r\n":
                    lookahead += 1
            elif sql.startswith("/*", lookahead):
                lookahead += 2
                depth = 1
                while lookahead < length and depth:
                    if sql.startswith("/*", lookahead):
                        depth += 1
                        lookahead += 2
                    elif sql.startswith("*/", lookahead):
                        depth -= 1
                        lookahead += 2
                    else:
                        newline |= sql[lookahead] in "\r\n"
                        lookahead += 1
            else:
                break
        if not newline or lookahead >= length or sql[lookahead] != "'":
            return position
        position = lookahead


def _statement_heads(sql: str, *, postgres: bool, backslash_strings: bool):
    position, length, head = 0, len(sql), True
    create_definition, routine_definition = False, False
    atomic_depth, case_depth = 0, 0
    previous_word = None
    while position < length:
        character = sql[position]
        if character.isspace() or character == "\ufeff":
            position += 1
        elif sql.startswith("--", position):
            position += 2
            while position < length and sql[position] not in "\r\n":
                position += 1
        elif sql.startswith("/*", position):
            position += 2
            depth = 1
            while position < length and depth:
                if postgres and sql.startswith("/*", position):
                    depth += 1
                    position += 2
                elif sql.startswith("*/", position):
                    depth -= 1
                    position += 2
                else:
                    position += 1
        elif character in "'\"":
            position = _quoted_end(
                sql, position, escapes=backslash_strings, postgres=postgres
            )
            head = False
            previous_word = None
        elif postgres and character == "$":
            # A tag is recognized only at a token boundary. Identifier scans
            # below consume embedded '$', so foo$tag$ is never a string body.
            # PostgreSQL's identifier grammar permits any high-bit byte, not
            # only Unicode letters: an emoji/non-ASCII tag is opaque too.
            match = re.match(
                r"\$(?:[A-Za-z_\x80-\U0010ffff][A-Za-z_0-9\x80-\U0010ffff]*|)\$",
                sql[position:],
            )
            if match:
                tag = match.group()
                end = sql.find(tag, position + len(tag))
                position = length if end < 0 else end + len(tag)
            else:
                position += 1
            head = False
            previous_word = None
        elif character == ";":
            position += 1
            if not atomic_depth:
                head = True
                create_definition = routine_definition = False
            previous_word = None
        elif _identifier_character(character):
            start = position
            while position < length and _identifier_character(sql[position]):
                position += 1
            word = sql[start:position]
            upper = word.upper()
            if head:
                yield upper
                create_definition = postgres and upper == "CREATE"
            if create_definition and upper in {"FUNCTION", "PROCEDURE"}:
                routine_definition = True
            if routine_definition and upper == "ATOMIC" and previous_word == "BEGIN":
                atomic_depth += 1
            elif atomic_depth and upper == "CASE":
                case_depth += 1
            elif atomic_depth and upper == "END":
                if case_depth:
                    case_depth -= 1
                else:
                    atomic_depth -= 1
            # PostgreSQL E'...' always uses backslash escapes, regardless of
            # the ordinary-string setting. Consume it in the same quote path.
            if (
                postgres
                and word.upper() == "E"
                and position < length
                and sql[position] == "'"
            ):
                position = _quoted_end(sql, position, escapes=True, postgres=True)
            head = False
            previous_word = upper
        else:
            position += 1
            head = False
            previous_word = None


def reject_transaction_control(sql: str, *, dialect: str) -> None:
    """Reject SQL transaction commands before any owning-connection I/O."""
    if not _CANDIDATE.search(sql):
        return
    if dialect == "sqlite":
        # A SQLite trigger is one complete statement; inner BEGIN/END belong
        # to its body, not this connection's transaction. Inspect only its head.
        for statement in sqlite_statements(sql):
            head = next(
                _statement_heads(statement, postgres=False, backslash_strings=False),
                None,
            )
            if head in _CONTROL:
                raise QueryError("owned SQL may not contain transaction control")
    elif dialect == "postgres":
        for backslash in (False, True):
            if any(
                head in _CONTROL
                for head in _statement_heads(
                    sql, postgres=True, backslash_strings=backslash
                )
            ):
                raise QueryError("owned SQL may not contain transaction control")
    else:
        raise ValueError("unsupported native transaction dialect")
