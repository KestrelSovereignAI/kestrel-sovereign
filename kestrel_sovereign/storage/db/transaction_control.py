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


def _statement_heads(sql: str, *, postgres: bool, backslash_strings: bool):
    position, length, head = 0, len(sql), True
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
            quote = character
            position += 1
            while position < length:
                if sql[position] == quote:
                    position += 1
                    if position < length and sql[position] == quote:
                        position += 1
                        continue
                    break
                if quote == "'" and backslash_strings and sql[position] == "\\":
                    position += 2
                else:
                    position += 1
            head = False
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
        elif character == ";":
            position += 1
            head = True
        elif _identifier_character(character):
            start = position
            while position < length and _identifier_character(sql[position]):
                position += 1
            word = sql[start:position]
            if head:
                yield word.upper()
            # PostgreSQL E'...' always uses backslash escapes, regardless of
            # the ordinary-string setting. Consume it in the same quote path.
            if (
                postgres
                and word.upper() == "E"
                and position < length
                and sql[position] == "'"
            ):
                position += 1
                while position < length:
                    if sql[position] == "\\":
                        position += 2
                    elif sql[position] == "'":
                        position += 1
                        if position < length and sql[position] == "'":
                            position += 1
                        else:
                            break
                    else:
                        position += 1
            head = False
        else:
            position += 1
            head = False


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
