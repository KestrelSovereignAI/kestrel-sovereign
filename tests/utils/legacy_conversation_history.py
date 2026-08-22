"""Seed a pre-#3009 ``conversation_history`` and open it (#3009).

``created_at`` now carries a CHECK, so a spelling this codebase used to hold
cannot be written through ``AsyncDatabase`` any more. Cases that are ABOUT such
a value therefore have to put it on disk the way an upgrading host already has
it: with the raw driver, against the shape the column had before the
constraint, and then open the database so ``_init_schema`` migrates it.

That is a real state, not a manufactured one — it is what every host that
upgraded through #2959 has right now. Writing these rows through the current
schema would be writing them through the very constraint the case is about, and
disabling the constraint to admit them would be testing a world that no longer
exists.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Sequence

from kestrel_sovereign.storage.async_database import AsyncDatabase

#: The table as it stood before #3009: ``created_at`` nullable and untyped in
#: practice, and none of the columns later migrations add. Copied from a live
#: agent database rather than trimmed from the current DDL by hand.
LEGACY_CONVERSATION_HISTORY_DDL = """CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    session_id TEXT DEFAULT NULL,
    created_at TIMESTAMP,
    deleted_at TIMESTAMP DEFAULT NULL
)"""


def write_legacy_history(
    path: str,
    rows: Iterable[Sequence],
    *,
    extra_ddl: Iterable[str] = (),
) -> str:
    """Create ``path`` holding ``rows`` in the pre-#3009 shape.

    Each row is ``(agent_id, role, content, metadata, session_id, created_at)``.
    ``extra_ddl`` runs after the table is created, for a case that needs a
    column a later ``ALTER`` adds.
    """
    connection = sqlite3.connect(path)
    try:
        connection.execute(LEGACY_CONVERSATION_HISTORY_DDL)
        for statement in extra_ddl:
            connection.execute(statement)
        connection.executemany(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            list(rows),
        )
        connection.commit()
    finally:
        connection.close()
    return path


async def open_legacy_history(
    path: str, rows: Iterable[Sequence], *, extra_ddl: Iterable[str] = ()
) -> AsyncDatabase:
    """Seed a pre-#3009 database and open it, running the migration."""
    write_legacy_history(path, rows, extra_ddl=extra_ddl)
    return await AsyncDatabase.sqlite(path)
