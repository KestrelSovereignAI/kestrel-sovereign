"""#3120's migration, run against a real PostgreSQL as well as SQLite.

The compare-and-set at its centre needs an equality that is true of two NULLs,
and the two engines spell that differently: SQLite reads ``IS`` as null-safe
equality and takes a parameter on the right, while PostgreSQL's ``IS`` is a
unary predicate — ``IS NULL``, ``IS TRUE`` — and refuses a bind parameter
outright. Spelled SQLite's way on PostgreSQL, the migration raises inside
``_init_schema`` on the first post-upgrade boot of any agent that has work to
do, which is every host this was written for.

A unit test can pin the string. Only this can say the engine accepts it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from kestrel_sovereign.storage import AsyncStorage
from kestrel_sovereign.storage.legacy_session_stamp import (
    STAMP_TABLE,
    stamp_legacy_sessions,
)

SESSION = "stamped-session-3120"
BASE = datetime(2026, 6, 1, 9, 0, 0)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_migration_stamps_a_legacy_row_on_both_backends(db_backend):
    agent_id = f"did:test:stamp:{uuid4()}"
    storage = AsyncStorage(backend=db_backend, agent_id=agent_id)
    await storage.initialize()
    db = storage.db

    ids = []
    for minute, metadata in (
        (0, {"session_id": SESSION}),
        (1, {"session_id": "999999"}),  # a bare integer naming no live row
    ):
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                agent_id,
                "user",
                f"turn at {minute}",
                json.dumps(metadata),
                BASE + timedelta(minutes=minute),
            ),
        )
        row = await db.fetchone(
            "SELECT id FROM conversation_history WHERE agent_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (agent_id,),
        )
        ids.append(row[0])

    # The agent was created before those rows existed, so its marker — if the
    # boot pass wrote one — describes an empty history. Clear it and run the
    # pass against the rows as they now stand.
    await db.execute(
        f"DELETE FROM {STAMP_TABLE} WHERE agent_id = ?", (agent_id,)
    )
    await stamp_legacy_sessions(db)

    stamped = await db.fetchone(
        "SELECT metadata, session_id FROM conversation_history WHERE id = ?",
        (ids[1],),
    )
    assert json.loads(stamped[0])["session_id"] == SESSION
    assert stamped[1] == SESSION
    recorded = await db.fetchone(
        f"SELECT rows_stamped FROM {STAMP_TABLE} WHERE agent_id = ?", (agent_id,)
    )
    assert recorded == (1,)
    await storage.close()
