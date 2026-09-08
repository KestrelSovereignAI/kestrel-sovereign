"""The conversation purge honours the exact instant on Postgres and floors it
on SQLite (#3227 review r3).

``conversation_history.created_at`` is whole-second on SQLite by CHECK, but
``TIMESTAMP`` written with ``NOW()`` on Postgres — microseconds. Handed the
whole-second watermark, the Postgres purge destroyed a NORMAL row written
30 ms before the transition. Handed the exact instant, Postgres compares
exactly; SQLite cannot tell a leak in the transition second from a row just
before it, and a privacy sweep must not leave leaks behind, so it floors the
watermark and purges the whole second, as it always did.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_storage import AsyncStorage


@pytest_asyncio.fixture
async def storage(db_backend):
    store = AsyncStorage(backend=db_backend, agent_id=f"did:test:conv-{uuid.uuid4().hex}")
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


async def _insert(storage, content: str, created_at: str) -> None:
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?)",
        (storage.agent_id, "user", content,
         storage.conversation._timestamp_query_param(created_at)
         if storage.db.backend_type == "postgres" else created_at[:19]),
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_conversation_same_second_boundary_on_backend(storage):
    transition = datetime(2026, 9, 7, 12, 0, 5, 500000, tzinfo=timezone.utc)
    before = (transition - timedelta(milliseconds=300)).strftime("%Y-%m-%d %H:%M:%S.%f")
    after = (transition + timedelta(milliseconds=300)).strftime("%Y-%m-%d %H:%M:%S.%f")
    later = (transition + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S.%f")
    await _insert(storage, "before", before)
    await _insert(storage, "after", after)
    await _insert(storage, "later", later)

    purged = await storage.purge_conversations_since(
        transition.strftime("%Y-%m-%d %H:%M:%S.%f"), reason="test"
    )
    rows = {r[0] for r in await storage.db.fetchall(
        "SELECT content FROM conversation_history WHERE agent_id = ?", (storage.agent_id,)
    )}
    if storage.db.backend_type == "postgres":
        # Microsecond column: exact.
        assert purged == 2 and rows == {"before"}, (purged, rows)
    else:
        # Whole-second column: the watermark is floored to 12:00:05 and the
        # whole transition second is purged — leak cleanup is not weakened.
        assert purged == 3 and rows == set(), (purged, rows)
