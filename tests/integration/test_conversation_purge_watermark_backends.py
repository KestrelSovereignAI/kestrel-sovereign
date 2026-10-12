"""The EPHEMERAL conversation sweep compares at the column's precision (#3227).

``conversation_history.created_at`` is ``NOW()``-stamped to the microsecond on
PostgreSQL and whole-second by CHECK on SQLite (#3009). The wrapper hands the
sweep the exact transition instant. PostgreSQL compares exactly, so a NORMAL
row written 300 ms before the transition survives; handed whole seconds, it
was destroyed. SQLite cannot tell a leak written in the transition second from
a row written just before it, and a privacy sweep must not leave leaks behind,
so it compares against the second the transition falls in and purges all of
it, as it always did.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.conversation_created_at import created_at_bind
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

TRANSITION = datetime(2026, 9, 7, 12, 0, 5, 500000, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def storage(db_backend):
    store = AsyncStorage(backend=db_backend, agent_id=f"did:test:conv-{uuid.uuid4().hex}")
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


async def _insert(storage, content: str, stamp: datetime) -> None:
    """A row stamped as its backend would stamp a write at ``stamp``."""
    await storage.db.execute_commit(
        "INSERT INTO conversation_history (agent_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?)",
        (storage.agent_id, "user", content, created_at_bind(storage.db.backend_type, stamp)),
    )


async def _contents(storage) -> set:
    rows = await storage.db.fetchall(
        "SELECT content FROM conversation_history WHERE agent_id = ?", (storage.agent_id,)
    )
    return {row[0] for row in rows}


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_conversation_same_second_boundary_through_the_wrapper_on_backend(
    storage, monkeypatch
):
    await _insert(storage, "before", TRANSITION - timedelta(milliseconds=300))
    monkeypatch.setattr(
        PrivacyEnforcingStorage, "_now_utc", staticmethod(lambda: TRANSITION)
    )
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    await _insert(storage, "after", TRANSITION + timedelta(milliseconds=300))
    await _insert(storage, "later", TRANSITION + timedelta(seconds=1))

    report = await wrapper.purge_ephemeral_session(reason="test-conversation-same-second")

    assert not report.required_sweep_failed, report.failed_stores
    remaining = await _contents(storage)
    if storage.db.backend_type == "postgres":
        # Microsecond column: exact.
        assert (report["conversation_history"], remaining) == (2, {"before"})
    else:
        # Whole-second column: "before" and "after" are both stored as
        # 12:00:05, so the whole transition second goes — leak cleanup is
        # not weakened.
        assert (report["conversation_history"], remaining) == (3, set())


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_an_undatable_watermark_is_refused_on_backend(storage):
    await _insert(storage, "row", TRANSITION)
    with pytest.raises(ValueError, match="datable watermark"):
        await storage.purge_conversations_since("not a timestamp", reason="test")
    assert await _contents(storage) == {"row"}
