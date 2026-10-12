"""The EPHEMERAL conversation sweep purges the whole transition second (#3227).

The wrapper hands every sweep the exact transition instant. The graph sweep
compares it exactly; this one does not, on either backend. SQLite's
``conversation_history.created_at`` is whole-second by CHECK (#3009), and
PostgreSQL's is ``NOW()``, the time the writing transaction *started*. On
neither can a row's stamp show that it was written before the watermark rather
than during the stint, and a privacy sweep must not leave leaks behind, so the
boundary is the start of the transition second, as it always was.
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
TRANSITION_SECOND = TRANSITION.replace(microsecond=0)


@pytest_asyncio.fixture
async def storage(db_backend):
    store = AsyncStorage(backend=db_backend, agent_id=f"did:test:conv-{uuid.uuid4().hex}")
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


def _pin_transition(monkeypatch, instant: datetime) -> None:
    """Pin the clock the wrapper reads its watermark from."""
    monkeypatch.setattr(PrivacyEnforcingStorage, "_now_utc", staticmethod(lambda: instant))


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
async def test_the_whole_transition_second_is_purged_through_the_wrapper_on_backend(
    storage, monkeypatch
):
    """Every row stamped in the transition second is purged, on both backends,
    including one stamped 300 ms before the watermark: on PostgreSQL that is
    what a leak looks like when its transaction began before the transition.
    The last microsecond of the previous second survives."""
    stamps = {
        "previous-second": TRANSITION_SECOND - timedelta(microseconds=1),
        "second-start": TRANSITION_SECOND,
        "before-watermark": TRANSITION - timedelta(milliseconds=300),
    }
    for content, stamp in stamps.items():
        await _insert(storage, content, stamp)
    _pin_transition(monkeypatch, TRANSITION)
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    await _insert(storage, "after-watermark", TRANSITION + timedelta(milliseconds=300))
    await _insert(storage, "later-second", TRANSITION + timedelta(seconds=1))

    report = await wrapper.purge_ephemeral_session(reason="test-conversation-same-second")

    assert not report.required_sweep_failed, report.failed_stores
    assert await _contents(storage) == {"previous-second"}, (
        "conversation rows stamped in the transition second were left behind"
    )
    assert report["conversation_history"] == 4


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_leak_stamped_when_its_transaction_began_is_purged_on_backend(
    storage, monkeypatch
):
    """The case the whole-second boundary exists for, on PostgreSQL's real
    clock. A transaction opens, the agent enters EPHEMERAL later in that
    second, and the transaction then writes a row through the real writer.
    ``NOW()`` stamps the row with the transaction's start, before the
    watermark; the sweep must still purge it."""
    if storage.db.backend_type != "postgres":
        pytest.skip("SQLite stamps a row when its statement runs, not when its transaction began")
    wrapper = PrivacyEnforcingStorage(storage, PrivacyMode.NORMAL)
    began = None
    for _ in range(10):
        async with storage.db.transaction():
            opened = await storage.db.fetchval("SELECT NOW()")
            if opened.microsecond >= 999_000:
                continue  # too close to the next second to fit a later transition into it
            began = opened.astimezone(timezone.utc)
            _pin_transition(
                monkeypatch, began + timedelta(microseconds=(1_000_000 - began.microsecond) // 2)
            )
            wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
            await storage.add_conversation("user", "written during the stint")
        break
    assert began is not None, "no transaction began early enough in its second"

    stamp = await storage.db.fetchval(
        "SELECT created_at FROM conversation_history WHERE agent_id = ?", (storage.agent_id,)
    )
    assert stamp == began.replace(tzinfo=None), "the row was not stamped with its transaction's start"
    assert wrapper._entered_ephemeral_at > stamp.strftime("%Y-%m-%d %H:%M:%S.%f"), (
        "the hazard is not live: the row's stamp does not precede the watermark"
    )

    report = await wrapper.purge_ephemeral_session(reason="test-conversation-straddling-txn")

    assert not report.required_sweep_failed, report.failed_stores
    assert report["conversation_history"] == 1
    assert await _contents(storage) == set()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_an_undatable_watermark_is_refused_on_backend(storage):
    await _insert(storage, "row", TRANSITION)
    with pytest.raises(ValueError, match="datable watermark"):
        await storage.purge_conversations_since("not a timestamp", reason="test")
    assert await _contents(storage) == {"row"}
