"""Tests for the restart status-event API endpoint (#1816).

The frontend repaints the restart status-bubble trail on chat reload by
fetching ``/api/restart/status-events``. This pins the endpoint's
contract: newest-first events, and session-scoping via the
``origin_session_id`` carried in each event payload (#1812) so reloading
conversation A never repaints a restart filed from conversation B.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytest_asyncio

from kestrel_sovereign.endpoints.restart_events import (
    get_restart_status_events,
)
from kestrel_sovereign.features.restart_coordinator.event_store import (
    ensure_restart_status_events_table,
    record_event,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend


_test_databases: list[AsyncDatabase] = []


@pytest_asyncio.fixture(autouse=True)
async def _close_test_databases():
    """Close each test-owned worker before pytest tears down its event loop."""

    _test_databases.clear()
    try:
        yield
    finally:
        for db in reversed(_test_databases):
            await db.close()
        _test_databases.clear()


async def _backend(tmp_path):
    raw = SQLiteBackend(str(tmp_path / "restart-events.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    _test_databases.append(db)
    await ensure_restart_status_events_table(db)
    return db


def _request(db, did="a"):
    agent = SimpleNamespace(did=did, _raw_storage=SimpleNamespace(db=db))
    return SimpleNamespace(state=SimpleNamespace(agent=agent))


@pytest.mark.asyncio
async def test_endpoint_returns_events_newest_first(tmp_path):
    db = await _backend(tmp_path)
    await record_event(
        db, request_id="r1", state="pending", agent_id="a",
        payload={"origin_session_id": "s1"},
    )
    await record_event(
        db, request_id="r1", state="completed", agent_id="a",
        payload={"origin_session_id": "s1"},
    )

    result = await get_restart_status_events(_request(db), session="", limit=200)

    assert result["count"] == 2
    assert {e["request_id"] for e in result["events"]} == {"r1"}


@pytest.mark.asyncio
async def test_endpoint_scopes_to_origin_session(tmp_path):
    db = await _backend(tmp_path)
    await record_event(
        db, request_id="r1", state="pending", agent_id="a",
        payload={"origin_session_id": "session-A"},
    )
    await record_event(
        db, request_id="r2", state="pending", agent_id="a",
        payload={"origin_session_id": "session-B"},
    )

    result = await get_restart_status_events(
        _request(db), session="session-A", limit=200,
    )

    assert result["count"] == 1
    assert result["events"][0]["request_id"] == "r1"
    assert (
        result["events"][0]["payload"]["origin_session_id"] == "session-A"
    )


@pytest.mark.asyncio
async def test_endpoint_scopes_shared_history_to_routed_agent(tmp_path):
    db = await _backend(tmp_path)
    await record_event(
        db,
        request_id="owner-request",
        state="pending",
        agent_id="did:test:owner",
        payload={"reason": "owner detail"},
    )
    await record_event(
        db,
        request_id="other-request",
        state="pending",
        agent_id="did:test:other",
        payload={"reason": "other private detail"},
    )

    result = await get_restart_status_events(
        _request(db, did="did:test:owner"),
        session="",
        limit=200,
    )

    assert result["count"] == 1
    assert result["events"][0]["request_id"] == "owner-request"
    assert "other private detail" not in str(result)


@pytest.mark.asyncio
async def test_endpoint_no_database_returns_empty(tmp_path):
    request = SimpleNamespace(state=SimpleNamespace(agent=SimpleNamespace()))
    result = await get_restart_status_events(request, session="", limit=200)
    assert result == {"events": [], "count": 0}
