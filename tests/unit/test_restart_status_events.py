"""Tests for the typed restart_status event store (#1562).

Pins the dedupe-signature contract, the persistence-on-emit path,
the chat-history listing, and the agent-context (preturn_state)
rendering. The companion frontend dedupe + stream-boundary work
lives under #1560 and consumes the ``dedupe_signature`` field these
events expose.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.restart_coordinator import (
    RestartCoordinatorFeature,
)
from kestrel_sovereign.features.restart_coordinator.event_store import (
    dedupe_signature,
    ensure_restart_status_events_table,
    latest_event_for_signature,
    list_events_for_request,
    list_recent_events_for_agent_context,
    list_recent_events_for_history,
    record_event,
)
from kestrel_sovereign.features.restart_coordinator.events import (
    build_restart_status_event,
)
from kestrel_sovereign.features.restart_coordinator.store import (
    ensure_restart_requests_table,
    insert_request,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _backend(tmp_path):
    raw = SQLiteBackend(str(tmp_path / "restart-events.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    await ensure_restart_requests_table(db)
    await ensure_restart_status_events_table(db)
    return db


def _make_agent(backend, did="did:test:agent", emit=None):
    raw_storage = SimpleNamespace(db=backend)
    if emit is None:
        captured = []

        async def emit(event_type, data):
            captured.append((event_type, data))

        emit.captured = captured  # type: ignore[attr-defined]
    return SimpleNamespace(
        did=did,
        agent_id=did,
        _raw_storage=raw_storage,
        storage=None,
        dispatcher=None,
        signal_registry=None,
        _active_request_ids=set(),
        _background_tasks=set(),
        emit_event=emit,
        features={"RestartCoordinatorFeature": True},
    )


# ---------------------------------------------------------------------------
# Dedupe signature contract
# ---------------------------------------------------------------------------


def test_dedupe_signature_is_request_id_colon_state():
    """The signature is the stable frontend dedupe key. Format must
    remain ``{request_id}:{state}`` so existing renderers don't
    re-parse on a schema change."""
    assert dedupe_signature("abc123", "pending") == "abc123:pending"
    assert dedupe_signature("abc123", "deferred") == "abc123:deferred"
    # Different state → different signature (the lifecycle MUST surface).
    assert (
        dedupe_signature("abc123", "pending")
        != dedupe_signature("abc123", "deferred")
    )
    # Different request → different signature.
    assert (
        dedupe_signature("abc123", "pending")
        != dedupe_signature("def456", "pending")
    )


def test_event_payload_includes_dedupe_signature():
    """``build_restart_status_event`` must surface the signature so
    the frontend can find-or-update by it (#1560 / #1562)."""
    req = SimpleNamespace(
        id="abc123",
        requested_by_agent="did:test:agent",
        operation="restart_only",
        urgency="normal",
        policy="idle_agents_only",
        reason="config landed",
        update_target_ref="",
        update_profile="",
        completed_at=None,
    )
    payload = build_restart_status_event(req, state="pending")
    assert payload["dedupe_signature"] == "abc123:pending"
    assert payload["status"] == "pending"
    assert payload["request_id"] == "abc123"


def test_event_payload_signature_excludes_volatile_deferral_reason():
    """The signature MUST stay stable across coordinator polls that
    only update the volatile ``oldest 63s of 900s stale window``
    age substring in ``deferral_reason`` — that volatility is what
    spawned duplicate bubbles on issue 7f9ee2dab18b (#1560).
    """
    req = SimpleNamespace(
        id="r1", requested_by_agent="a", operation="restart_only",
        urgency="normal", policy="idle_agents_only", reason="r",
        update_target_ref="", update_profile="", completed_at=None,
    )
    a = build_restart_status_event(
        req, state="pending",
        deferral_reason="agent busy (1 active request id(s); oldest 43s of 900s stale window)",
    )
    b = build_restart_status_event(
        req, state="pending",
        deferral_reason="agent busy (1 active request id(s); oldest 87s of 900s stale window)",
    )
    assert a["dedupe_signature"] == b["dedupe_signature"]


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_event_persists_typed_row(tmp_path):
    db = await _backend(tmp_path)
    payload = {"foo": "bar", "deferral_reason": "ignored by signature"}
    event = await record_event(
        db,
        request_id="r1",
        state="pending",
        agent_id="did:test:agent",
        payload=payload,
    )
    assert event.dedupe_signature == "r1:pending"
    rows = await list_events_for_request(db, "r1")
    assert len(rows) == 1
    public = rows[0].to_public_dict()
    assert public["dedupe_signature"] == "r1:pending"
    assert public["payload"] == payload


@pytest.mark.asyncio
async def test_record_event_audit_trail_keeps_all_polls(tmp_path):
    """Same (request_id, state) polled twice produces two rows in
    the audit trail (durable). Frontend collapses them into one
    bubble via the shared ``dedupe_signature`` — durability and
    UI dedupe are separate concerns.
    """
    db = await _backend(tmp_path)
    for _ in range(3):
        await record_event(
            db, request_id="r1", state="pending",
            agent_id="a", payload={"k": "v"},
        )
    rows = await list_events_for_request(db, "r1")
    assert len(rows) == 3
    # All three share the same dedupe_signature.
    assert {r.dedupe_signature for r in rows} == {"r1:pending"}


@pytest.mark.asyncio
async def test_list_recent_events_for_history_returns_newest_first(tmp_path):
    db = await _backend(tmp_path)
    for state in ("pending", "deferred", "executing", "completed"):
        await record_event(
            db, request_id="r1", state=state,
            agent_id="a", payload={"state": state},
        )
    rows = await list_recent_events_for_history(db, limit=10)
    assert [r.state for r in rows] == [
        "completed", "executing", "deferred", "pending",
    ]


@pytest.mark.asyncio
async def test_list_recent_events_since_paging(tmp_path):
    db = await _backend(tmp_path)
    first = await record_event(
        db, request_id="r1", state="pending", agent_id="a",
        payload={},
    )
    second = await record_event(
        db, request_id="r1", state="deferred", agent_id="a",
        payload={},
    )
    # Page newer than ``first.created_at`` should only return ``second``.
    rows = await list_recent_events_for_history(
        db, limit=10, since=first.created_at,
    )
    assert len(rows) == 1
    assert rows[0].id == second.id


@pytest.mark.asyncio
async def test_list_recent_events_for_agent_context_scopes_by_agent(tmp_path):
    db = await _backend(tmp_path)
    await record_event(
        db, request_id="r1", state="pending", agent_id="emma",
        payload={},
    )
    await record_event(
        db, request_id="r2", state="completed", agent_id="meridian",
        payload={},
    )
    rows = await list_recent_events_for_agent_context(
        db, agent_id="emma", limit=5,
    )
    assert len(rows) == 1
    assert rows[0].request_id == "r1"


@pytest.mark.asyncio
async def test_latest_event_for_signature(tmp_path):
    db = await _backend(tmp_path)
    a = await record_event(
        db, request_id="r1", state="pending", agent_id="a",
        payload={"i": 1},
    )
    b = await record_event(
        db, request_id="r1", state="pending", agent_id="a",
        payload={"i": 2},
    )
    latest = await latest_event_for_signature(db, "r1:pending")
    assert latest is not None
    assert latest.id == b.id


# ---------------------------------------------------------------------------
# Persist-on-emit through the feature surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_persists_and_sse_emits(tmp_path):
    """A real ``_emit_status_event`` call must (1) persist a row in
    ``restart_status_events`` with the correct dedupe_signature AND
    (2) push the SSE payload through ``agent.emit_event``. Persistence
    runs first so a failing SSE listener can't lose the audit row.
    """
    db = await _backend(tmp_path)
    agent = _make_agent(db, did="did:test:agent")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    req = await insert_request(
        db, requested_by_agent="did:test:agent", reason="ship",
    )
    await feat._emit_status_event(
        req, state="pending", deferral_reason="agent busy (1 active id)",
    )
    rows = await list_events_for_request(db, req.id)
    assert len(rows) == 1
    assert rows[0].dedupe_signature == f"{req.id}:pending"
    assert rows[0].state == "pending"
    # SSE side-channel fired with the same payload + signature.
    captured = agent.emit_event.captured  # type: ignore[attr-defined]
    assert len(captured) == 1
    event_type, payload = captured[0]
    assert event_type == "restart_status"
    assert payload["dedupe_signature"] == f"{req.id}:pending"


@pytest.mark.asyncio
async def test_emit_persists_even_when_sse_listener_raises(tmp_path):
    """Even if the SSE listener raises, the typed-event row is the
    audit primary and must already be persisted before the emit is
    attempted. Reload after a listener crash must still surface the
    full lifecycle.
    """
    db = await _backend(tmp_path)

    async def boom(event_type, data):
        raise RuntimeError("SSE pipe broken")

    agent = _make_agent(db, did="did:test:agent", emit=boom)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    req = await insert_request(
        db, requested_by_agent="did:test:agent", reason="r",
    )
    # The defensive try/except in _emit_status_event swallows the
    # listener exception; the durable row must still be present.
    await feat._emit_status_event(req, state="executing")
    rows = await list_events_for_request(db, req.id)
    assert len(rows) == 1
    assert rows[0].state == "executing"


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_restart_status_events_tool(tmp_path):
    db = await _backend(tmp_path)
    agent = _make_agent(db, did="did:test:agent")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    req = await insert_request(
        db, requested_by_agent="did:test:agent", reason="r",
    )
    await feat._emit_status_event(req, state="pending")
    await feat._emit_status_event(req, state="executing")
    res = await feat.list_restart_status_events()
    assert res.status is ToolResultStatus.OK
    events = res.data["events"]
    # Newest first; each event carries dedupe_signature.
    assert events[0]["state"] == "executing"
    assert events[1]["state"] == "pending"
    assert events[0]["dedupe_signature"] == f"{req.id}:executing"


@pytest.mark.asyncio
async def test_list_restart_status_events_tool_clamps_limit(tmp_path):
    """``limit`` must clamp to [1, 1000] so a hostile caller can't
    dump the entire table via a 10⁹ request and a negative number
    can't produce an empty page silently."""
    db = await _backend(tmp_path)
    agent = _make_agent(db, did="did:test:agent")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    # Negative → clamped to 1.
    res = await feat.list_restart_status_events(limit=-5)
    assert res.data["count"] == 0  # empty table; clamp didn't raise
    # Garbage → defaults to 100, no crash.
    res = await feat.list_restart_status_events(limit="not-an-int")
    assert res.status is ToolResultStatus.OK


# ---------------------------------------------------------------------------
# Preturn state agent-context section
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preturn_restart_status_section(tmp_path):
    """The pre-turn state block must surface a one-line summary of
    recent restart lifecycle events for the agent — non-instructional
    operational context, not a directive (#1562 acceptance criterion).
    """
    from kestrel_sovereign.agent.preturn_state import _restart_status_section

    db = await _backend(tmp_path)
    agent = _make_agent(db, did="did:test:agent")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {"RestartCoordinatorFeature": feat}

    # No events yet → section yields nothing.
    line = await _restart_status_section(_AgentWithGet(agent))
    assert line is None

    req = await insert_request(
        db, requested_by_agent="did:test:agent", reason="r",
    )
    await feat._emit_status_event(req, state="pending")
    await feat._emit_status_event(req, state="completed")

    line = await _restart_status_section(_AgentWithGet(agent))
    assert line is not None
    # Counts each state, names the latest transition with its id prefix.
    assert "1 pending" in line
    assert "1 completed" in line
    assert "→ completed" in line


@pytest.mark.asyncio
async def test_emit_skips_sse_when_persistence_fails(tmp_path):
    """Codex round 1 P2: if the audit row fails to persist, the SSE
    emit must be skipped — otherwise a live UI bubble appears that
    has no durable backing and vanishes on reload.
    """
    db = await _backend(tmp_path)
    sse_captured = []

    async def emit(event_type, data):
        sse_captured.append((event_type, data))

    agent = _make_agent(db, did="did:test:agent", emit=emit)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    req = await insert_request(
        db, requested_by_agent="did:test:agent", reason="r",
    )

    # Sabotage the persist path: monkey-patch the feature's db to
    # raise on execute. The emit_event side-channel must NOT fire.
    class _Boom:
        async def execute(self, *a, **kw):
            raise RuntimeError("disk full")

        async def fetchall(self, *a, **kw):
            return []

    feat._db = _Boom()
    await feat._emit_status_event(req, state="pending")
    assert sse_captured == [], (
        "SSE must not fire when audit-row persistence fails"
    )


@pytest.mark.asyncio
async def test_emit_with_no_db_still_sse_emits(tmp_path):
    """When there's no DB configured at all (headless host, test
    stub), the SSE side-channel still fires — there's no audit
    promise to break.
    """
    db = await _backend(tmp_path)
    sse_captured = []

    async def emit(event_type, data):
        sse_captured.append((event_type, data))

    agent = _make_agent(db, did="did:test:agent", emit=emit)
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    req = await insert_request(
        db, requested_by_agent="did:test:agent", reason="r",
    )

    # Drop the DB entirely.
    feat._db = None
    await feat._emit_status_event(req, state="pending")
    assert len(sse_captured) == 1


@pytest.mark.asyncio
async def test_preturn_restart_status_section_filters_by_agent_did(tmp_path):
    """Codex round 1 P2: in a shared multi-agent DB, the preturn
    state block must surface ONLY this agent's restart lifecycle
    — not events filed by peer agents.
    """
    from kestrel_sovereign.agent.preturn_state import _restart_status_section

    db = await _backend(tmp_path)
    agent = _make_agent(db, did="did:test:emma")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {"RestartCoordinatorFeature": feat}

    # Seed events for emma + meridian.
    await record_event(
        db, request_id="emma-r1", state="pending",
        agent_id="did:test:emma", payload={},
    )
    await record_event(
        db, request_id="meridian-r1", state="executing",
        agent_id="did:test:meridian", payload={},
    )

    line = await _restart_status_section(_AgentWithGet(agent))
    assert line is not None
    # Emma only sees her own row.
    assert "1 pending" in line
    assert "1 executing" not in line
    assert "emma-r1"[:8] in line


class _AgentWithGet:
    """Wrap a SimpleNamespace agent in ``get_feature`` so the
    pre-turn state module can resolve the feature by name."""

    def __init__(self, agent):
        self._agent = agent

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def get_feature(self, name):
        return self._agent.features.get(name)


# ---------------------------------------------------------------------------
# #1571 - always-on operational state block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_operational_block_surfaces_restart_events_without_config(
    tmp_path,
):
    """The operational state block must surface restart lifecycle
    events even when the proactive ``[preturn_state]`` feature is
    not configured (#1571). The block does not consult config at
    all, so no patch is needed — it is always-on.
    """
    from kestrel_sovereign.agent.preturn_state import (
        build_operational_state_block,
    )

    db = await _backend(tmp_path)
    agent = _make_agent(db, did="did:test:emma")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {"RestartCoordinatorFeature": feat}
    wrapped = _AgentWithGet(agent)

    # No events -> block silent.
    assert await build_operational_state_block(wrapped) is None

    # Seed a pending event; block must render with operational header.
    req = await insert_request(
        db, requested_by_agent="did:test:emma", reason="r",
    )
    await feat._emit_status_event(req, state="pending")

    block = await build_operational_state_block(wrapped)
    assert block is not None
    assert "OPERATIONAL STATE" in block
    assert "1 pending" in block
    assert "END OPERATIONAL STATE" in block


@pytest.mark.asyncio
async def test_operational_block_did_scoped(tmp_path):
    """Peer agents' restart events must not appear in this agent's
    operational block (#1571).
    """
    from kestrel_sovereign.agent.preturn_state import (
        build_operational_state_block,
    )

    db = await _backend(tmp_path)
    agent = _make_agent(db, did="did:test:emma")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {"RestartCoordinatorFeature": feat}
    wrapped = _AgentWithGet(agent)

    await record_event(
        db, request_id="emma-r1", state="pending",
        agent_id="did:test:emma", payload={},
    )
    await record_event(
        db, request_id="meridian-r1", state="executing",
        agent_id="did:test:meridian", payload={},
    )

    block = await build_operational_state_block(wrapped)
    assert block is not None
    assert "1 pending" in block
    assert "executing" not in block


@pytest.mark.asyncio
async def test_operational_block_silent_without_feature(tmp_path):
    """Agents without the RestartCoordinatorFeature must not see an
    empty operational block — silent return, no header noise.
    """
    from kestrel_sovereign.agent.preturn_state import (
        build_operational_state_block,
    )

    agent = SimpleNamespace(
        did="did:test:agent",
        features={},
        get_feature=lambda name: None,
    )
    assert await build_operational_state_block(agent) is None


@pytest.mark.asyncio
async def test_preturn_block_does_not_duplicate_restart_section(
    tmp_path, monkeypatch,
):
    """When ``[preturn_state]`` IS enabled, the opt-in state block
    must no longer carry the restart-status line — it lives on the
    always-on operational path now (#1571). Otherwise the agent
    would see the same lifecycle summary twice.
    """
    import kestrel_sovereign.config as cfg
    from kestrel_sovereign.agent.preturn_state import (
        build_operational_state_block,
        build_preturn_state_block,
    )

    # Patch the real source-of-truth: build_preturn_state_block does
    # ``from kestrel_sovereign.config import load_section`` inside
    # the function, so the local-module name in preturn_state is
    # never consulted.
    monkeypatch.setattr(
        cfg, "load_section",
        lambda s: {"enabled": True, "agents": ["Emma"], "max_tokens": 500}
        if s == "preturn_state" else {},
    )

    db = await _backend(tmp_path)
    agent = _make_agent(db, did="did:test:emma")
    feat = RestartCoordinatorFeature(agent)
    await feat.initialize()
    agent.features = {"RestartCoordinatorFeature": feat}
    agent._agent_name = "Emma"
    agent.storage_path = None
    wrapped = _AgentWithGet(agent)

    await record_event(
        db, request_id="emma-r1", state="pending",
        agent_id="did:test:emma", payload={},
    )

    op_block = await build_operational_state_block(wrapped)
    state_block = await build_preturn_state_block(wrapped)

    assert op_block is not None
    assert "1 pending" in op_block
    if state_block is not None:
        # The opt-in block may render other sections, but restart
        # must not duplicate here.
        assert "Restart events" not in state_block
