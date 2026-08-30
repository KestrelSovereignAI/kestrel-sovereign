"""Self-scoped, read-only Hold introspection (#3166)."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign import server
from kestrel_sovereign.features.identity.feature import IdentityFeature
from kestrel_sovereign.hold import (
    HOST_HOLD_TARGET,
    EffectiveHoldState,
    HoldScope,
    HoldState,
    HoldStore,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage.async_database import AsyncDatabase


SUBJECT_DID = "did:test:self-hold-subject"


def _state(
    scope: HoldScope,
    *,
    target_id: str,
    actor_id: str,
    reason: str,
    receipt_id: str,
) -> HoldState:
    return HoldState(
        scope=scope,
        target_id=target_id,
        reason=reason,
        actor_id=actor_id,
        set_at="2026-08-30T12:00:00+00:00",
        hold_receipt_id=receipt_id,
        revision=7,
    )


def _feature(snapshot=None, *, failure: Exception | None = None):
    async def reader():
        if failure is not None:
            raise failure
        return snapshot

    agent = SimpleNamespace(
        did=SUBJECT_DID,
        _self_hold_subject_did=SUBJECT_DID,
        _self_hold_state_reader=reader,
    )
    return IdentityFeature(agent)


@pytest.mark.asyncio
async def test_hold_introspection_keeps_host_and_agent_evidence_separate_and_redacted():
    host_actor = "did:sovereign:private-operator"
    host_receipt = "hold-receipt-private-host"
    agent_receipt = "hold-receipt-private-agent"
    feature = _feature(
        EffectiveHoldState(
            host=_state(
                HoldScope.HOST,
                target_id=HOST_HOLD_TARGET,
                actor_id=host_actor,
                reason="Pause during provider maintenance",
                receipt_id=host_receipt,
            ),
            agent=_state(
                HoldScope.AGENT,
                target_id=SUBJECT_DID,
                actor_id=SUBJECT_DID,
                reason="Rest before continuing",
                receipt_id=agent_receipt,
            ),
        )
    )

    result = await feature.inspect_hold_state()

    assert result.status is ToolResultStatus.OK
    assert result.data["state"] == "held"
    assert result.data["sources"] == ["host", "agent"]
    assert result.data["latches"]["host"] == {
        "scope": "host",
        "reason": "Pause during provider maintenance",
        "actor_role": "sovereign",
        "set_at": "2026-08-30T12:00:00+00:00",
        "revision": 7,
    }
    assert result.data["latches"]["agent"]["actor_role"] == "self"
    assert result.data["redaction_policy"] == {
        "subject_identity": "implicit_self",
        "target_identity": "omitted",
        "receipt_identity": "omitted",
        "actor_identity": "role_only",
        "reason": "visible_to_held_subject",
        "timestamp": "visible_to_held_subject",
    }
    rendered = json.dumps(result.to_dict(), sort_keys=True)
    for private_value in (
        SUBJECT_DID,
        host_actor,
        host_receipt,
        agent_receipt,
    ):
        assert private_value not in rendered


@pytest.mark.asyncio
async def test_hold_introspection_reports_known_not_held_state():
    result = await _feature(EffectiveHoldState(host=None, agent=None)).inspect_hold_state()

    assert result.status is ToolResultStatus.OK
    assert result.data["state"] == "not_held"
    assert result.data["held"] is False
    assert result.data["latches"] == {"host": None, "agent": None}


@pytest.mark.asyncio
async def test_hold_introspection_unbound_store_is_unknown_not_not_held():
    feature = IdentityFeature(SimpleNamespace(did=SUBJECT_DID))

    result = await feature.inspect_hold_state()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["state"] == "unknown"
    assert result.data["held"] is None
    assert "not bound" in result.error


@pytest.mark.asyncio
async def test_hold_introspection_read_failure_is_honest_and_sanitized():
    result = await _feature(
        failure=RuntimeError("postgres password=super-secret control outage")
    ).inspect_hold_state()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["state"] == "unknown"
    assert result.data["failure"] == "read_failed"
    assert result.data["cause_type"] == "RuntimeError"
    assert "super-secret" not in json.dumps(result.to_dict())


@pytest.mark.asyncio
async def test_hold_introspection_rejects_foreign_agent_snapshot():
    foreign = EffectiveHoldState(
        host=None,
        agent=_state(
            HoldScope.AGENT,
            target_id="did:test:other-agent",
            actor_id="did:sovereign:operator",
            reason="foreign",
            receipt_id="foreign-receipt",
        ),
    )

    result = await _feature(foreign).inspect_hold_state()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["state"] == "unknown"
    assert result.data["cause_type"] == "ValueError"
    assert "did:test:other-agent" not in json.dumps(result.to_dict())


def test_hold_introspection_tool_has_no_spoofable_subject_parameter():
    assert list(inspect.signature(IdentityFeature.inspect_hold_state).parameters) == [
        "self"
    ]


@pytest.mark.asyncio
async def test_host_reader_closure_pins_subject_at_binding_time():
    observed = []

    class Store:
        async def get_effective(self, subject_did):
            observed.append(subject_did)
            return EffectiveHoldState(host=None, agent=None)

    agent = SimpleNamespace(did=SUBJECT_DID)
    agent.bind_self_hold_state_reader = (
        lambda **kwargs: KestrelAgent.bind_self_hold_state_reader(agent, **kwargs)
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            host_context=SimpleNamespace(hold_store=Store()),
            agent=agent,
            agent_manager=None,
        )
    )

    assert server._bind_agent_self_hold_reader(app, agent) is True
    agent.did = "did:test:payload-spoofed-other-agent"
    await agent._self_hold_state_reader()

    assert observed == [SUBJECT_DID]
    assert agent._self_hold_subject_did == SUBJECT_DID


@pytest.mark.asyncio
async def test_reopened_hold_store_is_visible_after_reader_rebind(tmp_path):
    path = tmp_path / "host-control.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    first = HoldStore(first_db)
    await first.ensure_schema()
    await first.set_hold(
        operation_id="hold-before-restart",
        actor_id="did:sovereign:operator",
        reason="survive restart",
        scope=HoldScope.AGENT,
        target_id=SUBJECT_DID,
    )
    await first_db.close()

    reopened_db = await AsyncDatabase.sqlite(str(path))
    reopened = HoldStore(reopened_db)
    await reopened.ensure_schema()
    agent = SimpleNamespace(did=SUBJECT_DID)
    agent.bind_self_hold_state_reader = (
        lambda **kwargs: KestrelAgent.bind_self_hold_state_reader(agent, **kwargs)
    )
    app = SimpleNamespace(
        state=SimpleNamespace(host_context=SimpleNamespace(hold_store=reopened))
    )
    server._bind_agent_self_hold_reader(app, agent)

    result = await IdentityFeature(agent).inspect_hold_state()

    assert result.status is ToolResultStatus.OK
    assert result.data["state"] == "held"
    assert result.data["sources"] == ["agent"]
    assert result.data["latches"]["agent"]["reason"] == "survive restart"
    await reopened_db.close()


def test_kestrel_agent_refuses_foreign_hold_reader_binding():
    agent = SimpleNamespace(did=SUBJECT_DID)

    with pytest.raises(ValueError, match="does not match"):
        KestrelAgent.bind_self_hold_state_reader(
            agent,
            subject_did="did:test:other-agent",
            reader=lambda: None,
        )
