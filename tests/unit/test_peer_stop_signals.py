"""Peer Stop rides the authenticated signal rails (#3169)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kestrel_sdk.signals import CausationFrame, SignalMode, Status, Trust

import kestrel_sovereign.signals.durable as durable_module
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.context import reset_current_signal, set_current_signal
from kestrel_sovereign.signals.durable_payload_policy import (
    AlwaysElidedActionSourceRegistration,
)
from kestrel_sovereign.signals.sources.peer_stop import (
    PEER_STOP_RATE_LIMIT_PER_HOUR,
    PEER_STOP_RATE_LIMIT_PER_MINUTE,
    SOURCE_NAME,
    build_peer_stop_registration,
    build_peer_stop_signal,
    decode_peer_stop_intent,
    encode_peer_stop_intent,
    peer_stop_source_event_id,
    signal_result_to_peer_stop_response,
)
from kestrel_sovereign.stop import (
    StopDisposition,
    StopOutcome,
    StopRequest,
    StopScope,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock


class _Agent:
    def __init__(self, did: str = "did:test:target") -> None:
        self.did = did
        self.agent_id = did
        self.background_tasks: list[asyncio.Task] = []
        self.cancel_current_request = MagicMock(return_value=False)
        self._privacy_transition_lock = ReentrantTransitionLock()

    def _get_privacy_transition_lock(self):
        return self._privacy_transition_lock

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


@pytest.fixture
async def peer_dispatcher(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "signals.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    agent = _Agent()
    registry = SourceRegistry()
    registry.register(build_peer_stop_registration(agent))
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=store,
    )
    yield SimpleNamespace(agent=agent, dispatcher=dispatcher, backend=backend)
    pending = [task for task in agent.background_tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await backend.close()


def test_peer_stop_registration_is_bounded_trusted_action() -> None:
    registration = build_peer_stop_registration(_Agent())

    assert registration.name == SOURCE_NAME
    assert registration.default_mode is SignalMode.ACTION
    assert isinstance(registration, AlwaysElidedActionSourceRegistration)
    assert registration.allowed_modes == frozenset({SignalMode.ACTION})
    assert registration.trust is Trust.TRUSTED
    assert registration.allow_self_loops is False
    assert registration.rate_limit.per_minute == PEER_STOP_RATE_LIMIT_PER_MINUTE
    assert registration.rate_limit.per_hour == PEER_STOP_RATE_LIMIT_PER_HOUR


@pytest.mark.asyncio
async def test_peer_stop_dispatch_reaches_handler_while_turn_holds_privacy_lock(
    peer_dispatcher,
) -> None:
    c = peer_dispatcher
    private_chain = [
        CausationFrame(
            agent_id="did:test:private-ancestor",
            source="private.webhook",
            signal_id="private-signal-id",
            turn_id="private-turn-id",
            depth=1,
            emitted_at=datetime.now(timezone.utc),
        )
    ]
    signal = build_peer_stop_signal(
        agent=c.agent,
        actor_id="did:test:peer",
        intent={
            "scope": "agent",
            "target": None,
            "reason": "stop wedged stream",
            "cascade": True,
            "correlation_id": "peer-stop-live-stream",
        },
        causation_chain=private_chain,
    )

    blocked = False
    result = None
    async with c.agent._privacy_transition_lock:
        dispatch = asyncio.create_task(
            c.dispatcher.dispatch_signal(
                signal,
                source_event_id=signal.dedupe_key,
            )
        )
        try:
            result = await asyncio.wait_for(asyncio.shield(dispatch), timeout=1.0)
        except TimeoutError:
            blocked = True

    if blocked:
        dispatch.cancel()
        await asyncio.gather(dispatch, return_exceptions=True)
    assert blocked is False, "peer Stop waited behind the turn-held privacy lock"
    assert result is not None
    assert result.status is Status.OK
    c.agent.cancel_current_request.assert_called_once()
    durable_row = await c.backend.fetch_one(
        "SELECT payload, causation_chain FROM durable_signal_events "
        "WHERE agent_id = ?",
        (c.agent.did,),
    )
    assert durable_row is not None
    assert json.loads(durable_row[0]) == {"_privacy_gated": "source_policy"}
    assert "stop wedged stream" not in durable_row[0]
    assert "did:test:peer" not in durable_row[0]
    assert json.loads(durable_row[1]) == []
    assert "private-signal-id" not in durable_row[1]
    assert "private-turn-id" not in durable_row[1]

    audit_row = await c.backend.fetch_one(
        "SELECT dedupe_key, payload_redacted FROM signal_log WHERE id = ?",
        (signal.id,),
    )
    assert audit_row is not None
    enumerable_material = json.dumps(
        ["did:test:peer", "peer-stop-live-stream"],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert audit_row[0] != hashlib.sha256(enumerable_material).hexdigest()
    assert "did:test:peer" not in audit_row[1]
    assert "peer-stop-live-stream" not in audit_row[1]


def test_peer_stop_source_event_id_is_keyed_and_not_roster_enumerable(
    monkeypatch,
) -> None:
    monkeypatch.setenv("KESTREL_DATA_KEY", "first-private-data-binding-key")
    first = peer_stop_source_event_id("did:test:peer", "peer-stop-keyed")
    monkeypatch.setenv("KESTREL_DATA_KEY", "second-private-data-binding-key")
    second = peer_stop_source_event_id("did:test:peer", "peer-stop-keyed")
    enumerable_material = json.dumps(
        ["did:test:peer", "peer-stop-keyed"],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert first != second
    assert first != hashlib.sha256(enumerable_material).hexdigest()


def test_peer_stop_source_event_id_survives_transport_key_rotation(
    monkeypatch,
) -> None:
    """Transport credential rotation cannot reopen the durable action lane."""

    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_KEY", "first-transport-key")
    first = peer_stop_source_event_id(
        "did:test:peer",
        "peer-stop-restart-stable",
    )
    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_KEY", "rotated-transport-key")
    second = peer_stop_source_event_id(
        "did:test:peer",
        "peer-stop-restart-stable",
    )

    assert first == second


def test_durable_peer_stop_refuses_a_missing_restart_stable_data_key(
    monkeypatch,
) -> None:
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    monkeypatch.delenv("KESTREL_DATA_KEY_FILE", raising=False)
    monkeypatch.setenv("KESTREL_DEPLOYMENT_PERSISTENCE", "durable_sovereign")
    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_KEY", "disposable-transport-key")

    with pytest.raises(
        RuntimeError,
        match="Durable peer Stop replay binding requires KESTREL_DATA_KEY",
    ):
        peer_stop_source_event_id(
            "did:test:peer",
            "peer-stop-missing-stable-key",
        )


@pytest.mark.asyncio
async def test_always_elided_peer_stop_dispatch_does_not_require_data_key(
    peer_dispatcher,
    monkeypatch,
) -> None:
    monkeypatch.delenv("KESTREL_DATA_KEY", raising=False)
    c = peer_dispatcher
    signal = build_peer_stop_signal(
        agent=c.agent,
        actor_id="did:test:keyless-peer",
        intent={
            "scope": "agent",
            "target": None,
            "reason": "keyless stop",
            "cascade": True,
            "correlation_id": "peer-stop-keyless",
        },
    )

    result = await c.dispatcher.dispatch_signal(
        signal,
        source_event_id=signal.dedupe_key,
    )

    assert result.status is Status.OK
    durable_row = await c.backend.fetch_one(
        "SELECT payload, caller_identity FROM durable_signal_events "
        "WHERE agent_id = ?",
        (c.agent.did,),
    )
    assert durable_row is not None
    assert json.loads(durable_row[0]) == {"_privacy_gated": "source_policy"}
    assert durable_row[1] == "v1:none"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "scope": "host",
            "target": None,
            "reason": "not a peer scope",
            "cascade": True,
            "correlation_id": "peer-stop-1",
        },
        {
            "scope": "tool_call",
            "target": "tool-call-1",
            "reason": "unsupported address",
            "cascade": True,
            "correlation_id": "peer-stop-1",
        },
        {
            "scope": "agent",
            "target": None,
            "reason": "forged actor",
            "cascade": True,
            "correlation_id": "peer-stop-1",
            "actor_id": "did:attacker:forged",
        },
        {
            "scope": "agent",
            "target": "did:attacker:retarget",
            "reason": "forged target",
            "cascade": True,
            "correlation_id": "peer-stop-1",
        },
    ],
)
def test_peer_stop_schema_rejects_host_and_payload_principals(payload) -> None:
    registration = build_peer_stop_registration(_Agent())

    with pytest.raises(ValueError):
        registration.schema(payload)


@pytest.mark.asyncio
async def test_peer_stop_rejects_non_utf8_reason_before_action(
    peer_dispatcher,
) -> None:
    c = peer_dispatcher
    await c.dispatcher.initialize_durable_delivery()
    signal = build_peer_stop_signal(
        agent=c.agent,
        actor_id="did:test:peer",
        intent={
            "scope": "agent",
            "target": None,
            "reason": "valid before boundary mutation",
            "cascade": True,
            "correlation_id": "peer-stop-invalid-unicode",
        },
    )
    signal.payload["reason"] = "\ud800"

    refused = await c.dispatcher.dispatch_signal(
        signal,
        source_event_id=signal.dedupe_key,
    )

    assert refused.status is Status.DROPPED_VALIDATION
    assert "valid UTF-8" in (refused.error or "")
    c.agent.cancel_current_request.assert_not_called()
    assert await c.backend.fetch_val(
        "SELECT COUNT(*) FROM durable_signal_events WHERE source = ?",
        (SOURCE_NAME,),
    ) == 0
    audit = await c.backend.fetch_one(
        "SELECT status, error FROM signal_log WHERE id = ?",
        (signal.id,),
    )
    assert audit is not None
    assert audit[0] == Status.DROPPED_VALIDATION.value
    assert "valid UTF-8" in audit[1]


def test_peer_stop_intent_round_trip_is_canonical_and_has_no_principals() -> None:
    message = encode_peer_stop_intent(
        scope=StopScope.TURN,
        target="turn-123",
        reason="unsafe loop",
        cascade=True,
        correlation_id="peer-stop-123",
    )

    assert decode_peer_stop_intent(message) == {
        "scope": "turn",
        "target": "turn-123",
        "reason": "unsafe loop",
        "cascade": True,
        "correlation_id": "peer-stop-123",
    }
    assert "actor_id" not in message
    assert "target_agent_id" not in message


@pytest.mark.asyncio
async def test_handler_uses_signal_principals_not_payload_metadata() -> None:
    agent = _Agent()
    captured: list[StopRequest] = []

    class _Authority:
        async def stop(self, request: StopRequest):
            captured.append(request)
            return (
                StopOutcome(
                    scope=request.scope,
                    requested_target=request.target,
                    resolved_target=agent.did,
                    agent_id=agent.did,
                    disposition=StopDisposition.ALREADY_COMPLETE,
                    correlation_id=request.correlation_id,
                ),
            )

    signal = build_peer_stop_signal(
        agent=agent,
        actor_id="did:test:authenticated-peer",
        intent={
            "scope": "agent",
            "target": None,
            "reason": "andon cord",
            "cascade": True,
            "correlation_id": "peer-stop-actor",
        },
    )
    registration = build_peer_stop_registration(agent)
    token = set_current_signal(signal)
    try:
        with patch(
            "kestrel_sovereign.signals.sources.peer_stop."
            "build_agent_cancellation_authority",
            return_value=_Authority(),
        ):
            result = await registration.handler(signal.payload)
    finally:
        reset_current_signal(token)

    assert StopOutcome.from_dict(result[0]).disposition is StopDisposition.ALREADY_COMPLETE
    assert captured == [
        StopRequest(
            scope=StopScope.AGENT,
            actor_id="did:test:authenticated-peer",
            target=agent.did,
            reason="andon cord",
            cascade=True,
            correlation_id="peer-stop-actor",
        )
    ]


@pytest.mark.asyncio
async def test_handler_rejects_missing_or_retargeted_signal_principal() -> None:
    agent = _Agent()
    registration = build_peer_stop_registration(agent)
    base = {
        "scope": "agent",
        "target": None,
        "reason": None,
        "cascade": True,
        "correlation_id": "peer-stop-principal",
    }
    for actor, target in (
        (None, agent.did),
        ("did:test:peer", "did:test:other-target"),
    ):
        signal = build_peer_stop_signal(
            agent=agent,
            actor_id="did:test:peer",
            intent=base,
        )
        signal.caller = actor
        signal.target_agent = target
        token = set_current_signal(signal)
        try:
            with pytest.raises(ValueError, match="principal|target"):
                await registration.handler(signal.payload)
        finally:
            reset_current_signal(token)


def test_peer_stop_signal_binds_trusted_envelope_and_idempotency_key() -> None:
    agent = _Agent()
    chain = [
        CausationFrame(
            agent_id="did:test:sender",
            source="heartbeat",
            signal_id="prior",
            turn_id="turn-prior",
            depth=1,
            emitted_at=datetime.now(timezone.utc),
        )
    ]
    intent = {
        "scope": "turn",
        "target": "turn-123",
        "reason": "unsafe loop",
        "cascade": False,
        "correlation_id": "peer-stop-signal",
    }
    signal = build_peer_stop_signal(
        agent=agent,
        actor_id="did:test:sender",
        intent=intent,
        causation_chain=chain,
    )

    assert signal.source == SOURCE_NAME
    assert signal.mode is SignalMode.ACTION
    assert signal.caller == "did:test:sender"
    assert signal.target_agent == agent.did
    assert signal.payload == intent
    assert signal.causation_chain == chain
    assert signal.dedupe_key == peer_stop_source_event_id(
        "did:test:sender", "peer-stop-signal"
    )


def test_peer_stop_signal_refuses_an_unidentified_target() -> None:
    with pytest.raises(ValueError, match="stable target identity"):
        build_peer_stop_signal(
            agent=SimpleNamespace(),
            actor_id="did:test:sender",
            intent={
                "scope": "agent",
                "target": None,
                "reason": "no target principal",
                "cascade": True,
                "correlation_id": "peer-stop-no-target",
            },
        )


@pytest.mark.asyncio
async def test_peer_stop_inherits_cycle_and_depth_guards(peer_dispatcher) -> None:
    c = peer_dispatcher
    intent = {
        "scope": "agent",
        "target": None,
        "reason": "cycle test",
        "cascade": True,
        "correlation_id": "peer-stop-cycle",
    }
    repeated = CausationFrame(
        agent_id=c.agent.did,
        source=SOURCE_NAME,
        signal_id="prior-peer-stop",
        turn_id=None,
        depth=1,
        emitted_at=datetime.now(timezone.utc),
    )
    cycle = await c.dispatcher.dispatch_signal(
        build_peer_stop_signal(
            agent=c.agent,
            actor_id="did:test:peer",
            intent=intent,
            causation_chain=[repeated],
        )
    )

    depth_chain = [
        CausationFrame(
            agent_id=f"did:test:hop-{depth}",
            source=f"source-{depth}",
            signal_id=f"signal-{depth}",
            turn_id=None,
            depth=depth,
            emitted_at=datetime.now(timezone.utc),
        )
        for depth in range(1, 6)
    ]
    too_deep = await c.dispatcher.dispatch_signal(
        build_peer_stop_signal(
            agent=c.agent,
            actor_id="did:test:peer",
            intent={**intent, "correlation_id": "peer-stop-depth"},
            causation_chain=depth_chain,
        )
    )

    assert cycle.status is Status.DROPPED_CYCLE
    assert too_deep.status is Status.DROPPED_CYCLE
    c.agent.cancel_current_request.assert_not_called()


@pytest.mark.asyncio
async def test_authenticated_peer_action_stops_active_local_work(peer_dispatcher) -> None:
    c = peer_dispatcher
    c.agent._active_request_ids = {"request-active"}
    c.agent.cancel_current_request.return_value = True
    c.agent.wait_for_request_completion = AsyncMock(
        return_value=RequestCompletionDisposition.COMPLETED
    )
    signal = build_peer_stop_signal(
        agent=c.agent,
        actor_id="did:test:authenticated-peer",
        intent={
            "scope": "agent",
            "target": None,
            "reason": "active unsafe work",
            "cascade": True,
            "correlation_id": "peer-stop-active",
        },
    )

    result = await c.dispatcher.dispatch_signal(
        signal,
        source_event_id=signal.dedupe_key,
    )

    assert result.status is Status.OK
    assert StopOutcome.from_dict(result.action_result[0]).disposition is (
        StopDisposition.STOPPED
    )
    c.agent.cancel_current_request.assert_called_once_with(
        request_id="request-active"
    )
    c.agent.wait_for_request_completion.assert_awaited_once_with(
        "request-active"
    )


@pytest.mark.asyncio
async def test_peer_stop_inherits_rate_limit_and_replay_does_not_consume_it(
    peer_dispatcher,
) -> None:
    c = peer_dispatcher

    async def dispatch(correlation_id: str):
        signal = build_peer_stop_signal(
            agent=c.agent,
            actor_id="did:test:peer",
            intent={
                "scope": "agent",
                "target": None,
                "reason": "rate test",
                "cascade": True,
                "correlation_id": correlation_id,
            },
        )
        return await c.dispatcher.dispatch_signal(
            signal,
            source_event_id=signal.dedupe_key,
        )

    first = await dispatch("peer-stop-replay")
    calls_after_first = c.agent.cancel_current_request.call_count
    replay = await dispatch("peer-stop-replay")
    calls_after_replay = c.agent.cancel_current_request.call_count
    fresh = [
        await dispatch(f"peer-stop-fresh-{index}")
        for index in range(PEER_STOP_RATE_LIMIT_PER_MINUTE)
    ]

    assert first.status is Status.OK
    assert replay.status is Status.COALESCED
    assert calls_after_first == calls_after_replay == 1
    assert [result.status for result in fresh[:-1]] == [
        Status.OK
    ] * (PEER_STOP_RATE_LIMIT_PER_MINUTE - 1)
    assert fresh[-1].status is Status.DROPPED_RATE_LIMIT


@pytest.mark.asyncio
async def test_peer_stop_rate_limit_survives_dispatcher_restart(tmp_path) -> None:
    backend = SQLiteBackend(str(tmp_path / "durable-peer-rate.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()

    def build_dispatcher():
        agent = _Agent()
        registry = SourceRegistry()
        registry.register(build_peer_stop_registration(agent))
        return agent, SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )

    async def dispatch(agent, dispatcher, correlation_id):
        signal = build_peer_stop_signal(
            agent=agent,
            actor_id="did:test:peer",
            intent={
                "scope": "agent",
                "target": None,
                "reason": "restart-stable rate test",
                "cascade": True,
                "correlation_id": correlation_id,
            },
        )
        return await dispatcher.dispatch_signal(
            signal,
            source_event_id=signal.dedupe_key,
        )

    first_agent, first = build_dispatcher()
    second = None
    try:
        for index in range(PEER_STOP_RATE_LIMIT_PER_MINUTE):
            result = await dispatch(first_agent, first, f"before-restart-{index}")
            assert result.status is Status.OK
        await first.shutdown_durable_delivery()

        second_agent, second = build_dispatcher()
        refused = await dispatch(second_agent, second, "after-restart")

        assert refused.status is Status.DROPPED_RATE_LIMIT
        second_agent.cancel_current_request.assert_not_called()
        admitted = await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_rate_admissions "
            "WHERE agent_id = ? AND source = ?",
            (second_agent.did, SOURCE_NAME),
        )
        assert admitted == PEER_STOP_RATE_LIMIT_PER_MINUTE
    finally:
        if second is not None:
            await second.shutdown_durable_delivery()
        elif not first._durable_shutdown:
            await first.shutdown_durable_delivery()
        await backend.close()


@pytest.mark.asyncio
async def test_peer_stop_replay_survives_transport_rotation_and_restart(
    tmp_path,
    monkeypatch,
) -> None:
    """A verified retry cannot cancel replacement work after a cold restart."""

    backend = SQLiteBackend(str(tmp_path / "durable-peer-replay.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()

    def build_dispatcher():
        agent = _Agent()
        registry = SourceRegistry()
        registry.register(build_peer_stop_registration(agent))
        return agent, SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )

    async def dispatch(agent, dispatcher):
        signal = build_peer_stop_signal(
            agent=agent,
            actor_id="did:test:peer",
            intent={
                "scope": "agent",
                "target": None,
                "reason": "restart-stable replay",
                "cascade": True,
                "correlation_id": "peer-stop-across-cold-restart",
            },
        )
        return await dispatcher.dispatch_signal(
            signal,
            source_event_id=signal.dedupe_key,
        )

    first_agent, first = build_dispatcher()
    second = None
    try:
        monkeypatch.setenv("KESTREL_A2A_TRANSPORT_KEY", "first-transport-key")
        initial = await dispatch(first_agent, first)
        await first.shutdown_durable_delivery()

        monkeypatch.setenv("KESTREL_A2A_TRANSPORT_KEY", "rotated-transport-key")
        second_agent, second = build_dispatcher()
        replay = await dispatch(second_agent, second)

        assert initial.status is Status.OK
        assert replay.status is Status.COALESCED
        assert first_agent.cancel_current_request.call_count == 1
        second_agent.cancel_current_request.assert_not_called()
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_events "
            "WHERE agent_id = ? AND source = ?",
            (second_agent.did, SOURCE_NAME),
        ) == 1
    finally:
        if second is not None:
            await second.shutdown_durable_delivery()
        elif not first._durable_shutdown:
            await first.shutdown_durable_delivery()
        await backend.close()


@pytest.mark.asyncio
async def test_peer_stop_refuses_two_live_runtime_inventories(tmp_path) -> None:
    backend = SQLiteBackend(str(tmp_path / "peer-stop-two-runtimes.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()

    def build_dispatcher():
        agent = _Agent()
        registry = SourceRegistry()
        registry.register(build_peer_stop_registration(agent))
        return agent, SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )

    first_agent, first = build_dispatcher()
    second_agent, second = build_dispatcher()
    try:
        # Production boot initializes every dispatcher before serving traffic.
        # Two owner rows therefore prove the unsupported split inventory before
        # either runtime can acknowledge a cancellation result.
        await first.initialize_durable_delivery()
        await second.initialize_durable_delivery()
        signal = build_peer_stop_signal(
            agent=first_agent,
            actor_id="did:test:peer",
            intent={
                "scope": "agent",
                "target": None,
                "reason": "must not guess the worker",
                "cascade": True,
                "correlation_id": "two-runtime-refusal",
            },
        )

        refused = await first.dispatch_signal(
            signal,
            source_event_id=signal.dedupe_key,
        )

        assert refused.status is Status.FAILED
        assert "exactly one live runtime owner" in (refused.error or "")
        first_agent.cancel_current_request.assert_not_called()
        second_agent.cancel_current_request.assert_not_called()
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_events WHERE source = ?",
            (SOURCE_NAME,),
        ) == 0
    finally:
        await first.shutdown_durable_delivery()
        await second.shutdown_durable_delivery()
        await backend.close()


@pytest.mark.asyncio
async def test_peer_stop_rechecks_runtime_owner_after_event_commit(
    tmp_path, monkeypatch
) -> None:
    """A runtime appearing in the admission/action gap aborts before cancel."""
    backend = SQLiteBackend(str(tmp_path / "peer-stop-owner-race.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()

    def build_dispatcher():
        agent = _Agent()
        registry = SourceRegistry()
        registry.register(build_peer_stop_registration(agent))
        return agent, SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )

    first_agent, first = build_dispatcher()
    second_agent, second = build_dispatcher()
    original_persist = first._durable_store.persist_signal

    async def persist_then_register_other_runtime(*args, **kwargs):
        persisted = await original_persist(*args, **kwargs)
        await second.initialize_durable_delivery()
        return persisted

    monkeypatch.setattr(
        first._durable_store,
        "persist_signal",
        persist_then_register_other_runtime,
    )
    try:
        signal = build_peer_stop_signal(
            agent=first_agent,
            actor_id="did:test:peer",
            intent={
                "scope": "agent",
                "target": None,
                "reason": "owner appeared after durable admission",
                "cascade": True,
                "correlation_id": "peer-stop-owner-race",
            },
        )

        refused = await first.dispatch_signal(
            signal,
            source_event_id=signal.dedupe_key,
        )

        assert refused.status is Status.FAILED
        assert "exactly one live runtime owner" in (refused.error or "")
        first_agent.cancel_current_request.assert_not_called()
        second_agent.cancel_current_request.assert_not_called()
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_events WHERE source = ?",
            (SOURCE_NAME,),
        ) == 1
    finally:
        await first.shutdown_durable_delivery()
        await second.shutdown_durable_delivery()
        await backend.close()


@pytest.mark.asyncio
async def test_file_backed_sqlite_runtime_owner_fence_uses_windows_native_lock(
    tmp_path, monkeypatch
) -> None:
    """The default SQLite boot lane must not import POSIX-only ``fcntl``."""

    backend = SQLiteBackend(str(tmp_path / "peer-stop-windows-fence.db"))
    await backend.connect()
    store = durable_module.DurableSignalStore(backend)
    await store.initialize()
    token = object()
    windows_try = MagicMock(return_value=(True, token))
    windows_unlock = MagicMock()
    monkeypatch.setattr(
        durable_module,
        "_runtime_owner_lock_platform_name",
        lambda: "nt",
    )
    monkeypatch.setattr(
        durable_module,
        "_try_lock_windows_runtime_owner_descriptor",
        windows_try,
    )
    monkeypatch.setattr(
        durable_module,
        "_unlock_windows_runtime_owner_descriptor",
        windows_unlock,
    )

    try:
        await store.register_runtime_owner(
            agent_id="did:test:windows",
            owner_id="dispatcher:windows",
        )
        await store.heartbeat_runtime_owner(
            agent_id="did:test:windows",
            owner_id="dispatcher:windows",
        )
    finally:
        await backend.close()

    assert windows_try.call_count == 2
    assert windows_unlock.call_count == 2
    for call in windows_unlock.call_args_list:
        assert call.args[1] is token


def test_sqlite_runtime_owner_lock_is_db_adjacent_and_tmpdir_independent(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "trusted" / "signals.db"
    db_path.parent.mkdir(mode=0o700)
    db_path.touch(mode=0o600)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "first-process-temp"))
    first = Path(
        durable_module._runtime_owner_lock_path(
            str(db_path), agent_id="did:test:stable-lock"
        )
    )
    monkeypatch.setenv("TMPDIR", str(tmp_path / "second-process-temp"))
    second = Path(
        durable_module._runtime_owner_lock_path(
            str(db_path), agent_id="did:test:stable-lock"
        )
    )

    assert first == second
    assert first.parent == db_path.resolve().parent
    assert first.name.startswith(".kestrel-runtime-owner-")


@pytest.mark.skipif(os.name != "posix", reason="POSIX secure-open contract")
def test_sqlite_runtime_owner_lock_refuses_precreated_symlink(tmp_path) -> None:
    db_path = tmp_path / "signals.db"
    db_path.touch(mode=0o600)
    lock_path = Path(
        durable_module._runtime_owner_lock_path(
            str(db_path), agent_id="did:test:unsafe-lock"
        )
    )
    target = tmp_path / "attacker-selected"
    target.touch(mode=0o600)
    lock_path.symlink_to(target)

    with pytest.raises(RuntimeError, match="runtime-owner lock"):
        durable_module._open_runtime_owner_lock_descriptor(
            str(db_path), agent_id="did:test:unsafe-lock"
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX secure-open contract")
def test_sqlite_runtime_owner_lock_refuses_precreated_hardlink(tmp_path) -> None:
    db_path = tmp_path / "signals.db"
    db_path.touch(mode=0o600)
    lock_path = Path(
        durable_module._runtime_owner_lock_path(
            str(db_path), agent_id="did:test:hardlink-lock"
        )
    )
    target = tmp_path / "attacker-selected"
    target.touch(mode=0o600)
    os.link(target, lock_path)

    with pytest.raises(RuntimeError, match="runtime-owner lock"):
        durable_module._open_runtime_owner_lock_descriptor(
            str(db_path), agent_id="did:test:hardlink-lock"
        )


@pytest.mark.asyncio
async def test_peer_stop_action_fence_blocks_runtime_registration(
    tmp_path,
) -> None:
    backend = SQLiteBackend(str(tmp_path / "peer-stop-owner-fence.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()

    def build_dispatcher():
        agent = _Agent()
        registry = SourceRegistry()
        registry.register(build_peer_stop_registration(agent))
        return agent, SignalDispatcher(
            agent=agent,
            registry=registry,
            lock_manager=OrderedLockManager(),
            store=store,
        )

    first_agent, first = build_dispatcher()
    _second_agent, second = build_dispatcher()
    cancellation_started = asyncio.Event()
    release_cancellation = asyncio.Event()

    async def wait_for_completion(_request_id):
        cancellation_started.set()
        await release_cancellation.wait()
        return RequestCompletionDisposition.COMPLETED

    first_agent._active_request_ids = {"request-active"}
    first_agent.cancel_current_request.return_value = True
    first_agent.wait_for_request_completion = wait_for_completion
    second_initialization = None
    stale_owner_heartbeat = None
    dispatch = None
    try:
        signal = build_peer_stop_signal(
            agent=first_agent,
            actor_id="did:test:peer",
            intent={
                "scope": "agent",
                "target": None,
                "reason": "hold owner fence through cancellation",
                "cascade": True,
                "correlation_id": "peer-stop-owner-fence",
            },
        )
        dispatch = asyncio.create_task(
            first.dispatch_signal(signal, source_event_id=signal.dedupe_key)
        )
        await asyncio.wait_for(cancellation_started.wait(), timeout=1.0)
        second_initialization = asyncio.create_task(
            second.initialize_durable_delivery()
        )
        stale_owner_heartbeat = asyncio.create_task(
            first._durable_store.heartbeat_runtime_owner(
                agent_id=first_agent.did,
                owner_id="dispatcher:previously-stale-runtime",
            )
        )
        await asyncio.sleep(0.05)

        assert not second_initialization.done()
        assert not stale_owner_heartbeat.done()
        # The registration mutex must not be SQLite's global writer lock: the
        # cancelled request may need this same database to finish cleanup.
        await asyncio.wait_for(
            backend.execute(
                "UPDATE durable_signal_events SET source = source "
                "WHERE event_id = ?",
                (signal.id,),
            ),
            timeout=0.5,
        )
        release_cancellation.set()
        assert (await asyncio.wait_for(dispatch, timeout=1.0)).status is Status.OK
        await asyncio.wait_for(second_initialization, timeout=1.0)
        await asyncio.wait_for(stale_owner_heartbeat, timeout=1.0)
    finally:
        release_cancellation.set()
        for task in (dispatch, second_initialization, stale_owner_heartbeat):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await first.shutdown_durable_delivery()
        await second.shutdown_durable_delivery()
        await backend.close()


@pytest.mark.asyncio
async def test_replay_of_unexecuted_rate_limited_stop_never_claims_completion(
    peer_dispatcher,
) -> None:
    c = peer_dispatcher

    async def dispatch(correlation_id: str):
        intent = {
            "scope": "agent",
            "target": None,
            "reason": "rate test",
            "cascade": True,
            "correlation_id": correlation_id,
        }
        signal = build_peer_stop_signal(
            agent=c.agent,
            actor_id="did:test:peer",
            intent=intent,
        )
        result = await c.dispatcher.dispatch_signal(
            signal,
            source_event_id=signal.dedupe_key,
        )
        return result, signal_result_to_peer_stop_response(
            result,
            target_agent_id=c.agent.did,
            intent=intent,
        )

    for index in range(PEER_STOP_RATE_LIMIT_PER_MINUTE):
        result, _response = await dispatch(f"peer-stop-prime-{index}")
        assert result.status is Status.OK
    first, first_response = await dispatch("peer-stop-never-executed")
    retry, retry_response = await dispatch("peer-stop-never-executed")

    assert first.status is Status.DROPPED_RATE_LIMIT
    assert retry.status is Status.COALESCED
    assert StopOutcome.from_dict(
        first_response["stop_outcomes"][0]
    ).disposition is StopDisposition.REFUSED
    retry_outcome = StopOutcome.from_dict(retry_response["stop_outcomes"][0])
    assert retry_outcome.disposition is StopDisposition.UNREACHABLE
    assert "original outcome is unavailable" in retry_outcome.detail


@pytest.mark.parametrize(
    ("status", "disposition"),
    [
        (Status.DROPPED_CYCLE, StopDisposition.REFUSED),
        (Status.DROPPED_RATE_LIMIT, StopDisposition.REFUSED),
        (Status.DROPPED_VALIDATION, StopDisposition.REFUSED),
        (Status.COALESCED, StopDisposition.UNREACHABLE),
        (Status.FAILED, StopDisposition.UNREACHABLE),
    ],
)
def test_dispatch_drops_map_to_truthful_peer_stop_outcomes(status, disposition) -> None:
    response = signal_result_to_peer_stop_response(
        SimpleNamespace(
            signal_id="signal-result",
            status=status,
            mode=SignalMode.ACTION,
            action_result=None,
            error="guard detail",
        ),
        target_agent_id="did:test:target",
        intent={
            "scope": "agent",
            "target": None,
            "reason": "map result",
            "cascade": True,
            "correlation_id": "peer-stop-result",
        },
    )

    assert response["signal_receipt"]["signal_id"] == "signal-result"
    assert response["signal_receipt"]["status"] == status.value
    assert "guard detail" not in str(response)
    outcome = StopOutcome.from_dict(response["stop_outcomes"][0])
    assert outcome.disposition is disposition
    assert outcome.detail == response["signal_receipt"]["detail"]


def test_dispatch_failure_does_not_expose_internal_error_to_peer() -> None:
    internal_error = (
        "OperationalError: unable to open database file "
        "/secret/tenant/signals.db"
    )

    response = signal_result_to_peer_stop_response(
        SimpleNamespace(
            signal_id="signal-failed",
            status=Status.FAILED,
            mode=SignalMode.ACTION,
            action_result=None,
            error=internal_error,
        ),
        target_agent_id="did:test:target",
        intent={
            "scope": "agent",
            "target": None,
            "reason": "map failure",
            "cascade": True,
            "correlation_id": "peer-stop-failed",
        },
    )

    assert response["signal_receipt"]["status"] == "failed"
    assert response["signal_receipt"]["detail"] == (
        "Peer Stop signal failed before cancellation could be confirmed"
    )
    outcome = StopOutcome.from_dict(response["stop_outcomes"][0])
    assert outcome.disposition is StopDisposition.UNREACHABLE
    assert outcome.detail == response["signal_receipt"]["detail"]
    assert internal_error not in str(response)


def test_successful_turn_stop_redacts_private_resolved_request_address() -> None:
    outcome = StopOutcome(
        scope=StopScope.TURN,
        requested_target="turn-visible",
        resolved_target="request-private",
        agent_id="did:test:target",
        disposition=StopDisposition.STOPPED,
        correlation_id="peer-stop-turn",
    )

    response = signal_result_to_peer_stop_response(
        SimpleNamespace(
            signal_id="signal-turn",
            status=Status.OK,
            mode=SignalMode.ACTION,
            action_result=[outcome.to_dict()],
            error=None,
        ),
        target_agent_id="did:test:target",
        intent={
            "scope": "turn",
            "target": "turn-visible",
            "reason": "stop turn",
            "cascade": True,
            "correlation_id": "peer-stop-turn",
        },
    )

    public_outcome = StopOutcome.from_dict(response["stop_outcomes"][0])
    assert public_outcome.requested_target == "turn-visible"
    assert public_outcome.resolved_target == "did:test:target"
    assert "request-private" not in str(response)
