"""#3484 — the host contract Workflows' durable ``await_signal`` gate requires.

``WorkflowRunner.start_run`` refuses an await stage unless the agent exposes
two hooks:

* ``workflow_await_signal_privacy_transition_lock`` — held around "privacy
  check, then matcher/CAS"; no privacy transition may complete while it is
  held. It is the durable persistence gate held SHARED, never the
  privacy-transition lock every turn holds (that would be the #3316 stall);
* ``workflow_await_signal_event_trust_verifier(delivery)`` — returns the
  immutable ingress trust receipt the dispatcher committed with the event, or
  ``None``.

These tests drive the real ``KestrelAgent`` hooks and lock acquisitions
against a real ``SignalDispatcher`` over SQLite. The Workflows tests at the
bottom run only where ``kestrel_feature_workflows`` is installed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
    Trust,
)

from kestrel_sovereign.features.privacy import PrivacyAgent
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.signals import (
    MAX_INGRESS_TRUST_RECEIPT_BYTES,
    DurableConsumerRegistration,
    DurableIngressAttestation,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.durable import build_ingress_trust_receipt
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.db.interface import TransactionError

# Bounds a deadlock, not the work: a passing run returns as soon as the awaited
# event fires. Kept well inside CI's 60s per-test timeout (see #3454).
_HANG_GUARD_SECONDS = 30

_SANITIZED = "inbound.sanitized"  # UNTRUSTED ARTIFACT: the sanitizer runs
_TRUSTED = "inbound.trusted"  # TRUSTED ARTIFACT with a sanitizer it skips
_ACTION = "inbound.action"  # ACTION: sanitizers never run

_RECEIPTS = "durable_signal_ingress_receipts"


def _redaction() -> RedactionPolicy:
    return RedactionPolicy(summarize=lambda payload: "<redacted>")


class _Rig(SimpleNamespace):
    agent: KestrelAgent
    dispatcher: SignalDispatcher
    backend: SQLiteBackend
    sanitizer_calls: list


@pytest.fixture
async def rig(tmp_path):
    agent = KestrelAgent(did="did:test:3484", storage_path=":memory:")
    # Everything `process_input` touches before its lock span is stubbed, so
    # a turn exercises the real CONVERSATION -> privacy acquisition. The
    # storage and context stubs also let a real privacy transition apply.
    agent.storage = SimpleNamespace(set_privacy_mode=lambda mode: None)
    agent.context_manager = object()
    agent.bootstrap_service = None
    agent._safe_mode = False
    agent._maybe_audit = AsyncMock()
    agent._maybe_refresh_user_byok_resolver = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent.prepare_and_refresh_all_feature_context_clauses = AsyncMock()
    agent.privacy_agent = PrivacyAgent(storage=None, initial_mode=PrivacyMode.NORMAL)

    backend = SQLiteBackend(str(tmp_path / "await-contract.db"))
    await backend.connect()
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    sanitizer_calls: list = []

    def sanitize(payload):
        sanitizer_calls.append(dict(payload))
        return {key: value for key, value in payload.items() if key != "drop"}

    async def artifact(_signal):
        return {"ok": True}

    async def action(_payload):
        return {"ok": True}

    for name, trust in ((_SANITIZED, Trust.UNTRUSTED), (_TRUSTED, Trust.TRUSTED)):
        registry.register(
            SourceRegistration(
                name=name,
                schema=dict,
                default_mode=SignalMode.ARTIFACT,
                allowed_modes=frozenset({SignalMode.ARTIFACT}),
                artifact_handler=artifact,
                sanitizer=sanitize,
                trust=trust,
                log_redaction=_redaction(),
            )
        )
    registry.register(
        SourceRegistration(
            name=_ACTION,
            schema=dict,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=action,
            sanitizer=sanitize,
            trust=Trust.UNTRUSTED,
            log_redaction=_redaction(),
        )
    )
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=agent._get_lock_manager(),
        store=store,
    )
    agent.dispatcher = dispatcher
    await dispatcher.initialize_durable_delivery()
    for source in (_SANITIZED, _TRUSTED, _ACTION):
        await dispatcher.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id=f"consumer:{source}",
                source=source,
                agent_id=agent.did,
            )
        )
    try:
        yield _Rig(
            agent=agent,
            dispatcher=dispatcher,
            backend=backend,
            sanitizer_calls=sanitizer_calls,
        )
    finally:
        await dispatcher.shutdown_durable_delivery()
        await backend.close()


def _signal(agent: KestrelAgent, source: str, **overrides) -> Signal:
    mode = SignalMode.ACTION if source == _ACTION else SignalMode.ARTIFACT
    return Signal(
        source=source,
        kind="inbound",
        mode=mode,
        payload={"reply": "OK", "drop": "x"},
        target_agent=agent.did,
        **overrides,
    )


async def _dispatch_and_claim(rig: _Rig, signal: Signal):
    """Dispatch one signal and claim its durable delivery, as Workflows does."""
    result = await rig.dispatcher.dispatch_signal(signal)
    assert result.status is Status.OK, result.error
    delivery = await rig.dispatcher.claim_durable_delivery(
        consumer_id=f"consumer:{signal.source}", executor_id="workflow-test"
    )
    assert delivery is not None
    assert delivery.event.event_id == signal.id
    return delivery


async def _stored_receipt_row(rig: _Rig, event_id: str):
    return await rig.backend.fetch_one(
        f"SELECT receipt_version, receipt_id, agent_id, source, source_sequence, "
        f"policy_epoch, sanitized_at_ingress FROM {_RECEIPTS} WHERE event_id = ?",
        (event_id,),
    )


def _receipt_id(receipt: dict) -> str:
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# --------------------------------------------------------------------------
# 1. The receipt records what happened to this envelope at ingress
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receipt_for_a_sanitized_event_round_trips_through_the_verifier(rig):
    agent = rig.agent
    delivery = await _dispatch_and_claim(rig, _signal(agent, _SANITIZED))
    assert rig.sanitizer_calls, "the UNTRUSTED ARTIFACT envelope was sanitized"

    receipt = await agent.workflow_await_signal_event_trust_verifier(delivery)

    assert receipt == {
        "version": 1,
        "receipt_id": receipt["receipt_id"],
        "event_id": delivery.event.event_id,
        "agent_id": agent.did,
        "source": _SANITIZED,
        "source_sequence": delivery.event.source_sequence,
        "policy_epoch": agent._get_privacy_policy_epoch(),
        "sanitized_at_ingress": True,
    }
    assert receipt["receipt_id"] == _receipt_id(receipt)
    assert receipt["policy_epoch"].startswith("normal:")
    encoded = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert len(encoded) <= MAX_INGRESS_TRUST_RECEIPT_BYTES


@pytest.mark.asyncio
async def test_sanitized_at_ingress_is_false_when_the_sanitizer_did_not_run(rig):
    """A registered sanitizer is not evidence: only one that ran on this envelope."""
    agent = rig.agent
    trusted = await _dispatch_and_claim(rig, _signal(agent, _TRUSTED))
    action = await _dispatch_and_claim(rig, _signal(agent, _ACTION))
    assert rig.sanitizer_calls == [], "TRUSTED and ACTION envelopes skip the sanitizer"

    for delivery in (trusted, action):
        receipt = await agent.workflow_await_signal_event_trust_verifier(delivery)
        assert receipt is not None
        assert receipt["sanitized_at_ingress"] is False


@pytest.mark.asyncio
async def test_receipt_follows_the_envelope_not_the_source_registration(rig):
    """One TRUSTED source, two envelopes: only the self-downgraded one is sanitized."""
    agent = rig.agent
    plain = await _dispatch_and_claim(rig, _signal(agent, _TRUSTED))
    downgraded = await _dispatch_and_claim(
        rig, _signal(agent, _TRUSTED, origin_trust=Trust.UNTRUSTED)
    )

    assert len(rig.sanitizer_calls) == 1
    assert (
        await agent.workflow_await_signal_event_trust_verifier(plain)
    )["sanitized_at_ingress"] is False
    assert (
        await agent.workflow_await_signal_event_trust_verifier(downgraded)
    )["sanitized_at_ingress"] is True


@pytest.mark.asyncio
async def test_receipt_is_immutable_after_the_registry_changes(rig):
    """Re-registering the source later cannot rewrite what ingress observed."""
    agent = rig.agent
    delivery = await _dispatch_and_claim(rig, _signal(agent, _TRUSTED))
    before = await agent.workflow_await_signal_event_trust_verifier(delivery)

    registry = rig.dispatcher._registry
    registry.unregister(_TRUSTED)
    registry.register(
        SourceRegistration(
            name=_TRUSTED,
            schema=dict,
            default_mode=SignalMode.ARTIFACT,
            allowed_modes=frozenset({SignalMode.ARTIFACT}),
            artifact_handler=AsyncMock(return_value={"ok": True}),
            sanitizer=lambda payload: payload,
            trust=Trust.UNTRUSTED,
            log_redaction=_redaction(),
        )
    )

    assert await agent.workflow_await_signal_event_trust_verifier(delivery) == before
    assert before["sanitized_at_ingress"] is False


@pytest.mark.asyncio
async def test_policy_epoch_names_the_policy_installed_by_a_transition(rig):
    agent = rig.agent
    before = await _dispatch_and_claim(rig, _signal(agent, _SANITIZED))
    result = await agent.set_privacy_mode_with_effects(PrivacyMode.ANONYMOUS)
    assert result.applied
    after = await _dispatch_and_claim(rig, _signal(agent, _SANITIZED))

    first = await agent.workflow_await_signal_event_trust_verifier(before)
    second = await agent.workflow_await_signal_event_trust_verifier(after)

    mode, instance, epoch = first["policy_epoch"].split(":")
    assert (mode, epoch) == ("normal", "0")
    assert second["policy_epoch"] == f"anonymous:{instance}:1"


# --------------------------------------------------------------------------
# 2. The receipt commits with its event, and only then
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_receipt_is_written_in_the_event_transaction(rig):
    """A persist that rolls back leaves neither its event nor its receipt."""
    agent = rig.agent
    store = rig.dispatcher._durable_store
    signal = _signal(agent, _SANITIZED)

    def abort(_persisted):
        raise RuntimeError("abort before commit")

    with pytest.raises(TransactionError, match="abort before commit"):
        await store.persist_signal(
            signal,
            agent_id=agent.did,
            source_event_id=None,
            retention_days=1,
            before_commit=abort,
            ingress_attestation=DurableIngressAttestation(
                policy_epoch=agent._get_privacy_policy_epoch(),
                sanitized_at_ingress=True,
            ),
        )

    assert await rig.backend.fetch_val(
        "SELECT COUNT(*) FROM durable_signal_events WHERE event_id = ?",
        (signal.id,),
    ) == 0
    assert await _stored_receipt_row(rig, signal.id) is None

    persisted = await store.persist_signal(
        signal,
        agent_id=agent.did,
        source_event_id=None,
        retention_days=1,
        ingress_attestation=DurableIngressAttestation(
            policy_epoch=agent._get_privacy_policy_epoch(),
            sanitized_at_ingress=True,
        ),
    )
    row = await _stored_receipt_row(rig, signal.id)
    assert row is not None
    assert tuple(row)[2:5] == (agent.did, _SANITIZED, persisted.source_sequence)


@pytest.mark.asyncio
async def test_a_receipt_that_cannot_be_written_rolls_back_its_event(rig):
    """The event never commits without the receipt its dispatch attested."""
    agent = rig.agent
    # An identifier this long makes the encoded receipt exceed its bound.
    signal = _signal(agent, _SANITIZED, id="e" * MAX_INGRESS_TRUST_RECEIPT_BYTES)

    with pytest.raises(TransactionError, match="exceeds"):
        await rig.dispatcher._durable_store.persist_signal(
            signal,
            agent_id=agent.did,
            source_event_id=None,
            retention_days=1,
            ingress_attestation=DurableIngressAttestation(
                policy_epoch=agent._get_privacy_policy_epoch(),
                sanitized_at_ingress=True,
            ),
        )

    assert await rig.backend.fetch_val(
        "SELECT COUNT(*) FROM durable_signal_events WHERE event_id = ?",
        (signal.id,),
    ) == 0


@pytest.mark.asyncio
async def test_a_duplicate_source_event_keeps_its_original_receipt(rig):
    agent = rig.agent
    first = await rig.dispatcher.dispatch_signal(
        _signal(agent, _TRUSTED), source_event_id="provider-1"
    )
    assert first.status is Status.OK
    original = await rig.backend.fetch_all(f"SELECT * FROM {_RECEIPTS}")

    retry = await rig.dispatcher.dispatch_signal(
        _signal(agent, _TRUSTED, origin_trust=Trust.UNTRUSTED),
        source_event_id="provider-1",
    )

    assert retry.status is Status.COALESCED
    assert await rig.backend.fetch_all(f"SELECT * FROM {_RECEIPTS}") == original


# --------------------------------------------------------------------------
# 3. The verifier fails closed
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_returns_none_for_a_mismatched_delivery(rig):
    agent = rig.agent
    verify = agent.workflow_await_signal_event_trust_verifier
    delivery = await _dispatch_and_claim(rig, _signal(agent, _SANITIZED))
    other = await _dispatch_and_claim(rig, _signal(agent, _SANITIZED))
    event = delivery.event
    assert await verify(delivery) is not None

    mismatches = {
        "source_sequence": replace(event, source_sequence=event.source_sequence + 1),
        "source": replace(event, source=_TRUSTED),
        "agent_id": replace(event, agent_id="did:test:someone-else"),
        # Another event's receipt exists, but names a different sequence.
        "event_id": replace(event, event_id=other.event.event_id),
    }
    for field, mismatched in mismatches.items():
        assert await verify(replace(delivery, event=mismatched)) is None, field
    assert await verify(SimpleNamespace(event=None)) is None
    assert await verify(object()) is None


@pytest.mark.asyncio
async def test_verifier_returns_none_for_an_event_committed_without_a_receipt(rig):
    """Pre-receipt events (and lightweight embeddings) have no evidence."""
    agent = rig.agent
    signal = _signal(agent, _SANITIZED)
    await rig.dispatcher._durable_store.persist_signal(
        signal, agent_id=agent.did, source_event_id=None, retention_days=1
    )
    delivery = await rig.dispatcher.claim_durable_delivery(
        consumer_id=f"consumer:{_SANITIZED}", executor_id="workflow-test"
    )
    assert delivery is not None and delivery.event.event_id == signal.id
    assert await _stored_receipt_row(rig, signal.id) is None

    assert await agent.workflow_await_signal_event_trust_verifier(delivery) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statement", "params"),
    [
        (f"UPDATE {_RECEIPTS} SET sanitized_at_ingress = 1 WHERE event_id = ?", ()),
        (f"UPDATE {_RECEIPTS} SET sanitized_at_ingress = 2 WHERE event_id = ?", ()),
        (f"UPDATE {_RECEIPTS} SET policy_epoch = 'public:x:9' WHERE event_id = ?", ()),
        (
            f"UPDATE {_RECEIPTS} SET source_sequence = source_sequence + 1 "
            "WHERE event_id = ?",
            (),
        ),
        (f"UPDATE {_RECEIPTS} SET receipt_version = 2 WHERE event_id = ?", ()),
        (f"UPDATE {_RECEIPTS} SET receipt_id = 'not-a-digest' WHERE event_id = ?", ()),
        (f"UPDATE {_RECEIPTS} SET receipt_id = 'é' WHERE event_id = ?", ()),
        (
            f"UPDATE {_RECEIPTS} SET receipt_id = '{'0' * 64}' WHERE event_id = ?",
            (),
        ),
        # The receipt must also agree with the committed event row it covers.
        (
            "UPDATE durable_signal_events SET source_sequence = source_sequence + 100 "
            "WHERE event_id = ?",
            (),
        ),
    ],
)
async def test_verifier_returns_none_after_tampering_with_the_stored_row(
    rig, statement, params
):
    agent = rig.agent
    delivery = await _dispatch_and_claim(rig, _signal(agent, _TRUSTED))
    assert await agent.workflow_await_signal_event_trust_verifier(delivery) is not None

    await rig.backend.execute(statement, (*params, delivery.event.event_id))

    assert await agent.workflow_await_signal_event_trust_verifier(delivery) is None


@pytest.mark.asyncio
async def test_a_stored_boolean_other_than_zero_or_one_is_not_read_as_true(rig):
    """``2`` is not a stored ``True``, even on a receipt whose flag was true."""
    agent = rig.agent
    delivery = await _dispatch_and_claim(rig, _signal(agent, _SANITIZED))
    receipt = await agent.workflow_await_signal_event_trust_verifier(delivery)
    assert receipt["sanitized_at_ingress"] is True

    await rig.backend.execute(
        f"UPDATE {_RECEIPTS} SET sanitized_at_ingress = 2 WHERE event_id = ?",
        (delivery.event.event_id,),
    )

    assert await agent.workflow_await_signal_event_trust_verifier(delivery) is None


@pytest.mark.asyncio
async def test_a_coherently_rewritten_receipt_still_round_trips(rig):
    """Control for the tamper cases: rebuilding ``receipt_id`` is what binds it."""
    agent = rig.agent
    delivery = await _dispatch_and_claim(rig, _signal(agent, _TRUSTED))
    receipt = await agent.workflow_await_signal_event_trust_verifier(delivery)
    forged = build_ingress_trust_receipt(
        event_id=receipt["event_id"],
        agent_id=receipt["agent_id"],
        source=receipt["source"],
        source_sequence=receipt["source_sequence"],
        policy_epoch=receipt["policy_epoch"],
        sanitized_at_ingress=True,
    )
    await rig.backend.execute(
        f"UPDATE {_RECEIPTS} SET sanitized_at_ingress = 1, receipt_id = ? "
        "WHERE event_id = ?",
        (forged["receipt_id"], delivery.event.event_id),
    )

    assert await agent.workflow_await_signal_event_trust_verifier(delivery) == forged


def test_receipt_builder_refuses_an_oversized_or_malformed_receipt():
    valid = dict(
        event_id="evt-1",
        agent_id="did:test:3484",
        source="inbound.sanitized",
        source_sequence=1,
        policy_epoch="normal:abc:0",
        sanitized_at_ingress=True,
    )
    assert build_ingress_trust_receipt(**valid)["version"] == 1
    for field, value in (
        ("policy_epoch", ""),
        ("source_sequence", 0),
        ("source_sequence", True),
        ("sanitized_at_ingress", 1),
        ("event_id", "e" * MAX_INGRESS_TRUST_RECEIPT_BYTES),
    ):
        with pytest.raises(ValueError):
            build_ingress_trust_receipt(**{**valid, field: value})


# --------------------------------------------------------------------------
# 4. The privacy hook lock: a transition waits for it, a turn does not stall it
# --------------------------------------------------------------------------


def _block_turn(agent: KestrelAgent) -> tuple[asyncio.Event, asyncio.Event]:
    """Make the next turn hold its span until ``release`` is set."""
    in_body = asyncio.Event()
    release = asyncio.Event()

    async def traced(user_input, *args, **kwargs):
        in_body.set()
        await release.wait()
        return "turn done"

    agent._process_input_traced_locked = traced
    return in_body, release


async def _spin(turns: int = 20) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_privacy_transition_waits_for_a_held_hook_lock(rig):
    agent = rig.agent
    lock = agent.workflow_await_signal_privacy_transition_lock
    assert lock is agent.workflow_await_signal_privacy_transition_lock
    entered = asyncio.Event()

    async def transition() -> None:
        async with agent.privacy_transition():
            entered.set()

    async with lock:
        pending = asyncio.create_task(transition())
        await _spin()
        assert not entered.is_set(), "a transition completed inside the hook lock"
    await asyncio.wait_for(pending, timeout=_HANG_GUARD_SECONDS)
    assert entered.is_set()


@pytest.mark.asyncio
async def test_hook_lock_does_not_wait_for_an_in_flight_turn(rig):
    """The #3316 regression: the hook is not the turn-held privacy lock."""
    agent = rig.agent
    in_body, release = _block_turn(agent)
    turn = asyncio.create_task(agent.process_input("wake"))
    try:
        await asyncio.wait_for(in_body.wait(), timeout=_HANG_GUARD_SECONDS)
        assert agent._get_privacy_transition_lock().locked()

        async def hold_hook() -> str:
            async with agent.workflow_await_signal_privacy_transition_lock:
                return "held"

        hook = asyncio.create_task(hold_hook())
        done, _ = await asyncio.wait(
            {hook, turn},
            timeout=_HANG_GUARD_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
        assert done == {hook}, "the hook lock queued behind the in-flight turn"
        assert hook.result() == "held"
        assert not turn.done()
    finally:
        release.set()
        assert await asyncio.wait_for(turn, timeout=_HANG_GUARD_SECONDS) == "turn done"


async def _queue_transition(agent: KestrelAgent) -> tuple[asyncio.Task, asyncio.Event]:
    """Start a privacy transition and return once it waits on the gate."""
    gate = agent._get_durable_persistence_gate()
    entered = asyncio.Event()

    async def transition() -> None:
        async with agent.privacy_transition():
            entered.set()

    pending = asyncio.create_task(transition())
    for _ in range(200):
        if gate._lock._waiting_writers:
            break
        await asyncio.sleep(0)
    assert gate._lock._waiting_writers == 1, "the transition is queued on the gate"
    return pending, entered


@pytest.mark.asyncio
async def test_hook_holder_can_persist_while_a_transition_is_queued(rig):
    """A queued writer cannot wedge a holder that dispatches a signal."""
    agent = rig.agent
    async with agent.workflow_await_signal_privacy_transition_lock:
        pending, transitioned = await _queue_transition(agent)

        result = await asyncio.wait_for(
            rig.dispatcher.dispatch_signal(_signal(agent, _SANITIZED)),
            timeout=_HANG_GUARD_SECONDS,
        )
        assert result.status is Status.OK
        assert not transitioned.is_set()
    await asyncio.wait_for(pending, timeout=_HANG_GUARD_SECONDS)
    assert transitioned.is_set()


@pytest.mark.asyncio
async def test_hook_lock_reenters_on_the_holding_task_while_a_transition_is_queued(
    rig,
):
    agent = rig.agent
    lock = agent.workflow_await_signal_privacy_transition_lock
    gate = agent._get_durable_persistence_gate()
    # ``asyncio.timeout`` bounds a hang without leaving the current task.
    async with asyncio.timeout(_HANG_GUARD_SECONDS):
        async with lock:
            pending, transitioned = await _queue_transition(agent)
            async with lock:
                async with gate.shared():
                    assert not transitioned.is_set()
            result = await rig.dispatcher.dispatch_signal(_signal(agent, _SANITIZED))
            assert result.status is Status.OK
            assert not transitioned.is_set()
        await pending
    assert transitioned.is_set()
    assert not gate.locked()


async def _reenter_and_persist(rig: _Rig) -> Status:
    """What a hook holder's child does: take the hook lock, persist a signal."""
    async with rig.agent.workflow_await_signal_privacy_transition_lock:
        result = await rig.dispatcher.dispatch_signal(
            _signal(rig.agent, _SANITIZED)
        )
    return result.status


@pytest.mark.asyncio
async def test_hook_lock_reenters_from_wait_for_while_a_transition_is_queued(rig):
    """On Python 3.11 ``wait_for`` runs its coroutine on a separate task."""
    agent = rig.agent
    async with agent.workflow_await_signal_privacy_transition_lock:
        pending, transitioned = await _queue_transition(agent)
        status = await asyncio.wait_for(
            _reenter_and_persist(rig), timeout=_HANG_GUARD_SECONDS
        )
        assert status is Status.OK
        assert not transitioned.is_set()
    await asyncio.wait_for(pending, timeout=_HANG_GUARD_SECONDS)
    assert transitioned.is_set()


@pytest.mark.asyncio
async def test_hook_lock_reenters_from_an_awaited_child_task_while_a_transition_is_queued(
    rig,
):
    agent = rig.agent
    async with agent.workflow_await_signal_privacy_transition_lock:
        pending, transitioned = await _queue_transition(agent)
        child = asyncio.create_task(_reenter_and_persist(rig))
        async with asyncio.timeout(_HANG_GUARD_SECONDS):
            status = await child
        assert status is Status.OK
        assert not transitioned.is_set()
    await asyncio.wait_for(pending, timeout=_HANG_GUARD_SECONDS)
    assert transitioned.is_set()
    assert not agent._get_durable_persistence_gate().locked()


@pytest.mark.asyncio
async def test_a_detached_child_that_runs_after_release_does_not_hold_the_gate(rig):
    agent = rig.agent
    gate = agent._get_durable_persistence_gate()
    parent_released = asyncio.Event()
    child_entered = asyncio.Event()

    async def detached() -> None:
        await parent_released.wait()
        async with agent.workflow_await_signal_privacy_transition_lock:
            child_entered.set()

    async with agent.workflow_await_signal_privacy_transition_lock:
        child = asyncio.create_task(detached())
        await _spin()
    assert not gate.locked()

    in_transition = asyncio.Event()
    finish = asyncio.Event()

    async def transition() -> None:
        async with agent.privacy_transition():
            in_transition.set()
            await finish.wait()

    holder = asyncio.create_task(transition())
    try:
        await asyncio.wait_for(in_transition.wait(), timeout=_HANG_GUARD_SECONDS)
        parent_released.set()
        await _spin()
        assert not child_entered.is_set(), (
            "a child spawned under a released hold entered during a transition"
        )
    finally:
        finish.set()
        await asyncio.wait_for(holder, timeout=_HANG_GUARD_SECONDS)
    await asyncio.wait_for(child, timeout=_HANG_GUARD_SECONDS)
    assert child_entered.is_set()
    assert not gate.locked()


@pytest.mark.asyncio
async def test_a_detached_child_that_runs_after_release_can_transition(rig):
    """A dead inherited hold neither admits the child nor refuses its upgrade."""
    agent = rig.agent
    parent_released = asyncio.Event()

    async def detached() -> str:
        await parent_released.wait()
        async with agent.privacy_transition():
            return "transitioned"

    async with agent.workflow_await_signal_privacy_transition_lock:
        child = asyncio.create_task(detached())
        await _spin()
    parent_released.set()
    assert await asyncio.wait_for(child, timeout=_HANG_GUARD_SECONDS) == "transitioned"
    assert not agent._get_durable_persistence_gate().locked()


@pytest.mark.asyncio
async def test_a_child_still_inside_after_its_parent_left_keeps_the_gate_held(rig):
    agent = rig.agent
    lock = agent.workflow_await_signal_privacy_transition_lock
    gate = agent._get_durable_persistence_gate()
    child_in = asyncio.Event()
    child_release = asyncio.Event()
    nested = asyncio.Event()

    async def child() -> None:
        async with lock:
            child_in.set()
            await child_release.wait()
            # Its parent has left and a transition is queued; the child's own
            # hold still admits it.
            async with lock:
                async with gate.shared():
                    nested.set()

    async with lock:
        spawned = asyncio.create_task(child())
        await asyncio.wait_for(child_in.wait(), timeout=_HANG_GUARD_SECONDS)
    assert gate.locked(), "the parent's exit released a lease its child still holds"

    pending, transitioned = await _queue_transition(agent)
    await _spin()
    assert not transitioned.is_set(), "a transition completed inside a live hold"

    child_release.set()
    await asyncio.wait_for(spawned, timeout=_HANG_GUARD_SECONDS)
    assert nested.is_set()
    await asyncio.wait_for(pending, timeout=_HANG_GUARD_SECONDS)
    assert transitioned.is_set()
    assert not gate.locked()


@pytest.mark.asyncio
async def test_the_transition_waits_for_every_independent_holder(rig):
    agent = rig.agent
    lock = agent.workflow_await_signal_privacy_transition_lock
    gate = agent._get_durable_persistence_gate()
    releases = {name: asyncio.Event() for name in ("a", "b")}
    entered = {name: asyncio.Event() for name in ("a", "b")}

    async def holder(name: str) -> None:
        async with lock:
            entered[name].set()
            await releases[name].wait()

    holders = {name: asyncio.create_task(holder(name)) for name in releases}
    for event in entered.values():
        await asyncio.wait_for(event.wait(), timeout=_HANG_GUARD_SECONDS)
    pending, transitioned = await _queue_transition(agent)

    releases["a"].set()
    await asyncio.wait_for(holders["a"], timeout=_HANG_GUARD_SECONDS)
    await _spin()
    assert not transitioned.is_set(), "a transition completed inside a live hold"

    releases["b"].set()
    await asyncio.wait_for(holders["b"], timeout=_HANG_GUARD_SECONDS)
    await asyncio.wait_for(pending, timeout=_HANG_GUARD_SECONDS)
    assert transitioned.is_set()
    assert not gate.locked()


@pytest.mark.asyncio
async def test_hook_lock_admits_concurrent_and_nested_holders(rig):
    lock = rig.agent.workflow_await_signal_privacy_transition_lock
    gate = rig.agent._get_durable_persistence_gate()
    both_in = asyncio.Event()
    inside: list[str] = []

    async def holder(name: str) -> None:
        async with lock:
            async with lock:
                inside.append(name)
                if len(inside) == 2:
                    both_in.set()
                await asyncio.wait_for(both_in.wait(), timeout=_HANG_GUARD_SECONDS)

    await asyncio.wait_for(
        asyncio.gather(holder("a"), holder("b")), timeout=_HANG_GUARD_SECONDS
    )
    assert sorted(inside) == ["a", "b"]
    assert not gate.locked()


@pytest.mark.asyncio
async def test_a_hook_holder_cannot_take_the_gate_exclusive(rig):
    """The upgrade would wait on its own shared lease; it is refused instead."""
    gate = rig.agent._get_durable_persistence_gate()
    async with rig.agent.workflow_await_signal_privacy_transition_lock:
        with pytest.raises(RuntimeError, match="cannot take it exclusive"):
            async with gate.exclusive():
                pass
    assert not gate.locked()


@pytest.mark.asyncio
async def test_a_child_inside_its_own_reentered_hold_cannot_take_the_gate_exclusive(
    rig,
):
    gate = rig.agent._get_durable_persistence_gate()

    async def upgrade() -> None:
        async with gate.shared():
            async with gate.exclusive():
                pass

    async with rig.agent.workflow_await_signal_privacy_transition_lock:
        child = asyncio.create_task(upgrade())
        with pytest.raises(RuntimeError, match="cannot take it exclusive"):
            async with asyncio.timeout(_HANG_GUARD_SECONDS):
                await child
    assert not gate.locked()


# --------------------------------------------------------------------------
# 5. Workflows accepts the hooks (runs where the feature is installed)
# --------------------------------------------------------------------------


async def _workflow_runner(rig: _Rig, tmp_path, *, with_hooks: bool = True):
    wf_runner = pytest.importorskip("kestrel_feature_workflows.runner")
    from kestrel_feature_workflows.store import WorkflowStore
    from kestrel_sovereign.identity.runtime_identity import AgentIdentity
    from kestrel_sovereign.security.crypto_suite import (
        ALG_ECDSA_SECP256K1_SHA256,
        get_suite,
    )

    suite = get_suite(ALG_ECDSA_SECP256K1_SHA256)
    identity = AgentIdentity(
        legacy_did=rig.agent.did,
        legacy_keypair=suite.generate_keypair(),
        legacy_did_document={},
    )

    def resolve_public_key(did: str) -> bytes:
        if did != identity.legacy_did:
            raise KeyError(did)
        return suite.serialize_public_key(identity.legacy_keypair.public_key)

    def resolve_verification_methods(did: str) -> list:
        raise KeyError(did)

    async def schedule_deadline(*_args, **_kwargs):
        return "deadline-task"

    store = WorkflowStore(rig.backend)
    await store.initialize()
    hooks = {}
    if with_hooks:
        # Exactly the WorkflowsFeature wiring: plain getattr on the agent.
        hooks = dict(
            await_signal_event_trust_verifier=getattr(
                rig.agent, "workflow_await_signal_event_trust_verifier", None
            ),
            await_signal_privacy_transition_lock=getattr(
                rig.agent, "workflow_await_signal_privacy_transition_lock", None
            ),
        )
    runner = wf_runner.WorkflowRunner(
        store=store,
        dispatcher=rig.dispatcher,
        registry=rig.dispatcher._registry,
        agent_identity=identity,
        public_key_resolver=resolve_public_key,
        verification_methods_resolver=resolve_verification_methods,
        await_signal_deadline_scheduler=schedule_deadline,
        **hooks,
    )
    return runner, store, identity


async def _define_await_workflow(store, identity, *, name: str, await_source: str):
    from kestrel_feature_workflows.models import WorkflowSpec
    from kestrel_feature_workflows.signing import sign_workflow_spec

    stage = {
        "signal_source": _ACTION,
        "signal_mode": "action",
        "compensate": "noop_idempotent",
        "read_only": True,
    }
    spec = WorkflowSpec.from_dict(
        {
            "name": name,
            "version": 1,
            "stages": [
                {
                    **stage,
                    "name": "wait",
                    "gate": {
                        "type": "await_signal",
                        "params": {
                            "source": await_source,
                            "timeout_seconds": 30,
                            "matcher_version": 1,
                            "predicate": {"op": "eq", "path": "/reply", "value": "OK"},
                        },
                    },
                },
                {**stage, "name": "signal"},
                {**stage, "name": "timeout"},
            ],
            "edges": [
                {
                    "kind": "branch",
                    "from_stage": "wait",
                    "condition": "gate.passed",
                    "true_stage": "signal",
                    "false_stage": "timeout",
                }
            ],
        }
    )
    await store.put_definition(sign_workflow_spec(spec, identity))


@pytest.mark.asyncio
async def test_workflows_start_run_accepts_an_await_stage_with_the_host_hooks(
    rig, tmp_path
):
    runner, store, identity = await _workflow_runner(rig, tmp_path)
    await _define_await_workflow(
        store, identity, name="await-sanitized", await_source=_SANITIZED
    )

    run = await runner.start_run(name="await-sanitized", params={})

    assert await store.get_run(run.run_id) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("await_source", "with_hooks", "refusal"),
    [
        # ACTION ingress bypasses sanitizers, so it can never qualify.
        (_ACTION, True, "requires sanitized ARTIFACT/COGNITION ingress"),
        # A non-ACTION source that registers no sanitizer at all.
        ("inbound.unsanitized", True, "requires sanitized ARTIFACT/COGNITION ingress"),
        # The baseline the hooks remove: no host contract, no await stage.
        (_SANITIZED, False, "per-event ingress trust evidence"),
    ],
)
async def test_workflows_start_run_still_refuses_an_unqualified_await_stage(
    rig, tmp_path, await_source, with_hooks, refusal
):
    rig.dispatcher._registry.register(
        SourceRegistration(
            name="inbound.unsanitized",
            schema=dict,
            default_mode=SignalMode.ARTIFACT,
            allowed_modes=frozenset({SignalMode.ARTIFACT}),
            artifact_handler=AsyncMock(return_value={"ok": True}),
            trust=Trust.TRUSTED,
            log_redaction=_redaction(),
        )
    )
    runner, store, identity = await _workflow_runner(
        rig, tmp_path, with_hooks=with_hooks
    )
    await _define_await_workflow(
        store, identity, name="await-refused", await_source=await_source
    )
    from kestrel_feature_workflows.runner import WorkflowRunnerError

    with pytest.raises(WorkflowRunnerError, match=refusal):
        await runner.start_run(name="await-refused", params={})
    assert await store.list_runs(limit=10) == []


@pytest.mark.asyncio
async def test_workflows_accepts_the_receipt_only_for_a_sanitized_event(rig, tmp_path):
    """The runner's own receipt checks (shape, binding, size) accept ours."""
    runner, _store, _identity = await _workflow_runner(rig, tmp_path)
    sanitized = await _dispatch_and_claim(rig, _signal(rig.agent, _SANITIZED))
    trusted = await _dispatch_and_claim(rig, _signal(rig.agent, _TRUSTED))

    accepted = await runner._authenticated_await_event_trust_receipt(sanitized)

    assert accepted == await rig.agent.workflow_await_signal_event_trust_verifier(
        sanitized
    )
    assert await runner._authenticated_await_event_trust_receipt(trusted) is None
