"""Peer Stop rides the authenticated signal rails (#3169)."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import (
    CausationFrame,
    RateLimit,
    RedactionPolicy,
    SignalMode,
    Status,
    Trust,
)

from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from kestrel_sovereign.hold import HoldTurnRefusal
from kestrel_sovereign.signals import (
    InFlightControlActionRegistration,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.context import reset_current_signal, set_current_signal
from kestrel_sovereign.signals.registry import RegistrationError
from kestrel_sovereign.signals.sources import peer_stop
from kestrel_sovereign.signals.sources.peer_stop import (
    PEER_STOP_RATE_LIMIT_BURST,
    PEER_STOP_RATE_LIMIT_PER_HOUR,
    PEER_STOP_RATE_LIMIT_PER_MINUTE,
    SOURCE_NAME,
    PeerStopIntentError,
    attach_stop_evidence,
    build_peer_stop_registration,
    build_peer_stop_signal,
    decode_peer_stop_action_envelope,
    decode_peer_stop_intent,
    dispatch_peer_stop,
    encode_peer_stop_intent,
    peer_stop_operation_id,
    resolve_peer_stop_circuit_policy,
)
from kestrel_sovereign.stop import (
    PeerStopCircuitStore,
    StopCleanupRegistry,
    StopReceiptConflict,
    StopReceiptStore,
)
from kestrel_sovereign.stop.receipt import StopOperationClaim
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_DID = "did:test:target"
PEER_DID = "did:test:peer"


class _Agent:
    """The narrow live-agent surface the runtime Stop adapter reads."""

    def __init__(self, did: str = TARGET_DID) -> None:
        self.did = did
        self.background_tasks: list[asyncio.Task] = []
        self._privacy_transition_lock = ReentrantTransitionLock()
        self._active_request_ids: set[str] = set()
        self.cancelled: list[str | None] = []

    @property
    def agent_id(self) -> str:
        return self.did

    def cancel_current_request(self, request_id=None, **_kwargs) -> bool:
        self.cancelled.append(request_id)
        if request_id in self._active_request_ids:
            self._active_request_ids.discard(request_id)
            return True
        return False

    async def wait_for_request_completion(self, _request_id, **_kwargs):
        return RequestCompletionDisposition.COMPLETED

    def _get_privacy_transition_lock(self):
        return self._privacy_transition_lock

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        return task


def _intent(**overrides) -> dict:
    intent = {
        "scope": "agent",
        "target": None,
        "reason": "peer pulled the cord",
        "cascade": False,
        "correlation_id": "peer-stop-1",
    }
    intent.update(overrides)
    return intent


@pytest.fixture
async def rail(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "signals.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    receipt_db = await AsyncDatabase.sqlite(str(tmp_path / "receipts.db"))
    receipts = StopReceiptStore(receipt_db)
    await receipts.ensure_schema()
    circuit = PeerStopCircuitStore(
        receipt_db, policy=resolve_peer_stop_circuit_policy({})
    )
    await circuit.ensure_schema()

    agent = _Agent()
    registry = SourceRegistry()
    registry.register(build_peer_stop_registration(agent))
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    agent.dispatcher = dispatcher
    attach_stop_evidence(
        agent,
        receipt_store=receipts,
        cleanup_registry=StopCleanupRegistry(),
        circuit=circuit,
    )
    yield SimpleNamespace(
        agent=agent,
        dispatcher=dispatcher,
        backend=backend,
        receipts=receipts,
        circuit=circuit,
        receipt_db=receipt_db,
    )
    pending = [task for task in agent.background_tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    await receipt_db.close()
    await backend.close()


async def _receipts(rail) -> list:
    page = await rail.receipts.list_receipts(limit=100)
    return list(page.receipts)


async def _operation_receipt(rail, actor_id, correlation_id, **intent_fields):
    """The receipt under one peer Stop's *operation* id, or ``None``."""

    request = peer_stop.peer_stop_request(
        target_agent_id=peer_stop.peer_stop_target_identity(rail.agent),
        actor_id=actor_id,
        intent=_intent(correlation_id=correlation_id, **intent_fields),
    )
    return await rail.receipts.load(request)


# ---------------------------------------------------------------------------
# Registration contract
# ---------------------------------------------------------------------------


def test_registration_is_a_bounded_trusted_in_flight_action() -> None:
    registration = build_peer_stop_registration(_Agent())

    assert registration.name == SOURCE_NAME
    assert isinstance(registration, InFlightControlActionRegistration)
    assert registration.default_mode is SignalMode.ACTION
    assert registration.allowed_modes == frozenset({SignalMode.ACTION})
    assert registration.trust is Trust.TRUSTED
    assert registration.allow_self_loops is False
    assert registration.rate_limit.per_minute == PEER_STOP_RATE_LIMIT_PER_MINUTE
    assert registration.rate_limit.per_hour == PEER_STOP_RATE_LIMIT_PER_HOUR
    assert registration.rate_limit.burst == PEER_STOP_RATE_LIMIT_BURST
    assert registration.log_redaction.store_raw_trusted is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"allowed_modes": frozenset({SignalMode.ACTION, SignalMode.ARTIFACT})},
        {"trust": Trust.UNTRUSTED},
        {
            "log_redaction": RedactionPolicy(
                summarize=lambda _p: "", store_raw_trusted=True
            )
        },
    ],
)
def test_registry_confines_the_in_flight_exemption(overrides) -> None:
    base = build_peer_stop_registration(_Agent())
    fields = {
        "name": base.name,
        "schema": base.schema,
        "default_mode": base.default_mode,
        "allowed_modes": base.allowed_modes,
        "handler": base.handler,
        "artifact_handler": lambda _signal: None,
        "trust": base.trust,
        "sanitizer": lambda payload: payload,
        "rate_limit": base.rate_limit,
        "log_redaction": base.log_redaction,
    }
    fields.update(overrides)
    with pytest.raises(RegistrationError):
        SourceRegistry().register(InFlightControlActionRegistration(**fields))


def test_contract_signature_distinguishes_in_flight_registration() -> None:
    typed = build_peer_stop_registration(_Agent())
    plain = SourceRegistry.contract_signature(typed)
    assert plain[-1] is True


# ---------------------------------------------------------------------------
# Intent grammar: identities are never payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smuggled", ["actor_id", "sender", "target_agent", "caller"])
def test_payload_cannot_carry_an_identity(smuggled) -> None:
    assert peer_stop.parse_peer_stop_intent(_intent())
    with pytest.raises(PeerStopIntentError):
        peer_stop.parse_peer_stop_intent({**_intent(), smuggled: PEER_DID})


def test_agent_scope_target_comes_from_routing_not_payload() -> None:
    with pytest.raises(PeerStopIntentError):
        peer_stop.parse_peer_stop_intent(_intent(target="did:test:elsewhere"))


def test_intent_round_trips_only_in_canonical_form() -> None:
    message = encode_peer_stop_intent(
        scope="agent",
        target=None,
        reason="r",
        cascade=False,
        correlation_id="c-1",
    )
    assert decode_peer_stop_intent(message)["correlation_id"] == "c-1"
    with pytest.raises(PeerStopIntentError):
        decode_peer_stop_intent(message.replace(",", ", "))


def test_envelope_requires_verb_audience_and_matching_correlation() -> None:
    message = encode_peer_stop_intent(**_intent())
    envelope = {
        "id": "peer-stop-1",
        "sessionId": "peer-stop-session",
        "message": {"role": "user", "parts": [{"type": "text", "text": message}]},
        "metadata": {"a2a_verb": "peer_stop", "a2a_audience": TARGET_DID},
    }
    intent, correlation, _session, _metadata = decode_peer_stop_action_envelope(
        envelope
    )
    assert intent["correlation_id"] == correlation == "peer-stop-1"
    for broken in (
        {**envelope, "metadata": {"a2a_verb": "cancel_task", "a2a_audience": "x"}},
        {**envelope, "metadata": {"a2a_verb": "peer_stop"}},
        {**envelope, "id": "another-id"},
    ):
        with pytest.raises(PeerStopIntentError):
            decode_peer_stop_action_envelope(broken)


def test_cascade_is_grammatical_so_its_refusal_can_be_receipted() -> None:
    # A grammar failure has no receipt; the cascade refusal must, so the
    # grammar accepts it and the dispatcher's schema refuses it.
    assert peer_stop.parse_peer_stop_intent(_intent(cascade=True))["cascade"] is True
    assert peer_stop.peer_stop_policy_refusal(_intent(cascade=True))


def test_operation_id_is_actor_scoped_and_stable() -> None:
    first = peer_stop_operation_id(PEER_DID, "c-1")
    assert first == peer_stop_operation_id(PEER_DID, "c-1")
    assert first != peer_stop_operation_id("did:test:other-peer", "c-1")
    assert len(first.encode()) <= 256


# ---------------------------------------------------------------------------
# The rail end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_stop_is_receipted_under_the_verified_actor(rail) -> None:
    rail.agent._active_request_ids.add("req-live")

    response = await dispatch_peer_stop(rail.agent, actor_id=PEER_DID, intent=_intent())

    assert response["signal_receipt"]["status"] == Status.OK.value
    assert response["correlation_id"] == "peer-stop-1"
    [outcome] = response["stop_outcomes"]
    assert outcome["disposition"] == "stopped"
    assert outcome["agent_id"] == TARGET_DID
    assert outcome["receipt_id"]
    assert rail.agent.cancelled == ["req-live"]

    [receipt] = await _receipts(rail)
    assert receipt.actor_id == PEER_DID
    assert receipt.scope == "agent"
    assert receipt.cascade is False
    assert receipt.target_agent_id == TARGET_DID
    assert [o.disposition for o in receipt.outcomes] == ["stopped"]


@pytest.mark.asyncio
async def test_dispatch_reaches_handler_while_a_turn_holds_the_privacy_lock(
    rail,
) -> None:
    private_chain = [
        CausationFrame(
            agent_id="did:test:ancestor",
            source="private.webhook",
            signal_id="private-signal-id",
            turn_id="private-turn-id",
            depth=1,
            emitted_at=datetime.now(timezone.utc),
        )
    ]
    # The lock is task-reentrant, so a running turn is modelled by a separate
    # task holding it for as long as the Stop is in flight.
    locked = asyncio.Event()
    release = asyncio.Event()

    async def running_turn() -> None:
        async with rail.agent._privacy_transition_lock:
            locked.set()
            await release.wait()

    turn = asyncio.create_task(running_turn())
    await locked.wait()
    try:
        response = await asyncio.wait_for(
            dispatch_peer_stop(
                rail.agent,
                actor_id=PEER_DID,
                intent=_intent(correlation_id="under-lock"),
                causation_chain=private_chain,
            ),
            timeout=5.0,
        )
    finally:
        release.set()
        await turn
    assert response["signal_receipt"]["status"] == Status.OK.value

    row = await rail.backend.fetch_one(
        "SELECT payload, causation_chain FROM durable_signal_events "
        "WHERE agent_id = ?",
        (TARGET_DID,),
    )
    assert json.loads(row[0]) == {"_privacy_gated": "source_policy"}
    assert "peer pulled the cord" not in row[0]
    assert PEER_DID not in row[0]
    assert json.loads(row[1]) == []


@pytest.mark.asyncio
async def test_held_agent_still_receives_peer_stop(rail, monkeypatch) -> None:
    import kestrel_sovereign.hold as hold

    async def held(_agent):
        raise HoldTurnRefusal.__new__(HoldTurnRefusal)

    monkeypatch.setattr(hold, "require_turn_start_allowed", held)
    assert await rail.dispatcher._agent_is_held() is True
    rail.agent._active_request_ids.add("req-held")

    response = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="held")
    )

    assert response["signal_receipt"]["status"] == Status.OK.value
    assert response["stop_outcomes"][0]["disposition"] == "stopped"
    assert rail.agent.cancelled == ["req-held"]


@pytest.mark.asyncio
async def test_cascade_request_is_refused_with_a_receipt(rail) -> None:
    rail.agent._active_request_ids.add("req-live")

    response = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(cascade=True)
    )

    assert response["signal_receipt"]["status"] == Status.DROPPED_VALIDATION.value
    [outcome] = response["stop_outcomes"]
    assert outcome["disposition"] == "refused"
    assert "cascade" in outcome["detail"]
    assert rail.agent.cancelled == []
    [receipt] = await _receipts(rail)
    assert receipt.actor_id == PEER_DID
    assert receipt.cascade is True
    assert receipt.outcomes[0].disposition == "refused"
    assert "cascade" in receipt.outcomes[0].detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "target", "needle"),
    [("host", None, "host scope"), ("tool_call", "tool-call-1", "tool_call")],
)
async def test_out_of_authority_scopes_are_refused_with_receipts(
    rail, scope, target, needle
) -> None:
    response = await dispatch_peer_stop(
        rail.agent,
        actor_id=PEER_DID,
        intent=_intent(scope=scope, target=target, correlation_id=f"s-{scope}"),
    )

    assert response["signal_receipt"]["status"] == Status.DROPPED_VALIDATION.value
    assert response["stop_outcomes"][0]["disposition"] == "refused"
    assert needle in response["stop_outcomes"][0]["detail"]
    assert rail.agent.cancelled == []
    [receipt] = await _receipts(rail)
    assert receipt.scope == scope
    assert receipt.actor_id == PEER_DID


@pytest.mark.asyncio
async def test_causation_cycle_is_refused_with_a_receipt(rail) -> None:
    loop_frame = CausationFrame(
        agent_id=TARGET_DID,
        source=SOURCE_NAME,
        signal_id="earlier",
        turn_id=None,
        depth=1,
        emitted_at=datetime.now(timezone.utc),
    )
    response = await dispatch_peer_stop(
        rail.agent,
        actor_id=PEER_DID,
        intent=_intent(correlation_id="cycle"),
        causation_chain=[loop_frame],
    )

    assert response["signal_receipt"]["status"] == Status.DROPPED_CYCLE.value
    assert response["stop_outcomes"][0]["disposition"] == "refused"
    assert "cycle or depth" in response["stop_outcomes"][0]["detail"]
    assert rail.agent.cancelled == []
    [receipt] = await _receipts(rail)
    assert receipt.outcomes[0].disposition == "refused"


@pytest.mark.asyncio
async def test_depth_ttl_is_refused_with_a_receipt(rail) -> None:
    deep_chain = [
        CausationFrame(
            agent_id=f"did:test:hop-{depth}",
            source="a2a.task_complete",
            signal_id=f"hop-{depth}",
            turn_id=None,
            depth=depth,
            emitted_at=datetime.now(timezone.utc),
        )
        for depth in range(1, 6)
    ]
    response = await dispatch_peer_stop(
        rail.agent,
        actor_id=PEER_DID,
        intent=_intent(correlation_id="deep"),
        causation_chain=deep_chain,
    )

    assert response["signal_receipt"]["status"] == Status.DROPPED_CYCLE.value
    assert response["stop_outcomes"][0]["disposition"] == "refused"
    assert rail.agent.cancelled == []


@pytest.mark.asyncio
async def test_rate_limit_refusal_is_receipted_and_bounds_effects(rail) -> None:
    responses = [
        await dispatch_peer_stop(
            rail.agent,
            actor_id=PEER_DID,
            intent=_intent(correlation_id=f"burst-{index}"),
        )
        for index in range(PEER_STOP_RATE_LIMIT_BURST + 1)
    ]

    statuses = [r["signal_receipt"]["status"] for r in responses]
    assert statuses[:-1] == [Status.OK.value] * PEER_STOP_RATE_LIMIT_BURST
    assert statuses[-1] == Status.DROPPED_RATE_LIMIT.value
    refused = responses[-1]["stop_outcomes"][0]
    assert refused["disposition"] == "refused"
    assert "rate limit" in refused["detail"]
    # Each admitted Stop reached the cancellation seam once; the refused one
    # never did.
    assert len(rail.agent.cancelled) == PEER_STOP_RATE_LIMIT_BURST
    receipts = await _receipts(rail)
    assert len(receipts) == PEER_STOP_RATE_LIMIT_BURST + 1
    assert receipts[-1].outcomes[0].disposition == "refused"


@pytest.mark.asyncio
async def test_rate_limit_is_shared_across_peers_and_outlives_a_retry(rail) -> None:
    for index in range(PEER_STOP_RATE_LIMIT_BURST):
        await dispatch_peer_stop(
            rail.agent,
            actor_id=f"did:test:peer-{index}",
            intent=_intent(correlation_id="same"),
        )
    refused = await dispatch_peer_stop(
        rail.agent, actor_id="did:test:late-peer", intent=_intent(correlation_id="x")
    )
    assert refused["signal_receipt"]["status"] == Status.DROPPED_RATE_LIMIT.value
    [refusal] = refused["stop_outcomes"]
    assert refusal["disposition"] == "refused"
    assert "rate limits" in refusal["detail"]
    # The refusal is the delivery's, receipted under its signal id; it never
    # decides the operation.  A retry is therefore rate-limited again as its
    # own delivery (a slot per attempt, so retrying cannot amplify) rather
    # than replaying a refusal that would outlive the budget.
    assert refusal["correlation_id"] == (
        f"peer-stop-delivery:{refused['signal_receipt']['signal_id']}"
    )
    operation_id = peer_stop_operation_id("did:test:late-peer", "x")
    assert await _operation_receipt(rail, "did:test:late-peer", "x") is None
    retry = await dispatch_peer_stop(
        rail.agent, actor_id="did:test:late-peer", intent=_intent(correlation_id="x")
    )
    assert retry["signal_receipt"]["status"] == Status.DROPPED_RATE_LIMIT.value
    assert retry["stop_outcomes"][0]["disposition"] == "refused"
    assert retry["stop_outcomes"][0]["correlation_id"] != refusal["correlation_id"]

    # Once the budget refills, the same operation is admitted and executes.
    cancelled_before = list(rail.agent.cancelled)
    rail.agent._active_request_ids.add("req-live")
    rail.dispatcher._rate = type(rail.dispatcher._rate)()
    admitted = await dispatch_peer_stop(
        rail.agent, actor_id="did:test:late-peer", intent=_intent(correlation_id="x")
    )
    assert admitted["stop_outcomes"][0]["disposition"] == "stopped"
    assert admitted["stop_outcomes"][0]["correlation_id"] == operation_id
    assert rail.agent.cancelled[len(cancelled_before):] == ["req-live"]


@pytest.mark.asyncio
async def test_replay_is_idempotent_and_does_not_fan_out_again(rail) -> None:
    rail.agent._active_request_ids.add("req-live")
    first = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="once")
    )
    rail.agent._active_request_ids.add("req-second")
    replay = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="once")
    )

    assert replay["signal_receipt"]["status"] == Status.COALESCED.value
    assert replay["stop_outcomes"] == first["stop_outcomes"]
    assert rail.agent.cancelled == ["req-live"]
    assert len(await _receipts(rail)) == 1


async def _interrupt_after_source_event_commit(
    rail, monkeypatch, correlation_id, **intent_fields
):
    """Cancel one peer Stop after its durable source event committed.

    Everything past the commit (Hold, rate limit, locks, the handler) is
    replaced by a hang, then the dispatch is cancelled: the process-loss
    boundary where the dedup record exists but no Stop receipt does.
    """

    committed = asyncio.Event()
    original = rail.dispatcher._route_after_durable_persistence

    async def hang(signal, registration, start):
        if signal.source != SOURCE_NAME:
            return await original(signal, registration, start)
        committed.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(rail.dispatcher, "_route_after_durable_persistence", hang)
    first = asyncio.create_task(
        dispatch_peer_stop(
            rail.agent,
            actor_id=PEER_DID,
            intent=_intent(correlation_id=correlation_id, **intent_fields),
        )
    )
    await asyncio.wait_for(committed.wait(), timeout=5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    monkeypatch.setattr(rail.dispatcher, "_route_after_durable_persistence", original)
    assert await _receipts(rail) == []
    assert rail.agent.cancelled == []


@pytest.mark.asyncio
async def test_retry_completes_a_stop_interrupted_after_its_event_committed(
    rail, monkeypatch
) -> None:
    """The receipt, not the dedup record, decides whether a Stop happened."""

    rail.agent._active_request_ids.add("req-live")
    await _interrupt_after_source_event_commit(rail, monkeypatch, "interrupted")

    retry = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="interrupted")
    )

    assert retry["signal_receipt"]["status"] == Status.OK.value
    assert retry["stop_outcomes"][0]["disposition"] == "stopped"
    assert rail.agent.cancelled == ["req-live"]
    [receipt] = await _receipts(rail)
    assert receipt.actor_id == PEER_DID

    # Once the outcome is durable, a further retry is a pure replay.
    rail.agent._active_request_ids.add("req-second")
    replay = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="interrupted")
    )
    assert replay["signal_receipt"]["status"] == Status.COALESCED.value
    assert replay["stop_outcomes"] == retry["stop_outcomes"]
    assert rail.agent.cancelled == ["req-live"]
    assert len(await _receipts(rail)) == 1


@pytest.mark.asyncio
async def test_recovering_an_interrupted_stop_is_still_rate_limited(
    rail, monkeypatch
) -> None:
    rail.agent._active_request_ids.add("req-live")
    await _interrupt_after_source_event_commit(rail, monkeypatch, "late")
    for index in range(PEER_STOP_RATE_LIMIT_BURST):
        await dispatch_peer_stop(
            rail.agent,
            actor_id=PEER_DID,
            intent=_intent(correlation_id=f"budget-{index}"),
        )
    cancelled_before = list(rail.agent.cancelled)

    retry = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="late")
    )

    assert retry["signal_receipt"]["status"] == Status.DROPPED_RATE_LIMIT.value
    [refusal] = retry["stop_outcomes"]
    assert refusal["disposition"] == "refused"
    assert "rate limits" in refusal["detail"]
    assert rail.agent.cancelled == cancelled_before
    # The unadmitted retry is receipted under its own delivery id, never the
    # operation id, so it cannot strand the Stop: once the budget frees (for
    # example after the restart that interrupted it), a later retry completes
    # the operation.
    late_id = peer_stop_operation_id(PEER_DID, "late")
    assert len(await _receipts(rail)) == PEER_STOP_RATE_LIMIT_BURST + 1
    assert await _operation_receipt(rail, PEER_DID, "late") is None
    assert refusal["correlation_id"] == (
        f"peer-stop-delivery:{retry['signal_receipt']['signal_id']}"
    )
    # The response names the record it wrote: this delivery.
    assert retry["receipt_kind"] == "delivery"
    assert retry["stop_correlation_id"] == refusal["correlation_id"]
    rail.dispatcher._rate = type(rail.dispatcher._rate)()
    rail.agent._active_request_ids.add("req-after-restart")
    completed = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="late")
    )
    assert completed["stop_correlation_id"] == late_id
    assert completed["receipt_kind"] == "operation"
    assert completed["recorded"] is True
    assert completed["stop_outcomes"][0]["disposition"] == "stopped"
    assert rail.agent.cancelled == [*cancelled_before, "req-after-restart"]
    assert len(await _receipts(rail)) == PEER_STOP_RATE_LIMIT_BURST + 2


@pytest.mark.asyncio
async def test_unadmitted_retry_cannot_preempt_an_admitted_first_attempt(
    rail, monkeypatch
) -> None:
    """A rate-limited retry must not write the outcome of an admitted attempt.

    The first delivery passes the rate limit and is paused before it claims
    the Stop operation.  The budget is then filled and the same operation is
    retried: the retry is refused re-admission and must leave the operation's
    receipt to the delivery that was admitted.
    """

    rail.agent._active_request_ids.add("req-live")
    operation_id = peer_stop_operation_id(PEER_DID, "race")
    entered = asyncio.Event()
    release = asyncio.Event()
    original_claim = rail.receipts.claim

    async def gated_claim(request):
        if request.correlation_id == operation_id and not entered.is_set():
            entered.set()
            await release.wait()
        return await original_claim(request)

    monkeypatch.setattr(rail.receipts, "claim", gated_claim)
    first = asyncio.create_task(
        dispatch_peer_stop(
            rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="race")
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        for index in range(PEER_STOP_RATE_LIMIT_BURST - 1):
            await dispatch_peer_stop(
                rail.agent,
                actor_id=PEER_DID,
                # Turn scope at an absent turn: consumes the budget, stops
                # nothing.
                intent=_intent(
                    correlation_id=f"fill-{index}",
                    scope="turn",
                    target=f"absent-turn-{index}",
                ),
            )
        assert rail.agent.cancelled == []
        retry = await dispatch_peer_stop(
            rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="race")
        )
    finally:
        release.set()
    original = await asyncio.wait_for(first, timeout=5)

    assert original["signal_receipt"]["status"] == Status.OK.value
    assert original["stop_outcomes"][0]["disposition"] == "stopped"
    assert rail.agent.cancelled == ["req-live"]
    assert retry["signal_receipt"]["status"] == Status.DROPPED_RATE_LIMIT.value
    [refusal] = retry["stop_outcomes"]
    assert refusal["disposition"] == "refused"
    assert "rate limits" in refusal["detail"]
    # Exactly one receipt for the operation, and it is the admitted outcome;
    # the refused retry is receipted separately under its own delivery.
    replay = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="race")
    )
    assert replay["signal_receipt"]["status"] == Status.COALESCED.value
    assert replay["stop_outcomes"] == original["stop_outcomes"]
    receipts = await _receipts(rail)
    assert len(receipts) == PEER_STOP_RATE_LIMIT_BURST + 1
    operation_receipt = await _operation_receipt(rail, PEER_DID, "race")
    assert [o.disposition.value for o in operation_receipt.outcomes] == ["stopped"]
    # The delivery refusal is on the sovereign read surface with its actor
    # and the refusal named, beside -- not instead of -- the operation.
    refused_rows = [
        receipt
        for receipt in receipts
        if [o.disposition for o in receipt.outcomes] == ["refused"]
    ]
    assert len(refused_rows) == 1
    assert refused_rows[0].actor_id == PEER_DID
    assert "rate limits" in refused_rows[0].outcomes[0].detail


@pytest.mark.asyncio
async def test_retry_against_a_stranded_claim_is_an_honest_in_progress_refusal(
    rail, monkeypatch
) -> None:
    """A claim whose owner died is surfaced, not reinterpreted (#3356).

    Recovering a stranded operation claim is #3356's.  Until then the peer
    rail answers a retry with the typed "already in progress" refusal and
    writes nothing under the operation id that could pretend to decide it.
    """

    rail.agent._active_request_ids.add("req-live")
    await _interrupt_after_source_event_commit(rail, monkeypatch, "stranded")
    operation = peer_stop.peer_stop_request(
        target_agent_id=peer_stop.peer_stop_target_identity(rail.agent),
        actor_id=PEER_DID,
        intent=_intent(correlation_id="stranded"),
    )
    assert await rail.receipts.claim(operation) is not None  # owner then dies

    retry = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="stranded")
    )

    [outcome] = retry["stop_outcomes"]
    assert outcome["disposition"] == "refused"
    assert "already in progress" in outcome["detail"]
    assert rail.agent.cancelled == []
    assert await _operation_receipt(rail, PEER_DID, "stranded") is None


@pytest.mark.asyncio
async def test_retry_during_a_running_first_attempt_does_not_stop_twice(
    rail, monkeypatch
) -> None:
    rail.agent._active_request_ids.add("req-live")
    entered = asyncio.Event()
    release = asyncio.Event()
    original_cancel = rail.agent.cancel_current_request

    def slow_cancel(request_id=None, **kwargs):
        entered.set()
        return original_cancel(request_id, **kwargs)

    original_wait = rail.agent.wait_for_request_completion

    async def gated_wait(request_id, **kwargs):
        await release.wait()
        return await original_wait(request_id, **kwargs)

    monkeypatch.setattr(rail.agent, "cancel_current_request", slow_cancel)
    monkeypatch.setattr(rail.agent, "wait_for_request_completion", gated_wait)

    first = asyncio.create_task(
        dispatch_peer_stop(
            rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="busy")
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    retry = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="busy")
    )
    release.set()
    original = await first

    assert retry["stop_outcomes"][0]["disposition"] == "refused"
    assert "in progress" in retry["stop_outcomes"][0]["detail"]
    assert original["stop_outcomes"][0]["disposition"] == "stopped"
    assert rail.agent.cancelled == ["req-live"]
    assert len(await _receipts(rail)) == 1


@pytest.mark.asyncio
async def test_reused_correlation_for_a_different_request_is_refused(rail) -> None:
    await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="reuse")
    )
    changed = await dispatch_peer_stop(
        rail.agent,
        actor_id=PEER_DID,
        intent=_intent(correlation_id="reuse", reason="a different reason"),
    )
    assert changed["stop_outcomes"][0]["disposition"] == "refused"
    assert "reused" in changed["stop_outcomes"][0]["detail"]


@pytest.mark.asyncio
async def test_changed_intent_cannot_take_over_an_interrupted_operation(
    rail, monkeypatch
) -> None:
    """An operation identity binds its intent at first sight.

    The first request targets a turn and is interrupted after its source
    event committed, before it claimed the operation.  A second request
    reusing the correlation id for the whole agent is refused as identity
    reuse -- before dispatch, under its own delivery -- and never claims the
    operation; only the first intent can complete it.
    """

    rail.agent._active_request_ids.add("req-live")
    turn_intent = {"scope": "turn", "target": "turn-absent"}
    await _interrupt_after_source_event_commit(
        rail, monkeypatch, "taken", **turn_intent
    )
    dispatched: list[object] = []
    original_dispatch = rail.dispatcher.dispatch_signal

    async def counting_dispatch(signal, **kwargs):
        dispatched.append(signal)
        return await original_dispatch(signal, **kwargs)

    monkeypatch.setattr(rail.dispatcher, "dispatch_signal", counting_dispatch)

    changed = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="taken")
    )

    assert dispatched == []
    assert changed["receipt_kind"] == "delivery"
    assert changed["signal_receipt"]["status"] is None
    [refusal] = changed["stop_outcomes"]
    assert refusal["disposition"] == "refused"
    assert "reused for a different request" in refusal["detail"]
    assert refusal["correlation_id"] == changed["stop_correlation_id"]
    assert changed["stop_correlation_id"] == (
        f"peer-stop-delivery:{changed['signal_receipt']['signal_id']}"
    )
    assert rail.agent.cancelled == []
    assert await _operation_receipt(rail, PEER_DID, "taken") is None
    # The refusal is on the sovereign read surface under the verified actor.
    [refused_row] = await _receipts(rail)
    assert refused_row.actor_id == PEER_DID
    assert "reused" in refused_row.outcomes[0].detail

    # The first intent still completes its own operation, and only that.
    retry = await dispatch_peer_stop(
        rail.agent,
        actor_id=PEER_DID,
        intent=_intent(correlation_id="taken", **turn_intent),
    )
    assert retry["receipt_kind"] == "operation"
    assert retry["stop_correlation_id"] == peer_stop_operation_id(PEER_DID, "taken")
    assert retry["stop_outcomes"][0]["scope"] == "turn"
    assert rail.agent.cancelled == []
    operation = await _operation_receipt(
        rail, PEER_DID, "taken", **turn_intent
    )
    assert operation is not None and operation.scope == "turn"


@pytest.mark.asyncio
async def test_receipt_store_refuses_to_claim_an_operation_bound_elsewhere(
    rail,
) -> None:
    """The binding is honored by the claim itself, not only by the door."""

    target = peer_stop.peer_stop_target_identity(rail.agent)
    first = peer_stop.peer_stop_request(
        target_agent_id=target,
        actor_id=PEER_DID,
        intent=_intent(correlation_id="bound", scope="turn", target="turn-1"),
    )
    changed = peer_stop.peer_stop_request(
        target_agent_id=target,
        actor_id=PEER_DID,
        intent=_intent(correlation_id="bound"),
    )
    await rail.receipts.bind_operation(first)
    await rail.receipts.bind_operation(first)  # idempotent for its own intent

    with pytest.raises(StopReceiptConflict):
        await rail.receipts.bind_operation(changed)
    with pytest.raises(StopReceiptConflict):
        await rail.receipts.claim(changed)
    assert isinstance(await rail.receipts.claim(first), StopOperationClaim)


@pytest.mark.asyncio
async def test_two_peers_may_share_a_correlation_id(rail) -> None:
    first = await dispatch_peer_stop(
        rail.agent, actor_id=PEER_DID, intent=_intent(correlation_id="shared")
    )
    second = await dispatch_peer_stop(
        rail.agent,
        actor_id="did:test:other-peer",
        intent=_intent(correlation_id="shared"),
    )
    assert first["signal_receipt"]["status"] == Status.OK.value
    assert second["signal_receipt"]["status"] == Status.OK.value
    assert first["stop_correlation_id"] != second["stop_correlation_id"]
    actors = sorted(r.actor_id for r in await _receipts(rail))
    assert actors == sorted([PEER_DID, "did:test:other-peer"])


@pytest.mark.asyncio
async def test_missing_receipt_store_refuses_before_any_effect(tmp_path) -> None:
    backend = SQLiteBackend(str(tmp_path / "signals.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    agent = _Agent()
    agent._active_request_ids.add("req-live")
    registry = SourceRegistry()
    registry.register(build_peer_stop_registration(agent))
    agent.dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    try:
        response = await dispatch_peer_stop(agent, actor_id=PEER_DID, intent=_intent())
    finally:
        await backend.close()

    assert response["stop_outcomes"][0]["disposition"] == "refused"
    assert agent.cancelled == []


# ---------------------------------------------------------------------------
# Bypass guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_refuses_without_its_dispatcher_signal() -> None:
    agent = _Agent()
    agent._active_request_ids.add("req-live")
    handler = build_peer_stop_registration(agent).handler

    with pytest.raises(ValueError, match="dispatcher signal principal"):
        await handler(_intent())
    assert agent.cancelled == []


@pytest.mark.asyncio
async def test_handler_reads_the_actor_from_the_signal_principal(rail) -> None:
    signal = build_peer_stop_signal(
        agent=rail.agent, actor_id="did:test:verified", intent=_intent()
    )
    token = set_current_signal(signal)
    try:
        outcomes = await build_peer_stop_registration(rail.agent).handler(
            dict(signal.payload)
        )
    finally:
        reset_current_signal(token)

    assert outcomes[0]["correlation_id"] == peer_stop_operation_id(
        "did:test:verified", "peer-stop-1"
    )
    [receipt] = await _receipts(rail)
    assert receipt.actor_id == "did:test:verified"


@pytest.mark.asyncio
async def test_handler_never_takes_an_actor_from_the_payload(rail) -> None:
    rail.agent._active_request_ids.add("req-live")
    signal = build_peer_stop_signal(
        agent=rail.agent, actor_id="did:test:verified", intent=_intent()
    )
    token = set_current_signal(signal)
    try:
        with pytest.raises(PeerStopIntentError):
            await build_peer_stop_registration(rail.agent).handler(
                {**signal.payload, "actor_id": "did:test:forged"}
            )
    finally:
        reset_current_signal(token)
    assert rail.agent.cancelled == []
    assert await _receipts(rail) == []


@pytest.mark.asyncio
async def test_handler_refuses_a_signal_aimed_at_another_agent(rail) -> None:
    signal = build_peer_stop_signal(
        agent=_Agent("did:test:someone-else"), actor_id=PEER_DID, intent=_intent()
    )
    token = set_current_signal(signal)
    try:
        with pytest.raises(ValueError, match="target"):
            await build_peer_stop_registration(rail.agent).handler(
                dict(signal.payload)
            )
    finally:
        reset_current_signal(token)


def test_door_function_only_dispatches() -> None:
    source = inspect.getsource(dispatch_peer_stop)
    assert "dispatch_signal(" in source
    for forbidden in ("cancel_current_request", "CancellationAuthority", ".handler("):
        assert forbidden not in source


def _source(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_local_door_census() -> None:
    """Only the two peer doors build a peer Stop, and neither cancels."""

    users = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "kestrel_sovereign").rglob("*.py")
        if re.search(
            r"\b(dispatch_peer_stop|build_peer_stop_signal)\b",
            path.read_text(encoding="utf-8"),
        )
    )
    assert users == [
        "kestrel_sovereign/endpoints/agent.py",
        "kestrel_sovereign/multi_agent/agent_manager.py",
        "kestrel_sovereign/signals/sources/peer_stop.py",
    ]

    from kestrel_sovereign.endpoints import agent as agent_endpoints
    from kestrel_sovereign.endpoints import host_stop
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    for door in (
        agent_endpoints.stop_from_peer,
        AgentManager.stop_host_attested_local_peer,
    ):
        source = inspect.getsource(door)
        assert "dispatch_peer_stop(" in source
        for forbidden in (
            "cancel_current_request",
            "CancellationAuthority",
            "build_runtime_stop_target",
        ):
            assert forbidden not in source

    # Operator/local doors never mint or impersonate a peer signal.
    for door in (agent_endpoints.stop_agent_request, host_stop.stop_host):
        source = inspect.getsource(door)
        assert "peer_stop" not in source
        assert "dispatch_signal" not in source


def test_operator_stop_is_not_reachable_on_the_a2a_transport_lane() -> None:
    from kestrel_sovereign.a2a.transport_auth import is_a2a_transport_path

    assert is_a2a_transport_path("POST", "/api/agent/peer/stop")
    assert is_a2a_transport_path("POST", "/api/agents/kite/api/agent/peer/stop")
    assert not is_a2a_transport_path("POST", "/api/agent/stop")
    assert not is_a2a_transport_path("POST", "/api/host/stop")


def test_registration_is_wired_into_agent_boot() -> None:
    source = _source("kestrel_sovereign/kestrel_agent.py")
    assert "build_peer_stop_registration(self)" in source


def test_rate_limit_constants_are_the_registered_bounds() -> None:
    registration = build_peer_stop_registration(_Agent())
    assert registration.rate_limit == RateLimit(
        per_minute=PEER_STOP_RATE_LIMIT_PER_MINUTE,
        per_hour=PEER_STOP_RATE_LIMIT_PER_HOUR,
        burst=PEER_STOP_RATE_LIMIT_BURST,
    )


def test_server_attaches_stop_evidence_with_the_distributed_registry() -> None:
    from kestrel_sovereign.server import _attach_stop_runtime

    attached: list[object] = []
    receipts = object()
    circuit = PeerStopCircuitStore(
        None, policy=resolve_peer_stop_circuit_policy({})
    )
    state = SimpleNamespace(
        distributed_invocation_registry=SimpleNamespace(attach=attached.append),
        stop_receipt_store=receipts,
        stop_cleanup_registry=None,
        peer_stop_circuit=circuit,
    )
    agent = _Agent()

    _attach_stop_runtime(SimpleNamespace(state=state), agent)

    assert attached == [agent]
    assert isinstance(state.stop_cleanup_registry, StopCleanupRegistry)
    assert peer_stop._stop_evidence(agent) == (
        receipts,
        state.stop_cleanup_registry,
    )
    assert agent.__dict__[peer_stop._CIRCUIT_ATTRIBUTE] is circuit


def test_server_refuses_receipts_without_a_circuit_breaker() -> None:
    """Receipts without a breaker would honor peer Stops uncounted (#3170)."""

    from kestrel_sovereign.server import _attach_stop_runtime

    state = SimpleNamespace(
        distributed_invocation_registry=None,
        stop_receipt_store=object(),
        stop_cleanup_registry=None,
        peer_stop_circuit=None,
    )
    with pytest.raises(RuntimeError, match="circuit breaker"):
        _attach_stop_runtime(SimpleNamespace(state=state), _Agent())


def test_server_attach_sites_share_one_helper() -> None:
    source = _source("kestrel_sovereign/server.py")
    assert source.count("distributed_stop.attach(") == 1
    assert source.count("_attach_stop_runtime(app, ") == 3
