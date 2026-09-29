"""The durable row, not a reservation token, decides a held event's fate (#3283).

A payload-elided event is first reserved to the dispatcher that emitted it.
Other claimants may legitimately return or take that reservation while the
emitting dispatch is between its row claim and its reservation transfer: a
generic claim deferred by Hold returns every volatile reservation of its
consumer, and an unheld generic claim transfers one to its executor. The
emitting dispatch must then read the outcome from the ledger row rather than
treat the missing token as "delivery unavailable".

Every test runs a real dispatcher over a real SQLite ledger.
"""

from __future__ import annotations

import asyncio

import pytest
from kestrel_sdk.signals import Status

from kestrel_sovereign.hold import EffectiveHoldState, HoldEnforcementUnavailableError
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.signals import (
    ACKNOWLEDGED,
    LEASED,
    PENDING,
    RETRY,
    DurableAdmissionDisposition,
    DurableConsumerRegistration,
)
from kestrel_sovereign.signals.sources.channels import (
    DURABLE_COGNITION_CONSUMER_ID,
    DURABLE_COGNITION_MARKER,
    DURABLE_COGNITION_MARKER_VALUE,
    DURABLE_TERMINAL_CONSUMER_ID,
    DURABLE_TERMINAL_MARKER,
    DURABLE_TERMINAL_MARKER_VALUE,
)
from tests.unit.test_durable_signal_delivery import (
    _channel_dispatcher,
    _channel_signal,
    _close,
    _dispatcher,
    _held_state,
    _HoldSnapshots,
    _signal,
)

NOT_HELD = EffectiveHoldState(host=None, agent=None)
_CONTENT = "the only copy of this private content"


class _RaisingHoldRead:
    """A Hold store whose first read fails and whose later reads answer."""

    def __init__(self) -> None:
        self.reads = 0

    async def get_effective(self, _agent_id: str) -> EffectiveHoldState:
        self.reads += 1
        if self.reads == 1:
            raise RuntimeError("hold backend unavailable")
        return NOT_HELD


def _cognition_consumer(agent_id: str, *, max_attempts: int) -> DurableConsumerRegistration:
    return DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source="channel.message",
        agent_id=agent_id,
        correlation_selector=(
            f"payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}"
        ),
        max_attempts=max_attempts,
    )


async def _race_with_parked_emitting_dispatch(
    tmp_path,
    *,
    name: str,
    max_attempts: int,
    interloper,
) -> dict:
    """Park an ephemeral channel dispatch after its exact row claim misses.

    The miss is the ordinary first-delivery shape: the row is the emitting
    dispatch's own activated reservation, which only its reservation transfer
    may claim. ``interloper(dispatcher, hold, consumer_id)`` runs while the
    emitting dispatch is parked, before the transfer.
    """

    did = f"did:agent:hold-claim-ownership:{name}"
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / f"{name}.db", did)
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = _cognition_consumer(did, max_attempts=max_attempts)
    hold = _HoldSnapshots(NOT_HELD)
    agent._hold_store = hold

    prompts: list[str] = []

    async def record_turn(prompt: str):
        prompts.append(prompt)
        return "ok"

    agent.process_input = record_turn

    signal = _channel_signal(did, name)
    signal.payload["content"] = _CONTENT
    store = dispatcher._durable_store
    original_row_claim = store.claim_delivery_for_event
    paused = asyncio.Event()
    resume = asyncio.Event()
    parked = False
    observed: dict = {}

    async def park_after_first_row_claim(**kwargs):
        nonlocal parked
        claimed = await original_row_claim(**kwargs)
        if not parked and kwargs.get("event_id") == signal.id:
            parked = True
            observed["first_row_claim"] = claimed
            paused.set()
            await resume.wait()
        return claimed

    store.claim_delivery_for_event = park_after_first_row_claim

    async def the_row():
        [row] = await dispatcher.list_durable_deliveries(consumer_id=consumer.consumer_id)
        return row

    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            signal,
            source_event_id=f"telegram:update:{name}",
            consumer_id=consumer.consumer_id,
        )
        await asyncio.wait_for(paused.wait(), timeout=2.0)
        at_pause = await the_row()
        handoff = dispatcher._transient_durable_handoffs[at_pause.delivery_id]
        # The race window: the emitting dispatch owns an activated reservation.
        assert observed["first_row_claim"] is None
        assert at_pause.status == LEASED
        assert at_pause.lease_owner == dispatcher._durable_delivery_owner
        assert handoff.initial_lease_token is not None

        observed["interloper"] = await interloper(dispatcher, hold, consumer.consumer_id)
        observed["after_interloper"] = await the_row()

        resume.set()
        admission = await asyncio.wait_for(handle.wait_for_durable_admission(), timeout=2.0)
        result = await asyncio.wait_for(handle.wait(), timeout=2.0)
        observed["admission"] = admission.disposition
        observed["result"] = (result.status, result.error)
        observed["after_emit"] = await the_row()
        observed["prompts"] = list(prompts)
    finally:
        resume.set()
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)
    return observed


async def _hold_deferred_generic_claim(dispatcher, hold, consumer_id):
    """Hold arrives and a generic poller asks for work on the same consumer."""

    hold.snapshots[:] = [_held_state(dispatcher._agent.did)]
    claimed = await dispatcher.claim_durable_delivery(
        consumer_id=consumer_id, executor_id="generic-poller"
    )
    assert claimed is None
    return claimed


@pytest.mark.asyncio
@pytest.mark.parametrize("max_attempts", (0, 1), ids=("unbounded", "bounded"))
async def test_emitting_dispatch_claims_the_row_a_hold_deferral_returned(
    tmp_path, max_attempts
):
    """Hold returned the reservation and was released: run it, once, now.

    Before #3283 the emitting dispatch found its token gone and failed the
    first delivery as unavailable. The drain could not run the elided row
    (its caller exists only in the live envelope), so a bounded consumer
    spent its only attempt there and the message never ran.
    """

    async def interloper(dispatcher, hold, consumer_id):
        await _hold_deferred_generic_claim(dispatcher, hold, consumer_id)
        hold.snapshots[:] = [NOT_HELD]

    observed = await _race_with_parked_emitting_dispatch(
        tmp_path,
        name=f"returned-then-released-{max_attempts}",
        max_attempts=max_attempts,
        interloper=interloper,
    )

    returned = observed["after_interloper"]
    assert returned.status == RETRY
    assert returned.attempts == 0
    assert returned.last_error == "hold_deferred"

    assert observed["admission"] is DurableAdmissionDisposition.COMMITTED
    assert observed["result"] == (Status.OK, None)
    assert len(observed["prompts"]) == 1
    assert _CONTENT in observed["prompts"][0]
    settled = observed["after_emit"]
    assert settled.status == ACKNOWLEDGED
    assert settled.attempts == 1


@pytest.mark.asyncio
async def test_emitting_dispatch_stays_held_when_hold_outlives_the_returned_reservation(
    tmp_path,
):
    """Claiming the returned row must not run past a Hold that is still set."""

    observed = await _race_with_parked_emitting_dispatch(
        tmp_path,
        name="returned-still-held",
        max_attempts=1,
        interloper=_hold_deferred_generic_claim,
    )

    assert observed["admission"] is DurableAdmissionDisposition.HELD
    assert observed["result"] == (Status.COALESCED, "hold_deferred")
    assert observed["prompts"] == []
    deferred = observed["after_emit"]
    assert deferred.status == RETRY
    assert deferred.attempts == 0
    assert deferred.last_error == "hold_deferred"


@pytest.mark.asyncio
async def test_emitting_dispatch_does_not_take_a_row_another_claimant_owns(tmp_path):
    """An unheld generic claim that took the reservation keeps it."""

    async def interloper(dispatcher, _hold, consumer_id):
        return await dispatcher.claim_durable_delivery(
            consumer_id=consumer_id, executor_id="generic-poller"
        )

    observed = await _race_with_parked_emitting_dispatch(
        tmp_path,
        name="transferred",
        max_attempts=1,
        interloper=interloper,
    )

    transferred = observed["interloper"]
    assert transferred is not None
    assert transferred.lease_owner == "generic-poller"
    assert observed["admission"] is DurableAdmissionDisposition.DUPLICATE
    assert observed["result"][0] is Status.COALESCED
    assert observed["prompts"] == []
    still_theirs = observed["after_emit"]
    assert still_theirs.status == LEASED
    assert still_theirs.lease_owner == "generic-poller"
    assert still_theirs.lease_token == transferred.lease_token
    assert still_theirs.attempts == 1


@pytest.mark.asyncio
async def test_hold_read_failure_at_exact_claim_precheck_returns_the_reservation(
    tmp_path,
):
    """Unknown Hold state defers the event's own reservation, then raises.

    Before #3283 the reservation stayed leased to the dispatcher until its
    first lease expired, after which its payload sidecar was discarded.
    """

    backend, agent, dispatcher = await _dispatcher(
        tmp_path / "precheck-read-fails.db",
        "did:agent:hold-claim-ownership:precheck-read-fails",
    )
    agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = DurableConsumerRegistration(
        consumer_id="workflow-worker",
        source="provider.message",
        agent_id=agent.did,
        max_attempts=1,
    )
    try:
        await dispatcher.register_durable_consumer(consumer)
        result = await dispatcher.dispatch_signal(
            _signal(agent_id=agent.did),
            source_event_id="provider:precheck-read-fails",
        )
        [activated] = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        handoff = dispatcher._transient_durable_handoffs[activated.delivery_id]
        assert activated.status == LEASED
        assert handoff.initial_lease_token is not None

        hold_store = _RaisingHoldRead()
        agent._hold_store = hold_store
        with pytest.raises(HoldEnforcementUnavailableError) as caught:
            await dispatcher.claim_durable_delivery_for_event(
                consumer_id=consumer.consumer_id,
                event_id=result.signal_id,
                executor_id="workflow-executor",
            )
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert hold_store.reads == 1

        [deferred] = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        assert deferred.status == RETRY
        assert deferred.attempts == 0
        assert deferred.last_error == "hold_deferred"
        assert deferred.lease_owner is None
        assert handoff.initial_lease_token is None
        assert handoff.expires_at == handoff.retention_until

        # Once Hold is readable the returned row is ordinary claimable work
        # that still carries its payload.
        claimed = await dispatcher.claim_durable_delivery_for_event(
            consumer_id=consumer.consumer_id,
            event_id=result.signal_id,
            executor_id="workflow-executor",
        )
        assert claimed is not None and claimed.attempts == 1
        assert claimed.event.payload["message"] == "hello"
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_terminal_ingress_under_hold_reports_held(tmp_path):
    """The terminal route publishes the cognition route's Hold disposition."""

    did = "did:agent:hold-claim-ownership:terminal"
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "terminal-held.db", did
    )
    consumer = DurableConsumerRegistration(
        consumer_id=DURABLE_TERMINAL_CONSUMER_ID,
        source="channel.message",
        agent_id=did,
        correlation_selector=(
            f"payload.{DURABLE_TERMINAL_MARKER}={DURABLE_TERMINAL_MARKER_VALUE}"
        ),
        max_attempts=0,
    )
    agent._hold_store = _HoldSnapshots(_held_state(did))
    signal = _channel_signal(did, "malformed")
    signal.payload.pop(DURABLE_COGNITION_MARKER)
    signal.payload[DURABLE_TERMINAL_MARKER] = DURABLE_TERMINAL_MARKER_VALUE
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_terminal(
            signal,
            source_event_id="telegram:update:malformed-held",
            consumer_id=consumer.consumer_id,
        )
        admission = await asyncio.wait_for(handle.wait_for_durable_admission(), timeout=2.0)
        result = await asyncio.wait_for(handle.wait(), timeout=2.0)
        assert admission.disposition is DurableAdmissionDisposition.HELD
        assert (result.status, result.error) == (Status.COALESCED, "hold_deferred")
        [deferred] = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        assert deferred.status == PENDING
        assert deferred.attempts == 0
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)
