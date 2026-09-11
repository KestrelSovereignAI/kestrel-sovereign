"""Verification-only reproductions for two #3163 review claims.

Claim #4: the late turn-start Hold branch of
``SignalDispatcher._route_durable_cognition_delivery`` supposedly never
resolves ``durable_admission`` and so falls through to NOT_ADMITTED (channel
RETRYABLE) instead of HELD. These tests record the disposition every
Hold-involved route actually publishes.

Claim #5: the generic ``claim_durable_delivery`` poll path releases every
volatile initial reservation for a consumer while Hold is active, including
one a concurrent emitting dispatch still owns. These tests pause a real
emitting dispatch inside that window and compare the unscoped release with a
release scoped away from the generic path.

No production code is changed; every observation comes from the running
dispatcher over a real SQLite ledger.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kestrel_sdk.signals import RateLimit, Status

from kestrel_sovereign.features.channels.feature import (
    ChannelFeature,
    InboundAdmissionDisposition,
)
from kestrel_sovereign.features.channels.models import (
    ChannelConfig,
    ChannelMessage,
    MessageDirection,
)
from kestrel_sovereign.hold import (
    EffectiveHoldState,
    HoldEnforcementUnavailableError,
    HoldTurnRefusal,
)
from kestrel_sovereign.privacy import get_privacy_preset
from kestrel_sovereign.signals import (
    ACKNOWLEDGED,
    LEASED,
    PENDING,
    RETRY,
    DurableAdmissionDisposition,
    DurableConsumerRegistration,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
)
from kestrel_sovereign.signals.registry import SourceRegistry
from kestrel_sovereign.signals.sources.channels import (
    DURABLE_COGNITION_CONSUMER_ID,
    DURABLE_COGNITION_MARKER,
    DURABLE_COGNITION_MARKER_VALUE,
    DURABLE_TERMINAL_CONSUMER_ID,
    DURABLE_TERMINAL_MARKER,
    DURABLE_TERMINAL_MARKER_VALUE,
)
from kestrel_sovereign.storage.db import SQLiteBackend

from tests.unit.test_channels_feature import StubAdapter, _cursor_owned_telegram
from tests.unit.test_durable_signal_delivery import (
    _channel_dispatcher,
    _channel_signal,
    _close,
    _held_state,
    _HoldSnapshots,
)

NOT_HELD = EffectiveHoldState(host=None, agent=None)
_DISPATCHER_SUFFIX = os.path.join("signals", "dispatcher.py")


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


def _dispatcher_frames() -> tuple[str, ...]:
    """Innermost-first names of the dispatcher frames performing this read."""

    names: list[str] = []
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename.endswith(_DISPATCHER_SUFFIX):
            names.append(frame.f_code.co_name)
        frame = frame.f_back
    return tuple(names)


class _SiteHold:
    """A Hold store whose answer depends on the dispatcher site reading it."""

    def __init__(self, decide) -> None:
        self._decide = decide
        self.reads: list[tuple[str, ...]] = []

    async def get_effective(self, _agent_id: str) -> EffectiveHoldState:
        frames = _dispatcher_frames()
        self.reads.append(frames)
        return self._decide(frames, len(self.reads))


_ROUTE_HELD_CHECK = (
    "_agent_is_held",
    "_held_signal_result",
    "_route_durable_cognition_delivery",
)


def _is_route_held_check(frames: tuple[str, ...]) -> bool:
    """Top check (first) or post-miss re-check (second) in the cognition route."""

    return frames[:3] == _ROUTE_HELD_CHECK


def _is_route_final_check(frames: tuple[str, ...]) -> bool:
    """The direct ``get_effective_hold_state`` call at the final turn boundary."""

    return bool(frames) and frames[0] == "_route_durable_cognition_delivery"


def _is_claim_precheck(frames: tuple[str, ...]) -> bool:
    return (
        frames[:2] == ("_agent_is_held", "_durable_claim_deferred_by_hold")
        and "claim_durable_delivery_for_event" in frames
        and "_fence_claimed_durable_delivery_after_hold_race" not in frames
    )


def _is_post_claim_fence(frames: tuple[str, ...]) -> bool:
    return "_fence_claimed_durable_delivery_after_hold_race" in frames


def _row(delivery) -> dict:
    return {
        "status": delivery.status,
        "attempts": delivery.attempts,
        "last_error": delivery.last_error,
    }


# ---------------------------------------------------------------------------
# Claim #4
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy", ("normal", "ephemeral"))
@pytest.mark.parametrize("refusal", ("hold_turn_refusal", "hold_state_unavailable"))
async def test_claim4_late_turn_start_hold_branch_admission_disposition(
    tmp_path, privacy, refusal
):
    """Adapted from test_late_hold_refusal_preserves_durable_retry_budget.

    ``process_input`` stands in for the universal turn-start gate. It raises
    after the durable lease transfer, so ``_route_durable_cognition_delivery``
    reaches its ``if self._hold_prevented_execution(result):`` branch. The
    test records whether the admission receipt was already resolved when the
    gate refused, and what it finally reports.
    """

    did = f"did:agent:c4-late-{privacy}-{refusal}"
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / f"c4-late-{privacy}-{refusal}.db",
        did,
        rate_limit=RateLimit(per_minute=1, per_hour=1, burst=1),
    )
    if privacy == "ephemeral":
        agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = _cognition_consumer(did, max_attempts=1)
    agent._hold_store = _HoldSnapshots(NOT_HELD)
    handle_box: dict = {}
    at_refusal: dict = {}

    async def refuse_at_turn_start(_prompt: str):
        receipt = handle_box["handle"].durable_admission
        at_refusal["admission_done"] = receipt.done()
        at_refusal["admission"] = (
            receipt.result().disposition.value if receipt.done() else None
        )
        if refusal == "hold_turn_refusal":
            raise HoldTurnRefusal(agent_id=did, effective_state=_held_state(did))
        raise HoldEnforcementUnavailableError("turn-start Hold read failed")

    agent.process_input = refuse_at_turn_start
    release_spy = AsyncMock(wraps=dispatcher._release_cognition_hold_lease)
    dispatcher._release_cognition_hold_lease = release_spy
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(did, "late-hold"),
            source_event_id="telegram:update:late-hold",
            consumer_id=consumer.consumer_id,
        )
        handle_box["handle"] = handle
        admission = await asyncio.wait_for(
            handle.wait_for_durable_admission(), timeout=2.0
        )
        result = await asyncio.wait_for(handle.wait(), timeout=2.0)
        [delivery] = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        observed = {
            "at_refusal": at_refusal,
            "admission": admission.disposition.value,
            "result": (result.status.value, result.error),
            "hold_prevented_execution": dispatcher._hold_prevented_execution(result),
            "release_calls_with_claimed_delivery": sum(
                1
                for call in release_spy.await_args_list
                if call.kwargs.get("delivery") is not None
            ),
            "row": _row(delivery),
        }
        print(f"\nCLAIM4 late-branch privacy={privacy} refusal={refusal}: {observed}")

        # The branch really ran: Hold prevented execution after the lease
        # transfer, and the exact claimed lease was returned attempt-neutrally.
        assert observed["hold_prevented_execution"] is True
        assert observed["release_calls_with_claimed_delivery"] == 1
        assert observed["row"] == {
            "status": RETRY,
            "attempts": 0,
            "last_error": "hold_deferred",
        }
        # The receipt was resolved COMMITTED before the gate refused, so the
        # dispatch_signal fallback never gets to write NOT_ADMITTED.
        assert at_refusal == {"admission_done": True, "admission": "committed"}
        assert admission.disposition is DurableAdmissionDisposition.COMMITTED
        if refusal == "hold_turn_refusal":
            assert observed["result"] == ("coalesced", "hold_deferred")
        else:
            assert observed["result"] == (
                "failed",
                "hold_state_unavailable: HoldEnforcementUnavailableError",
            )
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy", ("normal", "ephemeral"))
@pytest.mark.parametrize("outcome", ("read_fails", "held"))
async def test_claim4_final_admission_hold_check_disposition(tmp_path, privacy, outcome):
    """The final ``get_effective_hold_state`` check (hold_state_unavailable/held).

    Only the direct read at the route's final turn boundary fails or reports
    Hold; every earlier read (route top, claim pre-check, post-claim fence)
    reports no Hold, so the claim succeeds first.
    """

    did = f"did:agent:c4-final-{privacy}-{outcome}"
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / f"c4-final-{privacy}-{outcome}.db", did
    )
    if privacy == "ephemeral":
        agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = _cognition_consumer(did, max_attempts=1)

    def decide(frames, _count):
        if _is_route_final_check(frames):
            if outcome == "read_fails":
                raise RuntimeError("hold backend unavailable")
            return _held_state(did)
        return NOT_HELD

    hold = _SiteHold(decide)
    agent._hold_store = hold
    agent.process_input = AsyncMock(return_value="must not run")
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(did, "final-check"),
            source_event_id="telegram:update:final-check",
            consumer_id=consumer.consumer_id,
        )
        admission = await asyncio.wait_for(
            handle.wait_for_durable_admission(), timeout=2.0
        )
        result = await asyncio.wait_for(handle.wait(), timeout=2.0)
        [delivery] = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        observed = {
            "final_check_reads": sum(1 for f in hold.reads if _is_route_final_check(f)),
            "admission": admission.disposition.value,
            "result": (result.status.value, result.error),
            "turn_ran": agent.process_input.await_count,
            "row": _row(delivery),
        }
        print(f"\nCLAIM4 final-check privacy={privacy} outcome={outcome}: {observed}")

        assert observed["final_check_reads"] == 1
        assert observed["turn_ran"] == 0
        assert admission.disposition is DurableAdmissionDisposition.COMMITTED
        assert observed["row"] == {
            "status": RETRY,
            "attempts": 0,
            "last_error": "hold_deferred",
        }
        if outcome == "read_fails":
            assert observed["result"] == (
                "failed",
                "hold_state_unavailable: HoldEnforcementUnavailableError",
            )
        else:
            assert observed["result"] == ("coalesced", "hold_deferred")
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_claim4_plain_enqueue_hold_in_route_after_persistence(tmp_path):
    """``_route_after_durable_persistence`` Hold on the non-consumer path."""

    did = "did:agent:c4-plain"
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "c4-plain.db", did
    )
    agent._hold_store = _HoldSnapshots(_held_state(did))
    agent.process_input = AsyncMock(return_value="must not run")
    try:
        handle = await dispatcher.enqueue_signal(
            _channel_signal(did, "plain"), source_event_id="telegram:update:plain"
        )
        admission = await asyncio.wait_for(
            handle.wait_for_durable_admission(), timeout=2.0
        )
        result = await asyncio.wait_for(handle.task, timeout=2.0)
        observed = {
            "admission": admission.disposition.value,
            "result": (result.status.value, result.error),
            "turn_ran": agent.process_input.await_count,
        }
        print(f"\nCLAIM4 plain-enqueue: {observed}")
        assert admission.disposition is DurableAdmissionDisposition.COMMITTED
        assert observed["result"] == ("dropped_quiet_hours", "hold_skipped")
        assert observed["turn_ran"] == 0
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


_NOT_ADMITTED_CASES = {
    # Contrast: an outage at the route's FIRST Hold read maps to HELD.
    "route_top_check_read_fails": (
        lambda frames, n, state: _raise_once(state, _is_route_held_check(frames))
    ),
    # An outage at the claim's own Hold pre-check escapes as an exception.
    "claim_precheck_read_fails": (
        lambda frames, n, state: _raise_once(state, _is_claim_precheck(frames))
    ),
    # An outage at the post-claim fence escapes as an exception.
    "post_claim_fence_read_fails": (
        lambda frames, n, state: _raise_once(state, _is_post_claim_fence(frames))
    ),
    # Hold defers the claim, then a Release commits before the route re-check.
    "claim_precheck_held_then_released": (
        lambda frames, n, state: _held_once(state, _is_claim_precheck(frames))
    ),
    # The post-claim fence sees Hold, then a Release commits before re-check.
    "post_claim_fence_held_then_released": (
        lambda frames, n, state: _held_once(state, _is_post_claim_fence(frames))
    ),
}


def _raise_once(state: dict, matches: bool) -> EffectiveHoldState:
    if matches and not state.get("fired"):
        state["fired"] = True
        raise RuntimeError("hold backend unavailable")
    return NOT_HELD


def _held_once(state: dict, matches: bool) -> EffectiveHoldState:
    if matches and not state.get("fired"):
        state["fired"] = True
        return _held_state(state["did"])
    return NOT_HELD


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy", ("normal", "ephemeral"))
@pytest.mark.parametrize("case", tuple(_NOT_ADMITTED_CASES))
async def test_claim4_hold_involved_paths_admission_disposition(tmp_path, privacy, case):
    """Every other Hold read in the cognition route, one at a time."""

    did = f"did:agent:c4-paths-{privacy}-{case}"
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / f"c4-paths-{privacy}-{case}.db", did
    )
    if privacy == "ephemeral":
        agent.privacy_config = get_privacy_preset("ephemeral")
    consumer = _cognition_consumer(did, max_attempts=1)
    state = {"did": did}
    decide = _NOT_ADMITTED_CASES[case]
    hold = _SiteHold(lambda frames, n: decide(frames, n, state))
    agent._hold_store = hold
    agent.process_input = AsyncMock(return_value="ran")
    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            _channel_signal(did, "paths"),
            source_event_id="telegram:update:paths",
            consumer_id=consumer.consumer_id,
        )
        admission = await asyncio.wait_for(
            handle.wait_for_durable_admission(), timeout=2.0
        )
        result = await asyncio.wait_for(handle.wait(), timeout=2.0)
        [delivery] = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        observed = {
            "site_fired": state.get("fired", False),
            "admission": admission.disposition.value,
            "result": (result.status.value, result.error),
            "turn_ran": agent.process_input.await_count,
            "row": {
                **_row(delivery),
                "leased_by_dispatcher": delivery.lease_owner
                == dispatcher._durable_delivery_owner,
            },
        }
        print(f"\nCLAIM4 path privacy={privacy} case={case}: {observed}")
        assert observed["site_fired"] is True
        assert observed["turn_ran"] == 0
        expected = {
            "route_top_check_read_fails": "held",
            "claim_precheck_read_fails": "not_admitted",
            "post_claim_fence_read_fails": "not_admitted",
            "claim_precheck_held_then_released": "not_admitted",
            "post_claim_fence_held_then_released": "not_admitted",
        }[case]
        assert observed["admission"] == expected
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_claim4_terminal_ingress_under_hold_disposition(tmp_path):
    """``_route_durable_terminal_delivery`` returns a Hold result unresolved."""

    did = "did:agent:c4-terminal"
    backend, agent, dispatcher = await _channel_dispatcher(
        tmp_path / "c4-terminal.db", did
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
            source_event_id="telegram:update:malformed",
            consumer_id=consumer.consumer_id,
        )
        admission = await asyncio.wait_for(
            handle.wait_for_durable_admission(), timeout=2.0
        )
        result = await asyncio.wait_for(handle.wait(), timeout=2.0)
        [delivery] = await dispatcher.list_durable_deliveries(
            consumer_id=consumer.consumer_id
        )
        observed = {
            "admission": admission.disposition.value,
            "result": (result.status.value, result.error),
            "row": _row(delivery),
        }
        print(f"\nCLAIM4 terminal-ingress under Hold: {observed}")
        assert observed["admission"] == "not_admitted"
        assert observed["result"] == ("coalesced", "hold_deferred")
        assert observed["row"]["status"] == PENDING
        assert observed["row"]["attempts"] == 0
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


class _ChannelHost:
    """Minimal host agent for a real ChannelFeature over a real dispatcher."""

    def __init__(self, backend, did: str, privacy: str) -> None:
        self.did = did
        self.storage = SimpleNamespace(db=backend, agent_id=did)
        self.signal_registry = SourceRegistry()
        self._privacy_transition_lock = asyncio.Lock()
        self.dispatcher = None
        self.tasks: list[asyncio.Task] = []
        self.turns = 0
        self._hold_store = _HoldSnapshots(NOT_HELD)
        if privacy == "ephemeral":
            self.privacy_config = get_privacy_preset("ephemeral")

    def _get_privacy_transition_lock(self):
        return self._privacy_transition_lock

    def _track_background_task(self, coro, *, name):
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        return task

    async def process_input(self, _prompt):
        self.turns += 1
        if self.turns == 1:
            # The universal turn-start gate refuses after the lease transfer.
            raise HoldTurnRefusal(agent_id=self.did, effective_state=_held_state(self.did))
        return "resumed"


@pytest.mark.asyncio
@pytest.mark.parametrize("privacy", ("normal", "ephemeral"))
async def test_claim4_channel_inbound_disposition_for_late_hold_refusal(tmp_path, privacy):
    """What the real channel feature reports when the late branch fires."""

    backend = SQLiteBackend(str(tmp_path / f"c4-channel-{privacy}.db"))
    await backend.connect()
    host = _ChannelHost(backend, f"did:agent:c4-channel-{privacy}", privacy)
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    dispatcher = SignalDispatcher(
        agent=host,
        registry=host.signal_registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    host.dispatcher = dispatcher
    await dispatcher.initialize_durable_delivery()
    handles: list = []
    original_enqueue = dispatcher.enqueue_durable_cognition

    async def capture_enqueue(*args, **kwargs):
        handle = await original_enqueue(*args, **kwargs)
        handles.append(handle)
        return handle

    dispatcher.enqueue_durable_cognition = capture_enqueue
    feature = ChannelFeature(host)
    try:
        await feature.initialize()
        feature.registry.register(
            StubAdapter(
                channel="telegram",
                config=ChannelConfig(channel_type="telegram", allowed_senders=["555"]),
            )
        )
        admission = await feature.handle_inbound(
            _cursor_owned_telegram(
                ChannelMessage(
                    channel_type="telegram",
                    direction=MessageDirection.INBOUND,
                    sender="555",
                    recipient="bot",
                    content="late hold refusal",
                )
            )
        )
        [handle] = handles
        receipt = await asyncio.wait_for(handle.wait_for_durable_admission(), timeout=2.0)
        result = await asyncio.wait_for(handle.wait(), timeout=2.0)
        observed = {
            "inbound_admission": admission.disposition.value,
            "dispatcher_receipt": receipt.disposition.value,
            "emitting_result": (result.status.value, result.error),
            "first_turn_refused": host.turns >= 1,
        }
        print(f"\nCLAIM4 channel privacy={privacy}: {observed}")
        assert observed["first_turn_refused"] is True
        assert observed["emitting_result"] == ("coalesced", "hold_deferred")
        assert receipt.disposition is DurableAdmissionDisposition.COMMITTED
        if privacy == "normal":
            assert admission.disposition is InboundAdmissionDisposition.DURABLY_ADMITTED
        else:
            assert admission.disposition is InboundAdmissionDisposition.HELD
    finally:
        await dispatcher.shutdown_durable_delivery()
        if host.tasks:
            await asyncio.gather(*host.tasks, return_exceptions=True)
        await backend.close()


# ---------------------------------------------------------------------------
# Claim #5
# ---------------------------------------------------------------------------

_CONTENT = "the only copy of this private content"
_SOURCE_EVENT_ID = "telegram:update:volatile-race"


async def _volatile_race(
    tmp_path,
    *,
    arm: str,
    pause_at: str,
    scenario: str,
    generic_reclaims: bool = False,
    max_attempts: int = 0,
) -> dict:
    """Drive one volatile channel message through the claim #5 window.

    ``arm``: ``unscoped`` is production; ``scoped`` makes the generic poll
    path's no-event_id release a no-op (the proposed fix's effect).
    ``pause_at``: where the emitting dispatch is parked after activation and
    before its own exact-event transfer.
    ``scenario``: ``held_through_resume`` keeps Hold set when the emitting
    dispatch resumes; ``released_before_resume`` lifts it first.
    After the emitting dispatch finishes, Hold is lifted and the drain runs;
    then, if the row is still not ACKed, the provider redelivers the identical
    update (Telegram keeps its cursor for anything not durably admitted).
    """

    tag = f"{arm}-{pause_at}-{scenario}-{int(generic_reclaims)}-{max_attempts}"
    did = f"did:agent:c5-{tag}"
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / f"c5-{tag}.db", did)
    agent.privacy_config = get_privacy_preset("ephemeral")
    # Production registers the channel cognition consumer with max_attempts=0.
    consumer = _cognition_consumer(did, max_attempts=max_attempts)
    hold = _HoldSnapshots(NOT_HELD)
    agent._hold_store = hold

    routed: list[tuple] = []
    prompts: list[str] = []

    async def record_turn(prompt: str):
        prompts.append(prompt)
        return "ok"

    agent.process_input = record_turn
    original_route = dispatcher._route_after_durable_persistence

    async def record_route(signal, registration, start):
        routed.append((signal.caller, signal.payload.get("content")))
        return await original_route(signal, registration, start)

    dispatcher._route_after_durable_persistence = record_route

    generic_release_calls: list[str] = []
    original_release = dispatcher._release_initial_reservations_deferred_by_hold

    async def release(*, consumer_id: str, event_id: str | None = None) -> None:
        if event_id is None:
            generic_release_calls.append(consumer_id)
            if arm == "scoped":
                return
        await original_release(consumer_id=consumer_id, event_id=event_id)

    dispatcher._release_initial_reservations_deferred_by_hold = release

    signal = _channel_signal(did, "volatile-race")
    signal.payload["content"] = _CONTENT
    paused = asyncio.Event()
    resume = asyncio.Event()
    obs: dict = {}

    if pause_at == "before_route_hold_check":
        original_held = dispatcher._held_signal_result
        calls = {"n": 0}

        async def pausing_held(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                paused.set()
                await resume.wait()
            return await original_held(*args, **kwargs)

        dispatcher._held_signal_result = pausing_held
    elif pause_at == "after_exact_store_claim_missed":
        store = dispatcher._durable_store
        original_store_claim = store.claim_delivery_for_event
        parked = {"done": False}

        async def pausing_store_claim(**kwargs):
            claimed = await original_store_claim(**kwargs)
            if not parked["done"] and kwargs.get("event_id") == signal.id:
                parked["done"] = True
                obs["emitting_store_claim"] = None if claimed is None else claimed.status
                paused.set()
                await resume.wait()
            return claimed

        store.claim_delivery_for_event = pausing_store_claim
    else:
        raise AssertionError(pause_at)

    async def the_row():
        [row] = await dispatcher.list_durable_deliveries(consumer_id=consumer.consumer_id)
        handoff = dispatcher._transient_durable_handoffs.get(row.delivery_id)
        return row, {
            **_row(row),
            "leased_by_dispatcher": row.lease_owner == dispatcher._durable_delivery_owner,
            "sidecar": handoff is not None,
            "initial_token": handoff is not None and handoff.initial_lease_token is not None,
        }

    try:
        await dispatcher.register_durable_consumer(consumer)
        handle = await dispatcher.enqueue_durable_cognition(
            signal, source_event_id=_SOURCE_EVENT_ID, consumer_id=consumer.consumer_id
        )
        await asyncio.wait_for(paused.wait(), timeout=2.0)
        _, obs["at_pause"] = await the_row()

        # Hold arrives while the emitting dispatch is parked; a generic poller
        # asks the dispatcher for work on the same consumer.
        hold.snapshots[:] = [_held_state(did)]
        generic = await dispatcher.claim_durable_delivery(
            consumer_id=consumer.consumer_id, executor_id="generic-poller"
        )
        obs["generic_claim_under_hold"] = generic
        obs["generic_release_calls"] = len(generic_release_calls)
        _, obs["after_generic_claim"] = await the_row()

        if scenario == "released_before_resume":
            hold.snapshots[:] = [NOT_HELD]
            if generic_reclaims:
                again = await dispatcher.claim_durable_delivery(
                    consumer_id=consumer.consumer_id, executor_id="generic-poller"
                )
                obs["generic_reclaim_after_release"] = (
                    None
                    if again is None
                    else {
                        "lease_owner": again.lease_owner,
                        "content": again.event.payload.get("content"),
                        "caller_identity": again.event.caller_identity,
                    }
                )
        elif scenario != "held_through_resume":
            raise AssertionError(scenario)

        resume.set()
        admission = await asyncio.wait_for(handle.wait_for_durable_admission(), timeout=2.0)
        result = await asyncio.wait_for(handle.wait(), timeout=2.0)
        obs["emit_admission"] = admission.disposition.value
        obs["emit_result"] = (result.status.value, result.error)
        _, obs["after_emit"] = await the_row()
        obs["routed_after_emit"] = list(routed)

        # Release (if still held) and let the dispatcher-owned drain run.
        hold.snapshots[:] = [NOT_HELD]
        dispatcher._start_durable_cognition_drain(consumer.consumer_id)
        drainer = dispatcher._durable_cognition_drainers.get(consumer.consumer_id)
        if drainer is not None:
            await asyncio.wait_for(drainer, timeout=2.0)
        row, obs["after_drain"] = await the_row()
        obs["routed_after_drain"] = list(routed)

        # Telegram retains its cursor for anything not durably admitted and
        # redelivers the identical update once the row is due again.
        if row.status != ACKNOWLEDGED:
            for timer in dispatcher._durable_cognition_drain_timers.values():
                timer.cancel()
            dispatcher._durable_cognition_drain_timers.clear()
            await backend.execute(
                "UPDATE durable_signal_deliveries SET next_attempt_at = ? "
                "WHERE delivery_id = ?",
                (
                    dispatcher._durable_store.to_timestamp_param(
                        datetime.now(timezone.utc) - timedelta(seconds=1)
                    ),
                    row.delivery_id,
                ),
            )
            dispatcher._coalescing.reset()
            redelivered = _channel_signal(did, "volatile-race")
            redelivered.payload["content"] = _CONTENT
            again_handle = await dispatcher.enqueue_durable_cognition(
                redelivered,
                source_event_id=_SOURCE_EVENT_ID,
                consumer_id=consumer.consumer_id,
            )
            redelivery_admission = await asyncio.wait_for(
                again_handle.wait_for_durable_admission(), timeout=2.0
            )
            redelivery_result = await asyncio.wait_for(again_handle.wait(), timeout=2.0)
            obs["redelivery_admission"] = redelivery_admission.disposition.value
            obs["redelivery_result"] = (
                redelivery_result.status.value,
                redelivery_result.error,
            )
            _, obs["after_redelivery"] = await the_row()
        obs["routed_final"] = list(routed)
        obs["turns"] = len(prompts)
        obs["turn_prompts_carry_content"] = [_CONTENT in prompt for prompt in prompts]
    finally:
        resume.set()
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)
    return obs


_ARM_SPECIFIC = ("generic_release_calls", "after_generic_claim")


def _outcome(obs: dict) -> dict:
    return {key: value for key, value in obs.items() if key not in _ARM_SPECIFIC}


def _print_arms(label: str, unscoped: dict, scoped: dict) -> None:
    print(f"\nCLAIM5 {label}")
    for key in sorted(set(unscoped) | set(scoped)):
        u = unscoped.get(key, "<absent>")
        s = scoped.get(key, "<absent>")
        marker = "  " if u == s else "!="
        print(f"  {marker} {key}:\n       unscoped={u}\n       scoped  ={s}")


def _assert_race_window_was_hit(unscoped: dict, scoped: dict) -> None:
    # At the pause the emitting dispatch owns an activated initial lease.
    for obs in (unscoped, scoped):
        assert obs["at_pause"]["status"] == LEASED
        assert obs["at_pause"]["leased_by_dispatcher"] is True
        assert obs["at_pause"]["initial_token"] is True
        assert obs["generic_claim_under_hold"] is None
        assert obs["generic_release_calls"] == 1
    # Production's unscoped release took the emitting dispatch's reservation.
    assert unscoped["after_generic_claim"] == {
        "status": RETRY,
        "attempts": 0,
        "last_error": "hold_deferred",
        "leased_by_dispatcher": False,
        "sidecar": True,
        "initial_token": False,
    }
    # The scoped arm left it with the emitting dispatch.
    assert scoped["after_generic_claim"]["status"] == LEASED
    assert scoped["after_generic_claim"]["initial_token"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pause_at", ("before_route_hold_check", "after_exact_store_claim_missed")
)
async def test_claim5_hold_stays_set_when_emitting_dispatch_resumes(tmp_path, pause_at):
    unscoped = await _volatile_race(
        tmp_path, arm="unscoped", pause_at=pause_at, scenario="held_through_resume"
    )
    scoped = await _volatile_race(
        tmp_path, arm="scoped", pause_at=pause_at, scenario="held_through_resume"
    )
    _print_arms(f"held_through_resume pause_at={pause_at}", unscoped, scoped)
    _assert_race_window_was_hit(unscoped, scoped)
    assert _outcome(unscoped) == _outcome(scoped)
    # The emitting dispatch defers in both arms; its live envelope is dropped
    # as Hold requires, and the provider redelivery is what executes it.
    assert unscoped["emit_admission"] == "held"
    assert unscoped["emit_result"] == ("coalesced", "hold_deferred")
    assert unscoped["routed_after_emit"] == []
    assert unscoped["routed_final"] == [("555", _CONTENT)]
    assert unscoped["turns"] == 1
    assert unscoped["turn_prompts_carry_content"] == [True]
    assert unscoped["after_redelivery"]["status"] == ACKNOWLEDGED


@pytest.mark.asyncio
async def test_claim5_hold_released_before_resume_parked_before_route_hold_check(tmp_path):
    kwargs = dict(pause_at="before_route_hold_check", scenario="released_before_resume")
    unscoped = await _volatile_race(tmp_path, arm="unscoped", **kwargs)
    scoped = await _volatile_race(tmp_path, arm="scoped", **kwargs)
    _print_arms("released_before_resume pause_at=before_route_hold_check", unscoped, scoped)
    _assert_race_window_was_hit(unscoped, scoped)
    assert _outcome(unscoped) == _outcome(scoped)
    assert unscoped["emit_admission"] == "committed"
    assert unscoped["emit_result"][0] == "ok"
    assert unscoped["routed_final"] == [("555", _CONTENT)]
    assert unscoped["turns"] == 1
    assert unscoped["after_emit"]["status"] == ACKNOWLEDGED


@pytest.mark.asyncio
async def test_claim5_hold_released_before_resume_parked_after_exact_store_claim(tmp_path):
    kwargs = dict(
        pause_at="after_exact_store_claim_missed", scenario="released_before_resume"
    )
    unscoped = await _volatile_race(tmp_path, arm="unscoped", **kwargs)
    scoped = await _volatile_race(tmp_path, arm="scoped", **kwargs)
    _print_arms(
        "released_before_resume pause_at=after_exact_store_claim_missed", unscoped, scoped
    )
    _assert_race_window_was_hit(unscoped, scoped)
    # The emitting dispatch's exact store claim missed because the row was its
    # own activated initial lease; it is parked before the transient transfer.
    assert unscoped["emitting_store_claim"] is None
    assert scoped["emitting_store_claim"] is None

    # Scoped: the emitting dispatch transfers its reservation and executes.
    assert scoped["emit_admission"] == "committed"
    assert scoped["emit_result"] == ("ok", None)
    assert scoped["routed_after_emit"] == [("555", _CONTENT)]
    assert scoped["after_emit"]["status"] == ACKNOWLEDGED
    assert scoped["after_emit"]["attempts"] == 1
    assert "redelivery_result" not in scoped

    # Unscoped (production): the reservation is gone when the emitting
    # dispatch looks for it, Hold no longer applies at its re-check, so it
    # fails the first delivery and drops its live envelope.
    assert unscoped["emit_admission"] == "not_admitted"
    assert unscoped["emit_result"] == (
        "failed",
        "Durable cognition delivery is unavailable; source must retry "
        "without advancing its cursor",
    )
    assert unscoped["routed_after_emit"] == []
    assert unscoped["after_emit"] == {
        "status": RETRY,
        "attempts": 0,
        "last_error": "hold_deferred",
        "leased_by_dispatcher": False,
        "sidecar": True,
        "initial_token": False,
    }
    # The drain cannot execute the elided row without its caller: one attempt
    # is spent and nothing runs.
    assert unscoped["after_drain"]["status"] == RETRY
    assert unscoped["after_drain"]["attempts"] == 1
    assert unscoped["after_drain"]["last_error"] == (
        "Durable cognition caller recovery failed"
    )
    assert unscoped["routed_after_drain"] == []
    # Only the provider's redelivery of the identical update executes it,
    # once, with the right caller and content (max_attempts=0 as production).
    assert unscoped["redelivery_admission"] == "duplicate"
    assert unscoped["redelivery_result"] == ("ok", None)
    assert unscoped["after_redelivery"]["status"] == ACKNOWLEDGED
    assert unscoped["after_redelivery"]["attempts"] == 2
    assert unscoped["routed_final"] == scoped["routed_final"] == [("555", _CONTENT)]
    assert unscoped["turns"] == scoped["turns"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("held_through_resume", "released_before_resume"))
async def test_claim5_bounded_consumer_parked_after_exact_store_claim(tmp_path, scenario):
    """Non-production max_attempts=1: can the divergence become a loss?"""

    kwargs = dict(
        pause_at="after_exact_store_claim_missed", scenario=scenario, max_attempts=1
    )
    unscoped = await _volatile_race(tmp_path, arm="unscoped", **kwargs)
    scoped = await _volatile_race(tmp_path, arm="scoped", **kwargs)
    _print_arms(f"max_attempts=1 {scenario} after_exact_store_claim_missed", unscoped, scoped)
    _assert_race_window_was_hit(unscoped, scoped)
    if scenario == "held_through_resume":
        # Baseline, identical in both arms: the post-Release drain cannot
        # recover an elided caller, spends the only attempt, and the row is
        # FAILED before the provider's redelivery arrives.
        assert _outcome(unscoped) == _outcome(scoped)
        assert unscoped["after_drain"]["status"] == "failed"
        assert unscoped["turns"] == 0
    else:
        # Unscoped: first delivery fails, the drain burns the only attempt,
        # the redelivery is refused, and the message never executes.
        assert unscoped["emit_admission"] == "not_admitted"
        assert unscoped["after_drain"]["status"] == "failed"
        assert unscoped["redelivery_admission"] == "not_admitted"
        assert unscoped["turns"] == 0
        assert unscoped["routed_final"] == []
        # Scoped: executed once on first delivery with caller and content.
        assert scoped["emit_admission"] == "committed"
        assert scoped["after_emit"]["status"] == ACKNOWLEDGED
        assert scoped["routed_final"] == [("555", _CONTENT)]
        assert scoped["turns"] == 1


@pytest.mark.asyncio
async def test_claim5_generic_poller_reclaims_after_release_before_resume(tmp_path):
    """Control: the NON-Hold generic path also takes the emitting reservation."""

    kwargs = dict(
        pause_at="before_route_hold_check",
        scenario="released_before_resume",
        generic_reclaims=True,
    )
    unscoped = await _volatile_race(tmp_path, arm="unscoped", **kwargs)
    scoped = await _volatile_race(tmp_path, arm="scoped", **kwargs)
    _print_arms("generic poller reclaims after Release", unscoped, scoped)
    _assert_race_window_was_hit(unscoped, scoped)
    assert _outcome(unscoped) == _outcome(scoped)
    assert unscoped["generic_reclaim_after_release"] == {
        "lease_owner": "generic-poller",
        "content": _CONTENT,
        "caller_identity": None,
    }
