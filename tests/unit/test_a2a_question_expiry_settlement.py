"""Sender-side question expiry is propagated to the recipient (#3440).

Observed 2026-10-02: Emma's ``send_a2a_question`` to Nellie expired on
Emma's side, but Nellie's task stayed ``submitted`` and was re-delivered to
Nellie two hours later. A later question was answered by Nellie after Emma's
expiry and Emma only learned of it by polling; another was already answered
when Emma's wake said ``expired`` with an empty reply.

Pins:

  - Settlement outcomes: the creator's signed cancellation withdraws a live
    task; a lifecycle conflict (the recipient answered first) yields the
    recipient's own state and reply; an unreachable recipient is
    unconfirmed. A cancellation of our own whose response was lost still
    reads as a withdrawal, never as the recipient declining.
  - Acceptance, with real sender stores and a real recipient TaskManager
    behind the host-attested local route:
      * the asker's expiry leaves the recipient's task terminal, and neither
        boot reconciliation nor a stale queued wake re-delivers it;
      * a recipient that answered before the expiry settled wakes the asker
        with that answer, not ``expired``;
      * a recipient unreachable at expiry that answers later wakes the asker
        again, flagged ``late``, without a manual poll — exactly once;
      * that late answer is held on the sender's row until its wake is
        delivered, so a crash after it is learned and before it is relayed
        loses nothing, and a concurrent pass cannot relay it twice;
      * so is the asker's first wake, whichever it is (the answer the expiry
        found, the ``expired`` of a withdrawn or unreached question, or the
        supervisor's on-time answer): a crash before it is delivered delays
        it until the dead process's claim lapses, never loses it, and no
        late answer overtakes an ``expired`` still owed;
      * a pass that stalls past its lease and is overtaken by another can
        no longer renew, confirm, or relay: each claim fences out the last;
      * questions that expired on a release before settlement existed are
        settled after the upgrade: withdrawn, or their late answer relayed;
      * settlement has no age cutoff: a recipient returning a month later,
        with its submission wake still retained in its durable ledger, has
        that submission withdrawn before it can run;
      * a recipient that hangs or cannot be reached costs a settlement pass
        one attempt, not one per question, so the recipients behind its
        backlog are still settled in that pass (#3586).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from kestrel_sdk.signals import SignalMode, SignalResult, Status
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.a2a.outbound_store import (
    ensure_a2a_outbound_tasks_table,
    get_outbound_task,
    record_outbound_dispatch,
    update_outbound_terminal_state,
)
from kestrel_sovereign.a2a.task_manager import create_task_manager
from kestrel_sovereign.a2a.types import Message, TaskSendParams, TaskState, TextPart
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.features.peers.directory import (
    PeerIdentity,
    PeerNotFoundError,
    PeerProtocolError,
    PeerRequester,
    PeerTransportError,
    PeerUnavailableError,
)
from kestrel_sovereign.features.peers.feature import (
    RECIPIENT_UNREACHABLE_ERROR_TYPE,
    PeersFeature,
    ReceiverSettlement,
    ReceiverSettlementOutcome,
)
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.hold import EffectiveHoldState
from kestrel_sovereign.signals import (
    PENDING,
    RETRY,
    DurableAdmissionDisposition,
    DurableConsumerRegistration,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals.sources.a2a import (
    build_a2a_task_complete_registration,
)
from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    DURABLE_COGNITION_CONSUMER_ID as SUBMITTED_CONSUMER,
    SOURCE_NAME as SUBMITTED_SOURCE,
    build_a2a_task_submitted_registration,
    build_signal_for_submitted_task,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.async_pending_a2a_question_store import (
    QUESTION_CLAIM_LEASE,
    RECEIVER_SETTLEMENT_CONFIRMED,
    RECEIVER_SETTLEMENT_LATE_ANSWER,
    RECEIVER_SETTLEMENT_PENDING,
    PendingA2AQuestionStore,
)
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
from tests.unit.test_durable_drain_replayable import _drain_once
from tests.unit.test_durable_signal_delivery import _held_state, _HoldSnapshots

SENDER = "did:test:emma"
RECIPIENT = "did:test:nellie"
PEER = PeerIdentity(agent_id=RECIPIENT, slug="nellie", routing_key="nellie")
KITE = "did:test:kite"
KITE_PEER = PeerIdentity(agent_id=KITE, slug="kite", routing_key="kite")


class _DeliveredHandle:
    async def wait(self) -> SignalResult:
        return SignalResult(
            signal_id="sig-question-answered",
            status=Status.OK,
            mode=SignalMode.COGNITION,
            duration_ms=1,
        )


class _UndeliveredHandle:
    async def wait(self) -> SignalResult:
        return SignalResult(
            signal_id="sig-question-answered",
            status=Status.FAILED,
            mode=SignalMode.COGNITION,
            duration_ms=1,
            error="turn failed",
        )


class _BlockedHandle:
    """A wake whose cognition turn is still running until ``release``."""

    def __init__(self):
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> SignalResult:
        self.waiting.set()
        await self.release.wait()
        return await _DeliveredHandle().wait()


class _ProcessDied(BaseException):
    """The sender's process ends at this point: nothing after it runs."""


class _DiesInFlight:
    """A wake whose cognition turn is under way when the sender's process
    ends: neither its delivery nor its failure is ever observed."""

    async def wait(self) -> SignalResult:
        raise _ProcessDied()


class _SenderDispatcher:
    """Records every ``a2a.question_answered`` wake the asker receives.

    ``die_on_enqueue`` ends the process as the next wake is enqueued, before
    the dispatcher accepts it."""

    def __init__(self):
        self.signals: list = []
        self.next_handles: list = []
        self.die_on_enqueue = False

    async def enqueue_signal(self, signal):
        if self.die_on_enqueue:
            raise _ProcessDied()
        self.signals.append(signal)
        if self.next_handles:
            return self.next_handles.pop(0)
        return _DeliveredHandle()


# ---------------------------------------------------------------------------
# Settlement outcomes
# ---------------------------------------------------------------------------


def _settling_feature(*, task_read=None, read_error=None) -> PeersFeature:
    feature = PeersFeature.__new__(PeersFeature)
    feature._own_name = "Emma"
    feature.agent = SimpleNamespace(did=SENDER)
    router = SimpleNamespace(
        get_a2a_task=AsyncMock(return_value=task_read, side_effect=read_error),
    )
    feature._question_recipient_agent_id = AsyncMock(return_value=RECIPIENT)
    feature._resolve_retained_automatic_peer = AsyncMock(
        return_value=(router, PeerRequester(SENDER, object()), PEER),
    )
    feature.router = router
    return feature


async def _settle(feature) -> ReceiverSettlement:
    return await feature._settle_question_with_receiver(
        task_id="q-1", recipient="Nellie", recipient_agent_id=RECIPIENT,
    )


def _conflict() -> ToolResult:
    return ToolResult.failed(
        "Task cancellation conflicts with the recipient's terminal state",
        data={"task_id": "q-1", "error_type": "lifecycle_conflict"},
    )


def _unreachable() -> ToolResult:
    """What ``cancel_outbound_task`` returns when its request failed in
    transport."""
    return ToolResult.failed(
        "Could not reach the task recipient",
        data={"task_id": "q-1", "error_type": RECIPIENT_UNREACHABLE_ERROR_TYPE},
    )


class TestSettlementOutcomes:
    @pytest.mark.asyncio
    async def test_committed_cancellation_withdraws_without_a_read(self):
        feature = _settling_feature()
        feature.cancel_outbound_task = AsyncMock(
            return_value=ToolResult.ok(
                confirmation="Cancelled", data={"status": "canceled"},
            ),
        )

        settlement = await _settle(feature)

        assert settlement.outcome is ReceiverSettlementOutcome.WITHDRAWN
        assert settlement.receiver_terminal
        feature.cancel_outbound_task.assert_awaited_once_with(
            "q-1", reason=PeersFeature.QUESTION_EXPIRY_CANCEL_REASON,
        )
        feature.router.get_a2a_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancellation_with_failed_audit_stamp_still_withdraws(self):
        """PARTIAL: the recipient's task is canceled; only our audit row
        missed its stamp."""
        feature = _settling_feature()
        feature.cancel_outbound_task = AsyncMock(
            return_value=ToolResult.partial(
                confirmation="Cancelled", error="audit row not updated",
                data={"status": "canceled"},
            ),
        )

        settlement = await _settle(feature)

        assert settlement.outcome is ReceiverSettlementOutcome.WITHDRAWN

    @pytest.mark.asyncio
    async def test_conflict_carries_the_recipients_answer(self):
        feature = _settling_feature(
            task_read={
                "id": "q-1", "status": "completed", "message": "42",
                "artifacts": [], "metadata": {},
            },
        )
        feature.cancel_outbound_task = AsyncMock(return_value=_conflict())

        settlement = await _settle(feature)

        assert settlement == ReceiverSettlement(
            ReceiverSettlementOutcome.ANSWERED, state="completed", reply_text="42",
        )

    @pytest.mark.asyncio
    async def test_live_task_behind_a_failed_cancellation_is_unconfirmed(self):
        feature = _settling_feature(
            task_read={"id": "q-1", "status": {"state": "working"}},
        )
        feature.cancel_outbound_task = AsyncMock(return_value=_unreachable())

        settlement = await _settle(feature)

        assert settlement.outcome is ReceiverSettlementOutcome.UNCONFIRMED
        assert not settlement.receiver_terminal
        # The read reached the recipient, which answered: still working.
        assert not settlement.recipient_unresponsive

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "read_error",
        [
            PeerTransportError("Could not reach local peer host"),
            PeerUnavailableError("Peer is unavailable"),
        ],
        ids=["transport", "unavailable"],
    )
    async def test_unreachable_recipient_is_unconfirmed_and_unresponsive(
        self, read_error,
    ):
        """The cancellation and the read both failed in transport, so the
        recipient's other questions would fail the same way: the sweep is
        told to leave them for its next pass (#3586)."""
        feature = _settling_feature(read_error=read_error)
        feature.cancel_outbound_task = AsyncMock(return_value=_unreachable())

        settlement = await _settle(feature)

        assert settlement.outcome is ReceiverSettlementOutcome.UNCONFIRMED
        assert settlement.recipient_unresponsive

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("withdrawal", "task_read", "read_error"),
        [
            pytest.param(
                _conflict(), None,
                PeerTransportError("Could not reach local peer host"),
                id="cancellation-conflicted",
            ),
            pytest.param(
                ToolResult.failed(
                    "Peer is not available in the automatic directory",
                    data={"task_id": "q-1"},
                ),
                None,
                PeerTransportError("Could not reach local peer host"),
                id="cancellation-refused",
            ),
            pytest.param(
                None, None,
                PeerTransportError("Could not reach local peer host"),
                id="no-outbound-route",
            ),
            pytest.param(
                _unreachable(), None, PeerNotFoundError("gone"),
                id="read-refused",
            ),
            pytest.param(
                _unreachable(), None, PeerProtocolError("malformed"),
                id="read-malformed",
            ),
            pytest.param(
                _unreachable(), None, RuntimeError("provider bug"),
                id="read-provider-fault",
            ),
            pytest.param(
                _unreachable(), {"id": "q-1", "status": "submitted"}, None,
                id="read-live-task",
            ),
        ],
    )
    async def test_recipient_reached_by_either_request_is_not_unresponsive(
        self, withdrawal, task_read, read_error,
    ):
        """A recipient is deferred only when both requests failed in
        transport. When either request got a decision, or failed for any
        other reason, the settlement stays an ordinary unconfirmed one and
        the sweep goes on to the recipient's next question (#3586)."""
        feature = _settling_feature(task_read=task_read, read_error=read_error)
        feature.cancel_outbound_task = AsyncMock(return_value=withdrawal)

        settlement = await _settle(feature)

        assert settlement.outcome is ReceiverSettlementOutcome.UNCONFIRMED
        assert not settlement.recipient_unresponsive

    @pytest.mark.asyncio
    async def test_own_cancellation_with_lost_response_is_a_withdrawal(self):
        """Our cancellation committed but its response was lost. The task
        reads canceled under OUR receipt: that is the withdrawal, not the
        recipient declining the question."""
        feature = _settling_feature(
            task_read={
                "id": "q-1", "status": "canceled",
                "message": "Task canceled by did:test:emma: withdrawn",
                "metadata": {
                    "cancellation_receipt": {
                        "actor_agent_id": SENDER,
                        "reason": "withdrawn",
                        "status_before": "submitted",
                    },
                },
            },
        )
        feature.cancel_outbound_task = AsyncMock(return_value=_unreachable())

        settlement = await _settle(feature)

        assert settlement.outcome is ReceiverSettlementOutcome.WITHDRAWN

    @pytest.mark.asyncio
    async def test_recipient_decline_is_the_recipients_answer(self):
        feature = _settling_feature(
            task_read={
                "id": "q-1", "status": "canceled",
                "message": "Task canceled by did:test:nellie: out of scope",
                "metadata": {
                    "cancellation_receipt": {
                        "actor_agent_id": RECIPIENT,
                        "reason": "out of scope",
                        "status_before": "working",
                    },
                },
            },
        )
        feature.cancel_outbound_task = AsyncMock(return_value=_conflict())

        settlement = await _settle(feature)

        assert settlement.outcome is ReceiverSettlementOutcome.ANSWERED
        assert settlement.state == "canceled"
        assert "out of scope" in settlement.reply_text

    @pytest.mark.asyncio
    async def test_settlement_is_bounded(self, monkeypatch):
        """The deadline wake waits for settlement, so a hung recipient must
        not hold it: past the bound the recipient is unconfirmed."""
        feature = _settling_feature()

        async def _hangs(*_args, **_kwargs):
            await asyncio.Event().wait()
        feature.cancel_outbound_task = _hangs
        monkeypatch.setattr(
            PeersFeature, "QUESTION_SETTLEMENT_TIMEOUT_SECONDS", 0.01,
        )

        settlement = await asyncio.wait_for(_settle(feature), timeout=30)

        assert settlement.outcome is ReceiverSettlementOutcome.UNCONFIRMED


class TestExpiryWakeFollowsSettlement:
    """Which wake the asker gets, and how the row is claimed."""

    async def _expire(self, settlement: ReceiverSettlement):
        feature = PeersFeature.__new__(PeersFeature)
        feature._own_name = "Emma"
        dispatcher = _SenderDispatcher()
        store = MagicMock()
        store.mark_expired = AsyncMock(return_value=True)
        store.mark_resolved = AsyncMock(return_value=True)
        store.renew_claim_lease = AsyncMock(return_value=True)
        store.confirm_wake_delivered = AsyncMock(return_value=True)
        feature.agent = SimpleNamespace(
            did=SENDER, dispatcher=dispatcher, pending_a2a_questions=store,
        )
        feature._settle_question_with_receiver = AsyncMock(return_value=settlement)
        await feature._expire_pending_question(
            store,
            task_id="q-1",
            recipient="Nellie",
            recipient_agent_id=RECIPIENT,
            original_question="How many?",
            sess_id="s",
            causation_chain=None,
        )
        return store, dispatcher

    @pytest.mark.asyncio
    async def test_answer_that_beat_the_deadline_is_delivered_not_expired(self):
        """The 2026-10-02 23:47Z case: the recipient had completed with a
        1022-char reply, and the asker's wake said ``expired`` with an empty
        body."""
        store, dispatcher = await self._expire(
            ReceiverSettlement(
                ReceiverSettlementOutcome.ANSWERED,
                state="completed",
                reply_text="the real answer",
            )
        )

        store.mark_resolved.assert_awaited_once_with(
            "q-1", state="completed", reply_text="the real answer", claim=ANY,
        )
        store.mark_expired.assert_not_awaited()
        [signal] = dispatcher.signals
        assert signal.payload["state"] == "completed"
        assert signal.payload["reply_text"] == "the real answer"
        assert signal.payload["late"] is False
        assert signal.payload["withdrawn"] is False
        # Delivered, the answer is retired under the claim that resolved it.
        claim = store.mark_resolved.await_args.kwargs["claim"]
        store.confirm_wake_delivered.assert_awaited_once_with("q-1", claim=claim)

    @pytest.mark.asyncio
    async def test_withdrawn_question_expires_with_the_recipient_settled(self):
        store, dispatcher = await self._expire(
            ReceiverSettlement(ReceiverSettlementOutcome.WITHDRAWN)
        )

        store.mark_expired.assert_awaited_once_with(
            "q-1", claim=ANY, receiver_settled=True,
        )
        [signal] = dispatcher.signals
        assert signal.payload["state"] == "expired"
        assert signal.payload["reply_text"] == ""
        assert signal.payload["withdrawn"] is True
        claim = store.mark_expired.await_args.kwargs["claim"]
        store.confirm_wake_delivered.assert_awaited_once_with("q-1", claim=claim)

    @pytest.mark.asyncio
    async def test_unreached_recipient_still_wakes_the_asker_at_the_deadline(self):
        store, dispatcher = await self._expire(
            ReceiverSettlement(ReceiverSettlementOutcome.UNCONFIRMED)
        )

        store.mark_expired.assert_awaited_once_with(
            "q-1", claim=ANY, receiver_settled=False,
        )
        [signal] = dispatcher.signals
        assert signal.payload["state"] == "expired"
        # The recipient's task may still be live: not reported withdrawn.
        assert signal.payload["withdrawn"] is False
        claim = store.mark_expired.await_args.kwargs["claim"]
        store.confirm_wake_delivered.assert_awaited_once_with("q-1", claim=claim)

    @pytest.mark.asyncio
    async def test_failed_claim_write_still_wakes_the_asker_unfenced(self):
        """A claim whose write raised may or may not have taken, so it cannot
        vouch for this pass: the wake fires anyway, without the ownership
        check or a retirement under that claim."""
        feature = PeersFeature.__new__(PeersFeature)
        feature._own_name = "Emma"
        dispatcher = _SenderDispatcher()
        store = MagicMock()
        store.mark_expired = AsyncMock(side_effect=RuntimeError("DB blip"))
        store.renew_claim_lease = AsyncMock(return_value=False)
        store.confirm_wake_delivered = AsyncMock(return_value=False)
        feature.agent = SimpleNamespace(
            did=SENDER, dispatcher=dispatcher, pending_a2a_questions=store,
        )
        feature._settle_question_with_receiver = AsyncMock(
            return_value=ReceiverSettlement(ReceiverSettlementOutcome.UNCONFIRMED),
        )

        await feature._expire_pending_question(
            store,
            task_id="q-1",
            recipient="Nellie",
            recipient_agent_id=RECIPIENT,
            original_question="How many?",
            sess_id="s",
            causation_chain=None,
        )

        [signal] = dispatcher.signals
        assert signal.payload["state"] == "expired"
        store.renew_claim_lease.assert_not_awaited()
        store.confirm_wake_delivered.assert_not_awaited()


# ---------------------------------------------------------------------------
# Acceptance: real sender stores, real recipient TaskManager
# ---------------------------------------------------------------------------


class _HostAttestedRouter:
    """The sender's peer router over the in-process host's attested route.

    Mirrors ``server.py``'s ``local_cancel`` / ``local_get`` wiring of the
    local peer directory: the recipient's own ``AgentManager`` admission and
    TaskManager decide every cancellation and read. ``reachable=False``
    models a recipient host that is down.
    """

    def __init__(self, manager: AgentManager, sender):
        self._manager = manager
        self._sender = sender
        self.reachable = True

    def _require_reachable(self) -> None:
        if not self.reachable:
            raise PeerTransportError("Could not reach local peer host")

    async def cancel_a2a_task(self, requester, peer, task_id, payload):
        self._require_reachable()
        return await self._manager.cancel_host_attested_local_a2a_task(
            sender=self._sender,
            requester=requester,
            peer=peer,
            task_id=task_id,
            payload=payload,
        )

    async def get_a2a_task(self, requester, peer, task_id):
        self._require_reachable()
        return await self._manager.get_host_attested_local_a2a_task(
            sender=self._sender,
            requester=requester,
            peer=peer,
            task_id=task_id,
        )


class _RecipientDispatcher:
    """Records the cognition wakes boot reconciliation re-admits."""

    def __init__(self):
        self.enqueued: list = []

    async def enqueue_durable_cognition(self, signal, *, source_event_id, consumer_id):
        self.enqueued.append(signal)
        return SimpleNamespace(
            wait_for_durable_admission=AsyncMock(
                return_value=SimpleNamespace(
                    disposition=DurableAdmissionDisposition.COMMITTED,
                ),
            ),
        )

    async def get_durable_delivery_for_source_event(self, **_key):
        return None


class _Recipient(EventManagerMixin):
    """Nellie: the real wake reconciliation and execution gate over her
    real task store."""

    did = RECIPIENT

    def __init__(self, task_manager):
        self.task_manager = task_manager
        self.dispatcher = _RecipientDispatcher()
        self.signal_registry = SourceRegistry()
        self.signal_registry.register(build_a2a_task_submitted_registration())
        self.signal_registry.register(build_a2a_task_complete_registration())


def _sender_feature(db, router, requester, store) -> PeersFeature:
    """Emma's PeersFeature as boot leaves it, over her real stores."""
    feature = PeersFeature.__new__(PeersFeature)
    feature._own_name = "Emma"
    feature._db = db
    feature._outbound_route_store_ready = True
    feature.agent = SimpleNamespace(
        did=SENDER,
        identity=None,
        dispatcher=_SenderDispatcher(),
        pending_a2a_questions=store,
    )
    feature._resolve_retained_automatic_peer = AsyncMock(
        return_value=(router, requester, PEER),
    )
    return feature


class _Fleet:
    def __init__(
        self, *, db, db_path, task_manager, store, feature, router, requester,
        recipient,
    ):
        self.db = db
        self.db_path = db_path
        self.task_manager = task_manager
        self.store = store
        self.feature = feature
        self.router = router
        self.requester = requester
        self.recipient = recipient
        self.recipient_tasks = TaskFeature(SimpleNamespace(did=RECIPIENT))
        self.recipient_tasks.set_task_manager(task_manager)
        # The settlement passes' clock. Every pass runs a lease later than
        # the previous one, as the hourly sweep does.
        self.clock = datetime.now(timezone.utc)

    async def settle(self) -> None:
        """One settlement pass, the sweep's (and boot's)."""
        self.clock += QUESTION_CLAIM_LEASE + timedelta(seconds=1)
        await self.feature._settle_expired_questions_with_receivers(
            self.store, now=self.clock,
        )

    async def recover(self) -> None:
        """The hourly sweep's first step, a lease after the last pass: wakes
        whose claim lapsed before their delivery was confirmed are
        delivered again."""
        self.clock += QUESTION_CLAIM_LEASE + timedelta(seconds=1)
        await self.feature._redeliver_abandoned_wakes(self.store, now=self.clock)

    async def restart_sender(self) -> None:
        """Emma's process is replaced: fresh connection, store, and feature
        over the same database file. Nothing in memory survives."""
        self.db = await AsyncDatabase.sqlite(self.db_path)
        self.store = PendingA2AQuestionStore(self.db, agent_id=SENDER)
        self.feature = _sender_feature(
            self.db, self.router, self.requester, self.store,
        )

    @property
    def wakes(self) -> list:
        return self.feature.agent.dispatcher.signals

    async def ask(
        self,
        task_id: str,
        question: str,
        *,
        recipient: str = "Nellie",
        recipient_agent_id: str = RECIPIENT,
        recipient_tasks=None,
    ) -> None:
        """What ``send_a2a_question`` leaves behind, with a lapsed deadline.

        Asked of Nellie unless another recipient and its task manager are
        named."""
        await (recipient_tasks or self.task_manager).create_task(
            TaskSendParams(
                id=task_id,
                sessionId=f"session-{task_id}",
                message=Message(role="user", parts=[TextPart(text=question)]),
                metadata={"a2a_verb": "question", "reply_expected": True},
            ),
            agent_name=recipient_agent_id,
            creator_agent_id=SENDER,
        )
        await record_outbound_dispatch(
            self.db,
            agent_id=SENDER,
            task_id=task_id,
            recipient=recipient,
            recipient_agent_id=recipient_agent_id,
            verb="question",
            session_id=f"session-{task_id}",
            dispatch_tool="send_a2a_question",
        )
        await self.store.insert(
            task_id=task_id,
            recipient=recipient,
            recipient_agent_id=recipient_agent_id,
            original_question=question,
            origin_turn_id=None,
            origin_session_id=f"session-{task_id}",
            deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

    async def expire(self, task_id: str) -> None:
        """The hourly sweep's (and boot replay's) expiry of this row."""
        [row] = [
            row for row in await self.store.list_waiting_past_deadline()
            if row.task_id == task_id
        ]
        await self.feature._handle_expired_row(self.store, row)

    async def answer(self, task_id: str, text: str) -> ToolResult:
        return await self.recipient_tasks.respond_to_a2a_task(task_id, text)

    async def recipient_state(self, task_id: str) -> TaskState:
        task = await self.task_manager.get_task_for_recipient(task_id, RECIPIENT)
        return task.status.state


@pytest.fixture
async def fleet(tmp_path):
    db_path = str(tmp_path / "emma.db")
    db = await AsyncDatabase.sqlite(db_path)
    await ensure_a2a_outbound_tasks_table(db)
    task_manager = await create_task_manager(str(tmp_path / "nellie-tasks.db"))
    manager = AgentManager()
    sender = SimpleNamespace(did=SENDER, agent_id=SENDER, identity=None)
    recipient_agent = SimpleNamespace(
        did=RECIPIENT, agent_id=RECIPIENT, task_manager=task_manager,
    )
    manager._register_agent("emma", sender)
    manager._register_agent("nellie", recipient_agent)
    requester = PeerRequester(SENDER, object())
    manager.install_a2a_hosted_policy(
        sender,
        resolver=object(),
        authorizer=object(),
        router=SimpleNamespace(),
        requester=requester,
    )
    manager.install_a2a_hosted_policy(
        recipient_agent,
        resolver=object(),
        authorizer=object(),
        router=SimpleNamespace(authorize_inbound_sender=AsyncMock(return_value=True)),
        requester=PeerRequester(RECIPIENT, object()),
    )
    router = _HostAttestedRouter(manager, sender)

    store = PendingA2AQuestionStore(db, agent_id=SENDER)
    fleet = _Fleet(
        db=db,
        db_path=db_path,
        task_manager=task_manager,
        store=store,
        feature=_sender_feature(db, router, requester, store),
        router=router,
        requester=requester,
        recipient=_Recipient(task_manager),
    )
    try:
        yield fleet
    finally:
        await task_manager.close()
        if fleet.db is not db:
            await fleet.db.close()
        await db.close()


@pytest.mark.asyncio
async def test_expiry_terminates_the_recipients_task_and_it_is_never_redelivered(
    fleet,
):
    """The 2026-10-02 19:38Z case: Emma's question expired, Nellie's task
    stayed ``submitted``, and it was re-delivered to Nellie at 21:37Z."""
    await fleet.ask("q-expired", "What are the live numbers?")

    await fleet.expire("q-expired")

    [wake] = fleet.wakes
    assert wake.payload["state"] == "expired"
    assert wake.payload["withdrawn"] is True
    assert await fleet.recipient_state("q-expired") is TaskState.CANCELED
    row = await fleet.store.get("q-expired")
    assert row.status == "EXPIRED"
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED
    outbound = await get_outbound_task(fleet.db, agent_id=SENDER, task_id="q-expired")
    assert outbound.terminal_state == "canceled"

    # No re-delivery afterwards. Boot reconciliation re-admits a submission
    # wake only for a LIVE task...
    await fleet.recipient.reconcile_a2a_cognition_wakes()
    assert [
        signal.source for signal in fleet.recipient.dispatcher.enqueued
    ] == ["a2a.task_complete"]
    # ...and a submission wake still queued from before the expiry is refused
    # at execution.
    task = await fleet.task_manager.get_task_for_recipient("q-expired", RECIPIENT)
    stale = build_signal_for_submitted_task(task, target_agent=RECIPIENT, sender=SENDER)
    assert stale.source == SUBMITTED_SOURCE
    refusal = await fleet.recipient.validate_cognition_signal_execution(stale)
    assert refusal is not None and "canceled" in refusal
    # The recipient cannot answer a question the asker has given up on.
    late = await fleet.answer("q-expired", "here you go")
    assert late.status is ToolResultStatus.ERROR

    # A second sweep owes nothing: no further wake, no settlement left.
    await fleet.settle()
    assert len(fleet.wakes) == 1


@pytest.mark.asyncio
async def test_answer_before_expiry_settles_reaches_the_asker(fleet):
    """The recipient's answer won the race with the expiry: the asker is
    woken with it, not with an empty ``expired``."""
    await fleet.ask("q-raced", "How many?")
    answered = await fleet.answer("q-raced", "1022 characters of real answer")
    assert answered.status is ToolResultStatus.OK

    await fleet.expire("q-raced")

    [wake] = fleet.wakes
    assert wake.payload["state"] == "completed"
    assert wake.payload["reply_text"] == "1022 characters of real answer"
    assert wake.payload["late"] is False
    assert await fleet.recipient_state("q-raced") is TaskState.COMPLETED
    row = await fleet.store.get("q-raced")
    assert row.status == "RESOLVED"
    assert row.receiver_settlement is None
    outbound = await get_outbound_task(fleet.db, agent_id=SENDER, task_id="q-raced")
    assert outbound.terminal_state == "completed"


@pytest.mark.asyncio
async def test_late_answer_from_unreached_recipient_wakes_the_asker(fleet):
    """The ``ea48db1e`` case: the expiry could not reach the recipient, who
    answered afterwards. The asker learns of it from a wake, not a poll."""
    await fleet.ask("q-late", "Live numbers?")
    fleet.router.reachable = False

    await fleet.expire("q-late")

    [expired] = fleet.wakes
    assert expired.payload["state"] == "expired"
    assert expired.payload["withdrawn"] is False
    row = await fleet.store.get("q-late")
    assert row.status == "EXPIRED"
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_PENDING
    assert await fleet.recipient_state("q-late") is TaskState.SUBMITTED

    # The recipient comes back and answers anyway.
    answered = await fleet.answer("q-late", "valid live numbers")
    assert answered.status is ToolResultStatus.OK
    fleet.router.reachable = True

    await fleet.settle()

    assert len(fleet.wakes) == 2
    late = fleet.wakes[1]
    assert late.payload["state"] == "completed"
    assert late.payload["reply_text"] == "valid live numbers"
    assert late.payload["late"] is True
    assert late.dedupe_key == "q-late:answered:late"
    row = await fleet.store.get("q-late")
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED
    outbound = await get_outbound_task(fleet.db, agent_id=SENDER, task_id="q-late")
    assert outbound.terminal_state == "completed"

    # Exactly once.
    await fleet.settle()
    assert len(fleet.wakes) == 2


@pytest.mark.asyncio
async def test_unreached_recipient_is_withdrawn_once_reachable(fleet):
    """An unreached recipient that never answered is withdrawn by the next
    settlement pass, silently: the asker already holds its ``expired``."""
    await fleet.ask("q-down", "Anyone there?")
    fleet.router.reachable = False
    await fleet.expire("q-down")
    fleet.router.reachable = True

    await fleet.settle()

    assert len(fleet.wakes) == 1
    assert await fleet.recipient_state("q-down") is TaskState.CANCELED
    row = await fleet.store.get("q-down")
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED


@pytest.mark.asyncio
async def test_undelivered_late_answer_is_retried(fleet):
    await fleet.ask("q-retry", "Numbers?")
    fleet.router.reachable = False
    await fleet.expire("q-retry")
    await fleet.answer("q-retry", "the numbers")
    fleet.router.reachable = True
    fleet.feature.agent.dispatcher.next_handles.append(_UndeliveredHandle())

    await fleet.settle()

    assert fleet.wakes[-1].payload["late"] is True
    row = await fleet.store.get("q-retry")
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_LATE_ANSWER
    assert row.receiver_late_reply_text == "the numbers"

    # The retry relays the answer held on the row; it does not need the
    # recipient again.
    fleet.router.reachable = False
    await fleet.settle()

    assert len(fleet.wakes) == 3
    assert fleet.wakes[-1].payload["reply_text"] == "the numbers"
    assert fleet.wakes[-1].payload["late"] is True
    row = await fleet.store.get("q-retry")
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED


async def _late_answer_after_unreached_expiry(fleet, task_id: str) -> None:
    await fleet.ask(task_id, "Live numbers?")
    fleet.router.reachable = False
    await fleet.expire(task_id)
    answered = await fleet.answer(task_id, "valid live numbers")
    assert answered.status is ToolResultStatus.OK
    fleet.router.reachable = True


@pytest.mark.asyncio
async def test_late_answer_survives_a_crash_before_its_wake_is_enqueued(fleet):
    """The settlement pass learns the recipient's answer, then the process
    dies before the late wake is enqueued. The answer must already be the
    sender's durable obligation: the restarted process relays it."""
    await _late_answer_after_unreached_expiry(fleet, "q-crash")
    fleet.feature.agent.dispatcher.die_on_enqueue = True

    with pytest.raises(_ProcessDied):
        await fleet.settle()

    assert len(fleet.wakes) == 1  # only the original ``expired``
    row = await fleet.store.get("q-crash")
    assert row.status == "EXPIRED"
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_LATE_ANSWER
    assert (row.receiver_late_state, row.receiver_late_reply_text) == (
        "completed", "valid live numbers",
    )

    await fleet.restart_sender()
    # The dead pass's lease still holds: an immediate pass does not relay.
    await fleet.feature._settle_expired_questions_with_receivers(
        fleet.store, now=fleet.clock,
    )
    assert fleet.wakes == []
    # Once it lapses, the restarted process relays the held answer, even
    # with the recipient unreachable again.
    fleet.router.reachable = False
    await fleet.settle()

    [late] = fleet.wakes
    assert late.payload["state"] == "completed"
    assert late.payload["reply_text"] == "valid live numbers"
    assert late.payload["late"] is True
    row = await fleet.store.get("q-crash")
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED

    await fleet.settle()
    assert len(fleet.wakes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("delay_inflight_commit_ack", [False, True])
async def test_crash_while_the_late_wake_is_in_flight_is_relayed_again(
    fleet, monkeypatch, delay_inflight_commit_ack,
):
    """The late wake was enqueued but its turn never finished: the row is not
    confirmed, so the restarted process relays it again. A duplicate is the
    price of never losing it."""
    await _late_answer_after_unreached_expiry(fleet, "q-inflight")
    in_flight = _BlockedHandle()
    fleet.feature.agent.dispatcher.next_handles.append(in_flight)
    renewed = asyncio.Event()
    commit_ack_pending = asyncio.Event()
    release_commit_ack = asyncio.Event()
    native_renew = fleet.store.renew_claim_lease
    connection = fleet.db.backend._connection
    native_commit = connection.commit

    async def observed_commit():
        await native_commit()
        if delay_inflight_commit_ack and in_flight.waiting.is_set():
            # The real native commit happened, but its ACK is still withheld.
            # Pre-enqueue renewal must not release the crash barrier here.
            commit_ack_pending.set()
            await release_commit_ack.wait()

    monkeypatch.setattr(connection, "commit", observed_commit)

    async def observed_renewal(*args, **kwargs):
        result = await native_renew(*args, **kwargs)
        if result and in_flight.waiting.is_set():
            renewed.set()
        return result

    monkeypatch.setattr(fleet.store, "renew_claim_lease", observed_renewal)

    first_pass = asyncio.create_task(fleet.settle())
    try:
        async with asyncio.timeout(30):
            await in_flight.waiting.wait()
            if delay_inflight_commit_ack:
                await commit_ack_pending.wait()
                assert not renewed.is_set(), "pre-enqueue renewal released the in-flight barrier"
                release_commit_ack.set()
            # Crash during the remote turn, after its actual SQLite renewal
            # ACK, not during COMMIT (which must retain UNKNOWN if interrupted).
            await renewed.wait()
        first_pass.cancel()  # the process dies mid-turn
        with pytest.raises(asyncio.CancelledError):
            await first_pass
    finally:
        from kestrel_sovereign._async_ownership import await_owned_task

        release_commit_ack.set()
        if not first_pass.done():
            first_pass.cancel()
        await await_owned_task(first_pass)
    row = await fleet.store.get("q-inflight")
    assert row.status == "EXPIRED"
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_LATE_ANSWER

    await fleet.restart_sender()
    await fleet.settle()

    [relayed] = fleet.wakes
    assert relayed.payload["reply_text"] == "valid live numbers"
    assert (await fleet.store.get("q-inflight")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )


# ---------------------------------------------------------------------------
# The asker's first wake survives a crash before it is delivered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_found_at_expiry_survives_a_crash_before_its_wake(fleet):
    """The expiry finds the recipient already answered and resolves the row,
    then the process dies before the answer's wake is enqueued. The answer
    is still owed on the row, and once the dead process's claim lapses the
    restarted process delivers it: on time, not late (#3440)."""
    await fleet.ask("q-answered-crash", "How many?")
    await fleet.answer("q-answered-crash", "1022 characters of real answer")
    fleet.feature.agent.dispatcher.die_on_enqueue = True

    with pytest.raises(_ProcessDied):
        await fleet.expire("q-answered-crash")

    assert fleet.wakes == []
    row = await fleet.store.get("q-answered-crash")
    assert row.status == "RESOLVED"
    assert (row.retry_state, row.retry_reply_text) == (
        "completed", "1022 characters of real answer",
    )

    await fleet.restart_sender()
    # The dead process's claim may still have the wake in flight: neither
    # boot replay nor a settlement pass delivers it yet.
    await fleet.feature._replay_pending_a2a_questions(fleet.store)
    await fleet.settle()
    assert fleet.wakes == []

    await fleet.recover()

    [wake] = fleet.wakes
    assert wake.payload["state"] == "completed"
    assert wake.payload["reply_text"] == "1022 characters of real answer"
    assert wake.payload["late"] is False
    row = await fleet.store.get("q-answered-crash")
    assert row.status == "RESOLVED"
    assert row.retry_state is None

    await fleet.recover()
    await fleet.settle()
    assert len(fleet.wakes) == 1


@pytest.mark.asyncio
async def test_withdrawal_wake_survives_a_crash_before_it_is_enqueued(fleet):
    """The expiry withdraws the question and confirms the recipient settled,
    then the process dies before the ``expired`` wake is enqueued. Nothing is
    left to settle, but the asker is still owed that wake."""
    await fleet.ask("q-withdrawn-crash", "Numbers?")
    fleet.feature.agent.dispatcher.die_on_enqueue = True

    with pytest.raises(_ProcessDied):
        await fleet.expire("q-withdrawn-crash")

    assert fleet.wakes == []
    assert await fleet.recipient_state("q-withdrawn-crash") is TaskState.CANCELED
    row = await fleet.store.get("q-withdrawn-crash")
    assert row.status == "EXPIRED"
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED
    assert row.retry_state == "expired"

    await fleet.restart_sender()
    await fleet.recover()

    [wake] = fleet.wakes
    assert wake.payload["state"] == "expired"
    assert wake.payload["reply_text"] == ""
    assert wake.payload["late"] is False
    assert wake.payload["withdrawn"] is True
    row = await fleet.store.get("q-withdrawn-crash")
    assert row.status == "EXPIRED"
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED
    assert row.retry_state is None
    assert await fleet.recipient_state("q-withdrawn-crash") is TaskState.CANCELED

    await fleet.recover()
    await fleet.settle()
    assert len(fleet.wakes) == 1


@pytest.mark.asyncio
async def test_unsettled_expiry_wake_survives_a_crash_ahead_of_the_late_answer(
    fleet,
):
    """The expiry cannot reach the recipient and the process dies before its
    ``expired`` wake is enqueued. The recipient answers meanwhile. The asker
    must hear ``expired`` first and the answer second, flagged late: no
    settlement pass relays the answer while the ``expired`` wake is owed."""
    await fleet.ask("q-unsettled-crash", "Numbers?")
    fleet.router.reachable = False
    fleet.feature.agent.dispatcher.die_on_enqueue = True

    with pytest.raises(_ProcessDied):
        await fleet.expire("q-unsettled-crash")

    answered = await fleet.answer("q-unsettled-crash", "the numbers")
    assert answered.status is ToolResultStatus.OK
    fleet.router.reachable = True
    await fleet.restart_sender()
    # The dead expiry's claim has lapsed, yet settlement leaves the row alone.
    await fleet.settle()
    assert fleet.wakes == []
    assert (await fleet.store.get("q-unsettled-crash")).receiver_settlement == (
        RECEIVER_SETTLEMENT_PENDING
    )

    await fleet.recover()
    await fleet.settle()

    assert [(w.payload["state"], w.payload["late"]) for w in fleet.wakes] == [
        ("expired", False), ("completed", True),
    ]
    # Recovered from the row, the ``expired`` wake still says unconfirmed.
    assert fleet.wakes[0].payload["withdrawn"] is False
    assert fleet.wakes[1].payload["reply_text"] == "the numbers"
    row = await fleet.store.get("q-unsettled-crash")
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED
    assert row.retry_state is None


@pytest.mark.asyncio
async def test_supervisor_answer_survives_a_crash_before_its_wake(fleet):
    """The supervisor's own on-time answer is claimed the same way: a crash
    between resolving the row and enqueueing the answer's wake delays the
    answer, and does not lose it."""
    await fleet.ask("q-on-time", "How many?")

    async def _answered(*_args, **_kwargs):
        yield SimpleNamespace(
            event="status",
            data=json.dumps({
                "id": "q-on-time",
                "status": {
                    "state": "completed",
                    "message": {"parts": [{"text": "42"}]},
                },
            }),
        )
    fleet.router.subscribe_a2a_task = _answered
    fleet.feature.agent.dispatcher.die_on_enqueue = True

    with pytest.raises(_ProcessDied):
        await fleet.feature._supervise_a2a_question(
            task_id="q-on-time",
            recipient="Nellie",
            recipient_agent_id=RECIPIENT,
            original_question="How many?",
            sess_id="session-q-on-time",
            deadline_utc=datetime.now(timezone.utc) + timedelta(minutes=5),
            causation_chain=None,
        )

    assert (await fleet.store.get("q-on-time")).retry_state == "completed"
    await fleet.restart_sender()
    await fleet.recover()

    [wake] = fleet.wakes
    assert (wake.payload["state"], wake.payload["reply_text"]) == ("completed", "42")
    assert wake.payload["late"] is False
    assert (await fleet.store.get("q-on-time")).retry_state is None


@pytest.mark.asyncio
async def test_crash_while_an_answer_wake_is_in_flight_delivers_it_again(fleet):
    """The answer's wake was enqueued and its turn was running when the
    process died, so its delivery was never confirmed. The restarted process
    delivers it again. A duplicate is the price of never losing it."""
    await fleet.ask("q-in-flight", "How many?")
    await fleet.answer("q-in-flight", "42")
    fleet.feature.agent.dispatcher.next_handles.append(_DiesInFlight())

    with pytest.raises(_ProcessDied):
        await fleet.expire("q-in-flight")

    [first] = fleet.wakes
    assert first.payload["reply_text"] == "42"
    assert (await fleet.store.get("q-in-flight")).retry_state == "completed"

    await fleet.restart_sender()
    await fleet.recover()

    [again] = fleet.wakes
    assert (again.payload["state"], again.payload["reply_text"]) == (
        "completed", "42",
    )
    assert (await fleet.store.get("q-in-flight")).retry_state is None


@pytest.mark.asyncio
async def test_concurrent_pass_does_not_relay_a_late_answer_twice(fleet):
    """Boot's settlement pass and the hourly sweep can overlap while a late
    wake's turn is running. The lease keeps the second off the row."""
    await _late_answer_after_unreached_expiry(fleet, "q-race")
    in_flight = _BlockedHandle()
    fleet.feature.agent.dispatcher.next_handles.append(in_flight)

    first_pass = asyncio.create_task(fleet.settle())
    await asyncio.wait_for(in_flight.waiting.wait(), timeout=30)
    await fleet.feature._settle_expired_questions_with_receivers(
        fleet.store, now=fleet.clock,
    )
    in_flight.release.set()
    await asyncio.wait_for(first_pass, timeout=30)

    late = [w for w in fleet.wakes if w.payload["late"]]
    assert len(late) == 1
    assert (await fleet.store.get("q-race")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )


@pytest.mark.asyncio
async def test_late_relay_answers_to_the_store_that_claimed_its_row(fleet):
    """The relay's ownership check and lease renewals go to the store whose
    claim it holds, not to whichever store the agent exposes when it
    relays."""
    await _late_answer_after_unreached_expiry(fleet, "q-own-store")
    fleet.feature.agent.pending_a2a_questions = None

    await fleet.settle()

    [expired, late] = fleet.wakes
    assert (expired.payload["state"], late.payload["late"]) == ("expired", True)
    assert (await fleet.store.get("q-own-store")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )

@pytest.mark.asyncio
async def test_pass_overtaken_while_stalled_neither_relays_nor_confirms(fleet):
    """Replica A learns the recipient's late answer, records it, and stalls
    before relaying it. Its lease lapses, and replica B claims the row and
    starts relaying the held answer. When A resumes it must not renew the
    lease, confirm the row, or wake the asker a second time."""
    await _late_answer_after_unreached_expiry(fleet, "q-takeover")
    store_a, feature_a = fleet.store, fleet.feature
    claims_a: list = []
    renewals_a: list = []
    stalled, resume = asyncio.Event(), asyncio.Event()
    real_claim = store_a.claim_receiver_settlement
    real_record = store_a.record_late_answer
    real_renew = store_a.renew_claim_lease

    async def _claim(task_id, **kwargs):
        claim = await real_claim(task_id, **kwargs)
        claims_a.append(claim)
        return claim

    async def _record_then_stall(task_id, **kwargs):
        recorded = await real_record(task_id, **kwargs)
        stalled.set()
        await resume.wait()
        return recorded

    async def _renew(task_id, **kwargs):
        renewed = await real_renew(task_id, **kwargs)
        renewals_a.append(renewed)
        return renewed
    store_a.claim_receiver_settlement = _claim
    store_a.record_late_answer = _record_then_stall
    store_a.renew_claim_lease = _renew

    pass_a = asyncio.create_task(fleet.settle())
    await asyncio.wait_for(stalled.wait(), timeout=30)
    # Replica B over the same database, a lease later.
    await fleet.restart_sender()
    in_flight = _BlockedHandle()
    fleet.feature.agent.dispatcher.next_handles.append(in_flight)
    pass_b = asyncio.create_task(fleet.settle())
    await asyncio.wait_for(in_flight.waiting.wait(), timeout=30)

    resume.set()
    await asyncio.wait_for(pass_a, timeout=30)

    assert [w.payload["state"] for w in feature_a.agent.dispatcher.signals] == [
        "expired",
    ]
    assert renewals_a == [False]
    [claim_a] = claims_a
    assert not await store_a.confirm_late_answer_delivered(
        "q-takeover", claim=claim_a.token,
    )
    assert (await fleet.store.get("q-takeover")).receiver_settlement == (
        RECEIVER_SETTLEMENT_LATE_ANSWER
    )

    in_flight.release.set()
    await asyncio.wait_for(pass_b, timeout=30)
    [late] = fleet.wakes
    assert late.payload["late"] is True
    assert late.payload["reply_text"] == "valid live numbers"
    assert (await fleet.store.get("q-takeover")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["answered", "unreached"])
async def test_wake_overtaken_while_stalled_is_never_enqueued(fleet, outcome):
    """Replica A's expiry claims the row to wake the asker (with the answer
    it found, or ``expired``) and stalls before enqueueing. Its lease lapses,
    and replica B takes the wake over and is delivering it. When A resumes it
    must not wake the asker too, nor pull the row back from under B."""
    task_id = f"q-stalled-{outcome}"
    await fleet.ask(task_id, "How many?")
    if outcome == "answered":
        await fleet.answer(task_id, "42")
        claim_method, expected = "mark_resolved", ("completed", "42")
    else:
        fleet.router.reachable = False
        claim_method, expected = "mark_expired", ("expired", "")
    store_a, feature_a = fleet.store, fleet.feature
    stalled, resume = asyncio.Event(), asyncio.Event()
    real_claim = getattr(store_a, claim_method)

    async def _claim_then_stall(*args, **kwargs):
        claimed = await real_claim(*args, **kwargs)
        stalled.set()
        await resume.wait()
        return claimed
    setattr(store_a, claim_method, _claim_then_stall)

    pass_a = asyncio.create_task(fleet.expire(task_id))
    await asyncio.wait_for(stalled.wait(), timeout=30)
    # Replica B over the same database, once A's lease has lapsed.
    await fleet.restart_sender()
    in_flight = _BlockedHandle()
    fleet.feature.agent.dispatcher.next_handles.append(in_flight)
    pass_b = asyncio.create_task(fleet.recover())
    await asyncio.wait_for(in_flight.waiting.wait(), timeout=30)

    resume.set()
    await asyncio.wait_for(pass_a, timeout=30)

    assert feature_a.agent.dispatcher.signals == []
    row = await fleet.store.get(task_id)
    assert row.status in ("RESOLVED", "EXPIRED")
    assert (row.retry_state, row.retry_reply_text or "") == expected

    in_flight.release.set()
    await asyncio.wait_for(pass_b, timeout=30)
    [wake] = fleet.wakes
    assert (wake.payload["state"], wake.payload["reply_text"]) == expected
    assert (await fleet.store.get(task_id)).retry_state is None


@pytest.mark.asyncio
async def test_no_late_answer_is_relayed_while_the_expired_wake_is_in_flight(
    fleet,
):
    """The expiry that left a row unsettled holds its lease: the asker hears
    ``expired`` before any late answer, never after it."""
    await fleet.ask("q-order", "Numbers?")
    fleet.router.reachable = False
    await fleet.expire("q-order")
    await fleet.answer("q-order", "the numbers")
    fleet.router.reachable = True

    # A pass inside the expiry's lease leaves the row alone.
    await fleet.feature._settle_expired_questions_with_receivers(fleet.store)

    assert [w.payload["state"] for w in fleet.wakes] == ["expired"]
    assert (await fleet.store.get("q-order")).receiver_settlement == (
        RECEIVER_SETTLEMENT_PENDING
    )


@pytest.mark.asyncio
async def test_unresponsive_recipient_costs_one_attempt_per_pass(
    fleet, monkeypatch,
):
    """With no age cutoff, every unsettled row is retried each pass. A
    recipient that hangs must not cost the settlement bound once per
    question, or one dead peer would stall the sweep."""
    for task_id in ("q-hang-1", "q-hang-2", "q-hang-3"):
        await fleet.ask(task_id, "Anyone?")
        fleet.router.reachable = False
        await fleet.expire(task_id)
    fleet.router.reachable = True
    attempts = []

    async def _hangs(task_id, *, reason):
        attempts.append(task_id)
        await asyncio.Event().wait()
    fleet.feature.cancel_outbound_task = _hangs
    monkeypatch.setattr(
        PeersFeature, "QUESTION_SETTLEMENT_TIMEOUT_SECONDS", 0.01,
    )

    await fleet.settle()

    assert len(attempts) == 1
    for task_id in ("q-hang-1", "q-hang-2", "q-hang-3"):
        row = await fleet.store.get(task_id)
        assert row.receiver_settlement == RECEIVER_SETTLEMENT_PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_ack", [False, True])
async def test_rows_behind_a_hung_row_are_reached_on_the_next_pass(
    fleet, monkeypatch, delayed_ack,
):
    """One row whose settlement hangs skips its recipient's other rows for
    that pass only: the next pass starts with them. The stuck row sorts first
    by expiry and by id, so only the lease order can put its sibling ahead."""
    for task_id in ("q-a-stuck", "q-b-fine"):
        await fleet.ask(task_id, "Anyone?")
        fleet.router.reachable = False
        await fleet.expire(task_id)
    fleet.router.reachable = True
    real_cancel = fleet.feature.cancel_outbound_task
    attempts = []
    hung_timeout = 0.05

    async def _hangs_for_one(task_id, *, reason):
        attempts.append(task_id)
        if task_id == "q-a-stuck":
            await asyncio.Event().wait()
        result = await real_cancel(task_id, reason=reason)
        if delayed_ack:
            # The receiver already committed. Delayed acknowledgement must
            # not make fairness depend on its round-trip fitting in 50 ms.
            await asyncio.sleep(2 * hung_timeout)
        return result
    fleet.feature.cancel_outbound_task = _hangs_for_one
    real_settle = fleet.feature._settle_question_with_receiver

    async def _short_deadline_only_for_hung_row(**kwargs):
        # Force the genuinely hung row to time out quickly on either pass.
        # Its healthy sibling retains the ordinary production deadline: the
        # fairness invariant does not require real SQLite/ACK I/O in 50 ms.
        with monkeypatch.context() as deadline:
            if kwargs["task_id"] == "q-a-stuck":
                deadline.setattr(
                    fleet.feature, "QUESTION_SETTLEMENT_TIMEOUT_SECONDS",
                    hung_timeout,
                )
            return await real_settle(**kwargs)

    fleet.feature._settle_question_with_receiver = _short_deadline_only_for_hung_row

    await fleet.settle()
    assert attempts == ["q-a-stuck"]
    await fleet.settle()

    assert attempts[0] == "q-a-stuck"
    assert "q-b-fine" in attempts
    assert await fleet.recipient_state("q-b-fine") is TaskState.CANCELED
    assert (await fleet.store.get("q-b-fine")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )


class _PerRecipientRouter:
    """The host-attested router with each recipient's host able to go down
    on its own. Records every request it is asked to carry, reached or not,
    as ``(operation, recipient, task_id)``."""

    def __init__(self, inner: _HostAttestedRouter):
        self._inner = inner
        self.down: set[str] = set()
        self.requests: list[tuple[str, str, str]] = []

    def _carry(self, operation: str, peer: PeerIdentity, task_id: str) -> None:
        self.requests.append((operation, peer.agent_id, task_id))
        if peer.agent_id in self.down:
            raise PeerTransportError("Could not reach local peer host")

    async def cancel_a2a_task(self, requester, peer, task_id, payload):
        self._carry("cancel", peer, task_id)
        return await self._inner.cancel_a2a_task(requester, peer, task_id, payload)

    async def get_a2a_task(self, requester, peer, task_id):
        self._carry("read", peer, task_id)
        return await self._inner.get_a2a_task(requester, peer, task_id)


def _add_kite(fleet, kite_tasks) -> _PerRecipientRouter:
    """Publish a second recipient, Kite, on Nellie's host, and route Emma's
    settlement through a router that can take either one down alone."""
    manager = fleet.router._manager
    kite = SimpleNamespace(did=KITE, agent_id=KITE, task_manager=kite_tasks)
    manager._register_agent("kite", kite)
    manager.install_a2a_hosted_policy(
        kite,
        resolver=object(),
        authorizer=object(),
        router=SimpleNamespace(authorize_inbound_sender=AsyncMock(return_value=True)),
        requester=PeerRequester(KITE, object()),
    )
    router = _PerRecipientRouter(fleet.router)
    peers = {RECIPIENT: PEER, KITE: KITE_PEER}

    async def _resolve(_recipient, recipient_agent_id):
        return router, fleet.requester, peers[recipient_agent_id]
    fleet.feature._resolve_retained_automatic_peer = AsyncMock(side_effect=_resolve)
    return router


@pytest.mark.asyncio
async def test_unreachable_recipient_costs_one_attempt_per_pass(fleet, tmp_path):
    """#3586: a recipient whose host refuses every connection fails inside
    the settlement bound, so the bound never deferred its other questions.
    Each was tried in turn, at up to the bound apiece, and one dead peer's
    backlog held up settlement for every recipient behind it.

    One attempt (its cancellation and the read that follows it) now stands
    for that recipient's whole pass: the healthy recipient behind it is
    settled in the same pass, and the deferred questions are taken up by the
    next one, until the host is back and they are all withdrawn."""
    kite_tasks = await create_task_manager(str(tmp_path / "kite-tasks.db"))
    try:
        router = _add_kite(fleet, kite_tasks)
        kite_questions = ("q-kite-1", "q-kite-2", "q-kite-3")
        for task_id in kite_questions:
            await fleet.ask(
                task_id, "Anyone?",
                recipient="Kite", recipient_agent_id=KITE,
                recipient_tasks=kite_tasks,
            )
        await fleet.ask("q-nellie", "Numbers?")
        # Both hosts are down at the deadlines: every question is left for
        # the sweep, Kite's ahead of Nellie's.
        router.down = {KITE, RECIPIENT}
        for task_id in (*kite_questions, "q-nellie"):
            await fleet.expire(task_id)
        assert [w.payload["state"] for w in fleet.wakes] == ["expired"] * 4

        # Nellie's host is back; Kite's is not.
        router.down = {KITE}
        router.requests.clear()
        await fleet.settle()

        assert router.requests[0][1] == KITE
        kite_requests = [r for r in router.requests if r[1] == KITE]
        assert kite_requests == [
            ("cancel", KITE, "q-kite-1"), ("read", KITE, "q-kite-1"),
        ]
        assert await fleet.recipient_state("q-nellie") is TaskState.CANCELED
        assert (await fleet.store.get("q-nellie")).receiver_settlement == (
            RECEIVER_SETTLEMENT_CONFIRMED
        )
        for task_id in kite_questions:
            assert (await fleet.store.get(task_id)).receiver_settlement == (
                RECEIVER_SETTLEMENT_PENDING
            )

        # Still down: the next pass takes up a question this one deferred.
        router.requests.clear()
        await fleet.settle()

        assert router.requests == [
            ("cancel", KITE, "q-kite-2"), ("read", KITE, "q-kite-2"),
        ]

        # Kite's host is back: nothing deferred was lost.
        router.down = set()
        await fleet.settle()

        for task_id in kite_questions:
            task = await kite_tasks.get_task_for_recipient(task_id, KITE)
            assert task.status.state is TaskState.CANCELED
            assert (await fleet.store.get(task_id)).receiver_settlement == (
                RECEIVER_SETTLEMENT_CONFIRMED
            )
        # Withdrawals are silent: the asker already holds each ``expired``.
        assert len(fleet.wakes) == 4
    finally:
        await kite_tasks.close()


# ---------------------------------------------------------------------------
# A row's claim lease is renewed while a wake for the row is in flight
# ---------------------------------------------------------------------------


async def _eventually(predicate, *, timeout: float = 30.0) -> None:
    """Wait for ``predicate()``; a guard that never holds fails, not hangs."""

    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(_poll(), timeout=timeout)


class _RecordingRenewal:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        return self.results.pop(0) if self.results else True


class TestClaimLeaseRenewingHandle:
    @pytest.mark.asyncio
    async def test_renews_until_the_wake_lands_and_returns_its_result(self):
        from kestrel_sovereign.features.peers.feature import (
            _ClaimLeaseRenewingHandle,
        )

        inner = _BlockedHandle()
        renew = _RecordingRenewal([])
        handle = _ClaimLeaseRenewingHandle(inner, interval=0.01, renew=renew)
        waiting = asyncio.create_task(handle.wait())
        await _eventually(lambda: renew.calls >= 3)
        inner.release.set()

        result = await asyncio.wait_for(waiting, timeout=30)

        assert result.status is Status.OK
        calls = renew.calls
        await asyncio.sleep(0.05)
        assert renew.calls == calls, "renewal must end with the wait"

    @pytest.mark.asyncio
    async def test_renews_as_soon_as_the_wait_begins(self):
        """Holding starts with the wait, not one interval later."""
        from kestrel_sovereign.features.peers.feature import (
            _ClaimLeaseRenewingHandle,
        )

        inner = _BlockedHandle()
        renew = _RecordingRenewal([])
        handle = _ClaimLeaseRenewingHandle(inner, interval=3600, renew=renew)
        waiting = asyncio.create_task(handle.wait())
        await _eventually(lambda: renew.calls >= 1)
        inner.release.set()

        assert (await asyncio.wait_for(waiting, timeout=30)).status is Status.OK
        assert renew.calls == 1

    @pytest.mark.asyncio
    async def test_delegates_other_handle_attributes(self):
        from kestrel_sovereign.features.peers.feature import (
            _ClaimLeaseRenewingHandle,
        )

        inner = _DeliveredHandle()
        inner.signal_id = "sig-1"
        handle = _ClaimLeaseRenewingHandle(
            inner, interval=3600, renew=_RecordingRenewal([]),
        )

        assert handle.signal_id == "sig-1"

    @pytest.mark.asyncio
    async def test_stops_renewing_when_nothing_is_left_to_hold(self):
        from kestrel_sovereign.features.peers.feature import (
            _ClaimLeaseRenewingHandle,
        )

        inner = _BlockedHandle()
        renew = _RecordingRenewal([True, False])
        handle = _ClaimLeaseRenewingHandle(inner, interval=0.01, renew=renew)
        waiting = asyncio.create_task(handle.wait())
        await _eventually(lambda: renew.calls >= 2)
        await asyncio.sleep(0.05)
        assert renew.calls == 2
        inner.release.set()

        assert (await asyncio.wait_for(waiting, timeout=30)).status is Status.OK

    @pytest.mark.asyncio
    async def test_cancelling_the_waiter_cancels_the_wait(self):
        from kestrel_sovereign.features.peers.feature import (
            _ClaimLeaseRenewingHandle,
        )

        inner_cancelled = asyncio.Event()

        class _Inner:
            async def wait(self):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    inner_cancelled.set()
                    raise

        handle = _ClaimLeaseRenewingHandle(
            _Inner(), interval=0.01, renew=_RecordingRenewal([]),
        )
        waiting = asyncio.create_task(handle.wait())
        await asyncio.sleep(0.03)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await asyncio.wait_for(inner_cancelled.wait(), timeout=30)

    @pytest.mark.asyncio
    async def test_a_cancelled_dispatch_is_still_reported_as_one(self):
        """``await_terminal_delivery`` tells "the dispatch died" from "we are
        being cancelled"; the wrapper must not blur the two."""
        from kestrel_sovereign.features.peers.feature import (
            _ClaimLeaseRenewingHandle,
        )
        from kestrel_sovereign.signals.delivery import (
            STATUS_DISPATCH_CANCELLED,
            await_terminal_delivery,
        )

        class _DispatchDied:
            async def wait(self):
                raise asyncio.CancelledError()

        handle = _ClaimLeaseRenewingHandle(
            _DispatchDied(), interval=0.01, renew=_RecordingRenewal([]),
        )

        outcome = await await_terminal_delivery(handle, label="test")

        assert outcome.status == STATUS_DISPATCH_CANCELLED
        assert not outcome.delivered


async def _spy_renewals(fleet) -> list:
    """Record each renewal of the real store's lease and its result."""
    real = fleet.store.renew_claim_lease
    results: list = []

    async def _spy(task_id, **kwargs):
        renewed = await real(task_id, **kwargs)
        results.append((task_id, renewed))
        return renewed
    fleet.store.renew_claim_lease = _spy
    fleet.feature.QUESTION_CLAIM_LEASE_RENEW_SECONDS = 0.01
    return results


async def _lease_until(fleet, task_id: str):
    [(value,)] = await fleet.db.fetchall(
        "SELECT claim_lease_until FROM pending_a2a_questions "
        "WHERE task_id = ?",
        (task_id,),
    )
    return value


@pytest.mark.asyncio
async def test_late_relay_keeps_its_lease_for_the_whole_turn(fleet):
    await _late_answer_after_unreached_expiry(fleet, "q-long")
    # Real-clock passes here: the expiry's own lease has lapsed.
    await fleet.db.execute(
        "UPDATE pending_a2a_questions SET claim_lease_until = NULL "
        "WHERE task_id = ?",
        ("q-long",),
    )
    renewals = await _spy_renewals(fleet)
    in_flight = _BlockedHandle()
    fleet.feature.agent.dispatcher.next_handles.append(in_flight)

    relay = asyncio.create_task(
        fleet.feature._settle_expired_questions_with_receivers(fleet.store),
    )
    await asyncio.wait_for(in_flight.waiting.wait(), timeout=30)
    claimed_lease = str(await _lease_until(fleet, "q-long"))

    async def _lease_grows() -> None:
        while str(await _lease_until(fleet, "q-long")) <= claimed_lease:
            await asyncio.sleep(0.01)
    await asyncio.wait_for(_lease_grows(), timeout=30)

    # The relay's own ownership check, then the in-flight renewals.
    assert len(renewals) >= 2
    assert all(renewal == ("q-long", True) for renewal in renewals)
    in_flight.release.set()
    await asyncio.wait_for(relay, timeout=30)
    assert (await fleet.store.get("q-long")).receiver_settlement == (
        RECEIVER_SETTLEMENT_CONFIRMED
    )


@pytest.mark.asyncio
async def test_expired_wake_keeps_the_rows_lease_while_in_flight(fleet):
    """No late answer can overtake an ``expired`` wake whose turn outlasts
    the lease."""
    await fleet.ask("q-slow", "Numbers?")
    fleet.router.reachable = False
    renewals = await _spy_renewals(fleet)
    in_flight = _BlockedHandle()
    fleet.feature.agent.dispatcher.next_handles.append(in_flight)

    expiry = asyncio.create_task(fleet.expire("q-slow"))
    await asyncio.wait_for(in_flight.waiting.wait(), timeout=30)
    await _eventually(lambda: bool(renewals))

    assert renewals[0] == ("q-slow", True)
    in_flight.release.set()
    await asyncio.wait_for(expiry, timeout=30)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", ["sweep", "near_term"])
async def test_retried_expired_wake_holds_a_claim_of_its_own(fleet, retry):
    """An ``expired`` wake that was not delivered returns its row to WAITING.
    Whichever retry expires it again claims the row afresh and holds that
    claim while the retried wake is in flight, so the recipient's late answer
    cannot overtake it."""
    await fleet.ask("q-retried", "Numbers?")
    fleet.router.reachable = False
    # The expiry's ``expired`` wake is not delivered, so the expiry restores
    # its row to WAITING with that wake. The near-term retry it schedules is
    # dropped here: this test drives the retry itself.
    fleet.feature.agent.dispatcher.next_handles.append(_UndeliveredHandle())
    fleet.feature.agent._track_background_task = (
        lambda coro, *, name: coro.close() or MagicMock()
    )
    await fleet.expire("q-retried")
    row = await fleet.store.get("q-retried")
    assert (row.status, row.retry_state, row.receiver_settlement) == (
        "WAITING", "expired", RECEIVER_SETTLEMENT_PENDING,
    )
    await fleet.answer("q-retried", "the numbers")
    fleet.router.reachable = True
    renewals = await _spy_renewals(fleet)
    in_flight = _BlockedHandle()
    fleet.feature.agent.dispatcher.next_handles.append(in_flight)

    if retry == "sweep":
        retried = asyncio.create_task(fleet.expire("q-retried"))
    else:
        retried = asyncio.create_task(
            fleet.feature._retry_restored_question_answered_signal(
                task_id="q-retried",
                recipient="Nellie",
                original_question="Numbers?",
                sess_id="session-q-retried",
                state="expired",
                reply_text="",
                causation_chain=None,
            )
        )
    await asyncio.wait_for(in_flight.waiting.wait(), timeout=30)
    await _eventually(lambda: bool(renewals))
    # A pass on the real clock, inside the retried wake's lease.
    await fleet.feature._settle_expired_questions_with_receivers(fleet.store)

    assert renewals[0] == ("q-retried", True)
    assert [w.payload["state"] for w in fleet.wakes] == ["expired", "expired"]
    in_flight.release.set()
    await asyncio.wait_for(retried, timeout=30)
    # Delivered, the retried wake makes way for the late answer, once.
    await fleet.settle()
    await fleet.settle()
    assert [(w.payload["state"], w.payload["late"]) for w in fleet.wakes] == [
        ("expired", False), ("expired", False), ("completed", True),
    ]


@pytest.mark.asyncio
async def test_near_term_retry_of_a_withdrawn_expiry_still_says_withdrawn(fleet):
    """The question was withdrawn but its ``expired`` wake failed, so the
    expiry restored the row and scheduled a near-term retry. The retried wake
    must say what the first one said."""
    await fleet.ask("q-withdrawn-retry", "Numbers?")
    fleet.feature.agent.dispatcher.next_handles.append(_UndeliveredHandle())
    scheduled: list = []
    fleet.feature.agent._track_background_task = (
        lambda coro, *, name: scheduled.append(coro) or MagicMock()
    )

    await fleet.expire("q-withdrawn-retry")
    [retry] = scheduled
    await asyncio.wait_for(retry, timeout=30)

    assert [(w.payload["state"], w.payload["withdrawn"]) for w in fleet.wakes] == [
        ("expired", True), ("expired", True),
    ]
    row = await fleet.store.get("q-withdrawn-retry")
    assert (row.status, row.retry_state) == ("EXPIRED", None)
    assert row.receiver_settlement == RECEIVER_SETTLEMENT_CONFIRMED


class _DurableRecipient(EventManagerMixin):
    """Nellie behind her real dispatcher and durable signal ledger."""

    did = RECIPIENT

    def __init__(self, task_manager, hold):
        self.task_manager = task_manager
        self._hold_store = hold
        self.process_input_calls: list = []
        self.background_tasks: list = []
        self._durable_persistence_gate = DurablePersistenceGate()

    async def process_input(self, prompt):
        self.process_input_calls.append(prompt)
        return "answered a withdrawn question"

    def _get_durable_persistence_gate(self):
        return self._durable_persistence_gate

    def _track_background_task(self, coroutine, *, name, admission=None):
        from kestrel_sovereign._async_ownership import create_owned_task
        task = create_owned_task(coroutine, name=name, admission=admission)
        self.background_tasks.append(task)
        return task


@pytest.mark.asyncio
async def test_recipient_returning_after_retention_has_its_submission_withdrawn(
    fleet, tmp_path,
):
    """Durable retention keeps undelivered work, and the recipient's
    execution gate checks only the task's state. So a recipient away for a
    month still runs a live submission when it returns, unless the sender is
    still settling it: settlement has no age cutoff."""
    backend = SQLiteBackend(str(tmp_path / "nellie-signals.db"))
    await backend.connect()
    log_store = SignalLogStore(backend)
    await log_store.initialize()
    hold = _HoldSnapshots(_held_state(RECIPIENT))
    nellie = _DurableRecipient(fleet.task_manager, hold)
    registry = SourceRegistry()
    registry.register(build_a2a_task_submitted_registration())
    dispatcher = SignalDispatcher(
        agent=nellie,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    try:
        await dispatcher.initialize_durable_delivery()
        await dispatcher.register_durable_consumer(
            DurableConsumerRegistration(
                consumer_id=SUBMITTED_CONSUMER,
                source=SUBMITTED_SOURCE,
                agent_id=RECIPIENT,
                max_attempts=0,
            )
        )
        await fleet.ask("q-away", "Live numbers?")
        task = await fleet.task_manager.get_task_for_recipient("q-away", RECIPIENT)
        # Nellie is away: her submission wake is admitted, and waits.
        handle = await dispatcher.enqueue_durable_cognition(
            build_signal_for_submitted_task(
                task, target_agent=RECIPIENT, sender=SENDER,
            ),
            source_event_id=task.id,
            consumer_id=SUBMITTED_CONSUMER,
        )
        admission = await asyncio.wait_for(
            handle.wait_for_durable_admission(), timeout=30,
        )
        assert admission.disposition is DurableAdmissionDisposition.HELD
        await asyncio.wait_for(handle.wait(), timeout=30)

        fleet.router.reachable = False
        await fleet.expire("q-away")
        assert (await fleet.store.get("q-away")).receiver_settlement == (
            RECEIVER_SETTLEMENT_PENDING
        )

        # A month passes: twice the submission source's 14-day retention.
        fleet.clock += timedelta(days=30)
        await dispatcher._durable_store.purge_expired(
            agent_id=RECIPIENT, now=fleet.clock,
        )
        [retained] = await dispatcher.list_durable_deliveries(
            consumer_id=SUBMITTED_CONSUMER,
        )
        assert retained.status in (PENDING, RETRY)

        # Nellie's host is reachable again; the sender's next pass settles.
        fleet.router.reachable = True
        await fleet.settle()

        assert await fleet.recipient_state("q-away") is TaskState.CANCELED
        assert (await fleet.store.get("q-away")).receiver_settlement == (
            RECEIVER_SETTLEMENT_CONFIRMED
        )
        assert [w.payload["state"] for w in fleet.wakes] == ["expired"]

        # Nellie's retained submission is refused, not run.
        hold.snapshots[:] = [EffectiveHoldState(host=None, agent=None)]
        await dispatcher.start_durable_cognition_consumer(SUBMITTED_CONSUMER)
        await _drain_once(dispatcher, SUBMITTED_CONSUMER)

        assert nellie.process_input_calls == []
        [settled] = await dispatcher.list_durable_deliveries(
            consumer_id=SUBMITTED_CONSUMER,
        )
        assert settled.is_terminal, settled.status
    finally:
        await dispatcher.shutdown_durable_delivery()
        pending = [t for t in nellie.background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await backend.close()


@pytest.mark.asyncio
async def test_late_answer_a_month_after_expiry_still_wakes_the_asker(fleet):
    await fleet.ask("q-month", "Live numbers?")
    fleet.router.reachable = False
    await fleet.expire("q-month")
    fleet.clock += timedelta(days=30)
    await fleet.answer("q-month", "numbers, a month on")
    fleet.router.reachable = True

    await fleet.settle()

    late = fleet.wakes[-1]
    assert late.payload["late"] is True
    assert late.payload["reply_text"] == "numbers, a month on"


async def _expire_on_a_release_before_3440(fleet, *task_ids: str) -> None:
    """What a release before #3440 left behind: each question expired on the
    sender alone, its recipient never told, and the asker woken ``expired``.
    That release's table had no settlement columns at all."""
    for task_id in task_ids:
        await fleet.db.execute(
            "UPDATE pending_a2a_questions "
            "SET status = 'EXPIRED', resolved_at = CURRENT_TIMESTAMP "
            "WHERE task_id = ?",
            (task_id,),
        )
        await update_outbound_terminal_state(
            fleet.db, agent_id=SENDER, task_id=task_id, terminal_state="expired",
        )
    for column in (
        "receiver_settlement",
        "claim_token",
        "claim_lease_until",
        "receiver_late_state",
        "receiver_late_reply_text",
    ):
        await fleet.db.execute(
            f"ALTER TABLE pending_a2a_questions DROP COLUMN {column}",
        )


@pytest.mark.asyncio
async def test_questions_expired_before_the_upgrade_are_settled(fleet):
    """The 2026-10-02 incidents expired on a release that never told the
    recipient: one question stayed live at Nellie and was re-delivered, and
    another was answered after Emma gave up. After the upgrade, Emma's first
    settlement pass withdraws the first and relays the second's answer."""
    await fleet.ask("q-238234", "What are the live numbers?")
    await fleet.ask("q-ea48db1e", "Live numbers, again?")
    await _expire_on_a_release_before_3440(fleet, "q-238234", "q-ea48db1e")
    answered = await fleet.answer("q-ea48db1e", "valid live numbers")
    assert answered.status is ToolResultStatus.OK

    await fleet.restart_sender()  # the upgraded process
    await fleet.settle()

    assert await fleet.recipient_state("q-238234") is TaskState.CANCELED
    task = await fleet.task_manager.get_task_for_recipient("q-238234", RECIPIENT)
    stale = build_signal_for_submitted_task(task, target_agent=RECIPIENT, sender=SENDER)
    assert await fleet.recipient.validate_cognition_signal_execution(stale)
    [late] = fleet.wakes
    assert late.payload["task_id"] == "q-ea48db1e"
    assert late.payload["state"] == "completed"
    assert late.payload["reply_text"] == "valid live numbers"
    assert late.payload["late"] is True
    for task_id in ("q-238234", "q-ea48db1e"):
        assert (await fleet.store.get(task_id)).receiver_settlement == (
            RECEIVER_SETTLEMENT_CONFIRMED
        )
    outbound = await get_outbound_task(fleet.db, agent_id=SENDER, task_id="q-ea48db1e")
    assert outbound.terminal_state == "completed"

    await fleet.settle()
    assert len(fleet.wakes) == 1


@pytest.mark.asyncio
async def test_supervisor_deadline_withdraws_the_question(fleet):
    """The live supervisor's deadline exit takes the same expiry path."""
    await fleet.ask("q-live", "Still there?")

    async def _no_events(*_args, **_kwargs):
        return
        yield  # pragma: no cover - async generator marker
    fleet.router.subscribe_a2a_task = _no_events

    await fleet.feature._supervise_a2a_question(
        task_id="q-live",
        recipient="Nellie",
        recipient_agent_id=RECIPIENT,
        original_question="Still there?",
        sess_id="session-q-live",
        deadline_utc=datetime.now(timezone.utc) - timedelta(seconds=1),
        causation_chain=None,
    )

    [wake] = fleet.wakes
    assert wake.payload["state"] == "expired"
    assert await fleet.recipient_state("q-live") is TaskState.CANCELED
