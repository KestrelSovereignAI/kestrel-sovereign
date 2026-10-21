"""Resident recovery of native question reads and canonical durable wake handoff."""

import asyncio
import json
import logging
import math
from dataclasses import replace
from time import monotonic

from a2a.utils.errors import A2AError

from kestrel_sovereign._async_ownership import gather_owned
from kestrel_sovereign.a2a.native_client import NativePeerProtocolError
from kestrel_sovereign.a2a.native_sender import NativeSenderProofError
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatch,
    OutboundDispatchConflict,
)
from kestrel_sovereign.a2a.stores.unified.question_store import (
    NativeQuestionStore,
    QuestionResponseError,
)
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.features.peers.directory import PeerDirectoryError
from kestrel_sovereign.signals.dispatcher import DurableAdmissionDisposition
from kestrel_sovereign.signals.sources.a2a_question_answered import (
    DURABLE_COGNITION_CONSUMER_ID,
    build_signal_for_question_answered,
)

logger = logging.getLogger(__name__)


class NativeQuestionSupervisor:
    """Observe retained routes without ever replaying their original sends.

    The feature owns this resident service. The dispatcher owns cognition after
    durable admission; a question wake checkpoint never waits for an LLM turn.
    Read recovery is a configured polling policy, independent of SDK streaming.
    """

    def __init__(
        self,
        *,
        store: NativeQuestionStore,
        open_dispatch,
        dispatcher,
        interval: float,
        is_current,
        max_interval: float | None = None,
        clock=monotonic,
    ):
        if not isinstance(store, NativeQuestionStore):
            raise TypeError(
                "Question supervision requires the canonical primary ledger"
            )
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError(
                "Question observation interval must be a positive finite duration"
            )
        if max_interval is None:
            max_interval = interval
        if (
            isinstance(max_interval, bool)
            or not isinstance(max_interval, (int, float))
            or not math.isfinite(max_interval)
            or max_interval < interval
        ):
            raise ValueError(
                "Maximum question observation interval must be finite and at least the initial interval"
            )
        self._clock = clock
        self._next_read: dict[str, tuple[float, float]] = {}
        self.max_interval = max_interval
        self._is_current = is_current
        self.store = store
        self.open_dispatch = open_dispatch
        self.dispatcher = dispatcher
        self.interval = interval
        self._changed = asyncio.Event()

    def notify(self) -> None:
        self._changed.set()

    async def run(self) -> None:
        # Peer reads may wait on network timeouts. Deadline settlement and
        # durable wake admission never wait behind those reads or an LLM turn.
        await gather_owned(
            (self._read_loop(), self._deadline_loop()),
            retain_originals=True,
            name_prefix="native-question-service",
        )

    async def _read_loop(self) -> None:
        while True:
            await self._read_pass()
            await asyncio.sleep(self.interval)

    async def _deadline_loop(self) -> None:
        while True:
            self._changed.clear()
            await self._deadline_pass()
            await self.handoff_wakes()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=self.interval)
            except TimeoutError:
                continue

    def _require_current(self) -> None:
        if not self._is_current():
            raise RuntimeError("Native question service belongs to a replaced runtime")

    async def recover_once(self) -> None:
        """One joined observation and handoff pass for explicit recovery."""
        await self._deadline_pass()
        await self._read_pass(force=True)
        await self.handoff_wakes()

    async def _deadline_pass(self) -> None:
        self._require_current()
        for question in await self.store.unsettled():
            self._require_current()
            await self._settle_local(question)

    @staticmethod
    def _observation_progress(
        dispatch: OutboundDispatch | None,
    ) -> OutboundDispatch | None:
        """Exclude receipt-write bookkeeping from the remote progress signal."""
        if dispatch is None:
            return None
        return replace(dispatch, revision=0, operation_id="", updated_at="")

    async def _read_pass(self, *, force: bool = False) -> None:
        self._require_current()
        questions = await self.store.unsettled()
        live_ids = {question.question_id for question in questions}
        self._next_read = {
            key: schedule
            for key, schedule in self._next_read.items()
            if key in live_ids
        }
        for question in questions:
            self._require_current()
            now = self._clock()
            due, delay = self._next_read.get(question.question_id, (now, self.interval))
            if not force and now < due:
                continue
            before = await self.store.outbound_store.get(question.dispatch_id)
            try:
                await self._observe(question)
            except (
                PeerDirectoryError,
                A2AError,
                NativePeerProtocolError,
                NativeSenderProofError,
                QuestionResponseError,
                OutboundDispatchConflict,
            ) as error:
                if is_execution_control_error(error):
                    raise
                # The retained route and deadline remain owed. A read retry is
                # safe; neither this branch nor restart resubmits a Message.
                logger.warning(
                    "Native question observation refused: dispatch=%s error=%s",
                    question.dispatch_id,
                    type(error).__name__,
                )
            after = await self.store.outbound_store.get(question.dispatch_id)
            delay = (
                self.interval
                if self._observation_progress(after)
                != self._observation_progress(before)
                else min(delay * 2, self.max_interval)
            )
            if question.status == "WAITING":
                # A growing retry interval must not skip the remaining answer
                # window. Tighten reads as its original deadline approaches;
                # the independent deadline lane still never waits for I/O.
                delay = min(
                    delay,
                    max(
                        self.interval,
                        self.store.deadline_remaining_seconds(question) / 2,
                    ),
                )
            self._next_read[question.question_id] = (self._clock() + delay, delay)

    async def _settle_local(self, question):
        expired = self.store.deadline_passed(question)
        if expired and question.status == "WAITING":
            question = await self.store.expire(question.question_id)
        local = await self.store.outbound_store.get(question.dispatch_id)
        if local is None:
            raise OutboundDispatchConflict("Question lost its sender-owned dispatch")
        if local.state in {DispatchState.RESERVED, DispatchState.NOT_SENT}:
            if expired or local.state is DispatchState.NOT_SENT:
                await self.store.settle_unsent(question.question_id, local)
        return await self.store.get(question.question_id)

    async def _observe(self, question) -> None:
        question = await self._settle_local(question)
        if question.receiver_settled:
            return
        expired = self.store.deadline_passed(question)
        local = await self.store.outbound_store.get(question.dispatch_id)
        if local.state in {
            DispatchState.RESERVED,
            DispatchState.NOT_SENT,
        } or self.store.outbound_store.submission_active(local.dispatch_id):
            return
        async with self.open_dispatch(question.dispatch_id) as (session, dispatch):
            if dispatch.state in {DispatchState.SUBMITTING, DispatchState.UNKNOWN}:
                dispatch = await session.reconcile(dispatch)
                if dispatch is None:
                    return
            if dispatch.state in {DispatchState.RESERVED, DispatchState.NOT_SENT}:
                return
            current = await self.store.get(question.question_id)
            if current.receiver_settled:
                return
            receipt = await session.get_task(dispatch)
            if receipt.dispatch.state is DispatchState.COMPLETE:
                return  # The native client atomically retained the answer and wake.
            if expired and receipt.dispatch.cancel_state in {
                None,
                DispatchState.NOT_SENT,
            }:
                await session.cancel_task(receipt.dispatch)
            # An uncertain cancellation is observed by reads only. An active
            # task after a lost response does not prove cancellation was unsent.

    async def handoff_wakes(self) -> None:
        self._require_current()
        for candidate in await self.store.pending_wakes():
            self._require_current()
            wake = await self.store.claim_wake(candidate)
            if wake is None:
                continue
            signal = build_signal_for_question_answered(
                event_id=wake.event_id,
                payload=json.loads(wake.payload_json),
                target_agent=self.store.sender_scope.recipient_id,
            )
            handle = await self.dispatcher.enqueue_durable_cognition(
                signal,
                source_event_id=wake.event_id,
                consumer_id=DURABLE_COGNITION_CONSUMER_ID,
            )
            admission = await handle.wait_for_durable_admission()
            if admission.signal_id != signal.id:
                raise RuntimeError("Question wake received another signal's admission")
            self._require_current()
            if admission.disposition is DurableAdmissionDisposition.DELIVERY_FAILED:
                if not await self.store.record_delivery_failure(wake):
                    raise RuntimeError("Failed question delivery lost its outbox lease")
                logger.error(
                    "Native question cognition already failed terminally: event=%s",
                    wake.event_id,
                )
            elif admission.acknowledged:
                if not await self.store.confirm_handoff(wake):
                    raise RuntimeError("Durable question handoff lost its outbox lease")
