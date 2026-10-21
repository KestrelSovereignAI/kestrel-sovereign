"""Sender-scoped native question intent, correlated answers and durable wake outbox."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, uuid4, uuid5

from a2a.types import Message, Role, SendMessageRequest, Task
from google.protobuf.json_format import MessageToDict, ParseDict

from kestrel_sovereign.a2a.extensions import parse_question_correlation
from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES, task_state_label
from kestrel_sovereign.a2a.native_sender import native_request_digest
from kestrel_sovereign.a2a.projections import message_text
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatch,
    OutboundStore,
)


class QuestionConflict(ValueError):
    """Correlation, native receipt or question ownership disagrees with durable intent."""


class QuestionResponseError(QuestionConflict):
    """An authenticated peer returned an invalid question answer."""


def answer_fields(question_id: str, response: Task | Message):
    """Validate peer assertions before a receipt transaction can wrap the error."""
    if not isinstance(response, (Task, Message)):
        raise TypeError("Question answers require official Task or Message values")
    if isinstance(response, Task):
        if response.status.state not in TERMINAL_STATES:
            raise QuestionResponseError("Question answer task is not terminal")
        try:
            original = parse_question_correlation(MessageToDict(response.metadata))
        except ValueError as error:
            raise QuestionResponseError(
                "Remote task has malformed question correlation"
            ) from error
        if original is None or (original.kind, original.question_id) != (
            "question",
            question_id,
        ):
            raise QuestionResponseError(
                "Remote task lost its original question correlation"
            )
        state = task_state_label(response.status.state)
        reply = response.status.message if response.status.HasField("message") else None
    else:
        if response.task_id:
            raise QuestionResponseError(
                "A standalone answer must retain its actual taskless identity"
            )
        state, reply = "completed", response
    if reply is not None:
        try:
            answer = parse_question_correlation(MessageToDict(reply.metadata))
        except ValueError as error:
            raise QuestionResponseError(
                "Answer has malformed question correlation"
            ) from error
        if (
            reply.role != Role.ROLE_AGENT
            or (answer is None and state == "completed")
            or (
                answer is not None
                and (answer.kind, answer.question_id) != ("answer", question_id)
            )
        ):
            raise QuestionResponseError(
                "Answer does not match the outstanding question identity"
            )
    elif state == "completed":
        raise QuestionResponseError(
            "Completed question has no correlated answer message"
        )
    return state, reply


@dataclass(frozen=True)
class NativeQuestion:
    question_id: str
    dispatch_id: str
    recipient_id: str
    recipient_tenant: str
    original_question: str
    request_json: str
    origin_turn_id: str
    origin_session_id: str
    causation_json: str
    deadline: str
    status: str
    receiver_settled: int
    response_json: str | None
    response_digest: str | None


@dataclass(frozen=True)
class QuestionWake:
    question_id: str
    wave: str
    event_id: str
    payload_json: str
    claim_token: str | None
    lease_until: str | None
    handed_off_at: str | None
    delivery_failed_at: str | None = None


_QUESTION_COLUMNS = ", ".join(NativeQuestion.__dataclass_fields__)
_WAKE_COLUMNS = ", ".join(QuestionWake.__dataclass_fields__)


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(
            "Question clocks and deadlines require timezone-aware UTC instants"
        )
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _now() -> datetime:
    return datetime.now(UTC)


def _text(value: str) -> bool:
    return isinstance(value, str) and bool(value.strip())


class NativeQuestionStore(UnifiedStoreBase):
    """One primary-store correlation ledger; a peer assertion never grants a wake.

    A receipt must match the exact response digest already admitted by the native
    outbound client. Wakes are immutable and keyed deterministically, so handing
    one to the canonical durable signal dispatcher can be retried after a crash.
    """

    def __init__(
        self,
        outbound_store: OutboundStore,
        *,
        clock: Callable[[], datetime] = _now,
        claim_lease: timedelta = timedelta(minutes=15),
    ):
        if not isinstance(outbound_store, OutboundStore):
            raise TypeError("Native questions require the canonical outbound ledger")
        if not isinstance(claim_lease, timedelta) or claim_lease <= timedelta():
            raise ValueError("Question claim leases must be positive durations")
        super().__init__(outbound_store.backend, owns_backend=False)
        self.outbound_store = outbound_store
        self._clock = clock
        self._claim_lease = claim_lease

    @property
    def sender_scope(self):
        return self.outbound_store.sender_scope

    @property
    def _owner(self):
        return self.sender_scope.recipient_id, self.sender_scope.tenant

    async def initialize(self) -> None:
        await self.backend.execute_script("""
            CREATE TABLE IF NOT EXISTS a2a_native_questions (
                sender_id TEXT NOT NULL, sender_tenant TEXT NOT NULL,
                question_id TEXT NOT NULL, dispatch_id TEXT NOT NULL,
                recipient_id TEXT NOT NULL, recipient_tenant TEXT NOT NULL,
                original_question TEXT NOT NULL, request_json TEXT NOT NULL,
                origin_turn_id TEXT NOT NULL,
                origin_session_id TEXT NOT NULL, causation_json TEXT NOT NULL,
                deadline TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('WAITING', 'RESOLVED', 'EXPIRED')),
                receiver_settled INTEGER NOT NULL CHECK (receiver_settled IN (0,1)),
                response_json TEXT, response_digest TEXT,
                PRIMARY KEY (sender_id, sender_tenant, question_id),
                UNIQUE (sender_id, sender_tenant, dispatch_id)
            );
            CREATE TABLE IF NOT EXISTS a2a_native_question_wakes (
                sender_id TEXT NOT NULL, sender_tenant TEXT NOT NULL,
                question_id TEXT NOT NULL, wave TEXT NOT NULL
                    CHECK (wave IN ('answer','deadline','late-answer')),
                event_id TEXT NOT NULL, payload_json TEXT NOT NULL,
                claim_token TEXT, lease_until TEXT, handed_off_at TEXT, delivery_failed_at TEXT,
                PRIMARY KEY (sender_id, sender_tenant, question_id, wave),
                UNIQUE (sender_id, sender_tenant, event_id)
            );
        """)

    async def get(self, question_id: str) -> NativeQuestion | None:
        row = await self.backend.fetch_one(
            f"SELECT {_QUESTION_COLUMNS} FROM a2a_native_questions "
            "WHERE sender_id = ? AND sender_tenant = ? AND question_id = ?",
            (*self._owner, question_id),
        )
        return NativeQuestion(*row) if row is not None else None

    async def for_dispatch(self, dispatch_id: str) -> NativeQuestion | None:
        row = await self.backend.fetch_one(
            f"SELECT {_QUESTION_COLUMNS} FROM a2a_native_questions "
            "WHERE sender_id = ? AND sender_tenant = ? AND dispatch_id = ?",
            (*self._owner, dispatch_id),
        )
        return NativeQuestion(*row) if row is not None else None

    async def unsettled(self) -> list[NativeQuestion]:
        rows = await self.backend.fetch_all(
            f"SELECT {_QUESTION_COLUMNS} FROM a2a_native_questions "
            "WHERE sender_id = ? AND sender_tenant = ? AND receiver_settled = 0 "
            "ORDER BY deadline, question_id",
            self._owner,
        )
        return [NativeQuestion(*row) for row in rows]

    def deadline_passed(self, question: NativeQuestion) -> bool:
        return question.deadline <= _timestamp(self._clock())

    async def retained_response(
        self, dispatch: OutboundDispatch
    ) -> Task | Message | None:
        question = await self.for_dispatch(dispatch.dispatch_id)
        if question is None:
            return None
        await self._binding(question.question_id, dispatch)
        if question.response_json is None:
            return None
        if dispatch.response_kind not in {"task", "message"}:
            raise QuestionConflict(
                "Retained question response has no native response kind"
            )
        response = ParseDict(
            json.loads(question.response_json),
            Task() if dispatch.response_kind == "task" else Message(),
        )
        digest = hashlib.sha256(
            response.SerializeToString(deterministic=True)
        ).hexdigest()
        if digest != question.response_digest or digest != dispatch.response_digest:
            raise QuestionConflict(
                "Retained question response differs from its native receipt"
            )
        return response

    async def acknowledge(
        self, dispatch: OutboundDispatch, response: Task | Message, *, operation_id: str
    ) -> OutboundDispatch | None:
        """Commit the native receipt and terminal answer in one primary transaction.

        In particular, a taskless Message has no remote task to read after a
        crash. Its answer must survive the same commit as its delivery receipt.
        """
        question = await self.for_dispatch(dispatch.dispatch_id)
        if question is None:
            raise QuestionConflict("Native question receipt has no registered intent")
        self.outbound_store.validate_response(dispatch, response)
        if (
            isinstance(response, Message)
            and not response.task_id
            or (isinstance(response, Task) and response.status.state in TERMINAL_STATES)
        ):
            answer_fields(question.question_id, response)
        async with self.write_transaction():
            receipt = await self.outbound_store.acknowledge(
                dispatch, response, operation_id=operation_id
            )
            if receipt is not None and receipt.state is DispatchState.COMPLETE:
                await self.record_answer(question.question_id, receipt, response)
        return receipt

    async def reserve(
        self,
        dispatch: OutboundDispatch,
        request: SendMessageRequest,
        *,
        origin_turn_id: str,
        origin_session_id: str,
        deadline: datetime,
    ) -> NativeQuestion:
        correlation = parse_question_correlation(MessageToDict(request.metadata))
        if (
            correlation is None
            or correlation.kind != "question"
            or dispatch.verb != "question"
            or dispatch.state is not DispatchState.RESERVED
            or await self.outbound_store.get(dispatch.dispatch_id) != dispatch
            or native_request_digest(request, "SendMessage") != dispatch.request_digest
        ):
            raise QuestionConflict(
                "Question intent requires its exact reserved native dispatch"
            )
        if not isinstance(origin_turn_id, str) or not isinstance(
            origin_session_id, str
        ):
            raise TypeError("Question origin lineage must be explicit strings")
        now = _timestamp(self._clock())
        deadline_text = _timestamp(deadline)
        if deadline_text <= now:
            raise ValueError("A question deadline must be in the future")
        from kestrel_sovereign.a2a.extensions import CAUSATION_URI

        metadata = MessageToDict(request.metadata)
        question = NativeQuestion(
            correlation.question_id,
            dispatch.dispatch_id,
            dispatch.recipient_id,
            dispatch.tenant,
            message_text(request.message) or "",
            json.dumps(MessageToDict(request), sort_keys=True),
            origin_turn_id,
            origin_session_id,
            json.dumps(metadata.get(CAUSATION_URI), sort_keys=True),
            deadline_text,
            "WAITING",
            0,
            None,
            None,
        )
        values = tuple(
            getattr(question, name) for name in NativeQuestion.__dataclass_fields__
        )
        changed = await self.backend.execute(
            f"INSERT INTO a2a_native_questions (sender_id, sender_tenant, {_QUESTION_COLUMNS}) "
            f"VALUES ({', '.join('?' for _ in range(len(values) + 2))}) ON CONFLICT DO NOTHING",
            (*self._owner, *values),
        )
        if changed != 1:
            raise QuestionConflict(
                "Question or dispatch already has durable correlation intent"
            )
        return question

    async def _binding(
        self, question_id: str, dispatch: OutboundDispatch
    ) -> NativeQuestion:
        question = await self.get(question_id)
        if (
            question is None
            or question.dispatch_id != dispatch.dispatch_id
            or (question.recipient_id, question.recipient_tenant)
            != (dispatch.recipient_id, dispatch.tenant)
            or await self.outbound_store.get(dispatch.dispatch_id) != dispatch
        ):
            raise QuestionConflict(
                "Question does not match the current sender-owned peer dispatch"
            )
        return question

    async def require_intent(
        self,
        dispatch: OutboundDispatch,
        request: SendMessageRequest,
    ) -> NativeQuestion:
        """Before peer HTTP, require the original live pre-dispatch correlation."""
        correlation = parse_question_correlation(MessageToDict(request.metadata))
        if correlation is None or correlation.kind != "question":
            raise QuestionConflict(
                "Native question dispatch has no question correlation"
            )
        question = await self._binding(correlation.question_id, dispatch)
        if (
            question.status != "WAITING"
            or question.receiver_settled
            or dispatch.verb != "question"
            or dispatch.state not in {DispatchState.RESERVED, DispatchState.SUBMITTING}
            or native_request_digest(request, "SendMessage") != dispatch.request_digest
            or question.deadline <= _timestamp(self._clock())
        ):
            raise QuestionConflict(
                "Question intent is no longer live for its exact original send"
            )
        return question

    async def require_continuation_intent(
        self, root: OutboundDispatch
    ) -> NativeQuestion:
        """A follow-up cannot extend or replace its original question deadline."""
        question = await self.for_dispatch(root.dispatch_id)
        if question is None:
            raise QuestionConflict(
                "Question continuation has no original durable intent"
            )
        question = await self._binding(question.question_id, root)
        if (
            root.verb != "question"
            or question.status != "WAITING"
            or question.receiver_settled
            or self.deadline_passed(question)
        ):
            raise QuestionConflict("Question continuation is no longer live")
        return question

    async def record_answer(
        self,
        question_id: str,
        dispatch: OutboundDispatch,
        response: Task | Message,
    ) -> NativeQuestion:
        question = await self._binding(question_id, dispatch)
        if not isinstance(response, (Task, Message)):
            raise TypeError("Question answers require official Task or Message values")
        digest = hashlib.sha256(
            response.SerializeToString(deterministic=True)
        ).hexdigest()
        if (
            digest != dispatch.response_digest
            or dispatch.state is not DispatchState.COMPLETE
        ):
            raise QuestionConflict(
                "Answer lacks a committed authorized terminal response receipt"
            )
        if response.context_id != dispatch.remote_context_id:
            raise QuestionConflict(
                "Answer context differs from the acknowledged native response"
            )
        if isinstance(response, Task):
            if (
                response.id != dispatch.remote_task_id
                or response.status.state not in TERMINAL_STATES
            ):
                raise QuestionConflict(
                    "Answer task differs from the acknowledged native task"
                )
        elif dispatch.remote_task_id:
            raise QuestionConflict(
                "A standalone answer differs from its native receipt"
            )
        state, reply = answer_fields(question_id, response)
        encoded = json.dumps(MessageToDict(response), sort_keys=True)
        if question.response_digest is not None:
            if question.response_digest != digest:
                raise QuestionConflict(
                    "Question already retains another terminal answer"
                )
            return question
        if question.status == "WAITING" and self.deadline_passed(question):
            await self.expire(
                question_id, withdrawal=dispatch if state == "canceled" else None
            )
        async with self.write_transaction():
            # The CAS fences concurrent deadline/answer observations, including
            # another connection on PostgreSQL. Re-read after the row lock winner.
            changed = await self.backend.execute(
                "UPDATE a2a_native_questions SET response_json = ?, response_digest = ?, "
                "receiver_settled = 1, status = CASE WHEN status = 'WAITING' THEN 'RESOLVED' ELSE status END "
                "WHERE sender_id = ? AND sender_tenant = ? AND question_id = ? AND response_digest IS NULL",
                (encoded, digest, *self._owner, question_id),
            )
            current = await self.get(question_id)
            if changed == 1 and (current is None or current.response_digest != digest):
                raise RuntimeError("Committed question settlement was not readable")
            # Cancellation after a deadline is withdrawal, not a late answer.
            if changed == 1 and (current.status != "EXPIRED" or state != "canceled"):
                await self._insert_wake(
                    current,
                    "late-answer" if current.status == "EXPIRED" else "answer",
                    state=state,
                    reply=(message_text(reply) or "") if reply is not None else "",
                    withdrawn=False,
                    requires_full_response=(
                        isinstance(response, Task) and bool(response.artifacts)
                    )
                    or (
                        reply is not None
                        and any(not part.HasField("text") for part in reply.parts)
                    ),
                )
        if current is None or current.response_digest != digest:
            raise QuestionConflict(
                "Question settlement lost its terminal response claim"
            )
        return current

    async def expire(
        self,
        question_id: str,
        *,
        withdrawal: OutboundDispatch | None = None,
    ) -> NativeQuestion:
        withdrawn = withdrawal is not None
        if withdrawal is not None:
            await self._binding(question_id, withdrawal)
            if (
                withdrawal.state is not DispatchState.COMPLETE
                or withdrawal.remote_task_state != "canceled"
            ):
                raise QuestionConflict(
                    "Withdrawal requires a committed native canceled-task receipt"
                )
        now = _timestamp(self._clock())
        async with self.write_transaction():
            changed = await self.backend.execute(
                "UPDATE a2a_native_questions SET status = 'EXPIRED', receiver_settled = ? "
                "WHERE sender_id = ? AND sender_tenant = ? AND question_id = ? "
                "AND status = 'WAITING' AND deadline <= ?",
                (int(withdrawn), *self._owner, question_id, now),
            )
            current = await self.get(question_id)
            if changed == 1 and current is None:
                raise RuntimeError("Committed question expiry was not readable")
            if changed == 1:
                await self._insert_wake(
                    current, "deadline", state="expired", reply="", withdrawn=withdrawn
                )
        if current is None:
            raise QuestionConflict("Question does not belong to this sender")
        if changed == 0 and current.status == "WAITING":
            raise QuestionConflict("Question deadline has not passed")
        return current

    async def settle_unsent(self, question_id: str, dispatch: OutboundDispatch) -> bool:
        """Retire proven local non-delivery without inventing a remote Task."""
        question = await self.get(question_id)
        if (
            question is None
            or (dispatch.sender_id, dispatch.sender_tenant) != self._owner
            or question.dispatch_id != dispatch.dispatch_id
            or (question.recipient_id, question.recipient_tenant)
            != (dispatch.recipient_id, dispatch.tenant)
        ):
            raise QuestionConflict("Question does not match this sender's dispatch")
        if dispatch.state is not DispatchState.NOT_SENT and not (
            dispatch.state is DispatchState.RESERVED
            and question.status == "EXPIRED"
            and self.deadline_passed(question)
        ):
            raise QuestionConflict("Question has no proof of local non-delivery")
        async with self.write_transaction():
            changed = await self.backend.execute(
                "UPDATE a2a_native_questions SET receiver_settled = 1, "
                "status = CASE WHEN status = 'WAITING' THEN 'RESOLVED' ELSE status END "
                "WHERE sender_id = ? AND sender_tenant = ? AND question_id = ? "
                "AND receiver_settled = 0 AND response_digest IS NULL "
                "AND EXISTS (SELECT 1 FROM a2a_outbound_dispatches AS dispatch "
                "WHERE dispatch.sender_id = ? AND dispatch.sender_tenant = ? "
                "AND dispatch.dispatch_id = ? AND dispatch.revision = ? AND dispatch.state = ?)",
                (
                    *self._owner,
                    question_id,
                    *self._owner,
                    dispatch.dispatch_id,
                    dispatch.revision,
                    dispatch.state.value,
                ),
            )
            if changed == 1 and question.status != "EXPIRED":
                await self._insert_wake(
                    question,
                    "answer",
                    state="failed",
                    reply="The original question was not delivered; no remote task exists.",
                    withdrawn=False,
                )
        return changed == 1

    async def _insert_wake(
        self,
        question: NativeQuestion,
        wave: str,
        *,
        state: str,
        reply: str,
        withdrawn: bool,
        requires_full_response: bool = False,
    ):
        event_id = str(
            uuid5(NAMESPACE_URL, json.dumps([*self._owner, question.question_id, wave]))
        )
        payload = {
            "question_id": question.question_id,
            "dispatch_id": question.dispatch_id,
            "recipient_id": question.recipient_id,
            "recipient_tenant": question.recipient_tenant,
            "original_question": question.original_question,
            "request": json.loads(question.request_json)["message"],
            "origin_turn_id": question.origin_turn_id,
            "origin_session_id": question.origin_session_id,
            "causation": json.loads(question.causation_json),
            "state": state,
            "reply_text": reply,
            "late": wave == "late-answer",
            "withdrawn": withdrawn,
            "requires_full_response": requires_full_response,
        }
        await self.backend.execute(
            "INSERT INTO a2a_native_question_wakes (sender_id, sender_tenant, question_id, wave, event_id, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (
                *self._owner,
                question.question_id,
                wave,
                event_id,
                json.dumps(payload, sort_keys=True),
            ),
        )

    async def pending_wakes(self) -> list[QuestionWake]:
        rows = await self.backend.fetch_all(
            f"SELECT {_WAKE_COLUMNS} FROM a2a_native_question_wakes "
            "WHERE sender_id = ? AND sender_tenant = ? AND handed_off_at IS NULL AND delivery_failed_at IS NULL "
            "ORDER BY question_id, CASE wave WHEN 'deadline' THEN 0 ELSE 1 END",
            self._owner,
        )
        return [QuestionWake(*row) for row in rows]

    async def wake_for_event(self, event_id: str) -> QuestionWake | None:
        """Rehydrate one owned durable event even after producer handoff."""
        row = await self.backend.fetch_one(
            f"SELECT {_WAKE_COLUMNS} FROM a2a_native_question_wakes "
            "WHERE sender_id = ? AND sender_tenant = ? AND event_id = ?",
            (*self._owner, event_id),
        )
        return QuestionWake(*row) if row is not None else None

    async def claim_wake(self, wake: QuestionWake) -> QuestionWake | None:
        now = self._clock()
        token = uuid4().hex
        changed = await self.backend.execute(
            "UPDATE a2a_native_question_wakes SET claim_token = ?, lease_until = ? "
            "WHERE sender_id = ? AND sender_tenant = ? AND question_id = ? AND wave = ? "
            "AND event_id = ? AND payload_json = ? AND handed_off_at IS NULL AND delivery_failed_at IS NULL "
            "AND (lease_until IS NULL OR lease_until <= ?) "
            "AND (wave != 'late-answer' OR NOT EXISTS ("
            "SELECT 1 FROM a2a_native_question_wakes earlier "
            "WHERE earlier.sender_id = a2a_native_question_wakes.sender_id "
            "AND earlier.sender_tenant = a2a_native_question_wakes.sender_tenant "
            "AND earlier.question_id = a2a_native_question_wakes.question_id "
            "AND earlier.wave = 'deadline' AND earlier.handed_off_at IS NULL AND earlier.delivery_failed_at IS NULL))",
            (
                token,
                _timestamp(now + self._claim_lease),
                *self._owner,
                wake.question_id,
                wake.wave,
                wake.event_id,
                wake.payload_json,
                _timestamp(now),
            ),
        )
        if changed != 1:
            return None
        return QuestionWake(
            wake.question_id,
            wake.wave,
            wake.event_id,
            wake.payload_json,
            token,
            _timestamp(now + self._claim_lease),
            None,
        )

    async def confirm_handoff(self, wake: QuestionWake) -> bool:
        """Only a committed durable signal admission may retire this outbox entry."""
        return await self._settle_handoff(wake, column="handed_off_at")

    async def record_delivery_failure(self, wake: QuestionWake) -> bool:
        """Retain permanent consumer failure separately from an admission ACK."""
        return await self._settle_handoff(wake, column="delivery_failed_at")

    async def _settle_handoff(self, wake: QuestionWake, *, column: str) -> bool:
        if column not in {"handed_off_at", "delivery_failed_at"}:
            raise ValueError("Unknown durable question handoff disposition")
        if not _text(wake.claim_token):
            raise QuestionConflict("Question wake has no lease claim")
        changed = await self.backend.execute(
            f"UPDATE a2a_native_question_wakes SET {column} = ? "
            "WHERE sender_id = ? AND sender_tenant = ? AND question_id = ? AND wave = ? "
            "AND event_id = ? AND payload_json = ? AND claim_token = ? AND handed_off_at IS NULL AND delivery_failed_at IS NULL",
            (
                _timestamp(self._clock()),
                *self._owner,
                wake.question_id,
                wake.wave,
                wake.event_id,
                wake.payload_json,
                wake.claim_token,
            ),
        )
        return changed == 1
