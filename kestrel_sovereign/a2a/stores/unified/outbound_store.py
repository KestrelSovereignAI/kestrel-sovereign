"""Native outbound dispatch evidence; local IDs never become remote task IDs."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

import rfc8785
from a2a.types import AgentInterface, Message, Role, SendMessageRequest, Task
from a2a.utils.proto_utils import validate_proto_required_fields
from google.protobuf.json_format import MessageToDict

from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES, task_state_label
from kestrel_sovereign.a2a.native_sender import native_request_digest
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase


class DispatchState(StrEnum):
    RESERVED = "reserved"
    SUBMITTING = "submitting"
    UNKNOWN = "unknown"
    ACKNOWLEDGED = "acknowledged"
    COMPLETE = "complete"


class OutboundMigrationRequired(RuntimeError):
    """Draft outbound rows require explicit offline provenance migration."""


class OutboundDispatchConflict(ValueError):
    """An outbound identity, original message or remote receipt conflicts."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Native outbound {field} must be a concrete string")


def _message_digest(message: Message, *, omit_context: bool) -> str:
    normalized = Message.FromString(message.SerializeToString())
    normalized.ClearField("task_id")
    if omit_context:
        normalized.ClearField("context_id")
    return hashlib.sha256(rfc8785.dumps(MessageToDict(normalized))).hexdigest()


@dataclass(frozen=True)
class OutboundDispatch:
    """A sender-owned immutable route plus its current delivery evidence."""

    dispatch_id: str
    sender_id: str
    sender_tenant: str
    recipient_id: str
    tenant: str
    signing_did: str
    interface_url: str
    protocol_binding: str
    message_id: str
    requested_context_id: str
    request_digest: str
    message_digest: str
    dispatch_tool: str
    verb: str
    created_at: str
    state: DispatchState
    revision: int
    remote_task_id: str | None
    remote_context_id: str | None
    remote_task_state: str | None
    response_digest: str | None
    response_kind: str | None
    updated_at: str
    operation_id: str


_COLUMNS = ", ".join(OutboundDispatch.__dataclass_fields__)
_MATCH_FIELDS = tuple(
    field for field in OutboundDispatch.__dataclass_fields__ if field != "revision"
)


class OutboundStore(UnifiedStoreBase):
    """One canonical native ledger, scoped to a host-authenticated sender.

    Message content and authentication headers are not persisted.
    The host supplies its independently verified peer binding before reserve.
    Receipt writes use compare-and-swap and return the exact committed row.
    UNKNOWN has no transition back to SUBMITTING; reconciliation is read-only.
    """

    def __init__(
        self,
        backend,
        *,
        sender_scope: NativeRecipientScope,
        owns_backend: bool = False,
        clock: Callable[[], datetime] = _utc_now,
    ):
        super().__init__(backend, owns_backend=owns_backend)
        if not isinstance(sender_scope, NativeRecipientScope):
            raise TypeError("Native outbound sender requires its host-selected scope")
        self._sender_scope = sender_scope
        self._clock = clock

    def _timestamp(self) -> str:
        timestamp = self._clock()
        if not isinstance(timestamp, datetime) or timestamp.utcoffset() is None:
            raise ValueError(
                "Outbound audit clock must return a timezone-aware datetime"
            )
        return timestamp.astimezone(UTC).isoformat()

    async def initialize(self) -> None:
        if await self.backend.table_exists("a2a_outbound_tasks"):
            legacy = await self.backend.fetch_one(
                "SELECT 1 FROM a2a_outbound_tasks LIMIT 1"
            )
            if legacy is not None:
                raise OutboundMigrationRequired(
                    "Draft A2A outbound rows require offline migration before native startup"
                )
        states = ", ".join(f"'{state.value}'" for state in DispatchState)
        await self.backend.execute_script(f"""
            CREATE TABLE IF NOT EXISTS a2a_outbound_dispatches (
                dispatch_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                sender_tenant TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                tenant TEXT NOT NULL,
                signing_did TEXT NOT NULL,
                interface_url TEXT NOT NULL,
                protocol_binding TEXT NOT NULL,
                message_id TEXT NOT NULL,
                requested_context_id TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                message_digest TEXT NOT NULL,
                dispatch_tool TEXT NOT NULL,
                verb TEXT NOT NULL,
                created_at TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ({states})),
                revision BIGINT NOT NULL CHECK (revision >= 0),
                remote_task_id TEXT,
                remote_context_id TEXT,
                remote_task_state TEXT,
                response_digest TEXT,
                response_kind TEXT,
                updated_at TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                PRIMARY KEY (sender_id, sender_tenant, dispatch_id),
                UNIQUE (sender_id, sender_tenant, message_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS a2a_outbound_remote_task_owner
            ON a2a_outbound_dispatches (sender_id, sender_tenant, recipient_id, tenant, remote_task_id)
            WHERE remote_task_id IS NOT NULL;
        """)

    @staticmethod
    def _snapshot(row) -> OutboundDispatch:
        values = dict(zip(OutboundDispatch.__dataclass_fields__, row, strict=True))
        values["state"] = DispatchState(values["state"])
        return OutboundDispatch(**values)

    async def get(self, dispatch_id: str) -> OutboundDispatch | None:
        row = await self.backend.fetch_one(
            f"SELECT {_COLUMNS} FROM a2a_outbound_dispatches WHERE sender_id = ? AND sender_tenant = ? AND dispatch_id = ?",
            (self._sender_scope.recipient_id, self._sender_scope.tenant, dispatch_id),
        )
        return self._snapshot(row) if row is not None else None

    async def reserve(
        self,
        dispatch_id: str,
        request: SendMessageRequest,
        *,
        recipient: NativeRecipientScope,
        signing_did: str,
        interface: AgentInterface,
        dispatch_tool: str,
        verb: str,
    ) -> OutboundDispatch:
        _text(dispatch_id, "dispatch_id")
        _text(signing_did, "signing_did")
        _text(dispatch_tool, "dispatch_tool")
        if verb not in {"message", "question", "task"}:
            raise ValueError("Native outbound verb must be message, question or task")
        validate_proto_required_fields(interface)
        if (
            interface.protocol_version != "1.0"
            or interface.protocol_binding not in {"JSONRPC", "HTTP+JSON"}
            or interface.tenant != recipient.tenant
            or request.tenant != recipient.tenant
            or request.message.role != Role.ROLE_USER
            or request.message.task_id
        ):
            raise ValueError(
                "Native outbound creation requires the selected native recipient interface"
            )
        digest = native_request_digest(request, "SendMessage")
        message = request.message
        now = self._timestamp()
        reservation = OutboundDispatch(
            dispatch_id=dispatch_id,
            sender_id=self._sender_scope.recipient_id,
            sender_tenant=self._sender_scope.tenant,
            recipient_id=recipient.recipient_id,
            tenant=recipient.tenant,
            signing_did=signing_did,
            interface_url=interface.url,
            protocol_binding=interface.protocol_binding,
            message_id=message.message_id,
            requested_context_id=message.context_id,
            request_digest=digest,
            message_digest=_message_digest(
                message, omit_context=not message.context_id
            ),
            dispatch_tool=dispatch_tool,
            verb=verb,
            created_at=now,
            state=DispatchState.RESERVED,
            revision=0,
            remote_task_id=None,
            remote_context_id=None,
            remote_task_state=None,
            response_digest=None,
            response_kind=None,
            updated_at=now,
            operation_id=uuid4().hex,
        )
        values = tuple(
            getattr(reservation, field)
            for field in OutboundDispatch.__dataclass_fields__
        )
        committed = None
        async with self.write_transaction():
            affected = await self.backend.execute(
                f"INSERT INTO a2a_outbound_dispatches ({_COLUMNS}) "
                f"VALUES ({', '.join('?' for _ in values)}) ON CONFLICT DO NOTHING",
                values,
            )
            if affected == 1:
                committed = await self.get(dispatch_id)
        if affected != 1:
            raise OutboundDispatchConflict(
                "Outbound dispatch or original message is already reserved"
            )
        if committed is None:
            raise RuntimeError("Committed outbound reservation was not readable")
        return committed

    async def _transition(
        self,
        dispatch: OutboundDispatch,
        *,
        sources: tuple[DispatchState, ...],
        state: DispatchState,
        operation_id: str,
        receipt: tuple[str | None, str, str, str, str | None] | None = None,
    ) -> OutboundDispatch | None:
        _text(operation_id, "operation_id")
        if (dispatch.sender_id, dispatch.sender_tenant) != (
            self._sender_scope.recipient_id,
            self._sender_scope.tenant,
        ):
            raise OutboundDispatchConflict(
                "Outbound dispatch belongs to another sender"
            )
        if (
            isinstance(dispatch.revision, bool)
            or not isinstance(dispatch.revision, int)
            or not 0 <= dispatch.revision < 2**63 - 1
        ):
            raise ValueError("Outbound revision must fit a nonnegative SQL integer")
        assignments = (
            "state = ?, revision = revision + 1, operation_id = ?, updated_at = ?"
        )
        values: list[str | int | None] = [
            state.value,
            operation_id,
            self._timestamp(),
        ]
        if receipt is not None:
            assignments += ", remote_task_id = ?, remote_context_id = ?, response_digest = ?, response_kind = ?, remote_task_state = ?"
            values.extend(receipt)
        # Snapshot objects are evidence supplied by our callers, not authority.
        # Match every original binding and previous receipt against the row.
        predicates = []
        for field in _MATCH_FIELDS:
            value = getattr(dispatch, field)
            predicates.append(f"{field} IS NULL" if value is None else f"{field} = ?")
            if value is not None:
                values.append(value)
        values.append(dispatch.revision)
        values.extend(source.value for source in sources)
        committed = None
        async with self.write_transaction():
            affected = await self.backend.execute(
                f"UPDATE a2a_outbound_dispatches SET {assignments} "
                f"WHERE {' AND '.join(predicates)} AND revision = ? "
                f"AND state IN ({', '.join('?' for _ in sources)})",
                tuple(values),
            )
            if affected == 1:
                committed = await self.get(dispatch.dispatch_id)
        if affected == 1 and committed is None:
            raise RuntimeError("Committed outbound transition was not readable")
        return committed

    async def begin_submission(self, dispatch: OutboundDispatch, *, operation_id: str):
        return await self._transition(
            dispatch,
            sources=(DispatchState.RESERVED,),
            state=DispatchState.SUBMITTING,
            operation_id=operation_id,
        )

    async def mark_unknown(self, dispatch: OutboundDispatch, *, operation_id: str):
        return await self._transition(
            dispatch,
            sources=(DispatchState.SUBMITTING,),
            state=DispatchState.UNKNOWN,
            operation_id=operation_id,
        )

    async def acknowledge(
        self,
        dispatch: OutboundDispatch,
        response: Task | Message,
        *,
        operation_id: str,
    ) -> OutboundDispatch | None:
        if not isinstance(response, (Task, Message)):
            raise TypeError(
                "Native outbound acknowledgement requires an official Task or Message"
            )
        validate_proto_required_fields(response)
        if not response.context_id or (
            dispatch.requested_context_id
            and response.context_id != dispatch.requested_context_id
        ):
            raise OutboundDispatchConflict(
                "Peer response changed the requested context"
            )
        if isinstance(response, Task):
            remote_id = response.id
            complete = response.status.state in TERMINAL_STATES
        else:
            if response.role != Role.ROLE_AGENT:
                raise OutboundDispatchConflict(
                    "Peer response message must have the agent role"
                )
            remote_id = response.task_id or None
            complete = remote_id is None
        if (
            dispatch.remote_task_id is not None and remote_id != dispatch.remote_task_id
        ) or (
            dispatch.remote_context_id is not None
            and response.context_id != dispatch.remote_context_id
        ):
            raise OutboundDispatchConflict(
                "Peer receipt changed the acknowledged task or context"
            )
        digest = hashlib.sha256(
            response.SerializeToString(deterministic=True)
        ).hexdigest()
        return await self._transition(
            dispatch,
            sources=(
                DispatchState.SUBMITTING,
                DispatchState.UNKNOWN,
                DispatchState.ACKNOWLEDGED,
            ),
            state=DispatchState.COMPLETE if complete else DispatchState.ACKNOWLEDGED,
            operation_id=operation_id,
            receipt=(
                remote_id,
                response.context_id,
                digest,
                "task" if isinstance(response, Task) else "message",
                task_state_label(response.status.state)
                if isinstance(response, Task)
                else None,
            ),
        )

    async def reconcile(
        self,
        dispatch: OutboundDispatch,
        candidates: list[Task],
        *,
        operation_id: str,
    ) -> OutboundDispatch | None:
        """Admit one exact original-message witness from authorized peer reads.

        Zero matches retain uncertainty. Multiple matching tasks are ambiguous;
        neither condition grants permission to resubmit the original message.
        """
        if dispatch.state is not DispatchState.UNKNOWN:
            raise OutboundDispatchConflict(
                "Only an unknown submission can be reconciled"
            )
        matches = {}
        for task in candidates:
            if not isinstance(task, Task):
                raise TypeError(
                    "Native reconciliation requires official Task snapshots"
                )
            validate_proto_required_fields(task)
            if (
                dispatch.requested_context_id
                and task.context_id != dispatch.requested_context_id
            ):
                continue
            if any(
                message.message_id == dispatch.message_id
                and message.context_id in {"", task.context_id}
                and message.task_id in {"", task.id}
                and _message_digest(
                    message, omit_context=not dispatch.requested_context_id
                )
                == dispatch.message_digest
                for message in task.history
            ):
                previous = matches.get(task.id)
                if previous is not None and previous != task:
                    raise OutboundDispatchConflict(
                        "Conflicting snapshots claim one remote task"
                    )
                matches[task.id] = task
        if not matches:
            return None
        if len(matches) != 1:
            raise OutboundDispatchConflict(
                "More than one remote task claims the original message"
            )
        return await self.acknowledge(
            dispatch, next(iter(matches.values())), operation_id=operation_id
        )
