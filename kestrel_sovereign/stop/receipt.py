"""Durable, idempotent evidence for cooperative Stop operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from kestrel_sovereign.storage.database_clock import database_now_sql

from .types import StopOutcome, StopRequest


_SCHEMA_LOCK = "stop_receipts_v1"
_RECEIPT_COLUMNS = (
    "receipt_id, operation_id, request_fingerprint, scope, actor_id, "
    "requested_target, target_agent_id, reason, cascade, occurred_at, "
    "turn_id, span_id, trace_id"
)
_OUTCOME_COLUMNS = (
    "receipt_id, ordinal, resolved_target, agent_id, disposition, detail"
)


class StopReceiptError(RuntimeError):
    """Base class for durable Stop-evidence failures."""


class StopReceiptConflict(StopReceiptError):
    """One operation identity was reused for a different Stop request."""


class StopReceiptCorruptError(StopReceiptError):
    """Persisted Stop evidence cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class StopOperationClaim:
    """Durable ownership of one operation before cancellation side effects."""

    operation_id: str
    request_fingerprint: str
    claim_id: str


@dataclass(frozen=True, slots=True)
class StopReceipt:
    receipt_id: str
    operation_id: str
    request_fingerprint: str
    scope: str
    actor_id: str
    requested_target: str | None
    target_agent_id: str | None
    reason: str | None
    cascade: bool
    occurred_at: str
    turn_id: str | None
    span_id: str | None
    trace_id: str | None
    outcomes: tuple[StopOutcome, ...]


def _fingerprint(request: StopRequest) -> str:
    semantic_request = request.to_dict()
    # Trace/span identify the transport attempt that first carried an
    # operation; they are evidence, not Stop semantics.  An exact retry may
    # arrive on a new span and must still replay the original receipt.
    semantic_request.pop("span_id", None)
    semantic_request.pop("trace_id", None)
    canonical = json.dumps(
        semantic_request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StopReceiptCorruptError(f"Stop receipt {field} is missing")
    return value


class StopReceiptStore:
    """Append-only Stop receipts stored on an ``AsyncDatabase`` backend."""

    def __init__(self, db: Any):
        self._db = db

    async def ensure_schema(self) -> None:
        async with self._db.migration_lock(_SCHEMA_LOCK):
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_receipts ("
                "receipt_id TEXT NOT NULL PRIMARY KEY, "
                "operation_id TEXT NOT NULL UNIQUE, "
                "request_fingerprint TEXT NOT NULL, "
                "scope TEXT NOT NULL, "
                "actor_id TEXT NOT NULL, "
                "requested_target TEXT, "
                "target_agent_id TEXT, "
                "reason TEXT, "
                "cascade INTEGER NOT NULL, "
                "occurred_at TEXT NOT NULL, "
                "turn_id TEXT, "
                "span_id TEXT, "
                "trace_id TEXT, "
                "CHECK (scope IN ('host', 'agent', 'turn', 'tool_call')), "
                "CHECK (cascade IN (0, 1)))"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_receipt_outcomes ("
                "receipt_id TEXT NOT NULL, "
                "ordinal INTEGER NOT NULL, "
                "resolved_target TEXT NOT NULL, "
                "agent_id TEXT NOT NULL, "
                "disposition TEXT NOT NULL, "
                "detail TEXT, "
                "PRIMARY KEY (receipt_id, ordinal), "
                "FOREIGN KEY (receipt_id) REFERENCES stop_receipts(receipt_id), "
                "CHECK (ordinal >= 0), "
                "CHECK (disposition IN "
                "('stopped', 'already_complete', 'refused', 'unreachable')))"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS stop_operation_claims ("
                "operation_id TEXT NOT NULL PRIMARY KEY, "
                "request_fingerprint TEXT NOT NULL, "
                "claim_id TEXT NOT NULL UNIQUE, "
                "claimed_at TEXT NOT NULL)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_receipts_target "
                "ON stop_receipts(scope, requested_target, occurred_at, receipt_id)"
            )

    async def _lock_operation(self, operation_id: str) -> None:
        if getattr(self._db, "backend_type", "") != "postgres":
            return
        await self._db.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (f"kestrel:stop:operation:{operation_id}",),
        )

    async def load(self, request: StopRequest) -> StopReceipt | None:
        """Return an exact replay or reject conflicting operation reuse."""

        row = await self._db.fetchone(
            f"SELECT {_RECEIPT_COLUMNS} FROM stop_receipts "
            "WHERE operation_id = ?",
            (request.correlation_id,),
        )
        if row is None:
            return None
        receipt = await self._receipt_from_row(row)
        expected = _fingerprint(request)
        self._assert_request_matches_receipt(request, receipt, expected)
        return receipt

    async def claim(
        self,
        request: StopRequest,
    ) -> StopReceipt | StopOperationClaim | None:
        """Claim an operation before effects, replay it, or report in-flight.

        A claim without a receipt is deliberately durable. If an owner dies
        after performing cancellation but before recording its outcomes, an
        exact retry must refuse instead of guessing that the effect is safe to
        repeat.
        """

        fingerprint = _fingerprint(request)
        claim_id = str(uuid4())
        async with self._db.transaction(immediate=True):
            await self._lock_operation(request.correlation_id)
            replay_row = await self._db.fetchone(
                f"SELECT {_RECEIPT_COLUMNS} FROM stop_receipts "
                "WHERE operation_id = ?",
                (request.correlation_id,),
            )
            if replay_row is not None:
                replay = await self._receipt_from_row(replay_row)
                self._assert_request_matches_receipt(
                    request, replay, fingerprint
                )
                return replay

            claim_row = await self._db.fetchone(
                "SELECT request_fingerprint, claim_id "
                "FROM stop_operation_claims WHERE operation_id = ?",
                (request.correlation_id,),
            )
            if claim_row is not None:
                if claim_row[0] != fingerprint:
                    raise StopReceiptConflict(
                        "Stop operation identity was reused for a different request"
                    )
                return None

            now_sql = database_now_sql(self._db)
            await self._db.execute(
                "INSERT INTO stop_operation_claims ("
                "operation_id, request_fingerprint, claim_id, claimed_at"
                f") VALUES (?, ?, ?, {now_sql})",
                (request.correlation_id, fingerprint, claim_id),
            )
            return StopOperationClaim(
                operation_id=request.correlation_id,
                request_fingerprint=fingerprint,
                claim_id=claim_id,
            )

    async def persist(
        self,
        request: StopRequest,
        outcomes: tuple[StopOutcome, ...],
        *,
        claim_id: str | None = None,
    ) -> StopReceipt:
        """Atomically append one request and its ordered per-target outcomes."""

        self._validate_outcomes(request, outcomes)
        fingerprint = _fingerprint(request)
        receipt_id = str(uuid4())
        try:
            async with self._db.transaction(immediate=True):
                await self._lock_operation(request.correlation_id)
                replay_row = await self._db.fetchone(
                    f"SELECT {_RECEIPT_COLUMNS} FROM stop_receipts "
                    "WHERE operation_id = ?",
                    (request.correlation_id,),
                )
                if replay_row is not None:
                    replay = await self._receipt_from_row(replay_row)
                    self._assert_request_matches_receipt(
                        request, replay, fingerprint
                    )
                    return replay

                claim_row = await self._db.fetchone(
                    "SELECT request_fingerprint, claim_id "
                    "FROM stop_operation_claims WHERE operation_id = ?",
                    (request.correlation_id,),
                )
                claim_matches = bool(
                    claim_row is not None
                    and len(claim_row) == 2
                    and claim_row[0] == fingerprint
                    and claim_row[1] == claim_id
                )
                if claim_row is not None and not claim_matches:
                    raise StopReceiptConflict(
                        "Stop operation claim is owned elsewhere"
                    )
                if claim_id is not None and not claim_matches:
                    raise StopReceiptConflict(
                        "Stop operation claim is missing"
                    )

                now_sql = database_now_sql(self._db)
                await self._db.execute(
                    "INSERT INTO stop_receipts ("
                    "receipt_id, operation_id, request_fingerprint, scope, "
                    "actor_id, requested_target, target_agent_id, reason, "
                    "cascade, occurred_at, turn_id, span_id, trace_id"
                    f") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {now_sql}, ?, ?, ?)",
                    (
                        receipt_id,
                        request.correlation_id,
                        fingerprint,
                        request.scope.value,
                        request.actor_id,
                        request.target,
                        request.target_agent_id,
                        request.reason,
                        int(request.cascade),
                        request.turn_id,
                        request.span_id,
                        request.trace_id,
                    ),
                )
                for ordinal, outcome in enumerate(outcomes):
                    await self._db.execute(
                        "INSERT INTO stop_receipt_outcomes ("
                        "receipt_id, ordinal, resolved_target, agent_id, "
                        "disposition, detail) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            receipt_id,
                            ordinal,
                            outcome.resolved_target,
                            outcome.agent_id,
                            outcome.disposition.value,
                            outcome.detail,
                        ),
                    )
                stored_row = await self._db.fetchone(
                    f"SELECT {_RECEIPT_COLUMNS} FROM stop_receipts "
                    "WHERE receipt_id = ?",
                    (receipt_id,),
                )
                stored = await self._receipt_from_row(stored_row)
                self._assert_request_matches_receipt(
                    request, stored, fingerprint
                )
                if claim_id is not None:
                    deleted = await self._db.execute(
                        "DELETE FROM stop_operation_claims "
                        "WHERE operation_id = ? AND claim_id = ?",
                        (request.correlation_id, claim_id),
                    )
                    if deleted != 1:
                        raise StopReceiptConflict(
                            "Stop operation claim changed before receipt commit"
                        )
                return stored
        except Exception as error:
            domain = self._domain_error(error)
            if domain is not None:
                raise domain from error
            # A concurrent exact replay can lose the unique-key race on a
            # backend without advisory locks. Read it back only after the
            # failed transaction has rolled back.
            replay = await self.load(request)
            if replay is not None:
                return replay
            raise

    async def _receipt_from_row(self, row: Any) -> StopReceipt:
        if row is None or len(row) != 13:
            raise StopReceiptCorruptError(
                "Stop receipt row has an unexpected shape"
            )
        receipt_id = _required_text(row[0], "receipt_id")
        operation_id = _required_text(row[1], "operation_id")
        fingerprint = _required_text(row[2], "request_fingerprint")
        scope = _required_text(row[3], "scope")
        actor_id = _required_text(row[4], "actor_id")
        occurred_at = _required_text(row[9], "occurred_at")
        try:
            cascade_int = int(row[8])
        except (TypeError, ValueError) as error:
            raise StopReceiptCorruptError(
                "Stop receipt cascade flag is invalid"
            ) from error
        if scope not in {"host", "agent", "turn", "tool_call"}:
            raise StopReceiptCorruptError("Stop receipt scope is invalid")
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise StopReceiptCorruptError("Stop receipt fingerprint is invalid")
        if cascade_int not in (0, 1):
            raise StopReceiptCorruptError("Stop receipt cascade flag is invalid")
        for field_name, value in (
            ("target_agent_id", row[6]),
            ("reason", row[7]),
            ("span_id", row[11]),
            ("trace_id", row[12]),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise StopReceiptCorruptError(
                    f"Stop receipt {field_name} is invalid"
                )
        if row[5] is not None and (
            not isinstance(row[5], str) or not row[5]
        ):
            raise StopReceiptCorruptError(
                "Stop receipt requested_target is invalid"
            )
        if row[10] is not None and (
            not isinstance(row[10], str) or not row[10]
        ):
            raise StopReceiptCorruptError("Stop receipt turn_id is invalid")
        outcome_rows = await self._db.fetchall(
            f"SELECT {_OUTCOME_COLUMNS} FROM stop_receipt_outcomes "
            "WHERE receipt_id = ? ORDER BY ordinal",
            (receipt_id,),
        )
        outcomes: list[StopOutcome] = []
        for expected_ordinal, outcome_row in enumerate(outcome_rows):
            if outcome_row is None or len(outcome_row) != 6:
                raise StopReceiptCorruptError(
                    "Stop outcome row has an unexpected shape"
                )
            if str(outcome_row[0]) != receipt_id:
                raise StopReceiptCorruptError("Stop outcome receipt link is invalid")
            try:
                ordinal = int(outcome_row[1])
            except (TypeError, ValueError) as error:
                raise StopReceiptCorruptError(
                    "Stop outcome ordinal is invalid"
                ) from error
            if ordinal != expected_ordinal:
                raise StopReceiptCorruptError("Stop outcome order is not contiguous")
            try:
                outcomes.append(
                    StopOutcome.from_dict(
                        {
                            "scope": scope,
                            "requested_target": row[5],
                            "resolved_target": outcome_row[2],
                            "agent_id": outcome_row[3],
                            "disposition": outcome_row[4],
                            "correlation_id": operation_id,
                            "detail": outcome_row[5],
                            "receipt_id": receipt_id,
                        }
                    )
                )
            except (TypeError, ValueError) as error:
                raise StopReceiptCorruptError(
                    "Stop outcome row violates its typed contract"
                ) from error
        if scope != "host" and not outcomes:
            raise StopReceiptCorruptError(
                "Non-host Stop receipt is missing its target outcome"
            )
        return StopReceipt(
            receipt_id=receipt_id,
            operation_id=operation_id,
            request_fingerprint=fingerprint,
            scope=scope,
            actor_id=actor_id,
            requested_target=row[5],
            target_agent_id=row[6],
            reason=row[7],
            cascade=bool(cascade_int),
            occurred_at=occurred_at,
            turn_id=row[10],
            span_id=row[11],
            trace_id=row[12],
            outcomes=tuple(outcomes),
        )

    @staticmethod
    def _assert_request_matches_receipt(
        request: StopRequest,
        receipt: StopReceipt,
        fingerprint: str,
    ) -> None:
        if receipt.request_fingerprint != fingerprint:
            raise StopReceiptConflict(
                "Stop operation identity was reused for a different request"
            )
        recorded = (
            receipt.operation_id,
            receipt.scope,
            receipt.actor_id,
            receipt.requested_target,
            receipt.target_agent_id,
            receipt.reason,
            receipt.cascade,
            receipt.turn_id,
        )
        supplied = (
            request.correlation_id,
            request.scope.value,
            request.actor_id,
            request.target,
            request.target_agent_id,
            request.reason,
            request.cascade,
            request.turn_id,
        )
        if recorded != supplied:
            raise StopReceiptCorruptError(
                "Stop receipt header does not match its request fingerprint"
            )

    @staticmethod
    def _validate_outcomes(
        request: StopRequest,
        outcomes: tuple[StopOutcome, ...],
    ) -> None:
        if not isinstance(outcomes, tuple):
            raise TypeError("Stop receipt outcomes must be a tuple")
        if request.scope.value != "host" and not outcomes:
            raise ValueError("Non-host Stop requires one target outcome")
        seen: set[str] = set()
        for outcome in outcomes:
            if not isinstance(outcome, StopOutcome):
                raise TypeError("Stop receipt received an untyped outcome")
            if (
                outcome.scope is not request.scope
                or outcome.requested_target != request.target
                or outcome.correlation_id != request.correlation_id
                or outcome.receipt_id is not None
            ):
                raise ValueError("Stop outcome does not belong to this request")
            if outcome.resolved_target in seen:
                raise ValueError("Stop receipt contains a duplicate target")
            seen.add(outcome.resolved_target)

    @staticmethod
    def _domain_error(error: BaseException) -> StopReceiptError | None:
        current: BaseException | None = error
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, StopReceiptError):
                return current
            current = current.__cause__ or current.__context__
        return None


class UnavailableStopReceiptStore:
    """Fail-closed writer used when the host evidence store did not open."""

    def __init__(self, reason: str = "Stop receipt storage is unavailable"):
        self._reason = reason

    async def load(self, _request: StopRequest) -> None:
        raise StopReceiptError(self._reason)

    async def persist(
        self,
        _request: StopRequest,
        _outcomes: tuple[StopOutcome, ...],
        **_kwargs: Any,
    ) -> StopReceipt:
        raise StopReceiptError(self._reason)


__all__ = [
    "StopReceipt",
    "StopOperationClaim",
    "StopReceiptConflict",
    "StopReceiptCorruptError",
    "StopReceiptError",
    "StopReceiptStore",
    "UnavailableStopReceiptStore",
]
