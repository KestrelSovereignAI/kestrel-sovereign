"""Durable, idempotent evidence for cooperative Stop operations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    raise_owned_outcome,
)
from kestrel_sovereign.agent.invocation import validate_invocation_id
from kestrel_sovereign.storage.database_clock import (
    database_lease_cutoff_sql,
    database_now_sql,
    database_timestamp_bound_text,
)
from kestrel_sovereign.storage.db.interface import DatabaseError
from kestrel_sovereign.storage.feed_sequence import ensure_feed_sequence

from .types import StopDoor, StopOutcome, StopRequest, StopScope

_SCHEMA_LOCK = "stop_receipts_v1"
# The lease every Stop owner holds on durable shared state: a distributed
# invocation owner's generations and an operation claim's owner (#3356) alike.
# Liveness is a heartbeat inside this lease, observed with the database clock.
STOP_OWNER_LEASE_SECONDS = 2.0
# A claim owner renews several times per lease, so one late renewal does not
# let a live owner's claim be taken over.
_CLAIM_RENEWALS_PER_LEASE = 4
# Additive (#3356): who holds an operation claim and when it last proved it
# was alive. Nullable: a claim written before liveness existed has neither, and
# its writer -- a binary without this code -- cannot still be renewing it, so
# it reads as expired and a retry may take it over.
_CLAIM_OWNER_COLUMNS = (
    ("owner_id", "TEXT"),
    ("heartbeat_at", "TEXT"),
)
_DOOR_VALUES = frozenset(door.value for door in StopDoor)
# Additive (#3170): which door wrote a receipt. Nullable so an older binary
# that never names it keeps appending during a rolling upgrade; such a row
# reads back as "door not recorded", never as a guessed door.
_DOOR_COLUMN = (
    "door",
    "TEXT CHECK (door IS NULL OR door IN ('peer', 'agent', 'host'))",
)
# Pre-#3170 rows carry no door. Only host scope proves its door: the host
# fan-out is its sole writer. An agent/turn row cannot be attributed -- the
# operator door records the caller's identity as its actor, and a caller may
# itself be a DID, so the actor's shape says nothing about which door ran.
# Those rows stay NULL and read back as "door not recorded". The backfill is
# provenance for display only: the circuit breaker counts its own admissions,
# never a backfilled row.
_DOOR_BACKFILL = (
    "UPDATE stop_receipts SET door = 'host' "
    "WHERE door IS NULL AND scope = 'host'",
    (),
)
_RECEIPT_COLUMNS = (
    "receipt_id, operation_id, request_fingerprint, scope, actor_id, "
    "requested_target, target_agent_id, reason, cascade, occurred_at, "
    "turn_id, span_id, trace_id"
)
_OUTCOME_COLUMNS = (
    "receipt_id, ordinal, resolved_target, agent_id, disposition, detail"
)
_OPAQUE_ID_DOMAIN = b"kestrel:stop-receipt-opaque-id:v1\0"
# Serializes every receipt append on PostgreSQL from ``feed_seq`` allocation to
# commit, so the feed's keyset order is commit order (#3159 R6). The allocating
# insert trigger takes it, so a writer holds it AFTER its per-operation lock;
# the schema backfill takes it inside the schema lock, which no writer holds,
# so the two orders cannot cycle.
_RECEIPT_FEED_LOCK_KEY = "kestrel:stop:receipt-feed"
# How long a claim written before owner liveness (#3356) is presumed live. Such
# a claim has no heartbeat to read, and during a rolling upgrade its writer may
# still be running. Pre-liveness Stop code bounds its own execution by its wait
# ceilings (seconds, not minutes), so a claim older than this cannot belong to a
# Stop still in progress and becomes retakable.
_LEGACY_CLAIM_GRACE_SECONDS = 300.0


logger = logging.getLogger(__name__)


class StopReceiptError(RuntimeError):
    """Base class for durable Stop-evidence failures."""


class StopReceiptConflict(StopReceiptError):
    """One operation identity was reused for a different Stop request."""


class StopReceiptCorruptError(StopReceiptError):
    """Persisted Stop evidence cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class StopOperationClaim:
    """Durable ownership of one operation before cancellation side effects.

    ``taken_over`` is true when this claim replaced one whose owner was proven
    dead (its lease expired): the Stop is executed again and records what it
    observes now.
    """

    operation_id: str
    request_fingerprint: str
    claim_id: str
    taken_over: bool = False


@dataclass(frozen=True, slots=True)
class StopReceiptOutcomeRecord:
    """One per-target outcome read back WITHOUT the originating request.

    ``resolved_target`` and ``agent_id`` are ``None`` when the row holds only
    the blinded digest of a caller-supplied address. A reader has no key to
    reverse it, and presenting a digest as an identity would invent one the
    receipt never recorded (#3159 R3).
    """

    ordinal: int
    resolved_target: str | None
    agent_id: str | None
    disposition: str
    detail: str | None


@dataclass(frozen=True, slots=True)
class StopReceiptRecord:
    """One immutable Stop receipt as the sovereign read surface renders it.

    ``feed_seq`` is the commit-ordered paging key; ``occurred_at`` is the
    database clock's reading and is for display, not for ordering.
    """

    feed_seq: int
    receipt_id: str
    scope: str
    door: str | None
    actor_id: str
    target_agent_id: str | None
    reason: str | None
    cascade: bool
    occurred_at: str
    span_id: str | None
    trace_id: str | None
    outcomes: tuple[StopReceiptOutcomeRecord, ...]


@dataclass(frozen=True, slots=True)
class StopReceiptPage:
    """One bounded page plus the keyset position that continues it."""

    receipts: tuple[StopReceiptRecord, ...]
    next_key: int | None


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
    # Trace/span/turn identify evidence inferred from the transport attempt;
    # they are not Stop semantics. An exact request-ID retry can arrive after
    # the live turn index is gone and must still replay the original receipt.
    semantic_request.pop("span_id", None)
    semantic_request.pop("trace_id", None)
    semantic_request.pop("turn_id", None)
    if request.scope is StopScope.AGENT:
        # An agent-scope request addresses its agent by ``target``. The DID in
        # ``target_agent_id`` is what the authority RESOLVED that address to
        # (#3159 R3) — evidence about the inventory at Stop time, exactly like
        # the trace above. A retry after that agent was unloaded resolves to
        # nothing and is still the same request.
        semantic_request["target_agent_id"] = None
    canonical = json.dumps(
        semantic_request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _identifier_digest(kind: str, value: str) -> str:
    """Return a domain-separated, non-reversible durable lookup identity."""

    encoded_kind = kind.encode("ascii")
    encoded_value = value.encode("utf-8")
    digest = hashlib.sha256(
        _OPAQUE_ID_DOMAIN
        + len(encoded_kind).to_bytes(2, "big")
        + encoded_kind
        + len(encoded_value).to_bytes(4, "big")
        + encoded_value
    ).hexdigest()
    return f"sha256:{digest}"


def opaque_stop_identifier(kind: str, value: str) -> str:
    """Share the receipt store's one-way lookup identity inside Stop."""

    return _identifier_digest(kind, value)


def _optional_identifier_digest(kind: str, value: str | None) -> str | None:
    return None if value is None else _identifier_digest(kind, value)


def _request_target_digest_kind(request: StopRequest) -> str:
    """Keep public turn handles disjoint from private request addresses."""

    return "public_turn_target" if request.target_is_turn_id else "target"


def _stored_outcome_identity(
    value: str,
    request: StopRequest,
    *,
    field: str,
) -> str:
    if request.target is not None and value == request.target:
        return _identifier_digest(_request_target_digest_kind(request), value)
    if field == "resolved_target" and request.target_is_turn_id:
        # A public turn address resolves to the process-private request key
        # used to cancel the live operation. That key may be caller supplied;
        # it is necessary in memory but is not durable receipt content.
        return _identifier_digest("resolved_target", value)
    return value


def _public_outcome_identity(value: str, request: StopRequest) -> str:
    if request.target is not None and value == _identifier_digest(
        _request_target_digest_kind(request), request.target
    ):
        return request.target
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StopReceiptCorruptError(f"Stop receipt {field} is missing")
    return value


def _required_opaque_identifier(value: object, field: str) -> str:
    try:
        return validate_invocation_id(value)
    except ValueError as error:
        raise StopReceiptCorruptError(
            f"Stop receipt {field} is invalid"
        ) from error


class StopReceiptStore:
    """Append-only Stop receipts stored on an ``AsyncDatabase`` backend.

    ``owner_id`` names this store's process as the owner of the operation
    claims it takes; each instance draws a fresh one, so a restarted host is a
    new owner and never inherits a previous boot's claims. ``claim_lease_seconds``
    is how long a claim stays live without a heartbeat.
    """

    def __init__(
        self,
        db: Any,
        *,
        claim_lease_seconds: float = STOP_OWNER_LEASE_SECONDS,
    ):
        if (
            not isinstance(claim_lease_seconds, (int, float))
            or isinstance(claim_lease_seconds, bool)
            or not math.isfinite(claim_lease_seconds)
            or claim_lease_seconds <= 0
        ):
            raise ValueError("Stop claim lease must be positive and finite")
        self._db = db
        self._owner_id = uuid4().hex
        self._claim_lease_seconds = float(claim_lease_seconds)

    @property
    def owner_id(self) -> str:
        """The owner identity this store writes into the claims it takes."""

        return self._owner_id

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
                "CREATE TABLE IF NOT EXISTS stop_operation_bindings ("
                "operation_id TEXT NOT NULL PRIMARY KEY, "
                "request_fingerprint TEXT NOT NULL, "
                "bound_at TEXT NOT NULL)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_stop_receipts_target "
                "ON stop_receipts(scope, requested_target, occurred_at, receipt_id)"
            )
            # The commit-ordered key the sovereign receipt feed pages on
            # (#3159 R2/R6), with its unique index. Numbering pre-existing rows
            # must exclude a concurrent append from another process, which
            # holds the feed lock but never this schema lock.
            await self._lock_receipt_feed()
            await ensure_feed_sequence(
                self._db,
                table="stop_receipts",
                lock_key=_RECEIPT_FEED_LOCK_KEY,
            )
        await self._db.migrate_columns_once(
            "stop_receipts",
            (_DOOR_COLUMN,),
            {"door": _DOOR_BACKFILL},
        )
        await self._db.migrate_columns_once(
            "stop_operation_claims", _CLAIM_OWNER_COLUMNS
        )

    async def _lock_operation(self, operation_id: str) -> None:
        if getattr(self._db, "backend_type", "") != "postgres":
            return
        await self._db.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (
                "kestrel:stop:operation:"
                f"{_identifier_digest('operation', operation_id)}",
            ),
        )

    async def _lock_receipt_feed(self) -> None:
        """Hold PostgreSQL's receipt-append slot until this transaction ends.

        Per-operation locks let two different Stops commit concurrently. Then
        transaction A can allocate a smaller key than B, B commits and is
        paged, and A commits BEHIND the cursor a consumer already holds —
        skipped forever. One writer at a time from ``feed_seq`` allocation to
        commit makes every later receipt sort after every earlier one.

        Appends take this inside the allocating insert trigger
        (:mod:`kestrel_sovereign.storage.feed_sequence`), which is what also
        serializes an older binary that never heard of the lock. Only the
        schema backfill takes it here. SQLite needs no key: ``BEGIN
        IMMEDIATE`` already grants the database's single writer slot.
        """

        if getattr(self._db, "backend_type", "") != "postgres":
            return
        await self._db.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            (_RECEIPT_FEED_LOCK_KEY,),
        )

    async def load(self, request: StopRequest) -> StopReceipt | None:
        """Return an exact replay or reject conflicting operation reuse."""

        row = await self._db.fetchone(
            f"SELECT {_RECEIPT_COLUMNS} FROM stop_receipts "
            "WHERE operation_id = ?",
            (_identifier_digest("operation", request.correlation_id),),
        )
        if row is None:
            return None
        receipt = await self._receipt_from_row(row, request=request)
        expected = _fingerprint(request)
        self._assert_request_matches_receipt(request, receipt, expected)
        return receipt

    async def bind_operation(self, request: StopRequest) -> None:
        """Bind an operation identity to its request at first sight.

        A door that commits other durable evidence of an operation before the
        operation is claimed -- peer Stop's signal source event -- binds the
        request first.  Every later sight of the same identity must then name
        the same request: a different one raises :class:`StopReceiptConflict`
        here and in :meth:`claim`, so an interrupted first attempt can only be
        completed by its own intent, never taken over by a changed one.  Only
        the request fingerprint is stored.
        """

        fingerprint = _fingerprint(request)
        stored_operation_id = _identifier_digest(
            "operation", request.correlation_id
        )
        try:
            async with self._db.transaction(immediate=True):
                await self._lock_operation(request.correlation_id)
                for table in ("stop_receipts", "stop_operation_claims"):
                    row = await self._db.fetchone(
                        f"SELECT request_fingerprint FROM {table} "
                        "WHERE operation_id = ?",
                        (stored_operation_id,),
                    )
                    if row is not None and row[0] != fingerprint:
                        raise StopReceiptConflict(
                            "Stop operation identity was reused for a "
                            "different request"
                        )
                if await self._binding_matches(stored_operation_id, fingerprint):
                    return
                now_sql = database_now_sql(self._db)
                await self._db.execute(
                    "INSERT INTO stop_operation_bindings ("
                    "operation_id, request_fingerprint, bound_at"
                    f") VALUES (?, ?, {now_sql})",
                    (stored_operation_id, fingerprint),
                )
        except Exception as error:
            domain = self._domain_error(error)
            if domain is not None:
                raise domain from error
            raise

    async def _binding_matches(
        self, stored_operation_id: str, fingerprint: str
    ) -> bool:
        """``True`` if bound to this request, ``False`` if unbound.

        A binding to a different request is a conflict, raised here so that
        every writer under the operation identity honors the first sight.
        """

        row = await self._db.fetchone(
            "SELECT request_fingerprint FROM stop_operation_bindings "
            "WHERE operation_id = ?",
            (stored_operation_id,),
        )
        if row is None:
            return False
        if row[0] != fingerprint:
            raise StopReceiptConflict(
                "Stop operation identity is bound to a different request"
            )
        return True

    async def has_acknowledged_turn_stop(
        self,
        agent_id: str,
        turn_id: str,
    ) -> bool:
        """Whether durable evidence forbids this exact turn from starting.

        A pre-registration Stop truthfully records ``already_complete`` because
        no live generation existed yet. That outcome still acknowledges the
        caller's exact-ID andon cord, so a delayed transport delivery must not
        resurrect the named turn after the short in-memory race fence expires.
        Refused or unreachable outcomes never establish this admission fence.
        """

        agent_id = _required_text(agent_id, "target agent identity")
        turn_id = _required_opaque_identifier(turn_id, "turn identity")
        row = await self._db.fetchone(
            "SELECT 1 FROM stop_receipts AS receipt "
            "JOIN stop_receipt_outcomes AS outcome "
            "ON outcome.receipt_id = receipt.receipt_id "
            "WHERE receipt.scope = 'turn' "
            "AND receipt.target_agent_id = ? "
            "AND receipt.requested_target = ? "
            "AND outcome.disposition IN ('stopped', 'already_complete') "
            "LIMIT 1",
            (agent_id, _identifier_digest("target", turn_id)),
        )
        return row is not None

    async def list_receipts(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        agent_id: str | None = None,
        trace_id: str | None = None,
        after: int | None = None,
        limit: int,
    ) -> StopReceiptPage:
        """Read one bounded page of immutable Stop evidence, oldest first.

        Ordering is ``feed_seq``, which the table's insert trigger allocates
        in commit order for every writer, including an older binary's. That is what makes the feed converge under late or out-of-order
        telemetry: the same window always yields the same rows in the same
        order however a consumer walked to it, and a receipt committed after a
        page was read always sorts after that page's cursor instead of being
        skipped by it (#3159 R6). ``since`` / ``until`` filter on the displayed
        ``occurred_at``.

        ``agent_id`` matches the receipt's own recorded target agent OR any of
        its per-target outcomes, so a host-scope fan-out — whose header names
        no single agent — is still findable by the agent it reached.

        A backend that cannot answer raises :class:`StopReceiptError`, never a
        raw ``DatabaseError``: the caller's promise is a 503 saying the history
        is unreadable, and an untyped backend failure would escape that
        translation and arrive as a sanitized 500 instead.
        """

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("Stop receipt page size must be a positive integer")

        try:
            return await self._read_receipt_page(
                since=since,
                until=until,
                agent_id=agent_id,
                trace_id=trace_id,
                after=after,
                limit=limit,
            )
        except StopReceiptError:
            # Corruption and conflict are already the precise answer. Only an
            # untyped backend failure needs a name.
            raise
        except DatabaseError as error:
            raise StopReceiptError(
                "Durable Stop evidence could not be read"
            ) from error

    async def _read_receipt_page(
        self,
        *,
        since: datetime | None,
        until: datetime | None,
        agent_id: str | None,
        trace_id: str | None,
        after: int | None,
        limit: int,
    ) -> StopReceiptPage:
        filters = ["receipt.feed_seq IS NOT NULL"]
        params: list[Any] = []
        if since is not None:
            filters.append("receipt.occurred_at >= ?")
            params.append(
                database_timestamp_bound_text(self._db, since)
            )
        if until is not None:
            filters.append("receipt.occurred_at < ?")
            params.append(
                database_timestamp_bound_text(self._db, until)
            )
        if trace_id is not None:
            filters.append("receipt.trace_id = ?")
            params.append(trace_id)
        if agent_id is not None:
            filters.append(
                "(receipt.target_agent_id = ? OR EXISTS ("
                "SELECT 1 FROM stop_receipt_outcomes AS scoped "
                "WHERE scoped.receipt_id = receipt.receipt_id "
                "AND scoped.agent_id = ?))"
            )
            params.extend((agent_id, agent_id))
        if after is not None:
            # Strict keyset successor: a page boundary can neither repeat a
            # receipt nor skip one.
            filters.append("receipt.feed_seq > ?")
            params.append(after)

        # One row over the page so the cursor is emitted only when more
        # evidence actually exists.
        params.append(limit + 1)
        rows = await self._db.fetchall(
            f"SELECT {_RECEIPT_COLUMNS}, feed_seq, door "
            "FROM stop_receipts AS receipt "
            f"WHERE {' AND '.join(filters)} "
            "ORDER BY receipt.feed_seq LIMIT ?",
            tuple(params),
        )
        rows = tuple(rows)
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        receipts = [
            await self._record_from_row(row) for row in page_rows
        ]
        next_key = receipts[-1].feed_seq if has_more and receipts else None
        return StopReceiptPage(receipts=tuple(receipts), next_key=next_key)

    async def _record_from_row(self, row: Any) -> StopReceiptRecord:
        """Project one stored receipt WITHOUT its originating request.

        :meth:`_receipt_from_row` exists to prove a retry against the request
        that produced it. A history reader has no such request, so this path
        validates the row's own shape and deliberately makes no attempt to
        de-blind an opaque address.
        """

        if row is None or len(row) != 15:
            raise StopReceiptCorruptError(
                "Stop receipt row has an unexpected shape"
            )
        feed_seq = row[13]
        door = row[14]
        if door is not None and door not in _DOOR_VALUES:
            raise StopReceiptCorruptError("Stop receipt door is invalid")
        if (
            isinstance(feed_seq, bool)
            or not isinstance(feed_seq, int)
            or feed_seq < 1
        ):
            raise StopReceiptCorruptError("Stop receipt feed sequence is invalid")
        receipt_id = _required_text(row[0], "receipt_id")
        scope = _required_text(row[3], "scope")
        if scope not in {"host", "agent", "turn", "tool_call"}:
            raise StopReceiptCorruptError("Stop receipt scope is invalid")
        try:
            cascade_int = int(row[8])
        except (TypeError, ValueError) as error:
            raise StopReceiptCorruptError(
                "Stop receipt cascade flag is invalid"
            ) from error
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
        outcome_rows = await self._db.fetchall(
            f"SELECT {_OUTCOME_COLUMNS} FROM stop_receipt_outcomes "
            "WHERE receipt_id = ? ORDER BY ordinal",
            (receipt_id,),
        )
        outcomes: list[StopReceiptOutcomeRecord] = []
        for expected_ordinal, outcome_row in enumerate(outcome_rows):
            if outcome_row is None or len(outcome_row) != 6:
                raise StopReceiptCorruptError(
                    "Stop outcome row has an unexpected shape"
                )
            try:
                ordinal = int(outcome_row[1])
            except (TypeError, ValueError) as error:
                raise StopReceiptCorruptError(
                    "Stop outcome ordinal is invalid"
                ) from error
            if ordinal != expected_ordinal:
                raise StopReceiptCorruptError(
                    "Stop outcome order is not contiguous"
                )
            disposition = _required_text(outcome_row[4], "outcome disposition")
            if disposition not in {
                "stopped",
                "already_complete",
                "refused",
                "unreachable",
            }:
                raise StopReceiptCorruptError(
                    "Stop outcome disposition is invalid"
                )
            outcomes.append(
                StopReceiptOutcomeRecord(
                    ordinal=ordinal,
                    resolved_target=_required_text(
                        outcome_row[2], "outcome resolved_target"
                    ),
                    agent_id=_required_text(outcome_row[3], "outcome agent_id"),
                    disposition=disposition,
                    detail=outcome_row[5],
                )
            )
        if not outcomes:
            raise StopReceiptCorruptError(
                "Stop receipt is missing its target outcome"
            )
        return StopReceiptRecord(
            feed_seq=feed_seq,
            receipt_id=receipt_id,
            scope=scope,
            door=door,
            actor_id=_required_text(row[4], "actor_id"),
            target_agent_id=row[6],
            reason=row[7],
            cascade=bool(cascade_int),
            occurred_at=_required_text(row[9], "occurred_at"),
            span_id=row[11],
            trace_id=row[12],
            outcomes=tuple(outcomes),
        )

    async def claim(
        self,
        request: StopRequest,
    ) -> StopReceipt | StopOperationClaim | None:
        """Claim an operation before effects, replay it, or report in-flight.

        Returns the stored receipt for an exact replay, a new
        :class:`StopOperationClaim` when this caller now owns the operation,
        or ``None`` when a LIVE owner holds it ("already in progress").

        A claim is live while its owner's heartbeat is inside the claim lease,
        read with the database clock. A claim whose owner is proven dead -- its
        heartbeat expired, or it predates owner liveness -- is taken over
        (#3356): the retry re-executes the Stop and records what it observes
        now. That is safe because Stop is idempotent. Cancelling work an
        earlier owner already cancelled yields ``already_complete``, never a
        second effect. A durable refusal would instead leave the operation
        unstoppable and its receipt forever unwritten.

        The takeover is one compare-and-set on the observed claim under the
        per-operation lock, so concurrent retries against one expired claim
        yield exactly one new owner. The dead owner cannot come back and
        write: :meth:`persist` deletes only its own ``claim_id``, which the
        takeover replaced.
        """

        fingerprint = _fingerprint(request)
        stored_operation_id = _identifier_digest(
            "operation", request.correlation_id
        )
        claim_id = str(uuid4())
        try:
            async with self._db.transaction(immediate=True):
                await self._lock_operation(request.correlation_id)
                replay_row = await self._db.fetchone(
                    f"SELECT {_RECEIPT_COLUMNS} FROM stop_receipts "
                    "WHERE operation_id = ?",
                    (stored_operation_id,),
                )
                if replay_row is not None:
                    replay = await self._receipt_from_row(
                        replay_row, request=request
                    )
                    self._assert_request_matches_receipt(
                        request, replay, fingerprint
                    )
                    return replay

                await self._binding_matches(stored_operation_id, fingerprint)
                claim_row = await self._db.fetchone(
                    "SELECT request_fingerprint, claim_id "
                    "FROM stop_operation_claims WHERE operation_id = ?",
                    (stored_operation_id,),
                )
                now_sql = database_now_sql(self._db)
                if claim_row is not None:
                    if claim_row[0] != fingerprint:
                        raise StopReceiptConflict(
                            "Stop operation identity was reused for a different request"
                        )
                    cutoff_sql, cutoff_args = database_lease_cutoff_sql(
                        self._db, self._claim_lease_seconds
                    )
                    # A claim with no heartbeat was written by a binary that
                    # predates owner liveness. During a rolling upgrade that
                    # binary may still be executing its Stop, so its claim is
                    # treated as live until it is older than any Stop that code
                    # could still be running, and only then retakable.
                    legacy_sql, legacy_args = database_lease_cutoff_sql(
                        self._db, _LEGACY_CLAIM_GRACE_SECONDS
                    )
                    taken = await self._db.execute(
                        "UPDATE stop_operation_claims SET claim_id = ?, "
                        f"owner_id = ?, claimed_at = {now_sql}, "
                        f"heartbeat_at = {now_sql} "
                        "WHERE operation_id = ? AND claim_id = ? "
                        f"AND ((heartbeat_at IS NULL AND claimed_at <= {legacy_sql}) "
                        f"OR heartbeat_at <= {cutoff_sql})",
                        (
                            claim_id,
                            self._owner_id,
                            stored_operation_id,
                            claim_row[1],
                            *legacy_args,
                            *cutoff_args,
                        ),
                    )
                    if taken == 0:
                        return None
                    if taken != 1:
                        raise StopReceiptCorruptError(
                            "Stop operation claim takeover changed multiple rows"
                        )
                    logger.warning(
                        "Stop operation claim taken over from an owner whose "
                        "lease expired; re-executing the Stop"
                    )
                    return StopOperationClaim(
                        operation_id=request.correlation_id,
                        request_fingerprint=fingerprint,
                        claim_id=claim_id,
                        taken_over=True,
                    )

                await self._db.execute(
                    "INSERT INTO stop_operation_claims ("
                    "operation_id, request_fingerprint, claim_id, claimed_at, "
                    "owner_id, heartbeat_at"
                    f") VALUES (?, ?, ?, {now_sql}, ?, {now_sql})",
                    (stored_operation_id, fingerprint, claim_id, self._owner_id),
                )
                return StopOperationClaim(
                    operation_id=request.correlation_id,
                    request_fingerprint=fingerprint,
                    claim_id=claim_id,
                )
        except Exception as error:
            # The transaction wrapper re-raises as a backend error; a typed
            # conflict must stay typed so callers do not read it as an
            # unavailable store.
            domain = self._domain_error(error)
            if domain is not None:
                raise domain from error
            raise

    async def renew_claim(self, claim: StopOperationClaim) -> bool:
        """Extend a still-live claim's lease; ``False`` once it is not live.

        Renewal is non-revivable, like an invocation owner's heartbeat: it
        succeeds only while this owner's heartbeat is still inside the lease.
        An owner that resumes after its lease expired has lost the claim even
        if no retry has taken it over yet, and must not make itself look alive
        again.
        """

        if not isinstance(claim, StopOperationClaim):
            raise TypeError("renew_claim requires a StopOperationClaim")
        stored_operation_id = _identifier_digest(
            "operation", claim.operation_id
        )
        async with self._db.transaction(immediate=True):
            now_sql = database_now_sql(self._db)
            cutoff_sql, cutoff_args = database_lease_cutoff_sql(
                self._db, self._claim_lease_seconds
            )
            renewed = await self._db.execute(
                "UPDATE stop_operation_claims "
                f"SET heartbeat_at = {now_sql} "
                "WHERE operation_id = ? AND claim_id = ? AND owner_id = ? "
                f"AND heartbeat_at > {cutoff_sql}",
                (
                    stored_operation_id,
                    claim.claim_id,
                    self._owner_id,
                    *cutoff_args,
                ),
            )
            if renewed == 1:
                return True
            if renewed != 0:
                raise StopReceiptCorruptError(
                    "Stop operation claim renewal changed multiple rows"
                )
            still_claimed = await self._db.fetchone(
                "SELECT 1 FROM stop_operation_claims WHERE operation_id = ?",
                (stored_operation_id,),
            )
        if still_claimed is not None:
            # Gone would mean the receipt committed; still claimed means the
            # lease expired or another owner took the operation over.
            logger.warning(
                "Stop operation claim lease was lost before its receipt "
                "committed; a retry may take the operation over"
            )
        return False

    @asynccontextmanager
    async def hold_claim(
        self, claim: StopOperationClaim
    ) -> AsyncIterator[None]:
        """Keep ``claim`` live while its owner executes the Stop.

        A renewal task heartbeats the claim several times per lease until the
        block exits, and stops for good once a renewal finds the claim no
        longer live (expired, taken over, or released by its receipt). Exiting
        joins that task before returning, even under cancellation, so no
        renewal outlives its owner and nothing renews a claim after its
        receipt committed.
        """

        if not isinstance(claim, StopOperationClaim):
            raise TypeError("hold_claim requires a StopOperationClaim")
        released = asyncio.Event()
        interval = self._claim_lease_seconds / _CLAIM_RENEWALS_PER_LEASE

        async def renew_until_released() -> None:
            while True:
                try:
                    await asyncio.wait_for(released.wait(), timeout=interval)
                    return
                except TimeoutError:
                    pass
                try:
                    if not await self.renew_claim(claim):
                        return
                except Exception:  # a missed heartbeat, not a crash
                    # Retry on the next tick: renewal is non-revivable, so a
                    # failure that outlasts the lease ends here as a lost
                    # claim, which a retry may then take over. The owner keeps
                    # executing either way; its receipt commit is fenced.
                    logger.warning(
                        "Stop operation claim heartbeat failed", exc_info=True
                    )

        renewal = asyncio.create_task(
            renew_until_released(), name="stop-operation-claim-lease"
        )
        try:
            yield
        finally:
            released.set()
            raise_owned_outcome(
                await await_owned_task(renewal),
                operation="Stop operation claim lease",
            )

    async def persist(
        self,
        request: StopRequest,
        outcomes: tuple[StopOutcome, ...],
        *,
        door: StopDoor,
        claim_id: str | None = None,
    ) -> StopReceipt:
        """Atomically append one request and its ordered per-target outcomes.

        ``door`` is required: every writer names the door its Stop arrived
        through, so the fleet circuit breaker (#3170) can never mistake an
        operator's Stop for a peer's.
        """

        if not isinstance(door, StopDoor):
            raise TypeError("Stop receipt door must be a StopDoor")
        self._validate_outcomes(request, outcomes)
        fingerprint = _fingerprint(request)
        stored_operation_id = _identifier_digest(
            "operation", request.correlation_id
        )
        receipt_id = str(uuid4())
        try:
            async with self._db.transaction(immediate=True):
                await self._lock_operation(request.correlation_id)
                replay_row = await self._db.fetchone(
                    f"SELECT {_RECEIPT_COLUMNS} FROM stop_receipts "
                    "WHERE operation_id = ?",
                    (stored_operation_id,),
                )
                if replay_row is not None:
                    replay = await self._receipt_from_row(
                        replay_row, request=request
                    )
                    self._assert_request_matches_receipt(
                        request, replay, fingerprint
                    )
                    return replay

                await self._binding_matches(stored_operation_id, fingerprint)
                claim_row = await self._db.fetchone(
                    "SELECT request_fingerprint, claim_id "
                    "FROM stop_operation_claims WHERE operation_id = ?",
                    (stored_operation_id,),
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

                # ``feed_seq`` is deliberately absent: the insert trigger
                # allocates it under the receipt-feed lock, held from here to
                # commit, so this receipt's feed position is its commit order.
                now_sql = database_now_sql(self._db)
                await self._db.execute(
                    "INSERT INTO stop_receipts ("
                    "receipt_id, operation_id, request_fingerprint, scope, "
                    "actor_id, requested_target, target_agent_id, reason, "
                    "cascade, occurred_at, turn_id, span_id, trace_id, door"
                    f") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {now_sql}, ?, ?, ?, ?)",
                    (
                        receipt_id,
                        stored_operation_id,
                        fingerprint,
                        request.scope.value,
                        request.actor_id,
                        _optional_identifier_digest(
                            _request_target_digest_kind(request),
                            request.target,
                        ),
                        request.target_agent_id,
                        request.reason,
                        int(request.cascade),
                        _optional_identifier_digest(
                            _request_target_digest_kind(request),
                            request.turn_id,
                        ),
                        request.span_id,
                        request.trace_id,
                        door.value,
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
                            _stored_outcome_identity(
                                outcome.resolved_target,
                                request,
                                field="resolved_target",
                            ),
                            _stored_outcome_identity(
                                outcome.agent_id,
                                request,
                                field="agent_id",
                            ),
                            outcome.disposition.value,
                            outcome.detail,
                        ),
                    )
                stored_row = await self._db.fetchone(
                    f"SELECT {_RECEIPT_COLUMNS} FROM stop_receipts "
                    "WHERE receipt_id = ?",
                    (receipt_id,),
                )
                stored = await self._receipt_from_row(
                    stored_row, request=request
                )
                self._assert_request_matches_receipt(
                    request, stored, fingerprint
                )
                if claim_id is not None:
                    deleted = await self._db.execute(
                        "DELETE FROM stop_operation_claims "
                        "WHERE operation_id = ? AND claim_id = ?",
                        (stored_operation_id, claim_id),
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

    async def _receipt_from_row(
        self, row: Any, *, request: StopRequest
    ) -> StopReceipt:
        if row is None or len(row) != 13:
            raise StopReceiptCorruptError(
                "Stop receipt row has an unexpected shape"
            )
        receipt_id = _required_text(row[0], "receipt_id")
        stored_operation_id = _required_text(row[1], "operation_id")
        fingerprint = _required_text(row[2], "request_fingerprint")
        scope = _required_text(row[3], "scope")
        actor_id = _required_text(row[4], "actor_id")
        occurred_at = _required_text(row[9], "occurred_at")
        expected_operation_id = _identifier_digest(
            "operation", request.correlation_id
        )
        expected_requested_target = _optional_identifier_digest(
            _request_target_digest_kind(request), request.target
        )
        # ``StopRequest`` defaults a private request-addressed turn's
        # ``turn_id`` to its request target.  That placeholder is not evidence
        # that an earlier live-index lookup must have recorded the same public
        # turn address, so an exact replay after the index disappears cannot
        # compare it to the stored inferred address.
        expected_turn_id = (
            None
            if not request.target_is_turn_id and request.turn_id == request.target
            else _optional_identifier_digest(
                _request_target_digest_kind(request), request.turn_id
            )
        )
        if stored_operation_id != expected_operation_id:
            raise StopReceiptCorruptError(
                "Stop receipt operation lookup identity is invalid"
            )
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise StopReceiptCorruptError("Stop receipt fingerprint is invalid")
        if fingerprint != _fingerprint(request):
            raise StopReceiptConflict(
                "Stop operation identity was reused for a different request"
            )
        if row[5] != expected_requested_target or (
            expected_turn_id is not None and row[10] != expected_turn_id
        ):
            raise StopReceiptCorruptError(
                "Stop receipt opaque target identity is invalid"
            )
        try:
            cascade_int = int(row[8])
        except (TypeError, ValueError) as error:
            raise StopReceiptCorruptError(
                "Stop receipt cascade flag is invalid"
            ) from error
        if scope not in {"host", "agent", "turn", "tool_call"}:
            raise StopReceiptCorruptError("Stop receipt scope is invalid")
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
            not isinstance(row[10], str)
            or not row[10].startswith("sha256:")
            or len(row[10]) != 71
            or any(
                character not in "0123456789abcdef"
                for character in row[10][7:]
            )
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
                            "requested_target": request.target,
                            "resolved_target": _public_outcome_identity(
                                outcome_row[2], request
                            ),
                            "agent_id": _public_outcome_identity(
                                outcome_row[3], request
                            ),
                            "disposition": outcome_row[4],
                            "correlation_id": request.correlation_id,
                            "detail": outcome_row[5],
                            "receipt_id": receipt_id,
                        }
                    )
                )
            except (TypeError, ValueError) as error:
                raise StopReceiptCorruptError(
                    "Stop outcome row violates its typed contract"
                ) from error
        if not outcomes:
            raise StopReceiptCorruptError(
                "Stop receipt is missing its target outcome"
            )
        return StopReceipt(
            receipt_id=receipt_id,
            operation_id=request.correlation_id,
            request_fingerprint=fingerprint,
            scope=scope,
            actor_id=actor_id,
            requested_target=request.target,
            target_agent_id=row[6],
            reason=row[7],
            cascade=bool(cascade_int),
            occurred_at=occurred_at,
            # Durable evidence must come from the receipt row, not from the
            # retry object used to locate it. The stored value is the blinded
            # canonical identity; the clear caller token remains on
            # ``StopRequest`` and was proven equal by the digest check above.
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
        # An agent-scope ``target_agent_id`` is resolution evidence, not part
        # of the request (see ``_fingerprint``): a retry replays whatever the
        # original resolution recorded, including nothing at all.
        agent_evidence = request.scope is StopScope.AGENT
        recorded = (
            receipt.operation_id,
            receipt.scope,
            receipt.actor_id,
            receipt.requested_target,
            None if agent_evidence else receipt.target_agent_id,
            receipt.reason,
            receipt.cascade,
        )
        supplied = (
            request.correlation_id,
            request.scope.value,
            request.actor_id,
            request.target,
            None if agent_evidence else request.target_agent_id,
            request.reason,
            request.cascade,
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
        if not outcomes:
            raise ValueError("Stop requires at least one target outcome")
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

    async def bind_operation(self, _request: StopRequest) -> None:
        raise StopReceiptError(self._reason)

    async def persist(
        self,
        _request: StopRequest,
        _outcomes: tuple[StopOutcome, ...],
        **_kwargs: Any,
    ) -> StopReceipt:
        raise StopReceiptError(self._reason)

    async def list_receipts(self, **_kwargs: Any) -> StopReceiptPage:
        # A history nobody can read is reported as unreadable, never as an
        # empty history (#3159 R2).
        raise StopReceiptError(self._reason)


__all__ = [
    "StopReceipt",
    "StopOperationClaim",
    "StopReceiptConflict",
    "StopReceiptCorruptError",
    "StopReceiptError",
    "StopReceiptOutcomeRecord",
    "StopReceiptPage",
    "StopReceiptRecord",
    "StopReceiptStore",
    "UnavailableStopReceiptStore",
]
