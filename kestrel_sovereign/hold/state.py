"""Typed, durable host and agent Hold latches.

A Hold is state, not a cancellation event.  The current latch and its immutable
mutation receipts live in the host control database so every worker observes
the same answer after a restart.  This module owns only storage and composition;
turn-start refusal is the separate enforcement seam tracked by #3162.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Mapping, Optional
from uuid import UUID, uuid4

try:  # pragma: no cover - exercised on Kestrel's POSIX deployment targets
    import fcntl
except ImportError:  # pragma: no cover - Windows has no POSIX advisory locks
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no cover - imported only on Windows
    import msvcrt
except ImportError:  # pragma: no cover - POSIX has no Windows byte-range locks
    msvcrt = None  # type: ignore[assignment]

from kestrel_sovereign.private_storage import (
    PrivateStorageError,
    absolute_without_following_leaf,
    open_private_file,
    path_exists,
)
from kestrel_sovereign.storage.database_clock import database_now_sql


HOST_HOLD_TARGET = "host"
_SCHEMA_LOCK = "hold_state_v1"
_HISTORY_LOCK_KEY = "kestrel:hold:history-anchor"
_WITNESS_BACKFILL = "hold_state_witness_ledgers_v1"
_INITIALIZATION_WITNESS_PAYLOAD = b"kestrel-hold-state-initialized-v1\n"
_BOOTSTRAP_INTENT_PAYLOAD = b"kestrel-hold-bootstrap-pending-v1\n"
_HISTORY_ANCHOR_HEADER = b"kestrel-hold-history-v1\n"
_HISTORY_ANCHOR_MAX_BYTES = 256
_BOOTSTRAP_INTENT_MAX_BYTES = (
    len(_BOOTSTRAP_INTENT_PAYLOAD) + _HISTORY_ANCHOR_MAX_BYTES
)
_EVIDENCE_LOCK_POLL_SECONDS = 0.01
_POSTGRES_WITNESS_AGENT_ID = "__kestrel_host_control__"
_POSTGRES_WITNESS_KEY = "hold_schema_initialized_v1"
_POSTGRES_HISTORY_ANCHOR_KEY = "hold_history_anchor_v1"
_POSTGRES_HISTORY_CANDIDATE_KEY = "hold_history_candidate_v1"
_POSTGRES_BOOTSTRAP_INTENT_KEY = "hold_bootstrap_pending_v1"
_POSTGRES_ROLLBACK_DOMAIN_KEY = "hold_rollback_domain_id_v1"
_POSTGRES_ROLLBACK_DOMAIN_PREFIX = "kestrel-hold-rollback-domain-v1:"
_POSTGRES_PRIMARY_BINDING_KEY = "hold_primary_custody_binding_v1"
_POSTGRES_EVIDENCE_BINDING_KEY = "hold_evidence_custody_binding_v1"
_POSTGRES_CUSTODY_BINDING_PREFIX = "kestrel-hold-custody-binding-v1:"
# Serialize the first core-schema publication independently on each PostgreSQL
# database.  PostgreSQL's CREATE TABLE IF NOT EXISTS catalog probe can race a
# peer cold start, so migration_lock() cannot be the first lock: its own table
# does not exist yet.
_POSTGRES_SCHEMA_BOOTSTRAP_LOCK = (0x004B4553, 0x5343484D)
# Two signed int32 values spelling ``KES`` / ``HOLD``. The lock lives on the
# independent evidence service and spans both primary commit and publication.
_POSTGRES_EVIDENCE_LOCK = (0x004B4553, 0x484F4C44)
_HOLD_SCHEMA_TABLES = frozenset(
    {
        "hold_latches",
        "hold_receipts",
        "hold_receipt_witnesses",
        "hold_receipt_content_witnesses",
        "hold_operation_witnesses",
        "hold_schema_migrations",
    }
)
_LATCH_COLUMNS = (
    "scope, target_id, active, hold_receipt_id, reason, actor_id, set_at, revision"
)
_RECEIPT_COLUMNS = (
    "receipt_id, operation_id, action, disposition, scope, target_id, reason, "
    "actor_id, occurred_at, expected_hold_receipt_id, prior_hold_receipt_id, "
    "resulting_hold_receipt_id"
)


class HoldScope(str, Enum):
    """The two scopes on which a durable Hold may latch."""

    HOST = "host"
    AGENT = "agent"


class HoldAction(str, Enum):
    HOLD = "hold"
    RELEASE = "release"


class HoldDisposition(str, Enum):
    APPLIED = "applied"
    ALREADY_IN_STATE = "already_in_state"
    REFUSED_STALE = "refused_stale"


class HoldStateError(RuntimeError):
    """Base class for a durable Hold-state failure."""


class HoldIdempotencyConflict(HoldStateError):
    """One operation id was reused for a different mutation."""


class HoldCorruptStateError(HoldStateError):
    """Persisted Hold state cannot be interpreted safely."""


@dataclass(frozen=True)
class PostgresHoldCustodySnapshot:
    """Read-only identity and role evidence from one PostgreSQL database."""

    cluster_identity: str
    domain_identity: str | None = None
    primary_binding: str | None = None
    evidence_binding: str | None = None


def postgres_hold_custody_binding_payload(
    pair_id: UUID,
    primary_identity: str,
    evidence_identity: str,
) -> str:
    """Encode the durable declaration binding both custody databases."""

    return (
        _POSTGRES_CUSTODY_BINDING_PREFIX
        + str(pair_id)
        + "|"
        + primary_identity
        + "|"
        + evidence_identity
    )


def _validate_postgres_domain_identity(identity: str, *, label: str) -> None:
    if not identity.startswith(_POSTGRES_ROLLBACK_DOMAIN_PREFIX):
        raise HoldStateError(
            f"could not verify PostgreSQL Hold {label} rollback domain"
        )
    try:
        parsed = UUID(identity.removeprefix(_POSTGRES_ROLLBACK_DOMAIN_PREFIX))
    except (ValueError, AttributeError) as exc:
        raise HoldStateError(
            f"could not verify PostgreSQL Hold {label} rollback domain"
        ) from exc
    if identity != _POSTGRES_ROLLBACK_DOMAIN_PREFIX + str(parsed):
        raise HoldStateError(
            f"could not verify PostgreSQL Hold {label} rollback domain"
        )


def _validate_postgres_custody_binding(
    payload: str,
    *,
    primary_identity: str,
    evidence_identity: str,
) -> UUID:
    if not payload.startswith(_POSTGRES_CUSTODY_BINDING_PREFIX):
        raise HoldStateError("PostgreSQL Hold custody binding is invalid")
    parts = payload.removeprefix(_POSTGRES_CUSTODY_BINDING_PREFIX).split("|")
    if len(parts) != 3:
        raise HoldStateError("PostgreSQL Hold custody binding is invalid")
    pair, bound_primary, bound_evidence = parts
    try:
        pair_id = UUID(pair)
    except ValueError as exc:
        raise HoldStateError("PostgreSQL Hold custody binding is invalid") from exc
    if str(pair_id) != pair or (
        bound_primary != primary_identity or bound_evidence != evidence_identity
    ):
        raise HoldStateError(
            "PostgreSQL Hold custody binding does not match the configured pair"
        )
    return pair_id


def validate_postgres_hold_custody(
    primary: PostgresHoldCustodySnapshot,
    evidence: PostgresHoldCustodySnapshot,
) -> None:
    """Fail closed on a read-only snapshot before either schema is mutated.

    A brand-new pair has no metadata table yet and is valid to initialize. Once
    either side contains role evidence, the persisted domains and pair binding
    must describe exactly the configured primary/evidence ordering.
    """

    if (
        not isinstance(primary.cluster_identity, str)
        or not primary.cluster_identity.strip()
    ):
        raise HoldStateError(
            "could not verify PostgreSQL Hold primary cluster identity"
        )
    if (
        not isinstance(evidence.cluster_identity, str)
        or not evidence.cluster_identity.strip()
    ):
        raise HoldStateError(
            "could not verify PostgreSQL Hold evidence cluster identity"
        )
    if primary.cluster_identity == evidence.cluster_identity:
        raise HoldStateError(
            "PostgreSQL Hold evidence requires an independent PostgreSQL cluster"
        )

    if (
        primary.evidence_binding is not None
        or evidence.primary_binding is not None
    ):
        raise HoldStateError(
            "PostgreSQL Hold database has the wrong durable custody role"
        )

    for label, identity in (
        ("primary", primary.domain_identity),
        ("evidence", evidence.domain_identity),
    ):
        if identity is not None:
            if not isinstance(identity, str):
                raise HoldStateError(
                    f"could not verify PostgreSQL Hold {label} rollback domain"
                )
            _validate_postgres_domain_identity(identity, label=label)

    if (
        primary.domain_identity is not None
        and primary.domain_identity == evidence.domain_identity
    ):
        raise HoldStateError(
            "PostgreSQL Hold evidence must use an independent rollback domain"
        )

    expected = tuple(
        binding
        for binding in (primary.primary_binding, evidence.evidence_binding)
        if binding is not None
    )
    if not expected:
        return
    if primary.domain_identity is None or evidence.domain_identity is None:
        raise HoldStateError(
            "PostgreSQL Hold custody binding lacks a durable rollback domain"
        )
    for binding in expected:
        if not isinstance(binding, str):
            raise HoldStateError("PostgreSQL Hold custody binding is invalid")
        _validate_postgres_custody_binding(
            binding,
            primary_identity=primary.domain_identity,
            evidence_identity=evidence.domain_identity,
        )
    if len(expected) == 2 and expected[0] != expected[1]:
        raise HoldStateError(
            "PostgreSQL Hold custody binding disagrees between databases"
        )


def postgres_hold_custody_snapshot_from_rows(
    cluster_identity: str,
    rows: list[Any] | tuple[Any, ...],
    *,
    label: str,
) -> PostgresHoldCustodySnapshot:
    """Parse the three allowlisted custody keys from read-only query rows."""

    values: dict[str, str] = {}
    allowed = {
        _POSTGRES_ROLLBACK_DOMAIN_KEY,
        _POSTGRES_PRIMARY_BINDING_KEY,
        _POSTGRES_EVIDENCE_BINDING_KEY,
    }
    for row in rows:
        if (
            len(row) != 2
            or not isinstance(row[0], str)
            or row[0] not in allowed
            or row[0] in values
            or not isinstance(row[1], str)
        ):
            raise HoldStateError(
                f"PostgreSQL Hold {label} custody metadata is invalid"
            )
        values[row[0]] = row[1]
    return PostgresHoldCustodySnapshot(
        cluster_identity=cluster_identity,
        domain_identity=values.get(_POSTGRES_ROLLBACK_DOMAIN_KEY),
        primary_binding=values.get(_POSTGRES_PRIMARY_BINDING_KEY),
        evidence_binding=values.get(_POSTGRES_EVIDENCE_BINDING_KEY),
    )


async def _gather_database_probes(
    *probes: Awaitable[Any],
) -> tuple[Any, ...]:
    """Cancel and await every sibling before a failed probe leaves its scope."""

    tasks = tuple(asyncio.ensure_future(probe) for probe in probes)
    try:
        return tuple(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        cleanup = asyncio.gather(*tasks, return_exceptions=True)
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # Repeated cancellation must not let a database probe escape
                # into host-context pool cleanup. Re-raise the original failure
                # only after every owned child has reached a terminal state.
                continue
        await cleanup
        raise


@asynccontextmanager
async def _postgres_custody_locks(
    primary_db: Any,
    evidence_db: Any,
    *,
    primary_cluster: str,
    evidence_cluster: str,
):
    """Lock both custody clusters in one deterministic global order.

    The same pair can be presented in opposite roles by two concurrent cold
    starts. Locking only the configured evidence side would then give each
    process a different serialization point and let both publish incompatible
    roles. Cluster identities are immutable PostgreSQL ``initdb`` identities,
    so sorting them gives every role ordering the same acquisition order.
    """

    if primary_cluster == evidence_cluster:
        raise HoldStateError(
            "PostgreSQL Hold evidence requires an independent PostgreSQL cluster"
        )
    databases = sorted(
        (
            (primary_cluster, "primary", primary_db),
            (evidence_cluster, "evidence", evidence_db),
        ),
        key=lambda item: item[0],
    )
    async with AsyncExitStack() as stack:
        for _cluster, label, database in databases:
            lock_owner = getattr(database, "backend", None) or database
            locks = getattr(lock_owner, "advisory_locks", None)
            if not callable(locks):
                raise HoldStateError(
                    f"PostgreSQL Hold {label} database cannot provide advisory locks"
                )
            await stack.enter_async_context(locks((_POSTGRES_EVIDENCE_LOCK,)))
        yield


async def _read_raw_postgres_cluster_identity(
    backend: Any,
    *,
    label: str,
) -> str:
    """Return the cluster identity from one connected raw PostgreSQL pool."""

    try:
        cluster_rows = await backend.fetch_all(
            "SELECT system_identifier::text FROM pg_catalog.pg_control_system()"
        )
    except Exception as exc:
        raise HoldStateError(
            f"could not verify PostgreSQL Hold {label} cluster identity; "
            "the runtime role requires EXECUTE on "
            "pg_catalog.pg_control_system()"
        ) from exc
    if (
        len(cluster_rows) != 1
        or len(cluster_rows[0]) != 1
        or not isinstance(cluster_rows[0][0], str)
        or not cluster_rows[0][0].strip()
    ):
        raise HoldStateError(
            f"could not verify PostgreSQL Hold {label} cluster identity"
        )
    return cluster_rows[0][0]


async def _read_postgres_hold_custody_snapshot(
    backend: Any,
    *,
    label: str,
) -> PostgresHoldCustodySnapshot:
    """Read custody evidence without creating or changing database objects."""

    cluster_identity = await _read_raw_postgres_cluster_identity(
        backend,
        label=label,
    )

    try:
        table_rows = await backend.fetch_all(
            "SELECT to_regclass('agent_metadata')::text"
        )
    except Exception as exc:
        raise HoldStateError(
            f"could not inspect PostgreSQL Hold {label} custody metadata"
        ) from exc
    if len(table_rows) != 1 or len(table_rows[0]) != 1:
        raise HoldStateError(
            f"could not inspect PostgreSQL Hold {label} custody metadata"
        )
    table_name = table_rows[0][0]
    if table_name is None:
        return PostgresHoldCustodySnapshot(cluster_identity=cluster_identity)
    if not isinstance(table_name, str) or not table_name.strip():
        raise HoldStateError(
            f"could not inspect PostgreSQL Hold {label} custody metadata"
        )

    try:
        rows = await backend.fetch_all(
            "SELECT key, value FROM agent_metadata "
            "WHERE agent_id = ? AND key IN (?, ?, ?)",
            (
                _POSTGRES_WITNESS_AGENT_ID,
                _POSTGRES_ROLLBACK_DOMAIN_KEY,
                _POSTGRES_PRIMARY_BINDING_KEY,
                _POSTGRES_EVIDENCE_BINDING_KEY,
            ),
        )
    except Exception as exc:
        raise HoldStateError(
            f"could not inspect PostgreSQL Hold {label} custody metadata"
        ) from exc

    return postgres_hold_custody_snapshot_from_rows(
        cluster_identity,
        rows,
        label=label,
    )


async def _close_postgres_preflight_backends(
    *backends: Any,
) -> tuple[BaseException, ...]:
    """Finish every raw-pool close even under repeated caller cancellation."""

    async def close_all() -> tuple[Any, ...]:
        return tuple(
            await asyncio.gather(
                *(backend.close() for backend in backends),
                return_exceptions=True,
            )
        )

    cleanup = asyncio.create_task(close_all())
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
            continue
    results = cleanup.result()
    if cancelled:
        raise asyncio.CancelledError()
    return tuple(result for result in results if isinstance(result, BaseException))


async def preflight_postgres_hold_custody(
    primary_dsn: str,
    evidence_dsn: str,
) -> None:
    """Verify existing custody roles through raw, read-only PostgreSQL pools."""

    primary_backend, evidence_backend = (
        await _connect_postgres_hold_custody_backends(
            primary_dsn,
            evidence_dsn,
        )
    )
    close_errors = await _close_postgres_preflight_backends(
        primary_backend,
        evidence_backend,
    )
    if close_errors:
        raise HoldStateError(
            "could not close PostgreSQL Hold custody preflight connections"
        ) from close_errors[0]


async def _connect_postgres_hold_custody_backends(
    primary_dsn: str,
    evidence_dsn: str,
) -> tuple[Any, Any]:
    """Return the exact connected pools whose custody roles were validated."""

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    primary_backend = PostgresBackend(
        dsn=primary_dsn,
        min_pool_size=1,
        max_pool_size=1,
    )
    evidence_backend = PostgresBackend(
        dsn=evidence_dsn,
        min_pool_size=1,
        max_pool_size=1,
    )
    try:
        await _gather_database_probes(
            primary_backend.connect(),
            evidence_backend.connect(),
        )
        primary_cluster, evidence_cluster = await _gather_database_probes(
            _read_raw_postgres_cluster_identity(
                primary_backend,
                label="primary",
            ),
            _read_raw_postgres_cluster_identity(
                evidence_backend,
                label="evidence",
            ),
        )
        async with _postgres_custody_locks(
            primary_backend,
            evidence_backend,
            primary_cluster=primary_cluster,
            evidence_cluster=evidence_cluster,
        ):
            # Re-read the entire pair only after both clusters are locked. A
            # concurrent first boot publishes domain and role rows in stages;
            # two unlocked per-database reads can otherwise construct a state
            # that never existed as one custody snapshot.
            primary, evidence = await _gather_database_probes(
                _read_postgres_hold_custody_snapshot(
                    primary_backend,
                    label="primary",
                ),
                _read_postgres_hold_custody_snapshot(
                    evidence_backend,
                    label="evidence",
                ),
            )
            if (
                primary.cluster_identity != primary_cluster
                or evidence.cluster_identity != evidence_cluster
            ):
                raise HoldStateError(
                    "PostgreSQL Hold cluster identity changed while acquiring "
                    "custody locks"
                )
            validate_postgres_hold_custody(primary, evidence)
    except BaseException as failure:
        close_errors: tuple[BaseException, ...] = ()
        try:
            close_errors = await _close_postgres_preflight_backends(
                primary_backend,
                evidence_backend,
            )
        except asyncio.CancelledError as cancellation:
            failure = cancellation
        for close_error in close_errors:
            failure.add_note(
                "Additional PostgreSQL Hold custody preflight close failure: "
                f"{close_error!r}"
            )
        raise failure.with_traceback(failure.__traceback__)
    return primary_backend, evidence_backend


async def initialize_postgres_hold_databases(
    primary_dsn: str,
    evidence_dsn: str,
) -> tuple[Any, Any]:
    """Validate, then initialize schema on those same connected PG pools."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase

    primary_backend, evidence_backend = (
        await _connect_postgres_hold_custody_backends(
            primary_dsn,
            evidence_dsn,
        )
    )
    primary_db = None
    evidence_db = None
    try:
        # Lock each database separately rather than holding both locks at once.
        # Besides bounding the critical section, this avoids a deadlock if two
        # still-unbound databases are accidentally presented in opposite roles
        # by concurrent starts.  Custody-role validation below remains the
        # authority that rejects that topology.
        primary_db = await AsyncDatabase.from_connected_backend(
            primary_backend,
            initialization_guard=primary_backend.advisory_locks(
                (_POSTGRES_SCHEMA_BOOTSTRAP_LOCK,)
            ),
        )
        evidence_db = await AsyncDatabase.from_connected_backend(
            evidence_backend,
            initialization_guard=evidence_backend.advisory_locks(
                (_POSTGRES_SCHEMA_BOOTSTRAP_LOCK,)
            ),
        )
        return primary_db, evidence_db
    except BaseException as failure:
        close_errors: tuple[BaseException, ...] = ()
        try:
            close_errors = await _close_postgres_preflight_backends(
                primary_db or primary_backend,
                evidence_db or evidence_backend,
            )
        except asyncio.CancelledError as cancellation:
            failure = cancellation
        for close_error in close_errors:
            failure.add_note(
                "Additional PostgreSQL Hold database initialization close failure: "
                f"{close_error!r}"
            )
        raise failure.with_traceback(failure.__traceback__)


def _domain_error_from_chain(error: BaseException) -> Optional[HoldStateError]:
    """Recover a typed Hold failure wrapped by a transaction backend."""

    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, HoldStateError):
            return current
        current = current.__cause__ or current.__context__
    return None


@dataclass(frozen=True)
class HoldState:
    scope: HoldScope
    target_id: str
    reason: str
    actor_id: str
    set_at: str
    hold_receipt_id: str
    revision: int


@dataclass(frozen=True)
class EffectiveHoldState:
    """Independent latches applying to one agent."""

    host: Optional[HoldState]
    agent: Optional[HoldState]

    @property
    def held(self) -> bool:
        return self.host is not None or self.agent is not None

    @property
    def sources(self) -> tuple[HoldScope, ...]:
        sources: list[HoldScope] = []
        if self.host is not None:
            sources.append(HoldScope.HOST)
        if self.agent is not None:
            sources.append(HoldScope.AGENT)
        return tuple(sources)


@dataclass(frozen=True)
class HoldReceipt:
    receipt_id: str
    operation_id: str
    action: HoldAction
    disposition: HoldDisposition
    scope: HoldScope
    target_id: str
    reason: str
    actor_id: str
    occurred_at: str
    expected_hold_receipt_id: str
    prior_hold_receipt_id: str
    resulting_hold_receipt_id: str


@dataclass(frozen=True)
class HoldMutation:
    receipt: HoldReceipt
    current: Optional[HoldState]


def hold_initialization_witness_path(control_db_path: str | Path) -> Path:
    """Return the external witness paired with one host control database."""

    path = absolute_without_following_leaf(Path(control_db_path))
    return Path(f"{path}.hold-initialized-v1")


def hold_history_anchor_path(control_db_path: str | Path) -> Path:
    """Return the external receipt-history anchor for one control database."""

    path = absolute_without_following_leaf(Path(control_db_path))
    return Path(f"{path}.hold-history-v1")


def _terminal_authority_ids(
    authorities: Mapping[str, HoldReceipt],
    consumers: Mapping[str, HoldReceipt],
) -> set[str]:
    """Return open Hold authorities with one linear visit per authority.

    Applied receipts form a functional graph: an authority has at most one
    successor, and a Hold successor is itself an authority.  Remembering every
    completed suffix avoids re-walking the full history from each ancestor.
    Any path that revisits its own unfinished suffix is a closed cycle.
    """

    terminal_authorities: set[str] = set()
    completed: set[str] = set()
    for authority_id in authorities:
        if authority_id in completed:
            continue
        cursor = authority_id
        path: list[str] = []
        path_positions: dict[str, int] = {}
        while cursor not in completed:
            if cursor in path_positions:
                raise HoldCorruptStateError("Hold receipt graph contains a cycle")
            path_positions[cursor] = len(path)
            path.append(cursor)
            successor = consumers.get(cursor)
            if successor is None:
                terminal_authorities.add(cursor)
                break
            if successor.action is HoldAction.RELEASE:
                break
            cursor = successor.receipt_id
        completed.update(path)
    return terminal_authorities


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _coerce_scope(value: HoldScope | str) -> HoldScope:
    if isinstance(value, HoldScope):
        return value
    try:
        return HoldScope(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("scope must be 'host' or 'agent'") from exc


def _target(scope: HoldScope, target_id: Optional[str]) -> str:
    if scope is HoldScope.HOST:
        if target_id not in (None, "", HOST_HOLD_TARGET):
            raise ValueError("host Hold target is fixed by the host control store")
        return HOST_HOLD_TARGET
    return _required_text(target_id, "target_id")


def _exact_nonnegative_revision(value: object) -> int:
    """Accept only the exact integer representation the latch schema promises."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HoldCorruptStateError("hold latch revision is invalid")
    return value


def _exact_active_flag(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in (0, 1)
    ):
        raise HoldCorruptStateError("hold latch active flag is invalid")
    return value


def _aware_timestamp(value: object, field: str) -> str:
    """Validate persisted Hold time evidence without rewriting its digest."""

    if not isinstance(value, str) or not value:
        raise HoldCorruptStateError(f"hold {field} timestamp is invalid")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise HoldCorruptStateError(
            f"hold {field} timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HoldCorruptStateError(
            f"hold {field} timestamp must be timezone-aware"
        )
    return value


def _latch_from_row(row: Any) -> Optional[HoldState]:
    if row is None:
        return None
    if len(row) != 8:
        raise HoldCorruptStateError("hold latch row has an unexpected shape")
    try:
        scope = HoldScope(str(row[0]))
        target_id = str(row[1])
        active = _exact_active_flag(row[2])
        revision = _exact_nonnegative_revision(row[7])
    except (TypeError, ValueError) as exc:
        raise HoldCorruptStateError("hold latch row has invalid typed fields") from exc
    hold_receipt_id = str(row[3] or "")
    reason = str(row[4] or "")
    actor_id = str(row[5] or "")
    set_at = str(row[6] or "")
    if not target_id.strip():
        raise HoldCorruptStateError("hold latch is missing its target identity")
    if scope is HoldScope.HOST and target_id != HOST_HOLD_TARGET:
        raise HoldCorruptStateError("host hold latch has a foreign target")
    evidence = (hold_receipt_id, reason, actor_id, set_at)
    if active == 0:
        if any(evidence):
            raise HoldCorruptStateError(
                "inactive latch retains active hold evidence"
            )
        return None
    if any(not value.strip() for value in evidence):
        raise HoldCorruptStateError("active hold latch is missing required evidence")
    _aware_timestamp(set_at, "latch")
    return HoldState(
        scope=scope,
        target_id=target_id,
        reason=reason,
        actor_id=actor_id,
        set_at=set_at,
        hold_receipt_id=hold_receipt_id,
        revision=revision,
    )


def _receipt_from_row(row: Any) -> HoldReceipt:
    if row is None or len(row) != 12:
        raise HoldCorruptStateError("hold receipt row has an unexpected shape")
    if any(value is None for value in row):
        # SQLite does not implicitly make a non-INTEGER PRIMARY KEY non-null,
        # and an imported/older schema may lack any of the v1 constraints.
        # Never manufacture the string ``"None"`` as durable audit evidence.
        raise HoldCorruptStateError("hold receipt has missing required evidence")
    if any(not isinstance(value, str) for value in row):
        raise HoldCorruptStateError("hold receipt has invalid evidence types")
    try:
        receipt = HoldReceipt(
            receipt_id=str(row[0]),
            operation_id=str(row[1]),
            action=HoldAction(str(row[2])),
            disposition=HoldDisposition(str(row[3])),
            scope=HoldScope(str(row[4])),
            target_id=str(row[5]),
            reason=str(row[6] or ""),
            actor_id=str(row[7]),
            occurred_at=str(row[8]),
            expected_hold_receipt_id=str(row[9] or ""),
            prior_hold_receipt_id=str(row[10] or ""),
            resulting_hold_receipt_id=str(row[11] or ""),
        )
    except (TypeError, ValueError) as exc:
        raise HoldCorruptStateError("hold receipt has invalid typed fields") from exc
    common_evidence = (
        receipt.receipt_id,
        receipt.operation_id,
        receipt.target_id,
        receipt.reason,
        receipt.actor_id,
        receipt.occurred_at,
    )
    if any(not value.strip() for value in common_evidence):
        raise HoldCorruptStateError("hold receipt invariant is invalid")
    _aware_timestamp(receipt.occurred_at, "receipt")
    if receipt.scope is HoldScope.HOST and receipt.target_id != HOST_HOLD_TARGET:
        raise HoldCorruptStateError("hold receipt invariant is invalid")

    prior = receipt.prior_hold_receipt_id
    resulting = receipt.resulting_hold_receipt_id
    expected = receipt.expected_hold_receipt_id
    if any(value and not value.strip() for value in (prior, resulting, expected)):
        raise HoldCorruptStateError("hold receipt invariant is invalid")
    if receipt.action is HoldAction.HOLD:
        valid = expected == "" and (
            (
                receipt.disposition is HoldDisposition.APPLIED
                and resulting == receipt.receipt_id
            )
            or (
                receipt.disposition is HoldDisposition.ALREADY_IN_STATE
                and bool(prior)
                and resulting == prior
            )
        )
    else:
        valid = bool(expected) and (
            (
                receipt.disposition is HoldDisposition.APPLIED
                and prior == expected
                and resulting == ""
            )
            or (
                receipt.disposition is HoldDisposition.ALREADY_IN_STATE
                and prior == ""
                and resulting == ""
            )
            or (
                receipt.disposition is HoldDisposition.REFUSED_STALE
                and bool(prior)
                and prior != expected
                and resulting == prior
            )
        )
    if not valid:
        raise HoldCorruptStateError("hold receipt invariant is invalid")
    return receipt


def _receipt_content_digest(row: Any) -> str:
    """Hash every typed receipt field with unambiguous length framing."""

    _receipt_from_row(row)
    digest = hashlib.sha256()
    for value in row:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class HoldStore:
    """Durable latch + append-only receipt store on an ``AsyncDatabase``."""

    def __init__(
        self,
        db: Any,
        *,
        initialization_witness_path: str | Path | None = None,
        history_anchor_path: str | Path | None = None,
        evidence_db: Any = None,
    ):
        self._db = db
        self._evidence_db = evidence_db
        is_postgres = getattr(db, "backend_type", "") == "postgres"
        explicit_file_evidence = (
            initialization_witness_path is not None
            or history_anchor_path is not None
        )
        if is_postgres and evidence_db is None and not explicit_file_evidence:
            raise HoldStateError(
                "PostgreSQL Hold requires an independent evidence database"
            )
        if evidence_db is not None:
            if not is_postgres:
                raise HoldStateError(
                    "an evidence database is only valid for PostgreSQL Hold"
                )
            if evidence_db is db:
                raise HoldStateError(
                    "PostgreSQL Hold evidence must use an independent rollback domain"
                )
            if getattr(evidence_db, "backend_type", "") != "postgres":
                raise HoldStateError(
                    "PostgreSQL Hold evidence requires a PostgreSQL database"
                )
        if initialization_witness_path is not None:
            self._initialization_witness_path = absolute_without_following_leaf(
                Path(initialization_witness_path)
            )
        elif getattr(db, "backend_type", "") == "sqlite":
            backend = getattr(db, "backend", None)
            db_path = getattr(backend, "db_path", None)
            self._initialization_witness_path = (
                None
                if not db_path or db_path == ":memory:"
                else hold_initialization_witness_path(db_path)
            )
        else:
            self._initialization_witness_path = None

        if history_anchor_path is not None:
            self._history_anchor_path = absolute_without_following_leaf(
                Path(history_anchor_path)
            )
        elif initialization_witness_path is not None:
            self._history_anchor_path = absolute_without_following_leaf(
                Path(f"{initialization_witness_path}.history-v1")
            )
        elif getattr(db, "backend_type", "") == "sqlite":
            backend = getattr(db, "backend", None)
            db_path = getattr(backend, "db_path", None)
            self._history_anchor_path = (
                None
                if not db_path or db_path == ":memory:"
                else hold_history_anchor_path(db_path)
            )
        else:
            self._history_anchor_path = None

        if self._history_anchor_path is None:
            self._history_candidate_path = None
            self._bootstrap_intent_path = None
            self._evidence_lock_path = None
        else:
            anchor = self._history_anchor_path
            self._history_candidate_path = Path(f"{anchor}.pending")
            self._bootstrap_intent_path = Path(f"{anchor}.bootstrap")
            self._evidence_lock_path = Path(f"{anchor}.lock")

    @asynccontextmanager
    async def _sqlite_evidence_lock(self):
        """Serialize SQLite DB snapshots with external evidence publication.

        SQLite cannot atomically commit a database transaction and replace a
        sidecar. Every Hold reader, writer, and initializer therefore takes the
        same cross-process lock while it observes or advances that pair. The
        nonblocking retry keeps a contended host from blocking its event loop.
        """

        path = self._evidence_lock_path
        if path is None:
            yield
            return
        if fcntl is None and msvcrt is None:
            raise HoldStateError(
                "durable SQLite Hold requires advisory file locks"
            )
        try:
            descriptor = open_private_file(
                path,
                os.O_RDWR | os.O_CREAT,
                label="Hold evidence protocol lock",
            )
        except PrivateStorageError as exc:
            raise HoldStateError(
                f"could not acquire Hold evidence protocol lock: {exc}"
            ) from exc
        acquired = False
        try:
            if fcntl is None:
                # ``msvcrt.locking`` locks bytes from the current file offset;
                # a zero-length file cannot provide a lock range. The byte is
                # protocol structure only, never evidence, so two creators
                # writing the same value before either locks remain benign.
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                    os.fsync(descriptor)
            while not acquired:
                try:
                    if fcntl is not None:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        assert msvcrt is not None
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError as exc:
                    windows_contention = (
                        fcntl is None
                        and (
                            exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
                            or getattr(exc, "winerror", None) in {33, 36}
                        )
                    )
                    if isinstance(exc, BlockingIOError) or windows_contention:
                        await asyncio.sleep(_EVIDENCE_LOCK_POLL_SECONDS)
                        continue
                    raise HoldStateError(
                        f"could not acquire Hold evidence protocol lock: {exc}"
                    ) from exc
            yield
        finally:
            if acquired:
                try:
                    if fcntl is not None:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    else:
                        assert msvcrt is not None
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            os.close(descriptor)

    async def _postgres_domain_identity(
        self,
        db: Any,
        *,
        label: str,
    ) -> str:
        """Read or found one durable identity for a PostgreSQL rollback domain.

        Network coordinates are not identities: two Cloud SQL Unix-socket
        connections both report a null address, while failover can change an
        address without changing the database. A unique marker stored in each
        domain rejects two pools aimed at the same database and also rejects a
        cloned evidence database until an operator deliberately separates it.
        """

        rows = await db.fetchall(
            "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
            (_POSTGRES_WITNESS_AGENT_ID, _POSTGRES_ROLLBACK_DOMAIN_KEY),
        )
        if not rows:
            candidate = _POSTGRES_ROLLBACK_DOMAIN_PREFIX + str(uuid4())
            await db.execute(
                "INSERT INTO agent_metadata (agent_id, key, value) "
                "VALUES (?, ?, ?) ON CONFLICT (agent_id, key) DO NOTHING",
                (
                    _POSTGRES_WITNESS_AGENT_ID,
                    _POSTGRES_ROLLBACK_DOMAIN_KEY,
                    candidate,
                ),
            )
            rows = await db.fetchall(
                "SELECT value FROM agent_metadata "
                "WHERE agent_id = ? AND key = ?",
                (_POSTGRES_WITNESS_AGENT_ID, _POSTGRES_ROLLBACK_DOMAIN_KEY),
            )
        if (
            len(rows) != 1
            or len(rows[0]) != 1
            or not isinstance(rows[0][0], str)
            or not rows[0][0].startswith(_POSTGRES_ROLLBACK_DOMAIN_PREFIX)
        ):
            raise HoldStateError(
                f"could not verify PostgreSQL Hold {label} rollback domain"
            )
        identity = rows[0][0]
        _validate_postgres_domain_identity(identity, label=label)
        return identity

    async def _postgres_cluster_identity(self, db: Any, *, label: str) -> str:
        """Return PostgreSQL's cluster-wide initdb identity."""

        try:
            rows = await db.fetchall(
                "SELECT system_identifier::text "
                "FROM pg_catalog.pg_control_system()"
            )
        except Exception as exc:
            raise HoldStateError(
                f"could not verify PostgreSQL Hold {label} cluster identity; "
                "the runtime role requires EXECUTE on "
                "pg_catalog.pg_control_system()"
            ) from exc
        if (
            len(rows) != 1
            or len(rows[0]) != 1
            or not isinstance(rows[0][0], str)
            or not rows[0][0].strip()
        ):
            raise HoldStateError(
                f"could not verify PostgreSQL Hold {label} cluster identity"
            )
        return rows[0][0]

    async def _assert_postgres_clusters_independent(self) -> tuple[str, str]:
        """Reject one backup unit and return both immutable cluster identities."""

        evidence_db = self._evidence_db
        if evidence_db is None:
            raise HoldStateError(
                "PostgreSQL Hold requires an independent evidence database"
            )
        primary, evidence = await _gather_database_probes(
            self._postgres_cluster_identity(self._db, label="primary"),
            self._postgres_cluster_identity(evidence_db, label="evidence"),
        )
        if primary == evidence:
            raise HoldStateError(
                "PostgreSQL Hold evidence requires an independent PostgreSQL cluster"
            )
        return primary, evidence

    async def _read_postgres_binding(
        self,
        db: Any,
        key: str,
        *,
        label: str,
    ) -> str | None:
        rows = await db.fetchall(
            "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
            (_POSTGRES_WITNESS_AGENT_ID, key),
        )
        if not rows:
            return None
        if len(rows) != 1 or len(rows[0]) != 1 or not isinstance(rows[0][0], str):
            raise HoldStateError(
                f"PostgreSQL Hold {label} custody binding is invalid"
            )
        return rows[0][0]

    async def _write_postgres_binding(self, db: Any, key: str, payload: str) -> None:
        await db.execute(
            "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT (agent_id, key) DO NOTHING",
            (_POSTGRES_WITNESS_AGENT_ID, key, payload),
        )

    async def _read_postgres_custody_roles(
        self,
        evidence_db: Any,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """Read the expected and forbidden role records from both databases."""

        return tuple(
            await _gather_database_probes(
                self._read_postgres_binding(
                    self._db,
                    _POSTGRES_PRIMARY_BINDING_KEY,
                    label="primary",
                ),
                self._read_postgres_binding(
                    evidence_db,
                    _POSTGRES_EVIDENCE_BINDING_KEY,
                    label="evidence",
                ),
                self._read_postgres_binding(
                    self._db,
                    _POSTGRES_EVIDENCE_BINDING_KEY,
                    label="primary",
                ),
                self._read_postgres_binding(
                    evidence_db,
                    _POSTGRES_PRIMARY_BINDING_KEY,
                    label="evidence",
                ),
            )
        )

    @staticmethod
    def _custody_binding_payload(
        pair_id: UUID,
        primary_identity: str,
        evidence_identity: str,
    ) -> str:
        return postgres_hold_custody_binding_payload(
            pair_id,
            primary_identity,
            evidence_identity,
        )

    @staticmethod
    def _validate_custody_binding(
        payload: str,
        *,
        primary_identity: str,
        evidence_identity: str,
    ) -> UUID:
        return _validate_postgres_custody_binding(
            payload,
            primary_identity=primary_identity,
            evidence_identity=evidence_identity,
        )

    async def _assert_postgres_evidence_domain_independent(self) -> None:
        """Bind each database permanently to one side of this custody pair."""

        evidence_db = self._evidence_db
        if evidence_db is None:
            return
        primary, evidence = await _gather_database_probes(
            self._postgres_domain_identity(self._db, label="primary"),
            self._postgres_domain_identity(evidence_db, label="evidence"),
        )
        if primary == evidence:
            raise HoldStateError(
                "PostgreSQL Hold evidence must use an independent rollback domain"
            )

        primary_binding, evidence_binding, primary_foreign, evidence_foreign = (
            await self._read_postgres_custody_roles(evidence_db)
        )
        if primary_foreign is not None or evidence_foreign is not None:
            raise HoldStateError(
                "PostgreSQL Hold database has the wrong durable custody role"
            )

        if primary_binding is None and evidence_binding is None:
            binding = self._custody_binding_payload(uuid4(), primary, evidence)
        else:
            binding = primary_binding or evidence_binding
            assert binding is not None
            self._validate_custody_binding(
                binding,
                primary_identity=primary,
                evidence_identity=evidence,
            )
            if (
                primary_binding is not None
                and evidence_binding is not None
                and primary_binding != evidence_binding
            ):
                raise HoldStateError(
                    "PostgreSQL Hold custody binding disagrees between databases"
                )

        if primary_binding is None:
            await self._write_postgres_binding(
                self._db,
                _POSTGRES_PRIMARY_BINDING_KEY,
                binding,
            )
            # INSERT .. DO NOTHING is a compare-and-declare boundary. The pair
            # locks serialize cooperating initializers, while the re-read also
            # covers a role left by an interrupted older boot or an out-of-band
            # writer. Adopt a compatible winner (same pair, different UUID),
            # but reject an incompatible winner before writing into evidence.
            primary_binding = await self._read_postgres_binding(
                self._db,
                _POSTGRES_PRIMARY_BINDING_KEY,
                label="primary",
            )
            primary_foreign = await self._read_postgres_binding(
                self._db,
                _POSTGRES_EVIDENCE_BINDING_KEY,
                label="primary",
            )
            if primary_foreign is not None:
                raise HoldStateError(
                    "PostgreSQL Hold database has the wrong durable custody role"
                )
            if primary_binding is None:
                raise HoldStateError(
                    "PostgreSQL Hold primary custody binding was not durably published"
                )
            self._validate_custody_binding(
                primary_binding,
                primary_identity=primary,
                evidence_identity=evidence,
            )
            binding = primary_binding
        if evidence_binding is None:
            await self._write_postgres_binding(
                evidence_db,
                _POSTGRES_EVIDENCE_BINDING_KEY,
                binding,
            )

        primary_binding, evidence_binding, primary_foreign, evidence_foreign = (
            await self._read_postgres_custody_roles(evidence_db)
        )
        if primary_foreign is not None or evidence_foreign is not None:
            raise HoldStateError(
                "PostgreSQL Hold database has the wrong durable custody role"
            )
        if (
            primary_binding is None
            or evidence_binding is None
            or primary_binding != evidence_binding
        ):
            raise HoldStateError(
                "PostgreSQL Hold custody binding was not durably published"
            )
        self._validate_custody_binding(
            primary_binding,
            primary_identity=primary,
            evidence_identity=evidence,
        )

    @asynccontextmanager
    async def _postgres_evidence_lock(self):
        """Serialize a primary snapshot across both PostgreSQL custody clusters."""

        evidence_db = self._evidence_db
        if evidence_db is None:
            yield
            return
        primary_cluster, evidence_cluster = (
            await self._assert_postgres_clusters_independent()
        )
        async with _postgres_custody_locks(
            self._db,
            evidence_db,
            primary_cluster=primary_cluster,
            evidence_cluster=evidence_cluster,
        ):
            locked_clusters = await self._assert_postgres_clusters_independent()
            if locked_clusters != (primary_cluster, evidence_cluster):
                raise HoldStateError(
                    "PostgreSQL Hold cluster identity changed while acquiring "
                    "custody locks"
                )
            await self._assert_postgres_evidence_domain_independent()
            yield

    async def _read_postgres_evidence(self, key: str, *, label: str) -> bytes | None:
        """Read one unique evidence value from the independent database."""

        evidence_db = self._evidence_db
        if evidence_db is None:
            raise HoldStateError(f"PostgreSQL Hold {label} has no evidence database")
        rows = await evidence_db.fetchall(
            "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
            (_POSTGRES_WITNESS_AGENT_ID, key),
        )
        if not rows:
            return None
        if len(rows) != 1 or len(rows[0]) != 1 or not isinstance(rows[0][0], str):
            raise HoldCorruptStateError(
                f"PostgreSQL Hold {label} has invalid durable evidence"
            )
        try:
            return rows[0][0].encode("ascii")
        except UnicodeEncodeError as exc:
            raise HoldCorruptStateError(
                f"PostgreSQL Hold {label} has invalid durable evidence"
            ) from exc

    async def _write_postgres_evidence(self, key: str, payload: bytes) -> None:
        """Commit one external protocol value before returning."""

        evidence_db = self._evidence_db
        if evidence_db is None:
            raise HoldStateError("PostgreSQL Hold has no evidence database")
        try:
            value = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise HoldStateError("PostgreSQL Hold evidence is not ASCII") from exc
        await evidence_db.execute(
            "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT (agent_id, key) DO UPDATE SET value = excluded.value",
            (_POSTGRES_WITNESS_AGENT_ID, key, value),
        )

    async def _remove_postgres_evidence(self, key: str) -> None:
        evidence_db = self._evidence_db
        if evidence_db is None:
            raise HoldStateError("PostgreSQL Hold has no evidence database")
        await evidence_db.execute(
            "DELETE FROM agent_metadata WHERE agent_id = ? AND key = ?",
            (_POSTGRES_WITNESS_AGENT_ID, key),
        )

    @staticmethod
    def _read_file_evidence(
        path: Path,
        *,
        label: str,
        max_bytes: int,
    ) -> bytes | None:
        """Read one complete private evidence file without following links."""

        if not path_exists(path):
            return None
        try:
            descriptor = open_private_file(path, os.O_RDONLY, label=label)
            try:
                return os.read(descriptor, max_bytes + 1)
            finally:
                os.close(descriptor)
        except PrivateStorageError as exc:
            raise HoldCorruptStateError(f"{label} cannot be trusted: {exc}") from exc

    def _read_file_initialization_witness(self) -> bool:
        """Return whether the external initialized marker is present and valid."""

        path = self._initialization_witness_path
        assert path is not None
        payload = self._read_file_evidence(
            path,
            label="Hold initialization witness",
            max_bytes=len(_INITIALIZATION_WITNESS_PAYLOAD),
        )
        if payload is None:
            return False
        if payload != _INITIALIZATION_WITNESS_PAYLOAD:
            raise HoldCorruptStateError(
                "Hold initialization witness has invalid durable evidence"
            )
        return True

    async def _read_initialization_witness(self) -> bool:
        """Read initialization evidence from restart-surviving custody."""

        if self._initialization_witness_path is not None:
            return self._read_file_initialization_witness()
        if getattr(self._db, "backend_type", "") != "postgres":
            raise HoldStateError(
                "durable Hold requires an external initialization witness"
            )
        payload = await self._read_postgres_evidence(
            _POSTGRES_WITNESS_KEY,
            label="initialization witness",
        )
        if payload is None:
            return False
        if payload != _INITIALIZATION_WITNESS_PAYLOAD:
            raise HoldCorruptStateError(
                "PostgreSQL Hold initialization witness has invalid durable evidence"
            )
        return True

    @staticmethod
    def _fsync_witness_directory(path: Path) -> None:
        if os.name == "nt":  # pragma: no cover - directory fsync is POSIX-only
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_file_evidence(self, path: Path, payload: bytes, *, label: str) -> None:
        """Atomically replace one private evidence file with fsynced content."""

        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = open_private_file(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                label=label,
            )
        except PrivateStorageError as exc:
            raise HoldStateError(f"could not persist {label}: {exc}") from exc
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(f"short write while persisting {label}")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1

            # Replace from the same private directory: readers can observe only
            # the previous complete payload or the new fsynced payload, never
            # the temporary inode while it is being written.
            os.replace(temporary, path)
            self._fsync_witness_directory(path)
        except OSError as exc:
            raise HoldStateError(f"could not persist {label}: {exc}") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if path_exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _remove_file_evidence(self, path: Path, *, label: str) -> None:
        """Durably remove one protocol marker without following its leaf."""

        try:
            os.unlink(path)
            self._fsync_witness_directory(path)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise HoldStateError(f"could not remove {label}: {exc}") from exc

    def _read_bootstrap_intent(self) -> bytes | None:
        path = self._bootstrap_intent_path
        if path is None:
            return None
        payload = self._read_file_evidence(
            path,
            label="Hold bootstrap intent",
            max_bytes=_BOOTSTRAP_INTENT_MAX_BYTES,
        )
        if payload is None:
            return None
        if not payload.startswith(_BOOTSTRAP_INTENT_PAYLOAD):
            raise HoldCorruptStateError(
                "Hold bootstrap intent has invalid durable evidence"
            )
        try:
            return self._validate_history_anchor_payload(
                payload.removeprefix(_BOOTSTRAP_INTENT_PAYLOAD)
            )
        except HoldCorruptStateError as exc:
            raise HoldCorruptStateError(
                "Hold bootstrap intent has invalid durable evidence"
            ) from exc

    def _write_bootstrap_intent(self, history_anchor: bytes) -> None:
        path = self._bootstrap_intent_path
        assert path is not None
        self._write_file_evidence(
            path,
            _BOOTSTRAP_INTENT_PAYLOAD + history_anchor,
            label="Hold bootstrap intent",
        )

    def _remove_bootstrap_intent(self) -> None:
        path = self._bootstrap_intent_path
        if path is not None:
            self._remove_file_evidence(path, label="Hold bootstrap intent")

    async def _read_external_bootstrap_intent(self) -> bytes | None:
        if self._bootstrap_intent_path is not None:
            return self._read_bootstrap_intent()
        payload = await self._read_postgres_evidence(
            _POSTGRES_BOOTSTRAP_INTENT_KEY,
            label="bootstrap intent",
        )
        if payload is None:
            return None
        if not payload.startswith(_BOOTSTRAP_INTENT_PAYLOAD):
            raise HoldCorruptStateError(
                "PostgreSQL Hold bootstrap intent has invalid durable evidence"
            )
        try:
            return self._validate_history_anchor_payload(
                payload.removeprefix(_BOOTSTRAP_INTENT_PAYLOAD)
            )
        except HoldCorruptStateError as exc:
            raise HoldCorruptStateError(
                "PostgreSQL Hold bootstrap intent has invalid durable evidence"
            ) from exc

    async def _write_external_bootstrap_intent(
        self,
        history_anchor: bytes,
    ) -> None:
        if self._bootstrap_intent_path is not None:
            self._write_bootstrap_intent(history_anchor)
            return
        await self._write_postgres_evidence(
            _POSTGRES_BOOTSTRAP_INTENT_KEY,
            _BOOTSTRAP_INTENT_PAYLOAD + history_anchor,
        )

    async def _remove_external_bootstrap_intent(self) -> None:
        if self._bootstrap_intent_path is not None:
            self._remove_bootstrap_intent()
            return
        await self._remove_postgres_evidence(_POSTGRES_BOOTSTRAP_INTENT_KEY)

    def _write_file_initialization_witness(self) -> None:
        """Atomically publish a complete local initialized marker."""

        path = self._initialization_witness_path
        assert path is not None
        if path_exists(path) and self._read_file_initialization_witness():
            return
        self._write_file_evidence(
            path,
            _INITIALIZATION_WITNESS_PAYLOAD,
            label="Hold initialization witness",
        )

    async def _write_initialization_witness(self) -> None:
        """Publish initialized evidence after the Hold schema commits."""

        if self._initialization_witness_path is not None:
            self._write_file_initialization_witness()
            return
        if getattr(self._db, "backend_type", "") != "postgres":
            raise HoldStateError(
                "durable Hold requires an external initialization witness"
            )
        await self._write_postgres_evidence(
            _POSTGRES_WITNESS_KEY,
            _INITIALIZATION_WITNESS_PAYLOAD,
        )
        if not await self._read_initialization_witness():
            raise HoldStateError(
                "could not persist PostgreSQL Hold initialization witness"
            )

    async def _current_history_anchor_payload(self) -> bytes:
        """Hash the complete immutable receipt set in a stable global order."""

        rows = await self._db.fetchall(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts ORDER BY receipt_id"
        )
        return self._history_anchor_payload_from_rows(rows)

    @classmethod
    def _is_immediate_history_predecessor(
        cls,
        predecessor: bytes,
        current_rows: list[Any] | tuple[Any, ...],
    ) -> bool:
        """Whether ``predecessor`` is exactly ``current_rows`` minus one receipt.

        A Hold mutation appends exactly one immutable receipt before staging its
        candidate head. Recovery is rare, so exhaustively proving which single
        receipt was appended is preferable to trusting only the monotonically
        increasing receipt count: two divergent histories can have the same
        count. This proof prevents a restored primary/candidate pair from
        replacing a newer stable external head.
        """

        predecessor = cls._validate_history_anchor_payload(predecessor)
        parts = predecessor.splitlines()
        if int(parts[1]) != len(current_rows) - 1:
            return False
        return any(
            cls._history_anchor_payload_from_rows(
                tuple(
                    row
                    for position, row in enumerate(current_rows)
                    if position != omitted
                )
            )
            == predecessor
            for omitted in range(len(current_rows))
        )

    @staticmethod
    def _history_anchor_payload_from_rows(rows: list[Any] | tuple[Any, ...]) -> bytes:
        """Build the canonical receipt head, including the empty history."""

        digest = hashlib.sha256()
        digest.update(_HISTORY_ANCHOR_HEADER)
        for row in rows:
            receipt = _receipt_from_row(row)
            for value in (receipt.receipt_id, _receipt_content_digest(row)):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        return (
            _HISTORY_ANCHOR_HEADER
            + str(len(rows)).encode("ascii")
            + b"\n"
            + digest.hexdigest().encode("ascii")
            + b"\n"
        )

    async def _bootstrap_history_anchor(self, existing: set[str]) -> bytes:
        """Read the receipt head a pending bootstrap is authorized to migrate."""

        if "hold_receipts" not in existing:
            return self._history_anchor_payload_from_rows([])
        return await self._current_history_anchor_payload()

    async def _read_history_anchor(self) -> bytes | None:
        """Read the receipt-history head from custody outside Hold tables."""

        if self._history_anchor_path is not None:
            payload = self._read_file_evidence(
                self._history_anchor_path,
                label="Hold history anchor",
                max_bytes=_HISTORY_ANCHOR_MAX_BYTES,
            )
        elif getattr(self._db, "backend_type", "") == "postgres":
            payload = await self._read_postgres_evidence(
                _POSTGRES_HISTORY_ANCHOR_KEY,
                label="history anchor",
            )
        else:
            raise HoldStateError("durable Hold requires an external history anchor")

        if payload is None:
            return None
        return self._validate_history_anchor_payload(payload)

    @staticmethod
    def _validate_history_anchor_payload(payload: bytes) -> bytes:
        """Validate one stable or staged history payload before trusting it."""

        parts = payload.splitlines()
        if (
            len(parts) != 3
            or parts[0] != _HISTORY_ANCHOR_HEADER.rstrip(b"\n")
            or not parts[1].isdigit()
            or str(int(parts[1])).encode("ascii") != parts[1]
            or len(parts[2]) != 64
            or any(byte not in b"0123456789abcdef" for byte in parts[2])
            or not payload.endswith(b"\n")
        ):
            raise HoldCorruptStateError(
                "Hold history anchor has invalid durable evidence"
            )
        return payload

    def _read_history_candidate(self) -> bytes | None:
        path = self._history_candidate_path
        if path is None:
            return None
        payload = self._read_file_evidence(
            path,
            label="Hold staged history anchor",
            max_bytes=_HISTORY_ANCHOR_MAX_BYTES,
        )
        if payload is None:
            return None
        return self._validate_history_anchor_payload(payload)

    def _stage_history_candidate(self, payload: bytes) -> None:
        path = self._history_candidate_path
        assert path is not None
        self._validate_history_anchor_payload(payload)
        self._write_file_evidence(
            path,
            payload,
            label="Hold staged history anchor",
        )

    def _remove_history_candidate(self) -> None:
        path = self._history_candidate_path
        if path is not None:
            self._remove_file_evidence(path, label="Hold staged history anchor")

    async def _read_external_history_candidate(self) -> bytes | None:
        if self._history_candidate_path is not None:
            return self._read_history_candidate()
        payload = await self._read_postgres_evidence(
            _POSTGRES_HISTORY_CANDIDATE_KEY,
            label="staged history anchor",
        )
        if payload is None:
            return None
        return self._validate_history_anchor_payload(payload)

    async def _stage_external_history_candidate(self, payload: bytes) -> None:
        self._validate_history_anchor_payload(payload)
        if self._history_candidate_path is not None:
            self._stage_history_candidate(payload)
            return
        await self._write_postgres_evidence(
            _POSTGRES_HISTORY_CANDIDATE_KEY,
            payload,
        )

    async def _remove_external_history_candidate(self) -> None:
        if self._history_candidate_path is not None:
            self._remove_history_candidate()
            return
        await self._remove_postgres_evidence(_POSTGRES_HISTORY_CANDIDATE_KEY)

    async def _recover_history_publication(self) -> None:
        """Resolve an interrupted primary commit from old/new durable evidence.

        The stable anchor is never changed before the database commit. A
        candidate written inside the transaction is the durable evidence of an
        intended new head. If the committed database matches it, publication
        can finish. If the database still matches the stable anchor, recovery
        cannot distinguish an interrupted rollback from a committed mutation
        followed by a primary restore, so it must fail closed. An ordinary
        in-process transaction failure removes its own candidate before the
        primary rollback through ``_primary_mutation_transaction``.
        """

        candidate = await self._read_external_history_candidate()
        if candidate is None:
            return
        current_rows = await self._db.fetchall(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts ORDER BY receipt_id"
        )
        current = self._history_anchor_payload_from_rows(current_rows)
        stable = await self._read_history_anchor()
        if current == candidate:
            if stable != candidate and (
                stable is None
                or not self._is_immediate_history_predecessor(stable, current_rows)
            ):
                raise HoldCorruptStateError(
                    "staged Hold history publication conflicts with the stable "
                    "history anchor"
                )
            if self._history_anchor_path is not None:
                self._write_file_evidence(
                    self._history_anchor_path,
                    candidate,
                    label="Hold history anchor",
                )
            else:
                await self._write_postgres_evidence(
                    _POSTGRES_HISTORY_ANCHOR_KEY,
                    candidate,
                )
            await self._remove_external_history_candidate()
            return
        if stable is not None and current == stable:
            raise HoldCorruptStateError(
                "ambiguous staged Hold history publication matches the stable "
                "anchor; refusing to discard possible committed evidence"
            )
        raise HoldCorruptStateError(
            "interrupted Hold history publication matches neither durable state"
        )

    @asynccontextmanager
    async def _sqlite_evidence_protocol(self):
        """Enter the one SQLite evidence boundary used by every live path."""

        async with self._sqlite_evidence_lock():
            if self._history_anchor_path is not None:
                await self._recover_history_publication()
            yield

    @asynccontextmanager
    async def _evidence_protocol(self):
        """Serialize every primary snapshot with its independent evidence."""

        if self._history_anchor_path is not None:
            async with self._sqlite_evidence_protocol():
                yield
            return
        async with self._postgres_evidence_lock():
            await self._recover_history_publication()
            yield

    async def _prepare_history_publication(self) -> bytes | None:
        """Stage the next external anchor before the primary commit."""

        payload = await self._current_history_anchor_payload()
        await self._stage_external_history_candidate(payload)
        return payload

    @asynccontextmanager
    async def _primary_mutation_transaction(self):
        """Rollback a known-failed mutation without leaving ambiguous evidence.

        The exception handler deliberately lives *inside* the database context:
        transaction-body failures are known not to have committed and may erase
        their candidate, while a commit/exit failure remains ambiguous and must
        leave the candidate for fail-closed recovery.
        """

        async with self._db.transaction(immediate=True):
            try:
                yield
            except BaseException:
                await self._remove_external_history_candidate()
                raise

    def _finish_history_publication(self, payload: bytes | None) -> None:
        """Promote a staged SQLite candidate only after the DB commit returns."""

        if payload is None:
            return
        candidate = self._read_history_candidate()
        if candidate != payload:
            raise HoldCorruptStateError(
                "Hold staged history anchor changed before publication"
            )
        path = self._history_anchor_path
        assert path is not None
        self._write_file_evidence(path, payload, label="Hold history anchor")
        self._remove_history_candidate()

    async def _complete_history_publication(self, payload: bytes | None) -> None:
        """Promote committed history evidence while the protocol lock is held."""

        if payload is None:
            return
        if self._history_anchor_path is not None:
            self._finish_history_publication(payload)
            return
        candidate = await self._read_external_history_candidate()
        if candidate != payload:
            raise HoldCorruptStateError(
                "Hold staged history anchor changed before publication"
            )
        await self._write_postgres_evidence(
            _POSTGRES_HISTORY_ANCHOR_KEY,
            payload,
        )
        await self._remove_external_history_candidate()

    async def _write_history_anchor(self) -> None:
        """Publish the receipt-history head before a mutation can return."""

        payload = await self._current_history_anchor_payload()
        if self._history_anchor_path is not None:
            self._write_file_evidence(
                self._history_anchor_path,
                payload,
                label="Hold history anchor",
            )
            return
        if getattr(self._db, "backend_type", "") != "postgres":
            raise HoldStateError("durable Hold requires an external history anchor")
        await self._write_postgres_evidence(
            _POSTGRES_HISTORY_ANCHOR_KEY,
            payload,
        )

    async def _assert_history_anchor_intact(self) -> None:
        """Fail closed when the database no longer matches its durable head."""

        anchored = await self._read_history_anchor()
        if anchored is None:
            raise HoldCorruptStateError("Hold history anchor is missing")
        if anchored != await self._current_history_anchor_payload():
            raise HoldCorruptStateError(
                "Hold history anchor does not match receipt history"
            )

    async def _existing_schema_tables(self) -> set[str]:
        placeholders = ", ".join("?" for _ in _HOLD_SCHEMA_TABLES)
        names = tuple(sorted(_HOLD_SCHEMA_TABLES))
        if getattr(self._db, "backend_type", "") == "postgres":
            rows = await self._db.fetchall(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema() "
                f"AND table_name IN ({placeholders})",
                names,
            )
        else:
            rows = await self._db.fetchall(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                f"AND name IN ({placeholders})",
                names,
            )
        return {str(row[0]) for row in rows}

    async def ensure_schema(self) -> None:
        """Create the Hold schema while preserving typed integrity failures."""

        try:
            if self._history_anchor_path is not None:
                async with self._sqlite_evidence_lock():
                    await self._ensure_external_schema_protocol()
            else:
                async with self._postgres_evidence_lock():
                    await self._ensure_external_schema_protocol()
        except BaseException as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                if domain_error is exc:
                    raise
                raise domain_error from exc
            raise

    @staticmethod
    def _validate_schema_evidence(
        *,
        initialized: bool,
        anchored: bytes | None,
        existing: set[str],
        bootstrap_pending: bool = False,
    ) -> None:
        legacy_tables = {"hold_latches", "hold_receipts"}
        if initialized and anchored is None:
            raise HoldCorruptStateError("Hold history anchor is missing")
        if not initialized and not bootstrap_pending:
            if anchored is not None:
                raise HoldCorruptStateError(
                    "Hold history anchor exists without its initialization witness"
                )
            if existing - legacy_tables:
                raise HoldCorruptStateError(
                    "Hold initialization witness is missing for an initialized schema"
                )

    async def _ensure_external_schema_protocol(self) -> None:
        """Run or recover bootstrap under the external protocol lock."""

        initialized = await self._read_initialization_witness()
        anchored = await self._read_history_anchor()
        existing = await self._existing_schema_tables()
        bootstrap_history = await self._read_external_bootstrap_intent()
        self._validate_schema_evidence(
            initialized=initialized,
            anchored=anchored,
            existing=existing,
            bootstrap_pending=bootstrap_history is not None,
        )
        current_bootstrap_history = await self._bootstrap_history_anchor(existing)
        if (
            bootstrap_history is not None
            and bootstrap_history != current_bootstrap_history
        ):
            raise HoldCorruptStateError(
                "Hold bootstrap intent does not match receipt history"
            )
        if (
            bootstrap_history is not None
            and anchored is not None
            and anchored != bootstrap_history
        ):
            raise HoldCorruptStateError(
                "Hold bootstrap intent conflicts with the stable history anchor"
            )
        if initialized:
            missing = sorted(_HOLD_SCHEMA_TABLES - existing)
            if missing:
                raise HoldCorruptStateError(
                    "initialized Hold schema is missing required tables: "
                    + ", ".join(missing)
                )
            await self._recover_history_publication()
            await self._ensure_schema_transaction(initialized=True)
            await self._assert_history_anchor_intact()
            if bootstrap_history is not None:
                await self._remove_external_bootstrap_intent()
            return

        if await self._read_external_history_candidate() is not None:
            raise HoldCorruptStateError(
                "Hold history publication exists without initialized schema"
            )
        if bootstrap_history is None:
            # Durable intent precedes DDL. If the process stops anywhere after
            # this write, a later initializer may finish exactly this bootstrap
            # rather than mistaking committed v1 tables for unexplained state.
            await self._write_external_bootstrap_intent(
                current_bootstrap_history
            )
        await self._ensure_schema_transaction(initialized=False)
        # The database transaction has committed while the cross-process lock
        # still excludes readers and peer initializers. Publish both pieces of
        # evidence, then retire the recovery authority last.
        await self._write_history_anchor()
        await self._write_initialization_witness()
        await self._remove_external_bootstrap_intent()

    async def _ensure_schema_transaction(
        self,
        *,
        initialized: bool,
    ) -> None:
        """Create both Hold tables as one serialized schema unit."""

        async with self._db.migration_lock(_SCHEMA_LOCK):
            if initialized:
                existing = await self._existing_schema_tables()
                missing = sorted(_HOLD_SCHEMA_TABLES - existing)
                if missing:
                    raise HoldCorruptStateError(
                        "initialized Hold schema is missing required tables: "
                        + ", ".join(missing)
                    )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_latches ("
                "scope TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "active INTEGER NOT NULL DEFAULT 0, "
                "hold_receipt_id TEXT NOT NULL DEFAULT '', "
                "reason TEXT NOT NULL DEFAULT '', "
                "actor_id TEXT NOT NULL DEFAULT '', "
                "set_at TEXT NOT NULL DEFAULT '', "
                "revision INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY (scope, target_id), "
                "CHECK (scope IN ('host', 'agent')), "
                "CHECK (scope <> 'host' OR target_id = 'host'), "
                "CHECK (active IN (0, 1)), "
                "CHECK (revision >= 0))"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_receipts ("
                "receipt_id TEXT NOT NULL PRIMARY KEY, "
                "operation_id TEXT NOT NULL UNIQUE, "
                "action TEXT NOT NULL, "
                "disposition TEXT NOT NULL, "
                "scope TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "reason TEXT NOT NULL DEFAULT '', "
                "actor_id TEXT NOT NULL, "
                "occurred_at TEXT NOT NULL, "
                "expected_hold_receipt_id TEXT NOT NULL DEFAULT '', "
                "prior_hold_receipt_id TEXT NOT NULL DEFAULT '', "
                "resulting_hold_receipt_id TEXT NOT NULL DEFAULT '', "
                "CHECK (action IN ('hold', 'release')), "
                "CHECK (disposition IN "
                "('applied', 'already_in_state', 'refused_stale')), "
                "CHECK (scope IN ('host', 'agent')), "
                "CHECK (scope <> 'host' OR target_id = 'host'))"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_hold_receipts_target "
                "ON hold_receipts(scope, target_id, occurred_at, receipt_id)"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_receipt_witnesses ("
                "scope TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "receipt_count INTEGER NOT NULL DEFAULT 0, "
                "PRIMARY KEY (scope, target_id), "
                "CHECK (scope IN ('host', 'agent')), "
                "CHECK (scope <> 'host' OR target_id = 'host'), "
                "CHECK (receipt_count >= 0))"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_receipt_content_witnesses ("
                "receipt_id TEXT NOT NULL PRIMARY KEY, "
                "scope TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "receipt_digest TEXT NOT NULL, "
                "CHECK (scope IN ('host', 'agent')), "
                "CHECK (scope <> 'host' OR target_id = 'host'))"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_hold_receipt_content_witnesses_target "
                "ON hold_receipt_content_witnesses(scope, target_id)"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_operation_witnesses ("
                "operation_id TEXT NOT NULL PRIMARY KEY, "
                "receipt_id TEXT NOT NULL UNIQUE)"
            )
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS hold_schema_migrations ("
                "name TEXT NOT NULL PRIMARY KEY)"
            )
            migration_complete = await self._db.fetchone(
                "SELECT 1 FROM hold_schema_migrations WHERE name = ?",
                (_WITNESS_BACKFILL,),
            )
            duplicate_operation = await self._db.fetchone(
                "SELECT operation_id FROM hold_receipts "
                "GROUP BY operation_id HAVING COUNT(*) > 1 LIMIT 1"
            )
            if duplicate_operation is not None:
                raise HoldCorruptStateError(
                    "Hold receipt history contains a duplicate operation id"
                )
            if migration_complete is not None:
                await self._assert_completed_witness_migration_intact()
                return
            if initialized:
                raise HoldCorruptStateError(
                    "initialized Hold schema is missing its required witness "
                    "migration marker"
                )
            # Seed the witness exactly once for upgraded databases. Future
            # schema checks never derive missing witnesses from mutable receipt
            # rows after the durable migration marker exists. Without that
            # gate, deleting a witness on a later boot re-blessed whatever
            # receipt content happened to remain.
            await self._db.execute(
                "INSERT INTO hold_receipt_witnesses "
                "(scope, target_id, receipt_count) "
                "SELECT scope, target_id, COUNT(*) FROM hold_receipts "
                "WHERE scope <> 'host' OR target_id = 'host' "
                "GROUP BY scope, target_id "
                "ON CONFLICT (scope, target_id) DO NOTHING"
            )
            # One-time backfill for upgraded databases. A witness is never
            # overwritten from receipt rows after it exists, so later in-place
            # mutation remains detectable across schema checks and restarts.
            missing_content_witnesses = await self._db.fetchall(
                "SELECT r.receipt_id, r.operation_id, r.action, r.disposition, "
                "r.scope, r.target_id, r.reason, r.actor_id, r.occurred_at, "
                "r.expected_hold_receipt_id, r.prior_hold_receipt_id, "
                "r.resulting_hold_receipt_id FROM hold_receipts AS r "
                "LEFT JOIN hold_receipt_content_witnesses AS w "
                "ON w.receipt_id = r.receipt_id WHERE w.receipt_id IS NULL"
            )
            for row in missing_content_witnesses:
                receipt = _receipt_from_row(row)
                await self._db.execute(
                    "INSERT INTO hold_receipt_content_witnesses "
                    "(receipt_id, scope, target_id, receipt_digest) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (receipt_id) DO NOTHING",
                    (
                        receipt.receipt_id,
                        receipt.scope.value,
                        receipt.target_id,
                        _receipt_content_digest(row),
                    ),
                )
            # Operation identities are global, not target-local. Keep a
            # separate append-only tombstone so deleting a receipt cannot make
            # its operation id available to a different target. The anti-join
            # keeps repeat startup proportional to genuinely missing legacy
            # witnesses rather than total receipt history.
            await self._db.execute(
                "INSERT INTO hold_operation_witnesses (operation_id, receipt_id) "
                "SELECT r.operation_id, r.receipt_id FROM hold_receipts AS r "
                "LEFT JOIN hold_operation_witnesses AS w "
                "ON w.operation_id = r.operation_id "
                "WHERE w.operation_id IS NULL "
                "ON CONFLICT (operation_id) DO NOTHING"
            )
            await self._db.execute(
                "INSERT INTO hold_receipt_witnesses "
                "(scope, target_id, receipt_count) "
                "SELECT scope, target_id, 0 FROM hold_latches "
                "WHERE scope <> 'host' OR target_id = 'host' "
                "ON CONFLICT (scope, target_id) DO NOTHING"
            )
            await self._db.execute(
                "INSERT INTO hold_schema_migrations (name) VALUES (?) "
                "ON CONFLICT (name) DO NOTHING",
                (_WITNESS_BACKFILL,),
            )

    async def _assert_completed_witness_migration_intact(self) -> None:
        """Fail closed if a completed migration later loses any witness."""

        missing_content = await self._db.fetchone(
            "SELECT r.receipt_id FROM hold_receipts AS r "
            "LEFT JOIN hold_receipt_content_witnesses AS w "
            "ON w.receipt_id = r.receipt_id "
            "WHERE w.receipt_id IS NULL LIMIT 1"
        )
        if missing_content is not None:
            raise HoldCorruptStateError(
                "completed Hold witness migration is missing a content witness"
            )

        await self._assert_no_missing_operation_witnesses(
            context="completed Hold witness migration",
        )
        await self._assert_no_duplicate_operation_witnesses()
        await self._assert_no_orphaned_operation_witnesses()

        missing_count = await self._db.fetchone(
            "SELECT source.scope, source.target_id FROM ("
            "SELECT scope, target_id FROM hold_receipts "
            "UNION SELECT scope, target_id FROM hold_latches"
            ") AS source LEFT JOIN hold_receipt_witnesses AS w "
            "ON w.scope = source.scope AND w.target_id = source.target_id "
            "WHERE w.scope IS NULL LIMIT 1"
        )
        if missing_count is not None:
            raise HoldCorruptStateError(
                "completed Hold witness migration is missing a receipt-count witness"
            )

    async def _assert_no_missing_operation_witnesses(
        self,
        *,
        context: str = "Hold receipt history",
    ) -> None:
        """Reject immutable receipts whose global operation evidence was lost."""

        missing_operation = await self._db.fetchone(
            "SELECT r.operation_id FROM hold_receipts AS r "
            "LEFT JOIN hold_operation_witnesses AS w "
            "ON w.operation_id = r.operation_id "
            "WHERE w.operation_id IS NULL LIMIT 1"
        )
        if missing_operation is not None:
            raise HoldCorruptStateError(f"{context} is missing an operation witness")

    async def _assert_no_duplicate_operation_witnesses(self) -> None:
        """Reject imported operation evidence whose nominal key is ambiguous."""

        duplicate = await self._db.fetchone(
            "SELECT operation_id FROM hold_operation_witnesses "
            "GROUP BY operation_id HAVING COUNT(*) > 1 LIMIT 1"
        )
        if duplicate is not None:
            raise HoldCorruptStateError(
                "Hold operation witness has a duplicate operation identity"
            )

    async def _assert_no_orphaned_operation_witnesses(self) -> None:
        """Reject append-only operation evidence without its immutable receipt."""

        orphaned = await self._db.fetchone(
            "SELECT w.operation_id FROM hold_operation_witnesses AS w "
            "LEFT JOIN hold_receipts AS r "
            "ON r.operation_id = w.operation_id AND r.receipt_id = w.receipt_id "
            "WHERE r.operation_id IS NULL LIMIT 1"
        )
        if orphaned is not None:
            raise HoldCorruptStateError(
                "Hold operation witness refers to a missing receipt"
            )

    async def _assert_global_history_intact(self) -> None:
        """Validate database-wide receipt evidence once per stable snapshot."""

        await self._assert_no_duplicate_operation_witnesses()
        await self._assert_no_missing_operation_witnesses()
        await self._assert_no_orphaned_operation_witnesses()
        await self._assert_history_anchor_intact()

    async def read_boot_state(self) -> tuple[HoldState, ...]:
        """Validate and return every active latch before work producers start."""

        async with self._evidence_protocol():
            rows = await self._db.fetchall(
                "SELECT scope, target_id FROM hold_latches "
                "UNION SELECT scope, target_id FROM hold_receipts "
                "UNION SELECT scope, target_id FROM hold_receipt_witnesses "
                "UNION SELECT scope, target_id FROM hold_receipt_content_witnesses"
            )
            targets: set[tuple[HoldScope, str]] = {
                (HoldScope.HOST, HOST_HOLD_TARGET)
            }
            for row in rows:
                if len(row) != 2:
                    raise HoldCorruptStateError(
                        "Hold boot-state target row has an unexpected shape"
                    )
                try:
                    scope = HoldScope(str(row[0]))
                except (TypeError, ValueError) as exc:
                    raise HoldCorruptStateError(
                        "Hold boot-state target has an invalid scope"
                    ) from exc
                target_id = row[1]
                if not isinstance(target_id, str) or not target_id.strip():
                    raise HoldCorruptStateError(
                        "Hold boot-state target is missing its identity"
                    )
                if target_id != target_id.strip():
                    raise HoldCorruptStateError(
                        "Hold boot-state target has a noncanonical identity"
                    )
                if scope is HoldScope.HOST and target_id != HOST_HOLD_TARGET:
                    raise HoldCorruptStateError(
                        "Hold boot-state target has a foreign host identity"
                    )
                targets.add((scope, target_id))

            active: list[HoldState] = []
            for scope, target_id in sorted(
                targets,
                key=lambda item: (item[0].value, item[1]),
            ):
                try:
                    state = await self._get_hold(
                        scope,
                        target_id,
                        validate_global_history=False,
                    )
                except Exception as exc:
                    domain_error = _domain_error_from_chain(exc)
                    if domain_error is not None:
                        raise domain_error from exc
                    raise
                if state is not None:
                    active.append(state)
            # The evidence protocol excludes every legitimate Hold writer for
            # this entire boot read. Validate the database-wide tombstones and
            # receipt anchor once after the target-local walks. Repeating these
            # global scans from ``_get_hold`` for every target makes boot
            # O(targets * total receipts).
            try:
                async with self._db.transaction():
                    await self._lock_read_history()
                    await self._assert_global_history_intact()
            except Exception as exc:
                domain_error = _domain_error_from_chain(exc)
                if domain_error is not None:
                    raise domain_error from exc
                raise
            return tuple(active)

    async def _lock_operation_and_target(
        self, operation_id: str, scope: HoldScope, target_id: str
    ) -> None:
        if getattr(self._db, "backend_type", "") != "postgres":
            return
        # One global acquisition order for every writer: history first,
        # operation second, target third. The global lock serializes the
        # database-wide receipt head across otherwise-independent targets; the
        # other locks close operation reuse and absent-row mutation gaps.
        for key in (
            _HISTORY_LOCK_KEY,
            f"kestrel:hold:operation:{operation_id}",
            f"kestrel:hold:target:{scope.value}:{target_id}",
        ):
            await self._db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                (key,),
            )

    async def _lock_read_history(self) -> None:
        """Stabilize PostgreSQL's global receipt set before inspecting it."""

        if getattr(self._db, "backend_type", "") != "postgres":
            return
        await self._db.execute(
            "SELECT pg_advisory_xact_lock_shared(hashtextextended(?, 0))",
            (_HISTORY_LOCK_KEY,),
        )

    async def _lock_read_targets(
        self,
        targets: tuple[tuple[HoldScope, str], ...],
        *,
        history_locked: bool = False,
    ) -> None:
        """Serialize a PostgreSQL read snapshot with legitimate target writers."""

        if getattr(self._db, "backend_type", "") != "postgres":
            return
        target_keys = sorted(
            {
                f"kestrel:hold:target:{scope.value}:{target_id}"
                for scope, target_id in targets
            }
        )
        # Every read validates the database-wide history anchor. Its shared
        # lock must therefore precede the same target locks as every writer.
        if not history_locked:
            await self._lock_read_history()
        for key in target_keys:
            await self._db.execute(
                "SELECT pg_advisory_xact_lock_shared(hashtextextended(?, 0))",
                (key,),
            )

    async def _ensure_latch_row(self, scope: HoldScope, target_id: str) -> None:
        await self._db.execute(
            "INSERT INTO hold_latches (scope, target_id) VALUES (?, ?) "
            "ON CONFLICT (scope, target_id) DO NOTHING",
            (scope.value, target_id),
        )
        await self._db.execute(
            "INSERT INTO hold_receipt_witnesses "
            "(scope, target_id, receipt_count) VALUES (?, ?, 0) "
            "ON CONFLICT (scope, target_id) DO NOTHING",
            (scope.value, target_id),
        )

    async def _read_latch_row(
        self, scope: HoldScope, target_id: str, *, for_update: bool = False
    ) -> Any:
        suffix = (
            " FOR UPDATE"
            if for_update and getattr(self._db, "backend_type", "") == "postgres"
            else ""
        )
        rows = await self._db.fetchall(
            f"SELECT {_LATCH_COLUMNS} FROM hold_latches "
            f"WHERE scope = ? AND target_id = ?{suffix}",
            (scope.value, target_id),
        )
        if len(rows) > 1:
            raise HoldCorruptStateError("duplicate hold latch key")
        return rows[0] if rows else None

    async def _assert_host_latch_shape(self, *, for_update: bool = False) -> None:
        """Fail closed if an upgraded/shared database has a foreign host row."""

        suffix = (
            " FOR UPDATE"
            if for_update and getattr(self._db, "backend_type", "") == "postgres"
            else ""
        )
        latch_rows = await self._db.fetchall(
            "SELECT target_id FROM hold_latches WHERE scope = ?" + suffix,
            (HoldScope.HOST.value,),
        )
        receipt_rows = await self._db.fetchall(
            "SELECT target_id FROM hold_receipts WHERE scope = ?" + suffix,
            (HoldScope.HOST.value,),
        )
        if any(
            str(row[0]) != HOST_HOLD_TARGET
            for row in (*latch_rows, *receipt_rows)
        ):
            raise HoldCorruptStateError(
                "host hold state has a foreign target identity"
            )

    async def _read_receipt_by_operation(self, operation_id: str) -> Any:
        rows = await self._db.fetchall(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts WHERE operation_id = ?",
            (operation_id,),
        )
        if len(rows) > 1:
            raise HoldCorruptStateError(
                "Hold receipt history has a duplicate operation identity"
            )
        return rows[0] if rows else None

    async def _validate_operation_witness(self, operation_id: str) -> Any:
        """Return the receipt row only when its global identity is intact."""

        witnesses = await self._db.fetchall(
            "SELECT receipt_id FROM hold_operation_witnesses "
            "WHERE operation_id = ?",
            (operation_id,),
        )
        if len(witnesses) > 1:
            raise HoldCorruptStateError(
                "Hold operation witness has a duplicate operation identity"
            )
        receipt_row = await self._read_receipt_by_operation(operation_id)
        if not witnesses:
            if receipt_row is not None:
                raise HoldCorruptStateError(
                    "Hold receipt is missing its global operation witness"
                )
            return None
        if receipt_row is None:
            raise HoldCorruptStateError(
                "Hold operation witness refers to a missing receipt"
            )
        if str(witnesses[0][0]) != str(receipt_row[0]):
            raise HoldCorruptStateError(
                "Hold operation witness does not match receipt identity"
            )
        return receipt_row

    async def _read_receipt_by_id(self, receipt_id: str) -> Any:
        return await self._db.fetchone(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts WHERE receipt_id = ?",
            (receipt_id,),
        )

    async def _validate_receipt_authority_graph(
        self,
        latch: Optional[HoldState],
        scope: HoldScope,
        target_id: str,
        *,
        validate_global_history: bool = True,
    ) -> None:
        """Prove the append-only authority graph agrees with its latch.

        Each applied Hold creates one authority. A later applied Hold or
        Release may consume it exactly once. Every chain must be acyclic and
        terminate either in an applied Release or in the one active authority
        named by the latch. This catches projection deletion, latch rewind,
        forked successors, and closed cycles rather than trusting a locally
        well-formed latch row.
        """

        rows = await self._db.fetchall(
            f"SELECT {_RECEIPT_COLUMNS} FROM hold_receipts "
            "WHERE scope = ? AND target_id = ?",
            (scope.value, target_id),
        )
        receipts = [_receipt_from_row(row) for row in rows]
        receipt_ids = {receipt.receipt_id for receipt in receipts}
        if len(receipt_ids) != len(receipts):
            raise HoldCorruptStateError("Hold receipt graph has duplicate identities")
        witness_rows = await self._db.fetchall(
            "SELECT receipt_id, receipt_digest "
            "FROM hold_receipt_content_witnesses "
            "WHERE scope = ? AND target_id = ?",
            (scope.value, target_id),
        )
        content_witnesses: dict[str, str] = {}
        content_witness_valid = True
        for witness in witness_rows:
            if (
                len(witness) != 2
                or not isinstance(witness[0], str)
                or not witness[0]
                or not isinstance(witness[1], str)
                or len(witness[1]) != 64
                or witness[0] in content_witnesses
            ):
                content_witness_valid = False
                continue
            content_witnesses[witness[0]] = witness[1]
        if set(content_witnesses) != receipt_ids:
            content_witness_valid = False
        for row, receipt in zip(rows, receipts):
            if content_witnesses.get(receipt.receipt_id) != _receipt_content_digest(row):
                content_witness_valid = False
        projection_row = await self._read_latch_row(scope, target_id)
        projection_revision = 0
        if projection_row is not None:
            if len(projection_row) != 8:
                raise HoldCorruptStateError(
                    "hold latch row has an unexpected shape"
                )
            try:
                projection_revision = _exact_nonnegative_revision(
                    projection_row[7]
                )
            except (TypeError, ValueError) as exc:
                raise HoldCorruptStateError(
                    "hold latch row has invalid typed fields"
                ) from exc
        witness_rows = await self._db.fetchall(
            "SELECT receipt_count FROM hold_receipt_witnesses "
            "WHERE scope = ? AND target_id = ?",
            (scope.value, target_id),
        )
        if not witness_rows and not receipts and projection_row is None:
            witnessed_receipts = 0
        elif not witness_rows:
            raise HoldCorruptStateError("Hold receipt-count witness is missing")
        elif len(witness_rows) != 1:
            raise HoldCorruptStateError("Hold receipt-count witness is duplicated")
        elif len(witness_rows[0]) != 1:
            raise HoldCorruptStateError(
                "Hold receipt-count witness has an unexpected shape"
            )
        else:
            try:
                witnessed_receipts = _exact_nonnegative_revision(witness_rows[0][0])
            except (TypeError, ValueError) as exc:
                raise HoldCorruptStateError(
                    "Hold receipt-count witness has invalid typed fields"
                ) from exc
        applied = [
            receipt
            for receipt in receipts
            if receipt.disposition is HoldDisposition.APPLIED
        ]
        authorities = {
            receipt.receipt_id: receipt
            for receipt in applied
            if receipt.action is HoldAction.HOLD
        }
        # Every prior/resulting reference names an applied Hold authority,
        # including non-applied audit outcomes.  ALREADY_IN_STATE and
        # REFUSED_STALE do not consume authority, but accepting a dangling
        # reference would let an idempotent replay return a receipt whose
        # requested latch never existed (or was deleted).
        for receipt in receipts:
            for authority_id in {
                receipt.prior_hold_receipt_id,
                receipt.resulting_hold_receipt_id,
            }:
                if authority_id and authority_id not in authorities:
                    raise HoldCorruptStateError(
                        "Hold history references missing authority receipt"
                    )
        consumers: dict[str, HoldReceipt] = {}
        for receipt in applied:
            prior = receipt.prior_hold_receipt_id
            if not prior:
                continue
            if prior not in authorities:
                raise HoldCorruptStateError(
                    "applied Hold history consumes missing authority"
                )
            if prior in consumers:
                raise HoldCorruptStateError(
                    "applied Hold authority has multiple successors"
                )
            consumers[prior] = receipt

        terminal_authorities = _terminal_authority_ids(authorities, consumers)

        if latch is None:
            if witnessed_receipts != len(receipts):
                raise HoldCorruptStateError(
                    "hold receipt-count revision does not match receipt history"
                )
            if not terminal_authorities:
                if projection_revision != len(applied):
                    raise HoldCorruptStateError(
                        "hold latch revision does not match applied receipt history"
                    )
                if not content_witness_valid:
                    raise HoldCorruptStateError(
                        "Hold receipt content witness does not match receipt history"
                    )
                # Operation witnesses are global tombstones. Without target
                # metadata an orphan cannot safely be attributed to this target
                # or ruled out, so a successful read must fail closed until the
                # database is repaired.
                if validate_global_history:
                    await self._assert_global_history_intact()
                return
            raise HoldCorruptStateError(
                "unheld projection retains active Hold authority"
            )

        receipt = authorities.get(latch.hold_receipt_id)
        if receipt is None:
            referenced_row = await self._read_receipt_by_id(latch.hold_receipt_id)
            if referenced_row is not None:
                _receipt_from_row(referenced_row)
                raise HoldCorruptStateError(
                    "active hold latch does not match its authority receipt"
                )
            raise HoldCorruptStateError(
                "active hold latch references a missing authority receipt"
            )
        if (
            receipt.scope is not latch.scope
            or receipt.target_id != latch.target_id
            or receipt.reason != latch.reason
            or receipt.actor_id != latch.actor_id
            or receipt.occurred_at != latch.set_at
        ):
            raise HoldCorruptStateError(
                "active hold latch does not match its authority receipt"
            )
        if terminal_authorities != {latch.hold_receipt_id}:
            raise HoldCorruptStateError(
                "active hold latch is not the receipt graph's terminal authority"
            )
        if witnessed_receipts != len(receipts):
            raise HoldCorruptStateError(
                "hold receipt-count revision does not match receipt history"
            )
        if projection_revision != len(applied):
            raise HoldCorruptStateError(
                "hold latch revision does not match applied receipt history"
            )
        if not content_witness_valid:
            raise HoldCorruptStateError(
                "Hold receipt content witness does not match receipt history"
            )
        if validate_global_history:
            await self._assert_global_history_intact()

    async def _validate_latch_projection(
        self,
        latch: Optional[HoldState],
        scope: HoldScope,
        target_id: str,
        *,
        validate_global_history: bool = True,
    ) -> None:
        await self._validate_receipt_authority_graph(
            latch,
            scope,
            target_id,
            validate_global_history=validate_global_history,
        )

    @staticmethod
    def _assert_replay(
        receipt: HoldReceipt,
        *,
        action: HoldAction,
        scope: HoldScope,
        target_id: str,
        reason: str,
        actor_id: str,
        expected_hold_receipt_id: str,
    ) -> None:
        supplied = (
            action,
            scope,
            target_id,
            reason,
            actor_id,
            expected_hold_receipt_id,
        )
        recorded = (
            receipt.action,
            receipt.scope,
            receipt.target_id,
            receipt.reason,
            receipt.actor_id,
            receipt.expected_hold_receipt_id,
        )
        if supplied != recorded:
            raise HoldIdempotencyConflict(
                "hold operation id was already used for a different mutation"
            )

    async def _insert_receipt(
        self,
        *,
        operation_id: str,
        action: HoldAction,
        disposition: HoldDisposition,
        scope: HoldScope,
        target_id: str,
        reason: str,
        actor_id: str,
        expected_hold_receipt_id: str,
        prior_hold_receipt_id: str,
        resulting_hold_receipt_id: str,
        receipt_id: Optional[str] = None,
    ) -> HoldReceipt:
        receipt_id = receipt_id or str(uuid4())
        now_sql = database_now_sql(self._db)
        await self._db.execute(
            "INSERT INTO hold_receipts ("
            "receipt_id, operation_id, action, disposition, scope, target_id, "
            "reason, actor_id, occurred_at, expected_hold_receipt_id, "
            "prior_hold_receipt_id, resulting_hold_receipt_id"
            f") VALUES (?, ?, ?, ?, ?, ?, ?, ?, {now_sql}, ?, ?, ?)",
            (
                receipt_id,
                operation_id,
                action.value,
                disposition.value,
                scope.value,
                target_id,
                reason,
                actor_id,
                expected_hold_receipt_id,
                prior_hold_receipt_id,
                resulting_hold_receipt_id,
            ),
        )
        receipt_row = await self._read_receipt_by_operation(operation_id)
        receipt = _receipt_from_row(receipt_row)
        await self._db.execute(
            "INSERT INTO hold_operation_witnesses (operation_id, receipt_id) "
            "VALUES (?, ?)",
            (receipt.operation_id, receipt.receipt_id),
        )
        await self._db.execute(
            "INSERT INTO hold_receipt_content_witnesses "
            "(receipt_id, scope, target_id, receipt_digest) "
            "VALUES (?, ?, ?, ?)",
            (
                receipt.receipt_id,
                receipt.scope.value,
                receipt.target_id,
                _receipt_content_digest(receipt_row),
            ),
        )
        await self._db.execute(
            "UPDATE hold_receipt_witnesses "
            "SET receipt_count = receipt_count + 1 "
            "WHERE scope = ? AND target_id = ?",
            (scope.value, target_id),
        )
        return receipt

    async def set_hold(
        self,
        *,
        scope: HoldScope | str,
        actor_id: str,
        reason: str,
        operation_id: str,
        target_id: Optional[str] = None,
    ) -> HoldMutation:
        """Set or replace one latch and append an immutable receipt."""

        try:
            async with self._evidence_protocol():
                return await self._set_hold(
                    scope=scope,
                    actor_id=actor_id,
                    reason=reason,
                    operation_id=operation_id,
                    target_id=target_id,
                )
        except Exception as exc:
            # AsyncDatabase deliberately wraps transaction-body exceptions.
            # Every Hold domain failure is part of this store's public
            # contract, so preserve its type across that backend boundary.
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _set_hold(
        self,
        *,
        scope: HoldScope | str,
        actor_id: str,
        reason: str,
        operation_id: str,
        target_id: Optional[str] = None,
    ) -> HoldMutation:
        resolved_scope = _coerce_scope(scope)
        resolved_target = _target(resolved_scope, target_id)
        actor = _required_text(actor_id, "actor_id")
        why = _required_text(reason, "reason")
        operation = _required_text(operation_id, "operation_id")

        publication: bytes | None = None
        async with self._primary_mutation_transaction():
            await self._assert_host_latch_shape(
                for_update=resolved_scope is HoldScope.HOST
            )
            await self._lock_operation_and_target(
                operation, resolved_scope, resolved_target
            )
            await self._ensure_latch_row(resolved_scope, resolved_target)
            prior_row = await self._read_latch_row(
                resolved_scope, resolved_target, for_update=True
            )
            prior = _latch_from_row(prior_row)
            await self._validate_latch_projection(
                prior, resolved_scope, resolved_target
            )
            replay_row = await self._validate_operation_witness(operation)
            if replay_row is not None:
                replay = _receipt_from_row(replay_row)
                self._assert_replay(
                    replay,
                    action=HoldAction.HOLD,
                    scope=resolved_scope,
                    target_id=resolved_target,
                    reason=why,
                    actor_id=actor,
                    expected_hold_receipt_id="",
                )
                return HoldMutation(receipt=replay, current=prior)

            if prior is not None and prior.actor_id == actor and prior.reason == why:
                receipt = await self._insert_receipt(
                    operation_id=operation,
                    action=HoldAction.HOLD,
                    disposition=HoldDisposition.ALREADY_IN_STATE,
                    scope=resolved_scope,
                    target_id=resolved_target,
                    reason=why,
                    actor_id=actor,
                    expected_hold_receipt_id="",
                    prior_hold_receipt_id=prior.hold_receipt_id,
                    resulting_hold_receipt_id=prior.hold_receipt_id,
                )
                current = prior
            else:
                receipt_id = str(uuid4())
                receipt = await self._insert_receipt(
                    operation_id=operation,
                    action=HoldAction.HOLD,
                    disposition=HoldDisposition.APPLIED,
                    scope=resolved_scope,
                    target_id=resolved_target,
                    reason=why,
                    actor_id=actor,
                    expected_hold_receipt_id="",
                    prior_hold_receipt_id=(prior.hold_receipt_id if prior else ""),
                    resulting_hold_receipt_id=receipt_id,
                    receipt_id=receipt_id,
                )
                await self._db.execute(
                    "UPDATE hold_latches SET active = 1, hold_receipt_id = ?, "
                    "reason = ?, actor_id = ?, set_at = ?, revision = revision + 1 "
                    "WHERE scope = ? AND target_id = ?",
                    (
                        receipt.receipt_id,
                        why,
                        actor,
                        receipt.occurred_at,
                        resolved_scope.value,
                        resolved_target,
                    ),
                )
                current = _latch_from_row(
                    await self._read_latch_row(resolved_scope, resolved_target)
                )
            # Every non-replay path inserts exactly one immutable receipt. The
            # external candidate is durable before primary commit, while the
            # stable anchor remains untouched until that commit returns.
            publication = await self._prepare_history_publication()
            mutation = HoldMutation(receipt=receipt, current=current)
        await self._complete_history_publication(publication)
        return mutation

    async def release_hold(
        self,
        *,
        scope: HoldScope | str,
        actor_id: str,
        reason: str,
        operation_id: str,
        expected_hold_receipt_id: str,
        target_id: Optional[str] = None,
    ) -> HoldMutation:
        """Release exactly the observed latch, refusing a stale release."""

        try:
            async with self._evidence_protocol():
                return await self._release_hold(
                    scope=scope,
                    actor_id=actor_id,
                    reason=reason,
                    operation_id=operation_id,
                    expected_hold_receipt_id=expected_hold_receipt_id,
                    target_id=target_id,
                )
        except Exception as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _release_hold(
        self,
        *,
        scope: HoldScope | str,
        actor_id: str,
        reason: str,
        operation_id: str,
        expected_hold_receipt_id: str,
        target_id: Optional[str] = None,
    ) -> HoldMutation:
        resolved_scope = _coerce_scope(scope)
        resolved_target = _target(resolved_scope, target_id)
        actor = _required_text(actor_id, "actor_id")
        why = _required_text(reason, "reason")
        operation = _required_text(operation_id, "operation_id")
        expected = _required_text(
            expected_hold_receipt_id, "expected_hold_receipt_id"
        )

        publication: bytes | None = None
        async with self._primary_mutation_transaction():
            await self._assert_host_latch_shape(
                for_update=resolved_scope is HoldScope.HOST
            )
            await self._lock_operation_and_target(
                operation, resolved_scope, resolved_target
            )
            await self._ensure_latch_row(resolved_scope, resolved_target)
            prior_row = await self._read_latch_row(
                resolved_scope, resolved_target, for_update=True
            )
            prior = _latch_from_row(prior_row)
            await self._validate_latch_projection(
                prior, resolved_scope, resolved_target
            )
            replay_row = await self._validate_operation_witness(operation)
            if replay_row is not None:
                replay = _receipt_from_row(replay_row)
                self._assert_replay(
                    replay,
                    action=HoldAction.RELEASE,
                    scope=resolved_scope,
                    target_id=resolved_target,
                    reason=why,
                    actor_id=actor,
                    expected_hold_receipt_id=expected,
                )
                return HoldMutation(receipt=replay, current=prior)

            if prior is None:
                disposition = HoldDisposition.ALREADY_IN_STATE
                prior_receipt_id = ""
                resulting_receipt_id = ""
            elif prior.hold_receipt_id != expected:
                disposition = HoldDisposition.REFUSED_STALE
                prior_receipt_id = prior.hold_receipt_id
                resulting_receipt_id = prior.hold_receipt_id
            else:
                disposition = HoldDisposition.APPLIED
                prior_receipt_id = prior.hold_receipt_id
                resulting_receipt_id = ""

            receipt = await self._insert_receipt(
                operation_id=operation,
                action=HoldAction.RELEASE,
                disposition=disposition,
                scope=resolved_scope,
                target_id=resolved_target,
                reason=why,
                actor_id=actor,
                expected_hold_receipt_id=expected,
                prior_hold_receipt_id=prior_receipt_id,
                resulting_hold_receipt_id=resulting_receipt_id,
            )
            if disposition is HoldDisposition.APPLIED:
                await self._db.execute(
                    "UPDATE hold_latches SET active = 0, hold_receipt_id = '', "
                    "reason = '', actor_id = '', set_at = '', revision = revision + 1 "
                    "WHERE scope = ? AND target_id = ? AND active = 1 "
                    "AND hold_receipt_id = ?",
                    (resolved_scope.value, resolved_target, expected),
                )
            current = _latch_from_row(
                await self._read_latch_row(resolved_scope, resolved_target)
            )
            publication = await self._prepare_history_publication()
            mutation = HoldMutation(receipt=receipt, current=current)
        await self._complete_history_publication(publication)
        return mutation

    async def get_hold(
        self, scope: HoldScope | str, target_id: Optional[str] = None
    ) -> Optional[HoldState]:
        try:
            async with self._evidence_protocol():
                return await self._get_hold(scope, target_id)
        except Exception as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _get_hold(
        self,
        scope: HoldScope | str,
        target_id: Optional[str] = None,
        *,
        validate_global_history: bool = True,
    ) -> Optional[HoldState]:
        resolved_scope = _coerce_scope(scope)
        resolved_target = _target(resolved_scope, target_id)
        targets = ((resolved_scope, resolved_target),)
        async with self._db.transaction():
            await self._lock_read_targets(targets)
            await self._assert_host_latch_shape()
            latch = _latch_from_row(
                await self._read_latch_row(resolved_scope, resolved_target)
            )
            if validate_global_history:
                await self._validate_latch_projection(
                    latch,
                    resolved_scope,
                    resolved_target,
                )
            else:
                await self._validate_latch_projection(
                    latch,
                    resolved_scope,
                    resolved_target,
                    validate_global_history=False,
                )
            return latch

    async def get_effective(self, agent_id: str) -> EffectiveHoldState:
        """Read host + agent latches in one database snapshot."""

        try:
            async with self._evidence_protocol():
                return await self._get_effective(agent_id)
        except Exception as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _get_effective(self, agent_id: str) -> EffectiveHoldState:
        """Read host + agent latches in one locked database snapshot."""

        agent = _required_text(agent_id, "agent_id")
        targets = (
            (HoldScope.HOST, HOST_HOLD_TARGET),
            (HoldScope.AGENT, agent),
        )
        async with self._db.transaction():
            await self._lock_read_targets(targets)
            await self._assert_host_latch_shape()
            rows = await self._db.fetchall(
                f"SELECT {_LATCH_COLUMNS} FROM hold_latches "
                "WHERE (scope = ? AND target_id = ?) "
                "OR (scope = ? AND target_id = ?)",
                (
                    HoldScope.HOST.value,
                    HOST_HOLD_TARGET,
                    HoldScope.AGENT.value,
                    agent,
                ),
            )
            host: Optional[HoldState] = None
            agent_state: Optional[HoldState] = None
            seen: set[tuple[str, str]] = set()
            for row in rows:
                key = (str(row[0]), str(row[1]))
                if key in seen:
                    raise HoldCorruptStateError("duplicate hold latch key")
                seen.add(key)
                state = _latch_from_row(row)
                if state is None:
                    continue
                if state.scope is HoldScope.HOST:
                    host = state
                elif state.target_id == agent:
                    agent_state = state
                else:
                    raise HoldCorruptStateError(
                        "effective hold query returned a foreign agent"
                    )
            await self._validate_latch_projection(
                host,
                HoldScope.HOST,
                HOST_HOLD_TARGET,
                validate_global_history=False,
            )
            await self._validate_latch_projection(
                agent_state,
                HoldScope.AGENT,
                agent,
                validate_global_history=False,
            )
            await self._assert_global_history_intact()
            return EffectiveHoldState(host=host, agent=agent_state)

    async def get_receipt(self, operation_id: str) -> Optional[HoldReceipt]:
        try:
            async with self._evidence_protocol():
                return await self._get_receipt(operation_id)
        except Exception as exc:
            domain_error = _domain_error_from_chain(exc)
            if domain_error is not None:
                raise domain_error from exc
            raise

    async def _get_receipt(self, operation_id: str) -> Optional[HoldReceipt]:
        """Read one receipt only after proving its target authority graph."""

        operation = _required_text(operation_id, "operation_id")
        async with self._db.transaction():
            # The absence path still depends on the global receipt/anchor pair.
            # Lock it before the first witness query; otherwise READ COMMITTED
            # can stitch together rows from opposite sides of a writer commit.
            await self._lock_read_history()
            row = await self._validate_operation_witness(operation)
            if row is None:
                await self._assert_global_history_intact()
                return None
            receipt = _receipt_from_row(row)
            targets = ((receipt.scope, receipt.target_id),)
            await self._lock_read_targets(targets, history_locked=True)
            await self._assert_host_latch_shape()
            latch = _latch_from_row(
                await self._read_latch_row(receipt.scope, receipt.target_id)
            )
            await self._validate_latch_projection(
                latch, receipt.scope, receipt.target_id
            )
            return receipt


__all__ = [
    "HOST_HOLD_TARGET",
    "EffectiveHoldState",
    "HoldAction",
    "HoldCorruptStateError",
    "HoldDisposition",
    "HoldIdempotencyConflict",
    "HoldMutation",
    "HoldReceipt",
    "HoldScope",
    "HoldState",
    "HoldStateError",
    "HoldStore",
]
