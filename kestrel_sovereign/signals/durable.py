"""Durable, scoped delivery for normalized signal envelopes.

``signal_log`` is an outcome/audit trail.  This module is deliberately a
separate ledger: it commits a normalized signal before a durable consumer can
claim it, and retains the lease/acknowledgement state required to resume after
a process loss.  It is used through :class:`SignalDispatcher`; callers should
not bypass the dispatcher to turn an external event into a workflow wake.

The ledger is at-least-once by design.  A consumer's side effects must be
idempotent on ``event_id``/``delivery_id``: a process can die after the side
effect and before ``ack_delivery``.  The lease token prevents a stale executor
from acknowledging a delivery that was reclaimed by another executor.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

from kestrel_sdk.signals import Signal

from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.signals.store import _json_default, _serialize_chain
from kestrel_sovereign.storage.db.interface import DatabaseBackend

PENDING = "pending"
INITIAL_RESERVED = "initial_reserved"
LEASED = "leased"
RETRY = "retry"
ACKNOWLEDGED = "acknowledged"
FAILED = "failed"
# A validation/cycle refusal has no effects to retry, but an ACK-bearing
# provider can redeliver it when its provider-side ACK was lost.  Keep that
# fact distinct from a normal terminal worker failure: only this state is a
# durable, idempotent receipt for a redelivery.
TERMINAL_ACKABLE = "terminal_ackable"
_TERMINAL_STATUSES = frozenset({ACKNOWLEDGED, FAILED, TERMINAL_ACKABLE})
_CLAIMABLE_STATUSES = frozenset({PENDING, RETRY})
_DEACTIVATED_CONSUMER_ERROR = "durable consumer deactivated"
_SELECTOR_KEY = re.compile(r"^(?:payload\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*|session_id|kind)=(.+)$")
_PERSISTED_PAYLOAD = object()
# Callers that own a managed dispatcher normally supply their configured
# threshold explicitly.  Keep direct-store compatibility conservative: a
# recently registered dispatcher must not lose a lease simply because a
# polling caller has no dispatcher instance from which to obtain that policy.
_DEFAULT_RUNTIME_OWNER_STALE_AFTER = timedelta(minutes=2)
_MAX_SOURCE_SEQUENCE = (1 << 63) - 1
_SOURCE_SEQUENCE_CHECK_EXPRESSION = (
    "source_sequence IS NOT NULL AND source_sequence >= 1"
)
_SQLITE_SOURCE_SEQUENCE_GUARD_PREFIX = (
    "durable_signal_events_require_source_sequence_"
)
_POSTGRES_SOURCE_SEQUENCE_RECOVERY_TRIGGER_PREFIX = (
    "durable_signal_events_source_sequence_recovery_"
)
_POSTGRES_SOURCE_SEQUENCE_RECOVERY_FUNCTION_PREFIX = (
    "kestrel_durable_signal_source_sequence_recovery_"
)
_SQLITE_SOURCE_SEQUENCE_COUNTER_FENCE_PREFIX = (
    "durable_signal_source_sequences_recovery_fence_"
)
_POSTGRES_SOURCE_SEQUENCE_COUNTER_TRIGGER_PREFIX = (
    "durable_signal_source_sequences_recovery_fence_"
)
_POSTGRES_SOURCE_SEQUENCE_COUNTER_FUNCTION_PREFIX = (
    "kestrel_durable_signal_counter_recovery_fence_"
)
_POSTGRES_SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE = 256
_POSTGRES_SOURCE_SEQUENCE_INDEX_LOCK = (0x4B455354, 0x53455149)
_SOURCE_SEQUENCE_LOSS_ERROR = (
    "both exact counter copies were lost for a previously seen scope "
    "or provide no positive high-water evidence"
)


def _canonical_schema_sql(value: str) -> str:
    """Normalize only insignificant schema whitespace and a final semicolon."""

    return " ".join(value.strip().rstrip(";").split()).casefold()


def _canonical_postgres_check_expression(value: Any) -> str:
    """Normalize PostgreSQL's deparsed form of our owned CHECK expression."""

    return re.sub(r"[\s()\"]+", "", str(value)).casefold().replace("::bigint", "")


def _postgres_catalog_char(value: Any) -> str:
    """Normalize PostgreSQL's internal ``char`` fields across drivers."""

    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("ascii")
    return str(value)


def _quoted_identifier(value: str) -> str:
    """Quote one catalog-sourced SQL identifier without changing its name."""

    return '"' + value.replace('"', '""') + '"'


def _sqlite_source_sequence_guard_definitions() -> tuple[tuple[str, str], ...]:
    """Return the definition-addressed SQLite guard family.

    The name fingerprint covers the pair as one mechanism, with names excluded
    from the digest. Changing either trigger therefore rotates both names and
    makes an older family enumerable rather than silently indistinguishable
    from the desired schema.
    """

    seen_without_high_water = """
                    EXISTS (
                        SELECT 1
                        FROM durable_signal_source_sequence_seen
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ) AND MAX(COALESCE((
                        SELECT recovery_sequence
                        FROM durable_signal_source_sequence_recovery
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ), 0), COALESCE((
                        SELECT high_water_sequence
                        FROM durable_signal_source_sequence_high_water
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ), 0)) < 1
    """.strip()
    templates = (
        (
            "insert",
            f"""
            CREATE TRIGGER {{name}}
            BEFORE INSERT ON durable_signal_events
            BEGIN
                SELECT CASE
                    WHEN NEW.source_sequence IS NULL OR NEW.source_sequence < 1
                    THEN RAISE(ABORT, 'durable signal source sequence is required')
                END;
                SELECT CASE
                    WHEN {seen_without_high_water}
                    THEN RAISE(ABORT, '{_SOURCE_SEQUENCE_LOSS_ERROR}')
                END;
                INSERT INTO durable_signal_source_sequence_recovery (
                    agent_id, source, recovery_sequence
                ) VALUES (NEW.agent_id, NEW.source, NEW.source_sequence)
                ON CONFLICT (agent_id, source) DO UPDATE
                SET recovery_sequence = MAX(
                    durable_signal_source_sequence_recovery.recovery_sequence,
                    excluded.recovery_sequence
                );
                INSERT INTO durable_signal_source_sequence_high_water (
                    agent_id, source, high_water_sequence
                ) VALUES (NEW.agent_id, NEW.source, NEW.source_sequence)
                ON CONFLICT (agent_id, source) DO UPDATE
                SET high_water_sequence = MAX(
                    durable_signal_source_sequence_high_water.high_water_sequence,
                    excluded.high_water_sequence
                );
                INSERT INTO durable_signal_source_sequence_seen (agent_id, source)
                VALUES (NEW.agent_id, NEW.source)
                ON CONFLICT (agent_id, source) DO NOTHING;
            END
            """.strip(),
        ),
        (
            "update",
            f"""
            CREATE TRIGGER {{name}}
            BEFORE UPDATE OF source_sequence ON durable_signal_events
            BEGIN
                SELECT CASE
                    WHEN NEW.source_sequence IS NULL OR NEW.source_sequence < 1
                    THEN RAISE(ABORT, 'durable signal source sequence is required')
                END;
                SELECT CASE
                    WHEN {seen_without_high_water}
                    THEN RAISE(ABORT, '{_SOURCE_SEQUENCE_LOSS_ERROR}')
                END;
                INSERT INTO durable_signal_source_sequence_recovery (
                    agent_id, source, recovery_sequence
                ) VALUES (NEW.agent_id, NEW.source, NEW.source_sequence)
                ON CONFLICT (agent_id, source) DO UPDATE
                SET recovery_sequence = MAX(
                    durable_signal_source_sequence_recovery.recovery_sequence,
                    excluded.recovery_sequence
                );
                INSERT INTO durable_signal_source_sequence_high_water (
                    agent_id, source, high_water_sequence
                ) VALUES (NEW.agent_id, NEW.source, NEW.source_sequence)
                ON CONFLICT (agent_id, source) DO UPDATE
                SET high_water_sequence = MAX(
                    durable_signal_source_sequence_high_water.high_water_sequence,
                    excluded.high_water_sequence
                );
                INSERT INTO durable_signal_source_sequence_seen (agent_id, source)
                VALUES (NEW.agent_id, NEW.source)
                ON CONFLICT (agent_id, source) DO NOTHING;
            END
            """.strip(),
        ),
    )
    material = "\n".join(template for _role, template in templates).encode("utf-8")
    fingerprint = hashlib.blake2s(material, digest_size=4).hexdigest()
    return tuple(
        (
            f"{_SQLITE_SOURCE_SEQUENCE_GUARD_PREFIX}{role}_{fingerprint}",
            template.format(
                name=f"{_SQLITE_SOURCE_SEQUENCE_GUARD_PREFIX}{role}_{fingerprint}"
            ),
        )
        for role, template in templates
    )


_SQLITE_SOURCE_SEQUENCE_GUARDS = _sqlite_source_sequence_guard_definitions()


def _sqlite_source_sequence_counter_fence_definitions(
) -> tuple[tuple[str, str], ...]:
    """Return the atomic SQLite fence for primary-only legacy writers."""

    seen_without_high_water = """
                    EXISTS (
                        SELECT 1
                        FROM durable_signal_source_sequence_seen
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ) AND MAX(COALESCE((
                        SELECT recovery_sequence
                        FROM durable_signal_source_sequence_recovery
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ), 0), COALESCE((
                        SELECT high_water_sequence
                        FROM durable_signal_source_sequence_high_water
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ), 0)) < 1
    """.strip()
    templates = (
        (
            "insert_check",
            f"""
            CREATE TRIGGER {{name}}
            BEFORE INSERT ON durable_signal_source_sequences
            BEGIN
                SELECT CASE
                    WHEN {seen_without_high_water}
                    THEN RAISE(ABORT, '{_SOURCE_SEQUENCE_LOSS_ERROR}')
                END;
            END
            """.strip(),
        ),
        (
            "update_check",
            f"""
            CREATE TRIGGER {{name}}
            BEFORE UPDATE OF current_sequence ON durable_signal_source_sequences
            BEGIN
                SELECT CASE
                    WHEN {seen_without_high_water}
                    THEN RAISE(ABORT, '{_SOURCE_SEQUENCE_LOSS_ERROR}')
                END;
            END
            """.strip(),
        ),
        (
            "insert_mirror",
            """
            CREATE TRIGGER {name}
            AFTER INSERT ON durable_signal_source_sequences
            BEGIN
                UPDATE durable_signal_source_sequences
                SET current_sequence = MAX(
                    current_sequence,
                    MAX(COALESCE((
                        SELECT recovery_sequence
                        FROM durable_signal_source_sequence_recovery
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ), 0), COALESCE((
                        SELECT high_water_sequence
                        FROM durable_signal_source_sequence_high_water
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ), 0))
                )
                WHERE agent_id = NEW.agent_id AND source = NEW.source;
                INSERT INTO durable_signal_source_sequence_recovery (
                    agent_id, source, recovery_sequence
                )
                SELECT agent_id, source, current_sequence
                FROM durable_signal_source_sequences
                WHERE agent_id = NEW.agent_id AND source = NEW.source
                ON CONFLICT (agent_id, source) DO UPDATE
                SET recovery_sequence = MAX(
                    durable_signal_source_sequence_recovery.recovery_sequence,
                    excluded.recovery_sequence
                );
                INSERT INTO durable_signal_source_sequence_high_water (
                    agent_id, source, high_water_sequence
                )
                SELECT agent_id, source, current_sequence
                FROM durable_signal_source_sequences
                WHERE agent_id = NEW.agent_id AND source = NEW.source
                ON CONFLICT (agent_id, source) DO UPDATE
                SET high_water_sequence = MAX(
                    durable_signal_source_sequence_high_water.high_water_sequence,
                    excluded.high_water_sequence
                );
                INSERT INTO durable_signal_source_sequence_seen (agent_id, source)
                SELECT agent_id, source
                FROM durable_signal_source_sequences
                WHERE agent_id = NEW.agent_id AND source = NEW.source
                  AND current_sequence > 0
                ON CONFLICT (agent_id, source) DO NOTHING;
            END
            """.strip(),
        ),
        (
            "update_mirror",
            """
            CREATE TRIGGER {name}
            AFTER UPDATE OF current_sequence
            ON durable_signal_source_sequences
            BEGIN
                UPDATE durable_signal_source_sequences
                SET current_sequence = MAX(
                    current_sequence,
                    MAX(COALESCE((
                        SELECT recovery_sequence
                        FROM durable_signal_source_sequence_recovery
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ), 0), COALESCE((
                        SELECT high_water_sequence
                        FROM durable_signal_source_sequence_high_water
                        WHERE agent_id = NEW.agent_id AND source = NEW.source
                    ), 0))
                )
                WHERE agent_id = NEW.agent_id AND source = NEW.source
                  AND current_sequence < MAX(COALESCE((
                          SELECT recovery_sequence
                          FROM durable_signal_source_sequence_recovery
                          WHERE agent_id = NEW.agent_id AND source = NEW.source
                      ), 0), COALESCE((
                          SELECT high_water_sequence
                          FROM durable_signal_source_sequence_high_water
                          WHERE agent_id = NEW.agent_id AND source = NEW.source
                      ), 0));
                INSERT INTO durable_signal_source_sequence_recovery (
                    agent_id, source, recovery_sequence
                )
                SELECT agent_id, source, current_sequence
                FROM durable_signal_source_sequences
                WHERE agent_id = NEW.agent_id AND source = NEW.source
                ON CONFLICT (agent_id, source) DO UPDATE
                SET recovery_sequence = MAX(
                    durable_signal_source_sequence_recovery.recovery_sequence,
                    excluded.recovery_sequence
                );
                INSERT INTO durable_signal_source_sequence_high_water (
                    agent_id, source, high_water_sequence
                )
                SELECT agent_id, source, current_sequence
                FROM durable_signal_source_sequences
                WHERE agent_id = NEW.agent_id AND source = NEW.source
                ON CONFLICT (agent_id, source) DO UPDATE
                SET high_water_sequence = MAX(
                    durable_signal_source_sequence_high_water.high_water_sequence,
                    excluded.high_water_sequence
                );
                INSERT INTO durable_signal_source_sequence_seen (agent_id, source)
                SELECT agent_id, source
                FROM durable_signal_source_sequences
                WHERE agent_id = NEW.agent_id AND source = NEW.source
                  AND current_sequence > 0
                ON CONFLICT (agent_id, source) DO NOTHING;
            END
            """.strip(),
        ),
    )
    material = "\n".join(template for _role, template in templates).encode("utf-8")
    fingerprint = hashlib.blake2s(material, digest_size=4).hexdigest()
    return tuple(
        (
            f"{_SQLITE_SOURCE_SEQUENCE_COUNTER_FENCE_PREFIX}{role}_{fingerprint}",
            template.format(
                name=(
                    f"{_SQLITE_SOURCE_SEQUENCE_COUNTER_FENCE_PREFIX}"
                    f"{role}_{fingerprint}"
                )
            ),
        )
        for role, template in templates
    )


_SQLITE_SOURCE_SEQUENCE_COUNTER_FENCES = (
    _sqlite_source_sequence_counter_fence_definitions()
)


@dataclass(frozen=True)
class _PostgresSourceSequenceRecoveryDefinition:
    """One exact member of PostgreSQL's statement-level mirror family."""

    role: str
    function_name: str
    trigger_name: str
    function_body: str
    function_ddl: str
    trigger_ddl: str
    trigger_type: int
    transition_table: str


def _postgres_source_sequence_recovery_function_body(
    transition_table: str,
) -> str:
    """Aggregate one statement's transition rows into exact scope maxima."""

    return f"""
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM (
                SELECT DISTINCT agent_id, source
                FROM {transition_table}
            ) AS changed_scope
            JOIN durable_signal_source_sequence_seen AS seen
              ON seen.agent_id = changed_scope.agent_id
             AND seen.source = changed_scope.source
            LEFT JOIN durable_signal_source_sequence_recovery AS recovery
              ON recovery.agent_id = changed_scope.agent_id
             AND recovery.source = changed_scope.source
            LEFT JOIN durable_signal_source_sequence_high_water AS high_water
              ON high_water.agent_id = changed_scope.agent_id
             AND high_water.source = changed_scope.source
            WHERE GREATEST(
                COALESCE(recovery.recovery_sequence, 0),
                COALESCE(high_water.high_water_sequence, 0)
            ) < 1
        ) THEN
            RAISE EXCEPTION '{_SOURCE_SEQUENCE_LOSS_ERROR}';
        END IF;

        INSERT INTO durable_signal_source_sequence_recovery (
            agent_id, source, recovery_sequence
        )
        SELECT agent_id, source, MAX(source_sequence)
        FROM {transition_table}
        GROUP BY agent_id, source
        HAVING MAX(source_sequence) IS NOT NULL
        ORDER BY agent_id, source
        ON CONFLICT (agent_id, source) DO UPDATE
        SET recovery_sequence = GREATEST(
            durable_signal_source_sequence_recovery.recovery_sequence,
            EXCLUDED.recovery_sequence
        );

        INSERT INTO durable_signal_source_sequence_high_water (
            agent_id, source, high_water_sequence
        )
        SELECT agent_id, source, MAX(source_sequence)
        FROM {transition_table}
        GROUP BY agent_id, source
        HAVING MAX(source_sequence) IS NOT NULL
        ORDER BY agent_id, source
        ON CONFLICT (agent_id, source) DO UPDATE
        SET high_water_sequence = GREATEST(
            durable_signal_source_sequence_high_water.high_water_sequence,
            EXCLUDED.high_water_sequence
        );

        INSERT INTO durable_signal_source_sequence_seen (agent_id, source)
        SELECT agent_id, source
        FROM {transition_table}
        WHERE source_sequence IS NOT NULL
        GROUP BY agent_id, source
        HAVING MAX(source_sequence) >= 1
        ORDER BY agent_id, source
        ON CONFLICT (agent_id, source) DO NOTHING;
        RETURN NULL;
    END
    """.strip()


def _postgres_source_sequence_recovery_definitions(
) -> tuple[_PostgresSourceSequenceRecoveryDefinition, ...]:
    """Return the fingerprint-addressed INSERT and UPDATE mirror family.

    PostgreSQL permits a transition relation only on a single-event trigger;
    INSERT and UPDATE therefore need distinct statement triggers and functions.
    The shared fingerprint covers all four definitions so an edit rotates and
    retires the mechanism as one atomic family.
    """

    roles = (
        ("insert", "i", "INSERT", 4, "kestrel_inserted_source_sequences"),
        ("update", "u", "UPDATE", 16, "kestrel_updated_source_sequences"),
    )
    templates: list[tuple[str, str, str, int, str, str]] = []
    for role, abbreviation, event, trigger_type, transition_table in roles:
        body = _postgres_source_sequence_recovery_function_body(transition_table)
        trigger_template = (
            "CREATE TRIGGER {trigger_name} "
            f"AFTER {event} ON durable_signal_events "
            f"REFERENCING NEW TABLE AS {transition_table} "
            "FOR EACH STATEMENT EXECUTE FUNCTION {function_name}()"
        )
        templates.append(
            (
                role,
                abbreviation,
                body,
                trigger_type,
                transition_table,
                trigger_template,
            )
        )
    material = "\n".join(
        f"{role}\n{body}\n{trigger_template}"
        for role, _abbreviation, body, _type, _table, trigger_template in templates
    ).encode("utf-8")
    fingerprint = hashlib.blake2s(material, digest_size=4).hexdigest()
    definitions: list[_PostgresSourceSequenceRecoveryDefinition] = []
    for (
        role,
        abbreviation,
        body,
        trigger_type,
        transition_table,
        trigger_template,
    ) in templates:
        function_name = (
            f"{_POSTGRES_SOURCE_SEQUENCE_RECOVERY_FUNCTION_PREFIX}"
            f"{abbreviation}_{fingerprint}"
        )
        trigger_name = (
            f"{_POSTGRES_SOURCE_SEQUENCE_RECOVERY_TRIGGER_PREFIX}"
            f"{abbreviation}_{fingerprint}"
        )
        function_ddl = f"""
            CREATE FUNCTION {function_name}()
            RETURNS trigger AS $kestrel$
            {body}
            $kestrel$ LANGUAGE plpgsql
        """.strip()
        trigger_ddl = trigger_template.format(
            trigger_name=trigger_name,
            function_name=function_name,
        )
        definitions.append(
            _PostgresSourceSequenceRecoveryDefinition(
                role=role,
                function_name=function_name,
                trigger_name=trigger_name,
                function_body=body,
                function_ddl=function_ddl,
                trigger_ddl=trigger_ddl,
                trigger_type=trigger_type,
                transition_table=transition_table,
            )
        )
    return tuple(definitions)


_POSTGRES_SOURCE_SEQUENCE_RECOVERY_DEFINITIONS = (
    _postgres_source_sequence_recovery_definitions()
)


@dataclass(frozen=True)
class _PostgresSourceSequenceCounterFenceDefinition:
    """One exact member of PostgreSQL's primary-counter fence family."""

    role: str
    function_name: str
    trigger_name: str
    function_body: str
    function_ddl: str
    trigger_ddl: str
    trigger_type: int


def _postgres_source_sequence_counter_fence_definitions(
) -> tuple[_PostgresSourceSequenceCounterFenceDefinition, ...]:
    """Return the BEFORE repair and AFTER mirror fence as one family."""

    before_body = f"""
    DECLARE
        recovered BIGINT;
    BEGIN
        SELECT GREATEST(
            COALESCE((
                SELECT recovery_sequence
                FROM durable_signal_source_sequence_recovery
                WHERE agent_id = NEW.agent_id AND source = NEW.source
            ), 0),
            COALESCE((
                SELECT high_water_sequence
                FROM durable_signal_source_sequence_high_water
                WHERE agent_id = NEW.agent_id AND source = NEW.source
            ), 0)
        ) INTO recovered;

        IF EXISTS (
            SELECT 1
            FROM durable_signal_source_sequence_seen
            WHERE agent_id = NEW.agent_id AND source = NEW.source
        ) AND recovered < 1 THEN
            RAISE EXCEPTION '{_SOURCE_SEQUENCE_LOSS_ERROR}';
        END IF;
        IF recovered IS NOT NULL AND NEW.current_sequence < recovered THEN
            NEW.current_sequence := recovered;
        END IF;
        RETURN NEW;
    END
    """.strip()
    after_body = """
    BEGIN
        INSERT INTO durable_signal_source_sequence_recovery (
            agent_id, source, recovery_sequence
        ) VALUES (NEW.agent_id, NEW.source, NEW.current_sequence)
        ON CONFLICT (agent_id, source) DO UPDATE
        SET recovery_sequence = GREATEST(
            durable_signal_source_sequence_recovery.recovery_sequence,
            EXCLUDED.recovery_sequence
        );
        INSERT INTO durable_signal_source_sequence_high_water (
            agent_id, source, high_water_sequence
        ) VALUES (NEW.agent_id, NEW.source, NEW.current_sequence)
        ON CONFLICT (agent_id, source) DO UPDATE
        SET high_water_sequence = GREATEST(
            durable_signal_source_sequence_high_water.high_water_sequence,
            EXCLUDED.high_water_sequence
        );
        IF NEW.current_sequence > 0 THEN
            INSERT INTO durable_signal_source_sequence_seen (agent_id, source)
            VALUES (NEW.agent_id, NEW.source)
            ON CONFLICT (agent_id, source) DO NOTHING;
        END IF;
        RETURN NULL;
    END
    """.strip()
    templates = (
        (
            "before",
            before_body,
            23,
            "BEFORE INSERT OR UPDATE ON durable_signal_source_sequences "
            "FOR EACH ROW EXECUTE FUNCTION {function_name}()",
        ),
        (
            "after",
            after_body,
            21,
            "AFTER INSERT OR UPDATE ON durable_signal_source_sequences "
            "FOR EACH ROW EXECUTE FUNCTION {function_name}()",
        ),
    )
    material = "\n".join(
        f"{role}\n{body}\n{trigger_template}"
        for role, body, _trigger_type, trigger_template in templates
    ).encode("utf-8")
    fingerprint = hashlib.blake2s(material, digest_size=4).hexdigest()
    definitions: list[_PostgresSourceSequenceCounterFenceDefinition] = []
    for role, body, trigger_type, trigger_template in templates:
        abbreviation = "b" if role == "before" else "a"
        function_name = (
            f"{_POSTGRES_SOURCE_SEQUENCE_COUNTER_FUNCTION_PREFIX}"
            f"{abbreviation}_{fingerprint}"
        )
        trigger_name = (
            f"{_POSTGRES_SOURCE_SEQUENCE_COUNTER_TRIGGER_PREFIX}"
            f"{abbreviation}_{fingerprint}"
        )
        function_ddl = f"""
            CREATE FUNCTION {function_name}()
            RETURNS trigger AS $kestrel$
            {body}
            $kestrel$ LANGUAGE plpgsql
        """.strip()
        trigger_ddl = "CREATE TRIGGER {trigger_name} ".format(
            trigger_name=trigger_name
        ) + trigger_template.format(function_name=function_name)
        definitions.append(
            _PostgresSourceSequenceCounterFenceDefinition(
                role=role,
                function_name=function_name,
                trigger_name=trigger_name,
                function_body=body,
                function_ddl=function_ddl,
                trigger_ddl=trigger_ddl,
                trigger_type=trigger_type,
            )
        )
    return tuple(definitions)


_POSTGRES_SOURCE_SEQUENCE_COUNTER_FENCE_DEFINITIONS = (
    _postgres_source_sequence_counter_fence_definitions()
)


@runtime_checkable
class SQLiteImmediateTransactionBackend(Protocol):
    """SQLite capability required to serialize durable-ledger bootstrap."""

    def transaction(self, *, immediate: bool = False) -> AsyncContextManager[None]: ...


@runtime_checkable
class PostgresAdvisoryLockBackend(Protocol):
    """PostgreSQL capability required for serialized autocommit DDL."""

    def polled_advisory_lock(
        self, key: tuple[int, int]
    ) -> AsyncContextManager[None]: ...


@dataclass(frozen=True)
class _SourceSequenceSchemaState:
    """Catalog evidence for the additive source-sequence migration."""

    column_exists: bool
    enforced: bool
    fence_exists: bool = False
    fence_validated: bool = False
    fence_definition_valid: bool = False
    column_not_null: bool = False


@dataclass(frozen=True)
class _SQLiteSourceSequenceIndexCatalog:
    """Raw SQLite catalog evidence for the owned source-sequence index."""

    object_type: str
    name: str
    relation_name: str
    index_list_row: Optional[Any] = None
    index_xinfo_rows: tuple[Any, ...] = ()


@dataclass(frozen=True)
class DurableConsumerRegistration:
    """A durable subscriber owned by one agent/tenant.

    ``correlation_selector`` is intentionally a tiny, non-SQL selector.  It
    is either ``None`` (receive every event from ``source``) or an exact
    comparison such as ``"payload.workflow_id=wf-42"``.  The left side may
    name a sanitized payload path, ``session_id``, or ``kind``.  Keeping the
    selector declarative makes subscriptions replayable after restart and
    prevents callers from injecting an ad-hoc database predicate.
    """

    consumer_id: str
    source: str
    agent_id: str
    correlation_selector: Optional[str] = None
    # Zero is intentional: a cursor-owning external producer must retain its
    # event until this consumer has acknowledged it, rather than converting a
    # transient outage into a terminal loss after an arbitrary retry budget.
    max_attempts: int = 5
    lease_seconds: int = 60
    active: bool = True


@dataclass(frozen=True)
class DurableSignalEvent:
    """The canonical, post-sanitization signal persisted for consumers."""

    event_id: str
    source_event_id: Optional[str]
    agent_id: str
    target_agent: str
    source: str
    kind: str
    mode: str
    payload: Any
    session_id: Optional[str]
    # ``caller_identity`` is an opaque, dispatcher-produced ciphertext for a
    # persistence-allowed caller, a keyless ``v2:opaque:...`` event label, or
    # the ``v1:none`` sentinel.  It is never a raw caller identifier in
    # storage. Payload-elided rows deliberately leave it NULL and bind the
    # caller only through their keyed integrity proof plus the verified live
    # retry envelope.
    caller_identity: Optional[str]
    visibility: str
    urgency: str
    dedupe_key: Optional[str]
    causation_chain: list[dict[str, Any]]
    arrived_at: datetime
    committed_at: datetime
    retention_until: datetime
    # Monotonic only within this event's exact ``(agent_id, source)`` scope.
    # Unlike ``committed_at``, this is safe evidence for effect-boundary
    # eligibility across hosts and database backends.
    source_sequence: int = 0


@dataclass(frozen=True)
class DurableSourceBoundary:
    """A durable linearization boundary for one agent-owned signal source.

    An event is after this boundary exactly when it has the same ``agent_id``
    and ``source`` and its committed ``source_sequence`` is strictly greater.
    Comparing a different tenant or source is an error rather than a false
    result that could hide an authority mix-up.
    """

    agent_id: str
    source: str
    sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or not 0 <= self.sequence <= _MAX_SOURCE_SEQUENCE
        ):
            raise ValueError("sequence must be a non-negative 64-bit integer")

    def is_event_eligible(self, event: DurableSignalEvent) -> bool:
        """Return whether ``event`` committed strictly after this boundary."""

        if event.agent_id != self.agent_id or event.source != self.source:
            raise ValueError(
                "Durable source boundaries can compare only events from the "
                "same agent_id and source"
            )
        if (
            not isinstance(event.source_sequence, int)
            or isinstance(event.source_sequence, bool)
            or not 1 <= event.source_sequence <= _MAX_SOURCE_SEQUENCE
        ):
            raise ValueError("event has no valid committed source sequence")
        return event.source_sequence > self.sequence

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-safe record a workflow must persist.

        The record intentionally contains the tenant and source as well as the
        sequence. Rehydration therefore cannot accidentally attach a scalar
        cursor to a different authority scope after a process restart.
        """

        return {
            "version": 1,
            "agent_id": self.agent_id,
            "source": self.source,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DurableSourceBoundary":
        """Rehydrate one exact persisted boundary record, failing closed."""

        if not isinstance(value, Mapping):
            raise ValueError("durable source boundary record must be a mapping")
        expected = {"version", "agent_id", "source", "sequence"}
        if (
            set(value) != expected
            or type(value.get("version")) is not int
            or value.get("version") != 1
        ):
            raise ValueError("unsupported durable source boundary record")
        return cls(
            agent_id=value["agent_id"],
            source=value["source"],
            sequence=value["sequence"],
        )


@dataclass(frozen=True)
class DurableEventPersistence:
    """Result of persisting a source event.

    ``created`` is false when the source event ID had already been accepted
    for the same source and agent/tenant.  The original ``event_id`` is
    returned so callers can record a useful audit result without creating a
    duplicate delivery.  ``delivery_ids`` identifies only the initial
    deliveries inserted by this persistence transaction.
    ``initial_reservations`` is populated only for a payload-elided live
    dispatch: those rows are atomically reserved to the emitting dispatcher
    before commit, but deliberately have no delivery lease deadline until the
    dispatcher activates them after the commit is visible.
    """

    event_id: str
    created: bool
    delivery_ids: tuple[str, ...] = ()
    retention_until: Optional[datetime] = None
    initial_reservations: tuple["DurableInitialDeliveryReservation", ...] = ()
    source_sequence: Optional[int] = None


@dataclass(frozen=True)
class DurableInitialDeliveryReservation:
    """A non-claimable initial reservation created with its event row.

    The token is a reservation capability, never user payload.  It lets the
    emitting dispatcher activate this row *after* commit, then transfer the
    resulting live lease to its chosen executor without exposing a newly
    committed marker-only delivery to another store instance first.
    """

    delivery_id: str
    consumer_id: str
    reservation_token: str
    created_at: datetime


@dataclass(frozen=True)
class DurableDelivery:
    """A delivery, optionally leased to an executor, plus its event."""

    delivery_id: str
    consumer_id: str
    agent_id: str
    event_id: str
    status: str
    attempts: int
    max_attempts: int
    lease_owner: Optional[str]
    lease_token: Optional[str]
    lease_expires_at: Optional[datetime]
    next_attempt_at: Optional[datetime]
    last_error: Optional[str]
    acknowledged_at: Optional[datetime]
    terminal_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    event: DurableSignalEvent

    @property
    def source_sequence(self) -> int:
        """The event's committed sequence in its agent/source scope."""

        return self.event.source_sequence


def _json_dump(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True)


def _json_load(value: Any) -> Any:
    if isinstance(value, (dict, list)) or value is None:
        return value
    return json.loads(value)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class DurableSignalStore(UnifiedStoreBase):
    """Backend-neutral pending-delivery ledger.

    Every agent-facing query includes ``agent_id`` in SQL.  This is an
    authorization boundary for a shared PostgreSQL database, not a caller-side
    post-filter.  SQLite uses the same predicates so standalone behaviour is
    identical.
    """

    EVENTS = "durable_signal_events"
    CONSUMERS = "durable_signal_consumers"
    DELIVERIES = "durable_signal_deliveries"
    RUNTIME_OWNERS = "durable_signal_runtime_owners"
    SOURCE_SEQUENCES = "durable_signal_source_sequences"
    SOURCE_SEQUENCE_RECOVERY = "durable_signal_source_sequence_recovery"
    SOURCE_SEQUENCE_HIGH_WATER = "durable_signal_source_sequence_high_water"
    SOURCE_SEQUENCE_SEEN = "durable_signal_source_sequence_seen"
    SOURCE_SEQUENCE_STATE = "durable_signal_source_sequence_state"
    SOURCE_SEQUENCE_BACKFILL_COLUMN = "backfill_completed"
    SOURCE_SEQUENCE_SCOPE_WORK = "durable_signal_source_sequence_scope_work"
    SOURCE_SEQUENCE_EVENT_WORK = "durable_signal_source_sequence_event_work"
    SOURCE_SEQUENCE_SCOPE_WORK_SEEDED_COLUMN = "scope_work_seeded"
    SOURCE_SEQUENCE_EVENT_WORK_SEEDED_COLUMN = "event_work_seeded"
    SOURCE_SEQUENCE_SCOPE_VALIDATION_COLUMN = "scope_validation_completed"
    SOURCE_SEQUENCE_SCOPE_VALIDATION_AGENT_COLUMN = (
        "scope_validation_after_agent_id"
    )
    SOURCE_SEQUENCE_SCOPE_VALIDATION_SOURCE_COLUMN = (
        "scope_validation_after_source"
    )
    SOURCE_SEQUENCE_GUARD_PREFIX = _SQLITE_SOURCE_SEQUENCE_GUARD_PREFIX
    SOURCE_SEQUENCE_GUARDS = _SQLITE_SOURCE_SEQUENCE_GUARDS
    SOURCE_SEQUENCE_INSERT_GUARD = SOURCE_SEQUENCE_GUARDS[0][0]
    SOURCE_SEQUENCE_UPDATE_GUARD = SOURCE_SEQUENCE_GUARDS[1][0]
    SOURCE_SEQUENCE_NOT_NULL_CHECK = "durable_signal_events_source_sequence_not_null"
    SOURCE_SEQUENCE_RECOVERY_FUNCTION_PREFIX = (
        _POSTGRES_SOURCE_SEQUENCE_RECOVERY_FUNCTION_PREFIX
    )
    SOURCE_SEQUENCE_RECOVERY_TRIGGER_PREFIX = (
        _POSTGRES_SOURCE_SEQUENCE_RECOVERY_TRIGGER_PREFIX
    )
    SOURCE_SEQUENCE_RECOVERY_DEFINITIONS = (
        _POSTGRES_SOURCE_SEQUENCE_RECOVERY_DEFINITIONS
    )
    SOURCE_SEQUENCE_COUNTER_FENCE_PREFIX = (
        _SQLITE_SOURCE_SEQUENCE_COUNTER_FENCE_PREFIX
    )
    SOURCE_SEQUENCE_COUNTER_FENCES = _SQLITE_SOURCE_SEQUENCE_COUNTER_FENCES
    SOURCE_SEQUENCE_COUNTER_TRIGGER_PREFIX = (
        _POSTGRES_SOURCE_SEQUENCE_COUNTER_TRIGGER_PREFIX
    )
    SOURCE_SEQUENCE_COUNTER_FUNCTION_PREFIX = (
        _POSTGRES_SOURCE_SEQUENCE_COUNTER_FUNCTION_PREFIX
    )
    SOURCE_SEQUENCE_COUNTER_FENCE_DEFINITIONS = (
        _POSTGRES_SOURCE_SEQUENCE_COUNTER_FENCE_DEFINITIONS
    )
    SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE = (
        _POSTGRES_SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE
    )
    SOURCE_SEQUENCE_SCOPE_INDEX = (
        "idx_durable_signal_events_scope_sequence"
    )
    # Payload-eliding privacy modes cannot retain their canonical input in the
    # event row. This side table stores only a fixed-width integrity binding.
    EVENT_INTEGRITY = "durable_signal_event_integrity"

    def __init__(self, backend: DatabaseBackend):
        # ``SignalLogStore`` historically accepts the ``AsyncDatabase``
        # compatibility facade as well as a native ``DatabaseBackend``.  The
        # dispatcher derives this ledger from that store, so unwrap only the
        # legacy facade (which exposes ``fetchall`` but not ``fetch_all``)
        # rather than requiring every existing signal-store embedding to
        # migrate its construction path. Durable delivery uses the native
        # ``fetch_one`` / ``fetch_all`` contract and must share the same
        # transaction domain as the signal log. The capability check leaves
        # native backends and test doubles untouched.
        native_backend = backend
        if not hasattr(backend, "fetch_all") and hasattr(backend, "backend"):
            native_backend = backend.backend
        super().__init__(native_backend)

    async def initialize(self) -> None:
        """Serialize PostgreSQL's transactional and autocommit schema phases."""

        if self.is_postgres:
            if not isinstance(self._backend, PostgresAdvisoryLockBackend):
                raise RuntimeError(
                    "PostgreSQL durable signal bootstrap requires session "
                    "advisory locks for concurrent index DDL"
                )
            polled_advisory_lock = cast(
                PostgresAdvisoryLockBackend, self._backend
            ).polled_advisory_lock
            async with polled_advisory_lock(
                _POSTGRES_SOURCE_SEQUENCE_INDEX_LOCK
            ):
                await self._initialize_locked()
            return
        await self._initialize_locked()

    async def _initialize_locked(self) -> None:
        """Bootstrap/evolve the ledger under one cross-process schema lock.

        The delivery tables are shared by independently restarted dispatchers.
        In particular, the integrity side table was added after the original
        ledger, so a plain sequence of ``CREATE IF NOT EXISTS`` calls is not a
        migration protocol: another process can observe a partial schema and
        begin routing before the additive migration finishes.
        """

        ts_type = self.timestamp_type()
        ts_default = self.now_default()
        json_type = self.json_type()
        bool_type = self.boolean_type()
        postgres_source_sequence_check = (
            f",\n                CONSTRAINT {self.SOURCE_SEQUENCE_NOT_NULL_CHECK} "
            f"CHECK ({_SOURCE_SEQUENCE_CHECK_EXPRESSION})"
            if self.is_postgres
            else ""
        )
        statements = (
            f"""
            CREATE TABLE IF NOT EXISTS {self.EVENTS} (
                event_id TEXT PRIMARY KEY,
                source_event_id TEXT,
                agent_id TEXT NOT NULL,
                target_agent TEXT NOT NULL,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                mode TEXT NOT NULL,
                payload {json_type} NOT NULL,
                session_id TEXT,
                caller_identity TEXT,
                visibility TEXT NOT NULL,
                urgency TEXT NOT NULL,
                dedupe_key TEXT,
                causation_chain {json_type} NOT NULL,
                arrived_at {ts_type} NOT NULL,
                committed_at {ts_type} {ts_default},
                retention_until {ts_type} NOT NULL,
                source_sequence BIGINT NOT NULL{postgres_source_sequence_check},
                UNIQUE (agent_id, source, source_event_id)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.SOURCE_SEQUENCES} (
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL,
                current_sequence BIGINT NOT NULL,
                PRIMARY KEY (agent_id, source),
                CHECK (current_sequence >= 0)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.SOURCE_SEQUENCE_RECOVERY} (
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL,
                recovery_sequence BIGINT NOT NULL,
                PRIMARY KEY (agent_id, source),
                CHECK (recovery_sequence >= 0)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.SOURCE_SEQUENCE_HIGH_WATER} (
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL,
                high_water_sequence BIGINT NOT NULL,
                PRIMARY KEY (agent_id, source),
                CHECK (high_water_sequence >= 0)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.SOURCE_SEQUENCE_SEEN} (
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY (agent_id, source)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.SOURCE_SEQUENCE_STATE} (
                singleton INTEGER PRIMARY KEY,
                counter_history_lost {bool_type} NOT NULL,
                {self.SOURCE_SEQUENCE_BACKFILL_COLUMN} {bool_type}
                    NOT NULL DEFAULT FALSE,
                {self.SOURCE_SEQUENCE_SCOPE_WORK_SEEDED_COLUMN} {bool_type}
                    NOT NULL DEFAULT FALSE,
                {self.SOURCE_SEQUENCE_EVENT_WORK_SEEDED_COLUMN} {bool_type}
                    NOT NULL DEFAULT FALSE,
                {self.SOURCE_SEQUENCE_SCOPE_VALIDATION_COLUMN} {bool_type}
                    NOT NULL DEFAULT FALSE,
                {self.SOURCE_SEQUENCE_SCOPE_VALIDATION_AGENT_COLUMN} TEXT,
                {self.SOURCE_SEQUENCE_SCOPE_VALIDATION_SOURCE_COLUMN} TEXT,
                CHECK (singleton = 1)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.SOURCE_SEQUENCE_SCOPE_WORK} (
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY (agent_id, source)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.SOURCE_SEQUENCE_EVENT_WORK} (
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY (agent_id, source, event_id)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.CONSUMERS} (
                agent_id TEXT NOT NULL,
                consumer_id TEXT NOT NULL,
                source TEXT NOT NULL,
                correlation_selector TEXT,
                max_attempts INTEGER NOT NULL,
                lease_seconds INTEGER NOT NULL,
                active {bool_type} NOT NULL,
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default},
                PRIMARY KEY (agent_id, consumer_id)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.DELIVERIES} (
                delivery_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                consumer_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL,
                lease_owner TEXT,
                lease_token TEXT,
                lease_expires_at {ts_type},
                next_attempt_at {ts_type},
                last_error TEXT,
                acknowledged_at {ts_type},
                terminal_at {ts_type},
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default},
                UNIQUE (agent_id, consumer_id, event_id),
                FOREIGN KEY (event_id) REFERENCES {self.EVENTS}(event_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (agent_id, consumer_id)
                    REFERENCES {self.CONSUMERS}(agent_id, consumer_id)
                    ON DELETE RESTRICT
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.RUNTIME_OWNERS} (
                agent_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                heartbeat_at {ts_type} NOT NULL,
                stopped_at {ts_type},
                created_at {ts_type} {ts_default},
                updated_at {ts_type} {ts_default},
                PRIMARY KEY (agent_id, owner_id)
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {self.EVENT_INTEGRITY} (
                event_id TEXT PRIMARY KEY,
                integrity_binding TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES {self.EVENTS}(event_id)
                    ON DELETE CASCADE
            )
            """,
        )
        postgres_finalization_required = False
        async with self._schema_bootstrap_transaction():
            source_sequences_existed = await self._table_exists(
                self.SOURCE_SEQUENCES
            )
            source_recovery_existed = await self._table_exists(
                self.SOURCE_SEQUENCE_RECOVERY
            )
            source_high_water_existed = await self._table_exists(
                self.SOURCE_SEQUENCE_HIGH_WATER
            )
            source_seen_existed = await self._table_exists(
                self.SOURCE_SEQUENCE_SEEN
            )
            for statement in statements:
                await self._backend.execute(statement)
            await self._ensure_source_sequence_backfill_marker_column()
            # ``CREATE TABLE IF NOT EXISTS`` cannot evolve an existing durable
            # ledger. Keep the caller representation additive: legacy rows
            # lack it and are rejected for caller-bearing replay rather than
            # silently accepting an unbound live caller.
            await self._ensure_caller_identity_column()
            source_state = await self._source_sequence_schema_state()
            await self._ensure_source_sequence_state()
            if self.is_postgres:
                backfill_completed = (
                    await self._source_sequence_backfill_completed()
                )
                recovery_sync_valid = (
                    await self._postgres_source_sequence_recovery_sync_valid()
                )
                # The marker remains proof only while the write fence and
                # recovery mirror that preserve its conclusion remain intact.
                # A newly recreated recovery table also needs the exact
                # primary counters adopted again. Invalidate before repairing
                # either shape so interruption cannot leave a false all-clear.
                if backfill_completed and (
                    not source_state.fence_definition_valid
                    or not recovery_sync_valid
                    or not await self._source_sequence_scope_validation_completed()
                    or (
                        source_sequences_existed
                        and not source_recovery_existed
                    )
                    or not source_high_water_existed
                    or not source_seen_existed
                ):
                    await self._set_source_sequence_backfill_completed(False)
                    backfill_completed = False
                if not source_state.enforced:
                    # Commit the additive column and NOT VALID write fence
                    # before touching legacy history. PostgreSQL retains an
                    # ACCESS EXCLUSIVE lock acquired by ALTER TABLE until the
                    # transaction ends; later phases keep history and index
                    # scans out of it.
                    await self._install_postgres_source_sequence_fence(source_state)
                await self._ensure_postgres_source_sequence_recovery_sync()
                await self._ensure_postgres_source_sequence_counter_fence()
                if not source_sequences_existed:
                    # CREATE TABLE and this exact recovery adoption commit as
                    # one schema phase. A primary-only legacy replica can
                    # therefore observe neither an empty recreated table nor a
                    # value below the independently durable watermark.
                    await self._restore_primary_source_sequences_from_recovery()
                postgres_finalization_required = (
                    not source_state.enforced or not backfill_completed
                )
            elif self.is_sqlite:
                await self._install_sqlite_source_sequence_counter_fence()
                if not source_sequences_existed:
                    await self._restore_primary_source_sequences_from_recovery()
                if not source_state.enforced:
                    await self._ensure_source_sequence_column(source_state)
                    # SQLite cannot add NOT NULL to an existing column. These
                    # guards are both the mixed-version fence and the durable
                    # completion marker for its atomic, reserved-writer
                    # migration transaction.
                    await self._install_sqlite_source_sequence_guards()
                    await self._backfill_source_sequences()
                elif not source_state.fence_exists:
                    # The desired pair is valid, but superseded family members
                    # remain. Retire them inside the same reserved-writer
                    # transaction so the durable shape converges as one set.
                    await self._install_sqlite_source_sequence_guards()
            else:
                raise RuntimeError(
                    "Durable signal source-sequence migration supports only "
                    "sqlite or postgres"
                )
            if (
                (not source_recovery_existed and source_sequences_existed)
                or not source_high_water_existed
                or not source_seen_existed
            ):
                # One-time adoption of every independent exact watermark and
                # the loss marker. Primary/recovery values include purged
                # history, unlike retained event reconstruction. The database
                # mirrors are already live here, closing the rolling-upgrade
                # race with an older writer.
                if self.is_postgres:
                    # Clearing the marker above makes phase two adopt these
                    # rows after this event-trigger DDL transaction commits.
                    postgres_finalization_required = True
                else:
                    await self._adopt_source_sequence_recovery()
            if not postgres_finalization_required:
                await self._ensure_ordinary_indexes()
                if self.is_sqlite:
                    await self._ensure_source_sequence_index()

        if self.is_postgres:
            # PostgreSQL unique indexes admit multiple NULL keys. Build the
            # final exact index before assigning legacy history so every
            # retained-maximum reconciliation performed by a batch is an
            # indexed lookup. The separately durable event work table orders
            # NULL rows by immutable event ID without repeatedly sorting the
            # wide event ledger.
            await self._ensure_postgres_source_sequence_index_concurrently()

        if postgres_finalization_required:
            # Each PostgreSQL history batch is its own durable transaction.
            # Restart resumes from the remaining NULL rows and the exact
            # counters committed beside earlier batches; the marker flips only
            # after a final empty check under the same bootstrap lock.
            await self._complete_postgres_source_sequence_backfill()

            # Counter/recovery row locks from the last batch are gone before
            # any relation-locking completion work begins. Keep ordinary-index
            # checks and CHECK validation in their own transaction as well, so
            # none of those locks survive into SET NOT NULL.
            async with self._schema_bootstrap_transaction():
                source_state = await self._source_sequence_schema_state()
                if not await self._source_sequence_backfill_completed():
                    raise RuntimeError(
                        "Durable signal source-sequence completion marker "
                        "disappeared before finalization"
                    )
                if (
                    not source_state.column_exists
                    or not source_state.fence_exists
                    or not source_state.fence_definition_valid
                ):
                    raise RuntimeError(
                        "Durable signal source-sequence write fence disappeared "
                        "before finalization"
                    )
                await self._ensure_ordinary_indexes()
                await self._validate_postgres_source_sequence_fence(source_state)

        if postgres_finalization_required:
            # Freshly recheck the catalog and durable marker once more. This
            # transaction requests only the final metadata ACCESS EXCLUSIVE
            # lock and commits immediately after it is granted.
            async with self._schema_bootstrap_transaction():
                source_state = await self._source_sequence_schema_state()
                if not await self._source_sequence_backfill_completed():
                    raise RuntimeError(
                        "Durable signal source-sequence completion marker "
                        "disappeared before NOT NULL enforcement"
                    )
                await self._enforce_postgres_source_sequence_required(source_state)

    @asynccontextmanager
    async def _schema_bootstrap_transaction(self) -> AsyncIterator[None]:
        """Serialize every fresh and additive ledger migration.

        PostgreSQL uses a fixed transaction advisory key; SQLite reserves its
        writer before schema inspection. Each phase keeps its catalog recheck
        and mutations in one transaction rather than relying on an
        instance-local ``initialized`` flag. PostgreSQL's source migration has
        separate fence, backfill, index/validation, and final NOT NULL phases;
        no scope-row or index lock is retained into ACCESS EXCLUSIVE metadata
        enforcement.
        """

        if self.is_postgres:
            async with self._backend.transaction():
                await self._backend.fetch_val(
                    "SELECT pg_advisory_xact_lock(hashtext('kestrel.durable_signal.bootstrap'))"
                )
                yield
            return
        if self.is_sqlite:
            transaction = self._sqlite_immediate_transaction()
            async with transaction:
                yield
            return
        raise RuntimeError("Durable signal delivery supports only sqlite or postgres databases")

    def _sqlite_immediate_transaction(self) -> AsyncContextManager[None]:
        """Return a SQLite transaction that reserves the writer before reads."""

        if not isinstance(self._backend, SQLiteImmediateTransactionBackend):
            raise RuntimeError(
                "SQLite durable signal delivery requires transaction(immediate=True) "
                "for schema bootstrap"
            )
        transaction = cast(SQLiteImmediateTransactionBackend, self._backend).transaction
        try:
            return transaction(immediate=True)
        except TypeError as exc:
            raise RuntimeError(
                "SQLite durable signal delivery requires transaction(immediate=True) "
                "for schema bootstrap"
            ) from exc

    async def _table_exists(self, table: str) -> bool:
        """Return whether one owned table exists in the current schema."""

        if self.is_postgres:
            return bool(
                await self._backend.fetch_val(
                    "SELECT to_regclass(?) IS NOT NULL", (table,)
                )
            )
        if self.is_sqlite:
            return bool(
                await self._backend.fetch_val(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = ?",
                    (table,),
                )
            )
        raise RuntimeError(
            "Durable signal schema inspection supports only sqlite or postgres"
        )

    async def _ensure_source_sequence_backfill_marker_column(self) -> None:
        """Add durable backfill proof/work markers to an interim state table.

        The state table was introduced with the independent recovery watermark.
        An initializer interrupted after that release candidate may encounter
        the table without later marker columns, so ``CREATE TABLE IF NOT
        EXISTS`` alone is insufficient even within the same feature rollout.
        """

        boolean_columns = (
            self.SOURCE_SEQUENCE_BACKFILL_COLUMN,
            self.SOURCE_SEQUENCE_SCOPE_WORK_SEEDED_COLUMN,
            self.SOURCE_SEQUENCE_EVENT_WORK_SEEDED_COLUMN,
            self.SOURCE_SEQUENCE_SCOPE_VALIDATION_COLUMN,
        )
        nullable_text_columns = (
            self.SOURCE_SEQUENCE_SCOPE_VALIDATION_AGENT_COLUMN,
            self.SOURCE_SEQUENCE_SCOPE_VALIDATION_SOURCE_COLUMN,
        )
        if self.is_postgres:
            existing = {
                str(row[0])
                for row in await self._backend.fetch_all(
                """
                SELECT attname
                FROM pg_attribute
                WHERE attrelid = to_regclass(?)
                  AND NOT attisdropped
                """,
                    (self.SOURCE_SEQUENCE_STATE,),
                )
            }
            for column in boolean_columns:
                if column not in existing:
                    await self._backend.execute(
                        f"ALTER TABLE {self.SOURCE_SEQUENCE_STATE} "
                        f"ADD COLUMN IF NOT EXISTS {column} "
                        f"{self.boolean_type()} NOT NULL DEFAULT FALSE"
                    )
            for column in nullable_text_columns:
                if column not in existing:
                    await self._backend.execute(
                        f"ALTER TABLE {self.SOURCE_SEQUENCE_STATE} "
                        f"ADD COLUMN IF NOT EXISTS {column} TEXT"
                    )
            return
        if self.is_sqlite:
            columns = await self._backend.fetch_all(
                f"PRAGMA table_info({self.SOURCE_SEQUENCE_STATE})"
            )
            existing = {str(row[1]) for row in columns}
            for column in boolean_columns:
                if column not in existing:
                    await self._backend.execute(
                        f"ALTER TABLE {self.SOURCE_SEQUENCE_STATE} "
                        f"ADD COLUMN {column} "
                        f"{self.boolean_type()} NOT NULL DEFAULT FALSE"
                    )
            for column in nullable_text_columns:
                if column not in existing:
                    await self._backend.execute(
                        f"ALTER TABLE {self.SOURCE_SEQUENCE_STATE} "
                        f"ADD COLUMN {column} TEXT"
                    )
            return
        raise RuntimeError(
            "Durable signal source-sequence marker migration supports only "
            "sqlite or postgres"
        )

    async def _ensure_source_sequence_state(self) -> None:
        """Ensure the transactional PostgreSQL backfill marker singleton.

        ``counter_history_lost`` remains in the additive table for compatibility
        with an interrupted earlier candidate migration, but exact loss is now
        represented by immutable rows in ``SOURCE_SEQUENCE_SEEN``. A singleton
        cannot identify which scope is unsafe and would incorrectly poison a
        genuinely fresh scope.
        """

        row = await self._backend.fetch_one(
            f"SELECT counter_history_lost FROM {self.SOURCE_SEQUENCE_STATE} "
            "WHERE singleton = 1"
        )
        if row is None:
            await self._backend.execute(
                f"INSERT INTO {self.SOURCE_SEQUENCE_STATE} "
                "(singleton, counter_history_lost) VALUES (1, ?)",
                (self.to_bool_param(False),),
            )

    async def _source_sequence_backfill_completed(self) -> bool:
        """Read the transactional proof that counters and legacy rows converged."""

        value = await self._backend.fetch_val(
            f"SELECT {self.SOURCE_SEQUENCE_BACKFILL_COLUMN} "
            f"FROM {self.SOURCE_SEQUENCE_STATE} WHERE singleton = 1"
        )
        if value is None:
            raise RuntimeError(
                "Durable signal source-sequence backfill state is missing"
            )
        return bool(value)

    async def _set_source_sequence_backfill_completed(self, completed: bool) -> None:
        """Persist or invalidate the PostgreSQL backfill proof transactionally."""

        if not completed:
            # A trusted fence/recovery family was lost. Discard both narrow
            # queues and their seed proofs in the same transaction that
            # invalidates completion; a restart can only resume after a fresh
            # one-time ledger/counter pass.
            await self._backend.execute(
                f"DELETE FROM {self.SOURCE_SEQUENCE_SCOPE_WORK}"
            )
            await self._backend.execute(
                f"DELETE FROM {self.SOURCE_SEQUENCE_EVENT_WORK}"
            )
            updated = await self._backend.execute(
                f"UPDATE {self.SOURCE_SEQUENCE_STATE} "
                f"SET {self.SOURCE_SEQUENCE_BACKFILL_COLUMN} = ?, "
                f"{self.SOURCE_SEQUENCE_SCOPE_WORK_SEEDED_COLUMN} = ?, "
                f"{self.SOURCE_SEQUENCE_EVENT_WORK_SEEDED_COLUMN} = ?, "
                f"{self.SOURCE_SEQUENCE_SCOPE_VALIDATION_COLUMN} = ?, "
                f"{self.SOURCE_SEQUENCE_SCOPE_VALIDATION_AGENT_COLUMN} = NULL, "
                f"{self.SOURCE_SEQUENCE_SCOPE_VALIDATION_SOURCE_COLUMN} = NULL "
                "WHERE singleton = 1",
                (
                    self.to_bool_param(False),
                    self.to_bool_param(False),
                    self.to_bool_param(False),
                    self.to_bool_param(False),
                ),
            )
        else:
            updated = await self._backend.execute(
                f"UPDATE {self.SOURCE_SEQUENCE_STATE} "
                f"SET {self.SOURCE_SEQUENCE_BACKFILL_COLUMN} = ? "
                f"WHERE singleton = 1 "
                f"AND {self.SOURCE_SEQUENCE_SCOPE_VALIDATION_COLUMN} = ?",
                (
                    self.to_bool_param(True),
                    self.to_bool_param(True),
                ),
            )
        if updated != 1:
            raise RuntimeError(
                "Durable signal source-sequence backfill state disappeared"
            )

    async def _source_sequence_scope_validation_completed(self) -> bool:
        """Return whether every pre-completion primary scope was adopted."""

        value = await self._backend.fetch_val(
            f"SELECT {self.SOURCE_SEQUENCE_SCOPE_VALIDATION_COLUMN} "
            f"FROM {self.SOURCE_SEQUENCE_STATE} WHERE singleton = 1"
        )
        if value is None:
            raise RuntimeError(
                "Durable signal source-sequence scope validation state is missing"
            )
        return bool(value)

    async def _source_sequence_work_seeded(self, column: str) -> bool:
        """Return one allowlisted durable work-population proof."""

        if column not in {
            self.SOURCE_SEQUENCE_SCOPE_WORK_SEEDED_COLUMN,
            self.SOURCE_SEQUENCE_EVENT_WORK_SEEDED_COLUMN,
        }:
            raise ValueError("unknown source-sequence work marker")
        value = await self._backend.fetch_val(
            f"SELECT {column} FROM {self.SOURCE_SEQUENCE_STATE} "
            "WHERE singleton = 1"
        )
        if value is None:
            raise RuntimeError(
                "Durable signal source-sequence backfill state is missing"
            )
        return bool(value)

    async def _mark_source_sequence_work_seeded(self, column: str) -> None:
        """Commit one allowlisted work-population proof."""

        if column not in {
            self.SOURCE_SEQUENCE_SCOPE_WORK_SEEDED_COLUMN,
            self.SOURCE_SEQUENCE_EVENT_WORK_SEEDED_COLUMN,
        }:
            raise ValueError("unknown source-sequence work marker")
        updated = await self._backend.execute(
            f"UPDATE {self.SOURCE_SEQUENCE_STATE} SET {column} = ? "
            "WHERE singleton = 1",
            (self.to_bool_param(True),),
        )
        if updated != 1:
            raise RuntimeError(
                "Durable signal source-sequence backfill state disappeared"
            )

    async def _adopt_source_sequence_recovery(self) -> None:
        """Adopt SQLite exact ledgers in deterministic relation/row order.

        SQLite owns its reserved writer for this whole operation. PostgreSQL
        uses ``_adopt_postgres_source_sequence_recovery_batch`` so it never
        retains an unbounded family of scope-row locks.
        """

        lock_clause = " FOR UPDATE" if self.is_postgres else ""
        primary_rows = await self._backend.fetch_all(
            f"SELECT agent_id, source, current_sequence "
            f"FROM {self.SOURCE_SEQUENCES} "
            f"ORDER BY agent_id, source{lock_clause}"
        )
        for agent_id, source, current_sequence in primary_rows:
            sequence = int(current_sequence)
            if sequence < 0 or sequence > _MAX_SOURCE_SEQUENCE:
                raise RuntimeError(
                    "Durable signal source sequence is out of range during adoption"
                )
            await self._backend.execute(
                f"""
                INSERT OR IGNORE INTO {self.SOURCE_SEQUENCE_RECOVERY} (
                    agent_id, source, recovery_sequence
                ) VALUES (?, ?, ?)
                """,
                (agent_id, source, sequence),
            )
            await self._merge_source_sequence_high_water_locked(
                agent_id=str(agent_id), source=str(source), sequence=sequence
            )

        # Every primary row is already locked before this relation is touched.
        # Lock existing recovery-only scopes afterwards as well so seeding the
        # independent marker observes one deterministic, exact snapshot.
        recovery_rows = await self._backend.fetch_all(
            f"SELECT agent_id, source, recovery_sequence "
            f"FROM {self.SOURCE_SEQUENCE_RECOVERY} "
            f"ORDER BY agent_id, source{lock_clause}"
        )
        for agent_id, source, recovery_sequence in recovery_rows:
            sequence = int(recovery_sequence)
            if sequence < 0 or sequence > _MAX_SOURCE_SEQUENCE:
                raise RuntimeError(
                    "Durable signal recovery sequence is out of range during adoption"
                )
            await self._merge_source_sequence_high_water_locked(
                agent_id=str(agent_id), source=str(source), sequence=sequence
            )

        high_water_rows = await self._backend.fetch_all(
            f"SELECT agent_id, source, high_water_sequence "
            f"FROM {self.SOURCE_SEQUENCE_HIGH_WATER} "
            f"ORDER BY agent_id, source{lock_clause}"
        )
        seen_scopes = {
            (str(agent_id), str(source))
            for agent_id, source, sequence in (
                *primary_rows,
                *recovery_rows,
                *high_water_rows,
            )
            if int(sequence) > 0
        }
        for agent_id, source in sorted(seen_scopes):
            await self._backend.execute(
                f"INSERT OR IGNORE INTO {self.SOURCE_SEQUENCE_SEEN} "
                "(agent_id, source) VALUES (?, ?)",
                (agent_id, source),
            )

    async def _merge_source_sequence_high_water_locked(
        self, *, agent_id: str, source: str, sequence: int
    ) -> None:
        """Merge exact evidence into the retention-independent ledger."""

        if sequence < 0 or sequence > _MAX_SOURCE_SEQUENCE:
            raise ValueError("source sequence is out of range")
        maximum = "GREATEST" if self.is_postgres else "MAX"
        await self._backend.execute(
            f"""
            INSERT INTO {self.SOURCE_SEQUENCE_HIGH_WATER} (
                agent_id, source, high_water_sequence
            ) VALUES (?, ?, ?)
            ON CONFLICT (agent_id, source) DO UPDATE
            SET high_water_sequence = {maximum}(
                {self.SOURCE_SEQUENCE_HIGH_WATER}.high_water_sequence,
                excluded.high_water_sequence
            )
            """,
            (agent_id, source, sequence),
        )

    async def _restore_primary_source_sequences_from_recovery(self) -> None:
        """Seed a recreated primary from every independent exact ledger."""

        await self._backend.execute(
            f"""
            INSERT OR IGNORE INTO {self.SOURCE_SEQUENCES} (
                agent_id, source, current_sequence
            )
            SELECT agent_id, source, MAX(sequence)
            FROM (
                SELECT agent_id, source, recovery_sequence AS sequence
                FROM {self.SOURCE_SEQUENCE_RECOVERY}
                UNION ALL
                SELECT agent_id, source, high_water_sequence AS sequence
                FROM {self.SOURCE_SEQUENCE_HIGH_WATER}
            ) AS exact_sequences
            GROUP BY agent_id, source
            """
        )

    async def _ensure_caller_identity_column(self) -> None:
        """Apply the additive caller-identity migration under the schema lock."""

        if self.is_postgres:
            exists = await self._backend.fetch_val(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_attribute
                    WHERE attrelid = to_regclass(?)
                      AND attname = 'caller_identity'
                      AND NOT attisdropped
                )
                """,
                (self.EVENTS,),
            )
            if exists:
                return
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} ADD COLUMN IF NOT EXISTS caller_identity TEXT"
            )
            return
        columns = await self._backend.fetch_all(f"PRAGMA table_info({self.EVENTS})")
        if not any(row[1] == "caller_identity" for row in columns):
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} ADD COLUMN caller_identity TEXT"
            )

    async def _source_sequence_schema_state(self) -> _SourceSequenceSchemaState:
        """Read O(1) catalog evidence for source-sequence enforcement."""

        if self.is_postgres:
            row = await self._backend.fetch_one(
                """
                SELECT attribute.attnotnull,
                       constraint_row.oid IS NOT NULL,
                       COALESCE(constraint_row.convalidated, FALSE),
                       pg_get_expr(
                           constraint_row.conbin, constraint_row.conrelid, TRUE
                       )
                FROM pg_attribute AS attribute
                LEFT JOIN pg_constraint AS constraint_row
                  ON constraint_row.conrelid = attribute.attrelid
                 AND constraint_row.conname = ?
                 AND constraint_row.contype = 'c'
                WHERE attribute.attrelid = to_regclass(?)
                  AND attribute.attname = 'source_sequence'
                  AND NOT attribute.attisdropped
                """,
                (self.SOURCE_SEQUENCE_NOT_NULL_CHECK, self.EVENTS),
            )
            if row is None:
                return _SourceSequenceSchemaState(False, False)
            column_not_null = bool(row[0])
            fence_exists = bool(row[1])
            fence_validated = bool(row[2])
            definition_valid = fence_exists and (
                _canonical_postgres_check_expression(row[3])
                == _canonical_postgres_check_expression(
                    _SOURCE_SEQUENCE_CHECK_EXPRESSION
                )
            )
            return _SourceSequenceSchemaState(
                column_exists=True,
                enforced=(
                    column_not_null and definition_valid and fence_validated
                ),
                fence_exists=fence_exists,
                fence_validated=fence_validated,
                fence_definition_valid=definition_valid,
                column_not_null=column_not_null,
            )

        if self.is_sqlite:
            columns = await self._backend.fetch_all(
                f"PRAGMA table_info({self.EVENTS})"
            )
            column = next(
                (row for row in columns if row[1] == "source_sequence"), None
            )
            if column is None:
                return _SourceSequenceSchemaState(False, False)
            installed = await self._sqlite_source_sequence_guard_family()
            installed_by_casefold = {
                name.casefold(): (name, relation_name, sql)
                for name, (relation_name, sql) in installed.items()
            }
            desired = dict(self.SOURCE_SEQUENCE_GUARDS)
            desired_valid = all(
                name.casefold() in installed_by_casefold
                and installed_by_casefold[name.casefold()][1] == self.EVENTS
                and installed_by_casefold[name.casefold()][2] is not None
                and _canonical_schema_sql(
                    cast(str, installed_by_casefold[name.casefold()][2])
                )
                == _canonical_schema_sql(ddl)
                for name, ddl in desired.items()
            )
            guarded = desired_valid and set(installed_by_casefold) == {
                name.casefold() for name in desired
            }
            column_not_null = bool(column[3])
            # SQLite's NOT NULL declaration does not reject zero or negatives,
            # so only the exact, definition-addressed pair is completion
            # evidence. It is installed in the same transaction that validates
            # history. A fresh table pays that empty validation once; every
            # normal boot takes the catalog-only fast path. A superseded family
            # still requires cleanup even if the desired pair is present.
            return _SourceSequenceSchemaState(
                column_exists=True,
                enforced=desired_valid,
                fence_exists=guarded,
                fence_validated=desired_valid,
                fence_definition_valid=desired_valid,
                column_not_null=column_not_null,
            )

        raise RuntimeError(
            "Durable signal source-sequence schema inspection supports only "
            "sqlite or postgres"
        )

    async def _sqlite_source_sequence_guard_family(
        self,
    ) -> dict[str, tuple[str, Optional[str]]]:
        """Enumerate every trigger in the owned SQLite guard namespace."""

        return await self._sqlite_trigger_family(self.SOURCE_SEQUENCE_GUARD_PREFIX)

    async def _sqlite_trigger_family(
        self, prefix: str
    ) -> dict[str, tuple[str, Optional[str]]]:
        """Enumerate one definition-addressed SQLite trigger namespace."""

        rows = await self._backend.fetch_all(
            "SELECT name, tbl_name, sql FROM sqlite_master WHERE type = 'trigger'"
        )
        return {
            str(row[0]): (str(row[1]), None if row[2] is None else str(row[2]))
            for row in rows
            if str(row[0]).casefold().startswith(prefix.casefold())
        }

    async def _ensure_source_sequence_column(
        self, state: _SourceSequenceSchemaState
    ) -> None:
        """Add the nullable staging column used by the sequence backfill."""

        if state.column_exists:
            return
        if self.is_postgres:
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} "
                "ADD COLUMN IF NOT EXISTS source_sequence BIGINT"
            )
            return
        if self.is_sqlite:
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} ADD COLUMN source_sequence BIGINT"
            )
            return
        raise RuntimeError(
            "Durable signal source-sequence column migration supports only "
            "sqlite or postgres"
        )

    async def _install_postgres_source_sequence_fence(
        self, state: _SourceSequenceSchemaState
    ) -> None:
        """Install a short-lock fence that rejects later legacy writes."""

        if not self.is_postgres:
            raise RuntimeError("PostgreSQL source-sequence fence requires postgres")
        await self._ensure_source_sequence_column(state)
        if state.fence_exists and not state.fence_definition_valid:
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} "
                f"DROP CONSTRAINT {self.SOURCE_SEQUENCE_NOT_NULL_CHECK}"
            )
        if not state.fence_definition_valid:
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} "
                f"ADD CONSTRAINT {self.SOURCE_SEQUENCE_NOT_NULL_CHECK} "
                f"CHECK ({_SOURCE_SEQUENCE_CHECK_EXPRESSION}) NOT VALID"
            )

    async def _postgres_source_sequence_recovery_sync_valid(self) -> bool:
        """Validate the whole PostgreSQL mirror family from exact catalog shape.

        Names alone are only an address. A trusted member must be an enabled,
        unconditional, ordinary, nondeferred AFTER statement trigger with the
        exact transition relation and exact zero-argument PL/pgSQL function.
        Set equality rejects superseded family members left beside the current
        fingerprint.
        """

        trigger_rows, function_rows = (
            await self._postgres_source_sequence_recovery_catalog()
        )
        wanted_triggers = {
            definition.trigger_name: definition
            for definition in self.SOURCE_SEQUENCE_RECOVERY_DEFINITIONS
        }
        wanted_functions = {
            definition.function_name: definition
            for definition in self.SOURCE_SEQUENCE_RECOVERY_DEFINITIONS
        }
        if {str(row[0]) for row in trigger_rows} != set(wanted_triggers):
            return False
        if {str(row[0]) for row in function_rows} != set(wanted_functions):
            return False

        for row in trigger_rows:
            definition = wanted_triggers[str(row[0])]
            if not self._postgres_trigger_catalog_row_matches(
                row,
                trigger_type=definition.trigger_type,
                transition_table=definition.transition_table,
                function_name=definition.function_name,
                function_body=definition.function_body,
            ):
                return False

        for row in function_rows:
            definition = wanted_functions[str(row[0])]
            if not self._postgres_function_catalog_row_matches(
                row, function_body=definition.function_body
            ):
                return False
        return True

    @staticmethod
    def _postgres_trigger_catalog_row_matches(
        row: Any,
        *,
        trigger_type: int,
        transition_table: Optional[str],
        function_name: str,
        function_body: str,
    ) -> bool:
        """Validate the exact trusted shape shared by owned trigger families."""

        return bool(
            int(row[1]) == trigger_type
            and _postgres_catalog_char(row[2]) == "O"
            and not str(row[3]).strip()
            and bool(row[4])  # tgqual IS NULL: no WHEN qualification
            and bool(row[5])  # tgconstraint = 0
            and not bool(row[6])
            and not bool(row[7])
            and row[8] == transition_table
            and row[9] is None
            and str(row[10]) == function_name
            and bool(row[11])  # function belongs to current_schema()
            and _canonical_schema_sql(str(row[12]))
            == _canonical_schema_sql(function_body)
            and str(row[13]) == "plpgsql"
            and int(row[14]) == 0
            and bool(row[15])
            and _postgres_catalog_char(row[16]) == "f"
            and not bool(row[17])
            and not bool(row[18])
            and not bool(row[19])
            and _postgres_catalog_char(row[20]) == "v"
            and _postgres_catalog_char(row[21]) == "u"
            and bool(row[22])
        )

    @staticmethod
    def _postgres_function_catalog_row_matches(
        row: Any, *, function_body: str
    ) -> bool:
        """Validate one zero-argument, invoker-rights PL/pgSQL trigger function."""

        return bool(
            str(row[1]) == ""
            and _canonical_schema_sql(str(row[2]))
            == _canonical_schema_sql(function_body)
            and str(row[3]) == "plpgsql"
            and int(row[4]) == 0
            and bool(row[5])
            and _postgres_catalog_char(row[6]) == "f"
            and not bool(row[7])
            and not bool(row[8])
            and not bool(row[9])
            and _postgres_catalog_char(row[10]) == "v"
            and _postgres_catalog_char(row[11]) == "u"
            and bool(row[12])
        )

    async def _postgres_source_sequence_recovery_catalog(
        self,
    ) -> tuple[list[Any], list[Any]]:
        """Enumerate every current-schema member of the owned PG family."""

        return await self._postgres_trigger_function_catalog(
            relation_name=self.EVENTS,
            trigger_prefix=self.SOURCE_SEQUENCE_RECOVERY_TRIGGER_PREFIX,
            function_prefix=self.SOURCE_SEQUENCE_RECOVERY_FUNCTION_PREFIX,
        )

    async def _postgres_trigger_function_catalog(
        self,
        *,
        relation_name: str,
        trigger_prefix: str,
        function_prefix: str,
    ) -> tuple[list[Any], list[Any]]:
        """Enumerate one owned current-schema trigger/function namespace."""

        trigger_rows = await self._backend.fetch_all(
            """
            SELECT trigger_row.tgname,
                   trigger_row.tgtype,
                   trigger_row.tgenabled,
                   trigger_row.tgattr::text,
                   trigger_row.tgqual IS NULL,
                   trigger_row.tgconstraint = 0,
                   trigger_row.tgdeferrable,
                   trigger_row.tginitdeferred,
                   trigger_row.tgnewtable,
                   trigger_row.tgoldtable,
                   function_row.proname,
                   function_namespace.nspname = current_schema(),
                   function_row.prosrc,
                   language_row.lanname,
                   function_row.pronargs,
                   function_row.prorettype = 'pg_catalog.trigger'::regtype,
                   function_row.prokind,
                   function_row.prosecdef,
                   function_row.proleakproof,
                   function_row.proisstrict,
                   function_row.provolatile,
                   function_row.proparallel,
                   function_row.proconfig IS NULL
            FROM pg_trigger AS trigger_row
            JOIN pg_class AS relation
              ON relation.oid = trigger_row.tgrelid
            JOIN pg_namespace AS relation_namespace
              ON relation_namespace.oid = relation.relnamespace
            JOIN pg_proc AS function_row
              ON function_row.oid = trigger_row.tgfoid
            JOIN pg_namespace AS function_namespace
              ON function_namespace.oid = function_row.pronamespace
            JOIN pg_language AS language_row
              ON language_row.oid = function_row.prolang
            WHERE relation_namespace.nspname = current_schema()
              AND relation.relname = ?
              AND NOT trigger_row.tgisinternal
            """,
            (relation_name,),
        )
        function_rows = await self._backend.fetch_all(
            """
            SELECT function_row.proname,
                   pg_get_function_identity_arguments(function_row.oid),
                   function_row.prosrc,
                   language_row.lanname,
                   function_row.pronargs,
                   function_row.prorettype = 'pg_catalog.trigger'::regtype,
                   function_row.prokind,
                   function_row.prosecdef,
                   function_row.proleakproof,
                   function_row.proisstrict,
                   function_row.provolatile,
                   function_row.proparallel,
                   function_row.proconfig IS NULL
            FROM pg_proc AS function_row
            JOIN pg_namespace AS function_namespace
              ON function_namespace.oid = function_row.pronamespace
            JOIN pg_language AS language_row
              ON language_row.oid = function_row.prolang
            WHERE function_namespace.nspname = current_schema()
            """
        )
        return (
            [
                row
                for row in trigger_rows
                if str(row[0]).startswith(trigger_prefix)
            ],
            [
                row
                for row in function_rows
                if str(row[0]).startswith(function_prefix)
            ],
        )

    async def _ensure_postgres_source_sequence_recovery_sync(self) -> None:
        """Install/repair the mixed-version recovery mirror under schema lock."""

        if not self.is_postgres:
            raise RuntimeError(
                "PostgreSQL source-sequence recovery sync requires postgres"
            )
        if await self._postgres_source_sequence_recovery_sync_valid():
            return

        trigger_rows, function_rows = (
            await self._postgres_source_sequence_recovery_catalog()
        )
        # Bootstrap holds the cross-process schema transaction here. PostgreSQL
        # also locks the target relation for trigger DDL, so dropping every old
        # family member and installing the desired fingerprint is one atomic,
        # externally invisible transition. Functions are retired only after
        # every trigger dependency is gone, and never with CASCADE.
        await self._replace_postgres_trigger_function_family(
            relation_name=self.EVENTS,
            trigger_rows=trigger_rows,
            function_rows=function_rows,
            definitions=self.SOURCE_SEQUENCE_RECOVERY_DEFINITIONS,
        )
        if not await self._postgres_source_sequence_recovery_sync_valid():
            raise RuntimeError(
                "PostgreSQL durable signal recovery trigger family failed validation"
            )

    async def _postgres_source_sequence_counter_fence_valid(self) -> bool:
        """Return whether primary-only legacy writes are durably fenced."""

        trigger_rows, function_rows = await self._postgres_trigger_function_catalog(
            relation_name=self.SOURCE_SEQUENCES,
            trigger_prefix=self.SOURCE_SEQUENCE_COUNTER_TRIGGER_PREFIX,
            function_prefix=self.SOURCE_SEQUENCE_COUNTER_FUNCTION_PREFIX,
        )
        wanted_triggers = {
            definition.trigger_name: definition
            for definition in self.SOURCE_SEQUENCE_COUNTER_FENCE_DEFINITIONS
        }
        wanted_functions = {
            definition.function_name: definition
            for definition in self.SOURCE_SEQUENCE_COUNTER_FENCE_DEFINITIONS
        }
        if {str(row[0]) for row in trigger_rows} != set(wanted_triggers):
            return False
        if {str(row[0]) for row in function_rows} != set(wanted_functions):
            return False
        for row in trigger_rows:
            definition = wanted_triggers[str(row[0])]
            if not self._postgres_trigger_catalog_row_matches(
                row,
                trigger_type=definition.trigger_type,
                transition_table=None,
                function_name=definition.function_name,
                function_body=definition.function_body,
            ):
                return False
        for row in function_rows:
            definition = wanted_functions[str(row[0])]
            if not self._postgres_function_catalog_row_matches(
                row, function_body=definition.function_body
            ):
                return False
        return True

    async def _ensure_postgres_source_sequence_counter_fence(self) -> None:
        """Install the database fence around the legacy primary counter API."""

        if not self.is_postgres:
            raise RuntimeError(
                "PostgreSQL source-sequence counter fence requires postgres"
            )
        if await self._postgres_source_sequence_counter_fence_valid():
            return
        trigger_rows, function_rows = await self._postgres_trigger_function_catalog(
            relation_name=self.SOURCE_SEQUENCES,
            trigger_prefix=self.SOURCE_SEQUENCE_COUNTER_TRIGGER_PREFIX,
            function_prefix=self.SOURCE_SEQUENCE_COUNTER_FUNCTION_PREFIX,
        )
        await self._replace_postgres_trigger_function_family(
            relation_name=self.SOURCE_SEQUENCES,
            trigger_rows=trigger_rows,
            function_rows=function_rows,
            definitions=self.SOURCE_SEQUENCE_COUNTER_FENCE_DEFINITIONS,
        )
        if not await self._postgres_source_sequence_counter_fence_valid():
            raise RuntimeError(
                "PostgreSQL durable signal counter fence failed validation"
            )

    async def _replace_postgres_trigger_function_family(
        self,
        *,
        relation_name: str,
        trigger_rows: list[Any],
        function_rows: list[Any],
        definitions: Iterable[Any],
    ) -> None:
        """Atomically replace one owned PostgreSQL trigger/function family."""

        definition_family = tuple(definitions)
        for row in sorted(trigger_rows, key=lambda item: str(item[0])):
            await self._backend.execute(
                f"DROP TRIGGER IF EXISTS {_quoted_identifier(str(row[0]))} "
                f"ON {_quoted_identifier(relation_name)}"
            )
        for row in sorted(
            function_rows, key=lambda item: (str(item[0]), str(item[1]))
        ):
            await self._backend.execute(
                f"DROP FUNCTION IF EXISTS {_quoted_identifier(str(row[0]))}"
                f"({str(row[1])})"
            )
        for definition in definition_family:
            await self._backend.execute(definition.function_ddl)
        for definition in definition_family:
            await self._backend.execute(definition.trigger_ddl)

    async def _backfill_source_sequences(self) -> None:
        """Assign stable SQLite sequences to rows created before this contract.

        SQLite owns its reserved writer for the whole operation. PostgreSQL's
        transition-table mirror requires the separately committed bounded
        batch path and is rejected here.

        Legacy rows are ordered by immutable event ID, not by process-clock
        timestamps.  Since no source boundary existed before this migration,
        every migrated row is necessarily at or below the first capturable
        boundary regardless of the order chosen within that history.
        """

        if self.is_postgres:
            raise RuntimeError(
                "PostgreSQL source-sequence history must use bounded batches"
            )

        scopes = await self._backend.fetch_all(
            f"SELECT DISTINCT agent_id, source FROM {self.EVENTS} "
            "ORDER BY agent_id, source"
        )
        for agent_id, source in scopes:
            current = await self._source_sequence_locked(
                agent_id=agent_id,
                source=source,
                allow_retained_reconstruction=True,
            )
            aggregate = await self._backend.fetch_one(
                f"SELECT MIN(source_sequence), MAX(source_sequence), "
                "COALESCE(SUM(CASE WHEN source_sequence IS NULL THEN 1 ELSE 0 END), 0) "
                f"FROM {self.EVENTS} WHERE agent_id = ? AND source = ?",
                (agent_id, source),
            )
            if aggregate is None:
                raise RuntimeError("Durable signal source scope disappeared")
            minimum, maximum, legacy_count = aggregate
            if minimum is not None and int(minimum) < 1:
                raise RuntimeError(
                    "Durable signal event has an invalid source sequence"
                )
            if maximum is not None:
                maximum = int(maximum)
                if maximum > _MAX_SOURCE_SEQUENCE:
                    raise RuntimeError(
                        "Durable signal event has an invalid source sequence"
                    )
                current = max(current, maximum)
            legacy_count = int(legacy_count or 0)
            if legacy_count > _MAX_SOURCE_SEQUENCE - current:
                raise OverflowError("Durable signal source sequence exhausted")

            if legacy_count:
                # One statement per exact scope, rather than one round trip per
                # legacy event. PostgreSQL must join the ranked relation once:
                # a scalar correlated lookup into the CTE for every target row
                # becomes quadratic after materialization. SQLite lacks
                # ``UPDATE ... FROM`` on older supported releases, so it keeps
                # the compatible correlated spelling for its serialized local
                # migration. Immutable event IDs give both engines the same
                # deterministic ordering.
                await self._backend.execute(
                    f"""
                    WITH ranked_source_events AS (
                        SELECT event_id,
                               ROW_NUMBER() OVER (ORDER BY event_id) AS sequence_offset
                        FROM {self.EVENTS}
                        WHERE agent_id = ? AND source = ?
                          AND source_sequence IS NULL
                    )
                    UPDATE {self.EVENTS}
                    SET source_sequence = ? + (
                        SELECT ranked.sequence_offset
                        FROM ranked_source_events AS ranked
                        WHERE ranked.event_id = {self.EVENTS}.event_id
                    )
                    WHERE event_id IN (
                        SELECT event_id FROM ranked_source_events
                    )
                      AND source_sequence IS NULL
                    """,
                    (agent_id, source, current),
                )
                # Retention may delete legacy rows while PostgreSQL migrates.
                # Advancing across the pre-update count can leave harmless
                # gaps, but can never reuse or move a committed sequence
                # backward. The source-state row lock excludes a current
                # ingress writer in this exact scope.
                current += legacy_count
            await self._set_source_sequence_locked(
                agent_id=agent_id, source=source, sequence=current
            )

        incomplete = await self._backend.fetch_val(
            f"SELECT 1 FROM {self.EVENTS} "
            "WHERE source_sequence IS NULL OR source_sequence < 1 LIMIT 1"
        )
        if incomplete is not None:
            raise RuntimeError(
                "Durable signal source-sequence backfill is incomplete"
            )

    async def _complete_postgres_source_sequence_backfill(self) -> None:
        """Converge exact counters and legacy history with bounded total work.

        Counter scopes and NULL legacy event identities are copied into narrow,
        durable primary-key work tables. Scope work is followed by a durable
        keyset pass over the exact primary ledger, so damaged work queues cannot
        become completion evidence. Every later transaction validates one
        scope or removes at most ``SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE`` event
        rows. A crash therefore resumes from committed queue/cursor state, and
        the migration never re-probes or re-sorts the wide event ledger per
        batch.
        """

        if not self.is_postgres:
            raise RuntimeError("PostgreSQL source-sequence backfill requires postgres")
        while True:
            async with self._schema_bootstrap_transaction():
                if await self._source_sequence_backfill_completed():
                    return
                source_state = await self._source_sequence_schema_state()
                if (
                    not source_state.column_exists
                    or not source_state.fence_exists
                    or not source_state.fence_definition_valid
                ):
                    raise RuntimeError(
                        "Durable signal source-sequence write fence disappeared "
                        "during migration"
                    )
                if not await self._source_sequence_work_seeded(
                    self.SOURCE_SEQUENCE_SCOPE_WORK_SEEDED_COLUMN
                ):
                    await self._seed_postgres_source_sequence_scope_work()
                    continue
                if await self._adopt_postgres_source_sequence_recovery_batch():
                    continue
                if not await self._source_sequence_scope_validation_completed():
                    await self._validate_postgres_source_sequence_scope_batch()
                    continue
                if not await self._source_sequence_work_seeded(
                    self.SOURCE_SEQUENCE_EVENT_WORK_SEEDED_COLUMN
                ):
                    await self._seed_postgres_source_sequence_event_work()
                    continue
                if await self._backfill_postgres_source_sequence_batch():
                    continue
                unsequenced = await self._backend.fetch_val(
                    f"SELECT 1 FROM {self.EVENTS} "
                    "WHERE source_sequence IS NULL LIMIT 1"
                )
                if unsequenced is not None:
                    # The seed bit is not sufficient evidence: an operator may
                    # recreate the narrow work table or lose only some queued
                    # rows while leaving the state singleton intact. Re-seed
                    # every still-NULL immutable identity transactionally. The
                    # write fence prevents new NULL rows, and subsequent work
                    # remains bounded by the queue primary key and batch size.
                    seeded = await self._seed_postgres_source_sequence_event_work()
                    if seeded < 1:
                        still_unsequenced = await self._backend.fetch_val(
                            f"SELECT 1 FROM {self.EVENTS} "
                            "WHERE source_sequence IS NULL LIMIT 1"
                        )
                        if still_unsequenced is not None:
                            raise RuntimeError(
                                "Durable signal source-sequence event work could "
                                "not be reconciled with NULL history"
                            )
                    continue
                invalid = await self._backend.fetch_val(
                    f"SELECT 1 FROM {self.EVENTS} "
                    "WHERE source_sequence < 1 LIMIT 1"
                )
                if invalid is not None:
                    raise RuntimeError(
                        "Durable signal event has an invalid source sequence"
                    )
                await self._set_source_sequence_backfill_completed(True)
                return

    async def _seed_postgres_source_sequence_scope_work(self) -> None:
        """Populate every pre-fence exact scope once without locking its rows."""

        for relation in (
            self.SOURCE_SEQUENCES,
            self.SOURCE_SEQUENCE_RECOVERY,
            self.SOURCE_SEQUENCE_HIGH_WATER,
            self.SOURCE_SEQUENCE_SEEN,
        ):
            await self._backend.execute(
                f"""
                INSERT INTO {self.SOURCE_SEQUENCE_SCOPE_WORK} (agent_id, source)
                SELECT agent_id, source FROM {relation}
                ON CONFLICT (agent_id, source) DO NOTHING
                """
            )
        await self._mark_source_sequence_work_seeded(
            self.SOURCE_SEQUENCE_SCOPE_WORK_SEEDED_COLUMN
        )

    async def _seed_postgres_source_sequence_event_work(self) -> int:
        """Snapshot pre-fence NULL event identities in one narrow ledger pass."""

        seeded = await self._backend.execute(
            f"""
            INSERT INTO {self.SOURCE_SEQUENCE_EVENT_WORK} (
                agent_id, source, event_id
            )
            SELECT agent_id, source, event_id
            FROM {self.EVENTS}
            WHERE source_sequence IS NULL
            ON CONFLICT (agent_id, source, event_id) DO NOTHING
            """
        )
        await self._mark_source_sequence_work_seeded(
            self.SOURCE_SEQUENCE_EVENT_WORK_SEEDED_COLUMN
        )
        return seeded

    async def _adopt_postgres_source_sequence_recovery_batch(self) -> bool:
        """Converge at most one primary/recovery/seen scope in lock order."""

        row = await self._backend.fetch_one(
            f"SELECT agent_id, source FROM {self.SOURCE_SEQUENCE_SCOPE_WORK} "
            "ORDER BY agent_id, source LIMIT 1"
        )
        if row is None:
            return False
        agent_id, source = str(row[0]), str(row[1])
        await self._lock_scope_handoff(agent_id=agent_id, source=source)
        sequence = await self._source_sequence_locked(
            agent_id=agent_id,
            source=source,
            allow_retained_reconstruction=True,
        )
        await self._set_source_sequence_locked(
            agent_id=agent_id, source=source, sequence=sequence
        )
        removed = await self._backend.execute(
            f"DELETE FROM {self.SOURCE_SEQUENCE_SCOPE_WORK} "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        if removed != 1:
            raise RuntimeError(
                "PostgreSQL source-sequence scope work disappeared"
            )
        return True

    async def _validate_postgres_source_sequence_scope_batch(self) -> bool:
        """Adopt one indexed primary scope and durably advance its cursor.

        ``scope_work_seeded`` proves only that a scope was once enqueued. A
        missing row or a recreated work table can therefore make an empty
        queue lie. This cursor independently visits every primary-key scope
        before completion. Each scope and cursor advancement share one
        transaction, so restart repeats at most one rolled-back scope.

        The keyset lookup is linear across the primary index rather than a
        repeated prefix scan. Deliberately process one scope per transaction:
        the scope handoff is always acquired before that scope's counter rows,
        without retaining an earlier scope's counter locks while waiting for a
        later handoff. A mixed-version writer that inserts, moves, or advances
        a key behind the cursor is already safe: the installed primary-counter
        fence clamps and mirrors that write into every exact ledger in its own
        transaction.
        """

        if not self.is_postgres:
            raise RuntimeError("PostgreSQL scope validation requires postgres")
        state = await self._backend.fetch_one(
            f"SELECT {self.SOURCE_SEQUENCE_SCOPE_VALIDATION_COLUMN}, "
            f"{self.SOURCE_SEQUENCE_SCOPE_VALIDATION_AGENT_COLUMN}, "
            f"{self.SOURCE_SEQUENCE_SCOPE_VALIDATION_SOURCE_COLUMN} "
            f"FROM {self.SOURCE_SEQUENCE_STATE} WHERE singleton = 1"
        )
        if state is None:
            raise RuntimeError(
                "Durable signal source-sequence scope validation state is missing"
            )
        completed, after_agent_id, after_source = state
        if bool(completed):
            return False
        if (after_agent_id is None) != (after_source is None):
            raise RuntimeError(
                "Durable signal source-sequence scope validation cursor is invalid"
            )

        if after_agent_id is None:
            row = await self._backend.fetch_one(
                f"SELECT agent_id, source FROM {self.SOURCE_SEQUENCES} "
                "ORDER BY agent_id, source LIMIT 1"
            )
        else:
            row = await self._backend.fetch_one(
                f"SELECT agent_id, source FROM {self.SOURCE_SEQUENCES} "
                "WHERE (agent_id, source) > (?, ?) "
                "ORDER BY agent_id, source LIMIT 1",
                (str(after_agent_id), str(after_source)),
            )
        if row is None:
            updated = await self._backend.execute(
                f"UPDATE {self.SOURCE_SEQUENCE_STATE} "
                f"SET {self.SOURCE_SEQUENCE_SCOPE_VALIDATION_COLUMN} = ? "
                "WHERE singleton = 1",
                (self.to_bool_param(True),),
            )
            if updated != 1:
                raise RuntimeError(
                    "Durable signal source-sequence scope validation state disappeared"
                )
            return False

        agent_id, source = str(row[0]), str(row[1])
        await self._lock_scope_handoff(agent_id=agent_id, source=source)
        sequence = await self._source_sequence_locked(
            agent_id=agent_id,
            source=source,
            allow_retained_reconstruction=True,
        )
        await self._set_source_sequence_locked(
            agent_id=agent_id, source=source, sequence=sequence
        )
        updated = await self._backend.execute(
            f"UPDATE {self.SOURCE_SEQUENCE_STATE} "
            f"SET {self.SOURCE_SEQUENCE_SCOPE_VALIDATION_AGENT_COLUMN} = ?, "
            f"{self.SOURCE_SEQUENCE_SCOPE_VALIDATION_SOURCE_COLUMN} = ? "
            "WHERE singleton = 1",
            (agent_id, source),
        )
        if updated != 1:
            raise RuntimeError(
                "Durable signal source-sequence scope validation state disappeared"
            )
        return True

    async def _backfill_postgres_source_sequence_batch(self) -> bool:
        """Assign one bounded event batch and advance its exact counter atomically."""

        scope = await self._backend.fetch_one(
            f"SELECT agent_id, source FROM {self.SOURCE_SEQUENCE_EVENT_WORK} "
            "ORDER BY agent_id, source, event_id LIMIT 1"
        )
        if scope is None:
            return False
        agent_id, source = str(scope[0]), str(scope[1])
        await self._lock_scope_handoff(agent_id=agent_id, source=source)
        current = await self._source_sequence_locked(
            agent_id=agent_id,
            source=source,
            allow_retained_reconstruction=True,
        )
        if current >= _MAX_SOURCE_SEQUENCE:
            raise OverflowError("Durable signal source sequence exhausted")
        batch_size = min(
            self.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE,
            _MAX_SOURCE_SEQUENCE - current,
        )
        result = await self._backend.fetch_one(
            self._postgres_source_sequence_backfill_update_sql(),
            (
                agent_id,
                source,
                batch_size,
                agent_id,
                source,
                current,
                agent_id,
                source,
                agent_id,
                source,
            ),
        )
        if result is None:
            raise RuntimeError("PostgreSQL source-sequence batch returned no result")
        updated_count, maximum, removed_count = (
            int(result[0]),
            result[1],
            int(result[2]),
        )
        if not 1 <= removed_count <= self.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE:
            raise RuntimeError(
                "PostgreSQL source-sequence event work batch was invalid"
            )
        if updated_count == 0:
            # Retention or a mixed-version repair can make queued identities
            # stale. The same statement removes that bounded stale batch.
            return True
        maximum = int(maximum)
        if not current < maximum <= _MAX_SOURCE_SEQUENCE:
            raise RuntimeError(
                "PostgreSQL source-sequence batch produced an invalid maximum"
            )
        await self._set_source_sequence_locked(
            agent_id=agent_id, source=source, sequence=maximum
        )
        return True

    def _postgres_source_sequence_backfill_update_sql(self) -> str:
        """Return PostgreSQL's bounded, joined legacy assignment."""

        if not self.is_postgres:
            raise RuntimeError("PostgreSQL source-sequence backfill requires postgres")
        return f"""
            WITH batch_event_ids AS MATERIALIZED (
                SELECT event_id
                FROM {self.SOURCE_SEQUENCE_EVENT_WORK}
                WHERE agent_id = ? AND source = ?
                ORDER BY event_id
                LIMIT ?
                FOR UPDATE
            ), ranked_source_events AS MATERIALIZED (
                SELECT queued.event_id,
                       ROW_NUMBER() OVER (
                           ORDER BY queued.event_id
                       ) AS sequence_offset
                FROM batch_event_ids AS queued
                JOIN {self.EVENTS} AS event
                  ON event.event_id = queued.event_id
                 AND event.agent_id = ? AND event.source = ?
                WHERE event.source_sequence IS NULL
            ), updated_events AS (
                UPDATE {self.EVENTS} AS target
                SET source_sequence = ? + ranked.sequence_offset
                FROM ranked_source_events AS ranked
                WHERE target.event_id = ranked.event_id
                  AND target.agent_id = ? AND target.source = ?
                  AND target.source_sequence IS NULL
                RETURNING target.source_sequence
            ), removed_work AS (
                DELETE FROM {self.SOURCE_SEQUENCE_EVENT_WORK} AS work
                USING batch_event_ids AS queued
                WHERE work.agent_id = ? AND work.source = ?
                  AND work.event_id = queued.event_id
                RETURNING work.event_id
            )
            SELECT (SELECT COUNT(*) FROM updated_events),
                   (SELECT MAX(source_sequence) FROM updated_events),
                   (SELECT COUNT(*) FROM removed_work)
        """

    async def _install_sqlite_source_sequence_guards(self) -> None:
        """Atomically converge SQLite's guard family on the desired shape."""

        if not self.is_sqlite:
            raise RuntimeError("SQLite source-sequence guards require sqlite")
        await self._install_sqlite_trigger_family(
            relation_name=self.EVENTS,
            prefix=self.SOURCE_SEQUENCE_GUARD_PREFIX,
            definitions=self.SOURCE_SEQUENCE_GUARDS,
            failure_label="source-sequence guards",
        )

    async def _install_sqlite_source_sequence_counter_fence(self) -> None:
        """Install the primary-counter legacy fence in the reserved transaction."""

        if not self.is_sqlite:
            raise RuntimeError("SQLite source-sequence counter fence requires sqlite")
        await self._install_sqlite_trigger_family(
            relation_name=self.SOURCE_SEQUENCES,
            prefix=self.SOURCE_SEQUENCE_COUNTER_FENCE_PREFIX,
            definitions=self.SOURCE_SEQUENCE_COUNTER_FENCES,
            failure_label="source-sequence counter fence",
        )

    async def _install_sqlite_trigger_family(
        self,
        *,
        relation_name: str,
        prefix: str,
        definitions: Iterable[tuple[str, str]],
        failure_label: str,
    ) -> None:
        """Atomically converge one exact SQLite trigger family."""

        desired = dict(definitions)
        installed = await self._sqlite_trigger_family(prefix)
        installed_by_casefold = {
            name.casefold(): name for name in installed
        }

        # Create the desired family before retiring superseded names. If a
        # desired name itself has malformed SQL, it must be dropped first; the
        # caller's IMMEDIATE transaction already excludes every SQLite writer,
        # so no unguarded shape can become externally visible.
        for name, ddl in desired.items():
            actual_name = installed_by_casefold.get(name.casefold())
            current = installed.get(actual_name) if actual_name is not None else None
            valid = (
                current is not None
                and current[0] == relation_name
                and current[1] is not None
                and _canonical_schema_sql(current[1])
                == _canonical_schema_sql(ddl)
            )
            if valid:
                continue
            if current is not None:
                await self._backend.execute(
                    f"DROP TRIGGER {_quoted_identifier(cast(str, actual_name))}"
                )
            await self._backend.execute(ddl)

        desired_casefold = {name.casefold() for name in desired}
        for name in sorted(
            installed_name
            for installed_name in installed
            if installed_name.casefold() not in desired_casefold
        ):
            await self._backend.execute(
                f"DROP TRIGGER {_quoted_identifier(name)}"
            )

        final = await self._sqlite_trigger_family(prefix)
        final_by_casefold = {name.casefold(): name for name in final}
        if set(final_by_casefold) != desired_casefold or any(
            final[final_by_casefold[name.casefold()]][0] != relation_name
            or final[final_by_casefold[name.casefold()]][1] is None
            or _canonical_schema_sql(
                cast(str, final[final_by_casefold[name.casefold()]][1])
            )
            != _canonical_schema_sql(ddl)
            for name, ddl in desired.items()
        ):
            raise RuntimeError(
                f"SQLite durable signal {failure_label} failed validation"
            )

    async def _validate_postgres_source_sequence_fence(
        self, state: _SourceSequenceSchemaState
    ) -> None:
        """Validate the migration fence without requesting ACCESS EXCLUSIVE.

        This is the mixed-version rollout fence.  A pre-boundary Core replica
        omits source_sequence from its insert; after bootstrap commits that
        write must fail rather than enter history with ambiguous eligibility.
        ``VALIDATE CONSTRAINT`` scans under SHARE UPDATE EXCLUSIVE and its proof
        later makes ``SET NOT NULL`` metadata-only.
        """

        if not self.is_postgres:
            raise RuntimeError("PostgreSQL source-sequence validation requires postgres")
        if not state.fence_validated:
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} "
                f"VALIDATE CONSTRAINT {self.SOURCE_SEQUENCE_NOT_NULL_CHECK}"
            )

    async def _enforce_postgres_source_sequence_required(
        self, state: _SourceSequenceSchemaState
    ) -> None:
        """Make PostgreSQL NOT NULL metadata in its own final transaction."""

        if not self.is_postgres:
            raise RuntimeError("PostgreSQL source-sequence enforcement requires postgres")
        if (
            not state.fence_exists
            or not state.fence_definition_valid
            or not state.fence_validated
        ):
            raise RuntimeError(
                "PostgreSQL source-sequence NOT NULL requires a validated write fence"
            )
        if not state.column_not_null:
            await self._backend.execute(
                f"ALTER TABLE {self.EVENTS} "
                "ALTER COLUMN source_sequence SET NOT NULL"
            )

    async def _ensure_source_sequence_index(self) -> None:
        """Atomically converge SQLite's per-scope uniqueness proof.

        ``CREATE UNIQUE INDEX IF NOT EXISTS`` validates only that the name is
        occupied.  It would therefore trust a non-unique, partial, expression,
        or wrong-key index left under our owned name.  The caller holds the
        bootstrap ``BEGIN IMMEDIATE`` transaction across this catalog probe
        and any drop/recreate, excluding another writer from observing or
        racing the replacement.
        """

        if not self.is_sqlite:
            raise RuntimeError(
                "PostgreSQL source-sequence index must be built concurrently"
            )

        catalog = await self._sqlite_source_sequence_index_catalog()
        if self._sqlite_source_sequence_index_catalog_valid(catalog):
            return
        if catalog is not None:
            if catalog.object_type != "index":
                raise RuntimeError(
                    "SQLite durable signal source-sequence index name is "
                    f"occupied by {catalog.object_type!r}"
                )
            await self._backend.execute(
                f"DROP INDEX {_quoted_identifier(catalog.name)}"
            )

        await self._backend.execute(
            "CREATE UNIQUE INDEX "
            f"{_quoted_identifier(self.SOURCE_SEQUENCE_SCOPE_INDEX)} "
            f"ON {_quoted_identifier(self.EVENTS)} "
            "(agent_id, source, source_sequence)"
        )
        final = await self._sqlite_source_sequence_index_catalog()
        if not self._sqlite_source_sequence_index_catalog_valid(final):
            raise RuntimeError(
                "SQLite durable signal source-sequence index failed "
                "exact catalog validation"
            )

    async def _sqlite_source_sequence_index_catalog(
        self,
    ) -> Optional[_SQLiteSourceSequenceIndexCatalog]:
        """Return exact SQLite catalog evidence for the owned index name."""

        if not self.is_sqlite:
            raise RuntimeError("SQLite source-sequence index requires sqlite")

        owner = await self._backend.fetch_one(
            "SELECT type, name, tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name = ? COLLATE NOCASE",
            (self.SOURCE_SEQUENCE_SCOPE_INDEX,),
        )
        if owner is None:
            # SQLite permits a trigger and an index to share a name, so a
            # trigger is neither our object nor a collision. Tables/views do
            # share the index namespace and deserve an explicit diagnostic
            # rather than an opaque CREATE failure.
            owner = await self._backend.fetch_one(
                "SELECT type, name, tbl_name FROM sqlite_master "
                "WHERE type IN ('table', 'view') "
                "AND name = ? COLLATE NOCASE",
                (self.SOURCE_SEQUENCE_SCOPE_INDEX,),
            )
            if owner is None:
                return None

        object_type, actual_name, relation_name = (
            str(owner[0]),
            str(owner[1]),
            str(owner[2]),
        )
        if object_type != "index":
            return _SQLiteSourceSequenceIndexCatalog(
                object_type=object_type,
                name=actual_name,
                relation_name=relation_name,
            )

        index_list_rows = await self._backend.fetch_all(
            f"PRAGMA index_list({_quoted_identifier(relation_name)})"
        )
        index_list_row = next(
            (
                row
                for row in index_list_rows
                if len(row) >= 2
                and str(row[1]).casefold() == actual_name.casefold()
            ),
            None,
        )
        index_xinfo_rows = tuple(
            await self._backend.fetch_all(
                f"PRAGMA index_xinfo({_quoted_identifier(actual_name)})"
            )
        )
        return _SQLiteSourceSequenceIndexCatalog(
            object_type=object_type,
            name=actual_name,
            relation_name=relation_name,
            index_list_row=index_list_row,
            index_xinfo_rows=index_xinfo_rows,
        )

    @classmethod
    def _sqlite_source_sequence_index_catalog_valid(
        cls,
        catalog: Optional[_SQLiteSourceSequenceIndexCatalog],
    ) -> bool:
        """Validate uniqueness and the exact ordered SQLite key definition."""

        if (
            catalog is None
            or catalog.object_type != "index"
            or catalog.name != cls.SOURCE_SEQUENCE_SCOPE_INDEX
            or catalog.relation_name != cls.EVENTS
            or catalog.index_list_row is None
        ):
            return False

        row = catalog.index_list_row
        try:
            if (
                len(row) < 5
                or str(row[1]) != cls.SOURCE_SEQUENCE_SCOPE_INDEX
                or int(row[2]) != 1
                or str(row[3]) != "c"
                or int(row[4]) != 0
            ):
                return False

            key_rows = []
            for xinfo_row in catalog.index_xinfo_rows:
                if len(xinfo_row) < 6 or int(xinfo_row[5]) not in (0, 1):
                    return False
                if int(xinfo_row[5]) == 1:
                    key_rows.append(xinfo_row)
        except (TypeError, ValueError):
            return False

        expected_columns = ("agent_id", "source", "source_sequence")
        if len(key_rows) != len(expected_columns):
            return False
        try:
            return all(
                int(key_row[0]) == position
                and int(key_row[1]) >= 0
                and str(key_row[2]) == column
                and int(key_row[3]) == 0
                and str(key_row[4]).casefold() == "binary"
                for position, (key_row, column) in enumerate(
                    zip(key_rows, expected_columns)
                )
            )
        except (TypeError, ValueError):
            return False

    async def _postgres_source_sequence_index_catalog(self) -> Optional[Any]:
        """Return exact catalog evidence for the owned PostgreSQL index name."""

        if not self.is_postgres:
            raise RuntimeError("PostgreSQL source-sequence index requires postgres")
        return await self._backend.fetch_one(
            """
            SELECT index_relation.oid,
                   index_relation.relkind,
                   index_row.indisunique,
                   index_row.indisvalid,
                   index_row.indisready,
                   index_row.indislive,
                   index_row.indisprimary,
                   index_row.indisexclusion,
                   access_method.amname,
                   index_row.indnkeyatts,
                   index_row.indnatts,
                   index_row.indpred IS NULL,
                   index_row.indexprs IS NULL,
                   ARRAY(
                       SELECT attribute.attname
                       FROM unnest(index_row.indkey::smallint[])
                            WITH ORDINALITY AS key_column(attnum, position)
                       LEFT JOIN pg_attribute AS attribute
                         ON attribute.attrelid = table_relation.oid
                        AND attribute.attnum = key_column.attnum
                       ORDER BY key_column.position
                   ),
                   index_row.indoption::smallint[],
                   index_row.indcollation::oid[],
                   ARRAY(
                       SELECT attribute.attcollation
                       FROM unnest(index_row.indkey::smallint[])
                            WITH ORDINALITY AS key_column(attnum, position)
                       LEFT JOIN pg_attribute AS attribute
                         ON attribute.attrelid = table_relation.oid
                        AND attribute.attnum = key_column.attnum
                       ORDER BY key_column.position
                   ),
                   index_row.indclass::oid[],
                   ARRAY(
                       SELECT default_opclass.oid
                       FROM unnest(index_row.indkey::smallint[])
                            WITH ORDINALITY AS key_column(attnum, position)
                       LEFT JOIN pg_attribute AS attribute
                         ON attribute.attrelid = table_relation.oid
                        AND attribute.attnum = key_column.attnum
                       LEFT JOIN pg_opclass AS default_opclass
                         ON default_opclass.opcmethod = index_relation.relam
                        AND default_opclass.opcintype = attribute.atttypid
                        AND default_opclass.opcdefault
                       ORDER BY key_column.position
                   ),
                   index_row.indimmediate,
                   index_row.indnullsnotdistinct,
                   constraint_row.conname
            FROM pg_class AS table_relation
            JOIN pg_namespace AS table_namespace
              ON table_namespace.oid = table_relation.relnamespace
            JOIN pg_class AS index_relation
              ON index_relation.relnamespace = table_namespace.oid
             AND index_relation.relname = ?
            LEFT JOIN pg_index AS index_row
              ON index_row.indexrelid = index_relation.oid
             AND index_row.indrelid = table_relation.oid
            LEFT JOIN pg_am AS access_method
              ON access_method.oid = index_relation.relam
            LEFT JOIN pg_constraint AS constraint_row
              ON constraint_row.conindid = index_relation.oid
             AND constraint_row.conrelid = table_relation.oid
            WHERE table_namespace.nspname = current_schema()
              AND table_relation.relname = ?
            """,
            (self.SOURCE_SEQUENCE_SCOPE_INDEX, self.EVENTS),
        )

    @staticmethod
    def _postgres_source_sequence_index_catalog_valid(row: Optional[Any]) -> bool:
        """Validate uniqueness, exact keys, and live/ready/valid build state."""

        if row is None:
            return False
        try:
            columns = tuple(str(value) for value in (row[13] or ()))
            options = tuple(int(value) for value in (row[14] or ()))
            collations = tuple(int(value) for value in (row[15] or ()))
            expected_collations = tuple(int(value) for value in (row[16] or ()))
            operator_classes = tuple(int(value) for value in (row[17] or ()))
            default_operator_classes = tuple(
                int(value) for value in (row[18] or ())
            )
            immediate = bool(row[19])
            nulls_not_distinct = bool(row[20])
        except (IndexError, TypeError, ValueError):
            return False
        return bool(
            row[0] is not None
            and _postgres_catalog_char(row[1]) == "i"
            and bool(row[2])
            and bool(row[3])
            and bool(row[4])
            and bool(row[5])
            and not bool(row[6])
            and not bool(row[7])
            and str(row[8]) == "btree"
            and int(row[9]) == 3
            and int(row[10]) == 3
            and bool(row[11])
            and bool(row[12])
            and columns == ("agent_id", "source", "source_sequence")
            and options == (0, 0, 0)
            and len(collations) == 3
            and collations == expected_collations
            and len(operator_classes) == 3
            and operator_classes == default_operator_classes
            and immediate
            and not nulls_not_distinct
        )

    async def _ensure_postgres_source_sequence_index_concurrently(self) -> None:
        """Repair/build the exact index under initialize's session lock."""

        if not self.is_postgres:
            raise RuntimeError("PostgreSQL source-sequence index requires postgres")
        catalog = await self._postgres_source_sequence_index_catalog()
        if self._postgres_source_sequence_index_catalog_valid(catalog):
            return
        if catalog is not None:
            # CREATE INDEX CONCURRENTLY can leave an invalid/indisready shell
            # after cancellation or failure. IF NOT EXISTS would trust that
            # name and permanently strand the migration, so remove every
            # non-exact owned shape before retrying.
            constraint_name = (
                str(catalog[21])
                if len(catalog) > 21 and catalog[21] is not None
                else None
            )
            if constraint_name is not None:
                await self._backend.execute(
                    f"ALTER TABLE {_quoted_identifier(self.EVENTS)} "
                    f"DROP CONSTRAINT {_quoted_identifier(constraint_name)}"
                )
            else:
                await self._backend.execute(
                    "DROP INDEX CONCURRENTLY IF EXISTS "
                    f"{_quoted_identifier(self.SOURCE_SEQUENCE_SCOPE_INDEX)}"
                )
        await self._backend.execute(
            "CREATE UNIQUE INDEX CONCURRENTLY "
            f"{_quoted_identifier(self.SOURCE_SEQUENCE_SCOPE_INDEX)} "
            f"ON {_quoted_identifier(self.EVENTS)} "
            "(agent_id, source, source_sequence)"
        )
        final = await self._postgres_source_sequence_index_catalog()
        if not self._postgres_source_sequence_index_catalog_valid(final):
            raise RuntimeError(
                "PostgreSQL durable signal source-sequence index failed "
                "exact catalog validation"
            )

    async def _ensure_ordinary_indexes(self) -> None:
        """Ensure the pre-source-boundary ledger indexes."""

        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.EVENTS}_scope_retention "
            f"ON {self.EVENTS}(agent_id, source, retention_until)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.DELIVERIES}_claim "
            f"ON {self.DELIVERIES}(agent_id, consumer_id, status, next_attempt_at)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.DELIVERIES}_lease "
            f"ON {self.DELIVERIES}(agent_id, consumer_id, lease_expires_at)"
        )
        await self._backend.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.RUNTIME_OWNERS}_liveness "
            f"ON {self.RUNTIME_OWNERS}(agent_id, heartbeat_at, stopped_at)"
        )

    # ------------------------------------------------------------------
    # Subscription registration and event persistence
    # ------------------------------------------------------------------

    async def capture_source_boundary(
        self, *, agent_id: str, source: str
    ) -> DurableSourceBoundary:
        """Capture the current sequence for one exact tenant/source scope.

        Capture and event persistence take the same source handoff lock inside
        their database transaction.  The returned value is therefore ordered
        against every event commit in that scope even across processes and
        PostgreSQL hosts; no process or database timestamp participates.
        """

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("source", source)
        async with self._backend.transaction():
            await self._lock_scope_handoff(agent_id=agent_id, source=source)
            if (
                self.is_postgres
                and not await self._source_sequence_backfill_completed()
            ):
                unsequenced = await self._backend.fetch_val(
                    f"SELECT 1 FROM {self.EVENTS} "
                    "WHERE agent_id = ? AND source = ? "
                    "AND source_sequence IS NULL LIMIT 1",
                    (agent_id, source),
                )
                if unsequenced is not None:
                    raise RuntimeError(
                        "Durable source boundary is unavailable while this "
                        "scope has unsequenced legacy history"
                    )
            sequence = await self._source_sequence_locked(
                agent_id=agent_id, source=source
            )
        return DurableSourceBoundary(
            agent_id=agent_id,
            source=source,
            sequence=sequence,
        )

    async def register_consumer(
        self, registration: DurableConsumerRegistration
    ) -> None:
        """Persist an idempotent consumer registration and backfill it.

        A different registration reusing an existing consumer ID is rejected;
        silently changing a workflow's selector/retry policy would make old
        pending deliveries ambiguous.  Backfill is idempotent because delivery
        identity is unique on ``(agent_id, consumer_id, event_id)``.
        """
        self._validate_registration(registration)
        async with self._backend.transaction():
            # Consumer registration and event persistence must share one
            # serialization point.  Without it on PostgreSQL, an event can
            # commit after this transaction fails to see it during backfill
            # while that event's consumer lookup still cannot see this
            # uncommitted registration — permanently losing the delivery.
            # SQLite's transaction writer lock is reserved explicitly below;
            # the advisory transaction lock gives hosted PostgreSQL identical
            # handoff semantics at the narrow (agent, source) scope.
            await self._lock_scope_handoff(
                agent_id=registration.agent_id, source=registration.source
            )
            # The SQLite writer reservation / PostgreSQL advisory lock is the
            # serialization point. Sampling before it can make a retention
            # backfill admit an event that is already expired by the time this
            # transaction owns the handoff.
            now = self.now_utc()
            row = await self._backend.fetch_one(
                f"""
                SELECT source, correlation_selector, max_attempts,
                       lease_seconds, active
                FROM {self.CONSUMERS}
                WHERE agent_id = ? AND consumer_id = ?
                """,
                (registration.agent_id, registration.consumer_id),
            )
            expected = (
                registration.source,
                registration.correlation_selector,
                registration.max_attempts,
                registration.lease_seconds,
                self.to_bool_param(registration.active),
            )
            if row is not None:
                actual = (row[0], row[1], int(row[2]), int(row[3]), row[4])
                if actual != expected:
                    raise ValueError(
                        "Durable consumer registration conflicts with the "
                        f"existing contract for '{registration.consumer_id}'."
                    )
            else:
                await self._backend.execute(
                    f"""
                    INSERT INTO {self.CONSUMERS} (
                        agent_id, consumer_id, source, correlation_selector,
                        max_attempts, lease_seconds, active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        registration.agent_id,
                        registration.consumer_id,
                        registration.source,
                        registration.correlation_selector,
                        registration.max_attempts,
                        registration.lease_seconds,
                        self.to_bool_param(registration.active),
                        self.to_timestamp_param(now),
                        self.to_timestamp_param(now),
                    ),
                )
            # Inactive registrations are retained as configuration but do not
            # materialize work that no executor is allowed to claim.
            if registration.active:
                await self._backfill_consumer(registration, now=now)

    async def deactivate_consumer(self, *, agent_id: str, consumer_id: str) -> bool:
        """Deactivate one existing consumer and terminalize its live work.

        ``False`` means this agent/tenant has no such consumer.  A present
        consumer is successful even when it was already inactive.  The
        transition shares the source handoff lock with registration and event
        persistence: an event committed before this transaction can have a
        delivery, but that delivery is terminalized here; an event committed
        after it observes the inactive consumer and materializes nothing.

        Historical registration and delivery rows remain auditable.  Pending,
        retrying, initially reserved, and leased rows become terminal
        ``failed`` rows, with their ownership capabilities cleared, so a stale
        worker cannot acknowledge, retry, release, or reactivate the work.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        # Read only the immutable source before beginning a SQLite write
        # transaction.  Starting that transaction with a read then waiting for
        # another writer would pin a stale SQLite snapshot and make the later
        # lifecycle update fail instead of serializing behind the handoff.
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None:
            return False
        async with self._backend.transaction():
            await self._lock_scope_handoff(agent_id=agent_id, source=str(consumer[0]))
            consumer = await self._get_consumer(agent_id, consumer_id)
            if consumer is None:
                return False
            # Source is immutable after registration.  This common lock is
            # the durable linearization point for event materialization and
            # consumer lifecycle changes on both supported backends.
            now = self.now_utc()
            transitioned = await self._backend.execute(
                f"""
                UPDATE {self.CONSUMERS}
                SET active = ?, updated_at = ?
                WHERE agent_id = ? AND consumer_id = ? AND active = ?
                """,
                (
                    self.to_bool_param(False),
                    self.to_timestamp_param(now),
                    agent_id,
                    consumer_id,
                    self.to_bool_param(True),
                ),
            )
            if transitioned == 0:
                # The row still exists (checked above), so another caller has
                # already committed this idempotent lifecycle transition.
                return True
            await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, next_attempt_at = NULL,
                    last_error = ?, terminal_at = ?, updated_at = ?
                WHERE agent_id = ? AND consumer_id = ?
                  AND status NOT IN ('{ACKNOWLEDGED}', '{FAILED}', '{TERMINAL_ACKABLE}')
                """,
                (
                    FAILED,
                    _DEACTIVATED_CONSUMER_ERROR,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    agent_id,
                    consumer_id,
                ),
            )
        return True

    async def persist_signal(
        self,
        signal: Signal,
        *,
        agent_id: str,
        source_event_id: Optional[str],
        retention_days: int,
        transient_selector_payload: Any = _PERSISTED_PAYLOAD,
        initial_lease_owner: Optional[str] = None,
        integrity_binding: Optional[str] = None,
        caller_identity: Optional[str] = None,
        caller_identity_factory: Optional[Callable[[], str]] = None,
        before_commit: Optional[Callable[[DurableEventPersistence], None]] = None,
        on_rollback: Optional[Callable[[DurableEventPersistence], None]] = None,
    ) -> DurableEventPersistence:
        """Commit a persisted signal and all matching initial deliveries.

        This is the durable boundary called by the dispatcher *after*
        sanitization/schema normalization and causation validation, but before
        handlers, cognition, or any durable consumer can execute.  When a
        privacy projection fully elides payload content, ``signal`` is that
        safe persisted projection while ``transient_selector_payload`` is the
        normalized in-memory payload used only to materialize deliveries for
        consumers already registered in this transaction.  It is never
        serialized, returned, or used for restart backfill.  Projections that
        retain a replayable payload, including ANONYMOUS redaction, leave this
        argument unset so initial and replayed selector behavior is identical.

        ``initial_lease_owner`` reserves each initially matched delivery to
        one live dispatcher in the same transaction as the event insert.  It
        creates an ``INITIAL_RESERVED`` capability, not a lease: there is no
        countdown to expire before the transaction is visible.  The dispatcher
        installs its process-local raw-payload sidecars through
        ``before_commit`` before this transaction becomes visible, then
        activates the reservation after this method returns from commit.  A
        transaction failure invokes ``on_rollback`` so those sidecars cannot
        outlive rows that did not commit.  Async database drivers can report
        cancellation after their worker has completed ``commit``, however, so
        the callback is an *ambiguous transaction-outcome* notification rather
        than proof of rollback. Callers that retained an owner/token capability
        must conditionally repair it; that repair is a no-op for a confirmed
        rollback and releases a row whose commit was already durable. Both
        callbacks are synchronous deliberately:
        yielding between installing the sidecar and committing would reopen
        the very visibility race this handoff closes.
        """
        if retention_days < 0:
            raise ValueError("retention_days must be >= 0")
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("source", signal.source)
        if initial_lease_owner is not None:
            self._require_nonempty("initial_lease_owner", initial_lease_owner)
        if integrity_binding is not None and (
            type(integrity_binding) is not str
            or re.fullmatch(r"[0-9a-f]{64}", integrity_binding) is None
        ):
            raise ValueError("integrity_binding must be a SHA-256 hex digest")
        if caller_identity is not None and type(caller_identity) is not str:
            raise ValueError("caller_identity must be an opaque string when set")
        if caller_identity is not None and caller_identity_factory is not None:
            raise ValueError(
                "caller_identity and caller_identity_factory are mutually exclusive"
            )
        if caller_identity_factory is not None and not callable(caller_identity_factory):
            raise ValueError("caller_identity_factory must be callable when set")
        source_event_id = self._normalize_source_event_id(source_event_id)
        payload_json = _json_dump(signal.payload)
        chain_json = _json_dump(_serialize_chain(signal.causation_chain))
        persistence: Optional[DurableEventPersistence] = None
        try:
            async with self._backend.transaction():
                await self._lock_scope_handoff(agent_id=agent_id, source=signal.source)
                existing = await self._existing_event_id_locked(
                    agent_id, signal, source_event_id
                )
                if existing is not None:
                    existing_sequence = await self._event_source_sequence_locked(
                        event_id=existing,
                        agent_id=agent_id,
                        source=signal.source,
                    )
                    return DurableEventPersistence(
                        event_id=existing,
                        created=False,
                        source_sequence=existing_sequence,
                    )
                # The transaction may have waited behind the cross-instance
                # handoff lock. Start persisted event timing only after that
                # contention has cleared, never from method entry.
                now = self.now_utc()
                retention_until = now + timedelta(days=retention_days)
                source_sequence = await self._advance_source_sequence_locked(
                    agent_id=agent_id, source=signal.source
                )
                inserted = await self._backend.execute(
                    f"""
                    INSERT OR IGNORE INTO {self.EVENTS} (
                        event_id, source_event_id, agent_id, target_agent, source, kind, mode,
                        payload, session_id, caller_identity, visibility, urgency,
                        dedupe_key, causation_chain, arrived_at, committed_at,
                        retention_until, source_sequence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal.id,
                        source_event_id,
                        agent_id,
                        signal.target_agent,
                        signal.source,
                        signal.kind,
                        signal.mode.value,
                        payload_json,
                        signal.session_id,
                        caller_identity,
                        signal.visibility.value,
                        signal.urgency.value,
                        signal.dedupe_key,
                        chain_json,
                        self.to_timestamp_param(signal.arrived_at),
                        self.to_timestamp_param(now),
                        self.to_timestamp_param(retention_until),
                        source_sequence,
                    ),
                )
                if inserted == 0:
                    # The source-scoped identity was absent after acquiring the
                    # handoff lock.  Any later conflict is a global event-ID
                    # collision or a writer that bypassed the lock.  Abort so
                    # the source counter increment rolls back with the insert.
                    existing = await self._find_existing_event_locked(
                        agent_id, signal, source_event_id
                    )
                    raise RuntimeError(
                        "durable event identity appeared outside its source "
                        f"handoff lock: {existing}"
                    )

                if caller_identity_factory is not None:
                    caller_identity = caller_identity_factory()
                    if type(caller_identity) is not str:
                        raise RuntimeError(
                            "caller_identity_factory returned a non-string value"
                        )
                    await self._backend.execute(
                        f"""
                        UPDATE {self.EVENTS}
                        SET caller_identity = ?
                        WHERE event_id = ? AND agent_id = ?
                        """,
                        (caller_identity, signal.id, agent_id),
                    )

                if integrity_binding is not None:
                    await self._backend.execute(
                        f"""
                        INSERT INTO {self.EVENT_INTEGRITY} (event_id, integrity_binding)
                        VALUES (?, ?)
                        """,
                        (signal.id, integrity_binding),
                    )

                consumer_rows = await self._backend.fetch_all(
                    f"""
                    SELECT consumer_id, correlation_selector, max_attempts, lease_seconds
                    FROM {self.CONSUMERS}
                    WHERE agent_id = ? AND source = ? AND active = ?
                    """,
                    (agent_id, signal.source, self.to_bool_param(True)),
                )
                event = self._event_from_signal(
                    signal,
                    agent_id=agent_id,
                    source_event_id=source_event_id,
                    caller_identity=caller_identity,
                    committed_at=now,
                    retention_until=retention_until,
                    source_sequence=source_sequence,
                )
                selector_event = (
                    event
                    if transient_selector_payload is _PERSISTED_PAYLOAD
                    else replace(event, payload=transient_selector_payload)
                )
                delivery_ids: list[str] = []
                initial_reservations: list[DurableInitialDeliveryReservation] = []
                for consumer_id, selector, max_attempts, lease_seconds in consumer_rows:
                    if not self._matches_selector(selector_event, selector):
                        continue
                    reservation_token = None
                    if initial_lease_owner is not None:
                        reservation_token = secrets.token_urlsafe(24)
                    delivery_id = await self._insert_delivery_locked(
                        agent_id=agent_id,
                        consumer_id=consumer_id,
                        event_id=signal.id,
                        max_attempts=int(max_attempts),
                        now=now,
                        initial_reservation_owner=initial_lease_owner,
                        initial_reservation_token=reservation_token,
                    )
                    if delivery_id is not None:
                        delivery_ids.append(delivery_id)
                        if reservation_token is not None:
                            initial_reservations.append(
                                DurableInitialDeliveryReservation(
                                    delivery_id=delivery_id,
                                    consumer_id=consumer_id,
                                    reservation_token=reservation_token,
                                    created_at=now,
                                )
                            )
                persistence = DurableEventPersistence(
                    event_id=signal.id,
                    created=True,
                    delivery_ids=tuple(delivery_ids),
                    retention_until=retention_until,
                    initial_reservations=tuple(initial_reservations),
                    source_sequence=source_sequence,
                )
                if before_commit is not None:
                    before_commit(persistence)
        except BaseException:
            if persistence is not None and on_rollback is not None:
                on_rollback(persistence)
            raise
        assert persistence is not None
        return persistence

    async def upgrade_legacy_delivery_for_redelivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        event_id: str,
        source_event_id: str,
        expected_signal: Signal,
        caller_identity_factory: Callable[[], str],
    ) -> bool:
        """Atomically add one delivery to a verified pre-consumer event.

        This deliberately is *not* a consumer backfill.  It upgrades only the
        immutable event named by a provider's current redelivery, after the
        caller has proved that its normalized live envelope matches the old
        retained event.  The caller identity was not protected by the
        pre-upgrade schema, so it is sealed and the exact delivery is created
        in the same transaction.  Privacy-elided rows carry an integrity row
        and are refused here; their keyed-MAC retry path remains fail-closed.
        """

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("event_id", event_id)
        self._require_nonempty("source_event_id", source_event_id)
        self._require_nonempty("source", expected_signal.source)
        if not callable(caller_identity_factory):
            raise ValueError("caller_identity_factory must be callable")

        async with self._backend.transaction():
            await self._lock_scope_handoff(
                agent_id=agent_id, source=expected_signal.source
            )
            # This upgrade is a live redelivery operation, not a historical
            # repair.  Sample once after the transaction has acquired the
            # source handoff lock and use that same instant for both retention
            # admission and the new delivery's timestamps.  Backfill retains
            # the exact boundary (``retention_until >= now``), so a row at the
            # boundary is still eligible here as well.
            now = self.now_utc()
            consumer = await self._get_consumer(agent_id, consumer_id)
            if consumer is None or not consumer[4] or consumer[0] != expected_signal.source:
                return False
            row = await self._backend.fetch_one(
                f"""
                SELECT event_id, source_event_id, agent_id, target_agent, source, kind, mode, payload,
                       session_id, caller_identity, visibility, urgency, dedupe_key,
                       causation_chain, arrived_at, committed_at, retention_until,
                       source_sequence
                FROM {self.EVENTS}
                WHERE event_id = ? AND agent_id = ? AND source = ?
                """,
                (event_id, agent_id, expected_signal.source),
            )
            if row is None:
                return False
            event = self._row_to_event(row)
            if (
                event.retention_until < now
                or
                event.caller_identity is not None
                or not self._legacy_event_matches_redelivery(
                    event, expected_signal, source_event_id
                )
                # An event that would already have matched the registered
                # selector is not a marker-era legacy row.  Never use a
                # redelivery to alter such historical work.
                or self._matches_selector(event, consumer[1])
            ):
                return False
            integrity = await self._backend.fetch_one(
                f"SELECT 1 FROM {self.EVENT_INTEGRITY} WHERE event_id = ?",
                (event_id,),
            )
            if integrity is not None:
                return False
            delivery = await self._backend.fetch_one(
                f"""
                SELECT 1 FROM {self.DELIVERIES}
                WHERE agent_id = ? AND consumer_id = ? AND event_id = ?
                """,
                (agent_id, consumer_id, event_id),
            )
            if delivery is not None:
                return False

            caller_identity = caller_identity_factory()
            if type(caller_identity) is not str or not caller_identity:
                raise RuntimeError(
                    "caller_identity_factory returned an invalid protected value"
                )

            updated = await self._backend.execute(
                f"""
                UPDATE {self.EVENTS}
                SET caller_identity = ?
                WHERE event_id = ? AND agent_id = ? AND caller_identity IS NULL
                """,
                (caller_identity, event_id, agent_id),
            )
            if updated != 1:
                return False
            delivery_id = await self._insert_delivery_locked(
                agent_id=agent_id,
                consumer_id=consumer_id,
                event_id=event_id,
                max_attempts=int(consumer[2]),
                now=now,
            )
            if delivery_id is None:
                # This transaction owns the source handoff lock, so an
                # unexpected duplicate would otherwise leave an identity-only
                # partial upgrade.  Raising rolls every change back together.
                raise RuntimeError("legacy delivery upgrade conflicted unexpectedly")
            return True

    # ------------------------------------------------------------------
    # Claim / lease / acknowledgement API
    # ------------------------------------------------------------------

    async def claim_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        executor_id: str,
        now: Optional[datetime] = None,
        runtime_owner_stale_before: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Atomically lease one due delivery for this scoped consumer.

        The conditional UPDATE is the ownership handoff.  Two SQLite
        connections or two PostgreSQL executors may choose the same candidate
        in their subqueries, but only one can change its still-claimable state;
        the loser observes a zero row count and receives no delivery.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("executor_id", executor_id)
        explicit_now = _as_utc(now) if now is not None else None
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        # Catch events committed while a consumer was restarting.  The unique
        # delivery key makes this safe to do on every poll.
        registration = DurableConsumerRegistration(
            agent_id=agent_id,
            consumer_id=consumer_id,
            source=consumer[0],
            correlation_selector=consumer[1],
            max_attempts=int(consumer[2]),
            lease_seconds=int(consumer[3]),
            active=bool(consumer[4]),
        )
        # Claim, recovery, and restart backfill all serialize with event
        # persistence for this source.  Besides avoiding a registration/event
        # visibility gap, this makes SQLite's single writer serialization
        # explicit. PostgreSQL still needs to serialize the actual delivery
        # row below before it observes an implicit lease clock.
        async with self._backend.transaction():
            await self._lock_scope_handoff(agent_id=agent_id, source=registration.source)
            # ``consumer`` was read before acquiring the source handoff lock.
            # Re-read it here so a deactivation that committed while this
            # claimant waited cannot backfill or lease historical work.
            consumer = await self._get_consumer(agent_id, consumer_id)
            if consumer is None or not consumer[4]:
                return None
            recovery_now = explicit_now or self.now_utc()
            await self._recover_expired_leases(
                agent_id=agent_id,
                consumer_id=consumer_id,
                now=recovery_now,
                runtime_owner_stale_before=(
                    _as_utc(runtime_owner_stale_before)
                    if runtime_owner_stale_before is not None
                    else recovery_now - _DEFAULT_RUNTIME_OWNER_STALE_AFTER
                ),
            )
            # Backfill is idempotent because delivery identity is unique.
            await self._backfill_consumer(registration, now=recovery_now)
            delivery_id = await self._lock_claimable_delivery(
                agent_id=agent_id,
                consumer_id=consumer_id,
                now=recovery_now,
            )
            if delivery_id is None:
                return None
            # A PostgreSQL row lock can wait behind a worker that is slower
            # than this consumer's entire lease.  Preserve an explicitly
            # supplied timestamp exactly, but otherwise sample only after the
            # selected delivery is serialized so the new lease is full-lived.
            effective_now = explicit_now or self.now_utc()
            lease_token = secrets.token_urlsafe(24)
            lease_expires_at = effective_now + timedelta(
                seconds=registration.lease_seconds
            )
            updated = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, attempts = attempts + 1, lease_owner = ?,
                    lease_token = ?, lease_expires_at = ?, next_attempt_at = NULL,
                    updated_at = ?
                WHERE delivery_id = ? AND agent_id = ? AND consumer_id = ?
                  AND status IN ('{PENDING}', '{RETRY}')
                  AND (max_attempts = 0 OR attempts < max_attempts)
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                """,
                (
                    LEASED,
                    executor_id,
                    lease_token,
                    self.to_timestamp_param(lease_expires_at),
                    self.to_timestamp_param(effective_now),
                    delivery_id,
                    agent_id,
                    consumer_id,
                    self.to_timestamp_param(effective_now),
                ),
            )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=lease_token,
        )

    async def claim_delivery_for_event(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        event_id: str,
        executor_id: str,
        now: Optional[datetime] = None,
        runtime_owner_stale_before: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Atomically claim this consumer's delivery for one persisted event.

        Cursor-owning ingress must never let a concurrent callback claim an
        unrelated pending event and then acknowledge the wrong provider
        cursor.  This is the exact-event counterpart of ``claim_delivery``.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("event_id", event_id)
        self._require_nonempty("executor_id", executor_id)
        explicit_now = _as_utc(now) if now is not None else None
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        async with self._backend.transaction():
            await self._lock_scope_handoff(agent_id=agent_id, source=consumer[0])
            # See ``claim_delivery``: the first read is only enough to find
            # this immutable source scope.  Lifecycle state must be observed
            # after this transaction owns the shared handoff lock.
            consumer = await self._get_consumer(agent_id, consumer_id)
            if consumer is None or not consumer[4]:
                return None
            recovery_now = explicit_now or self.now_utc()
            await self._recover_expired_leases(
                agent_id=agent_id,
                consumer_id=consumer_id,
                now=recovery_now,
                runtime_owner_stale_before=(
                    _as_utc(runtime_owner_stale_before)
                    if runtime_owner_stale_before is not None
                    else recovery_now - _DEFAULT_RUNTIME_OWNER_STALE_AFTER
                ),
            )
            delivery_id = await self._lock_claimable_delivery(
                agent_id=agent_id,
                consumer_id=consumer_id,
                event_id=event_id,
                now=recovery_now,
            )
            if delivery_id is None:
                return None
            # See claim_delivery: do not publish an already-expired implicit
            # lease after waiting for this exact delivery row.
            effective_now = explicit_now or self.now_utc()
            lease_token = secrets.token_urlsafe(24)
            lease_expires_at = effective_now + timedelta(seconds=int(consumer[3]))
            updated = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, attempts = attempts + 1, lease_owner = ?,
                    lease_token = ?, lease_expires_at = ?, next_attempt_at = NULL,
                    updated_at = ?
                WHERE delivery_id = ? AND agent_id = ? AND consumer_id = ? AND event_id = ?
                  AND status IN ('{PENDING}', '{RETRY}')
                  AND (max_attempts = 0 OR attempts < max_attempts)
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                """,
                (
                    LEASED,
                    executor_id,
                    lease_token,
                    self.to_timestamp_param(lease_expires_at),
                    self.to_timestamp_param(effective_now),
                    delivery_id,
                    agent_id,
                    consumer_id,
                    event_id,
                    self.to_timestamp_param(effective_now),
                ),
            )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=lease_token,
        )

    async def _lock_claimable_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        now: datetime,
        event_id: Optional[str] = None,
    ) -> Optional[str]:
        """Serialize one due delivery before assigning an implicit lease clock.

        ``_lock_scope_handoff`` protects event/consumer registration handoff,
        but a PostgreSQL transaction may still block on the selected delivery
        row itself (for example, an operator repair holding that row).  Select
        and lock that exact row before the caller samples its implicit clock.
        SQLite has already acquired its single writer in ``_lock_scope_handoff``;
        the same select keeps both backends on one claim contract.
        """

        where = [
            "agent_id = ?",
            "consumer_id = ?",
            f"status IN ('{PENDING}', '{RETRY}')",
            "(max_attempts = 0 OR attempts < max_attempts)",
            "(next_attempt_at IS NULL OR next_attempt_at <= ?)",
        ]
        params: list[Any] = [agent_id, consumer_id, self.to_timestamp_param(now)]
        if event_id is not None:
            where.insert(2, "event_id = ?")
            params.insert(2, event_id)
        lock_clause = " FOR UPDATE" if self.is_postgres else ""
        row = await self._backend.fetch_one(
            f"""
            SELECT delivery_id FROM {self.DELIVERIES}
            WHERE {' AND '.join(where)}
            ORDER BY created_at, delivery_id
            LIMIT 1{lock_clause}
            """,
            tuple(params),
        )
        return str(row[0]) if row is not None else None

    async def renew_delivery_lease(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Extend one still-owned lease using the consumer's persisted policy.

        A long cognition turn keeps its original token. This conditional update
        refuses an expired or foreign lease, so a late worker can never revive
        work another executor may already have claimed.
        """

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("lease_token", lease_token)
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        now = _as_utc(now or self.now_utc())
        lease_expires_at = now + timedelta(seconds=int(consumer[3]))
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET lease_expires_at = ?, updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_token = ? AND lease_expires_at > ?
            """,
            (
                self.to_timestamp_param(lease_expires_at),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                delivery_id,
                LEASED,
                lease_token,
                self.to_timestamp_param(now),
            ),
        )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=lease_token,
        )

    async def claim_initial_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        initial_lease_owner: str,
        initial_lease_token: str,
        executor_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Transfer one activated emitting-dispatcher lease to a worker.

        An initial reservation first becomes a real ``LEASED`` row through
        :meth:`activate_initial_delivery`, which runs only after the event
        transaction has committed.  A normal claimant cannot claim either the
        reservation or the activated owner lease.  Only the dispatcher holding
        this unpersisted capability may make the first worker claim; after
        that, ordinary retry/lease rules apply.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("initial_lease_owner", initial_lease_owner)
        self._require_nonempty("initial_lease_token", initial_lease_token)
        self._require_nonempty("executor_id", executor_id)
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        requested_now = _as_utc(now) if now is not None else None
        async with self._backend.transaction():
            # Acquire the delivery's write/row serialization point before
            # observing time.  SQLite transactions begin deferred, so the
            # targeted no-op update reserves the actual writer slot; PostgreSQL
            # locks this row with ``FOR UPDATE``.  Sampling first can publish a
            # worker lease that has already expired while waiting here.
            initial = await self._lock_initial_delivery_transfer(
                agent_id=agent_id,
                consumer_id=consumer_id,
                delivery_id=delivery_id,
            )
            if initial is None:
                return None
            status, owner, token, expires_at = initial
            transfer_now = requested_now or self.now_utc()
            if (
                status != LEASED
                or owner != initial_lease_owner
                or token != initial_lease_token
                or expires_at is None
                or _as_utc(self.from_timestamp_field(expires_at)) <= transfer_now
            ):
                return None
            lease_token = secrets.token_urlsafe(24)
            lease_expires_at = transfer_now + timedelta(seconds=int(consumer[3]))
            updated = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET attempts = attempts + 1, lease_owner = ?, lease_token = ?,
                    lease_expires_at = ?, next_attempt_at = NULL, updated_at = ?
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                  AND status = ? AND lease_owner = ? AND lease_token = ?
                  AND lease_expires_at > ?
                  AND EXISTS (
                      SELECT 1 FROM {self.CONSUMERS} consumer
                      WHERE consumer.agent_id = {self.DELIVERIES}.agent_id
                        AND consumer.consumer_id = {self.DELIVERIES}.consumer_id
                        AND consumer.active = ?
                  )
                """,
                (
                    executor_id,
                    lease_token,
                    self.to_timestamp_param(lease_expires_at),
                    self.to_timestamp_param(transfer_now),
                    agent_id,
                    consumer_id,
                    delivery_id,
                    LEASED,
                    initial_lease_owner,
                    initial_lease_token,
                    self.to_timestamp_param(transfer_now),
                    self.to_bool_param(True),
                ),
            )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=lease_token,
        )

    async def activate_initial_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        initial_lease_owner: str,
        initial_lease_token: str,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Start a reservation's first real lease after its event commits.

        ``persist_signal`` inserts an ``INITIAL_RESERVED`` row with no lease
        deadline.  This conditional transition is intentionally a separate
        post-commit operation: it is the first place a live lease countdown is
        calculated, so a paused commit can never publish an already-expired
        delivery.  The runtime-owner heartbeat and this transition share a
        transaction so stale-owner recovery cannot take a live emitter that is
        actively activating its own reservation.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("initial_lease_owner", initial_lease_owner)
        self._require_nonempty("initial_lease_token", initial_lease_token)
        consumer = await self._get_consumer(agent_id, consumer_id)
        if consumer is None or not consumer[4]:
            return None
        requested_now = _as_utc(now) if now is not None else None
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            # The transaction may wait behind a real database writer. Sample
            # time only after that contention has cleared and immediately
            # before the activation write; this is the first live delivery
            # deadline and must not inherit any pre-commit delay.
            activation_now = requested_now or self.now_utc()
            await self._touch_runtime_owner_locked(
                agent_id=agent_id,
                owner_id=initial_lease_owner,
                now=activation_now,
            )
            activation_now = requested_now or self.now_utc()
            lease_expires_at = activation_now + timedelta(
                seconds=int(consumer[3])
            )
            updated = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, lease_expires_at = ?, updated_at = ?
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                  AND status = ? AND lease_owner = ? AND lease_token = ?
                  AND lease_expires_at IS NULL
                  AND EXISTS (
                      SELECT 1 FROM {self.CONSUMERS} consumer
                      WHERE consumer.agent_id = {self.DELIVERIES}.agent_id
                        AND consumer.consumer_id = {self.DELIVERIES}.consumer_id
                        AND consumer.active = ?
                  )
                """,
                (
                    LEASED,
                    self.to_timestamp_param(lease_expires_at),
                    self.to_timestamp_param(activation_now),
                    agent_id,
                    consumer_id,
                    delivery_id,
                    INITIAL_RESERVED,
                    initial_lease_owner,
                    initial_lease_token,
                    self.to_bool_param(True),
                ),
            )
        if updated == 0:
            return None
        return await self._delivery_for_lease_locked(
            agent_id=agent_id,
            consumer_id=consumer_id,
            lease_token=initial_lease_token,
        )

    async def register_runtime_owner(
        self,
        *,
        agent_id: str,
        owner_id: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Record a live dispatcher generation before it can reserve work."""
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("owner_id", owner_id)
        requested_now = _as_utc(now) if now is not None else None
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            # Entering this transaction may wait behind a real writer.  A
            # default timestamp is liveness evidence, so sample it only after
            # that contention clears rather than publishing an old heartbeat.
            touch_now = requested_now or self.now_utc()
            await self._touch_runtime_owner_locked(
                agent_id=agent_id, owner_id=owner_id, now=touch_now
            )

    async def heartbeat_runtime_owner(
        self,
        *,
        agent_id: str,
        owner_id: str,
        now: Optional[datetime] = None,
    ) -> None:
        """Refresh one dispatcher generation's liveness record."""
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("owner_id", owner_id)
        requested_now = _as_utc(now) if now is not None else None
        async with self._backend.transaction():
            # Recovery uses this exact scope before it decides whether a
            # managed lease owner is stale.  Without the common lock a
            # PostgreSQL recovery snapshot can classify the old heartbeat as
            # stale while this refresh is concurrently committing.
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            # See register_runtime_owner: a heartbeat taken before waiting on
            # this transaction is not trustworthy liveness evidence.
            touch_now = requested_now or self.now_utc()
            await self._touch_runtime_owner_locked(
                agent_id=agent_id, owner_id=owner_id, now=touch_now
            )

    async def release_initial_reservations(
        self,
        *,
        agent_id: str,
        owner_id: str,
        now: Optional[datetime] = None,
        mark_owner_stopped: bool = True,
    ) -> int:
        """Release this runtime's unactivated rows into marker replay.

        This is deliberately scoped to one runtime owner.  A concurrent live
        dispatcher cannot release another emitter's raw-payload reservation.
        A cancellation-resistant cognition task can outlive ordinary shutdown;
        callers retain that owner's liveness fence by passing
        ``mark_owner_stopped=False`` until the task is actually settled.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("owner_id", owner_id)
        now = _as_utc(now or self.now_utc())
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            released = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, next_attempt_at = ?,
                    last_error = 'initial reservation owner stopped before activation',
                    updated_at = ?
                WHERE agent_id = ? AND status = ? AND lease_owner = ?
                """,
                (
                    RETRY,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    agent_id,
                    INITIAL_RESERVED,
                    owner_id,
                ),
            )
            if mark_owner_stopped:
                await self._backend.execute(
                    f"""
                    UPDATE {self.RUNTIME_OWNERS}
                    SET heartbeat_at = ?, stopped_at = ?, updated_at = ?
                    WHERE agent_id = ? AND owner_id = ?
                    """,
                    (
                        self.to_timestamp_param(now),
                        self.to_timestamp_param(now),
                        self.to_timestamp_param(now),
                        agent_id,
                        owner_id,
                    ),
                )
        return released

    async def abandon_initial_reservation(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        owner_id: str,
        reservation_token: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """Release one initial capability whose raw handoff cannot complete.

        The owner/token pair identifies both an unactivated reservation and
        its just-activated first lease.  The latter case is possible only if
        the activation write committed before its readback failed; no worker
        can own it yet because the dispatcher still holds its local handoff
        lock.  Both forms must become marker-only retry work.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("owner_id", owner_id)
        self._require_nonempty("reservation_token", reservation_token)
        now = _as_utc(now or self.now_utc())
        async with self._backend.transaction():
            released = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, next_attempt_at = ?,
                    last_error = 'initial reservation activation unavailable',
                    updated_at = ?
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                  AND status IN (?, ?) AND lease_owner = ? AND lease_token = ?
                """,
                (
                    RETRY,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    agent_id,
                    consumer_id,
                    delivery_id,
                    INITIAL_RESERVED,
                    LEASED,
                    owner_id,
                    reservation_token,
                ),
            )
        return released == 1

    async def recover_abandoned_initial_reservations(
        self,
        *,
        agent_id: str,
        recovering_owner_id: str,
        stale_before: datetime,
        now: Optional[datetime] = None,
    ) -> int:
        """Requeue reservations only from stale managed dispatcher owners.

        Generic claim/retry recovery intentionally never touches
        ``INITIAL_RESERVED``.  Startup uses this owner-aware path instead;
        another live dispatcher remains protected by its heartbeat even when
        it shares the same tenant and consumer IDs. Public executor owners and
        unknown owner namespaces are never recovery candidates.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("recovering_owner_id", recovering_owner_id)
        stale_before = _as_utc(stale_before)
        now = _as_utc(now or self.now_utc())
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            released = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = ?, lease_owner = NULL, lease_token = NULL,
                    lease_expires_at = NULL, next_attempt_at = ?,
                    last_error = 'initial reservation owner unavailable before activation',
                    updated_at = ?
                WHERE agent_id = ? AND status = ? AND lease_owner <> ?
                  AND lease_owner LIKE 'dispatcher:%'
                  AND EXISTS (
                      SELECT 1 FROM {self.RUNTIME_OWNERS} owner
                      WHERE owner.agent_id = {self.DELIVERIES}.agent_id
                        AND owner.owner_id = {self.DELIVERIES}.lease_owner
                        AND (
                            owner.stopped_at IS NOT NULL
                            OR owner.heartbeat_at < ?
                        )
                  )
                """,
                (
                    RETRY,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    agent_id,
                    INITIAL_RESERVED,
                    recovering_owner_id,
                    self.to_timestamp_param(stale_before),
                ),
            )
        return released

    async def recover_abandoned_leases(
        self,
        *,
        agent_id: str,
        recovering_owner_id: str,
        stale_before: datetime,
        now: Optional[datetime] = None,
    ) -> int:
        """Requeue only managed dispatcher leases whose owner is stale/stopped.

        A normal lease can be committed before cognition begins.  On restart,
        retaining that lease until its deadline would make a provider callback
        look like a duplicate even though its cognition was never made safe.
        Owner-aware recovery restores that delivery without disturbing a live
        sibling dispatcher sharing the same tenant. It deliberately preserves
        public executor leases and unknown ownership domains.
        """
        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("recovering_owner_id", recovering_owner_id)
        stale_before = _as_utc(stale_before)
        now = _as_utc(now or self.now_utc())
        timestamp = self._timestamp_placeholder()
        async with self._backend.transaction():
            await self._lock_runtime_owner_scope(agent_id=agent_id)
            released = await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET status = CASE
                        WHEN max_attempts > 0 AND attempts >= max_attempts THEN ?
                        ELSE ?
                    END,
                    lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                    next_attempt_at = CASE
                        WHEN max_attempts > 0 AND attempts >= max_attempts THEN NULL
                        ELSE {timestamp}
                    END,
                    last_error = 'lease owner unavailable before acknowledgement',
                    terminal_at = CASE
                        WHEN max_attempts > 0 AND attempts >= max_attempts
                        THEN {timestamp} ELSE NULL
                    END,
                    updated_at = ?
                WHERE agent_id = ? AND status = ? AND lease_owner <> ?
                  AND lease_owner LIKE 'dispatcher:%'
                  AND EXISTS (
                      SELECT 1 FROM {self.RUNTIME_OWNERS} owner
                      WHERE owner.agent_id = {self.DELIVERIES}.agent_id
                        AND owner.owner_id = {self.DELIVERIES}.lease_owner
                        AND (
                            owner.stopped_at IS NOT NULL
                            OR owner.heartbeat_at < ?
                        )
                  )
                """,
                (
                    FAILED,
                    RETRY,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    agent_id,
                    LEASED,
                    recovering_owner_id,
                    self.to_timestamp_param(stale_before),
                ),
            )
        return released

    async def ack_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """Acknowledge a live lease.  Stale/foreign tokens cannot ack it."""
        now = _as_utc(now or self.now_utc())
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = ?, lease_owner = NULL, lease_token = NULL,
                lease_expires_at = NULL, acknowledged_at = ?, terminal_at = ?,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_token = ? AND lease_expires_at > ?
            """,
            (
                ACKNOWLEDGED,
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                delivery_id,
                LEASED,
                lease_token,
                self.to_timestamp_param(now),
            ),
        )
        return updated == 1

    async def nack_delivery(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        error: str,
        retry_delay: timedelta = timedelta(),
        terminal: bool = False,
        terminal_ackable: bool = False,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Release a failed lease for retry or mark a terminal failure.

        Retry is bounded by the delivery's persisted ``max_attempts`` unless
        that value is zero, which intentionally means retry until an explicit
        terminal failure or acknowledgement.
        """
        self._require_nonempty("error", error)
        if retry_delay.total_seconds() < 0:
            raise ValueError("retry_delay must not be negative")
        if terminal_ackable and not terminal:
            raise ValueError("terminal_ackable deliveries must be terminal")
        now = _as_utc(now or self.now_utc())
        retry_at = now + retry_delay
        timestamp = self._timestamp_placeholder()
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = CASE
                    WHEN ? THEN ?
                    WHEN ? OR (max_attempts > 0 AND attempts >= max_attempts) THEN ?
                    ELSE ?
                END,
                lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                next_attempt_at = CASE
                    WHEN ? OR ? OR (max_attempts > 0 AND attempts >= max_attempts)
                        THEN NULL ELSE {timestamp} END,
                last_error = ?,
                terminal_at = CASE
                    WHEN ? OR ? OR (max_attempts > 0 AND attempts >= max_attempts)
                        THEN {timestamp} ELSE NULL
                    END,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_token = ? AND lease_expires_at > ?
            """,
            (
                self.to_bool_param(terminal_ackable),
                TERMINAL_ACKABLE,
                self.to_bool_param(terminal),
                FAILED,
                RETRY,
                self.to_bool_param(terminal_ackable),
                self.to_bool_param(terminal),
                self.to_timestamp_param(retry_at),
                error,
                self.to_bool_param(terminal_ackable),
                self.to_bool_param(terminal),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                delivery_id,
                LEASED,
                lease_token,
                self.to_timestamp_param(now),
            ),
        )
        if updated == 0:
            return None
        return await self.get_delivery(
            agent_id=agent_id, consumer_id=consumer_id, delivery_id=delivery_id
        )

    async def release_delivery_for_hold(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """Undo an exact lease transfer when Hold wins admission.

        Claim increments ``attempts`` as part of the ownership handoff. A Hold
        observed immediately afterward means no executor received the work,
        so the aborted handoff must not consume the retry budget.
        """

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("lease_token", lease_token)
        now = _as_utc(now or self.now_utc())
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = ?, attempts = CASE
                    WHEN attempts > 0 THEN attempts - 1 ELSE 0
                END,
                lease_owner = NULL, lease_token = NULL,
                lease_expires_at = NULL, next_attempt_at = ?,
                last_error = COALESCE(
                    last_error, 'hold committed during lease transfer'
                ),
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_token = ? AND lease_expires_at > ?
            """,
            (
                RETRY,
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                delivery_id,
                LEASED,
                lease_token,
                self.to_timestamp_param(now),
            ),
        )
        return updated == 1

    async def release_managed_delivery_after_task(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        delivery_id: str,
        lease_token: str,
        owner_id: str,
        error: str,
        terminal: bool = False,
        terminal_ackable: bool = False,
        now: Optional[datetime] = None,
    ) -> Optional[DurableDelivery]:
        """Release a completed local task's exact managed lease.

        A dispatcher may learn that its renewal path is lost while the
        cognition coroutine ignores cancellation.  Normal NACK intentionally
        refuses an expired lease, but the live managed-owner heartbeat keeps
        that lease out of generic expiry recovery until this exact coroutine
        has settled.  At that point this owner/token conditional transition is
        safe even after the nominal deadline: no other claimant could have
        acquired the row while the owner was live.  The owner predicate keeps
        this narrow escape hatch unavailable to public or unknown executors.
        """

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("consumer_id", consumer_id)
        self._require_nonempty("delivery_id", delivery_id)
        self._require_nonempty("lease_token", lease_token)
        self._require_nonempty("owner_id", owner_id)
        self._require_nonempty("error", error)
        if terminal_ackable and not terminal:
            raise ValueError("terminal_ackable deliveries must be terminal")
        if not owner_id.startswith("dispatcher:"):
            raise ValueError("owner_id must identify a managed dispatcher")
        now = _as_utc(now or self.now_utc())
        timestamp = self._timestamp_placeholder()
        updated = await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = CASE
                    WHEN ? THEN ?
                    WHEN ? OR (max_attempts > 0 AND attempts >= max_attempts) THEN ?
                    ELSE ?
                END,
                lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                next_attempt_at = CASE
                    WHEN ? OR ? OR (max_attempts > 0 AND attempts >= max_attempts)
                        THEN NULL ELSE {timestamp}
                END,
                last_error = ?,
                terminal_at = CASE
                    WHEN ? OR ? OR (max_attempts > 0 AND attempts >= max_attempts)
                        THEN {timestamp} ELSE NULL
                END,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
              AND status = ? AND lease_owner = ? AND lease_token = ?
            """,
            (
                self.to_bool_param(terminal_ackable),
                TERMINAL_ACKABLE,
                self.to_bool_param(terminal),
                FAILED,
                RETRY,
                self.to_bool_param(terminal_ackable),
                self.to_bool_param(terminal),
                self.to_timestamp_param(now),
                error,
                self.to_bool_param(terminal_ackable),
                self.to_bool_param(terminal),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                delivery_id,
                LEASED,
                owner_id,
                lease_token,
            ),
        )
        if updated == 0:
            return None
        return await self.get_delivery(
            agent_id=agent_id, consumer_id=consumer_id, delivery_id=delivery_id
        )

    async def get_delivery(
        self, *, agent_id: str, consumer_id: str, delivery_id: str
    ) -> Optional[DurableDelivery]:
        row = await self._backend.fetch_one(
            self._delivery_select_sql(
                "d.agent_id = ? AND d.consumer_id = ? AND d.delivery_id = ?"
            ),
            (agent_id, consumer_id, delivery_id),
        )
        return self._row_to_delivery(row) if row is not None else None

    async def get_delivery_for_event(
        self, *, agent_id: str, consumer_id: str, event_id: str
    ) -> Optional[DurableDelivery]:
        """Read one consumer delivery by its immutable persisted event ID."""
        row = await self._backend.fetch_one(
            self._delivery_select_sql(
                "d.agent_id = ? AND d.consumer_id = ? AND d.event_id = ?"
            ),
            (agent_id, consumer_id, event_id),
        )
        return self._row_to_delivery(row) if row is not None else None

    async def get_event_integrity(
        self, *, agent_id: str, event_id: str
    ) -> Optional[str]:
        """Return one agent-scoped privacy-safe durable event binding."""

        self._require_nonempty("agent_id", agent_id)
        self._require_nonempty("event_id", event_id)
        row = await self._backend.fetch_one(
            f"""
            SELECT integrity.integrity_binding
            FROM {self.EVENT_INTEGRITY} integrity
            JOIN {self.EVENTS} event ON event.event_id = integrity.event_id
            WHERE integrity.event_id = ? AND event.agent_id = ?
            """,
            (event_id, agent_id),
        )
        return str(row[0]) if row is not None else None

    async def list_deliveries(
        self,
        *,
        agent_id: str,
        consumer_id: Optional[str] = None,
        statuses: Optional[Iterable[str]] = None,
        limit: int = 100,
    ) -> list[DurableDelivery]:
        """List observable delivery state within one agent/tenant only."""
        self._require_nonempty("agent_id", agent_id)
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        where = ["d.agent_id = ?"]
        params: list[Any] = [agent_id]
        if consumer_id is not None:
            where.append("d.consumer_id = ?")
            params.append(consumer_id)
        if statuses is not None:
            wanted = tuple(statuses)
            valid = (
                _CLAIMABLE_STATUSES
                | {INITIAL_RESERVED, LEASED}
                | _TERMINAL_STATUSES
            )
            if not wanted or any(status not in valid for status in wanted):
                raise ValueError("statuses must be durable delivery states")
            where.append("d.status IN (" + ", ".join("?" for _ in wanted) + ")")
            params.extend(wanted)
        rows = await self._backend.fetch_all(
            self._delivery_select_sql(" AND ".join(where))
            + " ORDER BY d.created_at, d.delivery_id LIMIT ?",
            tuple(params + [limit]),
        )
        return [self._row_to_delivery(row) for row in rows]

    async def purge_expired(
        self, *, agent_id: str, now: Optional[datetime] = None
    ) -> int:
        """Delete only this agent's retained, terminal event histories.

        Pending, retriable, and leased work is never cleaned up by retention;
        operators must first resolve it to an observable terminal state.
        ``agent_id`` is mandatory because the periodic sweep is owned by one
        dispatcher even when multiple agents share a PostgreSQL database.
        """
        self._require_nonempty("agent_id", agent_id)
        now = _as_utc(now or self.now_utc())
        return await self._backend.execute(
            f"""
            DELETE FROM {self.EVENTS}
            WHERE agent_id = ?
              AND retention_until < ?
              AND NOT EXISTS (
                  SELECT 1 FROM {self.DELIVERIES} d
                  WHERE d.event_id = {self.EVENTS}.event_id
                    AND d.status NOT IN ('{ACKNOWLEDGED}', '{FAILED}', '{TERMINAL_ACKABLE}')
              )
            """,
            (agent_id, self.to_timestamp_param(now)),
        )

    # ------------------------------------------------------------------
    # Internal storage helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_nonempty(name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    def _validate_registration(self, registration: DurableConsumerRegistration) -> None:
        self._require_nonempty("consumer_id", registration.consumer_id)
        self._require_nonempty("source", registration.source)
        self._require_nonempty("agent_id", registration.agent_id)
        if (
            not isinstance(registration.max_attempts, int)
            or isinstance(registration.max_attempts, bool)
            or registration.max_attempts < 0
        ):
            raise ValueError("max_attempts must be >= 0")
        if (
            not isinstance(registration.lease_seconds, int)
            or isinstance(registration.lease_seconds, bool)
            or registration.lease_seconds < 1
        ):
            raise ValueError("lease_seconds must be >= 1")
        if registration.correlation_selector is not None:
            if (
                not isinstance(registration.correlation_selector, str)
                or not _SELECTOR_KEY.match(registration.correlation_selector)
            ):
                raise ValueError(
                    "correlation_selector must be 'payload.<path>=<value>', "
                    "'session_id=<value>', or 'kind=<value>'"
                )

    @staticmethod
    def _normalize_source_event_id(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("source_event_id must be a non-empty string when set")
        return value.strip()

    async def _find_existing_event_locked(
        self, agent_id: str, signal: Signal, source_event_id: Optional[str]
    ) -> str:
        existing = await self._existing_event_id_locked(
            agent_id, signal, source_event_id
        )
        if existing is None:
            raise RuntimeError("durable event insert conflicted without an existing event")
        return existing

    async def _existing_event_id_locked(
        self, agent_id: str, signal: Signal, source_event_id: Optional[str]
    ) -> Optional[str]:
        if source_event_id is not None:
            row = await self._backend.fetch_one(
                f"""
                SELECT event_id FROM {self.EVENTS}
                WHERE agent_id = ? AND source = ? AND source_event_id = ?
                """,
                (agent_id, signal.source, source_event_id),
            )
            if row is not None:
                return str(row[0])
        row = await self._backend.fetch_one(
            f"SELECT event_id FROM {self.EVENTS} WHERE event_id = ?",
            (signal.id,),
        )
        return str(row[0]) if row is not None else None

    async def _event_source_sequence_locked(
        self, *, event_id: str, agent_id: str, source: str
    ) -> int:
        """Return a previously committed event's sequence in this scope."""

        row = await self._backend.fetch_one(
            f"SELECT source_sequence FROM {self.EVENTS} "
            "WHERE event_id = ? AND agent_id = ? AND source = ?",
            (event_id, agent_id, source),
        )
        if row is None:
            raise RuntimeError(
                "durable event insert conflicted with a different agent/source scope"
            )
        if row[0] is None:
            raise RuntimeError("durable event is missing its committed source sequence")
        sequence = int(row[0])
        if sequence < 1 or sequence > _MAX_SOURCE_SEQUENCE:
            raise RuntimeError("durable event source sequence is out of range")
        return sequence

    @staticmethod
    def _legacy_event_matches_redelivery(
        event: DurableSignalEvent,
        signal: Signal,
        source_event_id: str,
    ) -> bool:
        """Compare the retained canonical envelope, excluding fresh attempt IDs.

        A provider retry receives a new signal/outcome ID and causation frame,
        neither of which was stable in the pre-consumer ledger.  Every source
        identity and normalized payload field that *was* retained must match.
        """

        return (
            event.source_event_id == source_event_id
            and event.target_agent == signal.target_agent
            and event.source == signal.source
            and event.kind == signal.kind
            and event.mode == signal.mode.value
            and event.payload == signal.payload
            and event.session_id == signal.session_id
            and event.visibility == signal.visibility.value
            and event.urgency == signal.urgency.value
            and event.dedupe_key == signal.dedupe_key
        )

    async def _lock_scope_handoff(self, *, agent_id: str, source: str) -> None:
        """Serialize registration and persistence for one subscription scope.

        PostgreSQL transactions otherwise use independent snapshots: a new
        registration could backfill before a concurrent event commits, while
        that event's consumer query runs before the registration commits.
        ``pg_advisory_xact_lock`` keeps those two atomic handoff paths in one
        order and releases automatically on commit or rollback.  SQLite
        transactions begin deferred, so a no-op write reserves its one writer
        slot before either handoff path can read stale state.  This matters
        across backend instances: each instance's in-memory write lock is
        local to that instance.
        """
        if self.is_postgres:
            await self._backend.fetch_val(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                (f"durable-signal:{agent_id}:{source}",),
            )
            return
        if self._backend.backend_type == "sqlite":
            await self._backend.execute(f"DELETE FROM {self.CONSUMERS} WHERE 0")
            return
        raise RuntimeError(
            "Durable signal handoff serialization does not support backend "
            f"{self._backend.backend_type!r}"
        )

    async def _source_sequence_locked(
        self,
        *,
        agent_id: str,
        source: str,
        allow_retained_reconstruction: bool = False,
    ) -> int:
        """Lock, validate, and if necessary repair one scope's counter.

        Normal callers must already own ``_lock_scope_handoff`` in the current
        transaction. Schema backfill takes that same handoff first on
        PostgreSQL and owns SQLite's writer reservation. PostgreSQL deliberately
        acquires ``FOR UPDATE`` only after the potentially blocking advisory
        lock, so capture samples the committed state that won the handoff rather
        than a pre-wait snapshot.

        The retained maximum is a descending lookup through the unique scope
        index, not a history scan. Normal capture/ingress use it only as a
        consistency lower bound: if it exceeds every exact ledger, arbitrary
        retention means the real high-water cannot be reconstructed and the
        operation fails closed. Only the additive legacy migration, before any
        source boundary could exist, may set ``allow_retained_reconstruction``
        and adopt that maximum. The independently retained recovery watermark
        and high-water ledger otherwise supply the exact value when another
        row/table is missing, even if newer history was already purged. An
        immutable marker distinguishes a genuinely fresh scope from a used
        scope after exact-row loss.
        """

        await self._backend.execute(
            f"""
            INSERT OR IGNORE INTO {self.SOURCE_SEQUENCES} (
                agent_id, source, current_sequence
            ) VALUES (?, ?, 0)
            """,
            (agent_id, source),
        )
        lock_clause = " FOR UPDATE" if self.is_postgres else ""
        row = await self._backend.fetch_one(
            f"SELECT current_sequence FROM {self.SOURCE_SEQUENCES} "
            f"WHERE agent_id = ? AND source = ?{lock_clause}",
            (agent_id, source),
        )
        if row is None:
            raise RuntimeError("Durable signal source sequence row disappeared")
        sequence = int(row[0])
        if sequence < 0 or sequence > _MAX_SOURCE_SEQUENCE:
            raise RuntimeError("Durable signal source sequence is out of range")

        await self._backend.execute(
            f"""
            INSERT OR IGNORE INTO {self.SOURCE_SEQUENCE_RECOVERY} (
                agent_id, source, recovery_sequence
            ) VALUES (?, ?, 0)
            """,
            (agent_id, source),
        )
        recovery_row = await self._backend.fetch_one(
            f"SELECT recovery_sequence FROM {self.SOURCE_SEQUENCE_RECOVERY} "
            f"WHERE agent_id = ? AND source = ?{lock_clause}",
            (agent_id, source),
        )
        if recovery_row is None:
            raise RuntimeError(
                "Durable signal source recovery watermark disappeared"
            )
        recovery_sequence = int(recovery_row[0])
        if recovery_sequence < 0 or recovery_sequence > _MAX_SOURCE_SEQUENCE:
            raise RuntimeError(
                "Durable signal source recovery watermark is out of range"
            )

        await self._backend.execute(
            f"""
            INSERT OR IGNORE INTO {self.SOURCE_SEQUENCE_HIGH_WATER} (
                agent_id, source, high_water_sequence
            ) VALUES (?, ?, 0)
            """,
            (agent_id, source),
        )
        high_water_row = await self._backend.fetch_one(
            f"SELECT high_water_sequence FROM {self.SOURCE_SEQUENCE_HIGH_WATER} "
            f"WHERE agent_id = ? AND source = ?{lock_clause}",
            (agent_id, source),
        )
        if high_water_row is None:
            raise RuntimeError("Durable signal source high-water row disappeared")
        high_water_sequence = int(high_water_row[0])
        if high_water_sequence < 0 or high_water_sequence > _MAX_SOURCE_SEQUENCE:
            raise RuntimeError("Durable signal source high-water is out of range")

        scope_was_seen = bool(
            await self._backend.fetch_val(
                f"SELECT 1 FROM {self.SOURCE_SEQUENCE_SEEN} "
                "WHERE agent_id = ? AND source = ?",
                (agent_id, source),
            )
        )
        if (
            scope_was_seen
            and recovery_sequence < 1
            and high_water_sequence < 1
        ):
            raise RuntimeError(
                "Durable signal source counter cannot be reconstructed: "
                f"{_SOURCE_SEQUENCE_LOSS_ERROR}"
            )

        retained_maximum = await self._backend.fetch_val(
            f"SELECT source_sequence FROM {self.EVENTS} "
            "WHERE agent_id = ? AND source = ? "
            "AND source_sequence IS NOT NULL "
            "ORDER BY source_sequence DESC LIMIT 1",
            (agent_id, source),
        )
        retained_sequence = 0
        if retained_maximum is not None:
            retained_sequence = int(retained_maximum)
            if not 1 <= retained_sequence <= _MAX_SOURCE_SEQUENCE:
                raise RuntimeError(
                    "Durable signal event source sequence is out of range"
                )

        exact_sequence = max(
            sequence,
            recovery_sequence,
            high_water_sequence,
        )
        if (
            retained_sequence > exact_sequence
            and not allow_retained_reconstruction
        ):
            raise RuntimeError(
                "Durable signal source counter cannot be reconstructed exactly: "
                "retained history exceeds every independent high-water"
            )

        repaired_sequence = max(
            exact_sequence,
            retained_sequence,
        )
        if (
            repaired_sequence != sequence
            or repaired_sequence != recovery_sequence
            or repaired_sequence != high_water_sequence
        ):
            await self._set_source_sequence_locked(
                agent_id=agent_id,
                source=source,
                sequence=repaired_sequence,
            )
        elif repaired_sequence > 0 and not scope_was_seen:
            # Equal surviving exact copies are already the fast-path value,
            # but they are also positive proof that this scope was used. Keep
            # the independent loss detector self-healing even when no counter
            # value itself needs repair.
            await self._mark_source_sequence_seen_locked(
                agent_id=agent_id, source=source
            )
        return repaired_sequence

    async def _mark_source_sequence_seen_locked(
        self, *, agent_id: str, source: str
    ) -> None:
        """Persist immutable evidence that an exact scope served a sequence."""

        await self._backend.execute(
            f"INSERT OR IGNORE INTO {self.SOURCE_SEQUENCE_SEEN} "
            "(agent_id, source) VALUES (?, ?)",
            (agent_id, source),
        )

    async def _set_source_sequence_locked(
        self, *, agent_id: str, source: str, sequence: int
    ) -> None:
        """Persist every nondecreasing exact copy while scope rows are locked."""

        if sequence < 0 or sequence > _MAX_SOURCE_SEQUENCE:
            raise ValueError("source sequence is out of range")
        updated = await self._backend.execute(
            f"""
            UPDATE {self.SOURCE_SEQUENCES}
            SET current_sequence = ?
            WHERE agent_id = ? AND source = ? AND current_sequence <= ?
            """,
            (sequence, agent_id, source, sequence),
        )
        if updated != 1:
            raise RuntimeError("Durable signal source sequence moved backwards")
        recovery_updated = await self._backend.execute(
            f"""
            UPDATE {self.SOURCE_SEQUENCE_RECOVERY}
            SET recovery_sequence = ?
            WHERE agent_id = ? AND source = ? AND recovery_sequence <= ?
            """,
            (sequence, agent_id, source, sequence),
        )
        if recovery_updated != 1:
            raise RuntimeError(
                "Durable signal source recovery watermark moved backwards"
            )
        high_water_updated = await self._backend.execute(
            f"""
            UPDATE {self.SOURCE_SEQUENCE_HIGH_WATER}
            SET high_water_sequence = ?
            WHERE agent_id = ? AND source = ? AND high_water_sequence <= ?
            """,
            (sequence, agent_id, source, sequence),
        )
        if high_water_updated != 1:
            raise RuntimeError("Durable signal source high-water moved backwards")
        if sequence > 0:
            await self._mark_source_sequence_seen_locked(
                agent_id=agent_id, source=source
            )

    async def _advance_source_sequence_locked(
        self, *, agent_id: str, source: str
    ) -> int:
        """Advance and return one scope's sequence in the owning transaction."""

        current = await self._source_sequence_locked(agent_id=agent_id, source=source)
        if current >= _MAX_SOURCE_SEQUENCE:
            raise OverflowError("Durable signal source sequence exhausted")
        sequence = current + 1
        await self._set_source_sequence_locked(
            agent_id=agent_id, source=source, sequence=sequence
        )
        return sequence

    async def _lock_runtime_owner_scope(self, *, agent_id: str) -> None:
        """Serialize liveness heartbeats and recovery for one tenant.

        The durable owner row is read by recovery predicates but updated by a
        separate heartbeat transaction.  PostgreSQL's statement snapshots do
        not make that read/update pair mutually exclusive on their own, so
        both paths take one transaction-scoped advisory key.  SQLite reserves
        its single writer before either path inspects owner liveness.  This is
        intentionally tenant-wide: recovery can assess several dispatcher
        generations in one statement, and a per-owner lock would leave the
        predicate race open for every other owner it scans.
        """
        if self.is_postgres:
            await self._backend.fetch_val(
                "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
                (f"durable-signal-runtime-owner:{agent_id}",),
            )
            return
        if self._backend.backend_type == "sqlite":
            await self._backend.execute(
                f"UPDATE {self.RUNTIME_OWNERS} SET updated_at = updated_at WHERE agent_id = ?",
                (agent_id,),
            )
            return
        raise RuntimeError(
            "Durable runtime-owner serialization does not support backend "
            f"{self._backend.backend_type!r}"
        )

    async def _lock_initial_delivery_transfer(
        self, *, agent_id: str, consumer_id: str, delivery_id: str
    ) -> Optional[tuple[Any, ...]]:
        """Lock an activated initial delivery before starting its worker lease.

        This deliberately uses the narrow delivery row rather than the
        registration/persistence scope lock: all we need here is stable
        ownership and a current lease deadline for one post-commit handoff.
        """
        params = (agent_id, consumer_id, delivery_id)
        if self.is_postgres:
            return await self._backend.fetch_one(
                f"""
                SELECT status, lease_owner, lease_token, lease_expires_at
                FROM {self.DELIVERIES}
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                FOR UPDATE
                """,
                params,
            )
        if self._backend.backend_type == "sqlite":
            # ``BEGIN`` is deferred in SQLite. Updating the exact delivery to
            # its current value claims the writer slot before the following
            # read/time sample, while preserving every persisted field.
            await self._backend.execute(
                f"""
                UPDATE {self.DELIVERIES}
                SET updated_at = updated_at
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                """,
                params,
            )
            return await self._backend.fetch_one(
                f"""
                SELECT status, lease_owner, lease_token, lease_expires_at
                FROM {self.DELIVERIES}
                WHERE agent_id = ? AND consumer_id = ? AND delivery_id = ?
                """,
                params,
            )
        raise RuntimeError(
            "Durable signal initial-delivery transfer does not support backend "
            f"{self._backend.backend_type!r}"
        )

    async def _get_consumer(
        self, agent_id: str, consumer_id: str
    ) -> Optional[tuple[Any, ...]]:
        return await self._backend.fetch_one(
            f"""
            SELECT source, correlation_selector, max_attempts, lease_seconds, active
            FROM {self.CONSUMERS}
            WHERE agent_id = ? AND consumer_id = ?
            """,
            (agent_id, consumer_id),
        )

    async def _touch_runtime_owner_locked(
        self, *, agent_id: str, owner_id: str, now: datetime
    ) -> None:
        """Upsert an active owner while the caller holds a transaction.

        A delayed heartbeat can finish after a newer activation or heartbeat.
        Preserve the newest liveness evidence rather than regressing it and
        allowing another dispatcher to misclassify this owner as stale.
        """
        updated = await self._backend.execute(
            f"""
            UPDATE {self.RUNTIME_OWNERS}
            SET heartbeat_at = CASE
                    WHEN heartbeat_at >= ? THEN heartbeat_at ELSE ? END,
                stopped_at = NULL,
                updated_at = CASE
                    WHEN heartbeat_at >= ? THEN heartbeat_at ELSE ? END
            WHERE agent_id = ? AND owner_id = ?
            """,
            (
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                owner_id,
            ),
        )
        if updated == 0:
            await self._backend.execute(
                f"""
                INSERT OR IGNORE INTO {self.RUNTIME_OWNERS} (
                    agent_id, owner_id, heartbeat_at, stopped_at, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, ?, ?)
                """,
                (
                    agent_id,
                    owner_id,
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                    self.to_timestamp_param(now),
                ),
            )

    async def _recover_expired_leases(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        now: datetime,
        runtime_owner_stale_before: datetime,
    ) -> None:
        """Requeue expired work without stealing from a live dispatcher.

        Leases owned by public or unknown executors retain ordinary expiry
        behavior because there is no durable liveness contract for them.  A
        ``dispatcher:`` owner is different: its heartbeat proves that an
        in-process cognition task may still be draining a cancellation.  Such
        a row is recoverable only once that managed owner is explicitly
        stopped or its heartbeat is stale.
        """

        # Claim/recovery calls this inside their existing transaction.  Take
        # the same tenant liveness scope as heartbeat before evaluating the
        # managed-owner predicate.
        await self._lock_runtime_owner_scope(agent_id=agent_id)
        timestamp = self._timestamp_placeholder()
        await self._backend.execute(
            f"""
            UPDATE {self.DELIVERIES}
            SET status = CASE WHEN max_attempts > 0 AND attempts >= max_attempts THEN ? ELSE ? END,
                lease_owner = NULL, lease_token = NULL, lease_expires_at = NULL,
                next_attempt_at = CASE WHEN max_attempts > 0 AND attempts >= max_attempts
                    THEN NULL ELSE {timestamp} END,
                last_error = 'lease expired before acknowledgement',
                terminal_at = CASE WHEN max_attempts > 0 AND attempts >= max_attempts
                    THEN {timestamp} ELSE NULL END,
                updated_at = ?
            WHERE agent_id = ? AND consumer_id = ? AND status = ?
              AND lease_expires_at <= ?
              AND (
                  lease_owner IS NULL
                  OR lease_owner NOT LIKE 'dispatcher:%'
                  OR NOT EXISTS (
                      SELECT 1 FROM {self.RUNTIME_OWNERS} owner
                      WHERE owner.agent_id = {self.DELIVERIES}.agent_id
                        AND owner.owner_id = {self.DELIVERIES}.lease_owner
                        AND owner.stopped_at IS NULL
                        AND owner.heartbeat_at >= ?
                  )
              )
            """,
            (
                FAILED,
                RETRY,
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
                agent_id,
                consumer_id,
                LEASED,
                self.to_timestamp_param(now),
                self.to_timestamp_param(runtime_owner_stale_before),
            ),
        )

    def _timestamp_placeholder(self) -> str:
        """Return a timestamp parameter expression for the active dialect.

        PostgreSQL cannot infer the type of a bind value in a ``CASE`` arm
        whose other arm is ``NULL``.  Without the cast asyncpg treats it as
        text and rejects the update against a ``TIMESTAMPTZ`` column.  SQLite
        stores timestamps as text, so its ordinary placeholder is correct.
        """
        return "?::TIMESTAMPTZ" if self.is_postgres else "?"

    async def _backfill_consumer(
        self, registration: DurableConsumerRegistration, *, now: datetime
    ) -> None:
        rows = await self._backend.fetch_all(
            f"""
            SELECT event_id, source_event_id, agent_id, target_agent, source, kind, mode, payload,
                   session_id, caller_identity, visibility, urgency, dedupe_key,
                   causation_chain, arrived_at, committed_at, retention_until,
                   source_sequence
            FROM {self.EVENTS}
            WHERE agent_id = ? AND source = ? AND retention_until >= ?
            """,
            (
                registration.agent_id,
                registration.source,
                self.to_timestamp_param(now),
            ),
        )
        for row in rows:
            event = self._row_to_event(row)
            if self._matches_selector(event, registration.correlation_selector):
                await self._insert_delivery_locked(
                    agent_id=registration.agent_id,
                    consumer_id=registration.consumer_id,
                    event_id=event.event_id,
                    max_attempts=registration.max_attempts,
                    now=now,
                )

    async def _insert_delivery_locked(
        self,
        *,
        agent_id: str,
        consumer_id: str,
        event_id: str,
        max_attempts: int,
        now: datetime,
        initial_reservation_owner: Optional[str] = None,
        initial_reservation_token: Optional[str] = None,
    ) -> Optional[str]:
        initial_reservation = initial_reservation_owner is not None
        if initial_reservation != (initial_reservation_token is not None):
            raise ValueError(
                "initial reservation owner and token must be set together"
            )
        delivery_id = secrets.token_urlsafe(18)
        inserted = await self._backend.execute(
            f"""
            INSERT OR IGNORE INTO {self.DELIVERIES} (
                delivery_id, agent_id, consumer_id, event_id, status,
                attempts, max_attempts, lease_owner, lease_token,
                lease_expires_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                delivery_id,
                agent_id,
                consumer_id,
                event_id,
                INITIAL_RESERVED if initial_reservation else PENDING,
                max_attempts,
                initial_reservation_owner,
                initial_reservation_token,
                None,
                self.to_timestamp_param(now),
                self.to_timestamp_param(now),
            ),
        )
        return delivery_id if inserted == 1 else None

    def _event_from_signal(
        self,
        signal: Signal,
        *,
        agent_id: str,
        source_event_id: Optional[str],
        caller_identity: Optional[str],
        committed_at: datetime,
        retention_until: datetime,
        source_sequence: int,
    ) -> DurableSignalEvent:
        return DurableSignalEvent(
            event_id=signal.id,
            source_event_id=source_event_id,
            agent_id=agent_id,
            target_agent=signal.target_agent,
            source=signal.source,
            kind=signal.kind,
            mode=signal.mode.value,
            payload=signal.payload,
            session_id=signal.session_id,
            caller_identity=caller_identity,
            visibility=signal.visibility.value,
            urgency=signal.urgency.value,
            dedupe_key=signal.dedupe_key,
            causation_chain=_serialize_chain(signal.causation_chain),
            arrived_at=_as_utc(signal.arrived_at),
            committed_at=committed_at,
            retention_until=retention_until,
            source_sequence=source_sequence,
        )

    def _row_to_event(self, row: tuple[Any, ...]) -> DurableSignalEvent:
        return DurableSignalEvent(
            event_id=row[0],
            source_event_id=row[1],
            agent_id=row[2],
            target_agent=row[3],
            source=row[4],
            kind=row[5],
            mode=row[6],
            payload=_json_load(row[7]),
            session_id=row[8],
            caller_identity=row[9],
            visibility=row[10],
            urgency=row[11],
            dedupe_key=row[12],
            causation_chain=_json_load(row[13]),
            arrived_at=_as_utc(self.from_timestamp_field(row[14])),
            committed_at=_as_utc(self.from_timestamp_field(row[15])),
            retention_until=_as_utc(self.from_timestamp_field(row[16])),
            source_sequence=int(row[17]),
        )

    @staticmethod
    def _matches_selector(
        event: DurableSignalEvent, selector: Optional[str]
    ) -> bool:
        if selector is None:
            return True
        match = _SELECTOR_KEY.match(selector)
        if match is None:  # registrations are validated; keep this fail-closed.
            return False
        left, expected = selector.split("=", 1)
        if left == "session_id":
            actual = event.session_id
        elif left == "kind":
            actual = event.kind
        else:
            actual: Any = event.payload
            for key in left.removeprefix("payload.").split("."):
                if not isinstance(actual, dict) or key not in actual:
                    return False
                actual = actual[key]
        if isinstance(actual, (dict, list)) or actual is None:
            return False
        return str(actual) == expected

    def _delivery_select_sql(self, where: str) -> str:
        return f"""
            SELECT
                d.delivery_id, d.consumer_id, d.agent_id, d.event_id, d.status,
                d.attempts, d.max_attempts, d.lease_owner, d.lease_token,
                d.lease_expires_at, d.next_attempt_at, d.last_error,
                d.acknowledged_at, d.terminal_at, d.created_at, d.updated_at,
                e.event_id, e.source_event_id, e.agent_id, e.target_agent,
                e.source, e.kind, e.mode, e.payload, e.session_id,
                e.caller_identity, e.visibility, e.urgency, e.dedupe_key,
                e.causation_chain,
                e.arrived_at, e.committed_at, e.retention_until,
                e.source_sequence
            FROM {self.DELIVERIES} d
            JOIN {self.EVENTS} e ON e.event_id = d.event_id
            WHERE {where}
        """

    async def _delivery_for_lease_locked(
        self, *, agent_id: str, consumer_id: str, lease_token: str
    ) -> DurableDelivery:
        row = await self._backend.fetch_one(
            self._delivery_select_sql(
                "d.agent_id = ? AND d.consumer_id = ? AND d.status = ? "
                "AND d.lease_token = ?"
            ),
            (agent_id, consumer_id, LEASED, lease_token),
        )
        if row is None:
            raise RuntimeError("claimed durable delivery disappeared before handoff")
        return self._row_to_delivery(row)

    def _row_to_delivery(self, row: tuple[Any, ...]) -> DurableDelivery:
        event = self._row_to_event(row[16:])
        return DurableDelivery(
            delivery_id=row[0],
            consumer_id=row[1],
            agent_id=row[2],
            event_id=row[3],
            status=row[4],
            attempts=int(row[5]),
            max_attempts=int(row[6]),
            lease_owner=row[7],
            lease_token=row[8],
            lease_expires_at=(
                _as_utc(self.from_timestamp_field(row[9])) if row[9] is not None else None
            ),
            next_attempt_at=(
                _as_utc(self.from_timestamp_field(row[10])) if row[10] is not None else None
            ),
            last_error=row[11],
            acknowledged_at=(
                _as_utc(self.from_timestamp_field(row[12])) if row[12] is not None else None
            ),
            terminal_at=(
                _as_utc(self.from_timestamp_field(row[13])) if row[13] is not None else None
            ),
            created_at=_as_utc(self.from_timestamp_field(row[14])),
            updated_at=_as_utc(self.from_timestamp_field(row[15])),
            event=event,
        )


__all__ = [
    "ACKNOWLEDGED",
    "FAILED",
    "INITIAL_RESERVED",
    "LEASED",
    "PENDING",
    "RETRY",
    "TERMINAL_ACKABLE",
    "DurableConsumerRegistration",
    "DurableDelivery",
    "DurableEventPersistence",
    "DurableInitialDeliveryReservation",
    "DurableSignalEvent",
    "DurableSourceBoundary",
    "DurableSignalStore",
]
