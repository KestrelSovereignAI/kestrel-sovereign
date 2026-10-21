"""Explicit offline A2A data cutover; never imported by the serving runtime."""

import base64
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import UUID, uuid4

from a2a.types import Task

from kestrel_sovereign.a2a.lifecycle import task_state_label
from kestrel_sovereign.a2a.migrate_legacy import (
    LegacyTaskMigrationError,
    convert_legacy_task,
)
from kestrel_sovereign.a2a.native_schema import (
    native_schema_write,
    publish_native_a2a_schema,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.context_store import (
    ContextOwnership,
    ContextStore,
)
from kestrel_sovereign.a2a.stores.unified.task_store import TaskStore, _task_payload
from kestrel_sovereign.signals.durable import DurableSignalStore, DurableSourceBoundary
from kestrel_sovereign.signals.sources.a2a import SOURCE_NAME as COMPLETE_SOURCE
from kestrel_sovereign.signals.sources.a2a_question_answered import (
    SOURCE_NAME as QUESTION_SOURCE,
)
from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    SOURCE_NAME as INPUT_SOURCE,
)

MIGRATION_PLAN_REVISION = 2

_TABLES = ("a2a_tasks", "a2a_outbound_tasks", "pending_a2a_questions")
_SIGNAL_SOURCES = (INPUT_SOURCE, COMPLETE_SOURCE, QUESTION_SOURCE)
_SIGNAL_HISTORY_TABLES = (
    (DurableSignalStore.EVENTS, "signal_events"),
    (DurableSignalStore.DELIVERIES, "signal_deliveries"),
    (DurableSignalStore.EVENT_INTEGRITY, "signal_integrity"),
    (DurableSignalStore.INGRESS_TRUST_RECEIPTS, "signal_ingress"),
)


def _source_selection(name):
    """Use one source predicate for reviewed snapshots and exact archives."""
    if name not in dict(_SIGNAL_HISTORY_TABLES):
        return "", ()
    placeholders = ", ".join("?" for _ in _SIGNAL_SOURCES)
    source_clause = f"source IN ({placeholders})"
    if name == DurableSignalStore.EVENTS:
        return f" WHERE {source_clause}", _SIGNAL_SOURCES
    return (
        f" WHERE event_id IN (SELECT event_id FROM {DurableSignalStore.EVENTS} WHERE {source_clause})",
        _SIGNAL_SOURCES,
    )


def _archive_name(migration_id, table_name):
    if table_name in dict(_SIGNAL_HISTORY_TABLES):
        suffix = dict(_SIGNAL_HISTORY_TABLES)[table_name]
    else:
        suffix = table_name.removeprefix("a2a_").replace(
            "pending_a2a_questions", "questions"
        )
    return f"a2a_legacy_{migration_id}_{suffix}"


_TERMINAL = frozenset({"completed", "failed", "canceled", "rejected"})
_QUESTION_ROUTE_FIELDS = ("agent_id", "task_id", "recipient_agent_id")
# Only Core's known task fences are suspended, inside the exclusive transaction.
_TASK_TRIGGERS = frozenset(
    {
        "a2a_tasks_canceled_terminal_v1",
        "a2a_tasks_canceled_terminal_v2",
        "a2a_tasks_canceled_replace_v3",
        "a2a_tasks_terminal_update_v3",
        "a2a_tasks_terminal_replace_v3",
        "a2a_tasks_terminal_replace_v4",
        "a2a_tasks_authority_insert_v3",
        "a2a_tasks_live_authority_update_v2",
        "a2a_tasks_live_authority_v2",
        "a2a_tasks_authority_fence_v2",
        "a2a_tasks_authority_fence_v3",
        "a2a_tasks_authority_fence_v4",
        "a2a_tasks_authority_fence_v5",
        "a2a_tasks_terminal_update_v5",
        "a2a_tasks_terminal_replace_v5",
        "a2a_tasks_authority_insert_v5",
        "a2a_tasks_live_authority_update_v5",
    }
)


def _ident(value):
    return '"' + value.replace('"', '""') + '"'


def _typed(value):
    """Fingerprint SQL values without losing type, timestamps, or binary data."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float) and math.isfinite(value):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, (tuple, list)):
        return ["array", [_typed(item) for item in value]]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return ["object", [[key, _typed(value[key])] for key in sorted(value)]]
    raise LegacyTaskMigrationError(
        "Legacy SQL data has an unsupported or non-finite value"
    )


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class LegacyTable:
    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple, ...]
    catalog: tuple[tuple, ...]

    def dictionaries(self):
        return [dict(zip(self.columns, row, strict=True)) for row in self.rows]


@dataclass(frozen=True)
class MigratedTask:
    task_id: str
    ownership: ContextOwnership
    native_bytes: bytes


@dataclass(frozen=True)
class LegacyMigrationPlan:
    migration_id: str
    source_digest: str
    tables: tuple[LegacyTable, ...]
    tasks: tuple[MigratedTask, ...]
    signal_boundaries: tuple[DurableSourceBoundary, ...] = ()

    def review(self):
        """A concrete review artifact, with no database credentials or message text."""
        return {
            "revision": MIGRATION_PLAN_REVISION,
            "migration_id": self.migration_id,
            "source_digest": self.source_digest,
            "tables": {table.name: len(table.rows) for table in self.tables},
            "retired_signal_histories": [
                boundary.to_dict() for boundary in self.signal_boundaries
            ],
            "tasks": [
                {
                    "task_id": item.task_id,
                    "ownership": asdict(item.ownership),
                    "native_sha256": hashlib.sha256(item.native_bytes).hexdigest(),
                }
                for item in self.tasks
            ],
        }


async def _table(backend, name):
    if backend.backend_type == "sqlite":
        columns = tuple(
            row[1]
            for row in await backend.fetch_all(f"PRAGMA table_info({_ident(name)})")
        )
        catalog = await backend.fetch_all(
            "SELECT type, name, sql FROM sqlite_master WHERE tbl_name = ? ORDER BY type, name",
            (name,),
        )
    else:
        columns = tuple(
            row[0]
            for row in await backend.fetch_all(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
                "AND table_name = ? ORDER BY ordinal_position",
                (name,),
            )
        )
        catalog = await backend.fetch_all(
            "SELECT 'column', column_name, data_type || ':' || is_nullable || ':' || COALESCE(column_default, '') "
            "FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = ? "
            "UNION ALL SELECT 'constraint', conname, pg_get_constraintdef(c.oid) FROM pg_constraint c "
            "JOIN pg_class t ON t.oid = c.conrelid JOIN pg_namespace n ON n.oid = t.relnamespace "
            "WHERE n.nspname = current_schema() AND t.relname = ? "
            "UNION ALL SELECT 'index', indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = current_schema() AND tablename = ? "
            "UNION ALL SELECT 'trigger', tr.tgname, pg_get_triggerdef(tr.oid) FROM pg_trigger tr "
            "JOIN pg_class t ON t.oid = tr.tgrelid JOIN pg_namespace n ON n.oid = t.relnamespace "
            "WHERE n.nspname = current_schema() AND t.relname = ? AND NOT tr.tgisinternal "
            "UNION ALL SELECT DISTINCT 'trigger-function', p.proname, pg_get_functiondef(p.oid) "
            "FROM pg_trigger tr JOIN pg_proc p ON p.oid = tr.tgfoid "
            "JOIN pg_class t ON t.oid = tr.tgrelid JOIN pg_namespace n ON n.oid = t.relnamespace "
            "WHERE n.nspname = current_schema() AND t.relname = ? AND NOT tr.tgisinternal ORDER BY 1, 2",
            (name, name, name, name, name),
        )
    where, parameters = _source_selection(name)
    rows = await backend.fetch_all(
        f"SELECT {', '.join(_ident(col) for col in columns)} FROM {_ident(name)}{where}",
        parameters,
    )
    return LegacyTable(
        name,
        columns,
        tuple(sorted(rows, key=lambda row: _encode(_typed(row)))),
        tuple(catalog),
    )


async def prepare_legacy_migration(
    backend, *, ownership: Mapping[str, ContextOwnership], migration_id=None
):
    """Validate a drained snapshot and explicit, reviewed task provenance."""
    if backend.backend_type not in {"sqlite", "postgres"}:
        raise ValueError("Offline A2A cutover supports SQLite and PostgreSQL")
    if await backend.table_exists("a2a_protocol_schema"):
        raise LegacyTaskMigrationError(
            "A2A data is already versioned; this is not a draft database"
        )
    for target in (
        "a2a_contexts",
        "a2a_outbound_dispatches",
        "a2a_native_questions",
        "a2a_native_question_wakes",
        "a2a_task_inputs",
        "a2a_native_store_schema",
    ):
        if await backend.table_exists(target) and await backend.fetch_one(
            f"SELECT 1 FROM {target} LIMIT 1"
        ):
            raise LegacyTaskMigrationError(
                "Native A2A data already exists without a protocol revision"
            )
    tables = tuple(
        [
            await _table(backend, name)
            for name in _TABLES
            if await backend.table_exists(name)
        ]
    )
    if await backend.table_exists(DurableSignalStore.EVENTS):
        tables += tuple(
            [
                await _table(backend, name)
                for name, _ in _SIGNAL_HISTORY_TABLES
                if await backend.table_exists(name)
            ]
        )
    records = {table.name: table.dictionaries() for table in tables}
    signal_events = records.get(DurableSignalStore.EVENTS, [])
    if signal_events and DurableSignalStore.DELIVERIES not in records:
        raise LegacyTaskMigrationError(
            "Durable signal storage is incomplete; finish its schema migration before cutover"
        )
    sequences = {}
    for event in signal_events:
        sequence = event.get("source_sequence")
        if type(sequence) is not int or sequence < 1:
            raise LegacyTaskMigrationError(
                "Durable A2A history has no committed source sequence; finish its storage migration before cutover"
            )
        key = (event["agent_id"], event["source"])
        sequences[key] = max(sequences.get(key, 0), sequence)
    try:
        signal_boundaries = tuple(
            DurableSourceBoundary(
                agent_id=agent, source=source, sequence=sequences[(agent, source)]
            )
            for agent, source in sorted(sequences)
        )
    except (TypeError, ValueError) as error:
        raise LegacyTaskMigrationError(
            "Durable A2A history has an invalid source boundary"
        ) from error
    signal_store = DurableSignalStore(backend)
    for boundary in signal_boundaries:
        if not await signal_store.source_history_is_drained(boundary):
            raise LegacyTaskMigrationError(
                "Owed durable A2A deliveries must drain before cutover"
            )
    tasks = records.get("a2a_tasks", [])
    if set(ownership) != {row["id"] for row in tasks}:
        raise LegacyTaskMigrationError(
            "Reviewed provenance must cover exactly every legacy task"
        )
    confirmed_expired_questions = set()
    for row in records.get("pending_a2a_questions", []):
        if (
            row.get("status") not in {"RESOLVED", "EXPIRED"}
            or row.get("retry_state") is not None
            or row.get("receiver_settlement") not in {None, "CONFIRMED"}
            or (
                row.get("status") == "EXPIRED"
                and row.get("receiver_settlement") != "CONFIRMED"
            )
            or row.get("receiver_late_state") is not None
        ):
            raise LegacyTaskMigrationError(
                "Unsettled draft questions or owed deliveries must drain before cutover"
            )
        route = tuple(row.get(field) for field in _QUESTION_ROUTE_FIELDS)
        if row.get("status") == "EXPIRED" and all(
            isinstance(value, str) and value for value in route
        ):
            confirmed_expired_questions.add(route)
    for row in records.get("a2a_outbound_tasks", []):
        if row.get("terminal_state") in _TERMINAL | {"dispatch_failed"}:
            continue
        route = tuple(row.get(field) for field in _QUESTION_ROUTE_FIELDS)
        if (
            row.get("terminal_state") == "expired"
            and row.get("verb") == "question"
            and route in confirmed_expired_questions
        ):
            continue
        raise LegacyTaskMigrationError(
            "Unsettled draft outbound work must drain before cutover"
        )
    converted = []
    for row in sorted(tasks, key=lambda row: row["id"]):
        decoded = dict(row)
        for field in ("message", "artifacts", "history", "metadata"):
            if isinstance(decoded.get(field), str):
                try:
                    decoded[field] = json.loads(decoded[field])
                except json.JSONDecodeError as error:
                    raise LegacyTaskMigrationError(
                        f"Legacy task has malformed JSON in {field}"
                    ) from error
        task = convert_legacy_task(decoded, ownership=ownership[row["id"]])
        converted.append(
            MigratedTask(
                task.id, ownership[task.id], task.SerializeToString(deterministic=True)
            )
        )
    context_creators = {}
    for item in converted:
        owner = item.ownership
        key = (owner.recipient_agent_id, owner.tenant, owner.context_id)
        creator = context_creators.setdefault(key, owner.creator_agent_id)
        if creator != owner.creator_agent_id:
            raise LegacyTaskMigrationError(
                "Reviewed tasks disagree on the creator owning one native context"
            )
    for table in tables:
        if table.name == "a2a_tasks" and any(
            kind == "trigger" and name not in _TASK_TRIGGERS
            for kind, name, _ in table.catalog
        ):
            raise LegacyTaskMigrationError(
                "Custom A2A task triggers require explicit operator schema review"
            )
    fingerprint = [
        [table.name, table.columns, _typed(table.rows), _typed(table.catalog)]
        for table in tables
    ]
    identifier = UUID(migration_id).hex if migration_id is not None else uuid4().hex
    return LegacyMigrationPlan(
        identifier,
        hashlib.sha256(_encode(fingerprint).encode()).hexdigest(),
        tables,
        tuple(converted),
        signal_boundaries,
    )


async def apply_legacy_migration(backend, plan: LegacyMigrationPlan):
    """Archive and convert atomically, after all old host writers have stopped.

    The source fingerprint is rechecked while holding database write locks.
    Historical tasks remain readable but cannot create fresh cognition wakes.
    Old outbound previews never become invented native delivery receipts.
    """
    if (
        not isinstance(plan, LegacyMigrationPlan)
        or UUID(plan.migration_id).hex != plan.migration_id
    ):
        raise TypeError("Apply requires an exact prepared legacy migration plan")
    async with native_schema_write(backend):
        signal_store = DurableSignalStore(backend)
        # Acquire source handoffs before relation locks, in the same order as
        # publishers/consumer registration, then fingerprint again under both.
        for boundary in plan.signal_boundaries:
            if (
                not isinstance(boundary, DurableSourceBoundary)
                or boundary.source not in _SIGNAL_SOURCES
            ):
                raise LegacyTaskMigrationError(
                    "Migration contains an unknown durable source boundary"
                )
            await signal_store._lock_scope_handoff(
                agent_id=boundary.agent_id, source=boundary.source
            )
        if backend.backend_type == "postgres":
            for table in plan.tables:
                if table.name not in (*_TABLES, *dict(_SIGNAL_HISTORY_TABLES)):
                    raise LegacyTaskMigrationError(
                        "Migration contains an unknown source table"
                    )
                await backend.execute(
                    f"LOCK TABLE {_ident(table.name)} IN ACCESS EXCLUSIVE MODE NOWAIT"
                )
        current = await prepare_legacy_migration(
            backend,
            ownership={item.task_id: item.ownership for item in plan.tasks},
            migration_id=plan.migration_id,
        )
        if current.review() != plan.review():
            raise LegacyTaskMigrationError(
                "Legacy data or reviewed conversion changed after preview; prepare a new plan"
            )
        await backend.execute_script("""
            CREATE TABLE IF NOT EXISTS a2a_legacy_migrations (
                migration_id TEXT PRIMARY KEY, source_digest TEXT NOT NULL,
                review_json TEXT NOT NULL, catalog_json TEXT NOT NULL
            )
        """)
        archives = {}
        for table in current.tables:
            archive = _archive_name(plan.migration_id, table.name)
            where, parameters = _source_selection(table.name)
            await backend.execute(
                f"CREATE TABLE {_ident(archive)} AS SELECT * FROM {_ident(table.name)}{where}",
                parameters,
            )
            archived = await backend.fetch_one(
                f"SELECT COUNT(*) FROM {_ident(archive)}"
            )
            if archived != (len(table.rows),):
                raise LegacyTaskMigrationError(
                    "Legacy archive did not retain every source row"
                )
            archives[table.name] = archive
        signal_events = next(
            (
                table.dictionaries()
                for table in current.tables
                if table.name == DurableSignalStore.EVENTS
            ),
            [],
        )
        for boundary in current.signal_boundaries:
            expected = sum(
                event["agent_id"] == boundary.agent_id
                and event["source"] == boundary.source
                and event["source_sequence"] <= boundary.sequence
                for event in signal_events
            )
            deleted = await signal_store.retire_source_history(boundary)
            if deleted != expected:
                raise LegacyTaskMigrationError(
                    "Durable A2A history changed during retirement"
                )
        if current.tasks:
            first = current.tasks[0].ownership
            tasks = TaskStore(
                backend,
                scope=NativeRecipientScope(first.recipient_agent_id, first.tenant),
                owns_backend=False,
            )
            await tasks._initialize_native_objects()
            catalog = (await _table(backend, "a2a_tasks")).catalog
            for kind, name, _ in catalog:
                if kind == "trigger":
                    if name not in _TASK_TRIGGERS:
                        raise LegacyTaskMigrationError(
                            "An unreviewed task trigger appeared during cutover"
                        )
                    suffix = (
                        " ON a2a_tasks" if backend.backend_type == "postgres" else ""
                    )
                    await backend.execute(f"DROP TRIGGER {_ident(name)}{suffix}")
            contexts = ContextStore(backend, owns_backend=False)
            await contexts.initialize()
            for item in current.tasks:
                task = Task.FromString(item.native_bytes)
                owner = item.ownership
                await contexts.reserve(
                    owner.context_id,
                    NativeRecipientScope(owner.recipient_agent_id, owner.tenant),
                    creator_agent_id=owner.creator_agent_id,
                )
                message, artifacts, history, metadata = _task_payload(task)
                changed = await backend.execute(
                    "UPDATE a2a_tasks SET session_id = ?, status = ?, message = ?, artifacts = ?, history = ?, metadata = ?, "
                    "creator_agent_id = ?, recipient_agent_id = ?, tenant = ?, status_timestamp_seconds = NULL, "
                    "status_timestamp_nanos = NULL, cognition_wake_eligible = 0 WHERE id = ?",
                    (
                        task.context_id,
                        task_state_label(task.status.state),
                        message,
                        artifacts,
                        history,
                        metadata,
                        owner.creator_agent_id,
                        owner.recipient_agent_id,
                        owner.tenant,
                        task.id,
                    ),
                )
                if changed != 1:
                    raise LegacyTaskMigrationError(
                        "A reviewed historical task disappeared during cutover"
                    )
            await tasks._install_canceled_terminal_fence()
        for table in current.tables:
            if table.name in _TABLES and table.name != "a2a_tasks":
                await backend.execute(f"DELETE FROM {_ident(table.name)}")
        await backend.execute(
            "INSERT INTO a2a_legacy_migrations (migration_id, source_digest, review_json, catalog_json) VALUES (?, ?, ?, ?)",
            (
                plan.migration_id,
                plan.source_digest,
                _encode(plan.review()),
                _encode(
                    {table.name: _typed(table.catalog) for table in current.tables}
                ),
            ),
        )
        await publish_native_a2a_schema(backend)
    return archives
