from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from kestrel_sovereign.hold import (
    HoldAction,
    HoldDisposition,
    HoldIdempotencyConflict,
    HoldReceipt,
    HoldScope,
    HoldStateError,
    HoldStore,
)
from kestrel_sovereign.hold.state import (
    HoldCorruptStateError,
    PostgresHoldCustodySnapshot,
    _latch_from_row,
    _receipt_from_row,
    _terminal_authority_ids,
    hold_history_anchor_path,
    hold_initialization_witness_path,
    initialize_postgres_hold_databases,
    preflight_postgres_hold_custody,
    postgres_hold_custody_binding_payload,
    validate_postgres_hold_custody,
)
from kestrel_sovereign.host_features.context import build_host_context
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture
async def hold_db(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
    store = HoldStore(db)
    await store.ensure_schema()
    try:
        yield db, store
    finally:
        await db.close()


async def _create_legacy_hold_tables(db) -> None:
    """Create the pre-witness Hold tables used by upgrade regressions."""

    await db.execute(
        "CREATE TABLE hold_latches ("
        "scope TEXT NOT NULL, target_id TEXT NOT NULL, "
        "active INTEGER NOT NULL DEFAULT 0, "
        "hold_receipt_id TEXT NOT NULL DEFAULT '', "
        "reason TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', "
        "set_at TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY (scope, target_id))"
    )
    await db.execute(
        "CREATE TABLE hold_receipts ("
        "receipt_id TEXT NOT NULL PRIMARY KEY, operation_id TEXT NOT NULL, "
        "action TEXT NOT NULL, disposition TEXT NOT NULL, scope TEXT NOT NULL, "
        "target_id TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', "
        "actor_id TEXT NOT NULL, occurred_at TEXT NOT NULL, "
        "expected_hold_receipt_id TEXT NOT NULL DEFAULT '', "
        "prior_hold_receipt_id TEXT NOT NULL DEFAULT '', "
        "resulting_hold_receipt_id TEXT NOT NULL DEFAULT '')"
    )


class _PostgresCustodyFacade:
    """Minimal two-domain PostgreSQL metadata/cluster double."""

    backend_type = "postgres"

    def __init__(self, cluster, *, domain_identity=None, backend=None):
        self.cluster = cluster
        self.metadata = {}
        if domain_identity is not None:
            self.metadata["hold_rollback_domain_id_v1"] = domain_identity
        self.backend = backend

    async def fetchall(self, query, params=()):
        if "pg_control_system" in query:
            return [(self.cluster,)]
        value = self.metadata.get(params[1])
        return [] if value is None else [(value,)]

    async def execute(self, query, params=()):
        if query.startswith("INSERT"):
            self.metadata.setdefault(params[1], params[2])
        elif query.startswith("DELETE"):
            self.metadata.pop(params[1], None)


@pytest.mark.asyncio
async def test_host_and_agent_holds_compose_and_release_independently(hold_db):
    _db, store = hold_db
    host = await store.set_hold(
        scope=HoldScope.HOST,
        actor_id="did:sovereign:operator",
        reason="fleet investigation",
        operation_id="hold-host-1",
    )
    agent = await store.set_hold(
        scope=HoldScope.AGENT,
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="agent investigation",
        operation_id="hold-agent-1",
    )

    effective = await store.get_effective("did:agent:kite")
    assert effective.held
    assert effective.sources == (HoldScope.HOST, HoldScope.AGENT)
    assert effective.host == host.current
    assert effective.agent == agent.current

    released = await store.release_hold(
        scope=HoldScope.HOST,
        actor_id="did:sovereign:operator",
        reason="fleet cleared",
        operation_id="release-host-1",
        expected_hold_receipt_id=host.receipt.receipt_id,
    )
    assert released.receipt.disposition is HoldDisposition.APPLIED
    assert released.current is None

    effective = await store.get_effective("did:agent:kite")
    assert effective.held
    assert effective.host is None
    assert effective.agent == agent.current


@pytest.mark.asyncio
async def test_effective_read_validates_global_history_once(hold_db, monkeypatch):
    """One effective snapshot pays for one complete history validation."""

    _db, store = hold_db
    validate_history = store._assert_global_history_intact
    history_validations = 0

    async def _count_history_validation() -> None:
        nonlocal history_validations
        history_validations += 1
        await validate_history()

    monkeypatch.setattr(
        store,
        "_assert_global_history_intact",
        _count_history_validation,
    )

    await store.get_effective("did:agent:kite")

    assert history_validations == 1


@pytest.mark.asyncio
async def test_state_and_receipts_survive_database_restart(tmp_path):
    path = tmp_path / "host.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    first = HoldStore(first_db)
    await first.ensure_schema()
    mutation = await first.set_hold(
        scope="agent",
        target_id="did:agent:durable",
        actor_id="did:sovereign:operator",
        reason="leave down",
        operation_id="durable-operation",
    )
    await first_db.close()

    second_db = await AsyncDatabase.sqlite(str(path))
    second = HoldStore(second_db)
    await second.ensure_schema()
    try:
        restored = await second.get_effective("did:agent:durable")
        boot_state = await second.read_boot_state()
        receipt = await second.get_receipt("durable-operation")
        assert restored.agent == mutation.current
        assert boot_state == (mutation.current,)
        assert receipt == mutation.receipt
        assert datetime.fromisoformat(receipt.occurred_at).tzinfo is not None
    finally:
        await second_db.close()


@pytest.mark.asyncio
async def test_receipt_primary_key_is_explicitly_not_null(hold_db):
    db, _store = hold_db
    columns = await db.fetchall("PRAGMA table_info(hold_receipts)")
    receipt_id = next(row for row in columns if row[1] == "receipt_id")
    assert receipt_id[3] == 1


@pytest.mark.asyncio
async def test_content_witness_lookup_has_target_index(hold_db):
    db, _store = hold_db
    columns = await db.fetchall(
        "PRAGMA index_info(idx_hold_receipt_content_witnesses_target)"
    )
    assert [row[2] for row in columns] == ["scope", "target_id"]


def test_existing_null_receipt_identity_fails_closed_on_read() -> None:
    with pytest.raises(HoldCorruptStateError, match="missing required evidence"):
        _receipt_from_row(
            (
                None,
                "legacy-operation",
                "release",
                "already_in_state",
                "agent",
                "did:agent:kite",
                "legacy import",
                "did:sovereign:operator",
                "2026-08-28T00:00:00+00:00",
                "expected-receipt",
                "",
                "",
            )
        )


@pytest.mark.parametrize("timestamp", ["not-a-timestamp", "2026-08-28T00:00:00"])
def test_malformed_or_naive_receipt_timestamp_fails_closed(timestamp) -> None:
    with pytest.raises(HoldCorruptStateError, match="receipt timestamp"):
        _receipt_from_row(
            (
                "receipt-id",
                "legacy-operation",
                "hold",
                "applied",
                "agent",
                "did:agent:kite",
                "legacy import",
                "did:sovereign:operator",
                timestamp,
                "",
                "",
                "receipt-id",
            )
        )


@pytest.mark.parametrize("timestamp", ["not-a-timestamp", "2026-08-28T00:00:00"])
def test_malformed_or_naive_latch_timestamp_fails_closed(timestamp) -> None:
    with pytest.raises(HoldCorruptStateError, match="latch timestamp"):
        _latch_from_row(
            (
                "agent",
                "did:agent:kite",
                1,
                "receipt-id",
                "legacy import",
                "did:sovereign:operator",
                timestamp,
                1,
            )
        )


@pytest.mark.parametrize("field_index", [0, 1, 5, 6, 7, 8])
def test_whitespace_only_required_receipt_evidence_fails_closed(field_index) -> None:
    row = [
        "receipt-id",
        "legacy-operation",
        "hold",
        "applied",
        "agent",
        "did:agent:kite",
        "legacy import",
        "did:sovereign:operator",
        "2026-08-28T00:00:00+00:00",
        "",
        "",
        "receipt-id",
    ]
    row[field_index] = "   "

    with pytest.raises(HoldCorruptStateError, match="invariant"):
        _receipt_from_row(tuple(row))


@pytest.mark.parametrize("field_index", [1, 3, 4, 5])
def test_whitespace_only_required_latch_evidence_fails_closed(field_index) -> None:
    row = [
        "agent",
        "did:agent:kite",
        1,
        "receipt-id",
        "legacy import",
        "did:sovereign:operator",
        "2026-08-28T00:00:00+00:00",
        1,
    ]
    row[field_index] = "   "

    with pytest.raises(HoldCorruptStateError, match="missing|required"):
        _latch_from_row(tuple(row))


@pytest.mark.parametrize("field_index", [1, 7, 8])
@pytest.mark.parametrize("bad_value", [None, 42])
def test_existing_invalid_required_receipt_evidence_fails_closed_on_read(
    field_index,
    bad_value,
) -> None:
    row = [
        "receipt-id",
        "legacy-operation",
        "hold",
        "applied",
        "agent",
        "did:agent:kite",
        "legacy import",
        "did:sovereign:operator",
        "2026-08-28T00:00:00+00:00",
        "",
        "",
        "receipt-id",
    ]
    row[field_index] = bad_value

    with pytest.raises(
        HoldCorruptStateError,
        match="missing required evidence|invalid evidence types",
    ):
        _receipt_from_row(tuple(row))


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_loss", ["delete", "reset"])
@pytest.mark.parametrize("operation", ["get", "effective", "set", "release"])
async def test_surviving_hold_authority_fails_closed_after_projection_loss(
    hold_db,
    projection_loss,
    operation,
):
    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="leave stopped",
        operation_id="hold-before-projection-loss",
    )
    if projection_loss == "delete":
        await db.execute(
            "DELETE FROM hold_latches WHERE scope = ? AND target_id = ?",
            ("agent", "did:agent:kite"),
        )
    else:
        await db.execute(
            "UPDATE hold_latches SET active = 0, hold_receipt_id = '', "
            "reason = '', actor_id = '', set_at = '' "
            "WHERE scope = ? AND target_id = ?",
            ("agent", "did:agent:kite"),
        )

    with pytest.raises(HoldCorruptStateError, match="active Hold authority"):
        if operation == "get":
            await store.get_hold("agent", "did:agent:kite")
        elif operation == "effective":
            await store.get_effective("did:agent:kite")
        elif operation == "set":
            await store.set_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:operator",
                reason="do not overwrite missing projection",
                operation_id="set-after-projection-loss",
            )
        else:
            await store.release_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:operator",
                reason="do not release missing projection",
                operation_id="release-after-projection-loss",
                expected_hold_receipt_id=held.receipt.receipt_id,
            )


@pytest.mark.asyncio
async def test_cyclic_applied_hold_history_fails_closed_when_projection_is_unheld(
    hold_db,
):
    db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="first hold",
        operation_id="cycle-first",
    )
    second = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="second hold",
        operation_id="cycle-second",
    )
    await db.execute(
        "UPDATE hold_receipts SET prior_hold_receipt_id = ? "
        "WHERE receipt_id = ?",
        (second.receipt.receipt_id, first.receipt.receipt_id),
    )
    await db.execute(
        "UPDATE hold_latches SET active = 0, hold_receipt_id = '', "
        "reason = '', actor_id = '', set_at = '' "
        "WHERE scope = ? AND target_id = ?",
        ("agent", "did:agent:kite"),
    )

    with pytest.raises(HoldCorruptStateError, match="cycle"):
        await store.get_hold("agent", "did:agent:kite")


@pytest.mark.asyncio
async def test_inactive_revision_rejects_deleted_closed_receipt_chain(hold_db):
    """Mutation tripwire: an empty graph cannot erase proven prior mutations."""

    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:closed-history",
        actor_id="did:sovereign:operator",
        reason="closed history",
        operation_id="closed-history-hold",
    )
    await store.release_hold(
        scope="agent",
        target_id="did:agent:closed-history",
        actor_id="did:sovereign:operator",
        reason="closed history release",
        operation_id="closed-history-release",
        expected_hold_receipt_id=held.receipt.receipt_id,
    )
    await db.execute(
        "DELETE FROM hold_receipts WHERE scope = ? AND target_id = ?",
        ("agent", "did:agent:closed-history"),
    )

    # Either the target-local revision witness or the global append-only
    # operation tombstone may detect the deletion first. Both are the same
    # required fail-closed outcome.
    with pytest.raises(HoldCorruptStateError):
        await store.get_effective("did:agent:closed-history")


@pytest.mark.asyncio
async def test_deleted_non_applied_receipt_cannot_reapply_old_operation(hold_db):
    """The receipt-count witness covers idempotent audit outcomes too."""

    db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:non-applied-loss",
        actor_id="did:sovereign:operator",
        reason="first reason",
        operation_id="first-applied",
    )
    duplicate = await store.set_hold(
        scope="agent",
        target_id="did:agent:non-applied-loss",
        actor_id="did:sovereign:operator",
        reason="first reason",
        operation_id="old-idempotent-operation",
    )
    assert duplicate.receipt.disposition is HoldDisposition.ALREADY_IN_STATE
    replacement = await store.set_hold(
        scope="agent",
        target_id="did:agent:non-applied-loss",
        actor_id="did:sovereign:operator",
        reason="newer reason",
        operation_id="newer-applied",
    )
    assert replacement.receipt.receipt_id != first.receipt.receipt_id
    await db.execute(
        "DELETE FROM hold_receipts WHERE operation_id = ?",
        ("old-idempotent-operation",),
    )

    with pytest.raises(HoldCorruptStateError, match="receipt-count"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:non-applied-loss",
            actor_id="did:sovereign:operator",
            reason="first reason",
            operation_id="old-idempotent-operation",
        )

    with pytest.raises(HoldCorruptStateError, match="missing receipt"):
        await store.get_receipt("old-idempotent-operation")
    with pytest.raises(HoldCorruptStateError, match="receipt-count"):
        await store.get_hold("agent", "did:agent:non-applied-loss")


@pytest.mark.asyncio
async def test_release_rejects_latch_rewound_to_consumed_hold_authority(hold_db):
    db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="first hold",
        operation_id="rewind-first",
    )
    await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="second hold",
        operation_id="rewind-second",
    )
    await db.execute(
        "UPDATE hold_latches SET active = 1, hold_receipt_id = ?, "
        "reason = ?, actor_id = ?, set_at = ? "
        "WHERE scope = ? AND target_id = ?",
        (
            first.receipt.receipt_id,
            first.receipt.reason,
            first.receipt.actor_id,
            first.receipt.occurred_at,
            "agent",
            "did:agent:kite",
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="terminal authority"):
        await store.release_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="must not release rewound projection",
            operation_id="rewind-release",
            expected_hold_receipt_id=first.receipt.receipt_id,
        )
    assert await store.get_receipt("rewind-release") is None


@pytest.mark.asyncio
async def test_host_context_reads_hold_store_from_control_database_at_boot(tmp_path):
    """The host boot context, not an agent-local DB, owns the durable latch."""

    path = str(tmp_path / "host-control.db")
    first = await build_host_context(db_path=path)
    try:
        assert first.hold_store is not None
        assert first.hold_store._history_anchor_path == hold_history_anchor_path(path)
        held = await first.hold_store.set_hold(
            scope="agent",
            target_id="did:agent:boot-held",
            actor_id="did:sovereign:operator",
            reason="remain held through restart",
            operation_id="boot-hold",
        )
    finally:
        if first.session_factory is not None:
            await first.session_factory.close()
        if first.db is not None:
            await first.db.close()

    reopened = await build_host_context(db_path=path)
    try:
        assert reopened.hold_store is not None
        effective = await reopened.hold_store.get_effective(
            "did:agent:boot-held"
        )
        assert effective.agent == held.current
        assert reopened.hold_boot_state == (held.current,)
    finally:
        if reopened.session_factory is not None:
            await reopened.session_factory.close()
        if reopened.db is not None:
            await reopened.db.close()


@pytest.mark.asyncio
async def test_boot_validates_global_history_once_for_all_targets(
    hold_db,
    monkeypatch,
):
    _db, store = hold_db
    for index in range(3):
        await store.set_hold(
            scope="agent",
            target_id=f"did:agent:boot-{index}",
            actor_id="did:sovereign:operator",
            reason="boot validation scaling",
            operation_id=f"boot-validation-{index}",
        )

    validate_anchor = AsyncMock(wraps=store._assert_history_anchor_intact)
    validate_operations = AsyncMock(
        wraps=store._assert_no_orphaned_operation_witnesses
    )
    monkeypatch.setattr(store, "_assert_history_anchor_intact", validate_anchor)
    monkeypatch.setattr(
        store,
        "_assert_no_orphaned_operation_witnesses",
        validate_operations,
    )

    states = await store.read_boot_state()

    assert len(states) == 3
    assert validate_anchor.await_count == 1
    assert validate_operations.await_count == 1


@pytest.mark.asyncio
async def test_boot_rejects_null_target_before_addressing_the_latch(hold_db):
    """A malformed persisted target cannot disappear behind ``str(None)``."""

    db, store = hold_db
    await db.execute("ALTER TABLE hold_latches RENAME TO hold_latches_valid")
    await db.execute(
        "CREATE TABLE hold_latches ("
        "scope TEXT, target_id TEXT, active INTEGER, hold_receipt_id TEXT, "
        "reason TEXT, actor_id TEXT, set_at TEXT, revision INTEGER)"
    )
    await db.execute(
        "INSERT INTO hold_latches VALUES "
        "('agent', NULL, 1, 'missing-authority', 'investigate', "
        "'did:sovereign:operator', '2026-08-31T00:00:00+00:00', 1)"
    )
    await db.execute("DROP TABLE hold_latches_valid")

    with pytest.raises(HoldCorruptStateError, match="missing its identity"):
        await store.read_boot_state()


@pytest.mark.asyncio
async def test_boot_rejects_whitespace_changing_persisted_target(hold_db):
    """Boot cannot normalize a malformed key and silently query another row."""

    db, store = hold_db
    await db.execute(
        "INSERT INTO hold_latches ("
        "scope, target_id, active, hold_receipt_id, reason, actor_id, set_at, revision"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "agent",
            " did:agent:noncanonical ",
            1,
            "missing-authority",
            "inspect malformed target",
            "did:sovereign:operator",
            "2026-08-31T00:00:00+00:00",
            1,
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="noncanonical identity"):
        await store.read_boot_state()


@pytest.mark.asyncio
async def test_host_context_uses_configured_postgres_for_durable_hold(
    monkeypatch,
    tmp_path,
):
    """Cloud Hold uses PG without cutting existing host features off SQLite."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.sqla import session as session_module

    events: list[object] = []

    class _DB:
        def __init__(self, backend_type):
            self.backend_type = backend_type

        async def close(self):
            events.append(("db-close", self.backend_type))

    class _InnerFactory:
        engine = object()

        async def close(self):
            events.append("factory-close")

    fake_host_db = _DB("sqlite")
    fake_hold_db = _DB("postgres")
    fake_evidence_db = _DB("postgres")

    async def _sqlite(_cls, path):
        events.append(("sqlite", path))
        return fake_host_db

    async def _ensure_schema(self):
        events.append(("hold-schema", self._db))

    async def _read_boot_state(self):
        events.append(("hold-boot-read", self._db))
        return ()

    async def _initialize(primary_dsn, evidence_dsn):
        events.append(("custody-initialize", primary_dsn, evidence_dsn))
        return fake_hold_db, fake_evidence_db

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://durable/host")
    monkeypatch.setenv(
        "KESTREL_HOLD_EVIDENCE_DATABASE_URL",
        "postgresql://independent/evidence",
    )
    monkeypatch.setattr(AsyncDatabase, "sqlite", classmethod(_sqlite))
    monkeypatch.setattr(
        session_module, "make_session_factory", lambda db: _InnerFactory()
    )
    monkeypatch.setattr(HoldStore, "ensure_schema", _ensure_schema)
    monkeypatch.setattr(HoldStore, "read_boot_state", _read_boot_state)
    monkeypatch.setattr(
        "kestrel_sovereign.hold.state.initialize_postgres_hold_databases",
        _initialize,
    )

    host_path = tmp_path / "existing-host-features.db"
    ctx = await build_host_context(db_path=str(host_path))

    assert ctx.db is fake_host_db
    assert ctx.hold_db is fake_hold_db
    assert ctx.hold_evidence_db is fake_evidence_db
    assert ctx.hold_store._db is fake_hold_db
    assert ctx.hold_store._evidence_db is fake_evidence_db
    assert ctx.hold_store._initialization_witness_path is None
    assert ctx.hold_store._history_anchor_path is None
    assert events[:4] == [
        ("sqlite", str(host_path)),
        (
            "custody-initialize",
            "postgresql://durable/host",
            "postgresql://independent/evidence",
        ),
        ("hold-schema", fake_hold_db),
        ("hold-boot-read", fake_hold_db),
    ]


@pytest.mark.asyncio
async def test_postgres_without_dsn_uses_runtime_sqlite_fallback(
    monkeypatch,
    tmp_path,
):
    """Hold selects the same effective backend as the agent runtime."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.sqla import session as session_module

    events: list[object] = []

    class _DB:
        backend_type = "sqlite"

        async def close(self):
            events.append("db-close")

    class _InnerFactory:
        engine = object()

        async def close(self):
            events.append("factory-close")

    fake_db = _DB()

    async def _sqlite(_cls, path):
        events.append(("sqlite", path))
        return fake_db

    async def _postgres_hold_initializer(_primary_dsn, _evidence_dsn):
        raise AssertionError("runtime fallback must not open PostgreSQL")

    async def _ensure_schema(self):
        events.append(("hold-schema", self._db))

    async def _read_boot_state(self):
        return ()

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    monkeypatch.setattr(AsyncDatabase, "sqlite", classmethod(_sqlite))
    monkeypatch.setattr(
        "kestrel_sovereign.hold.state.initialize_postgres_hold_databases",
        _postgres_hold_initializer,
    )
    monkeypatch.setattr(
        session_module, "make_session_factory", lambda _db: _InnerFactory()
    )
    monkeypatch.setattr(HoldStore, "ensure_schema", _ensure_schema)
    monkeypatch.setattr(HoldStore, "read_boot_state", _read_boot_state)

    path = tmp_path / "sqlite-fallback.db"
    ctx = await build_host_context(db_path=str(path))

    assert ctx.db is fake_db
    assert ctx.hold_db is fake_db
    assert ctx.hold_store is not None
    assert events == [
        ("sqlite", str(path)),
        ("hold-schema", fake_db),
    ]


@pytest.mark.asyncio
async def test_postgres_hold_without_independent_evidence_fails_closed_at_boot(
    monkeypatch,
    tmp_path,
):
    """A missing rollback witness is named and no host store is handed off."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.sqla import session as session_module

    events: list[str] = []

    class _DB:
        backend_type = "sqlite"

        async def close(self):
            events.append("db-close")

    class _InnerFactory:
        engine = object()

        async def close(self):
            events.append("factory-close")

    async def _sqlite(_cls, _path):
        return _DB()

    async def _postgres_hold_initializer(_primary_dsn, _evidence_dsn):
        raise AssertionError("missing evidence config must fail before PG opens")

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://durable/host")
    monkeypatch.delenv("KESTREL_HOLD_EVIDENCE_DATABASE_URL", raising=False)
    monkeypatch.setattr(AsyncDatabase, "sqlite", classmethod(_sqlite))
    monkeypatch.setattr(
        "kestrel_sovereign.hold.state.initialize_postgres_hold_databases",
        _postgres_hold_initializer,
    )
    monkeypatch.setattr(
        session_module,
        "make_session_factory",
        lambda _db: _InnerFactory(),
    )

    context = await build_host_context(db_path=str(tmp_path / "host.db"))

    assert context.db is None
    assert context.hold_store is None
    assert "KESTREL_HOLD_EVIDENCE_DATABASE_URL is required" in context.backend_error
    assert events == ["factory-close", "db-close"]


@pytest.mark.asyncio
async def test_postgres_custody_preflight_failure_precedes_schema_initialization(
    monkeypatch,
    tmp_path,
):
    """A wrong-role database is never handed to the schema-writing factory."""

    from kestrel_sovereign.storage.sqla import session as session_module

    events: list[str] = []

    class _DB:
        backend_type = "sqlite"

        async def close(self):
            events.append("db-close")

    class _InnerFactory:
        engine = object()

        async def close(self):
            events.append("factory-close")

    async def _sqlite(_cls, _path):
        return _DB()

    async def _reject_custody(_primary_dsn, _evidence_dsn):
        events.append("custody-preflight")
        raise HoldStateError("wrong durable custody role")

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://durable/host")
    monkeypatch.setenv(
        "KESTREL_HOLD_EVIDENCE_DATABASE_URL",
        "postgresql://independent/evidence",
    )
    monkeypatch.setattr(AsyncDatabase, "sqlite", classmethod(_sqlite))
    monkeypatch.setattr(
        session_module,
        "make_session_factory",
        lambda _db: _InnerFactory(),
    )
    monkeypatch.setattr(
        "kestrel_sovereign.hold.state.initialize_postgres_hold_databases",
        _reject_custody,
    )

    context = await build_host_context(db_path=str(tmp_path / "host.db"))

    assert context.hold_store is None
    assert "wrong durable custody role" in context.backend_error
    assert events == ["custody-preflight", "factory-close", "db-close"]


@pytest.mark.asyncio
async def test_postgres_initialization_witness_uses_durable_runtime_metadata(
    tmp_path,
):
    """A fresh Cloud Run instance reads a separately-custodied PG witness."""

    db = await AsyncDatabase.sqlite(str(tmp_path / "postgres-primary-facade.db"))
    evidence = await AsyncDatabase.sqlite(
        str(tmp_path / "postgres-evidence-facade.db")
    )

    class _PostgresWitnessFacade:
        backend_type = "postgres"

        def __init__(self, inner):
            self._inner = inner

        async def fetchall(self, query, params=()):
            return await self._inner.fetchall(query, params)

        async def execute(self, query, params=()):
            return await self._inner.execute(query, params)

    try:
        primary = _PostgresWitnessFacade(db)
        evidence_store = _PostgresWitnessFacade(evidence)
        first = HoldStore(primary, evidence_db=evidence_store)
        assert first._initialization_witness_path is None
        assert await first._read_initialization_witness() is False
        await first._write_initialization_witness()

        restarted = HoldStore(primary, evidence_db=evidence_store)
        assert await restarted._read_initialization_witness() is True
        await _create_legacy_hold_tables(db)
        await first._write_history_anchor()
        assert await restarted._read_history_anchor() == (
            await first._current_history_anchor_payload()
        )
    finally:
        await db.close()
        await evidence.close()


def test_postgres_hold_refuses_evidence_in_the_primary_rollback_domain() -> None:
    """The database being protected cannot also supply its rollback witness."""

    with pytest.raises(HoldStateError, match="independent.*evidence"):
        HoldStore(SimpleNamespace(backend_type="postgres"))


@pytest.mark.asyncio
async def test_postgres_factory_closes_pool_when_core_schema_init_fails(
    monkeypatch,
):
    """A factory failure cannot hide its newly connected pool from cleanup."""

    from kestrel_sovereign.storage import async_database as database_module
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db import postgres as postgres_module

    events: list[str] = []

    class _Backend:
        backend_type = "postgres"

        async def connect(self):
            events.append("connect")

        async def close(self):
            events.append("close")

    async def _fail_schema(_self):
        events.append("schema")
        raise RuntimeError("core schema failed")

    monkeypatch.setattr(
        postgres_module,
        "PostgresBackend",
        lambda **_kwargs: _Backend(),
    )
    monkeypatch.setattr(AsyncDatabase, "_init_schema", _fail_schema)

    with pytest.raises(RuntimeError, match="core schema failed"):
        await database_module.AsyncDatabase.postgres("postgresql://durable/host")

    assert events == ["connect", "schema", "close"]


@pytest.mark.asyncio
async def test_postgres_factory_forwards_explicit_pool_budget(monkeypatch):
    """Special-purpose stores can reserve only their justified connections."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db import postgres as postgres_module

    constructor_args: dict[str, object] = {}

    class _Backend:
        backend_type = "postgres"

        async def connect(self):
            return None

        async def close(self):
            return None

    def _backend(**kwargs):
        constructor_args.update(kwargs)
        return _Backend()

    async def _init_schema(_self):
        return None

    monkeypatch.setattr(postgres_module, "PostgresBackend", _backend)
    monkeypatch.setattr(AsyncDatabase, "_init_schema", _init_schema)

    db = await AsyncDatabase.postgres(
        "postgresql://durable/hold",
        min_pool_size=1,
        max_pool_size=1,
    )
    await db.close()

    assert constructor_args == {
        "dsn": "postgresql://durable/hold",
        "min_pool_size": 1,
        "max_pool_size": 1,
    }


@pytest.mark.asyncio
async def test_database_factory_preserves_schema_error_when_cleanup_also_fails(
    monkeypatch,
    caplog,
):
    """A secondary close failure cannot replace the initialization defect."""

    from kestrel_sovereign.storage import async_database as database_module
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db import postgres as postgres_module

    class _Backend:
        backend_type = "postgres"

        async def connect(self):
            return None

        async def close(self):
            raise RuntimeError("cleanup failed")

    async def _fail_schema(_self):
        raise RuntimeError("primary schema failure")

    monkeypatch.setattr(
        postgres_module,
        "PostgresBackend",
        lambda **_kwargs: _Backend(),
    )
    monkeypatch.setattr(AsyncDatabase, "_init_schema", _fail_schema)

    with pytest.raises(RuntimeError, match="primary schema failure"):
        await database_module.AsyncDatabase.postgres("postgresql://durable/host")

    assert "cleanup failed" in caplog.text


@pytest.mark.asyncio
async def test_cancelled_host_context_bootstrap_closes_partial_resources(
    monkeypatch,
    tmp_path,
):
    """Cancellation before context handoff cannot strand either database."""

    from kestrel_sovereign.storage.sqla import session as session_module

    events: list[object] = []
    schema_entered = asyncio.Event()
    factory_close_started = asyncio.Event()
    release_factory_close = asyncio.Event()

    class _DB:
        def __init__(self, backend_type):
            self.backend_type = backend_type

        async def close(self):
            events.append(("db-close", self.backend_type))

    class _InnerFactory:
        engine = object()

        async def close(self):
            events.append("factory-close-started")
            factory_close_started.set()
            await release_factory_close.wait()
            events.append("factory-close-finished")

    fake_host_db = _DB("sqlite")
    fake_hold_db = _DB("postgres")
    fake_evidence_db = _DB("postgres")

    async def _sqlite(_cls, _path):
        return fake_host_db

    async def _initialize(_primary_dsn, _evidence_dsn):
        return fake_hold_db, fake_evidence_db

    async def _ensure_schema(_self):
        schema_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://durable/host")
    monkeypatch.setenv(
        "KESTREL_HOLD_EVIDENCE_DATABASE_URL",
        "postgresql://independent/evidence",
    )
    monkeypatch.setattr(AsyncDatabase, "sqlite", classmethod(_sqlite))
    monkeypatch.setattr(
        session_module, "make_session_factory", lambda _db: _InnerFactory()
    )
    monkeypatch.setattr(HoldStore, "ensure_schema", _ensure_schema)
    monkeypatch.setattr(
        "kestrel_sovereign.hold.state.initialize_postgres_hold_databases",
        _initialize,
    )

    bootstrap = asyncio.create_task(
        build_host_context(db_path=str(tmp_path / "host.db"))
    )
    await schema_entered.wait()
    bootstrap.cancel()
    await factory_close_started.wait()
    bootstrap.cancel()
    release_factory_close.set()

    with pytest.raises(asyncio.CancelledError):
        await bootstrap

    assert events == [
        "factory-close-started",
        "factory-close-finished",
        ("db-close", "postgres"),
        ("db-close", "postgres"),
        ("db-close", "sqlite"),
    ]


@pytest.mark.asyncio
async def test_cancel_during_failed_host_bootstrap_cleanup_propagates(
    monkeypatch,
    tmp_path,
):
    """An opening error cannot turn later shutdown into a degraded context."""

    from kestrel_sovereign.storage.sqla import session as session_module

    events: list[object] = []
    factory_close_started = asyncio.Event()
    release_factory_close = asyncio.Event()

    class _DB:
        def __init__(self, backend_type):
            self.backend_type = backend_type

        async def close(self):
            events.append(("db-close", self.backend_type))

    class _InnerFactory:
        engine = object()

        async def close(self):
            events.append("factory-close-started")
            factory_close_started.set()
            await release_factory_close.wait()
            events.append("factory-close-finished")

    fake_host_db = _DB("sqlite")
    fake_hold_db = _DB("postgres")
    fake_evidence_db = _DB("postgres")

    async def _sqlite(_cls, _path):
        return fake_host_db

    async def _initialize(_primary_dsn, _evidence_dsn):
        return fake_hold_db, fake_evidence_db

    async def _fail_schema(_self):
        raise RuntimeError("schema opening failed")

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://durable/host")
    monkeypatch.setenv(
        "KESTREL_HOLD_EVIDENCE_DATABASE_URL",
        "postgresql://independent/evidence",
    )
    monkeypatch.setattr(AsyncDatabase, "sqlite", classmethod(_sqlite))
    monkeypatch.setattr(
        session_module,
        "make_session_factory",
        lambda _db: _InnerFactory(),
    )
    monkeypatch.setattr(HoldStore, "ensure_schema", _fail_schema)
    monkeypatch.setattr(
        "kestrel_sovereign.hold.state.initialize_postgres_hold_databases",
        _initialize,
    )

    bootstrap = asyncio.create_task(
        build_host_context(db_path=str(tmp_path / "host.db"))
    )
    await factory_close_started.wait()
    bootstrap.cancel()
    release_factory_close.set()

    with pytest.raises(asyncio.CancelledError):
        await bootstrap

    assert events == [
        "factory-close-started",
        "factory-close-finished",
        ("db-close", "postgres"),
        ("db-close", "postgres"),
        ("db-close", "sqlite"),
    ]


@pytest.mark.asyncio
async def test_receipt_lookup_rejects_missing_authority_history(hold_db):
    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:receipt-history",
        actor_id="did:sovereign:operator",
        reason="inspect authority",
        operation_id="receipt-authority-hold",
    )
    released = await store.release_hold(
        scope="agent",
        target_id="did:agent:receipt-history",
        actor_id="did:sovereign:operator",
        reason="inspect release",
        operation_id="receipt-authority-release",
        expected_hold_receipt_id=held.receipt.receipt_id,
    )
    assert released.receipt.disposition is HoldDisposition.APPLIED
    await db.execute(
        "DELETE FROM hold_receipts WHERE operation_id = ?",
        ("receipt-authority-hold",),
    )

    with pytest.raises(HoldCorruptStateError, match="missing authority"):
        await store.get_receipt("receipt-authority-release")


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["already_in_state", "refused_stale"])
async def test_non_applied_receipt_rejects_missing_referenced_authority(
    hold_db,
    disposition,
):
    """Audit-only outcomes cannot survive deletion of their authority proof."""

    db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:non-applied-history",
        actor_id="did:sovereign:operator",
        reason="first hold",
        operation_id=f"{disposition}-first",
    )
    if disposition == "already_in_state":
        non_applied = await store.set_hold(
            scope="agent",
            target_id="did:agent:non-applied-history",
            actor_id="did:sovereign:operator",
            reason="first hold",
            operation_id="already-in-state-receipt",
        )
    else:
        second = await store.set_hold(
            scope="agent",
            target_id="did:agent:non-applied-history",
            actor_id="did:sovereign:operator",
            reason="replacement hold",
            operation_id="refused-stale-second",
        )
        non_applied = await store.release_hold(
            scope="agent",
            target_id="did:agent:non-applied-history",
            actor_id="did:sovereign:operator",
            reason="stale release",
            operation_id="refused-stale-receipt",
            expected_hold_receipt_id=first.receipt.receipt_id,
        )
        assert non_applied.current == second.current
    assert non_applied.receipt.disposition.value == disposition

    await db.execute(
        "DELETE FROM hold_receipts WHERE disposition = 'applied'"
    )
    await db.execute(
        "UPDATE hold_latches SET active = 0, hold_receipt_id = '', "
        "reason = '', actor_id = '', set_at = '' "
        "WHERE scope = ? AND target_id = ?",
        ("agent", "did:agent:non-applied-history"),
    )

    with pytest.raises(HoldCorruptStateError, match="missing authority"):
        await store.get_receipt(non_applied.receipt.operation_id)


@pytest.mark.asyncio
async def test_operation_replay_is_exact_and_conflicting_reuse_fails(hold_db):
    _db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:one",
        actor_id="did:sovereign:operator",
        reason="inspect",
        operation_id="same-operation",
    )
    replay = await store.set_hold(
        scope="agent",
        target_id="did:agent:one",
        actor_id="did:sovereign:operator",
        reason="inspect",
        operation_id="same-operation",
    )
    assert replay.receipt == first.receipt
    assert replay.current == first.current

    with pytest.raises(HoldIdempotencyConflict):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:two",
            actor_id="did:sovereign:operator",
            reason="inspect",
            operation_id="same-operation",
        )


@pytest.mark.asyncio
async def test_legacy_receipts_gain_witnesses_once_before_marker(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-witness-upgrade.db"))
    await _create_legacy_hold_tables(db)
    receipt_id = "legacy-upgrade-receipt"
    target_id = "did:agent:legacy-upgrade"
    occurred_at = "2026-08-28T00:00:00+00:00"
    await db.execute(
        "INSERT INTO hold_receipts ("
        "receipt_id, operation_id, action, disposition, scope, target_id, "
        "reason, actor_id, occurred_at, resulting_hold_receipt_id"
        ") VALUES (?, ?, 'hold', 'applied', 'agent', ?, ?, ?, ?, ?)",
        (
            receipt_id,
            "legacy-upgrade-operation",
            target_id,
            "legacy import",
            "did:sovereign:operator",
            occurred_at,
            receipt_id,
        ),
    )
    await db.execute(
        "INSERT INTO hold_latches ("
        "scope, target_id, active, hold_receipt_id, reason, actor_id, "
        "set_at, revision"
        ") VALUES ('agent', ?, 1, ?, ?, ?, ?, 1)",
        (
            target_id,
            receipt_id,
            "legacy import",
            "did:sovereign:operator",
            occurred_at,
        ),
    )
    store = HoldStore(db)
    try:
        await store.ensure_schema()

        effective = await store.get_effective(target_id)
        receipt = await store.get_receipt("legacy-upgrade-operation")
        marker = await db.fetchone(
            "SELECT 1 FROM hold_schema_migrations "
            "WHERE name = 'hold_state_witness_ledgers_v1'"
        )

        assert effective.held is True
        assert effective.agent is not None
        assert effective.agent.hold_receipt_id == receipt_id
        assert receipt is not None and receipt.receipt_id == receipt_id
        assert marker is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_duplicate_operation_ids_are_rejected_during_backfill(
    tmp_path,
):
    """Migration cannot bind one operation witness to two legacy receipts."""

    db = await AsyncDatabase.sqlite(str(tmp_path / "duplicate-backfill.db"))
    await _create_legacy_hold_tables(db)
    for suffix in ("one", "two"):
        await db.execute(
            "INSERT INTO hold_receipts ("
            "receipt_id, operation_id, action, disposition, scope, target_id, "
            "reason, actor_id, occurred_at, resulting_hold_receipt_id"
            ") VALUES (?, 'duplicate-before-backfill', 'hold', 'applied', "
            "'agent', ?, 'legacy', 'did:sovereign:operator', "
            "'2026-08-28T00:00:00+00:00', ?)",
            (
                f"legacy-duplicate-{suffix}",
                f"did:agent:legacy-{suffix}",
                f"legacy-duplicate-{suffix}",
            ),
        )
    try:
        with pytest.raises(HoldCorruptStateError, match="duplicate operation"):
            await HoldStore(db).ensure_schema()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_duplicate_operation_ids_fail_closed(tmp_path):
    """Runtime reads cannot trust a UNIQUE constraint an old table may lack."""

    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-duplicate-operation.db"))
    await _create_legacy_hold_tables(db)
    store = HoldStore(db)
    await store.ensure_schema()
    try:
        for suffix in ("one", "two"):
            receipt_id = f"duplicate-operation-receipt-{suffix}"
            target_id = f"did:agent:{suffix}"
            await db.execute(
                "INSERT INTO hold_receipts ("
                "receipt_id, operation_id, action, disposition, scope, "
                "target_id, reason, actor_id, occurred_at, "
                "resulting_hold_receipt_id"
                ") VALUES (?, ?, 'hold', 'applied', 'agent', ?, ?, ?, ?, ?)",
                (
                    receipt_id,
                    "duplicate-operation",
                    target_id,
                    "legacy import",
                    "did:sovereign:operator",
                    "2026-08-28T00:00:00+00:00",
                    receipt_id,
                ),
            )
            await db.execute(
                "INSERT INTO hold_latches ("
                "scope, target_id, active, hold_receipt_id, reason, actor_id, "
                "set_at, revision"
                ") VALUES ('agent', ?, 1, ?, ?, ?, ?, 1)",
                (
                    target_id,
                    receipt_id,
                    "legacy import",
                    "did:sovereign:operator",
                    "2026-08-28T00:00:00+00:00",
                ),
            )

        with pytest.raises(HoldCorruptStateError, match="duplicate operation"):
            await store.get_receipt("duplicate-operation")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fractional_latch_revision_fails_closed(hold_db):
    """SQLite integer affinity cannot turn 1.9 into a valid revision one."""

    db, store = hold_db
    await store.set_hold(
        scope="agent",
        target_id="did:agent:fractional-revision",
        actor_id="did:sovereign:operator",
        reason="inspect revision",
        operation_id="fractional-revision-hold",
    )
    await db.execute(
        "UPDATE hold_latches SET revision = ? "
        "WHERE scope = 'agent' AND target_id = ?",
        (1.9, "did:agent:fractional-revision"),
    )

    with pytest.raises(HoldCorruptStateError, match="revision"):
        await store.get_hold("agent", "did:agent:fractional-revision")


@pytest.mark.asyncio
async def test_fractional_latch_active_flag_fails_closed(tmp_path):
    """Legacy tables cannot truncate a fractional flag into active state."""

    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-fractional-active.db"))
    await db.execute(
        "CREATE TABLE hold_latches ("
        "scope TEXT NOT NULL, target_id TEXT NOT NULL, active INTEGER NOT NULL, "
        "hold_receipt_id TEXT NOT NULL, reason TEXT NOT NULL, "
        "actor_id TEXT NOT NULL, set_at TEXT NOT NULL, revision INTEGER NOT NULL, "
        "PRIMARY KEY (scope, target_id))"
    )
    store = HoldStore(db)
    await store.ensure_schema()
    await db.execute(
        "INSERT INTO hold_latches VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "agent",
            "did:agent:fractional-active",
            1.5,
            "receipt",
            "inspect active flag",
            "did:sovereign:operator",
            "2026-08-28T00:00:00+00:00",
            1,
        ),
    )
    try:
        with pytest.raises(HoldCorruptStateError, match="active flag"):
            await store.get_hold("agent", "did:agent:fractional-active")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_repeated_semantic_hold_is_receipted_without_replacing_latch(hold_db):
    _db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="investigate",
        operation_id="hold-once",
    )
    second = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="investigate",
        operation_id="hold-twice",
    )

    assert second.receipt.disposition is HoldDisposition.ALREADY_IN_STATE
    assert second.receipt.receipt_id != first.receipt.receipt_id
    assert second.receipt.resulting_hold_receipt_id == first.receipt.receipt_id
    assert second.current == first.current


@pytest.mark.asyncio
async def test_mutated_receipt_content_cannot_reopen_an_operation_id(hold_db):
    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:mutated-receipt",
        actor_id="did:sovereign:operator",
        reason="inspect",
        operation_id="immutable-hold-operation",
    )
    await store.release_hold(
        scope="agent",
        target_id="did:agent:mutated-receipt",
        actor_id="did:sovereign:operator",
        reason="clear",
        operation_id="immutable-release-operation",
        expected_hold_receipt_id=held.receipt.receipt_id,
    )
    await db.execute(
        "UPDATE hold_receipts SET operation_id = ? WHERE operation_id = ?",
        ("tampered-operation", "immutable-hold-operation"),
    )

    with pytest.raises(HoldCorruptStateError, match="content witness"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:mutated-receipt",
            actor_id="did:sovereign:operator",
            reason="inspect",
            operation_id="immutable-hold-operation",
        )


@pytest.mark.asyncio
async def test_deleted_receipt_operation_id_cannot_be_rebound_to_other_target(
    hold_db,
):
    """A global operation tombstone survives loss of its receipt row."""

    db, store = hold_db
    original = await store.set_hold(
        scope="agent",
        target_id="did:agent:original-operation-owner",
        actor_id="did:sovereign:operator",
        reason="first use",
        operation_id="globally-retired-operation",
    )
    await db.execute(
        "DELETE FROM hold_receipts WHERE receipt_id = ?",
        (original.receipt.receipt_id,),
    )

    with pytest.raises(HoldCorruptStateError, match="missing receipt"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:attempted-new-owner",
            actor_id="did:sovereign:operator",
            reason="second use",
            operation_id="globally-retired-operation",
        )

    assert (
        await db.fetchone(
            "SELECT operation_id FROM hold_receipts WHERE target_id = ?",
            ("did:agent:attempted-new-owner",),
        )
        is None
    )


@pytest.mark.asyncio
async def test_orphaned_operation_witness_fails_closed_on_read_and_restart(hold_db):
    """A surviving append-only tombstone proves deleted Hold history existed."""

    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:orphaned-operation",
        actor_id="did:sovereign:operator",
        reason="evidence must survive",
        operation_id="orphaned-operation-witness",
    )
    await db.execute(
        "DELETE FROM hold_latches WHERE scope = ? AND target_id = ?",
        (held.receipt.scope.value, held.receipt.target_id),
    )
    await db.execute(
        "DELETE FROM hold_receipts WHERE receipt_id = ?",
        (held.receipt.receipt_id,),
    )
    await db.execute(
        "DELETE FROM hold_receipt_witnesses WHERE scope = ? AND target_id = ?",
        (held.receipt.scope.value, held.receipt.target_id),
    )
    await db.execute(
        "DELETE FROM hold_receipt_content_witnesses WHERE receipt_id = ?",
        (held.receipt.receipt_id,),
    )

    with pytest.raises(HoldCorruptStateError, match="missing receipt"):
        await store.get_hold("agent", held.receipt.target_id)
    with pytest.raises(HoldCorruptStateError, match="missing receipt"):
        await store.ensure_schema()


@pytest.mark.asyncio
async def test_absent_receipt_lookup_rejects_any_global_orphaned_tombstone(hold_db):
    """An unrelated absence cannot certify a globally corrupt operation ledger."""

    db, store = hold_db
    await db.execute(
        "INSERT INTO hold_operation_witnesses (operation_id, receipt_id) "
        "VALUES (?, ?)",
        ("orphaned-operation", "missing-receipt"),
    )

    with pytest.raises(HoldCorruptStateError, match="missing receipt"):
        await store.get_receipt("different-absent-operation")


@pytest.mark.asyncio
async def test_every_global_read_rejects_receipt_missing_operation_witness(hold_db):
    """Runtime reads cannot certify history after a required witness is lost."""

    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:missing-operation-witness",
        actor_id="did:sovereign:operator",
        reason="witness must remain durable",
        operation_id="operation-that-loses-witness",
    )
    await db.execute(
        "DELETE FROM hold_operation_witnesses WHERE operation_id = ?",
        (held.receipt.operation_id,),
    )

    with pytest.raises(HoldCorruptStateError, match="missing an operation witness"):
        await store.get_hold(held.receipt.scope, held.receipt.target_id)
    with pytest.raises(HoldCorruptStateError, match="missing an operation witness"):
        await store.read_boot_state()
    with pytest.raises(HoldCorruptStateError, match="missing an operation witness"):
        await store.get_receipt("different-absent-operation")


@pytest.mark.asyncio
async def test_boot_discovers_target_retained_only_by_content_and_count_witnesses(
    hold_db,
):
    """Surviving target-bearing evidence cannot vanish from the boot scan."""

    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:witness-only-target",
        actor_id="did:sovereign:operator",
        reason="evidence must remain visible",
        operation_id="witness-only-target-hold",
    )
    await db.execute(
        "DELETE FROM hold_latches WHERE scope = ? AND target_id = ?",
        (held.receipt.scope.value, held.receipt.target_id),
    )
    await db.execute(
        "DELETE FROM hold_receipts WHERE receipt_id = ?",
        (held.receipt.receipt_id,),
    )
    await db.execute(
        "DELETE FROM hold_operation_witnesses WHERE operation_id = ?",
        (held.receipt.operation_id,),
    )

    with pytest.raises(HoldCorruptStateError, match="witness|receipt-count"):
        await store.read_boot_state()


@pytest.mark.asyncio
@pytest.mark.parametrize("witness_kind", ["content", "operation", "receipt-count"])
async def test_completed_witness_migration_never_reblesses_missing_evidence(
    hold_db, witness_kind
):
    """A later boot treats missing ledgers as corruption, not legacy state."""

    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:backfill",
        actor_id="did:sovereign:operator",
        reason="original evidence",
        operation_id="witness-must-survive",
    )
    if witness_kind == "content":
        # Deleting the digest after mutating the row was the dangerous case:
        # repeat startup used to hash the attacker-controlled replacement and
        # thereby bless it as the new integrity witness.
        await db.execute(
            "UPDATE hold_receipts SET reason = ? WHERE receipt_id = ?",
            ("tampered evidence", held.receipt.receipt_id),
        )
        await db.execute(
            "DELETE FROM hold_receipt_content_witnesses WHERE receipt_id = ?",
            (held.receipt.receipt_id,),
        )
    elif witness_kind == "operation":
        await db.execute(
            "DELETE FROM hold_operation_witnesses WHERE operation_id = ?",
            (held.receipt.operation_id,),
        )
    else:
        await db.execute(
            "DELETE FROM hold_receipt_witnesses "
            "WHERE scope = 'agent' AND target_id = ?",
            (held.receipt.target_id,),
        )

    with pytest.raises(
        HoldCorruptStateError,
        match="completed Hold witness migration",
    ):
        await store.ensure_schema()


@pytest.mark.asyncio
async def test_initialized_store_rejects_deleted_witness_migration_marker(hold_db):
    """An initialized v1 store may not reclassify current rows as legacy."""

    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:deleted-migration-marker",
        actor_id="did:sovereign:operator",
        reason="original evidence",
        operation_id="deleted-migration-marker-hold",
    )
    await db.execute(
        "UPDATE hold_receipts SET reason = ? WHERE receipt_id = ?",
        ("tampered evidence", held.receipt.receipt_id),
    )
    await db.execute(
        "DELETE FROM hold_receipt_content_witnesses WHERE receipt_id = ?",
        (held.receipt.receipt_id,),
    )
    await db.execute(
        "DELETE FROM hold_schema_migrations WHERE name = ?",
        ("hold_state_witness_ledgers_v1",),
    )

    with pytest.raises(
        HoldCorruptStateError,
        match="initialized Hold schema is missing.*migration",
    ):
        await store.ensure_schema()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["ensure", "boot", "effective", "set"])
async def test_external_history_anchor_rejects_wholesale_active_target_erasure(
    hold_db,
    operation,
):
    """Deleting one target and all its in-database witnesses cannot release it."""

    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:wholesale-erasure",
        actor_id="did:sovereign:operator",
        reason="must remain held",
        operation_id="wholesale-erasure-hold",
    )
    await db.execute(
        "DELETE FROM hold_operation_witnesses WHERE operation_id = ?",
        (held.receipt.operation_id,),
    )
    await db.execute(
        "DELETE FROM hold_receipt_content_witnesses WHERE receipt_id = ?",
        (held.receipt.receipt_id,),
    )
    await db.execute(
        "DELETE FROM hold_receipt_witnesses WHERE scope = ? AND target_id = ?",
        (held.receipt.scope.value, held.receipt.target_id),
    )
    await db.execute(
        "DELETE FROM hold_receipts WHERE receipt_id = ?",
        (held.receipt.receipt_id,),
    )
    await db.execute(
        "DELETE FROM hold_latches WHERE scope = ? AND target_id = ?",
        (held.receipt.scope.value, held.receipt.target_id),
    )

    with pytest.raises(HoldCorruptStateError, match="history anchor"):
        if operation == "ensure":
            await store.ensure_schema()
        elif operation == "boot":
            await store.read_boot_state()
        elif operation == "effective":
            await store.get_effective(held.receipt.target_id)
        else:
            await store.set_hold(
                scope="agent",
                target_id=held.receipt.target_id,
                actor_id="did:sovereign:operator",
                reason="must not recreate erased authority",
                operation_id="hold-after-wholesale-erasure",
            )


@pytest.mark.asyncio
async def test_stale_release_cannot_clear_a_replaced_hold(hold_db):
    _db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:one",
        reason="first reason",
        operation_id="hold-first",
    )
    second = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:two",
        reason="replacement reason",
        operation_id="hold-second",
    )

    stale = await store.release_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:one",
        reason="release old observation",
        operation_id="release-stale",
        expected_hold_receipt_id=first.receipt.receipt_id,
    )
    assert stale.receipt.disposition is HoldDisposition.REFUSED_STALE
    assert stale.current == second.current
    assert await store.get_hold("agent", "did:agent:kite") == second.current


@pytest.mark.asyncio
async def test_release_rejects_latch_with_missing_authority_receipt(hold_db):
    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:kite",
        actor_id="did:sovereign:operator",
        reason="investigate",
        operation_id="hold-before-receipt-loss",
    )
    await db.execute(
        "DELETE FROM hold_receipts WHERE receipt_id = ?",
        (held.receipt.receipt_id,),
    )

    with pytest.raises(HoldCorruptStateError, match="missing authority receipt"):
        await store.release_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="must remain held",
            operation_id="release-without-authority",
            expected_hold_receipt_id=held.receipt.receipt_id,
        )

    row = await db.fetchone(
        "SELECT active, hold_receipt_id FROM hold_latches "
        "WHERE scope = ? AND target_id = ?",
        ("agent", "did:agent:kite"),
    )
    assert row == (1, held.receipt.receipt_id)
    assert (
        await db.fetchone(
            "SELECT receipt_id FROM hold_receipts WHERE operation_id = ?",
            ("release-without-authority",),
        )
        is None
    )


@pytest.mark.asyncio
async def test_release_rejects_latch_bound_to_another_targets_receipt(hold_db):
    db, store = hold_db
    first = await store.set_hold(
        scope="agent",
        target_id="did:agent:first",
        actor_id="did:sovereign:operator",
        reason="first investigation",
        operation_id="hold-first-target",
    )
    other = await store.set_hold(
        scope="agent",
        target_id="did:agent:other",
        actor_id="did:sovereign:operator",
        reason="other investigation",
        operation_id="hold-other-target",
    )
    await db.execute(
        "UPDATE hold_latches SET hold_receipt_id = ? "
        "WHERE scope = ? AND target_id = ?",
        (other.receipt.receipt_id, "agent", "did:agent:first"),
    )

    with pytest.raises(HoldCorruptStateError, match="does not match"):
        await store.release_hold(
            scope="agent",
            target_id="did:agent:first",
            actor_id="did:sovereign:operator",
            reason="must remain held",
            operation_id="release-mismatched-authority",
            expected_hold_receipt_id=other.receipt.receipt_id,
        )

    row = await db.fetchone(
        "SELECT active, hold_receipt_id FROM hold_latches "
        "WHERE scope = ? AND target_id = ?",
        ("agent", "did:agent:first"),
    )
    assert row == (1, other.receipt.receipt_id)
    assert first.receipt.receipt_id != other.receipt.receipt_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "target_id", "expected_lock"),
    [
        (HoldScope.AGENT, "did:agent:independent", False),
        (HoldScope.HOST, None, True),
    ],
)
async def test_mutation_locks_host_shape_only_for_host_scope(
    hold_db, monkeypatch, scope, target_id, expected_lock,
):
    _db, store = hold_db
    inspect_shape = AsyncMock(wraps=store._assert_host_latch_shape)
    monkeypatch.setattr(store, "_assert_host_latch_shape", inspect_shape)

    await store.set_hold(
        scope=scope,
        target_id=target_id,
        actor_id="did:sovereign:operator",
        reason="scope-specific lock",
        operation_id=f"lock-{scope.value}",
    )

    inspect_shape.assert_awaited_once_with(for_update=expected_lock)


@pytest.mark.asyncio
async def test_receipt_and_latch_roll_back_as_one_unit(hold_db, monkeypatch):
    _db, store = hold_db
    insert = store._insert_receipt

    async def fail_after_receipt(**kwargs):
        await insert(**kwargs)
        raise RuntimeError("injected crash after receipt")

    monkeypatch.setattr(store, "_insert_receipt", fail_after_receipt)
    with pytest.raises(Exception, match="injected crash"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="investigate",
            operation_id="rolled-back",
        )

    assert await store.get_hold("agent", "did:agent:kite") is None
    assert await store.get_receipt("rolled-back") is None


@pytest.mark.asyncio
async def test_failed_sqlite_mutation_does_not_publish_rolled_back_anchor(
    hold_db, monkeypatch
):
    """External evidence may not advance ahead of a failed DB transaction."""

    _db, store = hold_db
    stage = store._stage_history_candidate

    def fail_after_staging(payload):
        stage(payload)
        raise RuntimeError("injected failure after anchor staging")

    monkeypatch.setattr(store, "_stage_history_candidate", fail_after_staging)
    with pytest.raises(Exception, match="after anchor staging"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:rolled-back-anchor",
            actor_id="did:sovereign:operator",
            reason="must roll back together",
            operation_id="rolled-back-anchor",
        )

    monkeypatch.setattr(store, "_stage_history_candidate", stage)
    assert await store.get_hold("agent", "did:agent:rolled-back-anchor") is None
    assert await store.get_receipt("rolled-back-anchor") is None


@pytest.mark.asyncio
async def test_failed_sqlite_release_removes_known_rolled_back_candidate(
    hold_db, monkeypatch
):
    """A live release failure cleans its candidate before primary rollback."""

    _db, store = hold_db
    applied = await store.set_hold(
        scope="agent",
        target_id="did:agent:failed-release",
        actor_id="did:sovereign:operator",
        reason="hold remains active",
        operation_id="hold-before-failed-release",
    )
    stage = store._stage_history_candidate

    def fail_after_staging(payload):
        stage(payload)
        raise RuntimeError("injected release failure after anchor staging")

    monkeypatch.setattr(store, "_stage_history_candidate", fail_after_staging)
    with pytest.raises(Exception, match="release failure after anchor staging"):
        await store.release_hold(
            scope="agent",
            target_id="did:agent:failed-release",
            actor_id="did:sovereign:operator",
            reason="must roll back release",
            operation_id="rolled-back-release",
            expected_hold_receipt_id=applied.receipt.receipt_id,
        )

    monkeypatch.setattr(store, "_stage_history_candidate", stage)
    restored = await store.get_hold("agent", "did:agent:failed-release")
    assert restored is not None
    assert restored.hold_receipt_id == applied.receipt.receipt_id
    assert await store.get_receipt("rolled-back-release") is None


@pytest.mark.asyncio
async def test_committed_sqlite_mutation_recovers_interrupted_anchor_promotion(
    hold_db, monkeypatch
):
    """A committed candidate can finish publication on the next live path."""

    _db, store = hold_db
    finish = store._finish_history_publication

    def interrupt_promotion(_payload):
        raise RuntimeError("injected interruption before anchor promotion")

    monkeypatch.setattr(store, "_finish_history_publication", interrupt_promotion)
    with pytest.raises(RuntimeError, match="before anchor promotion"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:committed-candidate",
            actor_id="did:sovereign:operator",
            reason="recover committed publication",
            operation_id="committed-candidate",
        )

    monkeypatch.setattr(store, "_finish_history_publication", finish)
    recovered = await store.get_hold("agent", "did:agent:committed-candidate")
    assert recovered is not None
    assert recovered.reason == "recover committed publication"
    assert await store.get_receipt("committed-candidate") is not None


@pytest.mark.asyncio
async def test_sqlite_staged_evidence_rejects_ambiguous_primary_restore(
    tmp_path, monkeypatch
):
    """A staged candidate cannot be discarded after an ambiguous DB restore."""

    path = tmp_path / "ambiguous-publication.db"
    backup = tmp_path / "before-committed-hold.db"
    db = await AsyncDatabase.sqlite(str(path))
    store = HoldStore(db)
    await store.ensure_schema()
    await db.close()
    shutil.copyfile(path, backup)

    db = await AsyncDatabase.sqlite(str(path))
    store = HoldStore(db)
    await store.ensure_schema()

    def interrupt_promotion(_payload):
        raise RuntimeError("injected interruption after primary commit")

    monkeypatch.setattr(store, "_finish_history_publication", interrupt_promotion)
    with pytest.raises(RuntimeError, match="after primary commit"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:ambiguous-publication",
            actor_id="did:sovereign:operator",
            reason="must not disappear after restore",
            operation_id="ambiguous-publication",
        )
    await db.close()

    shutil.copyfile(backup, path)
    restored = await AsyncDatabase.sqlite(str(path))
    try:
        with pytest.raises(
            HoldCorruptStateError,
            match="ambiguous staged Hold history publication",
        ):
            await HoldStore(restored).ensure_schema()
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_postgres_external_candidate_recovers_committed_mutation(
    tmp_path,
    monkeypatch,
):
    """The external candidate bridges a crash after the primary PG commit."""

    primary = await AsyncDatabase.sqlite(str(tmp_path / "pg-primary-facade.db"))
    evidence = await AsyncDatabase.sqlite(str(tmp_path / "pg-evidence-facade.db"))
    store = HoldStore(primary)
    await store.ensure_schema()

    class _EvidenceFacade:
        backend_type = "postgres"

        async def fetchall(self, query, params=()):
            return await evidence.fetchall(query, params)

        async def execute(self, query, params=()):
            return await evidence.execute(query, params)

    @asynccontextmanager
    async def evidence_lock():
        yield

    # Keep the real SQLite primary so the mutation is a measured transaction,
    # but route its publication protocol through the same independent-DB code
    # used by PostgreSQL. This makes the crash boundary deterministic in unit
    # tests without requiring a second PostgreSQL service.
    store._history_anchor_path = None
    store._history_candidate_path = None
    store._bootstrap_intent_path = None
    store._evidence_lock_path = None
    store._evidence_db = _EvidenceFacade()
    monkeypatch.setattr(store, "_postgres_evidence_lock", evidence_lock)

    async def read_external_anchor():
        payload = await store._read_postgres_evidence(
            "hold_history_anchor_v1",
            label="history anchor",
        )
        if payload is None:
            return None
        return store._validate_history_anchor_payload(payload)

    monkeypatch.setattr(store, "_read_history_anchor", read_external_anchor)
    await store._write_postgres_evidence(
        "hold_history_anchor_v1",
        await store._current_history_anchor_payload(),
    )

    complete = store._complete_history_publication

    async def interrupt_promotion(_payload):
        raise RuntimeError("injected interruption after primary commit")

    monkeypatch.setattr(store, "_complete_history_publication", interrupt_promotion)
    with pytest.raises(RuntimeError, match="after primary commit"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:pg-publication",
            actor_id="did:sovereign:operator",
            reason="recover external candidate",
            operation_id="pg-external-candidate",
        )

    assert await store._read_external_history_candidate() is not None
    monkeypatch.setattr(store, "_complete_history_publication", complete)
    try:
        recovered = await store.get_hold("agent", "did:agent:pg-publication")
        assert recovered is not None
        assert recovered.reason == "recover external candidate"
        assert await store._read_external_history_candidate() is None
        assert await store._read_history_anchor() == (
            await store._current_history_anchor_payload()
        )
    finally:
        await primary.close()
        await evidence.close()


@pytest.mark.asyncio
async def test_sqlite_reader_waits_for_database_and_anchor_publication(
    tmp_path,
    monkeypatch,
):
    """A peer reader cannot observe either half of an evidence transition."""

    path = tmp_path / "serialized-publication.db"
    writer_db = await AsyncDatabase.sqlite(str(path))
    reader_db = await AsyncDatabase.sqlite(str(path))
    writer = HoldStore(writer_db)
    reader = HoldStore(reader_db)
    await writer.ensure_schema()
    candidate_staged = asyncio.Event()
    allow_commit = asyncio.Event()
    prepare = writer._prepare_history_publication

    async def pause_after_staging():
        payload = await prepare()
        candidate_staged.set()
        await allow_commit.wait()
        return payload

    monkeypatch.setattr(writer, "_prepare_history_publication", pause_after_staging)
    mutation = asyncio.create_task(
        writer.set_hold(
            scope="agent",
            target_id="did:agent:serialized-reader",
            actor_id="did:sovereign:operator",
            reason="publish as one protocol",
            operation_id="serialized-reader-hold",
        )
    )
    await candidate_staged.wait()
    read = asyncio.create_task(
        reader.get_hold("agent", "did:agent:serialized-reader")
    )
    await asyncio.sleep(0.05)
    reader_waited = not read.done()
    allow_commit.set()
    try:
        result, observed = await asyncio.gather(mutation, read)
        assert reader_waited
        assert observed == result.current
    finally:
        await writer_db.close()
        await reader_db.close()


@pytest.mark.asyncio
async def test_corrupt_active_latch_fails_closed(hold_db, monkeypatch):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_read_latch_row",
        AsyncMock(
            return_value=(
                "agent",
                "did:agent:kite",
                2,
                "receipt",
                "reason",
                "actor",
                "2026-08-28T00:00:00+00:00",
                1,
            )
        ),
    )
    with pytest.raises(HoldCorruptStateError, match="active flag"):
        await store.get_hold("agent", "did:agent:kite")


@pytest.mark.asyncio
async def test_greenfield_schema_rejects_foreign_host_target(hold_db):
    db, _store = hold_db
    with pytest.raises(Exception, match="CHECK constraint"):
        await db.execute(
            "INSERT INTO hold_latches (scope, target_id) VALUES (?, ?)",
            ("host", "foreign"),
        )
    with pytest.raises(Exception, match="CHECK constraint"):
        await db.execute(
            "INSERT INTO hold_receipts ("
            "receipt_id, operation_id, action, disposition, scope, target_id, "
            "reason, actor_id, occurred_at, expected_hold_receipt_id, "
            "prior_hold_receipt_id, resulting_hold_receipt_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "foreign-host-receipt",
                "foreign-host-operation",
                "hold",
                "applied",
                "host",
                "foreign",
                "invalid imported authority",
                "did:sovereign:operator",
                "2026-08-28T00:00:00+00:00",
                "",
                "",
                "foreign-host-receipt",
            ),
        )


@pytest.mark.asyncio
async def test_existing_foreign_host_receipt_fails_closed_on_every_state_path(
    hold_db,
):
    """Runtime validation remains load-bearing for upgraded/imported schemas."""

    db, store = hold_db
    await db.execute("PRAGMA ignore_check_constraints = ON")
    await db.execute(
        "INSERT INTO hold_receipts ("
        "receipt_id, operation_id, action, disposition, scope, target_id, "
        "reason, actor_id, occurred_at, expected_hold_receipt_id, "
        "prior_hold_receipt_id, resulting_hold_receipt_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "imported-foreign-host-receipt",
            "imported-foreign-host-operation",
            "hold",
            "applied",
            "host",
            "foreign",
            "invalid imported authority",
            "did:sovereign:operator",
            "2026-08-28T00:00:00+00:00",
            "",
            "",
            "imported-foreign-host-receipt",
        ),
    )
    await db.execute("PRAGMA ignore_check_constraints = OFF")

    with pytest.raises(HoldCorruptStateError, match="foreign target"):
        await store.get_hold("host")
    with pytest.raises(HoldCorruptStateError, match="foreign target"):
        await store.get_effective("did:agent:kite")
    with pytest.raises(HoldCorruptStateError, match="foreign target"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="must fail closed",
            operation_id="foreign-host-receipt-mutation",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("read", ["one", "effective"])
async def test_state_read_validates_projection_inside_one_locked_snapshot(
    hold_db,
    monkeypatch,
    read,
):
    db, store = hold_db
    held = await store.set_hold(
        scope="agent",
        target_id="did:agent:snapshot",
        actor_id="did:sovereign:operator",
        reason="snapshot authority",
        operation_id="snapshot-hold",
    )
    original_transaction = db.transaction
    original_validate = store._validate_latch_projection
    active = False
    transaction_modes: list[bool] = []
    locked_targets: list[tuple[tuple[HoldScope, str], ...]] = []

    @asynccontextmanager
    async def traced_transaction(*, immediate=False):
        nonlocal active
        transaction_modes.append(immediate)
        async with original_transaction(immediate=immediate):
            active = True
            try:
                yield
            finally:
                active = False

    async def inspect_projection(
        latch,
        scope,
        target_id,
        *,
        validate_global_history=True,
    ):
        assert active, "latch and receipt graph escaped their read snapshot"
        return await original_validate(
            latch,
            scope,
            target_id,
            validate_global_history=validate_global_history,
        )

    async def inspect_locks(targets):
        locked_targets.append(tuple(targets))

    monkeypatch.setattr(db, "transaction", traced_transaction)
    monkeypatch.setattr(store, "_validate_latch_projection", inspect_projection)
    monkeypatch.setattr(store, "_lock_read_targets", inspect_locks)

    if read == "one":
        assert await store.get_hold("agent", "did:agent:snapshot") == held.current
        assert locked_targets == [((HoldScope.AGENT, "did:agent:snapshot"),)]
    else:
        effective = await store.get_effective("did:agent:snapshot")
        assert effective.agent == held.current
        assert locked_targets == [
            (
                (HoldScope.HOST, "host"),
                (HoldScope.AGENT, "did:agent:snapshot"),
            )
        ]
    assert transaction_modes == [False]


@pytest.mark.asyncio
async def test_postgres_read_targets_take_shared_locks_in_global_order():
    execute = AsyncMock()
    store = HoldStore(
        SimpleNamespace(backend_type="postgres", execute=execute),
        evidence_db=SimpleNamespace(backend_type="postgres"),
    )

    await store._lock_read_targets(
        (
            (HoldScope.HOST, "host"),
            (HoldScope.AGENT, "did:agent:snapshot"),
            (HoldScope.HOST, "host"),
        )
    )

    assert [call.args for call in execute.await_args_list] == [
        (
            "SELECT pg_advisory_xact_lock_shared(hashtextextended(?, 0))",
            ("kestrel:hold:history-anchor",),
        ),
        (
            "SELECT pg_advisory_xact_lock_shared(hashtextextended(?, 0))",
            ("kestrel:hold:target:agent:did:agent:snapshot",),
        ),
        (
            "SELECT pg_advisory_xact_lock_shared(hashtextextended(?, 0))",
            ("kestrel:hold:target:host:host",),
        ),
    ]


@pytest.mark.asyncio
async def test_postgres_evidence_lock_rejects_same_database_identity():
    """Two pools do not become independent merely by being distinct objects."""

    @asynccontextmanager
    async def advisory_locks(_keys):
        yield

    identity = (
        "kestrel-hold-rollback-domain-v1:"
        "00000000-0000-0000-0000-000000000001"
    )
    primary = _PostgresCustodyFacade(
        "cluster-primary",
        domain_identity=identity,
    )
    evidence = _PostgresCustodyFacade(
        "cluster-evidence",
        domain_identity=identity,
        backend=SimpleNamespace(advisory_locks=advisory_locks),
    )
    store = HoldStore(primary, evidence_db=evidence)

    with pytest.raises(HoldStateError, match="independent rollback domain"):
        async with store._postgres_evidence_lock():
            pass


@pytest.mark.asyncio
async def test_postgres_evidence_lock_uses_independent_service_session():
    """The external session lock spans the caller's whole evidence boundary."""

    events: list[object] = []

    @asynccontextmanager
    async def advisory_locks(keys):
        events.append(("lock", keys))
        try:
            yield
        finally:
            events.append("unlock")

    primary = _PostgresCustodyFacade(
        "cluster-primary",
        domain_identity=(
            "kestrel-hold-rollback-domain-v1:"
            "00000000-0000-0000-0000-000000000001"
        ),
    )
    evidence = _PostgresCustodyFacade(
        "cluster-evidence",
        domain_identity=(
            "kestrel-hold-rollback-domain-v1:"
            "00000000-0000-0000-0000-000000000002"
        ),
        backend=SimpleNamespace(advisory_locks=advisory_locks),
    )
    store = HoldStore(primary, evidence_db=evidence)

    async with store._postgres_evidence_lock():
        events.append("body")

    assert events == [
        ("lock", ((0x004B4553, 0x484F4C44),)),
        "body",
        "unlock",
    ]


@pytest.mark.asyncio
async def test_postgres_evidence_lock_founds_distinct_domain_markers():
    """Fresh databases get durable identities before their first protocol lock."""

    @asynccontextmanager
    async def advisory_locks(_keys):
        yield

    primary = _PostgresCustodyFacade("cluster-primary")
    evidence = _PostgresCustodyFacade(
        "cluster-evidence",
        backend=SimpleNamespace(advisory_locks=advisory_locks),
    )
    store = HoldStore(primary, evidence_db=evidence)

    async with store._postgres_evidence_lock():
        pass

    primary_domain = primary.metadata["hold_rollback_domain_id_v1"]
    evidence_domain = evidence.metadata["hold_rollback_domain_id_v1"]
    assert primary_domain != evidence_domain


@pytest.mark.asyncio
async def test_postgres_evidence_lock_rejects_two_databases_in_one_cluster():
    """Different database markers do not prove an independent restore domain."""

    @asynccontextmanager
    async def advisory_locks(_keys):
        yield

    primary = _PostgresCustodyFacade("cluster-one")
    evidence = _PostgresCustodyFacade(
        "cluster-one",
        backend=SimpleNamespace(advisory_locks=advisory_locks),
    )

    with pytest.raises(HoldStateError, match="independent.*cluster"):
        async with HoldStore(primary, evidence_db=evidence)._postgres_evidence_lock():
            pass


@pytest.mark.asyncio
async def test_postgres_evidence_lock_rejects_swapped_custody_roles():
    """A configured evidence service can never become the protected primary."""

    @asynccontextmanager
    async def advisory_locks(_keys):
        yield

    backend = SimpleNamespace(advisory_locks=advisory_locks)
    primary = _PostgresCustodyFacade("cluster-primary", backend=backend)
    evidence = _PostgresCustodyFacade("cluster-evidence", backend=backend)
    async with HoldStore(primary, evidence_db=evidence)._postgres_evidence_lock():
        pass

    with pytest.raises(HoldStateError, match="custody role|binding"):
        async with HoldStore(evidence, evidence_db=primary)._postgres_evidence_lock():
            pass


def _custody_domain(number: int) -> str:
    return (
        "kestrel-hold-rollback-domain-v1:"
        f"00000000-0000-0000-0000-{number:012d}"
    )


def test_postgres_custody_preflight_accepts_fresh_independent_pair():
    validate_postgres_hold_custody(
        PostgresHoldCustodySnapshot("cluster-primary"),
        PostgresHoldCustodySnapshot("cluster-evidence"),
    )


def test_postgres_custody_preflight_rejects_wrong_durable_role():
    with pytest.raises(HoldStateError, match="wrong durable custody role"):
        validate_postgres_hold_custody(
            PostgresHoldCustodySnapshot(
                "cluster-primary",
                evidence_binding="old-evidence-role",
            ),
            PostgresHoldCustodySnapshot("cluster-evidence"),
        )


def test_postgres_custody_preflight_rejects_binding_without_domains():
    with pytest.raises(HoldStateError, match="lacks a durable rollback domain"):
        validate_postgres_hold_custody(
            PostgresHoldCustodySnapshot(
                "cluster-primary",
                primary_binding="persisted-binding",
            ),
            PostgresHoldCustodySnapshot("cluster-evidence"),
        )


def test_postgres_custody_preflight_accepts_valid_partial_publication():
    primary_domain = _custody_domain(1)
    evidence_domain = _custody_domain(2)
    binding = postgres_hold_custody_binding_payload(
        uuid4(),
        primary_domain,
        evidence_domain,
    )

    validate_postgres_hold_custody(
        PostgresHoldCustodySnapshot(
            "cluster-primary",
            domain_identity=primary_domain,
            primary_binding=binding,
        ),
        PostgresHoldCustodySnapshot(
            "cluster-evidence",
            domain_identity=evidence_domain,
        ),
    )


def test_postgres_custody_preflight_rejects_binding_for_another_pair():
    primary_domain = _custody_domain(1)
    evidence_domain = _custody_domain(2)
    binding = postgres_hold_custody_binding_payload(
        uuid4(),
        primary_domain,
        _custody_domain(3),
    )

    with pytest.raises(HoldStateError, match="does not match the configured pair"):
        validate_postgres_hold_custody(
            PostgresHoldCustodySnapshot(
                "cluster-primary",
                domain_identity=primary_domain,
                primary_binding=binding,
            ),
            PostgresHoldCustodySnapshot(
                "cluster-evidence",
                domain_identity=evidence_domain,
            ),
        )


def test_postgres_custody_preflight_rejects_disagreeing_bindings():
    primary_domain = _custody_domain(1)
    evidence_domain = _custody_domain(2)
    primary_binding = postgres_hold_custody_binding_payload(
        uuid4(), primary_domain, evidence_domain
    )
    evidence_binding = postgres_hold_custody_binding_payload(
        uuid4(), primary_domain, evidence_domain
    )

    with pytest.raises(HoldStateError, match="disagrees between databases"):
        validate_postgres_hold_custody(
            PostgresHoldCustodySnapshot(
                "cluster-primary",
                domain_identity=primary_domain,
                primary_binding=primary_binding,
            ),
            PostgresHoldCustodySnapshot(
                "cluster-evidence",
                domain_identity=evidence_domain,
                evidence_binding=evidence_binding,
            ),
        )


@pytest.mark.asyncio
async def test_postgres_custody_preflight_is_read_only_and_closes_raw_pools(
    monkeypatch,
):
    """The guard runs before schema init and owns both temporary pools."""

    from kestrel_sovereign.storage.db import postgres as postgres_module

    backends = []

    class _Backend:
        def __init__(self, *, dsn, min_pool_size, max_pool_size):
            self.dsn = dsn
            self.closed = False
            self.queries = []
            assert (min_pool_size, max_pool_size) == (1, 1)
            backends.append(self)

        async def connect(self):
            return None

        async def fetch_all(self, query, params=()):
            self.queries.append((query, params))
            assert query.lstrip().startswith("SELECT")
            if "pg_control_system" in query:
                return [(self.dsn,)]
            if "to_regclass" in query:
                return [(None,)]
            raise AssertionError(f"unexpected preflight query: {query}")

        async def close(self):
            self.closed = True

    monkeypatch.setattr(postgres_module, "PostgresBackend", _Backend)

    await preflight_postgres_hold_custody(
        "postgresql://primary/db",
        "postgresql://evidence/db",
    )

    assert len(backends) == 2
    assert all(backend.closed for backend in backends)
    assert all(len(backend.queries) == 2 for backend in backends)


@pytest.mark.asyncio
async def test_postgres_hold_schema_initializes_preflight_validated_backends(
    monkeypatch,
):
    """Custody proof and schema writes retain one connected-pool identity."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db import postgres as postgres_module

    backends = []
    events: list[tuple[str, object]] = []

    class _Backend:
        def __init__(self, *, dsn, min_pool_size, max_pool_size):
            self.dsn = dsn
            self.validated = False
            assert (min_pool_size, max_pool_size) == (1, 1)
            backends.append(self)

        async def connect(self):
            events.append(("connect", self))

        async def fetch_all(self, query, params=()):
            events.append(("validate", self))
            if "pg_control_system" in query:
                return [(self.dsn,)]
            if "to_regclass" in query:
                self.validated = True
                return [(None,)]
            raise AssertionError(f"unexpected custody query: {query}")

        async def close(self):
            events.append(("close", self))

    class _DB:
        def __init__(self, backend):
            self.backend = backend

        async def close(self):
            await self.backend.close()

    async def _from_connected(_cls, backend):
        assert backend.validated
        events.append(("schema", backend))
        return _DB(backend)

    async def _reopen_by_dsn(*_args, **_kwargs):
        pytest.fail("validated Hold pools were discarded before schema init")

    monkeypatch.setattr(postgres_module, "PostgresBackend", _Backend)
    monkeypatch.setattr(
        AsyncDatabase,
        "from_connected_backend",
        classmethod(_from_connected),
    )
    monkeypatch.setattr(
        AsyncDatabase,
        "postgres",
        classmethod(_reopen_by_dsn),
    )

    primary, evidence = await initialize_postgres_hold_databases(
        "postgresql://primary/db",
        "postgresql://evidence/db",
    )

    assert len(backends) == 2
    assert primary.backend is backends[0]
    assert evidence.backend is backends[1]
    for backend in backends:
        validation = max(
            index
            for index, event in enumerate(events)
            if event == ("validate", backend)
        )
        schema = events.index(("schema", backend))
        assert validation < schema


@pytest.mark.asyncio
async def test_postgres_hold_pair_initializer_closes_partial_schema_open(
    monkeypatch,
):
    """A failed second schema cannot strand either validated backend."""

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db import postgres as postgres_module

    backends = []

    class _Backend:
        def __init__(self, *, dsn, min_pool_size, max_pool_size):
            self.dsn = dsn
            self.close_count = 0
            assert (min_pool_size, max_pool_size) == (1, 1)
            backends.append(self)

        async def connect(self):
            return None

        async def fetch_all(self, query, params=()):
            if "pg_control_system" in query:
                return [(self.dsn,)]
            if "to_regclass" in query:
                return [(None,)]
            raise AssertionError(f"unexpected custody query: {query}")

        async def close(self):
            self.close_count += 1

    class _DB:
        def __init__(self, backend):
            self.backend = backend

        async def close(self):
            await self.backend.close()

    async def _from_connected(_cls, backend):
        if "evidence" in backend.dsn:
            raise RuntimeError("injected evidence schema failure")
        return _DB(backend)

    monkeypatch.setattr(postgres_module, "PostgresBackend", _Backend)
    monkeypatch.setattr(
        AsyncDatabase,
        "from_connected_backend",
        classmethod(_from_connected),
    )

    with pytest.raises(RuntimeError, match="evidence schema failure"):
        await initialize_postgres_hold_databases(
            "postgresql://primary/db",
            "postgresql://evidence/db",
        )

    assert len(backends) == 2
    assert [backend.close_count for backend in backends] == [1, 1]


@pytest.mark.asyncio
async def test_postgres_evidence_lock_recovers_partial_pair_binding(monkeypatch):
    """A crash between role writes resumes only the already-declared pair."""

    @asynccontextmanager
    async def advisory_locks(_keys):
        yield

    backend = SimpleNamespace(advisory_locks=advisory_locks)
    primary = _PostgresCustodyFacade("cluster-primary", backend=backend)
    evidence = _PostgresCustodyFacade("cluster-evidence", backend=backend)
    interrupted = HoldStore(primary, evidence_db=evidence)
    write = interrupted._write_postgres_binding

    async def crash_before_evidence_binding(db, key, payload):
        if db is evidence:
            raise RuntimeError("injected crash before evidence role binding")
        await write(db, key, payload)

    monkeypatch.setattr(
        interrupted,
        "_write_postgres_binding",
        crash_before_evidence_binding,
    )
    with pytest.raises(RuntimeError, match="before evidence role binding"):
        async with interrupted._postgres_evidence_lock():
            pass

    assert "hold_primary_custody_binding_v1" in primary.metadata
    assert "hold_evidence_custody_binding_v1" not in evidence.metadata

    async with HoldStore(primary, evidence_db=evidence)._postgres_evidence_lock():
        pass
    assert (
        primary.metadata["hold_primary_custody_binding_v1"]
        == evidence.metadata["hold_evidence_custody_binding_v1"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "fanout"),
    [("cluster", 2), ("domain", 2), ("bindings", 4)],
)
async def test_failed_parallel_custody_probe_awaits_cancelled_siblings(
    monkeypatch,
    phase,
    fanout,
):
    """Startup cleanup never races database probes left behind by a peer error."""

    primary = SimpleNamespace(backend_type="postgres")
    evidence = SimpleNamespace(backend_type="postgres")
    store = HoldStore(primary, evidence_db=evidence)
    all_started = asyncio.Event()
    never_finishes = asyncio.Event()
    started = 0
    cancelled: set[int] = set()

    async def probe(*_args, **_kwargs):
        nonlocal started
        probe_id = started
        started += 1
        if started == fanout:
            all_started.set()
        await all_started.wait()
        if probe_id == 0:
            raise RuntimeError("injected custody probe failure")
        try:
            await never_finishes.wait()
        finally:
            cancelled.add(probe_id)

    if phase == "cluster":
        monkeypatch.setattr(store, "_postgres_cluster_identity", probe)
        operation = store._assert_postgres_clusters_independent()
    elif phase == "domain":
        monkeypatch.setattr(store, "_postgres_domain_identity", probe)
        operation = store._assert_postgres_evidence_domain_independent()
    else:
        monkeypatch.setattr(store, "_read_postgres_binding", probe)
        operation = store._read_postgres_custody_roles(evidence)

    with pytest.raises(RuntimeError, match="injected custody probe failure"):
        await operation

    assert cancelled == set(range(1, fanout))


@pytest.mark.asyncio
async def test_postgres_public_read_uses_external_evidence_protocol(monkeypatch):
    """A live read cannot bypass recovery or the cross-service session lock."""

    events: list[str] = []
    store = HoldStore(
        SimpleNamespace(backend_type="postgres"),
        evidence_db=SimpleNamespace(backend_type="postgres"),
    )

    @asynccontextmanager
    async def evidence_lock():
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    async def recover():
        events.append("recover")

    async def read(_scope, _target_id):
        events.append("read")
        return None

    monkeypatch.setattr(store, "_postgres_evidence_lock", evidence_lock)
    monkeypatch.setattr(store, "_recover_history_publication", recover)
    monkeypatch.setattr(store, "_get_hold", read)

    assert await store.get_hold("agent", "did:agent:protocol") is None
    assert events == ["lock", "recover", "read", "unlock"]


@pytest.mark.asyncio
async def test_postgres_missing_receipt_locks_history_before_validation(monkeypatch):
    """An absent operation is proved only inside the global history lock."""

    events: list[str] = []

    @asynccontextmanager
    async def transaction():
        yield

    store = HoldStore(
        SimpleNamespace(backend_type="postgres", transaction=transaction),
        evidence_db=SimpleNamespace(backend_type="postgres"),
    )

    async def lock_history():
        events.append("history-lock")

    async def validate(_operation_id):
        events.append("validate-operation")
        return None

    async def validate_global_history():
        events.append("validate-global-history")

    monkeypatch.setattr(store, "_lock_read_history", lock_history, raising=False)
    monkeypatch.setattr(store, "_validate_operation_witness", validate)
    monkeypatch.setattr(
        store,
        "_assert_global_history_intact",
        validate_global_history,
    )

    assert await store._get_receipt("missing-operation") is None
    assert events == [
        "history-lock",
        "validate-operation",
        "validate-global-history",
    ]


@pytest.mark.asyncio
async def test_postgres_writers_serialize_global_history_before_local_keys():
    """Cross-target writers cannot publish lost-update history heads."""

    execute = AsyncMock()
    store = HoldStore(
        SimpleNamespace(backend_type="postgres", execute=execute),
        evidence_db=SimpleNamespace(backend_type="postgres"),
    )

    await store._lock_operation_and_target(
        "operation-one",
        HoldScope.AGENT,
        "did:agent:snapshot",
    )

    assert [call.args for call in execute.await_args_list] == [
        (
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            ("kestrel:hold:history-anchor",),
        ),
        (
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            ("kestrel:hold:operation:operation-one",),
        ),
        (
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            ("kestrel:hold:target:agent:did:agent:snapshot",),
        ),
    ]


def test_authority_graph_walk_is_linear_in_number_of_receipts():
    class CountingConsumers(dict[str, HoldReceipt]):
        get_count = 0

        def get(self, key, default=None):
            self.get_count += 1
            return super().get(key, default)

    authorities: dict[str, HoldReceipt] = {}
    consumers = CountingConsumers()
    previous = ""
    count = 2_000
    for index in range(count):
        receipt_id = f"hold-{index}"
        receipt = HoldReceipt(
            receipt_id=receipt_id,
            operation_id=f"operation-{index}",
            action=HoldAction.HOLD,
            disposition=HoldDisposition.APPLIED,
            scope=HoldScope.AGENT,
            target_id="did:agent:linear",
            reason="linear history",
            actor_id="did:sovereign:operator",
            occurred_at=f"2026-08-28T00:00:{index:05d}+00:00",
            expected_hold_receipt_id="",
            prior_hold_receipt_id=previous,
            resulting_hold_receipt_id=receipt_id,
        )
        authorities[receipt_id] = receipt
        if previous:
            consumers[previous] = receipt
        previous = receipt_id

    assert _terminal_authority_ids(authorities, consumers) == {previous}
    assert consumers.get_count <= count


@pytest.mark.asyncio
async def test_upgraded_foreign_host_row_fails_closed_for_reads_and_mutation(
    tmp_path,
):
    """Existing v1 tables lack the new CHECK, so runtime validation is load-bearing."""

    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy-hold.db"))
    await db.execute(
        "CREATE TABLE hold_latches ("
        "scope TEXT NOT NULL, target_id TEXT NOT NULL, "
        "active INTEGER NOT NULL DEFAULT 0, "
        "hold_receipt_id TEXT NOT NULL DEFAULT '', "
        "reason TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', "
        "set_at TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0, "
        "PRIMARY KEY (scope, target_id), "
        "CHECK (scope IN ('host', 'agent')), CHECK (active IN (0, 1)), "
        "CHECK (revision >= 0))"
    )
    await db.execute(
        "INSERT INTO hold_latches "
        "(scope, target_id, active, hold_receipt_id, reason, actor_id, set_at) "
        "VALUES ('host', 'foreign', 1, 'receipt', 'reason', 'actor', 'time')"
    )
    store = HoldStore(db)
    await store.ensure_schema()
    try:
        with pytest.raises(HoldCorruptStateError, match="foreign target"):
            await store.get_hold("host")
        with pytest.raises(HoldCorruptStateError, match="foreign target"):
            await store.get_effective("did:agent:kite")
        with pytest.raises(HoldCorruptStateError, match="foreign target"):
            await store.set_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:operator",
                reason="must fail closed",
                operation_id="foreign-host-row",
            )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_external_initialization_witness_rejects_total_hold_schema_loss(
    tmp_path,
):
    """The tables cannot erase the only evidence that Hold was initialized."""

    path = tmp_path / "replaceable-hold.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    first = HoldStore(first_db)
    await first.ensure_schema()
    await first.set_hold(
        scope="agent",
        target_id="did:agent:held-before-loss",
        actor_id="did:sovereign:operator",
        reason="must survive replacement",
        operation_id="hold-before-total-schema-loss",
    )
    await first_db.close()

    second_db = await AsyncDatabase.sqlite(str(path))
    try:
        for table in (
            "hold_operation_witnesses",
            "hold_receipt_content_witnesses",
            "hold_receipt_witnesses",
            "hold_receipts",
            "hold_latches",
            "hold_schema_migrations",
        ):
            await second_db.execute(f"DROP TABLE {table}")

        with pytest.raises(HoldCorruptStateError, match="schema.*missing"):
            await HoldStore(second_db).ensure_schema()
    finally:
        await second_db.close()


@pytest.mark.asyncio
async def test_external_history_anchor_rejects_empty_initialized_backup_restore(
    tmp_path,
):
    """Rolling back only the database cannot erase a later durable Hold."""

    path = tmp_path / "rollback-hold.db"
    backup = tmp_path / "empty-initialized-hold.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    first = HoldStore(first_db)
    await first.ensure_schema()
    await first_db.close()
    shutil.copyfile(path, backup)

    held_db = await AsyncDatabase.sqlite(str(path))
    held_store = HoldStore(held_db)
    await held_store.ensure_schema()
    await held_store.set_hold(
        scope="agent",
        target_id="did:agent:rolled-back",
        actor_id="did:sovereign:operator",
        reason="must survive rollback",
        operation_id="hold-before-empty-restore",
    )
    await held_db.close()

    shutil.copyfile(backup, path)
    restored_db = await AsyncDatabase.sqlite(str(path))
    try:
        with pytest.raises(HoldCorruptStateError, match="history anchor"):
            await HoldStore(restored_db).ensure_schema()
    finally:
        await restored_db.close()


@pytest.mark.asyncio
async def test_postgres_external_anchor_rejects_primary_snapshot_rollback(
    tmp_path,
):
    """A primary restore cannot restore the independent evidence head with it."""

    primary_path = tmp_path / "postgres-primary-facade.db"
    backup_path = tmp_path / "postgres-primary-empty-backup.db"
    evidence = await AsyncDatabase.sqlite(
        str(tmp_path / "postgres-independent-evidence-facade.db")
    )

    class _PostgresFacade:
        backend_type = "postgres"

        def __init__(self, inner):
            self._inner = inner

        async def fetchall(self, query, params=()):
            return await self._inner.fetchall(query, params)

        async def execute(self, query, params=()):
            return await self._inner.execute(query, params)

    evidence_facade = _PostgresFacade(evidence)
    primary = await AsyncDatabase.sqlite(str(primary_path))
    await _create_legacy_hold_tables(primary)
    empty = HoldStore(_PostgresFacade(primary), evidence_db=evidence_facade)
    await empty._write_history_anchor()
    await empty._write_initialization_witness()
    await primary.close()
    shutil.copyfile(primary_path, backup_path)

    held_primary = await AsyncDatabase.sqlite(str(primary_path))
    held = HoldStore(
        _PostgresFacade(held_primary),
        evidence_db=evidence_facade,
    )
    await held_primary.execute(
        "INSERT INTO hold_receipts ("
        "receipt_id, operation_id, action, disposition, scope, target_id, "
        "reason, actor_id, occurred_at, expected_hold_receipt_id, "
        "prior_hold_receipt_id, resulting_hold_receipt_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "postgres-rollback-receipt",
            "postgres-rollback-operation",
            "hold",
            "applied",
            "agent",
            "did:agent:postgres-rollback",
            "must remain held",
            "did:sovereign:operator",
            "2026-08-31T00:00:00+00:00",
            "",
            "",
            "postgres-rollback-receipt",
        ),
    )
    await held._write_history_anchor()
    await held_primary.close()

    shutil.copyfile(backup_path, primary_path)
    restored_primary = await AsyncDatabase.sqlite(str(primary_path))
    restored = HoldStore(
        _PostgresFacade(restored_primary),
        evidence_db=evidence_facade,
    )
    try:
        with pytest.raises(HoldCorruptStateError, match="history anchor"):
            await restored._assert_history_anchor_intact()
    finally:
        await restored_primary.close()
        await evidence.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "malformed"])
async def test_initialized_store_rejects_missing_or_malformed_history_anchor(
    tmp_path,
    damage,
):
    path = tmp_path / f"damaged-anchor-{damage}.db"
    db = await AsyncDatabase.sqlite(str(path))
    try:
        store = HoldStore(db)
        await store.ensure_schema()
        anchor_path = store._history_anchor_path
        assert anchor_path is not None
        if damage == "missing":
            anchor_path.unlink()
        else:
            anchor_path.write_bytes(b"not a valid history anchor\n")

        with pytest.raises(HoldCorruptStateError, match="history anchor"):
            await store.ensure_schema()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_initialization_witness_is_not_visible_until_payload_is_complete(
    tmp_path,
    monkeypatch,
):
    """Concurrent boot can observe only absence or a complete witness."""

    from kestrel_sovereign.hold import state as hold_state_module

    database_path = tmp_path / "atomic-witness.db"
    witness_path = hold_initialization_witness_path(database_path)
    db = await AsyncDatabase.sqlite(str(database_path))
    observed_final_path: list[bool] = []
    real_write = os.write

    def observe_before_write(descriptor, payload):
        observed_final_path.append(witness_path.exists())
        return real_write(descriptor, payload)

    monkeypatch.setattr(hold_state_module.os, "write", observe_before_write)
    try:
        await HoldStore(db).ensure_schema()
    finally:
        await db.close()

    assert observed_final_path
    assert not any(observed_final_path)


@pytest.mark.asyncio
async def test_sqlite_evidence_protocol_uses_windows_byte_range_lock(
    tmp_path,
    monkeypatch,
):
    """Durable Hold remains available on the advertised Windows platform."""

    from kestrel_sovereign.hold import state as hold_state_module

    events: list[tuple[int, int]] = []

    class _FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(descriptor, mode, length):
            assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
            assert os.fstat(descriptor).st_size >= 1
            events.append((mode, length))

    monkeypatch.setattr(hold_state_module, "fcntl", None)
    monkeypatch.setattr(hold_state_module, "msvcrt", _FakeMsvcrt, raising=False)
    path = tmp_path / "windows-lock.db"
    db = await AsyncDatabase.sqlite(str(path))
    try:
        await HoldStore(db).ensure_schema()
    finally:
        await db.close()

    assert events == [(_FakeMsvcrt.LK_NBLCK, 1), (_FakeMsvcrt.LK_UNLCK, 1)]


@pytest.mark.asyncio
async def test_concurrent_sqlite_initializers_share_one_evidence_protocol(
    tmp_path,
    monkeypatch,
):
    """A peer boot waits while the first boot publishes durable evidence."""

    path = tmp_path / "concurrent-initialization.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    second_db = await AsyncDatabase.sqlite(str(path))
    first = HoldStore(first_db)
    second = HoldStore(second_db)
    publication_entered = asyncio.Event()
    allow_publication = asyncio.Event()
    publish = first._write_history_anchor

    async def paused_publication():
        publication_entered.set()
        await allow_publication.wait()
        await publish()

    monkeypatch.setattr(first, "_write_history_anchor", paused_publication)
    first_boot = asyncio.create_task(first.ensure_schema())
    await publication_entered.wait()
    second_boot = asyncio.create_task(second.ensure_schema())
    await asyncio.sleep(0.05)
    second_waited = not second_boot.done()
    allow_publication.set()
    results = await asyncio.gather(first_boot, second_boot, return_exceptions=True)
    try:
        assert second_waited
        assert results == [None, None]
    finally:
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_sqlite_bootstrap_intent_recovers_after_schema_commit(
    tmp_path,
    monkeypatch,
):
    """Committed schema without final evidence resumes only from prior intent."""

    path = tmp_path / "interrupted-bootstrap.db"
    db = await AsyncDatabase.sqlite(str(path))
    store = HoldStore(db)
    publish = store._write_history_anchor

    async def interrupt_publication():
        raise RuntimeError("injected interruption after schema commit")

    monkeypatch.setattr(store, "_write_history_anchor", interrupt_publication)
    with pytest.raises(RuntimeError, match="after schema commit"):
        await store.ensure_schema()

    monkeypatch.setattr(store, "_write_history_anchor", publish)
    try:
        await store.ensure_schema()
        assert await store.get_hold("host") is None
        assert store._bootstrap_intent_path is not None
        assert not store._bootstrap_intent_path.exists()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_bootstrap_intent_cannot_reanchor_different_receipt_history(tmp_path):
    """A restored bootstrap marker authorizes only its recorded receipt head."""

    path = tmp_path / "bootstrap-history-binding.db"
    db = await AsyncDatabase.sqlite(str(path))
    store = HoldStore(db)
    await store.ensure_schema()
    await store.set_hold(
        scope="host",
        actor_id="did:sovereign:operator",
        reason="survives evidence restore",
        operation_id="bootstrap-binding-hold",
    )
    assert store._initialization_witness_path is not None
    assert store._history_anchor_path is not None
    store._initialization_witness_path.unlink()
    store._history_anchor_path.unlink()
    store._write_bootstrap_intent(
        store._history_anchor_payload_from_rows([])
    )

    try:
        with pytest.raises(
            HoldCorruptStateError,
            match="bootstrap intent does not match receipt history",
        ):
            await store.ensure_schema()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_ensure_schema_does_not_self_chain_typed_failure(
    hold_db,
    monkeypatch,
):
    """A direct Hold refusal remains inspectable as a normal exception chain."""

    _db, store = hold_db
    refusal = HoldCorruptStateError("injected typed refusal")
    monkeypatch.setattr(
        store,
        "_ensure_external_schema_protocol",
        AsyncMock(side_effect=refusal),
    )

    with pytest.raises(HoldCorruptStateError) as caught:
        await store.ensure_schema()

    assert caught.value is refusal
    assert caught.value.__cause__ is not caught.value


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "set", "release"])
async def test_imported_duplicate_latch_fails_closed_on_single_target_paths(
    tmp_path,
    monkeypatch,
    operation,
):
    """A legacy table without its key cannot make one duplicate authoritative."""

    db = await AsyncDatabase.sqlite(str(tmp_path / f"duplicate-{operation}.db"))
    await db.execute(
        "CREATE TABLE hold_latches ("
        "scope TEXT NOT NULL, target_id TEXT NOT NULL, "
        "active INTEGER NOT NULL DEFAULT 0, "
        "hold_receipt_id TEXT NOT NULL DEFAULT '', "
        "reason TEXT NOT NULL DEFAULT '', actor_id TEXT NOT NULL DEFAULT '', "
        "set_at TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0)"
    )
    await db.execute(
        "INSERT INTO hold_latches (scope, target_id) VALUES (?, ?)",
        ("agent", "did:agent:duplicate"),
    )
    await db.execute(
        "INSERT INTO hold_latches (scope, target_id) VALUES (?, ?)",
        ("agent", "did:agent:duplicate"),
    )
    store = HoldStore(db)
    await store.ensure_schema()
    if operation != "get":
        # The imported rows already exist. Bypass the insert-if-absent helper,
        # whose ON CONFLICT target also depends on the missing legacy key, so
        # this regression reaches the shared single-latch read itself.
        monkeypatch.setattr(store, "_ensure_latch_row", AsyncMock())
    try:
        with pytest.raises(HoldCorruptStateError, match="duplicate hold latch key"):
            if operation == "get":
                await store.get_hold("agent", "did:agent:duplicate")
            elif operation == "set":
                await store.set_hold(
                    scope="agent",
                    target_id="did:agent:duplicate",
                    actor_id="did:sovereign:operator",
                    reason="must reject ambiguous projection",
                    operation_id="set-duplicate-projection",
                )
            else:
                await store.release_hold(
                    scope="agent",
                    target_id="did:agent:duplicate",
                    actor_id="did:sovereign:operator",
                    reason="must reject ambiguous projection",
                    operation_id="release-duplicate-projection",
                    expected_hold_receipt_id="unknown-authority",
                )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_inactive_latch_with_retained_evidence_fails_closed(
    hold_db,
    monkeypatch,
):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_read_latch_row",
        AsyncMock(
            return_value=(
                "agent",
                "did:agent:kite",
                0,
                "retained-receipt",
                "retained reason",
                "did:sovereign:operator",
                "2026-08-28T00:00:00+00:00",
                2,
            )
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="inactive latch"):
        await store.get_hold("agent", "did:agent:kite")


@pytest.mark.asyncio
async def test_malformed_applied_hold_replay_fails_closed(
    hold_db,
    monkeypatch,
):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_validate_operation_witness",
        AsyncMock(
            return_value=(
                "receipt-id",
                "poisoned-replay",
                "hold",
                "applied",
                "agent",
                "did:agent:kite",
                "investigate",
                "did:sovereign:operator",
                "2026-08-28T00:00:00+00:00",
                "",
                "",
                "",
            )
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="receipt invariant"):
        await store.set_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="investigate",
            operation_id="poisoned-replay",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "disposition", "expected", "prior", "resulting"),
    [
        ("hold", "refused_stale", "", "prior", "prior"),
        ("hold", "already_in_state", "", "", ""),
        ("release", "applied", "expected", "other", ""),
        ("release", "already_in_state", "expected", "prior", "prior"),
        ("release", "refused_stale", "expected", "expected", "expected"),
    ],
)
async def test_impossible_receipt_state_combinations_fail_closed(
    hold_db,
    monkeypatch,
    action,
    disposition,
    expected,
    prior,
    resulting,
):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_validate_operation_witness",
        AsyncMock(
            return_value=(
                "receipt-id",
                "corrupt-combination",
                action,
                disposition,
                "agent",
                "did:agent:kite",
                "investigate",
                "did:sovereign:operator",
                "2026-08-28T00:00:00+00:00",
                expected,
                prior,
                resulting,
            )
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="receipt invariant"):
        await store.get_receipt("corrupt-combination")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["set", "release"])
async def test_mutations_preserve_typed_corrupt_state_error(
    hold_db,
    monkeypatch,
    operation,
):
    _db, store = hold_db
    monkeypatch.setattr(
        store,
        "_read_latch_row",
        AsyncMock(
            return_value=(
                "agent",
                "did:agent:kite",
                2,
                "receipt",
                "reason",
                "actor",
                "2026-08-28T00:00:00+00:00",
                1,
            )
        ),
    )

    with pytest.raises(HoldCorruptStateError, match="active flag"):
        if operation == "set":
            await store.set_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:operator",
                reason="replace corrupt latch",
                operation_id="corrupt-set",
            )
        else:
            await store.release_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:operator",
                reason="release corrupt latch",
                operation_id="corrupt-release",
                expected_hold_receipt_id="receipt",
            )


def test_package_exports_shared_hold_state_error() -> None:
    assert issubclass(HoldCorruptStateError, HoldStateError)
    assert issubclass(HoldIdempotencyConflict, HoldStateError)


@pytest.mark.asyncio
async def test_two_sqlite_workers_serialize_replacement_and_stale_release(tmp_path):
    path = tmp_path / "host.db"
    db_one = await AsyncDatabase.sqlite(str(path))
    db_two = await AsyncDatabase.sqlite(str(path))
    one = HoldStore(db_one)
    two = HoldStore(db_two)
    await asyncio.gather(one.ensure_schema(), two.ensure_schema())
    try:
        results = await asyncio.gather(
            one.set_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:one",
                reason="one",
                operation_id="race-one",
            ),
            two.set_hold(
                scope="agent",
                target_id="did:agent:kite",
                actor_id="did:sovereign:two",
                reason="two",
                operation_id="race-two",
            ),
        )
        current = await one.get_hold("agent", "did:agent:kite")
        assert current is not None
        winner = next(item for item in results if item.current == current)
        loser = next(item for item in results if item.receipt != winner.receipt)

        stale = await two.release_hold(
            scope="agent",
            target_id="did:agent:kite",
            actor_id="did:sovereign:operator",
            reason="stale concurrent observation",
            operation_id="race-release",
            expected_hold_receipt_id=loser.receipt.receipt_id,
        )
        assert stale.receipt.disposition is HoldDisposition.REFUSED_STALE
        assert stale.current == current
    finally:
        await db_one.close()
        await db_two.close()


def _portable_hold_store(db_backend, tmp_path):
    """Build the SQL parity store with disposable, backend-neutral custody."""

    db = AsyncDatabase(db_backend)
    # This test exercises SQL portability, not production custody topology.
    # Give both parameter branches an explicit disposable sidecar so the
    # PostgreSQL case does not silently depend on an unavailable second
    # cluster while still exercising the file-evidence protocol.
    store = HoldStore(
        db,
        initialization_witness_path=tmp_path / "backend-parity.hold-initialized-v1",
    )
    return db, store


def test_postgres_portability_harness_provides_disposable_evidence(tmp_path):
    _db, store = _portable_hold_store(
        SimpleNamespace(backend_type="postgres"),
        tmp_path,
    )

    assert store._initialization_witness_path == (
        tmp_path / "backend-parity.hold-initialized-v1"
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_hold_store_sql_is_backend_portable(db_backend, tmp_path):
    db, store = _portable_hold_store(db_backend, tmp_path)
    await store.ensure_schema()
    suffix = uuid4().hex
    target = f"did:agent:{suffix}"
    hold_operation = f"hold-{suffix}"
    release_operation = f"release-{suffix}"
    try:
        held = await store.set_hold(
            scope="agent",
            target_id=target,
            actor_id="did:sovereign:operator",
            reason="backend parity",
            operation_id=hold_operation,
        )
        released = await store.release_hold(
            scope="agent",
            target_id=target,
            actor_id="did:sovereign:operator",
            reason="backend parity complete",
            operation_id=release_operation,
            expected_hold_receipt_id=held.receipt.receipt_id,
        )
        assert released.receipt.disposition is HoldDisposition.APPLIED
        assert await store.get_hold("agent", target) is None
    finally:
        await db.execute(
            "DELETE FROM hold_operation_witnesses "
            "WHERE operation_id IN (?, ?)",
            (hold_operation, release_operation),
        )
        await db.execute(
            "DELETE FROM hold_receipt_content_witnesses WHERE receipt_id IN ("
            "SELECT receipt_id FROM hold_receipts "
            "WHERE operation_id IN (?, ?))",
            (hold_operation, release_operation),
        )
        await db.execute(
            "DELETE FROM hold_receipts WHERE operation_id IN (?, ?)",
            (hold_operation, release_operation),
        )
        await db.execute(
            "DELETE FROM hold_latches WHERE scope = ? AND target_id = ?",
            (HoldScope.AGENT.value, target),
        )
        await db.execute(
            "DELETE FROM hold_receipt_witnesses "
            "WHERE scope = ? AND target_id = ?",
            (HoldScope.AGENT.value, target),
        )

    # A reusable PostgreSQL test database must not retain append-only witness
    # rows after the generated receipt history is removed. Production has no
    # history-destruction API; this fixture-only cleanup must deliberately
    # advance the independent anchor to the empty test state.
    await store._write_history_anchor()
    await store.ensure_schema()
