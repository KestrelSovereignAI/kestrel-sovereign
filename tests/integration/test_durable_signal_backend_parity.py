"""SQLite/PostgreSQL parity for the durable signal delivery ledger."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from kestrel_sdk.signals import (
    RedactionPolicy,
    Signal,
    SignalMode,
    SourceRegistration,
    Status,
    Trust,
)
from kestrel_sovereign.signals import (
    DurableConsumerRegistration,
    DurableSignalStore,
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.features.channels.route_ownership import ChannelRouteOwnershipStore
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError


def _signal(agent_id: str) -> Signal:
    return Signal(
        source="provider.message",
        kind="inbound",
        mode=SignalMode.ACTION,
        payload={"workflow": "wf-1", "message": "normalized"},
        target_agent=agent_id,
    )


def _dispatcher_owner_id() -> str:
    """Return the same managed-owner shape a production dispatcher emits."""

    return f"dispatcher:{uuid4().hex}"


class _DispatcherAgent:
    """Minimal live-dispatch agent for the PostgreSQL commit-boundary race."""

    def __init__(self, did: str, privacy_config) -> None:
        self.did = did
        self.privacy_config = privacy_config
        self._privacy_transition_lock = asyncio.Lock()
        self.tasks: list[asyncio.Task] = []

    def _get_privacy_transition_lock(self):
        return self._privacy_transition_lock

    def _track_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        return task

    async def process_input(self, prompt: str):  # pragma: no cover - ACTION only
        return prompt


async def _ephemeral_dispatcher(db_backend, *, agent_id: str):
    """Build the actual dispatcher path with a payload-eliding projection."""
    from kestrel_sovereign.privacy import get_privacy_preset

    agent = _DispatcherAgent(agent_id, get_privacy_preset("ephemeral"))
    registry = SourceRegistry()

    async def handler(payload):
        return {"handled": payload}

    registry.register(
        SourceRegistration(
            name="provider.message",
            schema=lambda payload: payload,
            default_mode=SignalMode.ACTION,
            allowed_modes=frozenset({SignalMode.ACTION}),
            handler=handler,
            trust=Trust.TRUSTED,
            log_redaction=RedactionPolicy(summarize=lambda payload: "<redacted>"),
            retention_days=7,
        )
    )
    log_store = SignalLogStore(db_backend)
    await log_store.initialize()
    dispatcher = SignalDispatcher(
        agent=agent,
        registry=registry,
        lock_manager=OrderedLockManager(),
        store=log_store,
    )
    await dispatcher.initialize_durable_delivery()
    return agent, dispatcher


async def _finish_dispatcher(agent: _DispatcherAgent, dispatcher: SignalDispatcher):
    """Drain audit tasks before the shared backend fixture is torn down."""
    await dispatcher.shutdown_durable_delivery()
    if agent.tasks:
        await asyncio.gather(*agent.tasks, return_exceptions=True)


async def _cancel_and_drain(*tasks: asyncio.Task | None) -> None:
    """Cancel test choreography tasks before releasing their shared backend."""
    live_tasks = [task for task in tasks if task is not None]
    for task in live_tasks:
        if not task.done():
            task.cancel()
    if live_tasks:
        await asyncio.wait_for(
            asyncio.gather(*live_tasks, return_exceptions=True), timeout=5
        )


async def _independent_backend(db_backend):
    """Connect a second real backend instance to the same durable ledger."""
    if db_backend.backend_type == "sqlite":
        from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

        backend = SQLiteBackend(db_backend.db_path)
    else:
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        backend = PostgresBackend(db_backend._dsn)
    await backend.connect()
    return backend


@asynccontextmanager
async def _isolated_durable_schema(db_backend):
    """Isolate destructive durable-schema tests on the shared PostgreSQL DB."""

    if db_backend.backend_type == "sqlite":
        yield db_backend
        return

    # PostgreSQL source-sequence uniqueness is deliberately built with
    # CREATE INDEX CONCURRENTLY, which the server rejects inside even a caller-
    # owned outer transaction. Use the same schema-pinned autocommit backend as
    # production migration phases rather than hiding DDL in a test savepoint.
    async with _independent_postgres_schema_backend(db_backend) as backend:
        yield backend


@asynccontextmanager
async def _independent_postgres_schema_backend(db_backend):
    """Yield an autocommit-capable backend pinned to one disposable schema."""

    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    schema = f"durable_signal_independent_{uuid4().hex}"
    await db_backend.execute(f'CREATE SCHEMA "{schema}"')
    separator = "&" if "?" in db_backend._dsn else "?"
    dsn = db_backend._dsn + separator + f"options=-csearch_path%3D{schema}"
    backend = PostgresBackend(dsn)
    await backend.connect()
    try:
        yield backend
    finally:
        await backend.close()
        await db_backend.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def _is_scope_handoff_lock(query: str) -> bool:
    return (
        "pg_advisory_xact_lock" in query
        or query.strip() == "DELETE FROM durable_signal_consumers WHERE 0"
    )


async def _assert_one_pending_delivery(
    store: DurableSignalStore, *, agent_id: str, event_id: str
) -> None:
    deliveries = await store.list_deliveries(agent_id=agent_id)
    assert [delivery.event_id for delivery in deliveries] == [event_id]


async def _legacy_primary_only_advance(backend, *, agent_id: str, source: str) -> int:
    """Model the old counter implementation without touching recovery state."""

    async with backend.transaction():
        await backend.execute(
            "INSERT OR IGNORE INTO durable_signal_source_sequences "
            "(agent_id, source, current_sequence) VALUES (?, ?, 0)",
            (agent_id, source),
        )
        lock_clause = " FOR UPDATE" if backend.backend_type == "postgres" else ""
        current = await backend.fetch_val(
            "SELECT current_sequence FROM durable_signal_source_sequences "
            f"WHERE agent_id = ? AND source = ?{lock_clause}",
            (agent_id, source),
        )
        assert current is not None
        sequence = int(current) + 1
        await backend.execute(
            "UPDATE durable_signal_source_sequences SET current_sequence = ? "
            "WHERE agent_id = ? AND source = ?",
            (sequence, agent_id, source),
        )
        return sequence


async def _wait_for_postgres_concurrent_index_build(backend) -> None:
    """Wait until the real server reports the owned concurrent index DDL."""

    for _ in range(100):
        active = await backend.fetch_val(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_stat_activity
                WHERE pid <> pg_backend_pid()
                  AND state = 'active'
                  AND query LIKE '%CREATE UNIQUE INDEX CONCURRENTLY%'
                  AND query LIKE '%idx_durable_signal_events_scope_sequence%'
            )
            """
        )
        if active:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("PostgreSQL concurrent index build did not become active")


async def _drop_postgres_source_recovery_trigger_family(backend) -> None:
    """Drop the current fingerprinted trigger pair, leaving functions intact."""

    for definition in DurableSignalStore.SOURCE_SEQUENCE_RECOVERY_DEFINITIONS:
        await backend.execute(
            f'DROP TRIGGER IF EXISTS "{definition.trigger_name}" '
            'ON durable_signal_events'
        )


async def _drop_source_counter_trigger_family(backend) -> None:
    """Remove the active legacy-counter fence so tests can model corruption."""

    if backend.backend_type == "postgres":
        for definition in DurableSignalStore.SOURCE_SEQUENCE_COUNTER_FENCE_DEFINITIONS:
            await backend.execute(
                f'DROP TRIGGER IF EXISTS "{definition.trigger_name}" '
                'ON durable_signal_source_sequences'
            )
        return

    rows = await backend.fetch_all(
        "SELECT name FROM sqlite_master WHERE type = 'trigger'"
    )
    prefix = DurableSignalStore.SOURCE_SEQUENCE_COUNTER_FENCE_PREFIX.casefold()
    for (name,) in rows:
        actual_name = str(name)
        if actual_name.casefold().startswith(prefix):
            quoted = '"' + actual_name.replace('"', '""') + '"'
            await backend.execute(f"DROP TRIGGER {quoted}")


async def _create_pre_source_sequence_postgres_events(backend) -> None:
    """Create the exact pre-#3006 event relation, without migration artifacts."""

    await backend.execute(
        """
        CREATE TABLE durable_signal_events (
            event_id TEXT PRIMARY KEY,
            source_event_id TEXT,
            agent_id TEXT NOT NULL,
            target_agent TEXT NOT NULL,
            source TEXT NOT NULL,
            kind TEXT NOT NULL,
            mode TEXT NOT NULL,
            payload JSONB NOT NULL,
            session_id TEXT,
            caller_identity TEXT,
            visibility TEXT NOT NULL,
            urgency TEXT NOT NULL,
            dedupe_key TEXT,
            causation_chain JSONB NOT NULL,
            arrived_at TIMESTAMPTZ NOT NULL,
            committed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            retention_until TIMESTAMPTZ NOT NULL,
            UNIQUE (agent_id, source, source_event_id)
        )
        """
    )
    await backend.execute(
        "CREATE INDEX idx_durable_signal_events_scope_retention "
        "ON durable_signal_events(agent_id, source, retention_until)"
    )


async def _prepare_interrupted_postgres_source_backfill(
    backend, *, legacy_rows: int
) -> tuple[DurableSignalStore, str, str, list[str]]:
    """Leave PostgreSQL after durable work seeding but before its first batch."""

    await _create_pre_source_sequence_postgres_events(backend)
    agent_id = f"did:test:interrupted-source-backfill:{uuid4()}"
    source = "provider.message"
    token = uuid4().hex
    await backend.execute(
        """
        INSERT INTO durable_signal_events (
            event_id, source_event_id, agent_id, target_agent, source,
            kind, mode, payload, visibility, urgency, causation_chain,
            arrived_at, committed_at, retention_until
        )
        SELECT md5(? || series::text), ? || series::text,
               ?, ?, ?, 'inbound', 'action', '{}'::jsonb,
               'internal', 'normal', '[]'::jsonb,
               NOW(), NOW(), NOW() + INTERVAL '7 days'
        FROM generate_series(1, ?) AS series
        """,
        (
            token,
            f"interrupted-source-backfill:{token}:",
            agent_id,
            agent_id,
            source,
            legacy_rows,
        ),
    )
    event_ids = [
        str(row[0])
        for row in await backend.fetch_all(
            "SELECT event_id FROM durable_signal_events "
            "WHERE agent_id = ? AND source = ? ORDER BY event_id",
            (agent_id, source),
        )
    ]
    interrupted = DurableSignalStore(backend)
    original_batch = interrupted._backfill_postgres_source_sequence_batch

    async def stop_before_first_batch():
        raise RuntimeError("simulated interruption before first backfill batch")

    interrupted._backfill_postgres_source_sequence_batch = (  # type: ignore[method-assign]
        stop_before_first_batch
    )
    with pytest.raises(TransactionError, match="before first backfill batch"):
        await interrupted.initialize()
    interrupted._backfill_postgres_source_sequence_batch = (  # type: ignore[method-assign]
        original_batch
    )
    assert await backend.fetch_val(
        "SELECT event_work_seeded "
        "FROM durable_signal_source_sequence_state WHERE singleton = 1"
    ) is True
    assert await backend.fetch_val(
        "SELECT COUNT(*) FROM durable_signal_source_sequence_event_work"
    ) == legacy_rows
    return interrupted, agent_id, source, event_ids


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_schema_bootstrap_is_safe_under_independent_backend_contention(db_backend):
    """Fresh/additive durable and route bootstrap serializes across processes."""

    peer_backend = await _independent_backend(db_backend)
    try:
        first = DurableSignalStore(db_backend)
        second = DurableSignalStore(peer_backend)
        first_routes = ChannelRouteOwnershipStore(db_backend)
        second_routes = ChannelRouteOwnershipStore(peer_backend)
        await asyncio.wait_for(
            asyncio.gather(
                first.initialize(), second.initialize(),
                first_routes.initialize(), second_routes.initialize(),
            ),
            timeout=10,
        )
        assert await db_backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_event_integrity"
        ) == 0
        claim = await first_routes.claim(
            channel_type="telegram",
            canonical_route_identity=f"telegram-bot:bootstrap-{uuid4().hex}",
            agent_id="did:test:bootstrap",
        )
        assert claim is not None
    finally:
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_source_sequence_index_bootstrap_repairs_owned_wrong_shape(db_backend):
    """Both engines replace an owned-name index that is not the exact proof."""

    async with _isolated_durable_schema(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        if backend.backend_type == "postgres":
            await backend.execute(
                "DROP INDEX CONCURRENTLY "
                "idx_durable_signal_events_scope_sequence"
            )
        else:
            await backend.execute(
                "DROP INDEX idx_durable_signal_events_scope_sequence"
            )
        await backend.execute(
            "CREATE INDEX idx_durable_signal_events_scope_sequence "
            "ON durable_signal_events(target_agent)"
        )

        repaired = DurableSignalStore(backend)
        await repaired.initialize()
        if backend.backend_type == "postgres":
            assert repaired._postgres_source_sequence_index_catalog_valid(
                await repaired._postgres_source_sequence_index_catalog()
            )
        else:
            assert repaired._sqlite_source_sequence_index_catalog_valid(
                await repaired._sqlite_source_sequence_index_catalog()
            )

        agent_id = f"did:test:index-repair:{uuid4()}"
        first = await repaired.persist_signal(
            _signal(agent_id),
            agent_id=agent_id,
            source_event_id=f"index-repair:first:{uuid4()}",
            retention_days=7,
        )
        second = await repaired.persist_signal(
            _signal(agent_id),
            agent_id=agent_id,
            source_event_id=f"index-repair:second:{uuid4()}",
            retention_days=7,
        )
        with pytest.raises(QueryError):
            await backend.execute(
                "UPDATE durable_signal_events SET source_sequence = ? "
                "WHERE event_id = ?",
                (first.source_sequence, second.event_id),
            )


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("mutation", ("nondeterministic-collation", "opclass"))
async def test_postgres_concurrent_index_repair_rejects_mutated_equality_semantics(
    db_backend, mutation
):
    """Same keys are malformed when collation/opclass changes identity."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL collation/operator-class catalog regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        await backend.execute(
            "DROP INDEX CONCURRENTLY idx_durable_signal_events_scope_sequence"
        )
        if mutation == "nondeterministic-collation":
            await backend.execute(
                "CREATE COLLATION kestrel_nondeterministic "
                "(provider = icu, locale = 'und-u-ks-level2', "
                "deterministic = false)"
            )
            first_key = "agent_id COLLATE kestrel_nondeterministic"
        else:
            first_key = "agent_id text_pattern_ops"
        await backend.execute(
            "CREATE UNIQUE INDEX CONCURRENTLY "
            "idx_durable_signal_events_scope_sequence "
            f"ON durable_signal_events ({first_key}, source, source_sequence)"
        )
        malformed = await store._postgres_source_sequence_index_catalog()
        assert not store._postgres_source_sequence_index_catalog_valid(malformed)

        first_backend = await _independent_backend(backend)
        second_backend = await _independent_backend(backend)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    DurableSignalStore(first_backend).initialize(),
                    DurableSignalStore(second_backend).initialize(),
                ),
                timeout=20,
            )
        finally:
            await second_backend.close()
            await first_backend.close()

        repaired = await store._postgres_source_sequence_index_catalog()
        assert store._postgres_source_sequence_index_catalog_valid(repaired)


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "mutation",
    ("nulls-not-distinct", "deferred-uniqueness"),
)
async def test_postgres_repairs_noncanonical_unique_index_timing_and_null_semantics(
    db_backend, mutation
):
    """PostgreSQL 16 catalog flags are required, not inferred from UNIQUE."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL 16 unique-index catalog regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        server_version = int(await backend.fetch_val("SHOW server_version_num"))
        if server_version < 160000:
            pytest.skip("PostgreSQL 16 catalog semantics regression")

        store = DurableSignalStore(backend)
        await store.initialize()
        await backend.execute(
            "DROP INDEX CONCURRENTLY idx_durable_signal_events_scope_sequence"
        )
        if mutation == "nulls-not-distinct":
            await backend.execute(
                "CREATE UNIQUE INDEX CONCURRENTLY "
                "idx_durable_signal_events_scope_sequence "
                "ON durable_signal_events (agent_id, source, source_sequence) "
                "NULLS NOT DISTINCT"
            )
        else:
            await backend.execute(
                "ALTER TABLE durable_signal_events ADD CONSTRAINT "
                "idx_durable_signal_events_scope_sequence "
                "UNIQUE (agent_id, source, source_sequence) "
                "DEFERRABLE INITIALLY DEFERRED"
            )

        malformed = await store._postgres_source_sequence_index_catalog()
        assert malformed is not None
        assert not store._postgres_source_sequence_index_catalog_valid(malformed)
        if mutation == "nulls-not-distinct":
            assert malformed[19] is True
            assert malformed[20] is True
        else:
            assert malformed[19] is False
            assert malformed[20] is False
            assert malformed[21] == store.SOURCE_SEQUENCE_SCOPE_INDEX

        await DurableSignalStore(backend).initialize()
        repaired = await store._postgres_source_sequence_index_catalog()
        assert store._postgres_source_sequence_index_catalog_valid(repaired)
        assert repaired[19] is True
        assert repaired[20] is False
        assert repaired[21] is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_valid_source_sequence_index_is_preserved_concurrently(
    db_backend,
):
    """A canonical index survives concurrent initializers without rewrite."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL valid-index preservation regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        await DurableSignalStore(backend).initialize()
        before = await backend.fetch_one(
            "SELECT oid, relfilenode FROM pg_class "
            "WHERE oid = to_regclass('idx_durable_signal_events_scope_sequence')"
        )
        first_backend = await _independent_backend(backend)
        second_backend = await _independent_backend(backend)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    DurableSignalStore(first_backend).initialize(),
                    DurableSignalStore(second_backend).initialize(),
                ),
                timeout=20,
            )
        finally:
            await second_backend.close()
            await first_backend.close()
        after = await backend.fetch_one(
            "SELECT oid, relfilenode FROM pg_class "
            "WHERE oid = to_regclass('idx_durable_signal_events_scope_sequence')"
        )
        assert tuple(after) == tuple(before)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_completed_source_migration_boot_uses_only_schema_fast_path(
    db_backend, monkeypatch
):
    """Completed schemas never re-enter history scans or enforcement DDL."""

    await DurableSignalStore(db_backend).initialize()
    observed_sql: list[str] = []
    originals = {
        name: getattr(db_backend, name)
        for name in ("execute", "fetch_one", "fetch_all", "fetch_val")
    }

    def recorder(name):
        async def record(query, params=()):
            observed_sql.append(query)
            return await originals[name](query, params)

        return record

    for name in originals:
        monkeypatch.setattr(db_backend, name, recorder(name))

    restarted = DurableSignalStore(db_backend)
    forbidden_scope_validation = AsyncMock(
        side_effect=AssertionError(
            "completed schema repeated exact-scope validation"
        )
    )
    if db_backend.backend_type == "postgres":
        monkeypatch.setattr(
            restarted,
            "_validate_postgres_source_sequence_scope_batch",
            forbidden_scope_validation,
        )
    await restarted.initialize()
    forbidden_scope_validation.assert_not_awaited()
    normalized = "\n".join(" ".join(sql.split()) for sql in observed_sql)
    assert "SELECT DISTINCT agent_id, source" not in normalized
    assert "ROW_NUMBER() OVER" not in normalized
    assert "WHERE source_sequence IS NULL" not in normalized
    assert "WHERE (agent_id, source) >" not in normalized
    assert "ALTER TABLE durable_signal_events ADD COLUMN" not in normalized
    assert "ALTER COLUMN source_sequence SET NOT NULL" not in normalized
    assert "VALIDATE CONSTRAINT durable_signal_events_source_sequence" not in normalized


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("exact_event_claim", (False, True), ids=("ordinary", "exact"))
async def test_implicit_claim_clock_follows_contended_scope_handoff_on_both_backends(
    db_backend, monkeypatch, exact_event_claim
):
    """A delayed claim takes its implicit lease clock after the real DB lock."""

    peer_backend = await _independent_backend(db_backend)
    store = DurableSignalStore(db_backend)
    peer_store = DurableSignalStore(peer_backend)
    await store.initialize()
    await peer_store.initialize()
    agent_id = f"did:test:durable-claim-clock:{uuid4()}"
    consumer = DurableConsumerRegistration(
        consumer_id="clock-worker",
        source="provider.message",
        agent_id=agent_id,
        lease_seconds=1,
    )
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = {"now": base}
    monkeypatch.setattr(store, "now_utc", lambda: clock["now"])
    await store.register_consumer(consumer)
    persisted = await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"claim-clock:{uuid4()}",
        retention_days=7,
    )
    lock_acquired = asyncio.Event()
    entered_handoff = asyncio.Event()
    release_lock = asyncio.Event()
    original_handoff = store._lock_scope_handoff

    async def observe_handoff(**kwargs):
        entered_handoff.set()
        await original_handoff(**kwargs)

    monkeypatch.setattr(store, "_lock_scope_handoff", observe_handoff)

    async def hold_peer_scope() -> None:
        async with peer_backend.transaction():
            await peer_store._lock_scope_handoff(
                agent_id=agent_id, source=consumer.source
            )
            lock_acquired.set()
            await release_lock.wait()

    blocker = asyncio.create_task(hold_peer_scope())
    try:
        await asyncio.wait_for(lock_acquired.wait(), timeout=5)
        if exact_event_claim:
            claim = asyncio.create_task(
                store.claim_delivery_for_event(
                    agent_id=agent_id,
                    consumer_id=consumer.consumer_id,
                    event_id=persisted.event_id,
                    executor_id="worker",
                )
            )
        else:
            claim = asyncio.create_task(
                store.claim_delivery(
                    agent_id=agent_id,
                    consumer_id=consumer.consumer_id,
                    executor_id="worker",
                )
            )
        await asyncio.wait_for(entered_handoff.wait(), timeout=5)
        clock["now"] = base + timedelta(seconds=2)
        release_lock.set()
        claimed = await asyncio.wait_for(claim, timeout=5)
        assert claimed is not None
        assert claimed.lease_expires_at == base + timedelta(seconds=3)
    finally:
        release_lock.set()
        await _cancel_and_drain(blocker)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_registration_backfill_clock_follows_contended_scope_handoff_on_both_backends(
    db_backend, monkeypatch
):
    """Backfill evaluates retention after the real database serialization point."""

    peer_backend = await _independent_backend(db_backend)
    store = DurableSignalStore(db_backend)
    peer_store = DurableSignalStore(peer_backend)
    await store.initialize()
    await peer_store.initialize()
    agent_id = f"did:test:durable-registration-clock:{uuid4()}"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = {"now": base}
    monkeypatch.setattr(store, "now_utc", lambda: clock["now"])
    persisted = await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"registration-clock:{uuid4()}",
        retention_days=7,
    )
    await db_backend.execute(
        "UPDATE durable_signal_events SET retention_until = ? WHERE event_id = ?",
        (store.to_timestamp_param(base + timedelta(seconds=1)), persisted.event_id),
    )
    consumer = DurableConsumerRegistration(
        consumer_id="late-clock-worker",
        source="provider.message",
        agent_id=agent_id,
    )
    lock_acquired = asyncio.Event()
    entered_handoff = asyncio.Event()
    release_lock = asyncio.Event()
    original_handoff = store._lock_scope_handoff

    async def observe_handoff(**kwargs):
        entered_handoff.set()
        await original_handoff(**kwargs)

    monkeypatch.setattr(store, "_lock_scope_handoff", observe_handoff)

    async def hold_peer_scope() -> None:
        async with peer_backend.transaction():
            await peer_store._lock_scope_handoff(
                agent_id=agent_id, source=consumer.source
            )
            lock_acquired.set()
            await release_lock.wait()

    blocker = asyncio.create_task(hold_peer_scope())
    try:
        await asyncio.wait_for(lock_acquired.wait(), timeout=5)
        registration = asyncio.create_task(store.register_consumer(consumer))
        await asyncio.wait_for(entered_handoff.wait(), timeout=5)
        clock["now"] = base + timedelta(seconds=2)
        release_lock.set()
        await asyncio.wait_for(registration, timeout=5)
        assert await store.list_deliveries(agent_id=agent_id) == []
    finally:
        release_lock.set()
        await _cancel_and_drain(blocker)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "first",
    ("capture", "persistence"),
    ids=("capture-first", "persistence-first"),
)
async def test_source_boundary_and_ingress_share_one_cross_connection_order(
    db_backend, monkeypatch, first
):
    """The source handoff winner alone decides strict boundary eligibility."""

    peer_backend = await _independent_backend(db_backend)
    store = DurableSignalStore(db_backend)
    peer_store = DurableSignalStore(peer_backend)
    await store.initialize()
    await peer_store.initialize()
    agent_id = f"did:test:source-boundary-race:{uuid4()}"
    source = "provider.message"
    first_owned = asyncio.Event()
    release_first = asyncio.Event()
    contender_entered = asyncio.Event()
    capture_task = None
    persistence_task = None
    try:
        if first == "capture":
            original_sample = store._source_sequence_locked

            async def pause_captured_sample(**kwargs):
                sequence = await original_sample(**kwargs)
                first_owned.set()
                await release_first.wait()
                return sequence

            monkeypatch.setattr(store, "_source_sequence_locked", pause_captured_sample)
            capture_task = asyncio.create_task(
                store.capture_source_boundary(agent_id=agent_id, source=source)
            )
            await asyncio.wait_for(first_owned.wait(), timeout=5)

            original_handoff = peer_store._lock_scope_handoff

            async def observe_contending_persistence(**kwargs):
                contender_entered.set()
                await original_handoff(**kwargs)

            monkeypatch.setattr(
                peer_store, "_lock_scope_handoff", observe_contending_persistence
            )
            persistence_task = asyncio.create_task(
                peer_store.persist_signal(
                    _signal(agent_id),
                    agent_id=agent_id,
                    source_event_id=f"boundary-race:{uuid4()}",
                    retention_days=7,
                )
            )
        else:
            original_advance = store._advance_source_sequence_locked

            async def pause_advanced_sequence(**kwargs):
                sequence = await original_advance(**kwargs)
                first_owned.set()
                await release_first.wait()
                return sequence

            monkeypatch.setattr(
                store, "_advance_source_sequence_locked", pause_advanced_sequence
            )
            persistence_task = asyncio.create_task(
                store.persist_signal(
                    _signal(agent_id),
                    agent_id=agent_id,
                    source_event_id=f"boundary-race:{uuid4()}",
                    retention_days=7,
                )
            )
            await asyncio.wait_for(first_owned.wait(), timeout=5)

            original_handoff = peer_store._lock_scope_handoff

            async def observe_contending_capture(**kwargs):
                contender_entered.set()
                await original_handoff(**kwargs)

            monkeypatch.setattr(
                peer_store, "_lock_scope_handoff", observe_contending_capture
            )
            capture_task = asyncio.create_task(
                peer_store.capture_source_boundary(agent_id=agent_id, source=source)
            )

        await asyncio.wait_for(contender_entered.wait(), timeout=5)
        assert capture_task is not None and persistence_task is not None
        contender = persistence_task if first == "capture" else capture_task
        assert not contender.done()
        release_first.set()
        boundary, persisted = await asyncio.wait_for(
            asyncio.gather(capture_task, persistence_task), timeout=5
        )
        assert persisted.source_sequence == 1
        assert boundary.sequence == (0 if first == "capture" else 1)

        registration = DurableConsumerRegistration(
            consumer_id="late-workflow",
            source=source,
            agent_id=agent_id,
        )
        await store.register_consumer(registration)
        delivery = (
            await store.list_deliveries(
                agent_id=agent_id, consumer_id=registration.consumer_id
            )
        )[0]
        assert boundary.is_event_eligible(delivery.event) is (first == "capture")
    finally:
        release_first.set()
        await _cancel_and_drain(capture_task, persistence_task)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_boundary_samples_sequence_after_waiting_for_source_writer(
    db_backend, monkeypatch
):
    """A blocked capture observes the commit that held its handoff lock."""

    peer_backend = await _independent_backend(db_backend)
    store = DurableSignalStore(db_backend)
    peer_store = DurableSignalStore(peer_backend)
    await store.initialize()
    await peer_store.initialize()
    agent_id = f"did:test:source-boundary-post-lock:{uuid4()}"
    source = "provider.message"
    writer_sequenced = asyncio.Event()
    release_writer = asyncio.Event()
    capture_entered = asyncio.Event()
    holder = None
    capture_task = None
    try:
        async def hold_uncommitted_event() -> None:
            async with peer_backend.transaction():
                persisted = await peer_store.persist_signal(
                    _signal(agent_id),
                    agent_id=agent_id,
                    source_event_id=f"post-lock:{uuid4()}",
                    retention_days=7,
                )
                assert persisted.source_sequence == 1
                writer_sequenced.set()
                await release_writer.wait()

        holder = asyncio.create_task(hold_uncommitted_event())
        await asyncio.wait_for(writer_sequenced.wait(), timeout=5)

        original_handoff = store._lock_scope_handoff

        async def observe_capture_handoff(**kwargs):
            capture_entered.set()
            await original_handoff(**kwargs)

        monkeypatch.setattr(store, "_lock_scope_handoff", observe_capture_handoff)
        capture_task = asyncio.create_task(
            store.capture_source_boundary(agent_id=agent_id, source=source)
        )
        await asyncio.wait_for(capture_entered.wait(), timeout=5)
        assert not capture_task.done()

        release_writer.set()
        boundary = await asyncio.wait_for(capture_task, timeout=5)
        await asyncio.wait_for(holder, timeout=5)
        assert boundary.sequence == 1
    finally:
        release_writer.set()
        await _cancel_and_drain(holder, capture_task)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_missing_and_stale_source_counters_repair_per_scope_or_fail_closed(
    db_backend,
):
    """Scope evidence repairs either copy; losing both remains fail-closed."""

    async with _isolated_durable_schema(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        agent_a = f"did:test:counter-repair-a:{uuid4()}"
        agent_b = f"did:test:counter-repair-b:{uuid4()}"
        source = "provider.message"
        other_source = "provider.other"
        purged_source = "provider.purged"

        for index in range(2):
            assert (
                await store.persist_signal(
                    _signal(agent_a),
                    agent_id=agent_a,
                    source_event_id=f"repair-a:{index}:{uuid4()}",
                    retention_days=7,
                )
            ).source_sequence == index + 1
        assert (
            await store.persist_signal(
                Signal(
                    source=other_source,
                    kind="inbound",
                    mode=SignalMode.ACTION,
                    payload={"workflow": "wf-1"},
                    target_agent=agent_a,
                ),
                agent_id=agent_a,
                source_event_id=f"repair-other:{uuid4()}",
                retention_days=7,
            )
        ).source_sequence == 1
        assert (
            await store.persist_signal(
                _signal(agent_b),
                agent_id=agent_b,
                source_event_id=f"repair-b:{uuid4()}",
                retention_days=7,
            )
        ).source_sequence == 1
        purged = await store.persist_signal(
            Signal(
                source=purged_source,
                kind="inbound",
                mode=SignalMode.ACTION,
                payload={"workflow": "wf-1"},
                target_agent=agent_a,
            ),
            agent_id=agent_a,
            source_event_id=f"repair-purged:{uuid4()}",
            retention_days=7,
        )
        await backend.execute(
            "UPDATE durable_signal_events SET retention_until = ? "
            "WHERE event_id = ?",
            (
                store.to_timestamp_param(
                    datetime.now(timezone.utc) - timedelta(seconds=1)
                ),
                purged.event_id,
            ),
        )
        assert await store.purge_expired(agent_id=agent_a) == 1

        # A normal initialization does not scan history. The first real scope
        # operation notices and repairs this stale row through the scope index.
        await backend.execute(
            "UPDATE durable_signal_source_sequences SET current_sequence = 0 "
            "WHERE agent_id = ? AND source = ?",
            (agent_a, source),
        )
        await backend.execute(
            "UPDATE durable_signal_source_sequence_recovery "
            "SET recovery_sequence = 0 WHERE agent_id = ? AND source = ?",
            (agent_a, source),
        )
        await DurableSignalStore(backend).initialize()
        assert await backend.fetch_val(
            "SELECT current_sequence FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            (agent_a, source),
        ) == 2
        assert (
            await store.capture_source_boundary(agent_id=agent_a, source=source)
        ).sequence == 2
        assert (
            await store.persist_signal(
                _signal(agent_a),
                agent_id=agent_a,
                source_event_id=f"repair-a:next:{uuid4()}",
                retention_days=7,
            )
        ).source_sequence == 3

        # Recreate the whole primary table as in the review reproduction. The
        # independent per-scope watermark includes purged history, while the
        # indexed event maximum catches either copy if it is merely stale.
        await backend.execute("DROP TABLE durable_signal_source_sequences")
        recovered = DurableSignalStore(backend)
        await recovered.initialize()
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_a, source),
        ) == 1
        assert (
            await recovered.capture_source_boundary(
                agent_id=agent_a, source=source
            )
        ).sequence == 3
        assert (
            await recovered.capture_source_boundary(
                agent_id=agent_a, source=other_source
            )
        ).sequence == 1
        assert (
            await recovered.capture_source_boundary(
                agent_id=agent_b, source=source
            )
        ).sequence == 1
        assert (
            await recovered.persist_signal(
                Signal(
                    source=other_source,
                    kind="inbound",
                    mode=SignalMode.ACTION,
                    payload={"workflow": "wf-1"},
                    target_agent=agent_a,
                ),
                agent_id=agent_a,
                source_event_id=f"repair-other:next:{uuid4()}",
                retention_days=7,
            )
        ).source_sequence == 2

        # No event survives in this scope, but its independent watermark does.
        # The next value is 2 rather than a reused 1.
        assert (
            await recovered.capture_source_boundary(
                agent_id=agent_a, source=purged_source
            )
        ).sequence == 1
        assert (
            await recovered.persist_signal(
                Signal(
                    source=purged_source,
                    kind="inbound",
                    mode=SignalMode.ACTION,
                    payload={"workflow": "wf-1"},
                    target_agent=agent_a,
                ),
                agent_id=agent_a,
                source_event_id=f"repair-purged:next:{uuid4()}",
                retention_days=7,
            )
        ).source_sequence == 2

        # Recreating both older exact tables still recovers from the separate,
        # retention-independent high-water ledger.
        await backend.execute("DROP TABLE durable_signal_source_sequences")
        await backend.execute("DROP TABLE durable_signal_source_sequence_recovery")
        unrecoverable = DurableSignalStore(backend)
        await unrecoverable.initialize()
        assert (
            await unrecoverable.capture_source_boundary(
                agent_id=agent_a, source=source
            )
        ).sequence == 3

        # Retained maxima alone cannot prove that no higher, shorter-lived
        # event was purged. Losing every exact row leaves the immutable scope
        # marker behind and refuses even a scope that still has retained rows.
        for relation in (
            unrecoverable.SOURCE_SEQUENCES,
            unrecoverable.SOURCE_SEQUENCE_RECOVERY,
            unrecoverable.SOURCE_SEQUENCE_HIGH_WATER,
        ):
            await backend.execute(
                f"DELETE FROM {relation} WHERE agent_id = ? AND source = ?",
                (agent_a, source),
            )
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_a, source),
        ) == 1
        with pytest.raises(
            (RuntimeError, TransactionError), match="both exact counter copies"
        ):
            await unrecoverable.capture_source_boundary(
                agent_id=agent_a, source=source
            )
        assert (
            await unrecoverable.capture_source_boundary(
                agent_id=agent_a, source=f"fresh-after-loss:{uuid4()}"
            )
        ).sequence == 0


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_boundary_repairs_missing_seen_before_retention_and_exact_loss(
    db_backend,
):
    """Positive exact state recreates evidence on SQLite and PostgreSQL."""

    async with _isolated_durable_schema(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        agent_id = f"did:test:seen-fast-path-repair:{uuid4()}"
        source = "provider.message"
        persisted = await store.persist_signal(
            _signal(agent_id),
            agent_id=agent_id,
            source_event_id=f"seen-fast-path:{uuid4()}",
            retention_days=7,
        )
        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )

        assert (
            await store.capture_source_boundary(agent_id=agent_id, source=source)
        ).sequence == 1
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 1

        await backend.execute(
            "UPDATE durable_signal_events SET retention_until = ? "
            "WHERE event_id = ?",
            (
                store.to_timestamp_param(
                    datetime.now(timezone.utc) - timedelta(seconds=1)
                ),
                persisted.event_id,
            ),
        )
        assert await store.purge_expired(agent_id=agent_id) == 1

        # A pre-recovery replica advances only the primary API. Its database
        # fence mirrors exact state; deleting seen again reproduces the equal,
        # positive fast path without any retained event as fallback evidence.
        assert await _legacy_primary_only_advance(
            backend, agent_id=agent_id, source=source
        ) == 2
        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        assert (
            await store.capture_source_boundary(agent_id=agent_id, source=source)
        ).sequence == 2
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 1

        await backend.execute(
            "DELETE FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_recovery "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_high_water "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        with pytest.raises(
            (QueryError, TransactionError),
            match="both exact counter copies were lost",
        ):
            await _legacy_primary_only_advance(
                backend, agent_id=agent_id, source=source
            )
        with pytest.raises(
            (RuntimeError, TransactionError), match="both exact counter copies"
        ):
            await store.persist_signal(
                _signal(agent_id),
                agent_id=agent_id,
                source_event_id=f"seen-fast-path:must-not-reuse:{uuid4()}",
                retention_days=7,
            )


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "purged_indices",
    ((0, 1, 2), (1,)),
    ids=("complete-retention", "non-prefix-retention"),
)
async def test_both_exact_row_deletions_fail_closed_per_seen_scope_on_real_backends(
    db_backend, purged_indices
):
    """The immutable scope marker survives either retention deletion shape."""

    async with _isolated_durable_schema(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        agent_id = f"did:test:seen-row-loss:{uuid4()}"
        source = "provider.message"
        persisted = [
            await store.persist_signal(
                _signal(agent_id),
                agent_id=agent_id,
                source_event_id=f"seen-row-loss:{uuid4()}",
                retention_days=7,
            )
            for _ in range(3)
        ]
        expired = store.to_timestamp_param(
            datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        for index in purged_indices:
            await backend.execute(
                "UPDATE durable_signal_events SET retention_until = ? "
                "WHERE event_id = ?",
                (expired, persisted[index].event_id),
            )
        assert await store.purge_expired(agent_id=agent_id) == len(purged_indices)
        await backend.execute(
            "DELETE FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_recovery "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_high_water "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 1
        with pytest.raises(
            QueryError,
            match="both exact counter copies were lost for a previously seen scope",
        ):
            await backend.execute(
                "INSERT OR IGNORE INTO durable_signal_source_sequences "
                "(agent_id, source, current_sequence) VALUES (?, ?, 0)",
                (agent_id, source),
            )
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 0
        with pytest.raises(
            (RuntimeError, TransactionError),
            match="both exact counter copies were lost for a previously seen scope",
        ):
            await store.capture_source_boundary(agent_id=agent_id, source=source)
        assert (
            await store.capture_source_boundary(
                agent_id=agent_id, source=f"provider.fresh:{uuid4()}"
            )
        ).sequence == 0


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "purged_indices",
    ((0, 1, 2), (2,)),
    ids=("complete-retention", "newest-only-retention"),
)
@pytest.mark.parametrize(
    "corruption",
    ("all-exact-rows-zero", "primary-lost-independent-rows-zero"),
)
async def test_seen_scope_with_no_positive_exact_high_water_fails_closed_everywhere(
    db_backend, purged_indices, corruption
):
    """Zero-valued survivors cannot reset a used scope after any retention shape."""

    async with _isolated_durable_schema(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        agent_id = f"did:test:zero-high-water:{uuid4()}"
        source = "provider.message"
        template_agent = f"did:test:zero-high-water-template:{uuid4()}"
        template = await store.persist_signal(
            _signal(template_agent),
            agent_id=template_agent,
            source_event_id=f"zero-high-water-template:{uuid4()}",
            retention_days=7,
        )
        persisted = [
            await store.persist_signal(
                _signal(agent_id),
                agent_id=agent_id,
                source_event_id=f"zero-high-water:{uuid4()}",
                retention_days=7,
            )
            for _ in range(3)
        ]
        assert [item.source_sequence for item in persisted] == [1, 2, 3]

        expired = store.to_timestamp_param(
            datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        for index in purged_indices:
            await backend.execute(
                "UPDATE durable_signal_events SET retention_until = ? "
                "WHERE event_id = ?",
                (expired, persisted[index].event_id),
            )
        assert await store.purge_expired(agent_id=agent_id) == len(purged_indices)
        retained = await backend.fetch_all(
            "SELECT source_sequence FROM durable_signal_events "
            "WHERE agent_id = ? AND source = ? ORDER BY source_sequence",
            (agent_id, source),
        )
        expected_retained = [] if len(purged_indices) == 3 else [(1,), (2,)]
        assert [tuple(row) for row in retained] == expected_retained

        await _drop_source_counter_trigger_family(backend)
        await backend.execute(
            "UPDATE durable_signal_source_sequence_recovery "
            "SET recovery_sequence = 0 WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await backend.execute(
            "UPDATE durable_signal_source_sequence_high_water "
            "SET high_water_sequence = 0 WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        if corruption == "all-exact-rows-zero":
            await backend.execute(
                "UPDATE durable_signal_source_sequences "
                "SET current_sequence = 0 WHERE agent_id = ? AND source = ?",
                (agent_id, source),
            )
        else:
            await backend.execute(
                "DELETE FROM durable_signal_source_sequences "
                "WHERE agent_id = ? AND source = ?",
                (agent_id, source),
            )

        # Exercise the store-level detector itself while the compatibility
        # trigger family is absent. It must reject zero-valued survivors even
        # when one or all INSERT OR IGNORE calls find preexisting exact rows.
        unfenced = DurableSignalStore(backend)
        with pytest.raises(
            (RuntimeError, TransactionError), match="both exact counter copies"
        ):
            await unfenced.capture_source_boundary(agent_id=agent_id, source=source)

        restarted = DurableSignalStore(backend)
        await restarted.initialize()
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 1

        with pytest.raises(
            (RuntimeError, TransactionError), match="both exact counter copies"
        ):
            await restarted.capture_source_boundary(
                agent_id=agent_id, source=source
            )
        with pytest.raises(
            (RuntimeError, TransactionError), match="both exact counter copies"
        ):
            await restarted.persist_signal(
                _signal(agent_id),
                agent_id=agent_id,
                source_event_id=f"zero-high-water-next:{uuid4()}",
                retention_days=7,
            )

        if corruption == "all-exact-rows-zero":
            legacy_sql = (
                "UPDATE durable_signal_source_sequences SET current_sequence = 1 "
                "WHERE agent_id = ? AND source = ?"
            )
        else:
            legacy_sql = (
                "INSERT OR IGNORE INTO durable_signal_source_sequences "
                "(agent_id, source, current_sequence) VALUES (?, ?, 1)"
            )
        with pytest.raises(QueryError, match="both exact counter copies"):
            await backend.execute(legacy_sql, (agent_id, source))

        mixed_event_id = str(uuid4())
        with pytest.raises(QueryError, match="both exact counter copies"):
            await backend.execute(
                """
                INSERT INTO durable_signal_events (
                    event_id, source_event_id, agent_id, target_agent, source,
                    kind, mode, payload, session_id, caller_identity,
                    visibility, urgency, dedupe_key, causation_chain,
                    arrived_at, committed_at, retention_until, source_sequence
                )
                SELECT ?, ?, ?, ?, ?, kind, mode, payload, session_id,
                       caller_identity, visibility, urgency, dedupe_key,
                       causation_chain, arrived_at, committed_at,
                       retention_until, 4
                FROM durable_signal_events WHERE event_id = ?
                """,
                (
                    mixed_event_id,
                    f"zero-high-water-mixed:{uuid4()}",
                    agent_id,
                    agent_id,
                    source,
                    template.event_id,
                ),
            )
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_events WHERE event_id = ?",
            (mixed_event_id,),
        ) == 0

        fresh_source = f"provider.fresh:{uuid4()}"
        assert (
            await restarted.capture_source_boundary(
                agent_id=agent_id, source=fresh_source
            )
        ).sequence == 0
        fresh = await restarted.persist_signal(
            Signal(
                source=fresh_source,
                kind="inbound",
                mode=SignalMode.ACTION,
                payload={"workflow": "wf-1"},
                target_agent=agent_id,
            ),
            agent_id=agent_id,
            source_event_id=f"zero-high-water-fresh:{uuid4()}",
            retention_days=7,
        )
        assert fresh.source_sequence == 1


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_independent_high_water_survives_review_ordering_on_real_backends(
    db_backend,
):
    """Purge 3, lose older metadata, restart, and allocate 4 on both engines."""

    async with _isolated_durable_schema(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        agent_id = f"did:test:independent-high-water:{uuid4()}"
        source = "provider.message"
        persisted = [
            await store.persist_signal(
                _signal(agent_id),
                agent_id=agent_id,
                source_event_id=f"independent-high-water:{uuid4()}",
                retention_days=7,
            )
            for _ in range(3)
        ]
        assert [item.source_sequence for item in persisted] == [1, 2, 3]
        await backend.execute(
            "UPDATE durable_signal_events SET retention_until = ? "
            "WHERE event_id = ?",
            (
                store.to_timestamp_param(
                    datetime.now(timezone.utc) - timedelta(seconds=1)
                ),
                persisted[2].event_id,
            ),
        )
        assert await store.purge_expired(agent_id=agent_id) == 1

        # Exact review reproduction: only retained 1-2 survive, then the seen,
        # primary, and recovery rows disappear. The independent high-water is
        # deliberately outside retention and remains exact at 3.
        for relation in (
            store.SOURCE_SEQUENCE_SEEN,
            store.SOURCE_SEQUENCES,
            store.SOURCE_SEQUENCE_RECOVERY,
        ):
            await backend.execute(
                f"DELETE FROM {relation} WHERE agent_id = ? AND source = ?",
                (agent_id, source),
            )
        assert await backend.fetch_val(
            "SELECT high_water_sequence "
            "FROM durable_signal_source_sequence_high_water "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 3

        restarted = DurableSignalStore(backend)
        await restarted.initialize()
        assert (
            await restarted.capture_source_boundary(agent_id=agent_id, source=source)
        ).sequence == 3
        assert (
            await restarted.persist_signal(
                _signal(agent_id),
                agent_id=agent_id,
                source_event_id=f"independent-high-water:next:{uuid4()}",
                retention_days=7,
            )
        ).source_sequence == 4
        assert (
            await restarted.capture_source_boundary(
                agent_id=f"{agent_id}:other", source=source
            )
        ).sequence == 0


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    ("loss_kind", "purged_indices"),
    (
        ("table", (0, 1, 2)),
        ("table", (1,)),
        ("row", (0, 1, 2)),
        ("row", (1,)),
    ),
    ids=(
        "table-complete-retention",
        "table-non-prefix-retention",
        "row-complete-retention",
        "row-non-prefix-retention",
    ),
)
async def test_legacy_primary_writer_recovers_before_racing_live_ingress(
    db_backend, loss_kind, purged_indices
):
    """The DB fence seeds legacy state before either racing writer can use it."""

    async with _isolated_durable_schema(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        agent_id = f"did:test:legacy-counter-recovery:{uuid4()}"
        source = "provider.message"
        persisted = [
            await store.persist_signal(
                _signal(agent_id),
                agent_id=agent_id,
                source_event_id=f"legacy-counter-recovery:{uuid4()}",
                retention_days=7,
            )
            for _ in range(3)
        ]
        expired = store.to_timestamp_param(
            datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        for index in purged_indices:
            await backend.execute(
                "UPDATE durable_signal_events SET retention_until = ? "
                "WHERE event_id = ?",
                (expired, persisted[index].event_id),
            )
        assert await store.purge_expired(agent_id=agent_id) == len(purged_indices)

        if loss_kind == "table":
            await backend.execute("DROP TABLE durable_signal_source_sequences")
            store = DurableSignalStore(backend)
            await store.initialize()
        else:
            await backend.execute(
                "DELETE FROM durable_signal_source_sequences "
                "WHERE agent_id = ? AND source = ?",
                (agent_id, source),
            )

        peer = await _independent_backend(backend)
        try:
            legacy_sequence, live_event = await asyncio.wait_for(
                asyncio.gather(
                    _legacy_primary_only_advance(
                        peer, agent_id=agent_id, source=source
                    ),
                    store.persist_signal(
                        _signal(agent_id),
                        agent_id=agent_id,
                        source_event_id=f"legacy-counter-live:{uuid4()}",
                        retention_days=7,
                    ),
                ),
                timeout=10,
            )
        finally:
            await peer.close()

        # Both contenders advance from durable recovery=3. Their order may
        # vary, but neither can observe/recreate zero and both exact copies end
        # at the same post-race value.
        assert {legacy_sequence, live_event.source_sequence} == {4, 5}
        assert await backend.fetch_val(
            "SELECT current_sequence FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 5
        assert await backend.fetch_val(
            "SELECT recovery_sequence "
            "FROM durable_signal_source_sequence_recovery "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 5


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_partially_sequenced_rows_repair_counter_and_backfill_on_postgres(
    db_backend
):
    """Upgrade preserves legacy events and installs a stable current boundary."""

    if db_backend.backend_type != "postgres":
        pytest.skip("SQLite legacy-column migration is covered by the unit suite")

    store = DurableSignalStore(db_backend)
    await store.initialize()
    agent_id = f"did:test:source-boundary-migration:{uuid4()}"
    source = "provider.message"
    persisted = [
        await store.persist_signal(
            _signal(agent_id),
            agent_id=agent_id,
            source_event_id=f"migration:{uuid4()}",
            retention_days=7,
        )
        for _ in range(3)
    ]
    event_ids = {item.event_id for item in persisted}

    # Model a partially migrated PostgreSQL scope. The historical maximum,
    # rather than the stale counter, must be the base for remaining rows.
    seeded_event_id = min(event_ids)
    async with db_backend.transaction():
        await _drop_postgres_source_recovery_trigger_family(db_backend)
        await db_backend.execute(
            "ALTER TABLE durable_signal_events DROP CONSTRAINT IF EXISTS "
            "durable_signal_events_source_sequence_not_null"
        )
        await db_backend.execute(
            "ALTER TABLE durable_signal_events "
            "ALTER COLUMN source_sequence DROP NOT NULL"
        )
        await db_backend.execute(
            "UPDATE durable_signal_events SET source_sequence = NULL "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await db_backend.execute(
            "UPDATE durable_signal_events SET source_sequence = 7 "
            "WHERE event_id = ?",
            (seeded_event_id,),
        )
        await db_backend.execute(
            "UPDATE durable_signal_source_sequences SET current_sequence = 3 "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )

    migrated = DurableSignalStore(db_backend)
    await migrated.initialize()
    rows = await db_backend.fetch_all(
        "SELECT event_id, source_sequence FROM durable_signal_events "
        "WHERE agent_id = ? AND source = ? ORDER BY source_sequence",
        (agent_id, source),
    )
    assert {row[0] for row in rows} == event_ids
    assert [int(row[1]) for row in rows] == [7, 8, 9]
    assert (
        await migrated.capture_source_boundary(agent_id=agent_id, source=source)
    ).sequence == 9
    assert (
        await migrated.persist_signal(
            _signal(agent_id),
            agent_id=agent_id,
            source_event_id=f"migration-post:{uuid4()}",
            retention_days=7,
        )
    ).source_sequence == 10

    nullable = await db_backend.fetch_val(
        """
        SELECT is_nullable = 'YES'
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'durable_signal_events'
          AND column_name = 'source_sequence'
        """
    )
    assert nullable is False
    validated_fence = await db_backend.fetch_val(
        """
        SELECT convalidated
        FROM pg_constraint
        WHERE conrelid = to_regclass('durable_signal_events')
          AND conname = 'durable_signal_events_source_sequence_not_null'
          AND contype = 'c'
        """
    )
    assert validated_fence is True


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_large_scope_backfill_plan_and_work_scale_linearly(
    db_backend, monkeypatch
):
    """True legacy history is scanned once and drained by indexed batches."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL bounded-work migration regression")

    async with _isolated_durable_schema(db_backend) as backend:
        await _create_pre_source_sequence_postgres_events(backend)
        agent_id = f"did:test:large-backfill:{uuid4()}"
        source = "provider.message"
        token = uuid4().hex
        legacy_rows = 10_000
        await backend.execute(
            """
            INSERT INTO durable_signal_events (
                event_id, source_event_id, agent_id, target_agent, source,
                kind, mode, payload, visibility, urgency, causation_chain,
                arrived_at, committed_at, retention_until
            )
            SELECT md5(? || series::text), ? || series::text,
                   ?, ?, ?, 'inbound', 'action', '{}'::jsonb,
                   'internal', 'normal', '[]'::jsonb,
                   NOW(), NOW(), NOW() + INTERVAL '7 days'
            FROM generate_series(1, ?) AS series
            """,
            (
                token,
                f"large-backfill:{token}:",
                agent_id,
                agent_id,
                source,
                legacy_rows,
            ),
        )
        store = DurableSignalStore(backend)
        original_batch = store._backfill_postgres_source_sequence_batch
        first_batch_ready = asyncio.Event()
        release_first_batch = asyncio.Event()
        batch_calls = 0

        async def observe_batches():
            nonlocal batch_calls
            batch_calls += 1
            if batch_calls == 1:
                first_batch_ready.set()
                await release_first_batch.wait()
            return await original_batch()

        seed_statements = 0
        original_execute = backend.execute

        async def observe_seed(query, params=()):
            nonlocal seed_statements
            normalized = " ".join(query.split())
            if (
                "INSERT INTO durable_signal_source_sequence_event_work" in normalized
                and "FROM durable_signal_events" in normalized
            ):
                seed_statements += 1
            return await original_execute(query, params)

        monkeypatch.setattr(
            store, "_backfill_postgres_source_sequence_batch", observe_batches
        )
        monkeypatch.setattr(backend, "execute", observe_seed)
        probe = await _independent_backend(backend)
        migration_task = asyncio.create_task(store.initialize())
        try:
            await asyncio.wait_for(first_batch_ready.wait(), timeout=15)
            assert await probe.fetch_val(
                "SELECT COUNT(*) FROM durable_signal_source_sequence_event_work"
            ) == legacy_rows
            sql = store._postgres_source_sequence_backfill_update_sql()
            plan = await probe.fetch_val(
                "EXPLAIN (FORMAT JSON) " + sql,
                (
                    agent_id,
                    source,
                    store.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE,
                    agent_id,
                    source,
                    0,
                    agent_id,
                    source,
                    agent_id,
                    source,
                ),
            )
            rendered_plan = json.dumps(plan, default=str)
            assert "SubPlan" not in rendered_plan
            assert "WindowAgg" in rendered_plan
            assert "durable_signal_source_sequence_event_work_pkey" in rendered_plan
            assert "Index Scan" in rendered_plan or "Index Only Scan" in rendered_plan
            release_first_batch.set()
            await asyncio.wait_for(migration_task, timeout=30)
        finally:
            release_first_batch.set()
            await _cancel_and_drain(migration_task)
            await probe.close()

        assert seed_statements == 1
        assert batch_calls == (
            (legacy_rows + store.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE - 1)
            // store.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE
        ) + 1
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_event_work"
        ) == 0
        aggregate = await backend.fetch_one(
            "SELECT COUNT(*), MIN(source_sequence), MAX(source_sequence) "
            "FROM durable_signal_events WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        assert tuple(int(value) for value in aggregate) == (
            legacy_rows,
            1,
            legacy_rows,
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_wide_payload_backfill_bounds_transition_batches_and_temp_growth(
    db_backend,
):
    """Backfill never places a whole wide legacy scope in one NEW TABLE."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL transition-tuplestore batch regression")

    async with _isolated_durable_schema(db_backend) as backend:
        await _create_pre_source_sequence_postgres_events(backend)
        store = DurableSignalStore(backend)
        agent_id = f"did:test:wide-backfill:{uuid4()}"
        source = "provider.message"
        legacy_rows = store.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE * 4 + 7
        token = uuid4().hex
        await backend.execute(
            """
            INSERT INTO durable_signal_events (
                event_id, source_event_id, agent_id, target_agent, source,
                kind, mode, payload, visibility, urgency, causation_chain,
                arrived_at, committed_at, retention_until
            )
            SELECT md5(? || series::text), ? || series::text,
                   ?, ?, ?, 'inbound', 'action',
                   jsonb_build_object(
                       'message', (
                           SELECT string_agg(
                               md5((series * 1000 + chunk)::text), ''
                           )
                           FROM generate_series(1, 256) AS chunk
                       )
                   ),
                   'internal', 'normal', '[]'::jsonb,
                   NOW(), NOW(), NOW() + INTERVAL '7 days'
            FROM generate_series(1, ?) AS series
            """,
            (
                token,
                f"wide-backfill:{token}:",
                agent_id,
                agent_id,
                source,
                legacy_rows,
            ),
        )
        total_payload_bytes = int(
            await backend.fetch_val(
                "SELECT SUM(pg_column_size(payload)) "
                "FROM durable_signal_events WHERE agent_id = ? AND source = ?",
                (agent_id, source),
            )
        )

        await backend.execute(
            "CREATE TABLE kestrel_backfill_batch_probe "
            "(batch_rows INTEGER NOT NULL, payload_bytes BIGINT NOT NULL)"
        )
        await backend.execute(
            """
            CREATE FUNCTION kestrel_probe_backfill_batch()
            RETURNS trigger AS $probe$
            BEGIN
                INSERT INTO kestrel_backfill_batch_probe (
                    batch_rows, payload_bytes
                )
                SELECT COUNT(*), COALESCE(SUM(pg_column_size(payload)), 0)
                FROM kestrel_probed_rows;
                RETURN NULL;
            END
            $probe$ LANGUAGE plpgsql
            """
        )
        await backend.execute(
            """
            CREATE TRIGGER kestrel_probe_backfill_batch
            AFTER UPDATE ON durable_signal_events
            REFERENCING NEW TABLE AS kestrel_probed_rows
            FOR EACH STATEMENT EXECUTE FUNCTION kestrel_probe_backfill_batch()
            """
        )
        await backend.fetch_val("SELECT pg_stat_clear_snapshot()")
        temp_before = int(
            await backend.fetch_val(
                "SELECT temp_bytes FROM pg_stat_database "
                "WHERE datname = current_database()"
            )
            or 0
        )

        await store.initialize()

        await backend.fetch_val("SELECT pg_stat_clear_snapshot()")
        temp_after = int(
            await backend.fetch_val(
                "SELECT temp_bytes FROM pg_stat_database "
                "WHERE datname = current_database()"
            )
            or 0
        )
        batches = await backend.fetch_all(
            "SELECT batch_rows, payload_bytes "
            "FROM kestrel_backfill_batch_probe ORDER BY ctid"
        )
        assert sum(int(row[0]) for row in batches) == legacy_rows
        assert len(batches) >= 5
        assert max(int(row[0]) for row in batches) <= (
            store.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE
        )
        assert max(int(row[1]) for row in batches) < total_payload_bytes
        # The database statistic is cumulative and shared, so allow unrelated
        # noise while still rejecting scope-sized transition spill growth.
        assert temp_after - temp_before < total_payload_bytes


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_backfill_restarts_after_a_committed_batch(
    db_backend, monkeypatch
):
    """A crash between batches preserves offsets and resumes from the marker."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL restart-safe batch regression")

    async with _isolated_durable_schema(db_backend) as backend:
        await _create_pre_source_sequence_postgres_events(backend)
        store = DurableSignalStore(backend)
        agent_id = f"did:test:batch-restart:{uuid4()}"
        source = "provider.message"
        legacy_rows = store.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE + 9
        token = uuid4().hex
        await backend.execute(
            """
            INSERT INTO durable_signal_events (
                event_id, source_event_id, agent_id, target_agent, source,
                kind, mode, payload, visibility, urgency, causation_chain,
                arrived_at, committed_at, retention_until
            )
            SELECT md5(? || series::text), ? || series::text,
                   ?, ?, ?, 'inbound', 'action', '{}'::jsonb,
                   'internal', 'normal', '[]'::jsonb,
                   NOW(), NOW(), NOW() + INTERVAL '7 days'
            FROM generate_series(1, ?) AS series
            """,
            (
                token,
                f"batch-restart:{token}:",
                agent_id,
                agent_id,
                source,
                legacy_rows,
            ),
        )

        interrupted = DurableSignalStore(backend)
        original_batch = interrupted._backfill_postgres_source_sequence_batch
        calls = 0
        seed_statements = 0
        original_execute = backend.execute

        async def observe_seed(query, params=()):
            nonlocal seed_statements
            normalized = " ".join(query.split())
            if (
                "INSERT INTO durable_signal_source_sequence_event_work" in normalized
                and "FROM durable_signal_events" in normalized
            ):
                seed_statements += 1
            return await original_execute(query, params)

        async def commit_one_batch_then_fail():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated between-batch crash")
            return await original_batch()

        monkeypatch.setattr(
            interrupted,
            "_backfill_postgres_source_sequence_batch",
            commit_one_batch_then_fail,
        )
        monkeypatch.setattr(backend, "execute", observe_seed)
        with pytest.raises(TransactionError, match="between-batch crash"):
            await interrupted.initialize()

        assert await backend.fetch_val(
            "SELECT backfill_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is False
        assert await backend.fetch_val(
            "SELECT event_work_seeded "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is True
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_event_work"
        ) == legacy_rows - store.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE
        first_batch = await backend.fetch_all(
            "SELECT event_id, source_sequence FROM durable_signal_events "
            "WHERE agent_id = ? AND source = ? "
            "AND source_sequence IS NOT NULL ORDER BY event_id",
            (agent_id, source),
        )
        assert len(first_batch) == store.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE

        await DurableSignalStore(backend).initialize()
        assert await backend.fetch_val(
            "SELECT backfill_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is True
        assert seed_statements == 1
        assert await backend.fetch_all(
            "SELECT event_id, source_sequence FROM durable_signal_events "
            "WHERE agent_id = ? AND source = ? "
            "AND event_id = ANY(?) ORDER BY event_id",
            (agent_id, source, [row[0] for row in first_batch]),
        ) == first_batch
        aggregate = await backend.fetch_one(
            "SELECT COUNT(*), MIN(source_sequence), MAX(source_sequence) "
            "FROM durable_signal_events WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        assert tuple(int(value) for value in aggregate) == (
            legacy_rows,
            1,
            legacy_rows,
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_true_legacy_postgres_migration_admits_fenced_mixed_version_ingress(
    db_backend, monkeypatch
):
    """Queued history coexists with new and primary-counter-only writers."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL mixed-version legacy migration regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        await _create_pre_source_sequence_postgres_events(backend)
        agent_id = f"did:test:true-legacy-mixed:{uuid4()}"
        source = "provider.message"
        token = uuid4().hex
        legacy_rows = DurableSignalStore.SOURCE_SEQUENCE_BACKFILL_BATCH_SIZE * 2 + 7
        await backend.execute(
            """
            INSERT INTO durable_signal_events (
                event_id, source_event_id, agent_id, target_agent, source,
                kind, mode, payload, visibility, urgency, causation_chain,
                arrived_at, committed_at, retention_until
            )
            SELECT md5(? || series::text), ? || series::text,
                   ?, ?, ?, 'inbound', 'action', '{}'::jsonb,
                   'internal', 'normal', '[]'::jsonb,
                   NOW(), NOW(), NOW() + INTERVAL '7 days'
            FROM generate_series(1, ?) AS series
            """,
            (
                token,
                f"true-legacy-mixed:{token}:",
                agent_id,
                agent_id,
                source,
                legacy_rows,
            ),
        )

        migrating = DurableSignalStore(backend)
        original_batch = migrating._backfill_postgres_source_sequence_batch
        first_batch_ready = asyncio.Event()
        release_batch = asyncio.Event()

        async def pause_before_first_batch():
            if not first_batch_ready.is_set():
                first_batch_ready.set()
                await release_batch.wait()
            return await original_batch()

        monkeypatch.setattr(
            migrating,
            "_backfill_postgres_source_sequence_batch",
            pause_before_first_batch,
        )
        legacy_backend = await _independent_backend(backend)
        live_backend = await _independent_backend(backend)
        migration_task = asyncio.create_task(migrating.initialize())
        try:
            await asyncio.wait_for(first_batch_ready.wait(), timeout=15)

            # The committed NOT VALID fence applies to every later statement,
            # including a replica that still omits the additive column.
            with pytest.raises(QueryError, match="source_sequence|source sequence"):
                await legacy_backend.execute(
                    """
                    INSERT INTO durable_signal_events (
                        event_id, source_event_id, agent_id, target_agent,
                        source, kind, mode, payload, visibility, urgency,
                        causation_chain, arrived_at, committed_at,
                        retention_until
                    )
                    SELECT ?, ?, agent_id, target_agent, source, kind, mode,
                           payload, visibility, urgency, causation_chain,
                           arrived_at, committed_at, retention_until
                    FROM durable_signal_events LIMIT 1
                    """,
                    (str(uuid4()), f"unsequenced-old:{uuid4()}"),
                )

            legacy_advance, live_event = await asyncio.wait_for(
                asyncio.gather(
                    _legacy_primary_only_advance(
                        legacy_backend, agent_id=agent_id, source=source
                    ),
                    DurableSignalStore(live_backend).persist_signal(
                        _signal(agent_id),
                        agent_id=agent_id,
                        source_event_id=f"true-legacy-live:{uuid4()}",
                        retention_days=7,
                    ),
                ),
                timeout=10,
            )
            assert {legacy_advance, live_event.source_sequence} == {1, 2}

            release_batch.set()
            await asyncio.wait_for(migration_task, timeout=20)
        finally:
            release_batch.set()
            await _cancel_and_drain(migration_task)
            await live_backend.close()
            await legacy_backend.close()

        aggregate = await backend.fetch_one(
            "SELECT COUNT(*), MIN(source_sequence), MAX(source_sequence) "
            "FROM durable_signal_events WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        assert tuple(int(value) for value in aggregate) == (
            legacy_rows + 1,
            live_event.source_sequence,
            legacy_rows + 2,
        )
        assert await backend.fetch_val(
            "SELECT current_sequence FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == legacy_rows + 2
        assert await backend.fetch_val(
            "SELECT recovery_sequence "
            "FROM durable_signal_source_sequence_recovery "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == legacy_rows + 2


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_concurrent_unique_index_build_keeps_ingress_writable(db_backend):
    """Real CONCURRENTLY build admits a writer while an old writer delays it."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL concurrent-index availability regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        blocker_agent = f"did:test:index-blocker:{uuid4()}"
        ingress_agent = f"did:test:index-ingress:{uuid4()}"
        seed = await store.persist_signal(
            _signal(blocker_agent),
            agent_id=blocker_agent,
            source_event_id=f"index-blocker:{uuid4()}",
            retention_days=7,
        )
        await backend.execute(
            "DROP INDEX CONCURRENTLY idx_durable_signal_events_scope_sequence"
        )
        blocker = await _independent_backend(backend)
        builder = await _independent_backend(backend)
        ingress = await _independent_backend(backend)
        blocker_started = asyncio.Event()
        release_blocker = asyncio.Event()

        async def hold_prebuild_writer():
            async with blocker.transaction():
                await blocker.execute(
                    "UPDATE durable_signal_events SET urgency = urgency "
                    "WHERE event_id = ?",
                    (seed.event_id,),
                )
                blocker_started.set()
                await release_blocker.wait()

        blocker_task = asyncio.create_task(hold_prebuild_writer())
        build_task = None
        try:
            await asyncio.wait_for(blocker_started.wait(), timeout=5)
            build_task = asyncio.create_task(
                DurableSignalStore(
                    builder
                )._ensure_postgres_source_sequence_index_concurrently()
            )
            await _wait_for_postgres_concurrent_index_build(backend)
            assert not build_task.done()

            live = await asyncio.wait_for(
                DurableSignalStore(ingress).persist_signal(
                    _signal(ingress_agent),
                    agent_id=ingress_agent,
                    source_event_id=f"index-live:{uuid4()}",
                    retention_days=7,
                ),
                timeout=3,
            )
            assert live.source_sequence == 1
            assert not build_task.done()

            release_blocker.set()
            await asyncio.wait_for(
                asyncio.gather(blocker_task, build_task), timeout=15
            )
            catalog = await DurableSignalStore(
                backend
            )._postgres_source_sequence_index_catalog()
            assert DurableSignalStore._postgres_source_sequence_index_catalog_valid(
                catalog
            )
        finally:
            release_blocker.set()
            await _cancel_and_drain(blocker_task, build_task)
            await blocker.close()
            await builder.close()
            await ingress.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_retries_an_invalid_interrupted_concurrent_index(db_backend):
    """An interrupted owned-name shell is removed before a safe retry."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL interrupted concurrent-index regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        agent_id = f"did:test:index-interruption:{uuid4()}"
        seed = await store.persist_signal(
            _signal(agent_id),
            agent_id=agent_id,
            source_event_id=f"index-interruption:{uuid4()}",
            retention_days=7,
        )
        await backend.execute(
            "DROP INDEX CONCURRENTLY idx_durable_signal_events_scope_sequence"
        )
        blocker = await _independent_backend(backend)
        builder = await _independent_backend(backend)
        blocker_started = asyncio.Event()
        release_blocker = asyncio.Event()

        async def hold_prebuild_writer():
            async with blocker.transaction():
                await blocker.execute(
                    "UPDATE durable_signal_events SET urgency = urgency "
                    "WHERE event_id = ?",
                    (seed.event_id,),
                )
                blocker_started.set()
                await release_blocker.wait()

        blocker_task = asyncio.create_task(hold_prebuild_writer())
        interrupted_task = None
        try:
            await asyncio.wait_for(blocker_started.wait(), timeout=5)
            interrupted_task = asyncio.create_task(
                builder.execute(
                    "CREATE UNIQUE INDEX CONCURRENTLY "
                    "idx_durable_signal_events_scope_sequence "
                    "ON durable_signal_events "
                    "(agent_id, source, source_sequence)"
                )
            )
            await _wait_for_postgres_concurrent_index_build(backend)
            interrupted_task.cancel()
            await asyncio.gather(interrupted_task, return_exceptions=True)
            release_blocker.set()
            await asyncio.wait_for(blocker_task, timeout=10)

            interrupted_catalog = (
                await store._postgres_source_sequence_index_catalog()
            )
            assert interrupted_catalog is not None
            assert not store._postgres_source_sequence_index_catalog_valid(
                interrupted_catalog
            )

            await DurableSignalStore(backend).initialize()
            repaired_catalog = await store._postgres_source_sequence_index_catalog()
            assert store._postgres_source_sequence_index_catalog_valid(
                repaired_catalog
            )
        finally:
            release_blocker.set()
            await _cancel_and_drain(blocker_task, interrupted_task)
            await blocker.close()
            await builder.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_statement_recovery_mirror_aggregates_each_exact_scope(
    db_backend,
):
    """One INSERT/UPDATE statement mirrors one MAX per tenant/source group."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL transition-table trigger regression")

    async with _isolated_durable_schema(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        token = uuid4().hex
        await backend.execute(
            """
            INSERT INTO durable_signal_events (
                event_id, source_event_id, agent_id, target_agent, source,
                kind, mode, payload, visibility, urgency, causation_chain,
                arrived_at, committed_at, retention_until, source_sequence
            )
            SELECT md5(? || scope.agent_id || scope.source || series::text),
                   ? || scope.agent_id || ':' || scope.source || ':' || series,
                   scope.agent_id, scope.agent_id, scope.source,
                   'inbound', 'action', '{}'::jsonb, 'internal', 'normal',
                   '[]'::jsonb, NOW(), NOW(), NOW() + INTERVAL '7 days', series
            FROM (VALUES
                    ('did:test:mirror:a', 'provider.one', 5),
                    ('did:test:mirror:a', 'provider.two', 3),
                    ('did:test:mirror:b', 'provider.one', 7)
                 ) AS scope(agent_id, source, maximum)
            CROSS JOIN LATERAL generate_series(1, scope.maximum) AS series
            """,
            (token, f"mirror:{token}:"),
        )
        rows = await backend.fetch_all(
            "SELECT agent_id, source, recovery_sequence "
            "FROM durable_signal_source_sequence_recovery "
            "ORDER BY agent_id, source"
        )
        assert [tuple(row) for row in rows] == [
            ("did:test:mirror:a", "provider.one", 5),
            ("did:test:mirror:a", "provider.two", 3),
            ("did:test:mirror:b", "provider.one", 7),
        ]
        high_water_rows = await backend.fetch_all(
            "SELECT agent_id, source, high_water_sequence "
            "FROM durable_signal_source_sequence_high_water "
            "ORDER BY agent_id, source"
        )
        assert [tuple(row) for row in high_water_rows] == [
            ("did:test:mirror:a", "provider.one", 5),
            ("did:test:mirror:a", "provider.two", 3),
            ("did:test:mirror:b", "provider.one", 7),
        ]
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_seen"
        ) == 3

        # UPDATE must be a separate transition-table trigger (PostgreSQL does
        # not permit a column list or multi-event trigger with transition
        # relations) and must aggregate the NEW rows across all three scopes.
        await backend.execute(
            "UPDATE durable_signal_events "
            "SET source_sequence = source_sequence + 100"
        )
        rows = await backend.fetch_all(
            "SELECT agent_id, source, recovery_sequence "
            "FROM durable_signal_source_sequence_recovery "
            "ORDER BY agent_id, source"
        )
        assert [tuple(row) for row in rows] == [
            ("did:test:mirror:a", "provider.one", 105),
            ("did:test:mirror:a", "provider.two", 103),
            ("did:test:mirror:b", "provider.one", 107),
        ]
        high_water_rows = await backend.fetch_all(
            "SELECT agent_id, source, high_water_sequence "
            "FROM durable_signal_source_sequence_high_water "
            "ORDER BY agent_id, source"
        )
        assert [tuple(row) for row in high_water_rows] == [
            ("did:test:mirror:a", "provider.one", 105),
            ("did:test:mirror:a", "provider.two", 103),
            ("did:test:mirror:b", "provider.one", 107),
        ]
        assert await store._postgres_source_sequence_recovery_sync_valid()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_schema_upgrade_does_not_invert_source_handoff_lock(
    db_backend, monkeypatch
):
    """Final PostgreSQL DDL begins only after scope-row locks are committed."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL advisory/table lock ordering regression")

    peer_backend = await _independent_backend(db_backend)
    probe_backend = await _independent_backend(db_backend)
    seed_store = DurableSignalStore(db_backend)
    writer_store = DurableSignalStore(peer_backend)
    await seed_store.initialize()
    agent_id = f"did:test:source-boundary-upgrade-race:{uuid4()}"
    source_event_id = f"upgrade-race-seed:{uuid4()}"
    seed = await seed_store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=source_event_id,
        retention_days=7,
    )

    # Restore one nullable legacy row and remove the completed fence so the
    # next initializer must run both migration phases.
    async with db_backend.transaction():
        await _drop_postgres_source_recovery_trigger_family(db_backend)
        await db_backend.execute(
            "ALTER TABLE durable_signal_events DROP CONSTRAINT IF EXISTS "
            "durable_signal_events_source_sequence_not_null"
        )
        await db_backend.execute(
            "ALTER TABLE durable_signal_events "
            "ALTER COLUMN source_sequence DROP NOT NULL"
        )
        await db_backend.execute(
            "UPDATE durable_signal_events SET source_sequence = NULL "
            "WHERE event_id = ?",
            (seed.event_id,),
        )
        await db_backend.execute(
            "DELETE FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, "provider.message"),
        )
        await db_backend.execute(
            "DELETE FROM durable_signal_source_sequence_recovery "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, "provider.message"),
        )
        # This fixture models a pre-seen-marker legacy scope, not corruption
        # after a positive sequence was served under the new contract.
        await db_backend.execute(
            "DELETE FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, "provider.message"),
        )
        await db_backend.execute(
            "DELETE FROM durable_signal_source_sequence_high_water "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, "provider.message"),
        )

    migrating_store = DurableSignalStore(db_backend)
    counter_owned = asyncio.Event()
    release_migration = asyncio.Event()
    writer_owns_handoff = asyncio.Event()
    final_ddl_requested = asyncio.Event()
    migration_task = None
    writer_task = None
    original_source_sequence = migrating_store._source_sequence_locked
    original_migration_handoff = migrating_store._lock_scope_handoff
    original_handoff = writer_store._lock_scope_handoff
    original_enforce = migrating_store._enforce_postgres_source_sequence_required
    backfill_transaction_id = None
    migration_owns_handoff = asyncio.Event()

    async def observe_migration_handoff(**kwargs):
        await original_migration_handoff(**kwargs)
        if kwargs == {"agent_id": agent_id, "source": "provider.message"}:
            migration_owns_handoff.set()

    async def pause_with_counter_row_locked(**kwargs):
        nonlocal backfill_transaction_id
        sequence = await original_source_sequence(**kwargs)
        if (
            kwargs["agent_id"] == agent_id
            and kwargs["source"] == "provider.message"
        ):
            backfill_transaction_id = await db_backend.fetch_val(
                "SELECT txid_current()"
            )
            counter_owned.set()
            await release_migration.wait()
        return sequence

    async def observe_owned_handoff(**kwargs):
        await original_handoff(**kwargs)
        writer_owns_handoff.set()

    async def prove_counter_rows_committed_before_final_ddl(state):
        final_transaction_id = await db_backend.fetch_val("SELECT txid_current()")
        assert backfill_transaction_id is not None
        assert final_transaction_id != backfill_transaction_id
        # A distinct connection can lock both rows within a bounded timeout
        # only after the backfill transaction has committed. This is the
        # load-bearing proof: observing Python call order alone cannot
        # establish lock release.
        async with probe_backend.transaction():
            await probe_backend.execute("SET LOCAL lock_timeout = '2s'")
            assert (
                await probe_backend.fetch_one(
                    "SELECT current_sequence "
                    "FROM durable_signal_source_sequences "
                    "WHERE agent_id = ? AND source = ? FOR UPDATE",
                    (agent_id, "provider.message"),
                )
                is not None
            )
            assert (
                await probe_backend.fetch_one(
                    "SELECT recovery_sequence "
                    "FROM durable_signal_source_sequence_recovery "
                    "WHERE agent_id = ? AND source = ? FOR UPDATE",
                    (agent_id, "provider.message"),
                )
                is not None
            )
        final_ddl_requested.set()
        await original_enforce(state)

    monkeypatch.setattr(
        migrating_store, "_source_sequence_locked", pause_with_counter_row_locked
    )
    monkeypatch.setattr(
        migrating_store, "_lock_scope_handoff", observe_migration_handoff
    )
    monkeypatch.setattr(writer_store, "_lock_scope_handoff", observe_owned_handoff)
    monkeypatch.setattr(
        migrating_store,
        "_enforce_postgres_source_sequence_required",
        prove_counter_rows_committed_before_final_ddl,
    )
    try:
        migration_task = asyncio.create_task(migrating_store.initialize())
        # Reaching the target counter proves the ACCESS EXCLUSIVE fence phase
        # has committed; history work no longer holds that table lock.
        await asyncio.wait_for(counter_owned.wait(), timeout=5)
        assert migration_owns_handoff.is_set()
        writer_task = asyncio.create_task(
            writer_store.persist_signal(
                _signal(agent_id),
                agent_id=agent_id,
                source_event_id=f"upgrade-race-live:{uuid4()}",
                retention_days=7,
            )
        )
        await asyncio.sleep(0.1)
        assert writer_task is not None and not writer_task.done()
        assert not writer_owns_handoff.is_set()

        # Backfill owns handoff before counter rows. The live writer waits at
        # that first lock, so no handoff->counter / counter->handoff ABBA cycle
        # exists and it commits the next sequence only after repair.
        release_migration.set()
        _, persisted = await asyncio.wait_for(
            asyncio.gather(migration_task, writer_task), timeout=10
        )
        assert writer_owns_handoff.is_set()
        assert final_ddl_requested.is_set()
        assert persisted.source_sequence == 2
    finally:
        release_migration.set()
        await _cancel_and_drain(migration_task, writer_task)
        await probe_backend.close()
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "winner",
    ("capture", "backfill"),
    ids=("capture-before-backfill", "backfill-before-capture"),
)
async def test_postgres_boundary_and_repair_share_scope_handoff_order(
    db_backend, winner
):
    """Real PostgreSQL linearizes repair and capture without counter ABBA."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL repair/capture handoff regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        repair_store, agent_id, source, event_ids = (
            await _prepare_interrupted_postgres_source_backfill(
                backend, legacy_rows=1
            )
        )
        capture_backend = await _independent_backend(backend)
        capture_store = DurableSignalStore(capture_backend)
        handoff_owned = asyncio.Event()
        release_winner = asyncio.Event()
        repair_task = None
        capture_task = None

        async def run_repair_batch():
            async with backend.transaction():
                return await repair_store._backfill_postgres_source_sequence_batch()

        try:
            if winner == "capture":
                original_handoff = capture_store._lock_scope_handoff

                async def hold_capture_handoff(**kwargs):
                    await original_handoff(**kwargs)
                    handoff_owned.set()
                    await release_winner.wait()

                capture_store._lock_scope_handoff = (  # type: ignore[method-assign]
                    hold_capture_handoff
                )
                capture_task = asyncio.create_task(
                    capture_store.capture_source_boundary(
                        agent_id=agent_id, source=source
                    )
                )
                await asyncio.wait_for(handoff_owned.wait(), timeout=5)
                repair_task = asyncio.create_task(run_repair_batch())
                await asyncio.sleep(0.1)
                assert not repair_task.done()
                release_winner.set()

                capture_result, repaired = await asyncio.wait_for(
                    asyncio.gather(
                        capture_task, repair_task, return_exceptions=True
                    ),
                    timeout=10,
                )
                assert isinstance(capture_result, TransactionError)
                assert "unsequenced legacy history" in str(capture_result)
                assert repaired is True
            else:
                original_handoff = repair_store._lock_scope_handoff

                async def hold_repair_handoff(**kwargs):
                    await original_handoff(**kwargs)
                    handoff_owned.set()
                    await release_winner.wait()

                repair_store._lock_scope_handoff = (  # type: ignore[method-assign]
                    hold_repair_handoff
                )
                repair_task = asyncio.create_task(run_repair_batch())
                await asyncio.wait_for(handoff_owned.wait(), timeout=5)
                capture_task = asyncio.create_task(
                    capture_store.capture_source_boundary(
                        agent_id=agent_id, source=source
                    )
                )
                await asyncio.sleep(0.1)
                assert not capture_task.done()
                release_winner.set()

                repaired, capture_result = await asyncio.wait_for(
                    asyncio.gather(repair_task, capture_task), timeout=10
                )
                assert repaired is True
                assert capture_result.sequence == 1

            assert await backend.fetch_val(
                "SELECT source_sequence FROM durable_signal_events "
                "WHERE event_id = ?",
                (event_ids[0],),
            ) == 1
            assert (
                await capture_store.capture_source_boundary(
                    agent_id=agent_id, source=source
                )
            ).sequence == 1
        finally:
            release_winner.set()
            await _cancel_and_drain(repair_task, capture_task)
            await capture_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("drift", ("partial", "recreated"))
async def test_postgres_seeded_event_work_drift_reseeds_and_converges_after_restart(
    db_backend, drift
):
    """A true seed bit cannot strand NULL history after work-table loss."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL durable backfill work restart regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        _, agent_id, source, event_ids = (
            await _prepare_interrupted_postgres_source_backfill(
                backend, legacy_rows=3
            )
        )
        if drift == "partial":
            await backend.execute(
                "DELETE FROM durable_signal_source_sequence_event_work "
                "WHERE event_id = ?",
                (event_ids[1],),
            )
            assert await backend.fetch_val(
                "SELECT COUNT(*) FROM durable_signal_source_sequence_event_work"
            ) == 2
        else:
            await backend.execute(
                "DROP TABLE durable_signal_source_sequence_event_work"
            )
            assert await backend.fetch_val(
                "SELECT event_work_seeded "
                "FROM durable_signal_source_sequence_state WHERE singleton = 1"
            ) is True

        restarted = DurableSignalStore(backend)
        await restarted.initialize()
        rows = await backend.fetch_all(
            "SELECT event_id, source_sequence FROM durable_signal_events "
            "WHERE agent_id = ? AND source = ? ORDER BY source_sequence",
            (agent_id, source),
        )
        assert {str(row[0]) for row in rows} == set(event_ids)
        assert [int(row[1]) for row in rows] == [1, 2, 3]
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_event_work"
        ) == 0
        assert await backend.fetch_val(
            "SELECT event_work_seeded "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is True
        assert await backend.fetch_val(
            "SELECT backfill_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is True
        assert (
            await restarted.capture_source_boundary(
                agent_id=agent_id, source=source
            )
        ).sequence == 3


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "scope_queue_drift",
    ("empty", "partial", "missing-row", "recreated"),
)
async def test_postgres_scope_work_drift_validates_every_primary_before_completion(
    db_backend, monkeypatch, scope_queue_drift
):
    """A seed bit cannot skip a primary scope after narrow-queue damage."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL durable scope-work restart regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        await _create_pre_source_sequence_postgres_events(backend)
        await backend.execute(
            """
            CREATE TABLE durable_signal_source_sequences (
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL,
                current_sequence BIGINT NOT NULL,
                PRIMARY KEY (agent_id, source),
                CHECK (current_sequence >= 0)
            )
            """
        )
        token = uuid4().hex
        agent_a = f"did:test:scope-drift:a:{token}"
        agent_b = f"did:test:scope-drift:b:{token}"
        scopes = (
            (agent_a, "provider.alpha", 4),
            (agent_a, "provider.beta", 8),
            (agent_b, "provider.alpha", 12),
            (agent_b, "provider.gamma", 16),
        )
        for index, (agent_id, source, sequence) in enumerate(scopes):
            await backend.execute(
                "INSERT INTO durable_signal_source_sequences "
                "(agent_id, source, current_sequence) VALUES (?, ?, ?)",
                (agent_id, source, sequence),
            )
            event_id = f"scope-drift-{index}-{token}"
            await backend.execute(
                """
                INSERT INTO durable_signal_events (
                    event_id, source_event_id, agent_id, target_agent, source,
                    kind, mode, payload, visibility, urgency, causation_chain,
                    arrived_at, committed_at, retention_until
                ) VALUES (
                    ?, ?, ?, ?, ?, 'inbound', 'action', '{}'::jsonb,
                    'internal', 'normal', '[]'::jsonb,
                    NOW(), NOW(), NOW() + INTERVAL '7 days'
                )
                """,
                (
                    event_id,
                    f"scope-drift:{index}:{token}",
                    agent_id,
                    agent_id,
                    source,
                ),
            )

        interrupted = DurableSignalStore(backend)
        original_adoption = (
            interrupted._adopt_postgres_source_sequence_recovery_batch
        )

        async def stop_after_scope_seed():
            raise RuntimeError("simulated interruption after scope seed")

        monkeypatch.setattr(
            interrupted,
            "_adopt_postgres_source_sequence_recovery_batch",
            stop_after_scope_seed,
        )
        with pytest.raises(TransactionError, match="after scope seed"):
            await interrupted.initialize()
        monkeypatch.setattr(
            interrupted,
            "_adopt_postgres_source_sequence_recovery_batch",
            original_adoption,
        )
        assert await backend.fetch_val(
            "SELECT scope_work_seeded "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is True
        assert await backend.fetch_val(
            "SELECT scope_validation_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is False
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_scope_work"
        ) == len(scopes)

        if scope_queue_drift == "empty":
            await backend.execute(
                "DELETE FROM durable_signal_source_sequence_scope_work"
            )
        elif scope_queue_drift == "partial":
            async with backend.transaction():
                assert await original_adoption() is True
        elif scope_queue_drift == "missing-row":
            missing_agent, missing_source, _ = scopes[1]
            await backend.execute(
                "DELETE FROM durable_signal_source_sequence_scope_work "
                "WHERE agent_id = ? AND source = ?",
                (missing_agent, missing_source),
            )
        else:
            await backend.execute(
                "DROP TABLE durable_signal_source_sequence_scope_work"
            )
            await backend.execute(
                """
                CREATE TABLE durable_signal_source_sequence_scope_work (
                    agent_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY (agent_id, source)
                )
                """
            )

        restarted = DurableSignalStore(backend)
        await restarted.initialize()
        marker = await backend.fetch_one(
            "SELECT backfill_completed, scope_validation_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        )
        assert tuple(bool(value) for value in marker) == (True, True)
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_scope_work"
        ) == 0
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_source_sequence_event_work"
        ) == 0

        expected = {
            (agent_id, source): sequence + 1
            for agent_id, source, sequence in scopes
        }
        for relation, column in (
            ("durable_signal_source_sequences", "current_sequence"),
            (
                "durable_signal_source_sequence_recovery",
                "recovery_sequence",
            ),
            (
                "durable_signal_source_sequence_high_water",
                "high_water_sequence",
            ),
        ):
            rows = await backend.fetch_all(
                f"SELECT agent_id, source, {column} FROM {relation} "
                "ORDER BY agent_id, source"
            )
            assert {
                (str(agent_id), str(source)): int(sequence)
                for agent_id, source, sequence in rows
            } == expected
        assert {
            (str(row[0]), str(row[1]))
            for row in await backend.fetch_all(
                "SELECT agent_id, source "
                "FROM durable_signal_source_sequence_seen"
            )
        } == set(expected)
        assert {
            (str(row[0]), str(row[1])): int(row[2])
            for row in await backend.fetch_all(
                "SELECT agent_id, source, source_sequence "
                "FROM durable_signal_events"
            )
        } == expected

        expired = restarted.to_timestamp_param(
            datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        await backend.execute(
            "UPDATE durable_signal_events SET retention_until = ?",
            (expired,),
        )
        assert await restarted.purge_expired(agent_id=agent_a) == 2
        assert await restarted.purge_expired(agent_id=agent_b) == 2
        assert await backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_events"
        ) == 0

        row_loss_scope = (agent_a, "provider.beta")
        await backend.execute(
            "DELETE FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            row_loss_scope,
        )
        assert (
            await restarted.capture_source_boundary(
                agent_id=row_loss_scope[0], source=row_loss_scope[1]
            )
        ).sequence == expected[row_loss_scope]
        after_row_loss = await restarted.persist_signal(
            Signal(
                source=row_loss_scope[1],
                kind="inbound",
                mode=SignalMode.ACTION,
                payload={"workflow": "wf-1"},
                target_agent=row_loss_scope[0],
            ),
            agent_id=row_loss_scope[0],
            source_event_id=f"scope-drift-row-loss:{token}",
            retention_days=7,
        )
        assert after_row_loss.source_sequence == expected[row_loss_scope] + 1

        await backend.execute("DROP TABLE durable_signal_source_sequences")
        after_table_loss = DurableSignalStore(backend)
        await after_table_loss.initialize()
        table_loss_scope = (agent_b, "provider.alpha")
        assert (
            await after_table_loss.capture_source_boundary(
                agent_id=table_loss_scope[0], source=table_loss_scope[1]
            )
        ).sequence == expected[table_loss_scope]
        next_ingress = await after_table_loss.persist_signal(
            Signal(
                source=table_loss_scope[1],
                kind="inbound",
                mode=SignalMode.ACTION,
                payload={"workflow": "wf-1"},
                target_agent=table_loss_scope[0],
            ),
            agent_id=table_loss_scope[0],
            source_event_id=f"scope-drift-table-loss:{token}",
            retention_days=7,
        )
        assert next_ingress.source_sequence == expected[table_loss_scope] + 1

        fast_path = DurableSignalStore(backend)
        forbidden_validation = AsyncMock(
            side_effect=AssertionError("completed boot repeated scope validation")
        )
        monkeypatch.setattr(
            fast_path,
            "_validate_postgres_source_sequence_scope_batch",
            forbidden_validation,
        )
        observed_sql: list[str] = []
        originals = {
            name: getattr(backend, name)
            for name in ("execute", "fetch_one", "fetch_all", "fetch_val")
        }

        def recorder(name):
            async def record(query, params=()):
                observed_sql.append(" ".join(query.split()))
                return await originals[name](query, params)

            return record

        for name in originals:
            monkeypatch.setattr(backend, name, recorder(name))
        await fast_path.initialize()
        forbidden_validation.assert_not_awaited()
        normalized = "\n".join(observed_sql)
        assert "WHERE (agent_id, source) >" not in normalized
        assert "WHERE source_sequence IS NULL" not in normalized


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_scope_validation_cursor_restarts_on_primary_key_index(
    db_backend, monkeypatch
):
    """Cursor commits one scope at a time and resumes by indexed keyset."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL durable scope-validation cursor regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        await _create_pre_source_sequence_postgres_events(backend)
        await backend.execute(
            """
            CREATE TABLE durable_signal_source_sequences (
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL,
                current_sequence BIGINT NOT NULL,
                PRIMARY KEY (agent_id, source),
                CHECK (current_sequence >= 0)
            )
            """
        )
        token = uuid4().hex
        scopes = tuple(
            (
                f"did:test:scope-cursor:{tenant:02d}:{token}",
                f"provider.{source:02d}",
                tenant * 10 + source + 1,
            )
            for tenant in range(4)
            for source in range(4)
        )
        for scope in scopes:
            await backend.execute(
                "INSERT INTO durable_signal_source_sequences "
                "(agent_id, source, current_sequence) VALUES (?, ?, ?)",
                scope,
            )

        seeded = DurableSignalStore(backend)

        async def stop_after_scope_seed():
            raise RuntimeError("simulated interruption after cursor seed")

        monkeypatch.setattr(
            seeded,
            "_adopt_postgres_source_sequence_recovery_batch",
            stop_after_scope_seed,
        )
        with pytest.raises(TransactionError, match="after cursor seed"):
            await seeded.initialize()
        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_scope_work"
        )

        interrupted = DurableSignalStore(backend)
        original_validation = (
            interrupted._validate_postgres_source_sequence_scope_batch
        )
        validation_calls = 0

        async def commit_two_scopes_then_fail():
            nonlocal validation_calls
            validation_calls += 1
            if validation_calls == 3:
                raise RuntimeError("simulated scope-cursor interruption")
            return await original_validation()

        monkeypatch.setattr(
            interrupted,
            "_validate_postgres_source_sequence_scope_batch",
            commit_two_scopes_then_fail,
        )
        with pytest.raises(TransactionError, match="scope-cursor interruption"):
            await interrupted.initialize()

        cursor = await backend.fetch_one(
            "SELECT scope_validation_after_agent_id, "
            "scope_validation_after_source, scope_validation_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        )
        ordered_scopes = sorted(scopes)
        assert (str(cursor[0]), str(cursor[1])) == ordered_scopes[1][:2]
        assert cursor[2] is False

        async with backend.transaction():
            await backend.execute("SET LOCAL enable_seqscan = off")
            plan = await backend.fetch_val(
                """
                EXPLAIN (FORMAT JSON)
                SELECT agent_id, source
                FROM durable_signal_source_sequences
                WHERE (agent_id, source) > (?, ?)
                ORDER BY agent_id, source
                LIMIT 1
                """,
                (str(cursor[0]), str(cursor[1])),
            )
        rendered_plan = json.dumps(plan, default=str)
        assert "durable_signal_source_sequences_pkey" in rendered_plan
        assert "Index Scan" in rendered_plan or "Index Only Scan" in rendered_plan

        restarted = DurableSignalStore(backend)
        await restarted.initialize()
        assert await backend.fetch_val(
            "SELECT scope_validation_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is True
        assert await backend.fetch_val(
            "SELECT backfill_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is True
        for relation, column in (
            ("durable_signal_source_sequences", "current_sequence"),
            (
                "durable_signal_source_sequence_recovery",
                "recovery_sequence",
            ),
            (
                "durable_signal_source_sequence_high_water",
                "high_water_sequence",
            ),
        ):
            rows = await backend.fetch_all(
                f"SELECT agent_id, source, {column} FROM {relation} "
                "ORDER BY agent_id, source"
            )
            assert [tuple(row) for row in rows] == [
                (agent_id, source, sequence)
                for agent_id, source, sequence in ordered_scopes
            ]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_old_completion_without_scope_validation_is_rebuilt(
    db_backend,
):
    """An older all-clear cannot survive without the new adoption proof."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL scope-validation marker upgrade regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        agent_id = f"did:test:old-scope-marker:{uuid4()}"
        source = "provider.message"
        persisted = await store.persist_signal(
            _signal(agent_id),
            agent_id=agent_id,
            source_event_id=f"old-scope-marker:{uuid4()}",
            retention_days=7,
        )
        assert persisted.source_sequence == 1

        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_recovery "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_high_water "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await backend.execute(
            "DELETE FROM durable_signal_source_sequence_seen "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await backend.execute(
            "ALTER TABLE durable_signal_source_sequence_state "
            "DROP COLUMN scope_validation_completed, "
            "DROP COLUMN scope_validation_after_agent_id, "
            "DROP COLUMN scope_validation_after_source"
        )
        assert await backend.fetch_val(
            "SELECT backfill_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) is True

        restarted = DurableSignalStore(backend)
        await restarted.initialize()
        marker = await backend.fetch_one(
            "SELECT backfill_completed, scope_validation_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        )
        assert tuple(bool(value) for value in marker) == (True, True)
        assert (
            await restarted.capture_source_boundary(
                agent_id=agent_id, source=source
            )
        ).sequence == 1
        exact = await backend.fetch_one(
            "SELECT primary_counter.current_sequence, "
            "recovery.recovery_sequence, high_water.high_water_sequence, "
            "seen.agent_id IS NOT NULL "
            "FROM durable_signal_source_sequences AS primary_counter "
            "JOIN durable_signal_source_sequence_recovery AS recovery "
            "USING (agent_id, source) "
            "JOIN durable_signal_source_sequence_high_water AS high_water "
            "USING (agent_id, source) "
            "LEFT JOIN durable_signal_source_sequence_seen AS seen "
            "USING (agent_id, source) "
            "WHERE primary_counter.agent_id = ? "
            "AND primary_counter.source = ?",
            (agent_id, source),
        )
        assert tuple(exact) == (1, 1, 1, True)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_recovery_recreation_adopts_primary_before_live_ingress(
    db_backend, monkeypatch
):
    """Recovery-table recreation cannot form an ABBA cycle with ingress."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL primary/recovery row-lock race")

    peer_backend = await _independent_backend(db_backend)
    seed_store = DurableSignalStore(db_backend)
    writer_store = DurableSignalStore(peer_backend)
    await seed_store.initialize()
    target_agent = f"did:test:recovery-adoption-race:z:{uuid4()}"
    earlier_agent = f"did:test:recovery-adoption-race:a:{uuid4()}"
    source = "provider.message"
    for agent_id in (earlier_agent, target_agent):
        await seed_store.persist_signal(
            _signal(agent_id),
            agent_id=agent_id,
            source_event_id=f"recovery-adoption-seed:{uuid4()}",
            retention_days=7,
        )

    # Recreate the independently recoverable table while preserving primary
    # rows and immutable scope markers. The next initializer adopts one scope
    # per transaction, always primary before recovery; pause on the target
    # scope to prove a live writer cannot invert that edge.
    await db_backend.execute("DROP TABLE durable_signal_source_sequence_recovery")
    migrating_store = DurableSignalStore(db_backend)
    primary_family_locked = asyncio.Event()
    release_adoption = asyncio.Event()
    migration_task = None
    writer_task = None
    original_source_sequence_locked = migrating_store._source_sequence_locked

    async def pause_with_target_primary_locked(
        *, agent_id, source, allow_retained_reconstruction=False
    ):
        sequence = await original_source_sequence_locked(
            agent_id=agent_id,
            source=source,
            allow_retained_reconstruction=allow_retained_reconstruction,
        )
        if agent_id == target_agent:
            primary_family_locked.set()
            await release_adoption.wait()
        return sequence

    monkeypatch.setattr(
        migrating_store,
        "_source_sequence_locked",
        pause_with_target_primary_locked,
    )
    try:
        migration_task = asyncio.create_task(migrating_store.initialize())
        await asyncio.wait_for(primary_family_locked.wait(), timeout=5)
        writer_task = asyncio.create_task(
            writer_store.persist_signal(
                _signal(target_agent),
                agent_id=target_agent,
                source_event_id=f"recovery-adoption-live:{uuid4()}",
                retention_days=7,
            )
        )
        await asyncio.sleep(0.2)
        assert writer_task.done() is False

        release_adoption.set()
        _, persisted = await asyncio.wait_for(
            asyncio.gather(migration_task, writer_task), timeout=10
        )
        assert persisted.source_sequence == 2
        assert await db_backend.fetch_val(
            "SELECT current_sequence FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            (target_agent, source),
        ) == 2
        assert await db_backend.fetch_val(
            "SELECT recovery_sequence "
            "FROM durable_signal_source_sequence_recovery "
            "WHERE agent_id = ? AND source = ?",
            (target_agent, source),
        ) == 2
    finally:
        release_adoption.set()
        await _cancel_and_drain(migration_task, writer_task)
        # Leave the shared parity database in the completed family shape.
        await DurableSignalStore(db_backend).initialize()
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_restart_after_backfill_commit_uses_durable_completion_marker(
    db_backend, monkeypatch
):
    """A phase-three failure cannot lose or falsely repeat phase-two work."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL phased-bootstrap interruption regression")

    seed_store = DurableSignalStore(db_backend)
    await seed_store.initialize()
    agent_id = f"did:test:source-boundary-upgrade-restart:{uuid4()}"
    source = "provider.message"
    seed = await seed_store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"upgrade-restart-seed:{uuid4()}",
        retention_days=7,
    )

    # Model an interrupted pre-completion PostgreSQL shape. Removing the
    # trusted fence invalidates the old marker on phase one; phase two must
    # repair this row and its exact counter copies before writing a new marker.
    async with db_backend.transaction():
        await _drop_postgres_source_recovery_trigger_family(db_backend)
        await db_backend.execute(
            "ALTER TABLE durable_signal_events DROP CONSTRAINT IF EXISTS "
            "durable_signal_events_source_sequence_not_null"
        )
        await db_backend.execute(
            "ALTER TABLE durable_signal_events "
            "ALTER COLUMN source_sequence DROP NOT NULL"
        )
        await db_backend.execute(
            "UPDATE durable_signal_events SET source_sequence = NULL "
            "WHERE event_id = ?",
            (seed.event_id,),
        )
        await db_backend.execute(
            "UPDATE durable_signal_source_sequences SET current_sequence = 0 "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )
        await db_backend.execute(
            "UPDATE durable_signal_source_sequence_recovery "
            "SET recovery_sequence = 0 WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        )

    interrupted = DurableSignalStore(db_backend)
    monkeypatch.setattr(
        interrupted,
        "_ensure_ordinary_indexes",
        AsyncMock(side_effect=RuntimeError("simulated phase-three interruption")),
    )
    try:
        with pytest.raises(TransactionError, match="phase-three interruption"):
            await interrupted.initialize()

        # Phase two committed independently: legacy history and both exact
        # counters agree, and the marker is durable even though phase-three
        # index/validation work rolled back.
        assert await db_backend.fetch_val(
            "SELECT backfill_completed "
            "FROM durable_signal_source_sequence_state WHERE singleton = 1"
        ) in (1, True)
        assert await db_backend.fetch_val(
            "SELECT source_sequence FROM durable_signal_events WHERE event_id = ?",
            (seed.event_id,),
        ) == 2
        assert await db_backend.fetch_val(
            "SELECT current_sequence FROM durable_signal_source_sequences "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 2
        assert await db_backend.fetch_val(
            "SELECT recovery_sequence "
            "FROM durable_signal_source_sequence_recovery "
            "WHERE agent_id = ? AND source = ?",
            (agent_id, source),
        ) == 2

        restarted = DurableSignalStore(db_backend)
        forbidden_backfill = AsyncMock(
            side_effect=AssertionError(
                "committed PostgreSQL backfill repeated after interruption"
            )
        )
        monkeypatch.setattr(
            restarted,
            "_backfill_postgres_source_sequence_batch",
            forbidden_backfill,
        )
        await restarted.initialize()
        forbidden_backfill.assert_not_awaited()
        assert (await restarted._source_sequence_schema_state()).enforced
    finally:
        # Leave the shared PostgreSQL fixture in its completed shape even if a
        # preceding assertion fails midway through the interruption scenario.
        await DurableSignalStore(db_backend).initialize()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_completed_migration_rejects_legacy_unsequenced_writers(db_backend):
    """An old replica cannot create ambiguous post-boundary history."""

    store = DurableSignalStore(db_backend)
    await store.initialize()
    agent_id = f"did:test:source-boundary-legacy-fence:{uuid4()}"
    seed = await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"legacy-fence-seed:{uuid4()}",
        retention_days=7,
    )
    legacy_event_id = str(uuid4())
    with pytest.raises(QueryError, match="source_sequence|source sequence"):
        await db_backend.execute(
            """
            INSERT OR IGNORE INTO durable_signal_events (
                event_id, source_event_id, agent_id, target_agent, source,
                kind, mode, payload, session_id, caller_identity,
                visibility, urgency, dedupe_key, causation_chain,
                arrived_at, committed_at, retention_until
            )
            SELECT ?, ?, agent_id, target_agent, source, kind, mode, payload,
                   session_id, caller_identity, visibility, urgency,
                   dedupe_key, causation_chain, arrived_at, committed_at,
                   retention_until
            FROM durable_signal_events
            WHERE event_id = ?
            """,
            (legacy_event_id, f"legacy-fence:{uuid4()}", seed.event_id),
        )
    assert (
        await db_backend.fetch_val(
            "SELECT COUNT(*) FROM durable_signal_events WHERE event_id = ?",
            (legacy_event_id,),
        )
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("invalid_sequence", (0, -1), ids=("zero", "negative"))
async def test_completed_migration_rejects_nonpositive_mixed_writer_sequences(
    db_backend, invalid_sequence
):
    """Both durable fences reject values outside the public 1-based contract."""

    store = DurableSignalStore(db_backend)
    await store.initialize()
    agent_id = f"did:test:source-boundary-positive:{uuid4()}"
    seed = await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"positive-fence-seed:{uuid4()}",
        retention_days=7,
    )
    invalid_event_id = str(uuid4())
    with pytest.raises(QueryError, match="source_sequence|source sequence"):
        await db_backend.execute(
            """
            INSERT OR IGNORE INTO durable_signal_events (
                event_id, source_event_id, agent_id, target_agent, source,
                kind, mode, payload, session_id, caller_identity,
                visibility, urgency, dedupe_key, causation_chain,
                arrived_at, committed_at, retention_until, source_sequence
            )
            SELECT ?, ?, agent_id, target_agent, source, kind, mode, payload,
                   session_id, caller_identity, visibility, urgency,
                   dedupe_key, causation_chain, arrived_at, committed_at,
                   retention_until, ?
            FROM durable_signal_events
            WHERE event_id = ?
            """,
            (
                invalid_event_id,
                f"positive-fence:{uuid4()}",
                invalid_sequence,
                seed.event_id,
            ),
        )
    assert await db_backend.fetch_val(
        "SELECT COUNT(*) FROM durable_signal_events WHERE event_id = ?",
        (invalid_event_id,),
    ) == 0


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_repairs_a_stale_named_source_sequence_constraint(db_backend):
    """A familiar constraint name with the old non-NULL-only shape is stale."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL catalog-definition repair regression")

    async with _isolated_durable_schema(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        await backend.execute(
            "ALTER TABLE durable_signal_events DROP CONSTRAINT "
            "durable_signal_events_source_sequence_not_null"
        )
        await backend.execute(
            "ALTER TABLE durable_signal_events ADD CONSTRAINT "
            "durable_signal_events_source_sequence_not_null "
            "CHECK (source_sequence IS NOT NULL)"
        )

        stale = await DurableSignalStore(backend)._source_sequence_schema_state()
        assert stale.fence_exists and stale.fence_validated
        assert stale.column_not_null
        assert not stale.fence_definition_valid
        assert not stale.enforced

        repaired = DurableSignalStore(backend)
        await repaired.initialize()
        state = await repaired._source_sequence_schema_state()
        assert state.enforced and state.fence_definition_valid
        expression = await backend.fetch_val(
            """
            SELECT pg_get_expr(conbin, conrelid, TRUE)
            FROM pg_constraint
            WHERE conrelid = to_regclass('durable_signal_events')
              AND conname = 'durable_signal_events_source_sequence_not_null'
              AND contype = 'c'
            """
        )
        assert expression is not None and ">= 1" in str(expression)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_recovery_catalog_rejects_and_atomically_retires_bad_families(
    db_backend, monkeypatch
):
    """Real catalog state, not familiar names, controls mirror completion."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL trigger/function catalog regression")

    async with _independent_postgres_schema_backend(db_backend) as backend:
        store = DurableSignalStore(backend)
        await store.initialize()
        insert_definition, update_definition = (
            DurableSignalStore.SOURCE_SEQUENCE_RECOVERY_DEFINITIONS
        )

        async def repair_current_mutation():
            assert not await store._postgres_source_sequence_recovery_sync_valid()
            await store.initialize()
            assert await store._postgres_source_sequence_recovery_sync_valid()

        await backend.execute(
            "ALTER TABLE durable_signal_events DISABLE TRIGGER "
            f'"{insert_definition.trigger_name}"'
        )
        await repair_current_mutation()

        await backend.execute(
            f'DROP TRIGGER "{update_definition.trigger_name}" '
            "ON durable_signal_events"
        )
        await backend.execute(
            f'CREATE TRIGGER "{update_definition.trigger_name}" '
            "AFTER UPDATE ON durable_signal_events FOR EACH ROW "
            "WHEN (NEW.source_sequence IS NOT NULL) "
            f'EXECUTE FUNCTION "{update_definition.function_name}"()'
        )
        await repair_current_mutation()

        await backend.execute(
            f'DROP TRIGGER "{update_definition.trigger_name}" '
            "ON durable_signal_events"
        )
        await backend.execute(
            f'CREATE CONSTRAINT TRIGGER "{update_definition.trigger_name}" '
            "AFTER UPDATE ON durable_signal_events "
            "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
            f'EXECUTE FUNCTION "{update_definition.function_name}"()'
        )
        await repair_current_mutation()

        await backend.execute(
            f'DROP TRIGGER "{insert_definition.trigger_name}" '
            "ON durable_signal_events"
        )
        await backend.execute(
            f'DROP FUNCTION "{insert_definition.function_name}"()'
        )
        await backend.execute(
            f'CREATE FUNCTION "{insert_definition.function_name}"() '
            "RETURNS trigger AS $bad$ BEGIN RETURN NULL; END $bad$ "
            "LANGUAGE plpgsql"
        )
        await backend.execute(insert_definition.trigger_ddl)
        await repair_current_mutation()

        stale_function = (
            DurableSignalStore.SOURCE_SEQUENCE_RECOVERY_FUNCTION_PREFIX
            + "i_deadbeef"
        )
        stale_trigger = (
            DurableSignalStore.SOURCE_SEQUENCE_RECOVERY_TRIGGER_PREFIX
            + "i_deadbeef"
        )
        await backend.execute(
            f'CREATE FUNCTION "{stale_function}"() RETURNS trigger '
            "AS $stale$ BEGIN RETURN NULL; END $stale$ LANGUAGE plpgsql"
        )
        await backend.execute(
            f'CREATE TRIGGER "{stale_trigger}" AFTER INSERT '
            "ON durable_signal_events FOR EACH ROW "
            f'EXECUTE FUNCTION "{stale_function}"()'
        )
        assert not await store._postgres_source_sequence_recovery_sync_valid()

        original_execute = backend.execute

        async def fail_after_family_retirement(query, params=()):
            if query.lstrip().startswith(
                f"CREATE FUNCTION {insert_definition.function_name}()"
            ):
                raise RuntimeError("simulated family install interruption")
            return await original_execute(query, params)

        monkeypatch.setattr(backend, "execute", fail_after_family_retirement)
        with pytest.raises(TransactionError, match="family install interruption"):
            await store.initialize()
        monkeypatch.setattr(backend, "execute", original_execute)
        # The failed transaction retired nothing: both stale objects remain,
        # proving there was no externally committed half-family.
        trigger_names, function_names = (
            await store._postgres_source_sequence_recovery_catalog()
        )
        assert stale_trigger in {str(row[0]) for row in trigger_names}
        assert stale_function in {str(row[0]) for row in function_names}

        await store.initialize()
        assert await store._postgres_source_sequence_recovery_sync_valid()
        trigger_names, function_names = (
            await store._postgres_source_sequence_recovery_catalog()
        )
        assert {str(row[0]) for row in trigger_names} == {
            item.trigger_name
            for item in DurableSignalStore.SOURCE_SEQUENCE_RECOVERY_DEFINITIONS
        }
        assert {str(row[0]) for row in function_names} == {
            item.function_name
            for item in DurableSignalStore.SOURCE_SEQUENCE_RECOVERY_DEFINITIONS
        }


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_durable_delivery_claim_is_scoped_and_single_owner_on_both_backends(db_backend):
    store = DurableSignalStore(db_backend)
    await store.initialize()
    # The hosted PostgreSQL fixture shares its database between test runs, so
    # keep this durable scope unique instead of colliding with retained ledger
    # rows from an earlier invocation.
    agent_id = f"did:test:durable-parity:{uuid4()}"
    await store.register_consumer(
        DurableConsumerRegistration(
            consumer_id="workflow-wait",
            source="provider.message",
            agent_id=agent_id,
            correlation_selector="payload.workflow=wf-1",
        )
    )
    event = _signal(agent_id)
    assert (await store.persist_signal(
        event, agent_id=agent_id, source_event_id="provider-event-1", retention_days=7
    )).created
    # The same explicit provider ID is the durable dedup key on both engines.
    assert not (await store.persist_signal(
        _signal(agent_id), agent_id=agent_id,
        source_event_id="provider-event-1", retention_days=7
    )).created

    first, second = await asyncio.gather(
        store.claim_delivery(
            agent_id=agent_id, consumer_id="workflow-wait", executor_id="worker-a"
        ),
        store.claim_delivery(
            agent_id=agent_id, consumer_id="workflow-wait", executor_id="worker-b"
        ),
    )
    claims = [delivery for delivery in (first, second) if delivery is not None]
    assert len(claims) == 1
    delivery = claims[0]
    assert delivery.event.payload["message"] == "normalized"
    assert await store.ack_delivery(
        agent_id=agent_id,
        consumer_id="workflow-wait",
        delivery_id=delivery.delivery_id,
        lease_token=delivery.lease_token,
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_durable_retry_and_lease_expiry_transitions_work_on_both_backends(
    db_backend,
):
    """Timestamp-bearing failure transitions retain PostgreSQL type parity."""
    store = DurableSignalStore(db_backend)
    await store.initialize()
    agent_id = f"did:test:durable-retry:{uuid4()}"
    consumer_id = "workflow-wait"
    await store.register_consumer(
        DurableConsumerRegistration(
            consumer_id=consumer_id,
            source="provider.message",
            agent_id=agent_id,
            max_attempts=2,
            lease_seconds=1,
        )
    )
    await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"retry-{uuid4()}",
        retention_days=7,
    )
    now = datetime.now(timezone.utc)
    first = await store.claim_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        executor_id="worker-a",
        now=now,
    )
    assert first is not None
    retried = await store.nack_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        delivery_id=first.delivery_id,
        lease_token=first.lease_token,
        error="transient dependency outage",
        retry_delay=timedelta(seconds=1),
        now=now,
    )
    assert retried is not None
    assert retried.status == "retry"

    second = await store.claim_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        executor_id="worker-b",
        now=now + timedelta(seconds=1),
    )
    assert second is not None
    assert second.attempts == 2
    assert await store.claim_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        executor_id="worker-c",
        now=now + timedelta(seconds=2),
    ) is None
    terminal = await store.get_delivery(
        agent_id=agent_id,
        consumer_id=consumer_id,
        delivery_id=second.delivery_id,
    )
    assert terminal is not None
    assert terminal.status == "failed"
    assert terminal.last_error == "lease expired before acknowledgement"


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_durable_consumer_deactivation_has_backend_parity(db_backend):
    """The store retains evidence while eliminating future claimable work."""
    store = DurableSignalStore(db_backend)
    await store.initialize()
    agent_id = f"did:test:durable-deactivate:{uuid4()}"
    consumer_id = "workflow-wait"
    await store.register_consumer(
        DurableConsumerRegistration(
            consumer_id=consumer_id,
            source="provider.message",
            agent_id=agent_id,
        )
    )
    before = await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"before-deactivate:{uuid4()}",
        retention_days=7,
    )
    assert before.delivery_ids
    assert await store.deactivate_consumer(
        agent_id=agent_id, consumer_id=consumer_id
    )
    assert await store.deactivate_consumer(
        agent_id=agent_id, consumer_id=consumer_id
    )
    assert not await store.deactivate_consumer(
        agent_id=f"{agent_id}:other", consumer_id=consumer_id
    )

    retained = await store.get_delivery_for_event(
        agent_id=agent_id, consumer_id=consumer_id, event_id=before.event_id
    )
    assert retained is not None
    assert retained.status == "failed"
    assert retained.last_error == "durable consumer deactivated"
    assert retained.terminal_at is not None
    assert await store.claim_delivery(
        agent_id=agent_id, consumer_id=consumer_id, executor_id="late-worker"
    ) is None

    after = await store.persist_signal(
        _signal(agent_id),
        agent_id=agent_id,
        source_event_id=f"after-deactivate:{uuid4()}",
        retention_days=7,
    )
    assert after.created
    assert after.delivery_ids == ()
    assert await store.get_delivery_for_event(
        agent_id=agent_id, consumer_id=consumer_id, event_id=after.event_id
    ) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "first", ("persistence", "deactivation"), ids=("event-first", "deactivate-first")
)
async def test_deactivation_and_event_persistence_share_a_serialized_handoff(
    db_backend, first, monkeypatch
):
    """The handoff order decides whether evidence is terminalized or absent."""
    peer_backend = await _independent_backend(db_backend)
    try:
        persistence_backend, deactivation_backend = (
            (db_backend, peer_backend)
            if first == "persistence"
            else (peer_backend, db_backend)
        )
        persistence_store = DurableSignalStore(persistence_backend)
        deactivation_store = DurableSignalStore(deactivation_backend)
        await persistence_store.initialize()
        await deactivation_store.initialize()
        agent_id = f"did:test:durable-deactivate-handoff:{uuid4()}"
        consumer = DurableConsumerRegistration(
            consumer_id="workflow-wait",
            source="provider.message",
            agent_id=agent_id,
        )
        await persistence_store.register_consumer(consumer)
        event = _signal(agent_id)
        handoff_owned = asyncio.Event()
        release_handoff = asyncio.Event()

        if first == "persistence":
            original_fetch_all = persistence_backend.fetch_all

            async def pause_persistence_consumer_lookup(query, params=()):
                rows = await original_fetch_all(query, params)
                if (
                    f"FROM {DurableSignalStore.CONSUMERS}" in query
                    and "max_attempts" in query
                ):
                    handoff_owned.set()
                    await release_handoff.wait()
                return rows

            monkeypatch.setattr(
                persistence_backend, "fetch_all", pause_persistence_consumer_lookup
            )
            first_task = asyncio.create_task(
                persistence_store.persist_signal(
                    event,
                    agent_id=agent_id,
                    source_event_id=f"handoff:{uuid4()}",
                    retention_days=7,
                )
            )
        else:
            original_handoff = deactivation_store._lock_scope_handoff

            async def pause_deactivation_after_handoff(**kwargs):
                await original_handoff(**kwargs)
                handoff_owned.set()
                await release_handoff.wait()

            monkeypatch.setattr(
                deactivation_store,
                "_lock_scope_handoff",
                pause_deactivation_after_handoff,
            )
            first_task = asyncio.create_task(
                deactivation_store.deactivate_consumer(
                    agent_id=agent_id, consumer_id=consumer.consumer_id
                )
            )

        await asyncio.wait_for(handoff_owned.wait(), timeout=5)
        if first == "persistence":
            second_task = asyncio.create_task(
                deactivation_store.deactivate_consumer(
                    agent_id=agent_id, consumer_id=consumer.consumer_id
                )
            )
        else:
            second_task = asyncio.create_task(
                persistence_store.persist_signal(
                    event,
                    agent_id=agent_id,
                    source_event_id=f"handoff:{uuid4()}",
                    retention_days=7,
                )
            )
        await asyncio.sleep(0)
        assert not second_task.done()
        release_handoff.set()
        first_result, second_result = await asyncio.wait_for(
            asyncio.gather(first_task, second_task), timeout=5
        )

        if first == "persistence":
            assert first_result.created
            assert second_result is True
            delivery = await persistence_store.get_delivery_for_event(
                agent_id=agent_id,
                consumer_id=consumer.consumer_id,
                event_id=event.id,
            )
            assert delivery is not None and delivery.status == "failed"
        else:
            assert first_result is True
            assert second_result.created
            assert second_result.delivery_ids == ()
            assert await persistence_store.get_delivery_for_event(
                agent_id=agent_id,
                consumer_id=consumer.consumer_id,
                event_id=event.id,
            ) is None
    finally:
        release_handoff.set()
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_initial_volatile_reservation_blocks_a_peer_on_both_backends(db_backend):
    """The initial live owner is established before a marker event is visible."""
    peer_backend = await _independent_backend(db_backend)
    try:
        emitting_store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-initial-lease:{uuid4()}"
        consumer_id = "workflow-wait"
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
                correlation_selector="payload.workflow=wf-1",
                lease_seconds=10,
            )
        )
        secret = "initial-lease-customer@example.com"
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        installed_before_commit = []
        persisted = await emitting_store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"initial-lease-{uuid4()}",
            retention_days=7,
            transient_selector_payload={"workflow": "wf-1", "message": secret},
            initial_lease_owner="emitting-dispatcher",
            before_commit=installed_before_commit.append,
        )
        assert installed_before_commit == [persisted]
        assert len(persisted.initial_reservations) == 1
        reservation = persisted.initial_reservations[0]

        # A separate store sees the committed row but not a claimable
        # delivery. It must wait for the owning dispatcher to transfer or for
        # lease expiry recovery, never receive the transient selector payload.
        assert await peer_store.claim_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            executor_id="peer-worker",
        ) is None
        activated = await emitting_store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.reservation_token,
        )
        assert activated is not None
        owned = await emitting_store.claim_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.reservation_token,
            executor_id="owner-worker",
        )
        assert owned is not None
        assert owned.event.payload == {"_privacy_gated": "none"}
        row = await db_backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE event_id = ?",
            (marker_event.id,),
        )
        assert row is not None
        assert secret not in str(row[0])
    finally:
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_unactivated_reservation_survives_a_long_paused_commit_and_owner_activation_wins(
    db_backend,
):
    """No delivery lease exists until the post-commit owner activation."""
    peer_backend = await _independent_backend(db_backend)
    release_commit = asyncio.Event()
    persistence_task = None
    original_transaction = db_backend.transaction
    try:
        emitting_store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-paused-commit:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
                lease_seconds=1,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id
        )
        sidecar_installed = asyncio.Event()
        commit_paused = asyncio.Event()

        @asynccontextmanager
        async def pause_after_precommit_callback():
            async with original_transaction():
                yield
                commit_paused.set()
                await release_commit.wait()

        db_backend.transaction = pause_after_precommit_callback
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persistence_task = asyncio.create_task(
            emitting_store.persist_signal(
                marker_event,
                agent_id=agent_id,
                source_event_id=f"paused-commit-{uuid4()}",
                retention_days=7,
                transient_selector_payload={
                    "workflow": "wf-1",
                    "message": "commit-paused-customer@example.com",
                },
                initial_lease_owner=owner_id,
                before_commit=lambda _persisted: sidecar_installed.set(),
            )
        )
        await asyncio.wait_for(sidecar_installed.wait(), timeout=2)
        await asyncio.wait_for(commit_paused.wait(), timeout=2)
        # The event transaction is still invisible. Sleeping beyond the
        # consumer's one-second lease cannot expire a reservation because it
        # does not have a lease deadline yet.
        await asyncio.sleep(1.1)
        release_commit.set()
        persisted = await persistence_task
        reservation = persisted.initial_reservations[0]
        unactivated = await emitting_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert unactivated is not None
        assert unactivated.status == "initial_reserved"
        assert unactivated.lease_expires_at is None

        peer_claim, activated = await asyncio.gather(
            peer_store.claim_delivery(
                agent_id=agent_id,
                consumer_id=consumer_id,
                executor_id="peer-after-visibility",
            ),
            emitting_store.activate_initial_delivery(
                agent_id=agent_id,
                consumer_id=consumer_id,
                delivery_id=reservation.delivery_id,
                initial_lease_owner=owner_id,
                initial_lease_token=reservation.reservation_token,
            ),
        )
        assert peer_claim is None
        assert activated is not None
        assert activated.lease_expires_at > datetime.now(timezone.utc)
    finally:
        release_commit.set()
        db_backend.transaction = original_transaction
        if persistence_task is not None:
            await asyncio.gather(persistence_task, return_exceptions=True)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_startup_recovers_only_a_stale_unactivated_owner_as_marker_work(
    db_backend,
):
    """A crash before activation replays only the durable privacy marker."""
    peer_backend = await _independent_backend(db_backend)
    recovering_agent = None
    recovering_dispatcher = None
    try:
        emitting_store = DurableSignalStore(db_backend)
        await emitting_store.initialize()
        agent_id = f"did:test:durable-stale-owner:{uuid4()}"
        consumer_id = "workflow-wait"
        # Recovery deliberately considers only managed dispatcher owners;
        # arbitrary executor namespaces must never be stolen as crashed hosts.
        owner_id = _dispatcher_owner_id()
        now = datetime.now(timezone.utc)
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id,
            owner_id=owner_id,
            now=now - timedelta(minutes=10),
        )
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        secret = "crash-before-activation@example.com"
        persisted = await emitting_store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"stale-owner-{uuid4()}",
            retention_days=7,
            transient_selector_payload={"workflow": "wf-1", "message": secret},
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]
        # Startup initializes a distinct runtime owner and invokes the
        # owner-aware recovery path. The old owner heartbeat is ten minutes
        # stale, so the marker becomes ordinary retry work before any generic
        # worker can claim it.
        recovering_agent, recovering_dispatcher = await _ephemeral_dispatcher(
            peer_backend, agent_id=agent_id
        )
        recovered = await recovering_dispatcher._durable_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert recovered is not None
        assert recovered.status == "retry"
        replay = await recovering_dispatcher.claim_durable_delivery(
            consumer_id=consumer_id,
            executor_id="restart-worker",
        )
        assert replay is not None
        assert replay.event.payload == {"_privacy_gated": "none"}
        assert secret not in str(replay.event.payload)
    finally:
        if recovering_agent is not None and recovering_dispatcher is not None:
            await _finish_dispatcher(recovering_agent, recovering_dispatcher)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_live_runtime_owner_cannot_be_recovered_by_a_concurrent_dispatcher(
    db_backend,
):
    """Owner-aware recovery never steals a contemporaneously live emitter."""
    peer_backend = await _independent_backend(db_backend)
    try:
        emitting_store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-live-owner:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        peer_owner_id = _dispatcher_owner_id()
        now = datetime.now(timezone.utc)
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=now
        )
        await peer_store.register_runtime_owner(
            agent_id=agent_id, owner_id=peer_owner_id, now=now
        )
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persisted = await emitting_store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"live-owner-{uuid4()}",
            retention_days=7,
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]
        assert await peer_store.recover_abandoned_initial_reservations(
            agent_id=agent_id,
            recovering_owner_id=peer_owner_id,
            stale_before=now - timedelta(seconds=1),
            now=now,
        ) == 0
        assert await peer_store.claim_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            executor_id="peer-worker",
        ) is None
        assert await emitting_store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner=owner_id,
            initial_lease_token=reservation.reservation_token,
            now=now,
        ) is not None
    finally:
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_delayed_runtime_heartbeat_cannot_regress_owner_liveness_or_release_work(
    db_backend,
):
    """A heartbeat sampled before a wait must not undo newer liveness.

    Explicit times model a heartbeat which sampled ``older`` before blocking,
    while a newer owner touch completed first.  It then completes late on the
    same owner row.  Recovery from a concurrent dispatcher must still see the
    newer heartbeat and leave the initial reservation untouched.
    """
    peer_backend = await _independent_backend(db_backend)
    try:
        emitting_store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-monotonic-heartbeat:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        recovering_owner_id = _dispatcher_owner_id()
        base = datetime(2040, 1, 1, tzinfo=timezone.utc)
        older = base + timedelta(seconds=1)
        newer = base + timedelta(seconds=10)
        recovery_now = base + timedelta(seconds=11)
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=base
        )
        await peer_store.register_runtime_owner(
            agent_id=agent_id, owner_id=recovering_owner_id, now=newer
        )
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persisted = await emitting_store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"monotonic-heartbeat-{uuid4()}",
            retention_days=7,
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]

        # A newer activation/heartbeat touch wins; the delayed heartbeat that
        # was sampled earlier must not overwrite it at commit time.
        await emitting_store.heartbeat_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=newer
        )
        await emitting_store.heartbeat_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=older
        )

        assert await peer_store.recover_abandoned_initial_reservations(
            agent_id=agent_id,
            recovering_owner_id=recovering_owner_id,
            stale_before=base + timedelta(seconds=5),
            now=recovery_now,
        ) == 0
        remaining = await peer_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert remaining is not None
        assert remaining.status == "initial_reserved"
        assert await peer_store.claim_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            executor_id="peer-worker",
        ) is None
    finally:
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_postgres_recovery_waits_for_overlapping_live_runtime_heartbeat(
    db_backend,
):
    """A PostgreSQL recovery cannot classify a heartbeat mid-transaction stale."""

    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL advisory-lock overlap regression")

    peer_backend = await _independent_backend(db_backend)
    heartbeat_task = None
    recovery_task = None
    try:
        emitting_store = DurableSignalStore(db_backend)
        recovering_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await recovering_store.initialize()
        agent_id = f"did:test:durable-heartbeat-overlap:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        recovering_owner_id = _dispatcher_owner_id()
        base = datetime.now(timezone.utc)
        stale = base - timedelta(minutes=5)
        now = base + timedelta(seconds=1)
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
            )
        )
        await emitting_store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=stale
        )
        await recovering_store.register_runtime_owner(
            agent_id=agent_id, owner_id=recovering_owner_id, now=now
        )
        marker = _signal(agent_id)
        persisted = await emitting_store.persist_signal(
            marker,
            agent_id=agent_id,
            source_event_id=f"heartbeat-overlap:{uuid4()}",
            retention_days=7,
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]
        activated = await emitting_store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner=owner_id,
            initial_lease_token=reservation.reservation_token,
            now=stale,
        )
        assert activated is not None and activated.status == "leased"

        heartbeat_updated = asyncio.Event()
        release_heartbeat = asyncio.Event()
        original_touch = emitting_store._touch_runtime_owner_locked

        async def pause_after_owner_touch(**kwargs):
            await original_touch(**kwargs)
            heartbeat_updated.set()
            await release_heartbeat.wait()

        emitting_store._touch_runtime_owner_locked = pause_after_owner_touch
        heartbeat_task = asyncio.create_task(
            emitting_store.heartbeat_runtime_owner(
                agent_id=agent_id, owner_id=owner_id, now=now
            )
        )
        await asyncio.wait_for(heartbeat_updated.wait(), timeout=2)
        recovery_task = asyncio.create_task(
            recovering_store.recover_abandoned_leases(
                agent_id=agent_id,
                recovering_owner_id=recovering_owner_id,
                stale_before=base,
                now=now,
            )
        )
        await asyncio.sleep(0.05)
        assert recovery_task.done() is False

        release_heartbeat.set()
        await heartbeat_task
        assert await recovery_task == 0
        delivery = await recovering_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert delivery is not None and delivery.status == "leased"
    finally:
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
        if recovery_task is not None and not recovery_task.done():
            recovery_task.cancel()
        await asyncio.gather(
            *(task for task in (heartbeat_task, recovery_task) if task is not None),
            return_exceptions=True,
        )
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_same_dispatcher_initial_claim_waits_for_postgres_commit_boundary(
    db_backend,
):
    """A local claimant must not discard a sidecar before its row commits.

    PostgreSQL claims run in a separate transaction.  Pause the emitting
    transaction after the synchronous sidecar callback, start the same
    dispatcher's durable claim, then release commit.  Whether PostgreSQL
    blocks that claim on the source handoff lock or its initial-handoff lock,
    it must transfer the now-visible reservation and receive the raw payload.
    """
    if db_backend.backend_type != "postgres":
        pytest.skip("PostgreSQL-specific separate-transaction visibility race")

    agent_id = f"did:test:durable-local-commit-race:{uuid4()}"
    consumer_id = "workflow-wait"
    agent, dispatcher = await _ephemeral_dispatcher(db_backend, agent_id=agent_id)
    original_transaction = db_backend.transaction
    original_fetch_val = db_backend.fetch_val
    commit_boundary = asyncio.Event()
    claim_scope_waiting = asyncio.Event()
    release_commit = asyncio.Event()
    dispatch_task = None
    claim_task = None

    @asynccontextmanager
    async def pause_after_before_commit():
        async with original_transaction():
            yield
            # The raw sidecar is installed, but this PostgreSQL transaction
            # has not committed its event/delivery rows yet.
            commit_boundary.set()
            await release_commit.wait()

    try:
        await asyncio.wait_for(
            dispatcher.register_durable_consumer(
                DurableConsumerRegistration(
                    consumer_id=consumer_id,
                    source="provider.message",
                    agent_id=agent_id,
                    correlation_selector="payload.workflow=wf-1",
                    lease_seconds=10,
                )
            ),
            timeout=5,
        )
        # Register before installing the barrier.  The backend seam wraps every
        # transaction, and pausing registration here would block the test
        # before it starts the emitting transaction it is meant to observe.
        db_backend.transaction = pause_after_before_commit
        secret = "same-dispatcher-commit-boundary@example.com"
        event = _signal(agent_id)
        event.payload["message"] = secret
        dispatch_task = asyncio.create_task(
            dispatcher.dispatch_signal(
                event,
                source_event_id=f"same-dispatcher-commit-race:{uuid4()}",
            )
        )
        await asyncio.wait_for(commit_boundary.wait(), timeout=5)
        assert len(dispatcher._transient_durable_handoffs) == 1

        async def note_claim_scope_wait(query, params=()):
            if _is_scope_handoff_lock(query):
                claim_scope_waiting.set()
            return await original_fetch_val(query, params)

        # Install only after persistence reaches its pre-commit barrier; its
        # own earlier acquisition of this same lock is not the observation.
        db_backend.fetch_val = note_claim_scope_wait

        claim_task = asyncio.create_task(
            dispatcher.claim_durable_delivery(
                consumer_id=consumer_id,
                executor_id="local-worker",
            )
        )
        await asyncio.wait_for(claim_scope_waiting.wait(), timeout=5)
        # The source-level database handoff lock may serialize the ordinary
        # poll before it can observe the uncommitted row. If it does not, the
        # local handoff lock still protects the sidecar-to-reservation transfer.
        assert not claim_task.done()

        release_commit.set()
        assert (
            await asyncio.wait_for(asyncio.shield(dispatch_task), timeout=5)
        ).status is Status.OK
        delivery = await asyncio.wait_for(asyncio.shield(claim_task), timeout=5)
        assert delivery is not None
        assert delivery.event.payload == {"workflow": "wf-1", "message": secret}
        row = await db_backend.fetch_one(
            "SELECT payload FROM durable_signal_events WHERE event_id = ?",
            (event.id,),
        )
        assert row is not None
        assert secret not in str(row[0])
    finally:
        release_commit.set()
        db_backend.transaction = original_transaction
        db_backend.fetch_val = original_fetch_val
        try:
            await _cancel_and_drain(claim_task, dispatch_task)
        finally:
            await _finish_dispatcher(agent, dispatcher)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_initial_reservation_lease_starts_only_after_post_commit_activation_on_both_backends(
    db_backend,
):
    """A delayed persistence publishes no lease until the owner activates it."""
    peer_backend = await _independent_backend(db_backend)
    release_handoff = asyncio.Event()
    holder = None
    persistence_task = None
    try:
        emitting_store = DurableSignalStore(db_backend)
        blocking_store = DurableSignalStore(peer_backend)
        await emitting_store.initialize()
        await blocking_store.initialize()
        agent_id = f"did:test:durable-initial-contention:{uuid4()}"
        consumer_id = "workflow-wait"
        await emitting_store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
                lease_seconds=1,
            )
        )
        handoff_acquired = asyncio.Event()

        async def hold_scope_handoff() -> None:
            async with peer_backend.transaction():
                await blocking_store._lock_scope_handoff(
                    agent_id=agent_id, source="provider.message"
                )
                handoff_acquired.set()
                await release_handoff.wait()

        holder = asyncio.create_task(hold_scope_handoff())
        await asyncio.wait_for(handoff_acquired.wait(), timeout=2)
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persistence_task = asyncio.create_task(
            emitting_store.persist_signal(
                marker_event,
                agent_id=agent_id,
                source_event_id=f"initial-contention-{uuid4()}",
                retention_days=7,
                initial_lease_owner="emitting-dispatcher",
            )
        )
        # The old implementation could spend a full lease duration before it
        # even acquired this handoff lock. A reservation must still carry no
        # live deadline when its transaction later commits.
        await asyncio.sleep(1.1)
        release_handoff.set()
        await holder
        persisted = await persistence_task
        assert len(persisted.initial_reservations) == 1
        reservation = persisted.initial_reservations[0]
        unactivated = await emitting_store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert unactivated is not None
        assert unactivated.status == "initial_reserved"
        assert unactivated.lease_expires_at is None
        assert await blocking_store.claim_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            executor_id="peer-before-activation",
        ) is None
        activated = await emitting_store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.reservation_token,
        )
        assert activated is not None
        assert activated.lease_expires_at > datetime.now(timezone.utc)
        assert await emitting_store.claim_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner="emitting-dispatcher",
            initial_lease_token=reservation.reservation_token,
            executor_id="owner-worker",
        ) is not None
    finally:
        release_handoff.set()
        if holder is not None:
            await asyncio.gather(holder, return_exceptions=True)
        if persistence_task is not None:
            await asyncio.gather(persistence_task, return_exceptions=True)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_initial_worker_transfer_does_not_publish_an_expired_lease_after_contention(
    db_backend,
):
    """The initial owner lease is rechecked after the real write/row lock.

    A deterministic clock advances while another connection holds the exact
    serialization point.  The old transfer sampled before waiting and returned
    a worker lease already expired at the advanced clock; the fixed path sees
    the initial owner lease expired and declines the handoff.
    """
    peer_backend = await _independent_backend(db_backend)
    release_lock = asyncio.Event()
    lock_acquired = asyncio.Event()
    transfer_waiting = asyncio.Event()
    holder = None
    try:
        store = DurableSignalStore(db_backend)
        peer_store = DurableSignalStore(peer_backend)
        await store.initialize()
        await peer_store.initialize()
        agent_id = f"did:test:durable-initial-transfer:{uuid4()}"
        consumer_id = "workflow-wait"
        owner_id = _dispatcher_owner_id()
        base = datetime(2040, 1, 1, tzinfo=timezone.utc)
        current_time = {"value": base}
        store.now_utc = lambda: current_time["value"]
        await store.register_consumer(
            DurableConsumerRegistration(
                consumer_id=consumer_id,
                source="provider.message",
                agent_id=agent_id,
                lease_seconds=1,
            )
        )
        await store.register_runtime_owner(
            agent_id=agent_id, owner_id=owner_id, now=base
        )
        marker_event = _signal(agent_id)
        marker_event.payload = {"_privacy_gated": "none"}
        persisted = await store.persist_signal(
            marker_event,
            agent_id=agent_id,
            source_event_id=f"initial-transfer-contention:{uuid4()}",
            retention_days=7,
            initial_lease_owner=owner_id,
        )
        reservation = persisted.initial_reservations[0]
        assert await store.activate_initial_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
            initial_lease_owner=owner_id,
            initial_lease_token=reservation.reservation_token,
            now=base,
        ) is not None

        async def hold_transfer_serialization_point() -> None:
            async with peer_backend.transaction():
                if peer_backend.backend_type == "postgres":
                    await peer_backend.fetch_one(
                        "SELECT delivery_id FROM durable_signal_deliveries "
                        "WHERE delivery_id = ? FOR UPDATE",
                        (reservation.delivery_id,),
                    )
                else:
                    await peer_backend.execute(
                        "UPDATE durable_signal_deliveries "
                        "SET updated_at = updated_at WHERE delivery_id = ?",
                        (reservation.delivery_id,),
                    )
                lock_acquired.set()
                await release_lock.wait()

        holder = asyncio.create_task(hold_transfer_serialization_point())
        await asyncio.wait_for(lock_acquired.wait(), timeout=5)

        if db_backend.backend_type == "postgres":
            original_fetch_one = db_backend.fetch_one

            async def observe_transfer_lock(query, params=()):
                if "FOR UPDATE" in query and "durable_signal_deliveries" in query:
                    transfer_waiting.set()
                return await original_fetch_one(query, params)

            db_backend.fetch_one = observe_transfer_lock
        else:
            original_execute = db_backend.execute

            async def observe_transfer_lock(query, params=()):
                if (
                    "UPDATE durable_signal_deliveries" in query
                    and "SET updated_at = updated_at" in query
                ):
                    transfer_waiting.set()
                return await original_execute(query, params)

            db_backend.execute = observe_transfer_lock

        try:
            transfer_task = asyncio.create_task(
                store.claim_initial_delivery(
                    agent_id=agent_id,
                    consumer_id=consumer_id,
                    delivery_id=reservation.delivery_id,
                    initial_lease_owner=owner_id,
                    initial_lease_token=reservation.reservation_token,
                    executor_id="owner-worker",
                )
            )
            await asyncio.wait_for(transfer_waiting.wait(), timeout=5)
            current_time["value"] = base + timedelta(seconds=2)
            release_lock.set()
            assert await transfer_task is None
        finally:
            if db_backend.backend_type == "postgres":
                db_backend.fetch_one = original_fetch_one
            else:
                db_backend.execute = original_execute

        row = await store.get_delivery(
            agent_id=agent_id,
            consumer_id=consumer_id,
            delivery_id=reservation.delivery_id,
        )
        assert row is not None
        assert row.status == "leased"
        assert row.lease_owner == owner_id
        assert row.attempts == 0
    finally:
        release_lock.set()
        if holder is not None:
            await asyncio.gather(holder, return_exceptions=True)
        await peer_backend.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize(
    "first",
    ("registration", "persistence"),
    ids=("registration-first", "persistence-first"),
)
async def test_durable_registration_persistence_handoff_is_atomic_across_instances(
    db_backend, first
):
    """Exercise both handoff orders through the public durable-store API.

    The gates pause a real transaction after it has acquired its handoff
    primitive.  They prove the competing backend instance cannot take its
    first stale read until the owner commits, which is the SQLite failure mode
    that a same-instance or private-helper test cannot expose.
    """
    peer_backend = await _independent_backend(db_backend)
    try:
        registration_backend, persistence_backend = (
            (db_backend, peer_backend)
            if first == "registration"
            else (peer_backend, db_backend)
        )
        registration_store = DurableSignalStore(registration_backend)
        persistence_store = DurableSignalStore(persistence_backend)
        await registration_store.initialize()
        await persistence_store.initialize()

        agent_id = f"did:test:durable-handoff:{uuid4()}"
        registration = DurableConsumerRegistration(
            consumer_id="workflow-wait",
            source="provider.message",
            agent_id=agent_id,
            correlation_selector="payload.workflow=wf-1",
        )
        event = _signal(agent_id)

        if first == "registration":
            registration_read = asyncio.Event()
            allow_registration_write = asyncio.Event()
            original_fetch_one = registration_backend.fetch_one

            async def pause_after_registration_read(query, params=()):
                row = await original_fetch_one(query, params)
                if (
                    f"FROM {DurableSignalStore.CONSUMERS}" in query
                    and "consumer_id" in query
                ):
                    registration_read.set()
                    await allow_registration_write.wait()
                return row

            registration_backend.fetch_one = pause_after_registration_read
            registration_task = asyncio.create_task(
                registration_store.register_consumer(registration)
            )
            await asyncio.wait_for(registration_read.wait(), timeout=5)

            persistence_started = asyncio.Event()
            original_handoff = (
                persistence_backend.fetch_val
                if persistence_backend.backend_type == "postgres"
                else persistence_backend.execute
            )

            async def note_persistence_handoff(query, params=()):
                if _is_scope_handoff_lock(query):
                    persistence_started.set()
                return await original_handoff(query, params)

            if persistence_backend.backend_type == "postgres":
                persistence_backend.fetch_val = note_persistence_handoff
            else:
                persistence_backend.execute = note_persistence_handoff
            persistence_task = asyncio.create_task(
                persistence_store.persist_signal(
                    event,
                    agent_id=agent_id,
                    source_event_id=f"handoff-{uuid4()}",
                    retention_days=7,
                )
            )
            await asyncio.wait_for(persistence_started.wait(), timeout=5)
            assert not persistence_task.done()
            allow_registration_write.set()
            _, persisted = await asyncio.wait_for(
                asyncio.gather(registration_task, persistence_task), timeout=5
            )
        else:
            persistence_lookup = asyncio.Event()
            allow_persistence_lookup = asyncio.Event()
            original_fetch_all = persistence_backend.fetch_all

            async def pause_before_persistence_consumer_lookup(query, params=()):
                if (
                    f"FROM {DurableSignalStore.CONSUMERS}" in query
                    and "max_attempts" in query
                ):
                    persistence_lookup.set()
                    await allow_persistence_lookup.wait()
                return await original_fetch_all(query, params)

            persistence_backend.fetch_all = pause_before_persistence_consumer_lookup
            persistence_task = asyncio.create_task(
                persistence_store.persist_signal(
                    event,
                    agent_id=agent_id,
                    source_event_id=f"handoff-{uuid4()}",
                    retention_days=7,
                )
            )
            await asyncio.wait_for(persistence_lookup.wait(), timeout=5)

            registration_started = asyncio.Event()
            original_handoff = (
                registration_backend.fetch_val
                if registration_backend.backend_type == "postgres"
                else registration_backend.execute
            )

            async def note_registration_handoff(query, params=()):
                if _is_scope_handoff_lock(query):
                    registration_started.set()
                return await original_handoff(query, params)

            if registration_backend.backend_type == "postgres":
                registration_backend.fetch_val = note_registration_handoff
            else:
                registration_backend.execute = note_registration_handoff
            registration_task = asyncio.create_task(
                registration_store.register_consumer(registration)
            )
            await asyncio.wait_for(registration_started.wait(), timeout=5)
            assert not registration_task.done()
            allow_persistence_lookup.set()
            persisted, _ = await asyncio.wait_for(
                asyncio.gather(persistence_task, registration_task), timeout=5
            )

        assert persisted.created
        await _assert_one_pending_delivery(
            registration_store, agent_id=agent_id, event_id=event.id
        )
    finally:
        await peer_backend.close()
