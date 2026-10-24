"""Old durable A2A envelopes must not survive as native recovery work."""

import pytest
from kestrel_sdk.signals import Signal, SignalMode, Trust, Visibility

from kestrel_sovereign.a2a.legacy_cutover import (
    apply_legacy_migration,
    prepare_legacy_migration,
)
from kestrel_sovereign.a2a.migrate_legacy import LegacyTaskMigrationError
from kestrel_sovereign.signals.durable import (
    PENDING,
    DurableConsumerRegistration,
    DurableSignalStore,
)
from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    DURABLE_COGNITION_CONSUMER_ID,
    SOURCE_NAME,
)
from tests.unit.test_a2a_legacy_cutover import legacy, ownership


def old_signal():
    # This is the normalized envelope emitted by main3eb8, before native
    # input revisions. It is durable signal history, never a wire adapter.
    return Signal(
        source=SOURCE_NAME,
        kind="inbound",
        mode=SignalMode.COGNITION,
        payload={
            "task_id": "historical",
            "session_id": "draft-session",
            "sender": "did:creator",
            "skill_id": "",
            "a2a_verb": "task",
            "reply_expected": True,
            "request_content": "Historical input",
        },
        target_agent="did:recipient",
        caller="did:creator",
        origin_trust=Trust.TRUSTED,
        visibility=Visibility.INTERNAL,
        dedupe_key="historical",
    )


def consumer():
    return DurableConsumerRegistration(
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source=SOURCE_NAME,
        agent_id="did:recipient",
        max_attempts=0,
    )


async def test_preview_refuses_pending_old_durable_a2a_delivery(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await legacy(backend)
        await store.register_consumer(consumer())
        accepted = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        assert len(await store.list_deliveries(agent_id="did:recipient")) == 1
        with pytest.raises(LegacyTaskMigrationError, match="durable"):
            await prepare_legacy_migration(backend, ownership=ownership())
        assert (
            await store.get_delivery(
                agent_id="did:recipient",
                consumer_id=DURABLE_COGNITION_CONSUMER_ID,
                delivery_id=accepted.delivery_ids[0],
            )
        ).status == PENDING
        assert not await backend.table_exists("a2a_protocol_schema")
    finally:
        await backend.close()


async def test_cutover_retires_unselected_old_event_before_native_consumer_backfill(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await legacy(backend)
        accepted = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        assert accepted.created and accepted.delivery_ids == ()
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        await apply_legacy_migration(backend, plan)
        await store.register_consumer(consumer())
        deliveries = await store.list_deliveries(
            agent_id="did:recipient", consumer_id=DURABLE_COGNITION_CONSUMER_ID
        )
        assert deliveries == [], (
            "Old A2A input was backfilled into the native cognition consumer after cutover"
        )
    finally:
        await backend.close()


async def acknowledge_old(store, accepted):
    lease = await store.claim_delivery(
        agent_id="did:recipient",
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        executor_id="old-runtime",
    )
    assert lease is not None and lease.event.event_id == accepted.event_id
    assert await store.ack_delivery(
        agent_id="did:recipient",
        consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        delivery_id=lease.delivery_id,
        lease_token=lease.lease_token,
    )


async def test_archives_include_exact_signal_receipts_and_keep_unrelated_sources(
    native_a2a_backend_factory,
):
    import hashlib
    from dataclasses import replace

    from kestrel_sovereign.signals.durable import DurableIngressAttestation

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await legacy(backend)
        await store.register_consumer(consumer())
        accepted = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
            integrity_binding=hashlib.sha256(b"old normalized envelope").hexdigest(),
            ingress_attestation=DurableIngressAttestation(
                policy_epoch="old-runtime-privacy", sanitized_at_ingress=False
            ),
        )
        await acknowledge_old(store, accepted)
        await store.register_consumer(
            DurableConsumerRegistration(
                consumer_id="unrelated-cutover-test",
                source="cron.test_cutover",
                agent_id="did:recipient",
            )
        )
        unrelated = await store.persist_signal(
            replace(old_signal(), source="cron.test_cutover"),
            agent_id="did:recipient",
            source_event_id="unrelated",
            retention_days=14,
        )
        originals = {}
        tables = [
            store.EVENTS,
            store.DELIVERIES,
            store.EVENT_INTEGRITY,
            store.INGRESS_TRUST_RECEIPTS,
        ]
        for name in tables:
            key = "event_id"
            originals[name] = await backend.fetch_all(
                f'SELECT * FROM "{name}" WHERE {key} = ?', (accepted.event_id,)
            )
            assert originals[name], f"Fixture failed to persist {name}"
        counter = await backend.fetch_one(
            f"SELECT current_sequence FROM {store.SOURCE_SEQUENCES} WHERE agent_id = ? AND source = ?",
            ("did:recipient", SOURCE_NAME),
        )
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        review = plan.review()
        assert review["retired_signal_histories"] == [
            {
                "version": 1,
                "agent_id": "did:recipient",
                "source": SOURCE_NAME,
                "sequence": accepted.source_sequence,
            }
        ]
        archives = await apply_legacy_migration(backend, plan)
        for name in tables:
            assert name in archives, f"Cutover omitted the {name} archive"
            assert len(archives[name]) <= 63
            assert (
                await backend.fetch_all(f'SELECT * FROM "{archives[name]}"')
                == originals[name]
            )
        assert await backend.fetch_all(f"SELECT event_id FROM {store.EVENTS}") == [
            (unrelated.event_id,)
        ]
        remaining = await store.list_deliveries(agent_id="did:recipient")
        assert (
            len(remaining) == 1
            and remaining[0].event.event_id == unrelated.event_id
            and remaining[0].status == PENDING
        )
        assert (
            await backend.fetch_one(
                f"SELECT current_sequence FROM {store.SOURCE_SEQUENCES} WHERE agent_id = ? AND source = ?",
                ("did:recipient", SOURCE_NAME),
            )
            == counter
        )
        next_event = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="native-next",
            retention_days=14,
        )
        assert next_event.source_sequence > accepted.source_sequence
    finally:
        await backend.close()


async def test_changed_durable_history_invalidates_preview_before_conversion(
    native_a2a_backend_factory,
):
    from kestrel_sovereign.storage.db.interface import TransactionError

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await legacy(backend)
        await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="later-old-event",
            retention_days=14,
        )
        with pytest.raises(
            (LegacyTaskMigrationError, TransactionError), match="changed after preview"
        ):
            await apply_legacy_migration(backend, plan)
        assert not await backend.table_exists("a2a_protocol_schema")
        assert await backend.fetch_one(f"SELECT COUNT(*) FROM {store.EVENTS}") == (2,)
    finally:
        await backend.close()


async def test_failure_after_signal_retirement_rolls_back_originals_and_archives(
    native_a2a_backend_factory, monkeypatch
):
    from kestrel_sovereign.storage.db.interface import TransactionError

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await legacy(backend)
        accepted = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        original = await backend.fetch_all(f"SELECT * FROM {store.EVENTS}")
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        retire = DurableSignalStore.retire_source_history
        deleted = []

        async def fail_after_delete(self, boundary):
            count = await retire(self, boundary)
            deleted.append(count)
            raise RuntimeError("Injected offline retirement failure")

        monkeypatch.setattr(
            DurableSignalStore, "retire_source_history", fail_after_delete
        )
        with pytest.raises(
            (RuntimeError, TransactionError),
            match="Injected offline retirement failure",
        ):
            await apply_legacy_migration(backend, plan)
        assert deleted == [1]
        assert await backend.fetch_all(f"SELECT * FROM {store.EVENTS}") == original
        assert not await backend.table_exists("a2a_protocol_schema")
        assert not await backend.table_exists(
            f"a2a_legacy_{plan.migration_id}_signal_events"
        )
        assert (await backend.fetch_one(f"SELECT event_id FROM {store.EVENTS}"))[
            0
        ] == accepted.event_id
    finally:
        await backend.close()


async def test_generic_retirement_preserves_newer_events_and_other_agents(
    native_a2a_backend_factory,
):
    from dataclasses import replace

    from kestrel_sovereign.signals.durable import DurableSourceBoundary

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await store.register_consumer(consumer())
        old = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        await acknowledge_old(store, old)
        boundary = DurableSourceBoundary(
            "did:recipient", SOURCE_NAME, old.source_sequence
        )
        newer = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="newer",
            retention_days=14,
        )
        await store.register_consumer(
            DurableConsumerRegistration(
                consumer_id="unrelated-retirement-test",
                source="cron.test_cutover",
                agent_id="did:recipient",
            )
        )
        unrelated = await store.persist_signal(
            replace(old_signal(), source="cron.test_cutover"),
            agent_id="did:recipient",
            source_event_id="unrelated",
            retention_days=14,
        )
        await store.register_consumer(replace(consumer(), agent_id="did:other"))
        other = await store.persist_signal(
            replace(old_signal(), target_agent="did:other"),
            agent_id="did:other",
            source_event_id="other",
            retention_days=14,
        )
        assert await store.retire_source_history(boundary) == 1
        assert set(await backend.fetch_all(f"SELECT event_id FROM {store.EVENTS}")) == {
            (newer.event_id,),
            (other.event_id,),
            (unrelated.event_id,),
        }
        assert await store.retire_source_history(boundary) == 0
        assert len(await store.list_deliveries(agent_id="did:recipient")) == 2
        latest = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="latest",
            retention_days=14,
        )
        assert latest.source_sequence > newer.source_sequence > old.source_sequence
    finally:
        await backend.close()


async def test_generic_retirement_refuses_owed_deliveries_and_future_boundaries(
    native_a2a_backend_factory,
):
    from kestrel_sovereign.signals.durable import DurableSourceBoundary

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await store.register_consumer(consumer())
        old = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        with pytest.raises(ValueError, match="must drain"):
            await store.retire_source_history(
                DurableSourceBoundary("did:recipient", SOURCE_NAME, old.source_sequence)
            )
        await acknowledge_old(store, old)
        with pytest.raises(ValueError, match="beyond committed"):
            await store.retire_source_history(
                DurableSourceBoundary(
                    "did:recipient", SOURCE_NAME, old.source_sequence + 1
                )
            )
        assert len(await store.list_deliveries(agent_id="did:recipient")) == 1
        assert await backend.fetch_one(f"SELECT event_id FROM {store.EVENTS}") == (
            old.event_id,
        )
    finally:
        await backend.close()


async def test_retirement_validates_counter_custody_before_deleting_history(
    native_a2a_backend_factory,
):
    from kestrel_sovereign.signals.durable import DurableSourceBoundary
    from kestrel_sovereign.storage.db.interface import QueryError, TransactionError

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await store.register_consumer(consumer())
        old = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        await acknowledge_old(store, old)
        original = await backend.fetch_all(f"SELECT * FROM {store.EVENTS}")
        for name in (
            store.SOURCE_SEQUENCES,
            store.SOURCE_SEQUENCE_RECOVERY,
            store.SOURCE_SEQUENCE_HIGH_WATER,
        ):
            await backend.execute(f"DROP TABLE {name}")
        with pytest.raises(
            (QueryError, TransactionError), match="durable_signal_source"
        ):
            await store.retire_source_history(
                DurableSourceBoundary("did:recipient", SOURCE_NAME, old.source_sequence)
            )
        assert await backend.fetch_all(f"SELECT * FROM {store.EVENTS}") == original, (
            "Retirement removed the remaining history after losing counter custody"
        )
    finally:
        await backend.close()


@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
async def test_postgres_retirement_owns_source_handoff_before_counter_access(
    native_a2a_backend_factory, monkeypatch
):
    from kestrel_sovereign.signals.durable import DurableSourceBoundary

    backend, observer = native_a2a_backend_factory(), native_a2a_backend_factory()
    assert backend.backend_type == "postgres"
    await backend.connect()
    await observer.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        old = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        boundary = DurableSourceBoundary(
            "did:recipient", SOURCE_NAME, old.source_sequence
        )
        sequence = store._source_sequence_locked
        observed = []

        async def inspect_mutex(*, agent_id, source, **kwargs):
            obtained = await observer.fetch_val(
                "SELECT pg_try_advisory_xact_lock(hashtextextended(?, 0))",
                (f"durable-signal:{agent_id}:{source}",),
            )
            observed.append(obtained)
            assert obtained is False, (
                "Retirement reached source counters without owning the canonical handoff"
            )
            return await sequence(agent_id=agent_id, source=source, **kwargs)

        monkeypatch.setattr(store, "_source_sequence_locked", inspect_mutex)
        assert await store.retire_source_history(boundary) == 1
        assert observed == [False]
    finally:
        await observer.close()
        await backend.close()


@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
async def test_postgres_cutover_acquires_source_handoff_before_relation_locks(
    native_a2a_backend_factory, monkeypatch
):
    backend = native_a2a_backend_factory()
    assert backend.backend_type == "postgres"
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await legacy(backend)
        await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        handoff = DurableSignalStore._lock_scope_handoff
        visits = []

        async def inspect_phase(self, *, agent_id, source):
            if not visits:
                locks = await self.backend.fetch_all(
                    "SELECT relation.relname FROM pg_locks held JOIN pg_class relation ON relation.oid = held.relation WHERE held.pid = pg_backend_pid() AND held.granted AND held.mode = 'AccessExclusiveLock' AND relation.relname IN (?, ?, ?, ?)",
                    (
                        store.EVENTS,
                        store.DELIVERIES,
                        store.EVENT_INTEGRITY,
                        store.INGRESS_TRUST_RECEIPTS,
                    ),
                )
                assert locks == [], (
                    "Cutover took relation locks before the source handoff and can deadlock with a publisher"
                )
            visits.append((agent_id, source))
            await handoff(self, agent_id=agent_id, source=source)

        monkeypatch.setattr(DurableSignalStore, "_lock_scope_handoff", inspect_phase)
        await apply_legacy_migration(backend, plan)
        assert visits == [("did:recipient", SOURCE_NAME)] * 2
    finally:
        await backend.close()


async def test_preview_does_not_adopt_completed_native_store_metadata(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await legacy(backend)
        await backend.execute_script(
            "CREATE TABLE a2a_native_store_schema (component TEXT PRIMARY KEY, revision INTEGER NOT NULL)"
        )
        await backend.execute("INSERT INTO a2a_native_store_schema VALUES ('core', 1)")
        with pytest.raises(
            LegacyTaskMigrationError, match="Native A2A data already exists"
        ):
            await prepare_legacy_migration(backend, ownership=ownership())
        assert not await backend.table_exists("a2a_protocol_schema")
        assert await backend.fetch_one(
            "SELECT component, revision FROM a2a_native_store_schema"
        ) == ("core", 1)
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "operation", ["source_history_is_drained", "retire_source_history"]
)
async def test_source_history_rejects_untyped_boundary_objects_before_io(
    native_a2a_backend_factory, operation
):
    from types import SimpleNamespace

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        old = await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        claimed = SimpleNamespace(
            agent_id="did:recipient", source=SOURCE_NAME, sequence=old.source_sequence
        )
        with pytest.raises(TypeError, match="durable source boundary"):
            await getattr(store, operation)(claimed)
        assert await backend.fetch_one(f"SELECT event_id FROM {store.EVENTS}") == (
            old.event_id,
        )
    finally:
        await backend.close()


@pytest.mark.parametrize("sequence", [None, 0, "1"])
async def test_preview_refuses_old_signal_history_without_committed_sequences(
    native_a2a_backend_factory, sequence
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await legacy(backend)
        column = "TEXT" if isinstance(sequence, str) else "INTEGER"
        await backend.execute_script(
            f"CREATE TABLE durable_signal_events (event_id TEXT PRIMARY KEY, agent_id TEXT, source TEXT, source_sequence {column}); CREATE TABLE durable_signal_deliveries (event_id TEXT, status TEXT);"
        )
        await backend.execute(
            "INSERT INTO durable_signal_events VALUES (?, ?, ?, ?)",
            ("old-event", "did:recipient", SOURCE_NAME, sequence),
        )
        with pytest.raises(LegacyTaskMigrationError, match="committed source sequence"):
            await prepare_legacy_migration(backend, ownership=ownership())
        assert not await backend.table_exists("a2a_protocol_schema")
        assert await backend.fetch_one(
            "SELECT source_sequence FROM durable_signal_events"
        ) == (sequence,)
    finally:
        await backend.close()


async def test_preview_refuses_incomplete_durable_signal_storage(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await legacy(backend)
        await backend.execute_script(
            "CREATE TABLE durable_signal_events (event_id TEXT PRIMARY KEY, agent_id TEXT, source TEXT, source_sequence INTEGER)"
        )
        await backend.execute(
            "INSERT INTO durable_signal_events VALUES (?, ?, ?, ?)",
            ("old-event", "did:recipient", SOURCE_NAME, 1),
        )
        with pytest.raises(LegacyTaskMigrationError, match="storage is incomplete"):
            await prepare_legacy_migration(backend, ownership=ownership())
        assert not await backend.table_exists("a2a_protocol_schema")
    finally:
        await backend.close()


@pytest.mark.parametrize("kind", ["foreign-source", "untyped"])
async def test_apply_refuses_unreviewed_boundary_types_before_source_handoff(
    native_a2a_backend_factory, monkeypatch, kind
):
    from dataclasses import replace
    from types import SimpleNamespace

    from kestrel_sovereign.storage.db.interface import TransactionError

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        store = DurableSignalStore(backend)
        await store.initialize()
        await legacy(backend)
        await store.persist_signal(
            old_signal(),
            agent_id="did:recipient",
            source_event_id="historical",
            retention_days=14,
        )
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        boundary = plan.signal_boundaries[0]
        if kind == "foreign-source":
            claimed = replace(boundary, source="cron.test_cutover")
        else:
            claimed = SimpleNamespace(
                agent_id=boundary.agent_id,
                source=boundary.source,
                sequence=boundary.sequence,
                to_dict=boundary.to_dict,
            )
        forged = replace(plan, signal_boundaries=(claimed,))
        handoff = DurableSignalStore._lock_scope_handoff
        calls = []

        async def observe(self, *, agent_id, source):
            calls.append((agent_id, source))
            await handoff(self, agent_id=agent_id, source=source)

        monkeypatch.setattr(DurableSignalStore, "_lock_scope_handoff", observe)
        with pytest.raises((LegacyTaskMigrationError, TransactionError)):
            await apply_legacy_migration(backend, forged)
        assert calls == [], (
            "An unreviewed boundary reached source handoff before validation"
        )
        assert not await backend.table_exists("a2a_protocol_schema")
    finally:
        await backend.close()
