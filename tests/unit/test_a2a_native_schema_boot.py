"""A repeated native boot must not reacquire data-table migration locks."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar

import pytest

from kestrel_sovereign.a2a import native_schema
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from tests.utils.native_a2a import manager_for


@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
async def test_postgres_ready_boot_holds_no_data_table_migration_locks(
    native_a2a_backend_factory, monkeypatch
):
    reader, boot_backend = native_a2a_backend_factory(), native_a2a_backend_factory()
    assert reader.backend_type == "postgres"
    await reader.connect()
    await boot_backend.connect()
    first = manager_for(reader, NativeRecipientScope("did:first"))
    boot = manager_for(boot_backend, NativeRecipientScope("did:second"))
    paused, release = asyncio.Event(), asyncio.Event()
    depth = ContextVar("native-schema-test-depth", default=0)
    native_write = native_schema.native_schema_write
    observed = {}

    @asynccontextmanager
    async def pause_before_commit(backend):
        nesting = depth.get()
        token = depth.set(nesting + 1)
        try:
            async with native_write(backend):
                yield
                if nesting == 0:
                    observed["pid"] = (
                        await backend.fetch_one("SELECT pg_backend_pid()")
                    )[0]
                    paused.set()
                    await release.wait()
        finally:
            depth.reset(token)

    task = None
    try:
        await first.initialize()
        monkeypatch.setattr(native_schema, "native_schema_write", pause_before_commit)
        task = asyncio.create_task(boot.initialize())
        async with asyncio.timeout(5):
            await paused.wait()
        locks = await reader.fetch_all(
            "SELECT relation.relname, held.mode FROM pg_locks held "
            "JOIN pg_class relation ON relation.oid = held.relation "
            "JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace "
            "WHERE held.pid = ? AND namespace.nspname = current_schema() "
            "AND held.granted AND held.mode IN ('AccessExclusiveLock', 'ShareLock') "
            "ORDER BY relation.relname, held.mode",
            (observed["pid"],),
        )
        assert locks == [], (
            f"Ready native boot reacquired data migration locks: {locks}"
        )
        assert await reader.fetch_one(
            "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE pid = ? AND locktype = 'advisory' AND granted)",
            (observed["pid"],),
        ) == (True,), "Ready boot lost its atomic schema lease"
    finally:
        release.set()
        if task is not None:
            await task
        await boot.close()
        await first.close()


async def test_protocol_encoding_marker_does_not_prove_the_store_bundle(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:recipient"))
    try:
        await manager.task_store.initialize()
        assert not await backend.table_exists("a2a_native_store_schema")
        assert not await backend.table_exists("a2a_native_questions")
        await manager.initialize()
        assert await backend.table_exists("a2a_native_store_schema"), (
            "The task-only encoding stamp skipped full store initialization"
        )
        assert await backend.fetch_all(
            "SELECT component, revision FROM a2a_native_store_schema"
        ) == [native_schema.NativeStoreSchema.CORE.value]
        for table in (
            "a2a_native_questions",
            "a2a_contexts",
            "a2a_sessions",
            "a2a_llm_calls",
        ):
            assert await backend.table_exists(table)
    finally:
        await manager.close()


async def test_optional_stores_are_completed_only_when_requested(
    native_a2a_backend_factory,
):
    from kestrel_sovereign.a2a.task_manager import TaskManager

    backend = native_a2a_backend_factory()
    await backend.connect()
    scope = NativeRecipientScope("did:recipient")
    core = manager_for(backend, scope, owns_backend=False)
    full = TaskManager.from_backend(backend, scope=scope, owns_backend=False)
    try:
        await core.initialize()
        assert await backend.fetch_all(
            "SELECT component, revision FROM a2a_native_store_schema"
        ) == [native_schema.NativeStoreSchema.CORE.value]
        assert not await backend.table_exists("a2a_memory")
        assert not await backend.table_exists("a2a_feedback")
        await full.initialize()
        assert await backend.fetch_all(
            "SELECT component, revision FROM a2a_native_store_schema ORDER BY component"
        ) == [
            native_schema.NativeStoreSchema.CORE.value,
            ("feedback", 1),
            ("memory", 1),
        ]
        memory_id = await full.memory_service.add_memory(
            "session", "A searchable native archive"
        )
        assert memory_id
        await full.initialize()
        assert (
            await backend.fetch_one(
                "SELECT content FROM a2a_memory WHERE id = ?", (memory_id,)
            )
        )[0] == "A searchable native archive"
    finally:
        await full.close()
        await core.close()
        await backend.close()


@pytest.mark.parametrize(
    "component, revision",
    [
        ("core", 99),
        ("memory", 99),
        ("foreign", 1),
        ("memory", 1),
        ("core", 1),
        native_schema.NativeStoreSchema.CORE.value,
    ],
)
async def test_unsupported_or_incomplete_store_versions_refuse_before_writes(
    native_a2a_backend_factory, component, revision
):
    from kestrel_sovereign.a2a.native_schema import NativeA2AMigrationRequired

    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:recipient"))
    try:
        await backend.execute_script(
            "CREATE TABLE a2a_native_store_schema (component TEXT PRIMARY KEY, revision INTEGER NOT NULL)"
        )
        await backend.execute(
            "INSERT INTO a2a_native_store_schema VALUES (?, ?)", (component, revision)
        )
        protocol_present = (
            component,
            revision,
        ) != native_schema.NativeStoreSchema.CORE.value
        if protocol_present:
            await backend.execute_script(
                "CREATE TABLE a2a_protocol_schema (singleton INTEGER PRIMARY KEY, revision INTEGER NOT NULL)"
            )
            await backend.execute("INSERT INTO a2a_protocol_schema VALUES (1, 1)")
        with pytest.raises(NativeA2AMigrationRequired):
            await manager.initialize()
        assert await backend.fetch_all(
            "SELECT component, revision FROM a2a_native_store_schema"
        ) == [(component, revision)]
        assert await backend.table_exists("a2a_protocol_schema") is protocol_present
        assert not await backend.table_exists("a2a_tasks")
    finally:
        await manager.close()


@pytest.mark.parametrize(
    ("native_a2a_backend_factory", "failure"),
    [("sqlite", "observability"), ("postgres", "observability"), ("sqlite", "memory-fts")],
    indirect=["native_a2a_backend_factory"],
)
async def test_real_schema_failure_rolls_back_core_and_never_publishes_completion(
    native_a2a_backend_factory, monkeypatch, failure
):
    from kestrel_sovereign.a2a.task_manager import TaskManager
    from kestrel_sovereign.storage.db.interface import QueryError, TransactionError

    backend = native_a2a_backend_factory()
    if failure == "memory-fts":
        assert backend.backend_type == "sqlite"
    await backend.connect()
    manager = TaskManager.from_backend(
        backend,
        scope=NativeRecipientScope("did:recipient"),
        owns_backend=True,
        include_memory=failure == "memory-fts",
        include_feedback=False,
    )
    try:
        with monkeypatch.context() as patch:
            if failure == "observability":

                async def rejected_column(*args, **kwargs):
                    await backend.execute(
                        "ALTER TABLE a2a_llm_calls ADD COLUMN agent_did TEXT"
                    )

                patch.setattr(
                    manager.observability_store,
                    "add_column_if_missing",
                    rejected_column,
                )
            else:
                execute_script = backend.execute_script

                async def reject_fts(script):
                    if "CREATE VIRTUAL TABLE IF NOT EXISTS a2a_memory_fts" in script:
                        script = script.replace(
                            "USING fts5", "USING native_test_missing_fts_module"
                        )
                    await execute_script(script)

                patch.setattr(backend, "execute_script", reject_fts)
            with pytest.raises((QueryError, TransactionError)):
                await manager.initialize()
        for table in (
            "a2a_tasks",
            "a2a_native_questions",
            "a2a_protocol_schema",
            "a2a_native_store_schema",
        ):
            assert not await backend.table_exists(table), (
                f"Failed initialization left {table} published"
            )
        await manager.initialize()
        expected = (
            [native_schema.NativeStoreSchema.CORE.value, ("memory", 1)]
            if failure == "memory-fts"
            else [native_schema.NativeStoreSchema.CORE.value]
        )
        assert (
            await backend.fetch_all(
                "SELECT component, revision FROM a2a_native_store_schema ORDER BY component"
            )
            == expected
        )
    finally:
        await manager.close()


@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
async def test_postgres_memory_trigger_presence_is_scoped_to_its_relation(
    native_a2a_backend_factory,
):
    from kestrel_sovereign.a2a.stores.unified.memory_service import MemoryService

    backend = native_a2a_backend_factory()
    assert backend.backend_type == "postgres"
    await backend.connect()
    memory = MemoryService(backend)
    try:
        await backend.execute_script("""
            CREATE TABLE unrelated_memory_target (id INTEGER);
            CREATE FUNCTION unrelated_memory_trigger() RETURNS trigger AS $$ BEGIN RETURN NEW; END; $$ LANGUAGE plpgsql;
            CREATE TRIGGER a2a_memory_tsv_update BEFORE INSERT ON unrelated_memory_target
            FOR EACH ROW EXECUTE FUNCTION unrelated_memory_trigger();
        """)
        await memory.initialize()
        memory_id = await memory.add_memory("session", "A native searchable archive")
        assert await backend.fetch_one(
            "SELECT content_tsv IS NOT NULL FROM a2a_memory WHERE id = ?", (memory_id,)
        ) == (True,), (
            "A same-named trigger on another relation suppressed the native search trigger"
        )
    finally:
        await memory.close()


@pytest.mark.parametrize("native_a2a_backend_factory", ["postgres"], indirect=True)
async def test_postgres_table_probe_sees_another_connections_committed_ddl(
    native_a2a_backend_factory,
):
    first, second = native_a2a_backend_factory(), native_a2a_backend_factory()
    assert first.backend_type == "postgres"
    await first.connect()
    await second.connect()
    try:
        async with first.transaction():
            assert not await first.table_exists("native_concurrent_catalog_probe")
            await second.execute(
                "CREATE TABLE native_concurrent_catalog_probe (id INTEGER)"
            )
            assert await first.table_exists("native_concurrent_catalog_probe"), (
                "A committed relation remained invisible to the canonical table probe"
            )
            assert await first.fetch_one(
                "SELECT COUNT(*) FROM native_concurrent_catalog_probe"
            ) == (0,)
    finally:
        await second.close()
        await first.close()


async def test_table_probe_uses_literal_names_and_selected_scope(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await backend.execute_script('CREATE TABLE "Native Quoted Name" (id INTEGER)')
        assert await backend.table_exists("Native Quoted Name")
        assert not await backend.table_exists("native quoted name")
        assert not await backend.table_exists("unrelated_native_table")
    finally:
        await backend.close()
