"""Fresh native startup and fail-closed offline migration boundary on real stores."""

import pytest
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest

from kestrel_sovereign.a2a.native_schema import NativeA2AMigrationRequired, NativeStoreSchema
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.utils.native_a2a import manager_for


@pytest.mark.parametrize("previous", [("core", 4), ("push", 1)])
async def test_previous_global_capture_bundles_require_offline_migration(
    native_a2a_backend_factory, previous
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:recipient"), owns_backend=False)
    try:
        await backend.execute_script(
            "CREATE TABLE a2a_native_store_schema (component TEXT PRIMARY KEY, revision INTEGER NOT NULL);"
            "CREATE TABLE a2a_protocol_schema (singleton INTEGER PRIMARY KEY, revision INTEGER NOT NULL)"
        )
        await backend.execute("INSERT INTO a2a_protocol_schema VALUES (1, 1)")
        rows = [previous]
        if previous[0] != "core":
            rows.append(NativeStoreSchema.CORE.value)
        for row in rows:
            await backend.execute("INSERT INTO a2a_native_store_schema VALUES (?, ?)", row)
        with pytest.raises(NativeA2AMigrationRequired, match="not supported"):
            await manager.initialize()
        assert not await backend.table_exists("a2a_tasks")
        assert set(await backend.fetch_all("SELECT * FROM a2a_native_store_schema")) == set(rows)
    finally:
        await manager.close()
        await backend.close()


async def test_capture_cannot_start_before_the_optional_push_stores(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(
        backend, NativeRecipientScope("did:recipient"), owns_backend=False
    )
    try:
        await manager.initialize()
        with pytest.raises(TransactionError, match="initialized stores") as caught:
            await manager.task_store.changes.enable_push_capture()
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert not await backend.table_exists("a2a_push_configs")
    finally:
        await manager.close()
        await backend.close()


async def test_partial_old_capture_refuses_to_publish_a_completed_native_bundle(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(
        backend, NativeRecipientScope("did:recipient"), owns_backend=False
    )
    try:
        # A partial store has no completed manager bundle to validate. Its
        # actual legacy capture must still be refused before publication.
        await manager.task_store.initialize()
        protocol = await backend.fetch_all("SELECT * FROM a2a_protocol_schema")
        if backend.backend_type == "sqlite":
            await backend.execute_script(
                "CREATE TRIGGER a2a_task_changes_update_v1 AFTER UPDATE ON a2a_tasks "
                "BEGIN SELECT 1; END"
            )
        else:
            await backend.execute_script(
                "CREATE FUNCTION a2a_task_changes_v1() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END; $$; "
                "CREATE TRIGGER a2a_task_changes_v1 AFTER UPDATE ON a2a_tasks "
                "FOR EACH ROW EXECUTE FUNCTION a2a_task_changes_v1()"
            )
        with pytest.raises(TransactionError, match="capture.*offline") as caught:
            await manager.initialize()
        assert isinstance(caught.value.__cause__, NativeA2AMigrationRequired)
        assert not await backend.table_exists("a2a_native_store_schema")
        assert await backend.fetch_all("SELECT * FROM a2a_protocol_schema") == protocol
    finally:
        await manager.close()
        await backend.close()


@pytest.mark.parametrize(
    "table", ["a2a_tasks", "a2a_outbound_tasks", "pending_a2a_questions"]
)
async def test_unmigrated_data_refuses_startup_before_any_native_table_or_settlement(
    native_a2a_backend_factory, table
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(
        backend, NativeRecipientScope("did:recipient"), owns_backend=False
    )
    try:
        await backend.execute_script(
            f"CREATE TABLE {table} (id TEXT PRIMARY KEY, status TEXT, metadata TEXT)"
        )
        await backend.execute(
            f"INSERT INTO {table} (id, status, metadata) VALUES (?, ?, ?)",
            ("old-task", "submitted", '{"sender_verified":true,"sender":"claimed"}'),
        )
        before = await backend.fetch_all(f"SELECT id, status, metadata FROM {table}")
        with pytest.raises(NativeA2AMigrationRequired, match="offline"):
            await manager.initialize()
        assert (
            await backend.fetch_all(f"SELECT id, status, metadata FROM {table}")
            == before
        )
        for native in (
            "a2a_protocol_schema",
            "a2a_outbound_dispatches",
            "a2a_native_questions",
        ):
            assert not await backend.table_exists(native)
        with pytest.raises(NativeA2AMigrationRequired, match="offline"):
            await manager.task_store.initialize()
    finally:
        await manager.close()
        await backend.close()


async def test_native_revision_survives_restart_with_actual_task_data(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    scope = NativeRecipientScope("did:recipient", "tenant")
    manager = manager_for(backend, scope, owns_backend=False)
    try:
        await manager.initialize()
        assert await backend.fetch_one(
            "SELECT revision FROM a2a_protocol_schema WHERE singleton = 1"
        ) == (1,)
        task = await manager.create_task(
            SendMessageRequest(
                tenant=scope.tenant,
                message=new_text_message("Native input", role=Role.ROLE_USER),
            ),
            creator_agent_id="did:sender",
        )
        await manager.close()
        replacement = manager_for(backend, scope, owns_backend=False)
        try:
            await replacement.initialize()
            assert (
                await replacement.get_task_for_recipient(task.id, scope.recipient_id)
                == task
            )
        finally:
            await replacement.close()
    finally:
        await manager.close()
        await backend.close()


async def test_unknown_protocol_revision_is_not_silently_adopted(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(
        backend, NativeRecipientScope("did:recipient"), owns_backend=False
    )
    try:
        await backend.execute_script(
            "CREATE TABLE a2a_protocol_schema (singleton INTEGER PRIMARY KEY, revision INTEGER NOT NULL)"
        )
        await backend.execute(
            "INSERT INTO a2a_protocol_schema (singleton, revision) VALUES (1, 99)"
        )
        with pytest.raises(NativeA2AMigrationRequired, match="not supported"):
            await manager.initialize()
        assert not await backend.table_exists("a2a_tasks")
        assert await backend.fetch_one("SELECT revision FROM a2a_protocol_schema") == (
            99,
        )
    finally:
        await manager.close()
        await backend.close()


@pytest.mark.parametrize("legacy_present", [False, True])
async def test_core_storage_boot_neither_creates_nor_mutates_draft_question_state(
    native_a2a_backend_factory, legacy_present
):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    backend = native_a2a_backend_factory()
    await backend.connect()
    db = None
    manager = manager_for(
        backend, NativeRecipientScope("did:recipient"), owns_backend=False
    )
    try:
        if legacy_present:
            await backend.execute_script(
                "CREATE TABLE pending_a2a_questions (agent_id TEXT, task_id TEXT, recipient TEXT, original_question TEXT, deadline TEXT, status TEXT, receiver_settlement TEXT)"
            )
            await backend.execute(
                "INSERT INTO pending_a2a_questions (agent_id, task_id, recipient, original_question, deadline, status, receiver_settlement) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "did:recipient",
                    "old-question",
                    "Peer",
                    "Retain this old question",
                    "2025-04-01T00:00:00+00:00",
                    "EXPIRED",
                    None,
                ),
            )
            before = await backend.fetch_all("SELECT * FROM pending_a2a_questions")
        db = await AsyncDatabase.from_connected_backend(backend)
        if legacy_present:
            assert (
                await backend.fetch_all("SELECT * FROM pending_a2a_questions") == before
            )
            with pytest.raises(NativeA2AMigrationRequired, match="offline"):
                await manager.initialize()
        else:
            assert not await backend.table_exists("pending_a2a_questions")
            await manager.initialize()
            assert await backend.table_exists("a2a_native_questions")
    finally:
        await manager.close()
        if db is not None:
            await db.close()
        else:
            await backend.close()


async def test_two_native_managers_boot_concurrently_on_one_fresh_backend_scope(
    native_a2a_backend_factory,
):
    import asyncio

    first, second = native_a2a_backend_factory(), native_a2a_backend_factory()
    await first.connect()
    await second.connect()
    managers = [
        manager_for(first, NativeRecipientScope("did:first")),
        manager_for(second, NativeRecipientScope("did:second")),
    ]
    try:
        await asyncio.gather(*(manager.initialize() for manager in managers))
        assert await first.fetch_one(
            "SELECT revision FROM a2a_protocol_schema WHERE singleton = 1"
        ) == (1,)
    finally:
        await asyncio.gather(*(manager.close() for manager in managers))


async def test_manager_keeps_schema_lease_across_store_initializers(
    native_a2a_backend_factory, monkeypatch
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(backend, NativeRecipientScope("did:recipient"))
    initialize = manager.question_store.initialize
    checked = []

    async def initialize_questions():
        if backend.backend_type == "sqlite":
            assert backend.owns_open_transaction, (
                "Question DDL escaped the complete schema transaction"
            )
        else:
            assert await backend.fetch_one(
                "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE pid = pg_backend_pid() "
                "AND locktype = 'advisory' AND granted)"
            ) == (True,), "Question DDL escaped the shared native schema lease"
        checked.append(True)
        await initialize()

    monkeypatch.setattr(manager.question_store, "initialize", initialize_questions)
    try:
        await manager.initialize()
        assert checked == [True]
    finally:
        await manager.close()


async def test_previous_native_input_shape_is_refused_before_any_startup_write(
    native_a2a_backend_factory,
):
    from kestrel_sovereign.a2a.native_schema import require_native_a2a_schema

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await backend.execute_script("""
            CREATE TABLE a2a_protocol_schema (singleton INTEGER PRIMARY KEY, revision INTEGER);
            INSERT INTO a2a_protocol_schema VALUES (1, 1);
            CREATE TABLE a2a_task_inputs (task_id TEXT PRIMARY KEY, request_json TEXT, request_digest TEXT);
            INSERT INTO a2a_task_inputs VALUES ('old-input', '{"private":"unchanged"}', 'old-digest')
        """)
        before = await backend.fetch_all("SELECT * FROM a2a_task_inputs")
        with pytest.raises(NativeA2AMigrationRequired, match="input.*schema"):
            await require_native_a2a_schema(backend)
        assert await backend.fetch_all("SELECT * FROM a2a_task_inputs") == before
        assert not await backend.table_exists("a2a_tasks")
        assert not await backend.table_exists("a2a_native_store_schema")
    finally:
        await backend.close()


@pytest.mark.parametrize("revision", [2, 3])
async def test_pre_projection_core_bundle_revision_requires_offline_migration(
    native_a2a_backend_factory,
    revision,
):
    from kestrel_sovereign.a2a.native_schema import require_native_a2a_schema

    backend = native_a2a_backend_factory()
    await backend.connect()
    manager = manager_for(
        backend, NativeRecipientScope("did:recipient"), owns_backend=False
    )
    try:
        await manager.initialize()
        await backend.execute(
            "UPDATE a2a_native_store_schema SET revision = ? WHERE component = 'core'",
            (revision,),
        )
        before = await backend.fetch_all(
            "SELECT component, revision FROM a2a_native_store_schema"
        )
        with pytest.raises(NativeA2AMigrationRequired, match="schema revision"):
            await require_native_a2a_schema(backend)
        assert (
            await backend.fetch_all(
                "SELECT component, revision FROM a2a_native_store_schema"
            )
            == before
        )
    finally:
        await manager.close()
        await backend.close()
