"""Offline plan/apply on real storage, including rollback and no-wake history."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from a2a.helpers.proto_helpers import new_text_message
from a2a.types import Role, SendMessageRequest

from kestrel_sovereign.a2a.legacy_cutover import (
    apply_legacy_migration,
    prepare_legacy_migration,
)
from kestrel_sovereign.a2a.migrate_legacy import LegacyTaskMigrationError
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.base import UnifiedStoreBase
from kestrel_sovereign.a2a.stores.unified.context_store import ContextOwnership
from kestrel_sovereign.storage.db.interface import QueryError, TransactionError
from tests.utils.native_a2a import manager_for


def ownership():
    return {
        "historical": ContextOwnership(
            "reviewed-native-context", "tenant", "did:recipient", "did:creator"
        )
    }


async def legacy(backend, *, terminal_fence=False):
    store = UnifiedStoreBase(backend, owns_backend=False)
    await backend.execute_script(f"""
        CREATE TABLE a2a_tasks (
            id TEXT PRIMARY KEY, session_id TEXT, task_type TEXT NOT NULL,
            status TEXT, message {store.json_type()}, artifacts {store.json_type()},
            history {store.json_type()}, metadata {store.json_type()},
            created_at {store.timestamp_type()}, updated_at {store.timestamp_type()},
            creator_agent_id TEXT, recipient_agent_id TEXT
        );
        CREATE TABLE a2a_outbound_tasks (id TEXT PRIMARY KEY, terminal_state TEXT);
        CREATE TABLE pending_a2a_questions (
            task_id TEXT PRIMARY KEY, status TEXT, retry_state TEXT,
            receiver_settlement TEXT, receiver_late_state TEXT
        );
    """)
    await backend.execute(
        "INSERT INTO a2a_tasks (id, session_id, task_type, status, message, artifacts, history, metadata, created_at, updated_at, creator_agent_id, recipient_agent_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "historical",
            "draft-session",
            "generic",
            "completed",
            json.dumps(
                {
                    "role": "agent",
                    "parts": [{"type": "text", "text": "Historical answer"}],
                }
            ),
            "[]",
            json.dumps(
                [
                    {
                        "role": "user",
                        "parts": [{"type": "text", "text": "Historical input"}],
                    }
                ]
            ),
            '{"sender_verified":true,"sender":"did:forged","business":42}',
            store.to_timestamp_param(datetime(2026, 10, 20, tzinfo=UTC)),
            store.to_timestamp_param(datetime(2026, 10, 20, tzinfo=UTC)),
            "did:creator",
            "did:recipient",
        ),
    )
    if terminal_fence and store.is_postgres:
        await backend.execute_script("""
            CREATE FUNCTION a2a_tasks_enforce_authority_fence_v4() RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'legacy terminal task cannot be changed';
            END; $$ LANGUAGE plpgsql;
            CREATE TRIGGER a2a_tasks_authority_fence_v4 BEFORE UPDATE ON a2a_tasks
            FOR EACH ROW EXECUTE FUNCTION a2a_tasks_enforce_authority_fence_v4();
        """)
    elif terminal_fence:
        await backend.execute_script("""
            CREATE TRIGGER a2a_tasks_terminal_update_v3 BEFORE UPDATE ON a2a_tasks
            BEGIN SELECT RAISE(ABORT, 'legacy terminal task cannot be changed'); END;
        """)
    await backend.execute(
        "INSERT INTO a2a_outbound_tasks VALUES ('old-dispatch', 'completed')"
    )
    await backend.execute(
        "INSERT INTO pending_a2a_questions VALUES ('old-question', 'EXPIRED', NULL, 'CONFIRMED', NULL)"
    )


async def test_offline_cutover_preserves_archives_ownership_and_history_without_new_wakes(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await legacy(backend, terminal_fence=True)
        before = await backend.fetch_all("SELECT * FROM a2a_tasks")
        from kestrel_sovereign.a2a.native_schema import NativeA2AMigrationRequired

        unconverted = manager_for(
            backend, NativeRecipientScope("did:recipient", "tenant"), owns_backend=False
        )
        try:
            with pytest.raises(NativeA2AMigrationRequired, match="offline"):
                await unconverted.initialize()
        finally:
            await unconverted.close()
        assert not await backend.table_exists("a2a_protocol_schema")
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        assert not await backend.table_exists("a2a_protocol_schema")
        assert await backend.fetch_all("SELECT * FROM a2a_tasks") == before
        archives = await apply_legacy_migration(backend, plan)
        assert (
            await backend.fetch_all(f'SELECT * FROM "{archives["a2a_tasks"]}"')
            == before
        )
        assert await backend.fetch_one("SELECT COUNT(*) FROM a2a_outbound_tasks") == (
            0,
        )
        assert await backend.fetch_one(
            "SELECT COUNT(*) FROM pending_a2a_questions"
        ) == (0,)
        manager = manager_for(
            backend, NativeRecipientScope("did:recipient", "tenant"), owns_backend=False
        )
        try:
            await manager.initialize()
            restored = await manager.get_task_for_recipient(
                "historical", manager.host_agent_id
            )
            assert (
                restored.id == "historical"
                and restored.context_id == "reviewed-native-context"
            )
            assert restored.status.message.parts[0].text == "Historical answer"
            assert not restored.status.HasField("timestamp")
            await manager.context_store.require_owned(
                restored.context_id,
                manager.task_store.scope,
                creator_agent_id="did:creator",
            )
            assert (
                "sender_verified" not in restored.metadata
                and "sender" not in restored.metadata
            )
            fresh = await manager.create_task(
                SendMessageRequest(
                    tenant="tenant",
                    message=new_text_message("Native input", role=Role.ROLE_USER),
                    metadata={"cognition_wake_eligible": 0},
                ),
                creator_agent_id="did:creator",
            )
            candidates = await manager.task_store.list_cognition_wake_candidates(
                recipient_agent_id=manager.host_agent_id,
                live_changed_since=datetime(2020, 1, 1, tzinfo=UTC),
                terminal_changed_since=datetime(2020, 1, 1, tzinfo=UTC),
            )
            assert [item.task.id for item in candidates] == [fresh.id]
            with pytest.raises(QueryError, match="terminal A2A task"):
                await backend.execute(
                    "UPDATE a2a_tasks SET status = 'working' WHERE id = 'historical'"
                )
        finally:
            await manager.close()
    finally:
        await backend.close()


@pytest.mark.parametrize(
    "change",
    [
        "task-live",
        "outbound-unknown",
        "question-waiting",
        "question-owed-wake",
        "question-unconfirmed",
        "old-expiry-unconfirmed",
        "foreign-owner",
        "missing-owner",
    ],
)
async def test_preview_refuses_undrained_work_or_unreviewed_provenance(
    native_a2a_backend_factory, change
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await legacy(backend)
        owners = ownership()
        sql = {
            "task-live": "UPDATE a2a_tasks SET status = 'working'",
            "outbound-unknown": "UPDATE a2a_outbound_tasks SET terminal_state = NULL",
            "question-waiting": "UPDATE pending_a2a_questions SET status = 'WAITING'",
            "question-owed-wake": "UPDATE pending_a2a_questions SET retry_state = 'expired'",
            "question-unconfirmed": "UPDATE pending_a2a_questions SET receiver_settlement = 'PENDING'",
            "old-expiry-unconfirmed": "UPDATE pending_a2a_questions SET receiver_settlement = NULL",
        }
        if change in sql:
            await backend.execute(sql[change])
        elif change == "missing-owner":
            owners = {}
        else:
            owners = {
                "historical": replace(
                    owners["historical"], creator_agent_id="did:forged"
                )
            }
        with pytest.raises(LegacyTaskMigrationError):
            await prepare_legacy_migration(backend, ownership=owners)
        assert not await backend.table_exists("a2a_legacy_migrations")
        assert not await backend.table_exists("a2a_protocol_schema")
    finally:
        await backend.close()


async def test_apply_refuses_changed_source_before_any_archive_or_conversion(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await legacy(backend)
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        await backend.execute(
            "UPDATE a2a_tasks SET metadata = '{}' WHERE id = 'historical'"
        )
        with pytest.raises(
            (TransactionError, LegacyTaskMigrationError), match="changed after preview"
        ):
            await apply_legacy_migration(backend, plan)
        assert not await backend.table_exists("a2a_legacy_migrations")
        assert not await backend.table_exists("a2a_protocol_schema")
    finally:
        await backend.close()


async def test_failed_cutover_rolls_back_payload_schema_archives_and_marker(
    native_a2a_backend_factory, monkeypatch
):
    from kestrel_sovereign.a2a import legacy_cutover

    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await legacy(backend, terminal_fence=True)
        plan = await prepare_legacy_migration(backend, ownership=ownership())

        async def fail_publish(backend):
            assert await backend.fetch_one(
                "SELECT cognition_wake_eligible FROM a2a_tasks"
            ) == (0,)
            raise OSError("Injected failure after conversion and before marker")

        monkeypatch.setattr(legacy_cutover, "publish_native_a2a_schema", fail_publish)
        with pytest.raises((TransactionError, OSError), match="Injected failure"):
            await apply_legacy_migration(backend, plan)
        original = next(table for table in plan.tables if table.name == "a2a_tasks")
        assert await backend.fetch_all("SELECT * FROM a2a_tasks") == list(
            original.rows
        ), "Failed migration changed historical rows or schema"
        assert (
            await prepare_legacy_migration(
                backend, ownership=ownership(), migration_id=plan.migration_id
            )
        ).review() == plan.review()
        assert not await backend.table_exists("a2a_protocol_schema")
        assert not await backend.table_exists("a2a_legacy_migrations")
        for table in plan.tables:
            archive = f"a2a_legacy_{plan.migration_id}_{table.name.removeprefix('a2a_').replace('pending_a2a_questions', 'questions')}"
            assert not await backend.table_exists(archive)
    finally:
        await backend.close()


async def test_changed_postgres_trigger_function_invalidates_the_review(
    native_a2a_backend_factory,
):
    backend = native_a2a_backend_factory()
    if backend.backend_type != "postgres":
        pytest.skip("PostgreSQL stores trigger function bodies separately")
    await backend.connect()
    try:
        await legacy(backend, terminal_fence=True)
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        await backend.execute_script("""
            CREATE OR REPLACE FUNCTION a2a_tasks_enforce_authority_fence_v4()
            RETURNS trigger AS $$ BEGIN RETURN NEW; END; $$ LANGUAGE plpgsql;
        """)
        with pytest.raises(
            (TransactionError, LegacyTaskMigrationError), match="changed after preview"
        ):
            await apply_legacy_migration(backend, plan)
        assert not await backend.table_exists("a2a_legacy_migrations")
        assert not await backend.table_exists("a2a_protocol_schema")
    finally:
        await backend.close()


async def test_archive_truncation_aborts_before_conversion_and_rolls_back(
    native_a2a_backend_factory, monkeypatch
):
    backend = native_a2a_backend_factory()
    await backend.connect()
    try:
        await legacy(backend)
        plan = await prepare_legacy_migration(backend, ownership=ownership())
        execute = backend.execute
        truncated = []

        async def truncate_archive(query, *args, **kwargs):
            if query.startswith('CREATE TABLE "a2a_legacy_') and " AS SELECT " in query:
                query += " WHERE 1 = 0"
                truncated.append(True)
            return await execute(query, *args, **kwargs)

        monkeypatch.setattr(backend, "execute", truncate_archive)
        with pytest.raises(
            (TransactionError, LegacyTaskMigrationError),
            match="retain every source row",
        ):
            await apply_legacy_migration(backend, plan)
        assert truncated == [True]
        assert (
            await prepare_legacy_migration(
                backend, ownership=ownership(), migration_id=plan.migration_id
            )
        ).review() == plan.review()
        assert not await backend.table_exists("a2a_protocol_schema")
    finally:
        await backend.close()
