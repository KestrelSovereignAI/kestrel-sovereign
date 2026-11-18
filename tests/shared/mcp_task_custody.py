"""Canonical owned resources and native commit faults for MCP Task tests."""

from contextlib import AsyncExitStack
from dataclasses import replace
from types import SimpleNamespace

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio

from kestrel_sovereign.mcp_task_store import MCPTaskBinding, MCPTaskStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.utils.original_failure import lost_original_database_commit_ack as lost_commit_ack


@pytest_asyncio.fixture(params=['sqlite', pytest.param('postgres', marks=pytest.mark.native_postgres)])
async def custody(tmp_path, monkeypatch, request):
    key = tmp_path / 'owned-task-key'
    key.write_bytes(Fernet.generate_key())
    key.chmod(0o600)
    monkeypatch.setenv('KESTREL_DATA_KEY_FILE', str(key))
    monkeypatch.delenv('KESTREL_DATA_KEY', raising=False)

    async def no_core_schema(database):
        pass

    async with AsyncExitStack() as cleanup:
        if request.param == 'postgres':
            from tests.utils.postgres_schema import disposable_postgres_schema, postgres_test_url, with_search_path
            dsn = postgres_test_url()
            if not dsn:
                pytest.skip('PostgreSQL test URL is not configured')
            admin = await AsyncDatabase.postgres(dsn, min_pool_size=1, max_pool_size=2,
                schema_initializer=no_core_schema)
            cleanup.push_async_callback(admin.close)
            schema = await cleanup.enter_async_context(disposable_postgres_schema(admin, 'mcp_task_test'))

            async def database():
                return await AsyncDatabase.postgres(with_search_path(dsn, schema),
                    min_pool_size=1, max_pool_size=2, schema_initializer=no_core_schema)
        else:
            async def database():
                return await AsyncDatabase.sqlite(str(tmp_path / 'tasks.db'), schema_initializer=no_core_schema)

        async def reopen(**changes):
            db = await database()
            cleanup.push_async_callback(db.close)
            options = dict(agent_did='did:test:task-owner', max_active=2,
                max_records=8, max_payload_bytes=4096)
            options.update(changes)
            store = MCPTaskStore(db, **options)
            await store.initialize()
            cleanup.push_async_callback(store.close)
            return store

        yield SimpleNamespace(store=await reopen(), reopen=reopen, key=key, tmp_path=tmp_path)


def binding(**changes):
    return replace(MCPTaskBinding('did:test:task-owner', 'private-operator', 'private-session',
        'original-leaf-claim', 'a' * 64, 'b' * 64, 'private-native-tool', 'c' * 64), **changes)


def claim(task_id='remote-private-task'):
    # Native schema validation belongs to the feature; Core preserves the JSON.
    return {'version': 1, 'native': {'resultType': 'task', 'taskId': task_id,
        'status': 'working', 'createdAt': '2026-11-18T00:00:00Z',
        'lastUpdatedAt': '2026-11-18T00:00:00Z', 'ttlMs': None,
        'vendor.example/private': {'marker': 'private-server-state'}}}


async def attach(store, issuance, task_id='remote-private-task'):
    return await store.attach_task(issuance, remote_task_id=task_id, envelope=claim(task_id))
