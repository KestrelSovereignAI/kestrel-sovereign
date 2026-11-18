"""Private Task custody on real SQLite/PostgreSQL and actual encryption keys."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
from types import SimpleNamespace

from cryptography.fernet import Fernet
import pytest
import pytest_asyncio

from kestrel_sovereign.execution_custody import execution_commit_outcome
from kestrel_sovereign.mcp_task_store import (
    MCPTaskBinding, MCPTaskConflict, MCPTaskIntegrityError, MCPTaskStore,
)
from kestrel_sovereign.security.key_rotation import KeyRotationService, RotationStatus
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.interface import TransactionError


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


@pytest.mark.asyncio
@pytest.mark.parametrize('task_id', ['', 'é/任務/🔒', ' opaque\u0000task\r\n'])
async def test_reservation_and_exact_private_claim_survive_fresh_connections(custody, task_id):
    store = custody.store
    reserved = await store.reserve(binding())
    assert reserved.fresh and reserved.issuance.record == reserved.record
    # A second physical connection sees custody BEFORE a caller could dispatch.
    restarted = await custody.reopen()
    assert await restarted.lookup(binding()) == reserved.record
    assert not (await restarted.reserve(binding())).fresh
    attached = await attach(store, reserved.issuance, task_id)
    assert await restarted.read(binding(), attached.handle) == attached
    assert attached.remote_task_id == task_id and attached.envelope == claim(task_id)
    detached = attached.envelope
    detached['native']['ttlMs'] = 123
    assert attached.envelope['native']['ttlMs'] is None
    assert await restarted.list_attached() == (attached,)
    rows = await store.database.fetchall('SELECT * FROM mcp_task_records')
    serialized = repr(rows)
    for private in ('private-operator', 'private-session', 'original-leaf-claim',
                    'private-native-tool', 'private-server-state', reserved.issuance.secret):
        assert private not in serialized
    assert 'secret' not in vars(reserved.record)


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [
    ('agent_did', 'did:test:foreign'), ('principal_id', 'foreign'),
    ('session_id', 'foreign'), ('operation_id', 'foreign'),
    ('connection_id', 'd' * 64), ('authorization_id', 'd' * 64),
    ('tool_name', 'foreign'), ('contract_digest', 'd' * 64),
])
async def test_identifiers_never_replace_original_binding_or_issuance(custody, field, value):
    store = custody.store
    reserved = await store.reserve(binding())
    foreign = binding(**{field: value})
    with pytest.raises(MCPTaskConflict):
        await store.read(foreign, reserved.record.handle)
    if field != 'operation_id':
        with pytest.raises(MCPTaskConflict):
            await store.reserve(foreign)
    with pytest.raises(MCPTaskConflict):
        await attach(store, replace(reserved.issuance, secret='x' * 43))
    with pytest.raises(MCPTaskConflict):
        await attach(store, replace(reserved.issuance,
            record=replace(reserved.record, binding=foreign)))
    assert (await store.read(binding(), reserved.record.handle)).phase == 'ISSUING'


@pytest.mark.asyncio
async def test_two_native_writers_share_active_capacity_and_lowered_limits_keep_inventory(custody):
    first = await custody.reopen(max_active=1)
    second = await custody.reopen(max_active=1)
    outcomes = await asyncio.gather(first.reserve(binding(operation_id='one')),
        second.reserve(binding(operation_id='two')), return_exceptions=True)
    assert sum(isinstance(item, MCPTaskConflict) for item in outcomes) == 1
    admitted, = [item for item in outcomes if not isinstance(item, BaseException)]
    assert admitted.fresh
    assert (await second.database.fetchone('SELECT COUNT(*) FROM mcp_task_records')) == (1,)
    await attach(first, admitted.issuance, 'one')
    third = await custody.store.reserve(binding(operation_id='third'))
    await attach(custody.store, third.issuance, 'two')
    lowered = await custody.reopen(max_active=1, max_records=1, max_payload_bytes=1)
    assert len(await lowered.list_attached()) == 2
    assert not (await lowered.reserve(admitted.record.binding)).fresh
    with pytest.raises(MCPTaskConflict, match='capacity'):
        await lowered.reserve(binding(operation_id='new'))


@pytest.mark.asyncio
@pytest.mark.parametrize('disposition', ['SETTLED', 'NOT_SENT'])
async def test_confirmed_initial_outcome_releases_active_slot_but_keeps_consumed_identity(custody, disposition):
    store = await custody.reopen(max_active=1, max_records=2)
    reserved = await store.reserve(binding())
    lowered = await custody.reopen(max_active=1, max_records=1, max_payload_bytes=1)
    evidence = {'version': 1, 'native': {'resultType': 'complete', 'content': []}} if disposition == 'SETTLED' else {
        'version': 1, 'refusal': {'reason': 'native submission never began'}}
    result = await lowered.settle_initial(reserved.issuance, disposition=disposition, envelope=evidence)
    assert result.phase == disposition and result.envelope == evidence
    existing = await store.reserve(binding())
    assert not existing.fresh and existing.issuance is None
    assert (await store.reserve(binding(operation_id='second'))).fresh
    with pytest.raises(MCPTaskConflict, match='retained record capacity'):
        await store.reserve(binding(operation_id='third'))
    with pytest.raises(MCPTaskConflict):
        await attach(store, reserved.issuance)


@pytest.mark.asyncio
async def test_duplicate_remote_identity_and_oversize_claim_never_reopen_initial_dispatch(custody):
    store = await custody.reopen(max_active=3)
    first = await store.reserve(binding())
    second_binding = binding(operation_id='another-leaf', principal_id='another-operator', session_id='another-session')
    second = await store.reserve(second_binding)
    other = await custody.reopen(max_active=3)
    outcomes = await asyncio.gather(attach(store, first.issuance), attach(other, second.issuance),
        return_exceptions=True)
    assert sum(isinstance(item, MCPTaskConflict) for item in outcomes) == 1
    assert len(await other.list_attached()) == 1
    third = await store.reserve(binding(operation_id='oversize-result'))
    oversized = claim('third')
    oversized['native']['large'] = 'x' * 4096
    with pytest.raises(TransactionError, match='storage budget'):
        await store.attach_task(third.issuance, remote_task_id='third', envelope=oversized)
    assert (await store.lookup(third.record.binding)).phase == 'ISSUING'
    assert not (await other.reserve(third.record.binding)).fresh
    with pytest.raises(MCPTaskConflict, match='active capacity'):
        await other.reserve(binding(operation_id='replacement'))


@pytest.mark.asyncio
@pytest.mark.parametrize('column,value', [
    ('handle', 'changed'), ('operation_digest', 'd' * 64), ('binding_digest', 'd' * 64),
    ('remote_digest', 'd' * 64), ('revision', 200), ('phase', 'SETTLED'),
    ('claim_hash', 'd' * 64), ('payload_limit', 5000), ('payload', b'damaged'),
])
async def test_every_retained_routing_or_cas_column_is_authenticated_before_new_admission(custody, column, value):
    store = custody.store
    reserved = await store.reserve(binding())
    await attach(store, reserved.issuance)
    # Column names are the parametrized constant allowlist above, never input.
    await store.database.execute(f'UPDATE mcp_task_records SET {column}=?', (value,))
    with pytest.raises((MCPTaskIntegrityError, TransactionError)):
        await store.reserve(binding(operation_id='new'))
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (1,)
    with pytest.raises(MCPTaskIntegrityError):
        await store.list_attached()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['envelope', 'missing', 'key'])
async def test_lost_original_envelope_or_key_is_never_silently_replaced(custody, failure):
    store = custody.store
    await store.reserve(binding())
    if failure == 'envelope':
        await store.database.execute('UPDATE mcp_task_keys SET wrapped_key=?', (b'broken',))
    elif failure == 'key':
        custody.key.write_bytes(Fernet.generate_key())
    elif store.is_postgres:
        await store.database.execute('ALTER TABLE mcp_task_records DROP CONSTRAINT mcp_task_records_scope_fkey')
        await store.database.execute('DELETE FROM mcp_task_keys')
    else:
        await store.database.execute('PRAGMA foreign_keys=OFF')
        try:
            await store.database.execute('DELETE FROM mcp_task_keys')
        finally:
            await store.database.execute('PRAGMA foreign_keys=ON')
    with pytest.raises(MCPTaskIntegrityError):
        await custody.reopen()
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (1,)
    if failure == 'missing':
        with pytest.raises(TransactionError, match='admission scope is missing'):
            await store.reserve(binding(operation_id='new'))
        assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_keys') == (0,)


@pytest.mark.asyncio
async def test_actual_key_rotation_rewraps_task_root_and_preserves_remote_custody(custody):
    store = custody.store
    reserved = await store.reserve(binding())
    attached = await attach(store, reserved.issuance)
    before = await store.database.fetchall('SELECT * FROM mcp_task_records')
    wrapped_before = await store.database.fetchone('SELECT wrapped_key FROM mcp_task_keys')
    # The generic registry also owns these required master-key stores. Empty
    # genuine tables represent this otherwise unused fresh test database.
    await store.database.execute('CREATE TABLE conversation_history (id INTEGER PRIMARY KEY, content TEXT, rendered_content TEXT, agent_id TEXT, metadata TEXT)')
    binary = 'BYTEA' if store.is_postgres else 'BLOB'
    await store.database.execute(f'CREATE TABLE files (content_hash TEXT PRIMARY KEY, content {binary}, metadata TEXT)')
    next_key = custody.tmp_path / 'rotated-key'
    next_key.write_bytes(Fernet.generate_key())
    next_key.chmod(0o600)
    service = KeyRotationService(SimpleNamespace(database=store.database))
    await service.initialize()
    try:
        rotation = await service.start_rotation(new_key_file=str(next_key))
        await service.drain_rotations()
        status = await service.get_status(rotation)
        assert status.status == RotationStatus.COMPLETED and status.records_processed == 1
    finally:
        await service.shutdown()
    assert await store.database.fetchall('SELECT * FROM mcp_task_records') == before
    assert await store.database.fetchone('SELECT wrapped_key FROM mcp_task_keys') != wrapped_before
    custody.key.write_bytes(next_key.read_bytes())
    restarted = await custody.reopen()
    assert await restarted.read(binding(), attached.handle) == attached
    assert not (await restarted.reserve(binding())).fresh


@pytest.mark.asyncio
async def test_ambient_transactions_cannot_publish_issuance_and_close_preserves_primary(custody):
    store = custody.store
    async with store.database.transaction():
        with pytest.raises(MCPTaskConflict, match='own durable transaction'):
            await store.reserve(binding())
        with pytest.raises(MCPTaskConflict, match='own durable transaction'):
            await store.initialize()
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (0,)
    await store.close()
    assert await store.database.fetchone('SELECT 1') == (1,)
    with pytest.raises(RuntimeError, match='not initialized'):
        await store.reserve(binding())
    await store.initialize()
    assert (await store.reserve(binding())).fresh


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['reserve', 'attach'])
async def test_native_commit_ack_loss_preserves_uncertainty_without_reissuing(custody, monkeypatch, operation):
    from tests.utils.original_failure import fail_original_file_read
    store = custody.store
    issuance = (await store.reserve(binding())).issuance if operation == 'attach' else None
    originals = []

    with monkeypatch.context() as patch:
        if store.is_sqlite:
            import aiosqlite
            native_execute = aiosqlite.Connection._execute

            async def lost_sqlite_ack(connection, function, *args, **kwargs):
                result = await native_execute(connection, function, *args, **kwargs)
                committed = function.__name__ == 'commit' or (
                    function.__name__ == 'execute' and args
                    and isinstance(args[0], str) and args[0].strip().upper() == 'COMMIT')
                if not originals and committed:
                    fail_original_file_read(originals)
                return result

            patch.setattr(aiosqlite.Connection, '_execute', lost_sqlite_ack)
        else:
            import asyncpg
            native_exit = asyncpg.transaction.Transaction.__aexit__

            async def lost_postgres_ack(transaction, error_type, error, traceback):
                result = await native_exit(transaction, error_type, error, traceback)
                if error_type is None and not transaction._nested and not originals:
                    fail_original_file_read(originals)
                return result

            patch.setattr(asyncpg.transaction.Transaction, '__aexit__', lost_postgres_ack)
        # The real driver commits the real SQL before a real descriptor read
        # fails at its acknowledgment boundary. No commit outcome is fabricated.
        with pytest.raises(BaseException) as caught:
            if operation == 'reserve':
                await store.reserve(binding())
            else:
                await attach(store, issuance)
        assert originals and execution_commit_outcome(caught.value) == 'unknown'
    restarted = await custody.reopen()
    recovered = await restarted.lookup(binding())
    assert recovered is not None and recovered.phase == ('ISSUING' if operation == 'reserve' else 'ATTACHED')
    assert not (await restarted.reserve(binding())).fresh
    if operation == 'attach':
        with pytest.raises(MCPTaskConflict):
            await attach(restarted, issuance)
