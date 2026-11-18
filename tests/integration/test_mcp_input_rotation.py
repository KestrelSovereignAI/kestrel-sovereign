"""Real input-envelope migration, rotation and native failure custody (#4152)."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from cryptography.fernet import Fernet
import pytest

from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sdk.security.aead import AEADCipher
from kestrel_sdk.security.encryption import get_agent_key, get_fernet
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_evidence_carriers
from kestrel_sovereign.mcp_input_store import MCPInputIntegrityError, MCPInputStore
from kestrel_sovereign.security.key_rotation import KeyRotationService, RotationStatus, _hash_key
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.integration.test_key_rotation_recovery import reject_writes
from tests.integration.test_mcp_input_store import custody as custody, binding, create
from tests.utils.original_failure import lost_original_database_commit_ack


async def legacy(store, *, schema_one=True):
    """Seed actual v1 purpose wrapping/schema before a fresh initializer runs."""
    root = store._root
    old = AEADCipher(get_agent_key(store.agent_did, 'service-keys')).encrypt(root)
    await store.database.execute('UPDATE mcp_input_scopes SET wrapped_key=?,wrap_version=1 WHERE scope=?', (old, store.scope))
    if schema_one:
        await store.database.execute('ALTER TABLE mcp_input_scopes DROP COLUMN wrap_version')
        await store.database.execute('UPDATE mcp_input_schema SET revision=1')
    return root, old


async def rotation_service(store, key, tmp_path):
    db = store.database
    await db.execute('CREATE TABLE conversation_history (id INTEGER PRIMARY KEY, content TEXT, rendered_content TEXT, agent_id TEXT, metadata TEXT)')
    binary = 'BYTEA' if store.is_postgres else 'BLOB'
    await db.execute(f'CREATE TABLE files (content_hash TEXT PRIMARY KEY, content {binary}, metadata TEXT)')
    await db.execute('INSERT INTO files VALUES (?,?,?)', ('owned-file', get_fernet().encrypt(b'owned-private-file'), '{"enc":true}'))
    next_key = tmp_path / 'next-owned-input-key'
    next_key.write_bytes(Fernet.generate_key())
    next_key.chmod(0o600)
    service = KeyRotationService(SimpleNamespace(database=db))
    await service.initialize()
    return service, next_key


@pytest.mark.asyncio
async def test_fresh_rotation_reopens_identical_checkpoint_and_response_remains_single_use(custody, tmp_path):
    store, reopen, key = custody
    saved = await create(store)
    service, next_key = await rotation_service(store, key, tmp_path)
    try:
        job = await service.start_rotation(new_key_file=str(next_key))
        await service.drain_rotations()
        status = await service.get_status(job)
        assert status.status == RotationStatus.COMPLETED and status.records_processed == status.records_total == 2
    finally:
        await service.shutdown()
    key.write_bytes(next_key.read_bytes())
    restarted = await reopen()
    assert await restarted.read(binding(), saved.request_id) == saved
    claimed = await restarted.claim_response(saved, {}, now=101)
    with pytest.raises(ValueError):
        await restarted.claim_response(saved, {}, now=101)
    assert (await restarted.settle(claimed, state='RESOLVED')).state == 'RESOLVED'


@pytest.mark.asyncio
async def test_actual_legacy_root_migration_preserves_ciphertext_indexes_and_admission(custody):
    store, reopen, _ = custody
    saved = await create(store)
    before = await store.database.fetchall('SELECT * FROM mcp_inputs')
    gate = await store.database.fetchone('SELECT revision,observed_at FROM mcp_input_scopes')
    root, old = await legacy(store)
    fresh = await reopen()
    assert fresh._root == root and await fresh.read(binding(), saved.request_id) == saved
    assert await fresh.database.fetchall('SELECT * FROM mcp_inputs') == before
    assert await fresh.database.fetchone('SELECT revision,observed_at FROM mcp_input_scopes') == gate
    row = await fresh.database.fetchone('SELECT wrapped_key,wrap_version FROM mcp_input_scopes')
    assert bytes(row[0]) != old and row[1] == 2
    assert await fresh.database.fetchone('SELECT revision FROM mcp_input_schema') == (2,)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['start', 'resume'])
@pytest.mark.parametrize('fault', ['schema', 'legacy', 'version'])
async def test_all_scope_readiness_refuses_before_journal_or_any_ciphertext_change(custody, tmp_path, operation, fault):
    store, reopen, key = custody
    await create(store)
    other = await reopen(agent_did='did:test:unmigrated-input-owner')
    await legacy(other, schema_one=fault == 'schema')
    if fault == 'version':
        await store.database.execute('UPDATE mcp_input_scopes SET wrap_version=-1 WHERE scope=?', (other.scope,))
    service, next_key = await rotation_service(store, key, tmp_path)
    try:
        if operation == 'resume':
            await store.database.execute('INSERT INTO key_rotations VALUES (?,?,?,?,?,?,?,?,?)', (
                'old-registry-job', datetime.now(UTC).isoformat(), None,
                _hash_key(key.read_text().strip()), _hash_key(next_key.read_text().strip()),
                'failed', 0, 1, 'original-error'))
        before = await store.database.fetchall('SELECT * FROM key_rotations')
        files = await store.database.fetchall('SELECT * FROM files')
        scopes = await store.database.fetchall('SELECT * FROM mcp_input_scopes ORDER BY scope')
        with pytest.raises(RuntimeError, match='original-agent envelope migration'):
            if operation == 'start':
                await service.start_rotation(new_key_file=str(next_key))
            else:
                await service.resume_rotation('old-registry-job', str(next_key))
        assert await store.database.fetchall('SELECT * FROM key_rotations') == before
        assert await store.database.fetchall('SELECT * FROM files') == files
        assert await store.database.fetchall('SELECT * FROM mcp_input_scopes ORDER BY scope') == scopes
        assert not service._rotation_tasks
    finally:
        await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['raise', 'ignored'])
async def test_native_migration_write_refusal_keeps_original_schema_and_envelope(custody, failure):
    store, _, _ = custody
    await create(store)
    _, old = await legacy(store)
    # Actual backend triggers refuse the migration UPDATE; no return is mocked.
    await reject_writes(store.database, 'mcp_input_scopes', failure)
    fresh = MCPInputStore(store.database, agent_did=store.agent_did, max_pending=2, max_payload_bytes=4096)
    try:
        with pytest.raises((MCPInputIntegrityError, TransactionError), match='update denied|migration was not stored'):
            await fresh.initialize()
        assert fresh._root is None
        assert await store.database.fetchone('SELECT revision FROM mcp_input_schema') == (1,)
        assert bytes((await store.database.fetchone('SELECT wrapped_key FROM mcp_input_scopes'))[0]) == old
    finally:
        await fresh.close()


@pytest.mark.asyncio
async def test_real_migration_commit_ack_loss_retains_original_and_recovers_same_root(custody, monkeypatch):
    store, reopen, _ = custody
    saved = await create(store)
    root, _ = await legacy(store)
    fresh = MCPInputStore(store.database, agent_did=store.agent_did, max_pending=2, max_payload_bytes=4096)
    try:
        with lost_original_database_commit_ack(monkeypatch, store) as originals:
            with pytest.raises(ResourceOutcomeUnknown) as caught:
                await fresh.initialize()
            assert len(originals) == 1 and execution_commit_outcome(caught.value) == 'unknown'
            assert any(error is originals[0] for error in execution_evidence_carriers(caught.value))
        assert fresh._root is None
        recovered = await reopen()
        assert recovered._root == root and await recovered.read(binding(), saved.request_id) == saved
    finally:
        await fresh.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('fault', ['foreign', 'version', 'downgrade'])
async def test_invalid_original_envelope_remains_unusable_before_and_after_rotation(custody, tmp_path, rotated, fault):
    store, reopen, key = custody
    await create(store)
    other = await reopen(agent_did='did:test:other-input-owner')
    if rotated:
        service, next_key = await rotation_service(store, key, tmp_path)
        try:
            await service.start_rotation(new_key_file=str(next_key))
            await service.drain_rotations()
        finally:
            await service.shutdown()
        key.write_bytes(next_key.read_bytes())
    if fault == 'foreign':
        envelope = await other.database.fetchone('SELECT wrapped_key FROM mcp_input_scopes WHERE scope=?', (other.scope,))
        await store.database.execute('UPDATE mcp_input_scopes SET wrapped_key=? WHERE scope=?', (envelope[0], store.scope))
    else:
        await store.database.execute('UPDATE mcp_input_scopes SET wrap_version=? WHERE scope=?',
            (3 if fault == 'version' else 1, store.scope))
    with pytest.raises(MCPInputIntegrityError, match='Unsupported MCP input envelope version|encryption key is unavailable'):
        await reopen()


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['file', 'input'])
async def test_completed_native_rotation_failure_is_not_discarded_and_resume_refreshes_totals(custody, tmp_path, monkeypatch, boundary):
    store, reopen, key = custody
    saved = await create(store)
    service, next_key = await rotation_service(store, key, tmp_path)
    release = asyncio.Event()
    execute = service._execute_rotation
    async def gated(*args):
        await release.wait()
        await execute(*args)
    monkeypatch.setattr(service, '_execute_rotation', gated)
    try:
        job = await service.start_rotation(new_key_file=str(next_key))
        original, = service._rotation_tasks
        probe = await reopen()
        async def accept_commit():
            # An independent real connection sees only durable progress. Do not
            # reenter the original backend from its held native commit owner.
            return boundary == 'file' or (await probe.database.fetchone(
                'SELECT COUNT(*) FROM rotation_progress WHERE rotation_id=? AND table_name=?',
                (job, 'mcp_input_scopes')))[0] == 1
        with lost_original_database_commit_ack(monkeypatch, store, accept_commit=accept_commit) as originals:
            release.set()
            observed = await await_owned_task(original, retain_cancellations=True)
            assert original.done() and original in service._rotation_tasks
            assert observed.error is not None
            with pytest.raises(BaseException) as caught:
                await service.drain_rotations()
            assert len(originals) == 1 and execution_commit_outcome(caught.value) == 'unknown'
            assert any(error is originals[0] for error in execution_evidence_carriers(caught.value))
        assert not service._rotation_tasks
        assert (await service.get_status(job)).status == RotationStatus.FAILED
        await store.database.execute('UPDATE key_rotations SET records_total=1 WHERE id=?', (job,))
        assert await service.resume_rotation(job, str(next_key))
        await service.drain_rotations()
        completed = await service.get_status(job)
        assert completed.status == RotationStatus.COMPLETED
        assert completed.records_total == completed.records_processed == 2
    finally:
        release.set()
        await service.shutdown()
    key.write_bytes(next_key.read_bytes())
    restarted = await reopen()
    assert await restarted.read(binding(), saved.request_id) == saved


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['own', 'foreign', 'mismatched', 'observer', 'publication'])
async def test_original_rotation_owner_is_retained_and_only_authenticated_shutdown_is_suppressed(custody, tmp_path, monkeypatch, mode):
    from kestrel_sovereign.security import key_rotation as module
    from tests.utils.original_failure import fail_original_file_read
    store, _, key = custody
    await create(store)
    service, next_key = await rotation_service(store, key, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    execute = service._execute_rotation
    async def gated(*args):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            if mode == 'mismatched':
                raise asyncio.CancelledError('foreign replacement') from None
            raise
        await execute(*args)
    monkeypatch.setattr(service, '_execute_rotation', gated)
    observer = None
    try:
        if mode == 'publication':
            native_create = module.create_owned_task
            failures = []
            def refused_publication(*args, **kwargs):
                native_create(*args, **kwargs)
                fail_original_file_read(failures)
            with monkeypatch.context() as patch:
                patch.setattr(module, 'create_owned_task', refused_publication)
                with pytest.raises(OSError) as caught:
                    await service.start_rotation(new_key_file=str(next_key))
                assert len(failures) == 1 and caught.value is failures[0]
        else:
            await service.start_rotation(new_key_file=str(next_key))
        original, = service._rotation_tasks
        await entered.wait()
        if mode == 'foreign':
            original.cancel('foreign owner withdrawal')
            observed = await await_owned_task(original, retain_cancellations=True)
            assert observed.error is not None
            with pytest.raises(asyncio.CancelledError) as caught:
                await service.shutdown()
            assert any(error is observed.error for error in execution_evidence_carriers(caught.value))
        elif mode == 'mismatched':
            with pytest.raises(asyncio.CancelledError, match='foreign replacement'):
                await service.shutdown()
        elif mode == 'own':
            await service.shutdown()
            assert original.cancelled()
        elif mode == 'observer':
            joining = asyncio.Event()
            native_join = module.await_owned_task
            async def observed_join(*args, **kwargs):
                joining.set()
                return await native_join(*args, **kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(module, 'await_owned_task', observed_join)
                observer = asyncio.create_task(service.drain_rotations())
                await joining.wait()
                observer.cancel('drain observer withdrawal')
                assert not original.cancelling()
                release.set()
                result, = await asyncio.gather(observer, return_exceptions=True)
                assert isinstance(result, asyncio.CancelledError)
        else:
            release.set()
            await service.drain_rotations()
        assert original.done() and not service._rotation_tasks
    finally:
        release.set()
        if observer is not None:
            await asyncio.gather(observer, return_exceptions=True)
        await service.shutdown()
