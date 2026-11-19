"""Private Task custody on real SQLite/PostgreSQL and actual encryption keys."""

import asyncio
from dataclasses import replace
import hashlib

from cryptography.fernet import Fernet
import pytest

from kestrel_sdk.security.aead import AEADCipher
from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_evidence_carriers
from kestrel_sovereign.mcp_task_store import (
    MCPTaskConflict, MCPTaskIntegrityError,
)
from kestrel_sovereign.security.key_rotation import RotationStatus
from kestrel_sovereign.storage.db.interface import TransactionError
from kestrel_sovereign.mcp_task_submission import MCPTaskInteractiveOrigin
from tests.shared.mcp_task_custody import custody as custody, binding, claim, attach, lost_commit_ack, rotate_task_root, submission


@pytest.mark.asyncio
@pytest.mark.parametrize('task_id', ['', 'é/任務/🔒', ' opaque\u0000task\r\n'])
async def test_reservation_and_exact_private_claim_survive_fresh_connections(custody, task_id):
    store = custody.store
    reserved = await store.reserve(binding(), submission())
    assert reserved.fresh and reserved.issuance.record == reserved.record
    # A second physical connection sees custody BEFORE a caller could dispatch.
    restarted = await custody.reopen()
    assert await restarted.lookup(binding()) == reserved.record
    assert not (await restarted.reserve(binding(), submission())).fresh
    attached = await attach(store, reserved.issuance, task_id)
    assert await restarted.read(binding(), attached.handle) == attached
    assert attached.remote_task_id == task_id and attached.envelope == claim(task_id)
    detached = attached.envelope
    detached['native']['ttlMs'] = 123
    assert attached.envelope['native']['ttlMs'] is None
    assert await restarted.list_remote_tasks() == (attached,)
    rows = await store.database.fetchall('SELECT * FROM mcp_task_records')
    serialized = repr(rows)
    for private in (binding().origin.principal_digest, 'private-session', 'original-leaf-claim',
                    'private-native-tool', 'private-server-state', reserved.issuance.secret):
        assert private not in serialized
    assert 'secret' not in vars(reserved.record)


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [
    ('agent_did', 'did:test:foreign'),
    ('origin', MCPTaskInteractiveOrigin('d' * 64, 'private-session')),
    ('origin', MCPTaskInteractiveOrigin('f' * 64, 'foreign')), ('operation_id', 'foreign'),
    ('connection_id', 'd' * 64), ('authorization_id', 'd' * 64),
    ('tool_name', 'foreign'), ('contract_digest', 'd' * 64),
])
async def test_identifiers_never_replace_original_binding_or_issuance(custody, field, value):
    store = custody.store
    reserved = await store.reserve(binding(), submission())
    foreign = binding(**{field: value})
    foreign_submission = submission(origin=foreign.origin)
    if field == 'contract_digest':
        foreign_submission = submission(origin=foreign.origin, contract=b'other-original-contract')
        foreign = replace(foreign, contract_digest=hashlib.sha256(foreign_submission.contract).hexdigest())
    with pytest.raises(MCPTaskConflict):
        await store.read(foreign, reserved.record.handle)
    if field != 'operation_id':
        with pytest.raises(MCPTaskConflict):
            await store.reserve(foreign, foreign_submission)
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
    outcomes = await asyncio.gather(first.reserve(binding(operation_id='one'), submission()),
        second.reserve(binding(operation_id='two'), submission()), return_exceptions=True)
    assert sum(isinstance(item, MCPTaskConflict) for item in outcomes) == 1
    admitted, = [item for item in outcomes if not isinstance(item, BaseException)]
    assert admitted.fresh
    assert (await second.database.fetchone('SELECT COUNT(*) FROM mcp_task_records')) == (1,)
    await attach(first, admitted.issuance, 'one')
    third = await custody.store.reserve(binding(operation_id='third'), submission())
    await attach(custody.store, third.issuance, 'two')
    lowered = await custody.reopen(max_active=1, max_records=1, max_payload_bytes=1)
    assert len(await lowered.list_remote_tasks()) == 2
    assert not (await lowered.reserve(admitted.record.binding, submission(origin=(admitted.record.binding).origin))).fresh
    with pytest.raises(MCPTaskConflict, match='capacity'):
        await lowered.reserve(binding(operation_id='new'), submission())


@pytest.mark.asyncio
@pytest.mark.parametrize('disposition', ['SETTLED', 'NOT_SENT'])
async def test_confirmed_initial_outcome_releases_active_slot_but_keeps_consumed_identity(custody, disposition):
    store = await custody.reopen(max_active=1, max_records=2)
    reserved = await store.reserve(binding(), submission())
    lowered = await custody.reopen(max_active=1, max_records=1, max_payload_bytes=1)
    evidence = {'version': 1, 'native': {'resultType': 'complete', 'content': []}} if disposition == 'SETTLED' else {
        'version': 1, 'refusal': {'reason': 'native submission never began'}}
    result = await lowered.settle_initial(reserved.issuance, disposition=disposition, envelope=evidence)
    assert result.phase == disposition and result.envelope == evidence
    existing = await store.reserve(binding(), submission())
    assert not existing.fresh and existing.issuance is None
    assert (await store.reserve(binding(operation_id='second'), submission())).fresh
    with pytest.raises(MCPTaskConflict, match='retained record capacity'):
        await store.reserve(binding(operation_id='third'), submission())
    with pytest.raises(MCPTaskConflict):
        await attach(store, reserved.issuance)


@pytest.mark.asyncio
async def test_duplicate_remote_identity_and_oversize_claim_never_reopen_initial_dispatch(custody):
    store = await custody.reopen(max_active=3)
    first = await store.reserve(binding(), submission())
    second_binding = binding(operation_id='another-leaf', origin=MCPTaskInteractiveOrigin('d' * 64, 'another-session'))
    second = await store.reserve(second_binding, submission(origin=(second_binding).origin))
    other = await custody.reopen(max_active=3)
    outcomes = await asyncio.gather(attach(store, first.issuance), attach(other, second.issuance),
        return_exceptions=True)
    assert sum(isinstance(item, MCPTaskConflict) for item in outcomes) == 1
    assert len(await other.list_remote_tasks()) == 1
    third = await store.reserve(binding(operation_id='oversize-result'), submission())
    oversized = claim('third')
    oversized['native']['large'] = 'x' * 4096
    with pytest.raises(TransactionError, match='storage budget'):
        await store.attach_task(third.issuance, remote_task_id='third', envelope=oversized)
    assert (await store.lookup(third.record.binding)).phase == 'ISSUING'
    assert not (await other.reserve(third.record.binding, submission(origin=(third.record.binding).origin))).fresh
    with pytest.raises(MCPTaskConflict, match='active capacity'):
        await other.reserve(binding(operation_id='replacement'), submission())


@pytest.mark.asyncio
@pytest.mark.parametrize('column,value,authenticated', [
    ('handle', 'changed', False), ('operation_digest', 'd' * 64, False),
    ('binding_digest', 'd' * 64, False), ('remote_digest', 'd' * 64, False),
    ('revision', 200, False), ('phase', 'SETTLED', False),
    ('claim_hash', 'd' * 64, False), ('payload_limit', 5000, False), ('payload', b'damaged', False),
    ('operation_digest', 'd' * 64, True), ('binding_digest', 'd' * 64, True),
    ('remote_digest', 'd' * 64, True),
])
async def test_every_retained_routing_or_cas_column_is_authenticated_before_new_admission(custody, column, value, authenticated):
    store = custody.store
    reserved = await store.reserve(binding(), submission())
    if column != 'claim_hash':
        await attach(store, reserved.issuance)
    # A live ISSUING hash remains structurally valid, so its change must fail
    # authentication rather than the ATTACHED phase's null-hash invariant.
    ciphertext = None
    if authenticated:
        columns = 'scope,handle,operation_digest,binding_digest,remote_digest,revision,phase,claim_hash,payload_limit,payload,submission_limit,submission_digest'
        row = list(await store.database.fetchone(f'SELECT {columns} FROM mcp_task_records'))
        cipher = AEADCipher(store._root)
        raw = cipher.decrypt(bytes(row[9]), aad=store._aad((*row[:9], *row[10:12])))
        row[columns.split(',').index(column)] = value
        ciphertext = cipher.encrypt(raw, aad=store._aad((*row[:9], *row[10:12])))
        assert cipher.decrypt(ciphertext, aad=store._aad((*row[:9], *row[10:12]))) == raw
    # Column names are the parametrized constant allowlist above, never input.
    if authenticated:
        await store.database.execute(f'UPDATE mcp_task_records SET {column}=?,payload=?', (value, ciphertext))
    else:
        await store.database.execute(f'UPDATE mcp_task_records SET {column}=?', (value,))
    match = 'custody indexes are invalid' if authenticated else None
    with pytest.raises((MCPTaskIntegrityError, TransactionError), match=match):
        await store.reserve(binding(operation_id='new'), submission())
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (1,)
    with pytest.raises(MCPTaskIntegrityError, match=match):
        await store.list_remote_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['envelope', 'missing', 'key'])
async def test_lost_original_envelope_or_key_is_never_silently_replaced(custody, failure):
    store = custody.store
    await store.reserve(binding(), submission())
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
            await store.reserve(binding(operation_id='new'), submission())
        assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_keys') == (0,)


@pytest.mark.asyncio
async def test_actual_key_rotation_rewraps_task_root_and_preserves_remote_custody(custody):
    store = custody.store
    reserved = await store.reserve(binding(), submission())
    attached = await attach(store, reserved.issuance)
    before = await store.database.fetchall('SELECT * FROM mcp_task_records')
    wrapped_before = await store.database.fetchone('SELECT wrapped_key FROM mcp_task_keys')
    status = await rotate_task_root(custody)
    assert status.status == RotationStatus.COMPLETED and status.records_processed == 1
    assert await store.database.fetchall('SELECT * FROM mcp_task_records') == before
    assert await store.database.fetchone('SELECT wrapped_key FROM mcp_task_keys') != wrapped_before
    restarted = await custody.reopen()
    assert await restarted.read(binding(), attached.handle) == attached
    assert await restarted.open_submission(attached) == submission()
    assert not (await restarted.reserve(binding(), submission())).fresh


@pytest.mark.asyncio
async def test_ambient_transactions_cannot_publish_issuance_and_close_preserves_primary(custody):
    store = custody.store
    async with store.database.transaction():
        with pytest.raises(MCPTaskConflict, match='own durable transaction'):
            await store.reserve(binding(), submission())
        with pytest.raises(MCPTaskConflict, match='own durable transaction'):
            await store.initialize()
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (0,)
    await store.close()
    assert await store.database.fetchone('SELECT 1') == (1,)
    with pytest.raises(RuntimeError, match='not initialized'):
        await store.reserve(binding(), submission())
    await store.initialize()
    assert (await store.reserve(binding(), submission())).fresh


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['reserve', 'attach'])
async def test_native_commit_ack_loss_preserves_uncertainty_without_reissuing(custody, monkeypatch, operation):
    store = custody.store
    issuance = (await store.reserve(binding(), submission())).issuance if operation == 'attach' else None
    with lost_commit_ack(monkeypatch, store) as originals:
        # The real driver commits the real SQL before a real descriptor read
        # fails at its acknowledgment boundary. No commit outcome is fabricated.
        with pytest.raises(BaseException) as caught:
            if operation == 'reserve':
                await store.reserve(binding(), submission())
            else:
                await attach(store, issuance)
        assert len(originals) == 1 and execution_commit_outcome(caught.value) == 'unknown'
        assert any(error is originals[0] for error in execution_evidence_carriers(caught.value)), (
            'The native commit failure lost its original injected source')
    restarted = await custody.reopen()
    recovered = await restarted.lookup(binding())
    assert recovered is not None and recovered.phase == ('ISSUING' if operation == 'reserve' else 'ATTACHED')
    assert not (await restarted.reserve(binding(), submission())).fresh
    if operation == 'attach':
        with pytest.raises(MCPTaskConflict):
            await attach(restarted, issuance)
