"""Native database continuation encryption, durable consumption and refusals."""

import asyncio
import base64
from contextlib import AsyncExitStack
from dataclasses import replace

import pytest
import pytest_asyncio

from kestrel_sovereign.mcp_input_store import (
    MCPInputBinding, MCPInputClaim, MCPInputConflict, MCPInputIntegrityError, MCPInputStore,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest_asyncio.fixture(params=['sqlite', pytest.param('postgres', marks=pytest.mark.native_postgres)])
async def custody(tmp_path, monkeypatch, request):
    key = tmp_path / 'owned-input-data-key'
    key.write_text(base64.urlsafe_b64encode(b'i' * 32).decode())
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
            schema = await cleanup.enter_async_context(disposable_postgres_schema(admin, 'mcp_input_test'))
            async def database():
                return await AsyncDatabase.postgres(with_search_path(dsn, schema),
                    min_pool_size=1, max_pool_size=2, schema_initializer=no_core_schema)
        else:
            async def database():
                return await AsyncDatabase.sqlite(str(tmp_path / 'input.db'), schema_initializer=no_core_schema)
        async def reopen(*, max_pending=2, max_payload_bytes=4096, agent_did='did:test:input-owner'):
            db = await database()
            cleanup.push_async_callback(db.close)
            store = MCPInputStore(db, agent_did=agent_did,
                max_pending=max_pending, max_payload_bytes=max_payload_bytes)
            await store.initialize()
            return store
        yield await reopen(), reopen, key


def binding(**changes):
    return replace(MCPInputBinding('did:test:input-owner', 'operator-private', 'private-session',
        'original-dispatch-claim', 'original-connection', 'a' * 64, 'tools/call'), **changes)


def checkpoint():
    return {'requestState': ' opaque\u0000signed-state\n', 'arguments': {'private': 'secret-input'},
        'inputRequests': {'form': {'message': 'Private city?'}}}


async def create(store, **changes):
    return await store.create(binding(**changes), checkpoint(), expires_at=1000, now=100)


@pytest.mark.asyncio
async def test_continuation_chain_reuses_one_reservation_and_keeps_uncertain_claims_counted(custody):
    _, reopen, _ = custody
    store = await reopen(max_pending=1)
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    with pytest.raises(MCPInputConflict, match='capacity'):
        await create(store, operation_id='other-operation')
    child = await store.create(binding(operation_id='successor-operation'), checkpoint(),
        now=102, expires_at=1000, predecessor=claim)
    assert child.reservation_id == first.request_id == first.reservation_id
    assert child.predecessor_id == first.request_id and first.predecessor_id is None
    restored = await reopen(max_pending=1)
    assert (await restored.read(first.binding, first.request_id)).state == 'RESPONDING'
    with pytest.raises(MCPInputConflict, match='capacity'):
        await create(restored, operation_id='other-operation')
    await store.settle(claim, state='RESOLVED', continuation={'next_request_id': child.request_id})
    with pytest.raises(MCPInputConflict, match='capacity'):
        await create(restored, operation_id='other-operation')
    assert await restored.read(child.binding, child.request_id) == child


@pytest.mark.asyncio
async def test_two_native_writers_cannot_fork_one_response_reservation(custody):
    store, reopen, _ = custody
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    second = await reopen()
    outcomes = await asyncio.gather(*[
        owner.create(binding(operation_id='child-' + str(index)), checkpoint(),
            now=102, expires_at=1000, predecessor=claim)
        for index, owner in enumerate((store, second))
    ], return_exceptions=True)
    assert sum(isinstance(value, MCPInputConflict) for value in outcomes) == 1
    children = [value for value in outcomes if not isinstance(value, BaseException)]
    assert len(children) == 1
    assert children[0].reservation_id == first.reservation_id
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (2,)


@pytest.mark.asyncio
async def test_chain_listing_exposes_successor_before_its_unsettled_parent(custody, monkeypatch):
    from uuid import UUID
    from kestrel_sovereign import mcp_input_store

    _, reopen, _ = custody
    store = await reopen(max_pending=1)
    ids = iter((UUID(int=1), UUID(int=2)))
    monkeypatch.setattr(mcp_input_store, 'uuid4', lambda: next(ids))
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    child = await store.create(binding(operation_id='successor-operation'), checkpoint(),
        now=102, expires_at=1000, predecessor=claim)
    rows = await store.list_for(first.binding.principal_id, first.binding.session_id, now=103)
    assert rows == (child,), 'A consumed predecessor hid its confirmed successor at the chain capacity limit'


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['secret', 'principal', 'settled'])
async def test_successor_requires_the_exact_original_live_response_claim(custody, fault):
    store, _, _ = custody
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    successor_binding = binding(operation_id='successor-operation')
    if fault == 'secret':
        claim = replace(claim, secret='x' * 43)
    elif fault == 'principal':
        successor_binding = replace(successor_binding, principal_id='foreign-reader')
    else:
        await store.settle(claim, state='UNKNOWN')
    with pytest.raises(MCPInputConflict):
        await store.create(successor_binding, checkpoint(), now=102, expires_at=1000, predecessor=claim)
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (1,)


@pytest.mark.asyncio
async def test_descriptive_claim_cannot_change_the_authenticated_reservation(custody):
    store, _, _ = custody
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    description = replace(claim, record=replace(claim.record, reservation_id='forged-reservation'))
    child = await store.create(binding(operation_id='successor-operation'), checkpoint(),
        now=102, expires_at=1000, predecessor=description)
    assert child.reservation_id == first.reservation_id


@pytest.mark.asyncio
@pytest.mark.parametrize('column', ['reservation_id', 'predecessor_id'])
async def test_reservation_lineage_is_authenticated_with_its_checkpoint(custody, column):
    store, _, _ = custody
    first = await create(store)
    claim = await store.claim_response(first, {}, now=101)
    child = await store.create(binding(operation_id='successor-operation'), checkpoint(),
        now=102, expires_at=1000, predecessor=claim)
    other = await create(store, operation_id='other-reservation')
    assert column in {'reservation_id', 'predecessor_id'}
    await store.database.execute('UPDATE mcp_inputs SET ' + column + '=? WHERE request_id=?',
        (other.request_id, child.request_id))
    with pytest.raises(MCPInputIntegrityError):
        await store.read(child.binding, child.request_id)


@pytest.mark.asyncio
async def test_encrypted_checkpoint_and_exact_state_survive_new_native_connection(custody):
    store, reopen, _ = custody
    record = await create(store)
    assert record.state == 'WAITING' and record.revision == 0 and record.responses is None
    payload = record.payload
    payload['requestState'] = 'substitute'
    second = await reopen()
    recovered = await second.read(binding(), record.request_id)
    assert recovered == record and recovered.payload == checkpoint()
    rows = await store.database.fetchall('SELECT * FROM mcp_inputs')
    assert 'secret-input' not in repr(rows) and 'operator-private' not in repr(rows)
    assert 'signed-state' not in repr(rows) and 'Private city?' not in repr(record)
    assert 'operator-private' not in repr(record.binding)


@pytest.mark.asyncio
async def test_claim_is_single_use_across_two_independent_writers(custody):
    store, reopen, _ = custody
    record = await create(store)
    second = await reopen()
    results = await asyncio.gather(
        store.claim_response(record, {'form': {'action': 'accept', 'content': {'city': 'private-answer'}}}, now=101),
        second.claim_response(record, {'form': {'action': 'decline'}}, now=101), return_exceptions=True)
    claims = [r for r in results if isinstance(r, MCPInputClaim)]
    assert len(claims) == 1
    assert sum(isinstance(r, MCPInputConflict) for r in results) == 1
    claim = claims[0]
    assert claim.record.state == 'RESPONDING' and claim.record.revision == 1
    assert claim.record.payload == checkpoint()
    assert claim.record.responses in [
        {'form': {'action': 'accept', 'content': {'city': 'private-answer'}}}, {'form': {'action': 'decline'}}]
    assert 'private-answer' not in repr(claim)
    restarted = await reopen()
    with pytest.raises(MCPInputConflict, match='consumed'):
        await restarted.claim_response(record, {}, now=102)


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['RESOLVED', 'NON_RESUMABLE', 'UNKNOWN'])
async def test_original_claim_settles_once_and_never_reopens(custody, state):
    store, reopen, _ = custody
    record = await create(store)
    claim = await store.claim_response(record, {'form': {'action': 'decline'}}, now=101)
    second = await reopen()
    settled = await second.settle(claim, state=state)
    assert settled.state == state and settled.revision == 2
    assert settled.payload == checkpoint() and settled.responses == {'form': {'action': 'decline'}}
    with pytest.raises(MCPInputConflict):
        await store.settle(claim, state='RESOLVED')
    with pytest.raises(MCPInputConflict):
        await second.claim_response(record, {}, now=102)
    with pytest.raises(MCPInputConflict, match='already has a checkpoint'):
        await create(second)


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [
    {'principal_id': 'other'}, {'session_id': 'other'}, {'operation_id': 'other'},
    {'connection_id': 'other'}, {'contract_digest': 'b' * 64}, {'method': 'resources/read'},
    {'agent_did': 'did:test:other'},
])
async def test_wrong_original_binding_cannot_read_or_claim(custody, changes):
    store, _, _ = custody
    record = await create(store)
    with pytest.raises(MCPInputConflict):
        await store.read(binding(**changes), record.request_id)
    with pytest.raises(MCPInputConflict):
        await store.claim_response(replace(record, binding=binding(**changes)), {}, now=101)
    assert (await store.read(binding(), record.request_id)).state == 'WAITING'


@pytest.mark.asyncio
async def test_snapshot_cannot_replace_original_checkpoint(custody):
    store, _, _ = custody
    record = await create(store)
    forged = replace(record, _payload=b'{"checkpoint":{"requestState":"forged"},"responses":null}')
    claim = await store.claim_response(forged, {}, now=101)
    assert claim.record.payload == checkpoint()
    forged_claim = replace(claim, record=replace(claim.record, _payload=forged._payload))
    assert (await store.settle(forged_claim, state='UNKNOWN')).payload == checkpoint()


@pytest.mark.asyncio
async def test_forged_claim_secret_does_not_settle_or_free_capacity(custody):
    store, _, _ = custody
    record = await create(store)
    claim = await store.claim_response(record, {}, now=101)
    with pytest.raises(MCPInputConflict, match='no longer current'):
        await store.settle(replace(claim, secret='x' * 43), state='RESOLVED')
    assert (await store.read(binding(), record.request_id)).state == 'RESPONDING'


@pytest.mark.asyncio
async def test_expired_checkpoint_is_readable_but_cannot_be_claimed(custody):
    store, _, _ = custody
    record = await create(store)
    with pytest.raises(MCPInputConflict, match='expired'):
        await store.claim_response(record, {}, now=1000)
    with pytest.raises(MCPInputConflict, match='consumed'):
        await store.claim_response(replace(record, expires_at=1001), {}, now=1000)
    assert (await store.read(binding(), record.request_id)).state == 'WAITING'


@pytest.mark.asyncio
async def test_native_capacity_is_serialized_and_stranded_claims_stay_counted(custody):
    store, reopen, _ = custody
    record = await create(store)
    await store.claim_response(record, {}, now=101)
    second = await reopen()
    results = await asyncio.gather(create(store, operation_id='next-a'),
        create(second, operation_id='next-b'), return_exceptions=True)
    assert sum(isinstance(r, MCPInputConflict) for r in results) == 1
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (2,)
    bounded = await reopen(max_pending=1)
    with pytest.raises(MCPInputConflict, match='capacity'):
        await bounded.create(binding(operation_id='after-expiry'), {}, expires_at=3000, now=2000)


@pytest.mark.asyncio
async def test_storage_budget_and_expired_create_refuse_before_insertion(custody):
    store, _, _ = custody
    with pytest.raises(ValueError, match='storage budget'):
        await store.create(binding(), {'private': 'x' * 4096}, expires_at=1000, now=100)
    with pytest.raises(MCPInputConflict, match='expired'):
        await store.create(binding(), {}, expires_at=100, now=100)
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (0,)


@pytest.mark.asyncio
async def test_original_operation_cannot_be_rebound_to_new_connection(custody):
    store, _, _ = custody
    await create(store)
    with pytest.raises(MCPInputConflict, match='already has a checkpoint'):
        await create(store, connection_id='replacement-connection')


@pytest.mark.asyncio
async def test_metadata_tampering_is_authenticated(custody):
    store, _, _ = custody
    record = await create(store)
    await store.database.execute('UPDATE mcp_inputs SET expires_at=expires_at+1 WHERE request_id=?', (record.request_id,))
    with pytest.raises(MCPInputIntegrityError, match='cannot be opened'):
        await store.read(binding(), record.request_id)


@pytest.mark.asyncio
@pytest.mark.parametrize('custody', ['sqlite'], indirect=True)
async def test_missing_original_key_is_not_replaced(custody):
    store, reopen, _ = custody
    await create(store)
    await store.database.execute('PRAGMA foreign_keys=OFF')
    await store.database.execute('DELETE FROM mcp_input_scopes')
    with pytest.raises(MCPInputIntegrityError, match='envelope is missing'):
        await reopen()


@pytest.mark.asyncio
async def test_rotated_host_key_fails_without_discarding_checkpoint(custody):
    store, reopen, key = custody
    await create(store)
    key.write_text(base64.urlsafe_b64encode(b'j' * 32).decode())
    with pytest.raises(MCPInputIntegrityError, match='encryption key is unavailable'):
        await reopen()
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (1,)


@pytest.mark.asyncio
async def test_private_reader_can_recover_binding_but_not_another_session(custody):
    store, reopen, _ = custody
    record = await create(store)
    restarted = await reopen()
    assert await restarted.list_for('operator-private', 'private-session', now=100) == (record,)
    assert await restarted.list_for('other', 'private-session', now=100) == ()
    assert await restarted.list_for('operator-private', 'other', now=100) == ()
    assert await restarted.list_for('operator-private', 'private-session', now=1000) == ()
    recovered = await restarted.read_for('operator-private', 'private-session', record.request_id)
    assert recovered.binding == binding() and recovered.payload == checkpoint()
    with pytest.raises(MCPInputConflict):
        await restarted.read_for('other', 'private-session', record.request_id)
    claim = await restarted.claim_response(record, {}, now=101)
    await restarted.settle(claim, state='RESOLVED')
    assert await restarted.list_for('operator-private', 'private-session', now=100) == ()


@pytest.mark.asyncio
async def test_machine_declines_have_durable_content_free_decisions(custody):
    store, reopen, _ = custody
    await store.audit_decisions('private-original-operation', 'private-state-digest',
        (('private-form-key', 'no_approver'), ('private-url-key', 'url_mode_declined'),
            ('', 'url_mode_declined')), now=100)
    restarted = await reopen()
    rows = await restarted.database.fetchall('SELECT operation_digest,input_digest,reason,decided_at FROM mcp_input_decisions ORDER BY reason')
    assert [(row[2], row[3]) for row in rows] == [('no_approver', 100), ('url_mode_declined', 100), ('url_mode_declined', 100)]
    assert all(len(row[0]) == len(row[1]) == 64 for row in rows)
    assert 'private' not in repr(rows)
    with pytest.raises(ValueError, match='audit decisions'):
        await store.audit_decisions('operation', 'digest', (('key', 'invented-allow'),), now=101)
    assert await store.database.fetchone('SELECT COUNT(*) FROM mcp_input_decisions') == (3,)


@pytest.mark.asyncio
async def test_another_agent_envelope_cannot_replace_original_recovery_root(custody):
    store, reopen, _ = custody
    await create(store)
    other = await reopen(agent_did='did:test:other-input-agent')
    envelope = await other.database.fetchone('SELECT wrapped_key FROM mcp_input_scopes WHERE scope=?', (other.scope,))
    await store.database.execute('UPDATE mcp_input_scopes SET wrapped_key=? WHERE scope=?',
        (envelope[0], store.scope))
    with pytest.raises(MCPInputIntegrityError, match='encryption key is unavailable'):
        await reopen()
