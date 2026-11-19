"""Lifetime Task input identity and pre-wire mutation receipts on native SQL."""

import asyncio
from dataclasses import replace
import json

import pytest

from kestrel_sdk.security.aead import AEADCipher
from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_evidence_carriers
from kestrel_sovereign.mcp_task_store import MCPTaskConflict, MCPTaskIntegrityError
from kestrel_sovereign.security.key_rotation import RotationStatus
from kestrel_sovereign.storage.db.interface import TransactionError
from kestrel_sovereign.mcp_task_submission import MCPTaskInteractiveOrigin
from tests.shared.mcp_task_custody import (
    submission,
    custody as custody, binding, claim, attach, attached, observation, lost_commit_ack, rotate_task_root, paused_task_write,
)
from tests.integration.test_key_rotation_recovery import reject_writes


def inputs(**requests):
    return observation('input_required', inputRequests=requests)


def response(value='private-answer'):
    return {'resultType': 'complete', 'action': 'accept', 'content': {'private': value}}


async def waiting(store, **requests):
    return await store.observe(await attached(store), inputs(**requests))


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['', 'é/任務/🔒', ' opaque\u0000input\r\n'])
async def test_input_identity_survives_gaps_and_connections_and_canonical_type_changes(custody, key):
    store = custody.store
    request = {'method': 'roots/list', 'private-value': True}
    current = await waiting(store, **{key: request})
    assert current.pending_inputs == {key: request}
    detached = current.pending_inputs
    detached[key]['private-value'] = 'changed-reader'
    assert current.pending_inputs[key]['private-value'] is True
    current = await store.observe(current, inputs(**{key: dict(reversed(tuple(request.items())))}))
    assert current.control['inputs'] == {key: request}
    current = await store.observe(current, observation())
    reopened = await custody.reopen()
    assert (await reopened.read(binding(), current.handle)).control['inputs'] == {key: request}
    for changed in ({'method': 'roots/list', 'private-value': 1}, {'method': 'sampling/createMessage'}):
        with pytest.raises(MCPTaskConflict, match='reused'):
            await reopened.observe(current, inputs(**{key: changed}))
        assert await reopened.read(binding(), current.handle) == current
    current = await reopened.observe(current, inputs(**{key: request}))
    assert current.pending_inputs == {key: request}


@pytest.mark.asyncio
async def test_partial_inputs_are_durably_consumed_before_wire_and_acks_preserve_newer_state(custody):
    store = custody.store
    request = {'method': 'roots/list'}
    current = await waiting(store, first=request, second=request)
    first = await store.reserve_mutation(current, operation_id='first-leaf', kind='update',
        input_responses={'first': response()})
    restarted = await custody.reopen()
    # An independent connection sees consumption BEFORE any native send.
    recovered = await restarted.read(binding(), current.handle)
    assert recovered == first.record and recovered.pending_inputs == {'second': request}
    assert recovered.control['mutations']['first-leaf']['intent'] == {'method': 'tasks/update',
        'params': {'taskId': 'remote-private-task', 'inputResponses': {'first': response()}}}
    assert first.issuance.secret not in repr(recovered) + recovered._payload.decode()
    duplicate = await restarted.reserve_mutation(current, operation_id='first-leaf', kind='update',
        input_responses={'first': response()})
    assert not duplicate.fresh and duplicate.issuance is None
    current = await store.observe(recovered, inputs(first=request, second=request))
    assert current.pending_inputs == {'second': request}
    second = await store.reserve_mutation(current, operation_id='second-leaf', kind='update',
        input_responses={'second': response('other')})
    # Reverse acknowledgement order, with terminal observation in between.
    current = await store.acknowledge_mutation(second.issuance, {'resultType': 'complete'})
    terminal = await store.observe(current, observation('completed'))
    current = await restarted.acknowledge_mutation(first.issuance, {'resultType': 'complete', '_meta': {'private': 'ack'}})
    assert current.phase == 'SETTLED' and current.observation == terminal.observation
    assert current.envelope['native'] == terminal.envelope['native']
    assert current.pending_inputs == {} and set(current.control['inputs']) == {'first', 'second'}
    assert all(r['claim_hash'] is None and r['acknowledgement'] is not None
        for r in current.control['mutations'].values())
    with pytest.raises(MCPTaskConflict):
        await store.acknowledge_mutation(first.issuance, {'resultType': 'complete'})
    assert not (await store.reserve_mutation(current, operation_id='first-leaf', kind='update',
        input_responses={'first': response()})).fresh
    rows = repr(await store.database.fetchall('SELECT * FROM mcp_task_records'))
    assert 'private-answer' not in rows and 'first-leaf' not in rows and 'private-value' not in rows


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['foreign', 'stale', 'forged', 'unknown', 'consumed', 'invalid', 'initial-leaf', 'changed-intent', 'initial-over-mutation'])
async def test_original_binding_revision_and_consumption_govern_new_mutation_custody(custody, case):
    store = custody.store
    current = await waiting(store, key={'method': 'roots/list'})
    if case == 'invalid':
        for kind, responses in [('unknown', None), ('cancel', {}), ('update', {}), ('update', {'key': []})]:
            with pytest.raises(ValueError):
                await store.reserve_mutation(current, operation_id='invalid', kind=kind, input_responses=responses)
        assert await store.read(binding(), current.handle) == current
        return
    if case == 'foreign':
        current = replace(current, binding=binding(origin=MCPTaskInteractiveOrigin('d' * 64, 'private-session')))
    elif case == 'stale':
        await store.observe(current, inputs(key={'method': 'roots/list'}))
    elif case == 'forged':
        raw = json.loads(current._payload)
        raw['envelope']['observation']['inputRequests']['invented'] = {'method': 'roots/list'}
        current = replace(current, _payload=json.dumps(raw).encode())
    if case in {'consumed', 'changed-intent', 'initial-over-mutation'}:
        admitted = await store.reserve_mutation(current, operation_id='prior', kind='update', input_responses={'key': response(True)})
        current = admitted.record
    if case == 'initial-over-mutation':
        with pytest.raises(MCPTaskConflict, match='already owns a mutation'):
            await store.reserve(binding(operation_id='prior'), submission())
        assert await store.read(binding(), current.handle) == current
        return
    operation = binding().operation_id if case == 'initial-leaf' else ('prior' if case == 'changed-intent' else 'new-leaf')
    key = 'unknown' if case in {'unknown', 'forged'} else 'key'
    with pytest.raises(MCPTaskConflict):
        await store.reserve_mutation(current, operation_id=operation, kind='update', input_responses={key: response(1)})


@pytest.mark.asyncio
@pytest.mark.parametrize('same_task', [True, False])
async def test_native_writers_cannot_reuse_inputs_or_a_governed_operation_across_tasks(custody, same_task):
    first, second = custody.store, await custody.reopen()
    current = await waiting(first, key={'method': 'roots/list'})
    if same_task:
        other = current
        second_operation = 'competing-leaf'
    else:
        fresh = await first.reserve(binding(operation_id='other-original'), submission())
        other = await attach(first, fresh.issuance, 'other-remote')
        other = await first.observe(other, observation('input_required', taskId='other-remote', inputRequests={'key': {'method': 'roots/list'}}))
        second_operation = 'one-leaf'
    results = await asyncio.gather(
        first.reserve_mutation(current, operation_id='one-leaf', kind='update', input_responses={'key': response()}),
        second.reserve_mutation(other, operation_id=second_operation, kind='update', input_responses={'key': response()}),
        return_exceptions=True)
    assert sum(isinstance(value, MCPTaskConflict) for value in results) == 1
    winner, = [value for value in results if not isinstance(value, BaseException)]
    assert winner.fresh
    finishes = await asyncio.gather(first.acknowledge_mutation(winner.issuance, {'resultType': 'complete'}),
        second.acknowledge_mutation(winner.issuance, {'resultType': 'complete'}), return_exceptions=True)
    assert sum(isinstance(value, MCPTaskConflict) for value in finishes) == 1


@pytest.mark.asyncio
async def test_cancel_ack_is_only_intent_and_never_releases_active_capacity(custody):
    store = await custody.reopen(max_active=1)
    original = await attached(store)
    cancel = await store.reserve_mutation(original, operation_id='cancel-leaf', kind='cancel')
    assert cancel.record.control['mutations']['cancel-leaf']['intent'] == {
        'method': 'tasks/cancel', 'params': {'taskId': original.remote_task_id}}
    acked = await store.acknowledge_mutation(cancel.issuance, {'resultType': 'complete'})
    assert acked.phase == 'ATTACHED' and acked.observation is None
    with pytest.raises(MCPTaskConflict, match='active capacity'):
        await store.reserve(binding(operation_id='replacement'), submission())
    assert not (await store.reserve_mutation(original, operation_id='cancel-leaf', kind='cancel')).fresh
    new_intent = await store.reserve_mutation(acked, operation_id='explicit-new-cancel', kind='cancel')
    final = await store.observe(new_intent.record, observation('cancelled'))
    with pytest.raises(MCPTaskConflict):
        await store.reserve_mutation(final, operation_id='terminal-new-cancel', kind='cancel')
    assert not (await store.reserve_mutation(original, operation_id='cancel-leaf', kind='cancel')).fresh
    assert (await store.reserve(binding(operation_id='replacement'), submission())).fresh


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['reserve', 'ack'])
async def test_native_commit_ack_loss_keeps_original_error_and_never_recovers_mutation_secret(custody, monkeypatch, stage):
    store = custody.store
    current = await waiting(store, key={'method': 'roots/list'})
    if stage == 'ack':
        issuance = (await store.reserve_mutation(current, operation_id='lost-ack', kind='update', input_responses={'key': response()})).issuance
    with lost_commit_ack(monkeypatch, store) as originals:
        with pytest.raises(BaseException) as caught:
            if stage == 'reserve':
                await store.reserve_mutation(current, operation_id='lost-ack', kind='update', input_responses={'key': response()})
            else:
                await store.acknowledge_mutation(issuance, {'resultType': 'complete'})
        assert len(originals) == 1 and execution_commit_outcome(caught.value) == 'unknown'
        assert any(error is originals[0] for error in execution_evidence_carriers(caught.value))
    restarted = await custody.reopen()
    recovered = await restarted.read(binding(), current.handle)
    receipt = recovered.control['mutations']['lost-ack']
    assert recovered.pending_inputs == {}
    assert (receipt['acknowledgement'] is not None) == (stage == 'ack')
    assert not (await restarted.reserve_mutation(current, operation_id='lost-ack', kind='update', input_responses={'key': response()})).fresh
    with pytest.raises(MCPTaskConflict):
        await restarted.reserve_mutation(recovered, operation_id='different-leaf', kind='update', input_responses={'key': response()})


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['observation-budget', 'reservation-budget', 'ack-budget', 'ignored', 'raised', 'ambient', 'snapshot', 'snapshot-observe', 'snapshot-ack'])
async def test_custody_refusals_keep_original_history_and_snapshot_before_native_wait(custody, monkeypatch, case):
    store = custody.store
    current = await waiting(store, key={'method': 'roots/list'})
    if case.startswith('snapshot'):
        if case == 'snapshot-ack':
            reserved = await store.reserve_mutation(current, operation_id='snapshot', kind='cancel')
        submitted = {'key': response()}
        native_observation = inputs(next={'method': 'roots/list'})
        native_ack = {'resultType': 'complete', '_meta': {'private': 'original-ack'}}
        if case == 'snapshot-observe':
            operation = store.observe(current, native_observation)
        elif case == 'snapshot-ack':
            operation = store.acknowledge_mutation(reserved.issuance, native_ack)
        else:
            operation = store.reserve_mutation(current, operation_id='snapshot', kind='update', input_responses=submitted)
        async with paused_task_write(store, monkeypatch, operation) as held:
            await held.entered.wait()
            submitted['key']['content']['private'] = 'retargeted-after-validation'
            native_observation['inputRequests']['next']['method'] = 'sampling/createMessage'
            native_ack['_meta']['private'] = 'retargeted-ack'
        result = held.result
        if case == 'snapshot-observe':
            assert result.control['inputs']['next'] == {'method': 'roots/list'}
        elif case == 'snapshot-ack':
            assert result.control['mutations']['snapshot']['acknowledgement']['_meta'] == {'private': 'original-ack'}
        else:
            assert result.record.control['mutations']['snapshot']['intent']['params']['inputResponses'] == {'key': response()}
        return
    if case == 'ambient':
        async with store.database.transaction():
            with pytest.raises(MCPTaskConflict, match='own durable transaction'):
                await store.reserve_mutation(current, operation_id='ambient', kind='cancel')
        return
    if case in {'ignored', 'raised'}:
        await reject_writes(store.database, 'mcp_task_records', 'raise' if case == 'raised' else 'ignored')
        try:
            with pytest.raises((MCPTaskConflict, TransactionError)):
                await store.reserve_mutation(current, operation_id='rejected', kind='cancel')
        finally:
            sql = 'DROP TRIGGER rotation_unavailable' + (' ON mcp_task_records' if store.is_postgres else '')
            await store.database.execute(sql)
    elif case == 'observation-budget':
        with pytest.raises(TransactionError, match='storage budget'):
            await store.observe(current, inputs(other={'method': 'roots/list', 'large': 'x' * 4096}))
    elif case == 'reservation-budget':
        with pytest.raises(TransactionError, match='storage budget'):
            await store.reserve_mutation(current, operation_id='oversize', kind='update', input_responses={'key': response('x' * 4096)})
    else:
        issuance = (await store.reserve_mutation(current, operation_id='ack-budget', kind='cancel')).issuance
        current = issuance.record
        with pytest.raises(TransactionError, match='storage budget'):
            await store.acknowledge_mutation(issuance, {'resultType': 'complete', '_meta': {'large': 'x' * 4096}})
    assert await store.read(binding(), current.handle) == current
    lowered = await custody.reopen(max_payload_bytes=1, max_records=1, max_active=1)
    advanced = await lowered.observe(current, observation('completed'))
    assert advanced.phase == 'SETTLED' and advanced.control == current.control


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['secret', 'intent', 'duplicate-key', 'bad-ack', 'bad-hash', 'schema1', 'rotation'])
async def test_original_mutation_integrity_schema_boundary_and_key_rotation(custody, case):
    store = custody.store
    current = await waiting(store, key={'method': 'roots/list'})
    reserved = await store.reserve_mutation(current, operation_id='original-mutation', kind='update', input_responses={'key': response()})
    if case == 'secret':
        with pytest.raises(MCPTaskConflict):
            await store.acknowledge_mutation(replace(reserved.issuance, secret='x' * 43), {'resultType': 'complete'})
    elif case == 'schema1':
        row = list(await store.database.fetchone('SELECT scope,handle,operation_digest,binding_digest,remote_digest,revision,phase,claim_hash,payload_limit,payload,submission_limit,submission_digest FROM mcp_task_records'))
        raw = json.loads(AEADCipher(store._root).decrypt(bytes(row[9]), aad=store._aad((*row[:9], *row[10:12]))))
        del raw['envelope']['control']
        ciphertext = AEADCipher(store._root).encrypt(json.dumps(raw).encode(), aad=store._aad((*row[:9], *row[10:12])))
        await store.database.execute('UPDATE mcp_task_records SET payload=?', (ciphertext,))
        before = await store.database.fetchall('SELECT * FROM mcp_task_records')
        await store.database.execute('UPDATE mcp_task_schema SET revision=1')
        with pytest.raises(MCPTaskIntegrityError, match='Unsupported MCP Task schema'):
            await custody.reopen()
        assert await store.database.fetchall('SELECT * FROM mcp_task_records') == before
        assert await store.database.fetchone('SELECT revision FROM mcp_task_schema') == (1,)
    elif case == 'rotation':
        status = await rotate_task_root(custody)
        assert status.status is RotationStatus.COMPLETED
        restarted = await custody.reopen()
        assert await restarted.read(binding(), reserved.record.handle) == reserved.record
        assert not (await restarted.reserve_mutation(current, operation_id='original-mutation', kind='update', input_responses={'key': response()})).fresh
    else:
        row = list(await store.database.fetchone('SELECT scope,handle,operation_digest,binding_digest,remote_digest,revision,phase,claim_hash,payload_limit,payload,submission_limit,submission_digest FROM mcp_task_records'))
        raw = json.loads(AEADCipher(store._root).decrypt(bytes(row[9]), aad=store._aad((*row[:9], *row[10:12]))))
        control = raw['envelope']['control']
        receipt = control['mutations']['original-mutation']
        if case == 'intent':
            receipt['intent']['params']['taskId'] = 'retargeted-task'
        elif case == 'duplicate-key':
            control['mutations']['other-mutation'] = json.loads(json.dumps(receipt))
        elif case == 'bad-ack':
            receipt['acknowledgement'] = {'resultType': 'complete'}
        else:
            receipt['claim_hash'] = 'invalid'
        ciphertext = AEADCipher(store._root).encrypt(json.dumps(raw).encode(), aad=store._aad((*row[:9], *row[10:12])))
        await store.database.execute('UPDATE mcp_task_records SET payload=? WHERE scope=? AND handle=?', (ciphertext, store.scope, current.handle))
        with pytest.raises(MCPTaskIntegrityError):
            await store.read(binding(), current.handle)
        # The native transaction wraps ordinary integrity failures while
        # retaining their original cause; do not replace that contract.
        with pytest.raises(TransactionError, match='Invalid private MCP Task payload') as rejected:
            await store.reserve(binding(operation_id='cannot-hide-corrupt-history'), submission())
        assert isinstance(rejected.value.__cause__, MCPTaskIntegrityError)


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['attach', 'settle'])
async def test_initial_json_cannot_change_across_its_original_native_lock_wait(custody, monkeypatch, outcome):
    store = custody.store
    issuance = (await store.reserve(binding(), submission())).issuance
    envelope = claim() if outcome == 'attach' else {'version': 1, 'native': {'resultType': 'complete', 'content': []}}
    original = json.loads(json.dumps(envelope))
    operation = (store.attach_task(issuance, remote_task_id='remote-private-task', envelope=envelope)
        if outcome == 'attach' else store.settle_initial(issuance, disposition='SETTLED', envelope=envelope))
    async with paused_task_write(store, monkeypatch, operation) as held:
        await held.entered.wait()
        envelope['native']['unvalidated-after-wait'] = {'private': 'retargeted'}
    assert held.result.envelope == original, 'Initial custody retained JSON changed after native validation and before commit'
