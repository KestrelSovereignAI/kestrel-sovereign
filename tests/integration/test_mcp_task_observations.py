"""Native detailed observations retain creation custody and confirmed outcomes."""

import asyncio
from dataclasses import replace
import json

import pytest

from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_evidence_carriers
from kestrel_sovereign.mcp_task_store import MCPTaskConflict
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.shared.mcp_task_custody import custody as custody, binding, claim, attach, lost_commit_ack


def observation(status='working', **changes):
    value = {**claim()['native'], 'resultType': 'complete', 'status': status}
    if status == 'input_required':
        value['inputRequests'] = {'private-key': {'method': 'elicitation/create', 'params': {
            'mode': 'form', 'message': 'Private input?', 'requestedSchema': {'type': 'object'}}}}
    elif status == 'completed':
        value['result'] = {'resultType': 'complete', 'content': [], 'isError': True,
            'structuredContent': {'private': 'private-terminal-content'}}
    elif status == 'failed':
        value['error'] = {'code': -32603, 'message': 'private-protocol-error'}
    value.update(changes)
    return value


async def attached(store):
    reserved = await store.reserve(binding())
    return await attach(store, reserved.issuance)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['working', 'input_required'])
async def test_nonterminal_observation_preserves_original_claim_across_connections(custody, status):
    store = await custody.reopen(max_active=1)
    original = await attached(store)
    supplied = json.loads(original._payload)
    supplied['envelope']['native']['vendor.example/private'] = {'marker': 'forged-reader-payload'}
    current = await store.observe(replace(original, _payload=json.dumps(supplied).encode()), observation(status))
    assert current.phase == 'ATTACHED' and current.revision == original.revision + 1
    assert current.envelope['native'] == original.envelope['native']
    assert current.observation == observation(status)
    detached = current.observation
    detached['status'] = 'cancelled'
    assert current.observation['status'] == status
    restarted = await custody.reopen(max_active=1, max_records=1, max_payload_bytes=1)
    assert await restarted.read(binding(), original.handle) == current
    assert await restarted.list_attached() == (current,)
    with pytest.raises(MCPTaskConflict, match='capacity'):
        await restarted.reserve(binding(operation_id='unconfirmed-replacement'))


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['completed', 'failed', 'cancelled'])
async def test_confirmed_terminal_observation_releases_active_slot_and_keeps_remote_consumed(custody, status):
    store = await custody.reopen(max_active=1)
    original = await attached(store)
    result = await store.observe(original, observation(status))
    assert result.phase == 'SETTLED' and result.envelope['native'] == claim()['native']
    assert result.observation == observation(status)
    restarted = await custody.reopen(max_active=1)
    assert await restarted.list_attached() == ()
    assert await restarted.read(binding(), result.handle) == result
    assert not (await restarted.reserve(binding())).fresh
    fresh = await restarted.reserve(binding(operation_id='other', principal_id='other', session_id='other'))
    with pytest.raises(MCPTaskConflict, match='remote identity'):
        await attach(restarted, fresh.issuance)
    rows = repr(await store.database.fetchall('SELECT * FROM mcp_task_records'))
    assert 'private-terminal-content' not in rows and 'private-protocol-error' not in rows


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [
    {'taskId': 'another-task'}, {'resultType': 'task'}, {'status': 'invented'},
    {'status': 'completed', 'result': None}, {'status': 'failed', 'error': []},
    {'status': 'input_required', 'inputRequests': None},
    {'status': []},
])
async def test_invalid_native_observation_cannot_change_current_custody(custody, changes):
    store = custody.store
    original = await attached(store)
    with pytest.raises(MCPTaskConflict):
        await store.observe(replace(original, revision=True), observation('completed'))
    with pytest.raises(TransactionError):
        await store.observe(original, observation(**changes))
    assert await store.read(binding(), original.handle) == original


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [
    ('principal_id', 'other'), ('session_id', 'other'), ('operation_id', 'other'),
    ('connection_id', 'd' * 64), ('authorization_id', 'd' * 64), ('contract_digest', 'd' * 64),
])
async def test_reader_labels_cannot_retarget_an_observation(custody, field, value):
    store = custody.store
    original = await attached(store)
    foreign = replace(original, binding=replace(original.binding, **{field: value}))
    with pytest.raises(MCPTaskConflict):
        await store.observe(foreign, observation('completed'))
    assert await store.read(binding(), original.handle) == original


@pytest.mark.asyncio
async def test_terminal_or_stale_revision_never_regresses_or_overwrites_the_original_result(custody):
    store = custody.store
    original = await attached(store)
    pending = await store.observe(original, observation())
    with pytest.raises(MCPTaskConflict):
        await store.observe(original, observation('completed'))
    terminal = await store.observe(pending, observation('completed'))
    for candidate in (pending, terminal, replace(terminal, phase='ATTACHED')):
        with pytest.raises(MCPTaskConflict):
            await store.observe(candidate, observation('cancelled'))
    assert await store.read(binding(), original.handle) == terminal


@pytest.mark.asyncio
async def test_two_native_observers_cannot_overwrite_a_terminal_winner(custody):
    first, second = custody.store, await custody.reopen()
    original = await attached(first)
    outcomes = await asyncio.gather(first.observe(original, observation('completed')),
        second.observe(original, observation('cancelled')), return_exceptions=True)
    assert sum(isinstance(item, MCPTaskConflict) for item in outcomes) == 1
    winner, = [item for item in outcomes if not isinstance(item, BaseException)]
    assert winner.phase == 'SETTLED' and winner.observation['status'] in {'completed','cancelled'}
    assert await second.read(binding(), original.handle) == winner


@pytest.mark.asyncio
async def test_oversize_observation_does_not_release_capacity_and_lowered_budgets_preserve_progress(custody):
    store = await custody.reopen(max_active=1)
    original = await attached(store)
    oversized = observation('completed', result={'private': 'x' * 4096})
    with pytest.raises(TransactionError, match='storage budget'):
        await store.observe(original, oversized)
    assert await store.read(binding(), original.handle) == original
    with pytest.raises(MCPTaskConflict, match='active capacity'):
        await store.reserve(binding(operation_id='new'))
    lowered = await custody.reopen(max_active=1, max_records=1, max_payload_bytes=1)
    final = await lowered.observe(original, observation('completed'))
    assert final.phase == 'SETTLED'


@pytest.mark.asyncio
async def test_original_commit_ack_loss_recovers_terminal_result_and_actual_capacity(custody, monkeypatch):
    store = await custody.reopen(max_active=1)
    original = await attached(store)
    with lost_commit_ack(monkeypatch, store) as originals:
        with pytest.raises(BaseException) as caught:
            await store.observe(original, observation('completed'))
        assert len(originals) == 1 and execution_commit_outcome(caught.value) == 'unknown'
        assert any(error is originals[0] for error in execution_evidence_carriers(caught.value))
    restarted = await custody.reopen(max_active=1)
    recovered = await restarted.read(binding(), original.handle)
    assert recovered.phase == 'SETTLED' and recovered.observation == observation('completed')
    assert recovered.envelope['native'] == original.envelope['native']
    assert (await restarted.reserve(binding(operation_id='next'))).fresh
    with pytest.raises(MCPTaskConflict):
        await restarted.observe(original, observation('completed'))


@pytest.mark.asyncio
async def test_attach_cannot_smuggle_observation_or_unbound_initial_settlement(custody):
    store = custody.store
    reserved = await store.reserve(binding())
    with pytest.raises(ValueError):
        await store.attach_task(reserved.issuance, remote_task_id='remote-private-task',
            envelope={**claim(), 'observation': observation()})
    with pytest.raises(ValueError):
        await store.settle_initial(reserved.issuance, disposition='SETTLED',
            envelope={'version': 1, 'native': {}, 'observation': observation('completed')})
    assert await store.lookup(binding()) == reserved.record
