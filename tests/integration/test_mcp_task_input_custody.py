"""Joint credentials and input expiry on actual native SQLite/PostgreSQL stores."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
import pytest_asyncio

from kestrel_sovereign.mcp_input_store import MCPInputBinding, MCPInputConflict, MCPInputStore
from kestrel_sovereign.mcp_task_input import MCPTaskInputCustody
from kestrel_sovereign.mcp_task_store import MCPTaskConflict
from kestrel_sovereign.mcp_task_submission import MCPTaskServiceOrigin
from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_evidence_carriers
from kestrel_sovereign.security.key_rotation import RotationStatus
from tests.shared.mcp_task_custody import (
    custody as custody, binding, submission, attach, observation, lost_commit_ack, rotate_task_root,
)
from tests.utils.original_failure import fail_original_file_read


@pytest_asyncio.fixture
async def joint(custody):
    async def pair(**limits):
        tasks = custody.store if not limits else await custody.reopen(**limits)
        inputs = MCPInputStore(tasks.database, agent_did=tasks.agent_did,
            max_pending=2, max_payload_bytes=4096)
        await inputs.initialize()
        return SimpleNamespace(tasks=tasks, inputs=inputs, bridge=MCPTaskInputCustody(tasks, inputs))

    first = await pair()

    async def reopen(**limits):
        tasks = await custody.reopen(**limits)
        inputs = MCPInputStore(tasks.database, agent_did=tasks.agent_did,
            max_pending=2, max_payload_bytes=4096)
        await inputs.initialize()
        return SimpleNamespace(tasks=tasks, inputs=inputs, bridge=MCPTaskInputCustody(tasks, inputs))

    first.reopen = reopen
    return first


def input_binding(operation_id='original-leaf-claim', **changes):
    original = binding()
    return replace(MCPInputBinding(original.agent_did, original.origin.principal_digest,
        original.origin.session_id, operation_id, 'actual-feature-connection-name',
        original.contract_digest, 'tools/call'), **changes)


def checkpoint():
    return {'native': {'resultType': 'input_required', 'requestState': 'opaque-original-state'},
        'arguments': {'private-answer': 'original-private-value'}}


async def waiting(joint):
    reserved = await joint.tasks.reserve(binding(), submission())
    record = await joint.bridge.suspend(reserved.issuance, input_binding(), checkpoint(),
        now=100, expires_at=1000)
    return reserved, record


@pytest.mark.asyncio
async def test_joint_wait_and_resume_keep_original_submission_and_mint_one_pair_after_commit(joint):
    assert joint.tasks.database is joint.inputs.database
    assert joint.tasks._root != joint.inputs._root
    reserved, pending = await waiting(joint)
    current = await joint.tasks.read(binding(), reserved.record.handle)
    assert current.phase == 'INPUT_WAITING' and pending.state == 'WAITING'
    assert pending.binding.custody == current.handle
    assert current.continuity['input']['request_id'] == pending.request_id
    assert await joint.tasks.open_submission(current) == submission()
    with pytest.raises(MCPInputConflict, match='joint custody'):
        await joint.inputs.claim_response(pending, {}, now=101)
    with pytest.raises(MCPTaskConflict):
        await attach(joint.tasks, reserved.issuance)
    fresh = await joint.reopen()
    authentic = await fresh.inputs.read(pending.binding, pending.request_id)
    claimed, issued = await fresh.bridge.resume(replace(authentic, _payload=b'forged-caller-payload'),
        {'form': {'action': 'accept'}}, operation_id='fresh-continuation', now=101)
    assert claimed.record.state == 'RESPONDING' and issued.record.phase == 'ISSUING'
    assert issued.record.handle == current.handle and issued.record.current_operation_id == 'fresh-continuation'
    assert issued.record.input_lineage_digest == current.input_lineage_digest
    assert await fresh.tasks.open_submission(issued.record) == submission()
    assert not (await fresh.tasks.reserve(binding(), submission())).fresh
    with pytest.raises(MCPInputConflict):
        await fresh.bridge.resume(authentic, {}, operation_id='another-continuation', now=102)
    with pytest.raises(MCPTaskConflict, match='retained work'):
        await fresh.tasks.reserve(binding(operation_id='fresh-continuation'), submission())


@pytest.mark.asyncio
async def test_successive_joint_rounds_keep_one_lineage_and_never_rewrite_original_bytes(joint, monkeypatch):
    reserved, first = await waiting(joint)
    before = await joint.tasks.database.fetchone('SELECT submission_limit,submission_digest,submission,input_lineage_digest FROM mcp_task_records')
    statements = []
    execute = joint.tasks.database.execute

    async def observed(query, *args, **kwargs):
        statements.append(query)
        return await execute(query, *args, **kwargs)

    monkeypatch.setattr(joint.tasks.database, 'execute', observed)
    first_claim, first_issuance = await joint.bridge.resume(first, {}, operation_id='first-continuation', now=101)
    second = await joint.bridge.suspend(first_issuance, input_binding('first-continuation'),
        checkpoint(), predecessor=first_claim, now=102, expires_at=1200)
    assert second.reservation_id == first.reservation_id and second.predecessor_id == first.request_id
    await joint.inputs.settle(first_claim, state='RESOLVED', continuation={'next_request_id': second.request_id})
    second_claim, second_issuance = await joint.bridge.resume(second, {}, operation_id='second-continuation', now=103)
    attached = await attach(joint.tasks, second_issuance)
    terminal = await joint.tasks.observe(attached, observation('completed'))
    await joint.inputs.settle(second_claim, state='RESOLVED', continuation={'task': terminal.handle})
    assert terminal.continuity['operations'] == ['first-continuation', 'second-continuation']
    assert await joint.tasks.open_submission(terminal) == submission()
    assert await joint.tasks.database.fetchone('SELECT submission_limit,submission_digest,submission,input_lineage_digest FROM mcp_task_records') == before
    for query in statements:
        if query.startswith('UPDATE mcp_task_records SET '):
            assignments = query.split(' SET ', 1)[1].split(' WHERE ', 1)[0]
            assert 'submission' not in assignments and 'input_lineage_digest' not in assignments


@pytest.mark.asyncio
async def test_proved_local_expiry_closes_both_waits_without_a_wire_credential(joint):
    reserved, pending = await waiting(joint)
    current = await joint.tasks.read(binding(), reserved.record.handle)
    with pytest.raises(MCPInputConflict, match='not expired'):
        await joint.bridge.close_expired(current, now=999)
    closed = await joint.bridge.close_expired(replace(current, _payload=b'caller-substitution'), now=1000)
    assert closed.phase == 'INPUT_CLOSED' and closed.remote_task_id is None and closed.envelope is None
    actual = await joint.inputs.read(pending.binding, pending.request_id)
    assert actual.state == 'NON_RESUMABLE'
    assert not (await joint.tasks.reserve(binding(), submission())).fresh
    with pytest.raises(MCPInputConflict):
        await joint.bridge.resume(pending, {}, operation_id='late-continuation', now=1001)
    with pytest.raises(MCPTaskConflict):
        await attach(joint.tasks, reserved.issuance)


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['principal', 'session', 'contract', 'operation', 'method'])
async def test_failed_joint_link_rolls_back_input_and_retains_the_original_issuer(joint, changed):
    reserved = await joint.tasks.reserve(binding(), submission())
    alterations = {'principal': {'principal_id': 'd' * 64}, 'session': {'session_id': 'foreign-session'},
        'contract': {'contract_digest': 'd' * 64}, 'operation': {'operation_id': 'foreign-operation'},
        'method': {'method': 'resources/read'}}
    with pytest.raises(MCPTaskConflict):
        await joint.bridge.suspend(reserved.issuance, input_binding(**alterations[changed]),
            checkpoint(), now=100, expires_at=1000)
    assert await joint.inputs.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (0,)
    assert await joint.tasks.lookup(binding()) == reserved.record
    assert (await attach(joint.tasks, reserved.issuance)).phase == 'ATTACHED'


@pytest.mark.asyncio
async def test_joint_store_requires_actual_database_identity_and_initialized_roots(joint):
    other = await joint.reopen()
    with pytest.raises(MCPTaskConflict, match='original database'):
        MCPTaskInputCustody(joint.tasks, other.inputs)
    uninitialized = MCPInputStore(joint.tasks.database, agent_did=joint.tasks.agent_did,
        max_pending=1, max_payload_bytes=4096)
    with pytest.raises(RuntimeError, match='not initialized'):
        MCPTaskInputCustody(joint.tasks, uninitialized)
    async with joint.inputs.write_transaction():
        with pytest.raises(MCPTaskConflict, match='own durable transaction'):
            MCPTaskInputCustody(joint.tasks, joint.inputs)


@pytest.mark.asyncio
@pytest.mark.parametrize('conversation', [None, 'private-session'])
async def test_service_attribution_cannot_become_an_interactive_input_wait(joint, conversation):
    origin = MCPTaskServiceOrigin('signal', 'f' * 64, conversation)
    original = binding(origin=origin)
    reserved = await joint.tasks.reserve(original, submission(origin=origin))
    with pytest.raises(MCPTaskConflict, match='original Task attribution'):
        await joint.bridge.suspend(reserved.issuance, input_binding(), checkpoint(),
            now=100, expires_at=1000)
    assert await joint.inputs.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (0,)
    assert await joint.tasks.lookup(original) == reserved.record
    assert (await attach(joint.tasks, reserved.issuance)).phase == 'ATTACHED'


@pytest.mark.asyncio
async def test_initialized_input_for_another_agent_cannot_join_the_task_store(joint):
    foreign = MCPInputStore(joint.tasks.database, agent_did='did:test:foreign-input-owner',
        max_pending=2, max_payload_bytes=4096)
    await foreign.initialize()
    try:
        with pytest.raises(MCPTaskConflict, match='original database, agent and scope'):
            MCPTaskInputCustody(joint.tasks, foreign)
    finally:
        await foreign.close()
    assert await joint.inputs.database.fetchone('SELECT COUNT(*) FROM mcp_inputs') == (0,)
    assert await joint.tasks.database.fetchone('SELECT COUNT(*) FROM mcp_task_records') == (0,)


@pytest.mark.asyncio
async def test_two_real_connections_cannot_both_resume_the_same_wait(joint):
    reserved, pending = await waiting(joint)
    other = await joint.reopen()
    outcomes = await asyncio.gather(joint.bridge.resume(pending, {}, operation_id='first-responder', now=101),
        other.bridge.resume(pending, {}, operation_id='second-responder', now=101), return_exceptions=True)
    pairs = [result for result in outcomes if isinstance(result, tuple)]
    conflicts = [result for result in outcomes if isinstance(result, MCPInputConflict)]
    assert len(pairs) == len(conflicts) == 1
    actual = await joint.tasks.read(binding(), reserved.record.handle)
    assert actual.phase == 'ISSUING' and len(actual.continuity['operations']) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['suspend', 'resume', 'close'])
async def test_native_lost_joint_commit_ack_returns_no_credentials_and_never_replays(joint, monkeypatch, operation):
    if operation == 'suspend':
        reserved = await joint.tasks.reserve(binding(), submission())
        pending = None
    else:
        reserved, pending = await waiting(joint)
    current = await joint.tasks.read(binding(), reserved.record.handle)
    returned = None
    with lost_commit_ack(monkeypatch, joint.inputs) as originals:
        with pytest.raises(BaseException) as caught:
            if operation == 'suspend':
                returned = await joint.bridge.suspend(reserved.issuance, input_binding(), checkpoint(),
                    now=100, expires_at=1000)
            elif operation == 'resume':
                returned = await joint.bridge.resume(pending, {}, operation_id='lost-ack-operation', now=101)
            else:
                returned = await joint.bridge.close_expired(current, now=1000)
        assert returned is None and len(originals) == 1
        assert execution_commit_outcome(caught.value) == 'unknown'
        assert any(error is originals[0] for error in execution_evidence_carriers(caught.value))
    fresh = await joint.reopen()
    actual = await fresh.tasks.read(binding(), current.handle)
    expected = {'suspend': ('INPUT_WAITING', 'WAITING'), 'resume': ('ISSUING', 'RESPONDING'),
        'close': ('INPUT_CLOSED', 'NON_RESUMABLE')}[operation]
    assert actual.phase == expected[0]
    request_id = actual.continuity['input']['request_id']
    retained = await fresh.inputs.read_for(binding().origin.principal_digest,
        binding().origin.session_id, request_id)
    assert retained.state == expected[1]
    assert not (await fresh.tasks.reserve(binding(), submission())).fresh
    with pytest.raises(MCPTaskConflict):
        await attach(fresh.tasks, reserved.issuance)
    if operation == 'resume':
        assert actual.continuity['operations'] == ['lost-ack-operation']
        with pytest.raises(MCPInputConflict):
            await fresh.bridge.resume(pending, {}, operation_id='would-replay', now=102)


@pytest.mark.asyncio
async def test_expiry_and_stale_timestamp_response_race_has_one_joint_winner(joint):
    reserved, pending = await waiting(joint)
    original = await joint.tasks.read(binding(), reserved.record.handle)
    other = await joint.reopen()
    outcomes = await asyncio.gather(joint.bridge.resume(pending, {}, operation_id='racing-response', now=999),
        other.bridge.close_expired(original, now=1000), return_exceptions=True)
    successes = [item for item in outcomes if not isinstance(item, BaseException)]
    refusals = [item for item in outcomes if isinstance(item, (MCPInputConflict, MCPTaskConflict))]
    assert len(successes) == len(refusals) == 1
    task = await joint.tasks.read(binding(), original.handle)
    actual = await joint.inputs.read(pending.binding, pending.request_id)
    assert (task.phase, actual.state) in {('ISSUING', 'RESPONDING'), ('INPUT_CLOSED', 'NON_RESUMABLE')}


@pytest.mark.asyncio
@pytest.mark.parametrize('owner', ['initial', 'foreign-initial', 'continuation', 'mutation'])
async def test_continuation_cannot_reuse_any_scope_wide_operation_identity(joint, owner):
    reserved, pending = await waiting(joint)
    if owner == 'initial':
        operation = binding().operation_id
    elif owner == 'foreign-initial':
        await joint.tasks.reserve(binding(operation_id='foreign-existing-initial'), submission())
        operation = 'foreign-existing-initial'
    elif owner == 'continuation':
        claimed, issued = await joint.bridge.resume(pending, {}, operation_id='consumed-continuation', now=101)
        pending = await joint.bridge.suspend(issued, input_binding('consumed-continuation'),
            checkpoint(), predecessor=claimed, now=102, expires_at=1000)
        operation = 'consumed-continuation'
    else:
        second = await joint.tasks.reserve(binding(operation_id='another-initial-operation'), submission())
        attached = await attach(joint.tasks, second.issuance, 'another-remote-task')
        await joint.tasks.reserve_mutation(attached, operation_id='consumed-native-mutation', kind='cancel')
        operation = 'consumed-native-mutation'
    before = await joint.tasks.read(binding(), reserved.record.handle)
    with pytest.raises(MCPTaskConflict, match='already has original custody'):
        await joint.bridge.resume(pending, {}, operation_id=operation, now=103)
    assert await joint.tasks.read(binding(), before.handle) == before
    assert (await joint.inputs.read(pending.binding, pending.request_id)).state == 'WAITING'


@pytest.mark.asyncio
async def test_continuation_identity_cannot_be_reused_by_a_remote_task_mutation(joint):
    _, pending = await waiting(joint)
    _, issued = await joint.bridge.resume(pending, {}, operation_id='original-continuation', now=101)
    actual = await attach(joint.tasks, issued)
    with pytest.raises(MCPTaskConflict, match='already has original custody'):
        await joint.tasks.reserve_mutation(actual, operation_id='original-continuation', kind='cancel')


@pytest.mark.asyncio
async def test_expiry_releases_active_capacity_while_unknown_issuance_stays_held(joint):
    limited = await joint.reopen(max_active=1)
    reserved, pending = await waiting(limited)
    with pytest.raises(MCPTaskConflict, match='active capacity'):
        await limited.tasks.reserve(binding(operation_id='another-operation'), submission())
    await limited.bridge.close_expired(await limited.tasks.read(binding(), reserved.record.handle), now=1000)
    unknown = await limited.tasks.reserve(binding(operation_id='unknown-operation'), submission())
    assert unknown.record.phase == 'ISSUING'
    with pytest.raises(MCPTaskConflict, match='active capacity'):
        await limited.tasks.reserve(binding(operation_id='after-unknown'), submission())
    assert (await limited.inputs.read(pending.binding, pending.request_id)).state == 'NON_RESUMABLE'
    assert {record.phase for record in await limited.tasks.list_retained()} == {'INPUT_CLOSED', 'ISSUING'}


@pytest.mark.asyncio
async def test_lowered_limits_do_not_strand_an_admitted_joint_continuation(joint):
    reserved, pending = await waiting(joint)
    lowered = await joint.reopen(max_active=1, max_records=1, max_payload_bytes=1, max_submission_bytes=1)
    _, issued = await lowered.bridge.resume(pending, {}, operation_id='admitted-continuation', now=101)
    assert issued.record.handle == reserved.record.handle
    assert await lowered.tasks.open_submission(issued.record) == submission()
    assert (await attach(lowered.tasks, issued)).phase == 'ATTACHED'


@pytest.mark.asyncio
async def test_actual_master_rotation_preserves_both_independent_joint_roots(custody, joint):
    reserved, pending = await waiting(joint)
    task_rows = await joint.tasks.database.fetchall('SELECT * FROM mcp_task_records')
    input_rows = await joint.inputs.database.fetchall('SELECT * FROM mcp_inputs')
    task_root, input_root = joint.tasks._root, joint.inputs._root
    status = await rotate_task_root(custody)
    assert status.status == RotationStatus.COMPLETED and status.records_processed == 2
    fresh = await joint.reopen()
    assert fresh.tasks._root == task_root and fresh.inputs._root == input_root
    assert await fresh.tasks.database.fetchall('SELECT * FROM mcp_task_records') == task_rows
    assert await fresh.inputs.database.fetchall('SELECT * FROM mcp_inputs') == input_rows
    assert await fresh.tasks.open_submission(await fresh.tasks.read(binding(), reserved.record.handle)) == submission()
    assert (await fresh.bridge.resume(pending, {}, operation_id='after-rotation', now=101))[1].record.phase == 'ISSUING'


@pytest.mark.asyncio
async def test_actual_store_identity_is_rechecked_after_input_preparation(joint, monkeypatch):
    reserved, pending = await waiting(joint)
    original = joint.inputs._read_for
    original_scope = joint.tasks.scope

    async def changed_owner(*args):
        result = await original(*args)
        joint.tasks.scope = 'a' * 64
        return result

    monkeypatch.setattr(joint.inputs, '_read_for', changed_owner)
    try:
        with pytest.raises(MCPTaskConflict, match='original database, agent and scope'):
            await joint.bridge.resume(pending, {}, operation_id='owner-changed-operation', now=101)
    finally:
        joint.tasks.scope = original_scope
        monkeypatch.setattr(joint.inputs, '_read_for', original)
    assert (await joint.inputs.read(pending.binding, pending.request_id)).state == 'WAITING'
    assert (await joint.tasks.read(binding(), reserved.record.handle)).phase == 'INPUT_WAITING'


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['suspend', 'resume', 'close'])
async def test_genuine_failure_after_input_write_rolls_back_both_stores(joint, monkeypatch, operation):
    if operation == 'suspend':
        reserved = await joint.tasks.reserve(binding(), submission())
        pending = None
    else:
        reserved, pending = await waiting(joint)
    current = await joint.tasks.read(binding(), reserved.record.handle)
    input_before = await joint.inputs.database.fetchall('SELECT * FROM mcp_inputs')
    task_before = await joint.tasks.database.fetchall('SELECT * FROM mcp_task_records')
    original = joint.tasks._lock
    errors = []

    async def native_failure_after_lock():
        await original()
        fail_original_file_read(errors)

    with monkeypatch.context() as patch:
        patch.setattr(joint.tasks, '_lock', native_failure_after_lock)
        with pytest.raises(BaseException) as caught:
            if operation == 'suspend':
                await joint.bridge.suspend(reserved.issuance, input_binding(), checkpoint(), now=100, expires_at=1000)
            elif operation == 'resume':
                await joint.bridge.resume(pending, {}, operation_id='failed-continuation', now=101)
            else:
                await joint.bridge.close_expired(current, now=1000)
        assert len(errors) == 1 and any(error is errors[0] for error in execution_evidence_carriers(caught.value))
    assert await joint.inputs.database.fetchall('SELECT * FROM mcp_inputs') == input_before, 'Input mutation escaped the joint rollback'
    assert await joint.tasks.database.fetchall('SELECT * FROM mcp_task_records') == task_before, 'Task mutation escaped the joint rollback'


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['suspend', 'resume', 'close'])
async def test_cancellation_before_joint_commit_joins_real_rollback_without_credentials(joint, monkeypatch, operation):
    if operation == 'suspend':
        reserved = await joint.tasks.reserve(binding(), submission())
        pending = None
    else:
        reserved, pending = await waiting(joint)
    current = await joint.tasks.read(binding(), reserved.record.handle)
    before_input = await joint.inputs.database.fetchall('SELECT * FROM mcp_inputs')
    before_task = await joint.tasks.database.fetchall('SELECT * FROM mcp_task_records')
    entered, release = asyncio.Event(), asyncio.Event()
    original = joint.tasks._lock

    async def held_actual_lock():
        await original()
        entered.set()
        await release.wait()

    async def invoke():
        if operation == 'suspend':
            return await joint.bridge.suspend(reserved.issuance, input_binding(), checkpoint(), now=100, expires_at=1000)
        if operation == 'resume':
            return await joint.bridge.resume(pending, {}, operation_id='cancelled-continuation', now=101)
        return await joint.bridge.close_expired(current, now=1000)

    with monkeypatch.context() as patch:
        patch.setattr(joint.tasks, '_lock', held_actual_lock)
        worker = asyncio.create_task(invoke())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            worker.cancel('owned joint caller withdrew before commit')
            result, = await asyncio.gather(worker, return_exceptions=True)
            assert isinstance(result, BaseException), 'Cancellation released a joint credential'
            assert any(isinstance(error, asyncio.CancelledError)
                for error in execution_evidence_carriers(result))
        finally:
            release.set()
            await asyncio.gather(worker, return_exceptions=True)
    assert await joint.inputs.database.fetchall('SELECT * FROM mcp_inputs') == before_input
    assert await joint.tasks.database.fetchall('SELECT * FROM mcp_task_records') == before_task


@pytest.mark.asyncio
async def test_authenticated_input_cannot_point_to_a_different_tasks_wait(joint):
    first, pending = await waiting(joint)
    second = await joint.tasks.reserve(binding(operation_id='another-original-operation'), submission())
    other_input = await joint.bridge.suspend(second.issuance, input_binding('another-original-operation'),
        checkpoint(), now=100, expires_at=1000)
    changed = replace(pending.binding, custody=second.record.handle)
    digest = joint.inputs._binding_digest(changed)
    encrypted = joint.inputs._encode(changed, pending.request_id, pending.revision, 'WAITING',
        pending.expires_at, None, pending.payload, reservation_id=pending.reservation_id,
        predecessor_id=pending.predecessor_id)
    await joint.inputs.database.execute('UPDATE mcp_inputs SET binding_digest=?,payload=? WHERE request_id=?',
        (digest, encrypted, pending.request_id))
    actual = await joint.inputs.read(changed, pending.request_id)
    with pytest.raises(MCPTaskConflict, match='original waiting Task'):
        await joint.bridge.resume(actual, {}, operation_id='cross-linked-response', now=101)
    assert (await joint.inputs.read(changed, pending.request_id)).state == 'WAITING'
    assert (await joint.tasks.read(binding(), first.record.handle)).phase == 'INPUT_WAITING'
    assert (await joint.inputs.read(other_input.binding, other_input.request_id)).state == 'WAITING'


@pytest.mark.asyncio
async def test_an_assigned_task_lineage_cannot_be_replaced_even_by_authenticated_metadata(joint):
    reserved, pending = await waiting(joint)
    current = await joint.tasks.read(binding(), reserved.record.handle)
    before = await joint.tasks.database.fetchall('SELECT * FROM mcp_task_records')
    lineage = joint.tasks._digest('mcp-task-input-lineage-v1', 'another-reservation')
    continuity = current.continuity
    continuity['input']['reservation_id'] = 'another-reservation'
    with pytest.raises(MCPTaskConflict, match='lineage cannot change'):
        async with joint.tasks.write_transaction():
            await joint.tasks._lock()
            actual, row = joint.tasks._select(await joint.tasks._all(), current)
            await joint.tasks._persist(actual, row, 'INPUT_WAITING', None,
                continuity=continuity, input_lineage_digest=lineage)
    assert await joint.tasks.database.fetchall('SELECT * FROM mcp_task_records') == before
    assert (await joint.inputs.read(pending.binding, pending.request_id)).state == 'WAITING'
