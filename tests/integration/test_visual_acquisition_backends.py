"""Original acquisition checkpoints use only the canonical encrypted row."""
import asyncio
import copy
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import SecretBytes
from kestrel_sdk.computer_use import ComputerSession, ControlRequest, EnvironmentQuote
from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionPreparation
from kestrel_sdk.resources import ResourceConflict
from kestrel_sdk.storage.database.interface import TransactionError
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, bind_execution_custody_snapshot,
    is_execution_control_error,
)
from kestrel_sovereign.visual_computer.repository import CustodyRecord, VisualCustodyRepository
from tests.integration.test_visual_custody_backends import custody as custody, record, SCOPE
from tests.integration.test_visual_original_result_backends import OriginalTurn


async def quoted(store):
    value = await store.admit(SCOPE, record(), lambda: None)
    request = value.request
    value.quote = EnvironmentQuote(quote_id='original-quote', request_id=request.request_id,
        request_sha256=request.digest(), scope=request.scope, provider_id=request.provider_id,
        location=request.location, expires_at=request.expires_at, hourly_cost_usd=Decimal(0),
        estimated_total_cost_usd=Decimal(0), capabilities=dict(provider_id=request.provider_id,
            location=request.location, surfaces=frozenset({'browser'}), operations=request.required_capabilities))
    value.phase = 'quoted'
    await store.save(SCOPE, value)
    return value


def preparation(payload=b'PRIVATE-CREATE-BODY', identity=b'PRIVATE-WORKER-IDENTITY'):
    return EnvironmentAcquisitionPreparation(create_payload=SecretBytes(payload), worker_identity=SecretBytes(identity))


@pytest.mark.asyncio
@pytest.mark.parametrize('contract', ['return-value', 'coroutine'])
async def test_acquisition_requires_original_synchronous_raising_checker(custody, contract):
    store, _ = custody
    value = await quoted(store)
    def returning():
        return True
    async def asynchronous():
        pytest.fail('a synchronous restriction must never await a new grant')
    check = returning if contract == 'return-value' else asynchronous
    with pytest.raises(TransactionError, match='must return None') as denied:
        async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=check):
            pytest.fail('an unsupported original checker cannot issue an attempt')
    assert type(denied.value.__cause__) is TypeError
    assert await store.get(SCOPE, request_id=value.request.request_id) == value


@pytest.mark.asyncio
@pytest.mark.parametrize('custody', ['postgres'], indirect=True)
async def test_original_acquisition_owner_keeps_full_native_pg_program_guard(custody):
    store, _ = custody
    value = await quoted(store)
    async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
        pool = attempt._result._native_backing.postgres_pool.pool
        original = pool._init
        try:
            pool._init = lambda connection: None
            with pytest.raises(ExecutionAuthorityError, match='application pool programs'):
                _ = attempt.checkpoint
        finally:
            pool._init = original


@pytest.mark.asyncio
async def test_acquisition_original_attempt_and_private_preparation_survive_reopen(custody):
    store, reopen = custody
    value = await quoted(store)
    async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
        first = attempt.checkpoint
        assert first.state == 'unprepared' and first.sequence == 0
        original = await attempt.prepare_once(preparation())
        assert original == preparation()
        assert attempt.checkpoint.state == 'prepared' and attempt.checkpoint.sequence == 1
        await attempt.prepare_once(preparation())
        assert attempt.checkpoint.sequence == 1
        with pytest.raises(ResourceConflict):
            await attempt.prepare_once(preparation(identity=b'different original credential'))
        with pytest.raises(ExecutionAuthorityError, match='producer absent'):
            await attempt.begin_dispatch()
    raw = bytes((await store.db.fetchone('SELECT payload FROM visual_computer_requests'))[0])
    assert b'PRIVATE-CREATE-BODY' not in raw and b'PRIVATE-WORKER-IDENTITY' not in raw
    current = await store.get(SCOPE, request_id=value.request.request_id)
    assert 'acquisition' not in current.model_dump() and 'PRIVATE' not in repr(current)
    with pytest.raises(ResourceConflict, match='native checkpoint CAS'):
        await store.save(SCOPE, current.model_copy(update={'acquisition': None}))
    with pytest.raises(ResourceConflict):
        await store.save(SCOPE, value)
    db = await reopen()
    try:
        restarted = VisualCustodyRepository(db)
        current = await restarted.get(SCOPE, request_id=value.request.request_id)
        async with restarted.original_acquisition_result(SCOPE, current, current.request, current.quote, guard=lambda: None) as attempt:
            assert attempt.checkpoint.attempt_id == first.attempt_id
            assert attempt.checkpoint.sequence == 1
            assert attempt.preparation == original
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_acquisition_prepared_retirement_keeps_original_attempt_and_no_quota_recharge(custody):
    store, _ = custody
    store.max_requests = 1
    value = await quoted(store)
    async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
        await attempt.prepare_once(preparation())
        identity = attempt.checkpoint.attempt_id
    current = await store.get(SCOPE, request_id=value.request.request_id)
    retired = await store.retire_unacquired(SCOPE, current, at=datetime.now(UTC))
    assert retired.phase == 'retired' and retired.acquisition == current.acquisition
    assert (await store.db.fetchone('SELECT retired, released FROM visual_computer_requests')) == (1, 0)
    with pytest.raises(ResourceConflict):
        async with store.original_acquisition_result(SCOPE, retired, retired.request, retired.quote, guard=lambda: None):
            pytest.fail('retired attempt must not reopen')
    original = await store.admit(SCOPE, value, lambda: None)
    assert original.phase == 'retired'
    from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionStorage
    assert EnvironmentAcquisitionStorage.deserialize_private(original.acquisition.storage).checkpoint.attempt_id == identity
    fresh = record()
    fresh.request = fresh.request.model_copy(update={'request_id': 'fresh-request'})
    fresh.session_id = 'fresh-session'
    assert (await store.admit(SCOPE, fresh, lambda: None)).phase == 'admitted'


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['pending_control', 'last_control', 'session'])
async def test_acquisition_prepared_refuses_unissued_control_derivatives(custody, field):
    store, _ = custody
    value = await quoted(store)
    async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
        await attempt.prepare_once(preparation())
    current = await store.get(SCOPE, request_id=value.request.request_id)
    if field == 'session':
        fact = ComputerSession(session_id=current.session_id, scope=SCOPE,
            lease_id='unissued-lease', driver_id='test-driver', surface='browser',
            capabilities=current.request.required_capabilities, state='paused',
            controller='none', control_epoch=0, expires_at=current.request.expires_at)
    else:
        fact = ControlRequest(request_id='unissued-control', session_id=current.session_id,
            expected_control_epoch=0, operation='pause')
    from kestrel_sovereign.visual_computer.repository import CustodyIntegrityError
    with pytest.raises(CustodyIntegrityError, match='effect facts'):
        await store.save(SCOPE, current.model_copy(update={field: fact}))
    assert await store.get(SCOPE, request_id=value.request.request_id) == current


@pytest.mark.parametrize(('field', 'fact'), [
    ('human_private', True), ('human_control_id', 'unissued-human-control'), ('human_control_epoch', 0),
])
def test_shared_schema_retains_human_fields_without_acquisition(field, fact):
    # Shared Human facts remain valid schema. Acquisition authority is enforced
    # independently on the authenticated original row and native checkpoint.
    value = CustodyRecord.model_validate({**record().model_dump(), field: fact})
    assert getattr(value, field) == fact and value.acquisition is None


@pytest.mark.asyncio
@pytest.mark.parametrize('wrong', ['copy', 'row', 'ciphertext', 'key', 'sql', 'lifetime', 'foreign-task', 'closed'])
async def test_acquisition_original_writer_refuses_rebound_or_closed_owners(custody, wrong):
    store, _ = custody
    value = await quoted(store)
    original_key = store._key
    try:
        async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
            root = attempt._result
            candidate = root
            if wrong == 'copy': candidate = copy.copy(root)
            elif wrong == 'row': root._identity = (root._identity[0], 'other-request', root._identity[2])
            elif wrong == 'ciphertext': root._expected = b'other-ciphertext'
            elif wrong == 'key': store._key = b'x'*32
            elif wrong == 'sql': root._query = 'DELETE FROM visual_computer_requests'
            elif wrong == 'lifetime': root._expires_at += timedelta(seconds=1)
            elif wrong == 'closed': root.close()
            with pytest.raises(ExecutionAuthorityError):
                if wrong == 'foreign-task': await asyncio.create_task(candidate.prepare_once(preparation()))
                else: await candidate.prepare_once(preparation())
    finally:
        store._key = original_key
    current = await store.get(SCOPE, request_id=value.request.request_id)
    assert current.phase == 'quoted'


@pytest.mark.asyncio
async def test_acquisition_revoked_operation_cannot_prepare_or_claim_late_native_result(custody):
    store, _ = custody
    value = await quoted(store)
    turn = ExecutionCustody(OriginalTurn(store.db.backend_type))
    with bind_execution_custody_snapshot((turn,)):
        async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
            turn.revoke('global stop before preparation')
            with pytest.raises(ExecutionAuthorityError): await attempt.prepare_once(preparation())
            with pytest.raises(ExecutionAuthorityError): await attempt.record_unknown()
            with pytest.raises(Exception) as denied: await store.db.fetchone('SELECT 1')
            assert is_execution_control_error(denied.value)
    current = await store.get(SCOPE, request_id=value.request.request_id)
    retired = await store.retire_unacquired(SCOPE, current, at=datetime.now(UTC))
    assert retired.phase == 'retired'


@pytest.mark.asyncio
async def test_acquisition_lost_allocation_ack_reuses_committed_original_attempt(custody, monkeypatch):
    store, _ = custody
    value = await quoted(store)
    original = store.db.backend._settle_original_acquisition_result
    identities = []
    async def lose_after_native_ack(result):
        identities.append(result)
        await original(result)
        raise RuntimeError('lost original allocation acknowledgment publication')
    with monkeypatch.context() as patch:
        patch.setattr(store.db.backend, '_settle_original_acquisition_result', lose_after_native_ack)
        with pytest.raises(BaseException):
            async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None):
                pytest.fail('lost ACK must not yield an acquisition operation')
    root = identities[0]
    assert root._phase == 'closed' and root._revision == value.revision
    current = await store.get(SCOPE, request_id=value.request.request_id)
    from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionStorage
    committed = EnvironmentAcquisitionStorage.deserialize_private(current.acquisition.storage).checkpoint
    async with store.original_acquisition_result(SCOPE, current, current.request, current.quote, guard=lambda: None) as attempt:
        assert attempt.checkpoint.attempt_id == committed.attempt_id
        assert attempt.checkpoint.sequence == 0
        await attempt.prepare_once(preparation())
    assert len(identities) == 1


@pytest.mark.asyncio
async def test_acquisition_allocation_race_has_one_original_row_attempt(custody):
    store, reopen = custody
    value = await quoted(store)
    db = await reopen()
    try:
        other = VisualCustodyRepository(db)
        await other.ensure_initialized()
        ready = asyncio.Event()
        async def allocate(repository):
            await ready.wait()
            async with repository.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
                return attempt.checkpoint.attempt_id
        tasks = [asyncio.create_task(allocate(repository)) for repository in (store, other)]
        ready.set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        successes = [item for item in outcomes if type(item) is str]
        assert len(successes) == 1
        assert all(type(item) is str or isinstance(item, BaseException) for item in outcomes)
        current = await store.get(SCOPE, request_id=value.request.request_id)
        async with other.original_acquisition_result(SCOPE, current, current.request, current.quote, guard=lambda: None) as attempt:
            assert attempt.checkpoint.attempt_id == successes[0]
        assert (await store.db.fetchone('SELECT COUNT(*) FROM visual_computer_requests'))[0] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_present_corrupt_acquisition_envelope_never_becomes_fresh_attempt(custody):
    store, _ = custody
    value = await quoted(store)
    from kestrel_sdk.security.aead import AEADCipher
    from kestrel_sovereign.visual_computer.repository import CustodyIntegrityError
    import json
    owner = store._owner(SCOPE)
    payload = value.model_dump(mode='json')
    payload['acquisition'] = {'schema_version': 1, 'storage': 'e30=', 'binding': None}
    corrupt = AEADCipher(store._key).encrypt(json.dumps(payload).encode(), aad=f'{owner}:{value.request.request_id}'.encode())
    await store.db.execute('UPDATE visual_computer_requests SET payload = ? WHERE owner = ? AND request_id = ?',
        (corrupt, owner, value.request.request_id))
    with pytest.raises(CustodyIntegrityError):
        await store.get(SCOPE, request_id=value.request.request_id)
    with pytest.raises(CustodyIntegrityError):
        async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None):
            pytest.fail('corrupt original storage cannot allocate')


@pytest.mark.asyncio
async def test_legacy_possible_create_cannot_gain_new_acquisition_attempt(custody):
    store, _ = custody
    value = await quoted(store)
    value.phase = 'acquiring'
    await store.save(SCOPE, value)
    with pytest.raises(ResourceConflict):
        async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None):
            pytest.fail('legacy possible create may not allocate or retry')
    with pytest.raises(ResourceConflict):
        await store.retire_unacquired(SCOPE, value, at=datetime.now(UTC))


@pytest.mark.asyncio
async def test_acquisition_stage_requires_exact_old_ciphertext_not_only_revision(custody):
    store, _ = custody
    value = await quoted(store)
    async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
        root = attempt._result
        current = await store.get(SCOPE, request_id=value.request.request_id)
        replacement = store._seal(root._identity[0], value.request.request_id, current)
        assert replacement != root._expected
        await store.db.execute('UPDATE visual_computer_requests SET payload = ? WHERE owner = ? AND request_id = ?',
            (replacement, root._identity[0], value.request.request_id))
        with pytest.raises(BaseException) as refused:
            await attempt.prepare_once(preparation())
        assert is_execution_control_error(refused.value)
        assert root._revision == current.revision
    assert bytes((await store.db.fetchone('SELECT payload FROM visual_computer_requests'))[0]) == replacement


@pytest.mark.asyncio
async def test_acquisition_owner_cannot_use_agent_or_signal_native_program(custody):
    store, _ = custody
    value = await quoted(store)
    async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
        for entry in (store.db.backend._settle_original_action_result, store.db.backend._append_original_signal_outcome):
            with pytest.raises(ExecutionAuthorityError):
                await entry(attempt._result)


@pytest.mark.asyncio
@pytest.mark.parametrize(('field', 'fact'), [
    ('human_private', True), ('human_control_id', 'unissued-human-control'), ('human_control_epoch', 0),
])
async def test_acquisition_prepared_refuses_unissued_human_effect_facts(custody, field, fact):
    from kestrel_sovereign.visual_computer.repository import CustodyIntegrityError

    store, _ = custody
    value = await quoted(store)
    async with store.original_acquisition_result(SCOPE, value, value.request, value.quote, guard=lambda: None) as attempt:
        await attempt.prepare_once(preparation())
        checkpoint = attempt.checkpoint
    current = await store.get(SCOPE, request_id=value.request.request_id)
    # Human facts belong to the shared schema, but have no original Human
    # issuance on this authenticated, never-dispatched acquisition attempt.
    candidate = CustodyRecord.model_validate({**current.model_dump(), field: fact,
        'acquisition': current.acquisition})
    before = await store.db.fetchone('SELECT payload, revision, released, retired FROM visual_computer_requests')
    with pytest.raises(CustodyIntegrityError, match='effect facts'):
        await store.save(SCOPE, candidate)
    after = await store.db.fetchone('SELECT payload, revision, released, retired FROM visual_computer_requests')
    assert tuple(after) == tuple(before)
    restored = await store.get(SCOPE, request_id=value.request.request_id)
    assert restored == current and restored.phase == 'quoted'
    from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionStorage
    assert EnvironmentAcquisitionStorage.deserialize_private(restored.acquisition.storage).checkpoint == checkpoint
