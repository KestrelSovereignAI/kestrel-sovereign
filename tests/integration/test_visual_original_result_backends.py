"""Only the original fixed native action result survives turn retirement."""
import asyncio
import copy
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.computer_use import ActionRequest, ComputerSession
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, ExecutionCustody, bind_execution_custody_snapshot, require_execution_work, is_execution_control_error
from kestrel_sovereign.visual_computer.dispatch import original_dispatch, native_dispatch
from tests.integration.test_visual_custody_backends import custody, record, SCOPE


class OriginalTurn:
    def __init__(self, backend_type):self.backend_type = backend_type
    def require_work(self):pass
    async def lock_and_validate(self, connection):pass


async def prepared(store):
    current = await store.admit(SCOPE, record(), lambda:None)
    session = ComputerSession(session_id=current.session_id, scope=SCOPE, lease_id='lease',
        driver_id='driver', surface='browser', capabilities=frozenset({'click'}), state='ready',
        controller='agent', control_epoch=0, expires_at=current.request.expires_at)
    current.session = session
    current.phase = 'active'
    await store.save(SCOPE, current)
    action = ActionRequest(action_id='original-action', session_id=session.session_id,
        observation_id='original-frame', control_epoch=0, not_after=datetime.now(UTC)+timedelta(seconds=30),
        action={'kind':'click','x':1,'y':1})
    await store.action(SCOPE,session.session_id,action.action_id,accept=action,prepare=True)
    return current, session, action


@pytest.mark.asyncio
async def test_global_turn_stop_keeps_only_exact_original_negative_fact(custody):
    store,reopen = custody
    current,session,action = await prepared(store)
    # Original trusted test-owner restriction, installed BEFORE any capture.
    # No hosted-Postgres grant is synthesized for standalone SQLite.
    turn = ExecutionCustody(OriginalTurn(store.db.backend_type))
    with bind_execution_custody_snapshot((turn,)):
        with original_dispatch(session, action) as origin:
            async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
                await store.action(SCOPE,session.session_id,action.action_id,
                    dispatch_revision=current.revision,_result_owner=result)
                turn.revoke('global turn stopped')
                for operation in (lambda:store.db.execute('DELETE FROM visual_computer_actions'),
                                  lambda:store.db.fetchone('SELECT 1'),
                                  lambda:store.action(SCOPE,session.session_id,action.action_id)):
                    with pytest.raises(Exception) as denied:await operation()
                    assert is_execution_control_error(denied.value)
                with pytest.raises(ExecutionAuthorityError):require_execution_work()
                await result.not_dispatched(origin,origin.refuse_unentered(asyncio.CancelledError('original stop')))
                with pytest.raises(Exception) as denied:await store.db.fetchone('SELECT 1')
                assert is_execution_control_error(denied.value)
    recovered = await store.action(SCOPE,session.session_id,action.action_id)
    assert recovered.dispatch_started and recovered.request == action
    assert recovered.receipt.outcome=='denied' and recovered.receipt.reason_code=='host_not_dispatched'
    assert (await store.db.fetchone('SELECT pending FROM visual_computer_actions'))[0]==0
    assert not await store.has_pending_actions(SCOPE,session.session_id)
    db=await reopen()
    try:
        from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
        other=VisualCustodyRepository(db)
        assert await other.action(SCOPE,session.session_id,action.action_id)==recovered
    finally:await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('wrong',['copy','owner','row','request','sql','expected','foreign-task','closed'])
async def test_original_result_refuses_forged_stale_or_rebound_uses(custody,wrong):
    store,_=custody;current,session,action=await prepared(store)
    with original_dispatch(session,action) as origin:
        async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
            await store.action(SCOPE,session.session_id,action.action_id,
                dispatch_revision=current.revision,_result_owner=result)
            fact=origin.refuse_unentered(RuntimeError('not dispatched'))
            candidate=result
            if wrong=='copy':candidate=copy.copy(result)
            elif wrong=='owner':candidate._identity=('other-owner',*result._identity[1:])
            elif wrong=='row':candidate._identity=(*result._identity[:2],'other-action')
            elif wrong=='request':candidate._stored.request=action.model_copy(update={'observation_id':'other'})
            elif wrong=='sql':candidate._query='DELETE FROM visual_computer_actions'
            elif wrong=='expected':candidate._expected=b'other revision'
            elif wrong=='closed':candidate.close()
            async def attempt():await candidate.not_dispatched(origin,fact)
            with pytest.raises(ExecutionAuthorityError):
                if wrong=='foreign-task':await asyncio.create_task(attempt())
                else:await attempt()
    assert await store.has_pending_actions(SCOPE,session.session_id)


@pytest.mark.asyncio
async def test_original_result_cas_cannot_overwrite_later_original_revision(custody):
    store,_=custody;current,session,action=await prepared(store)
    with original_dispatch(session,action) as origin:
        async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
            await store.action(SCOPE,session.session_id,action.action_id,
                dispatch_revision=current.revision,_result_owner=result)
            # Same original row rewritten with a new encrypted revision. The
            # retained exact-ciphertext CAS must not silently overwrite it.
            await store.action(SCOPE,session.session_id,action.action_id,accept=action)
            from kestrel_sovereign.visual_computer.repository import ActionRecord
            owner,key,identifier=result._identity
            later=store._seal(owner,f'{key}:{identifier}',ActionRecord(request=action,dispatch_started=True))
            await store.db.execute('UPDATE visual_computer_actions SET payload = ? WHERE owner = ? AND session_key = ? AND action_id = ?',
                (later,owner,key,identifier))
            with pytest.raises(BaseException) as refused:
                await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no entry')))
            assert is_execution_control_error(refused.value)
    assert await store.has_pending_actions(SCOPE,session.session_id)


@pytest.mark.asyncio
async def test_possible_partial_send_never_uses_result_permission(custody):
    store,_=custody;current,session,action=await prepared(store)
    with original_dispatch(session,action) as origin:
        async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
            await store.action(SCOPE,session.session_id,action.action_id,
                dispatch_revision=current.revision,_result_owner=result)
            origin.enter_driver()
            with native_dispatch(session,action,'driver') as trace:
                trace.input_started=True
            assert trace.refusal(ResourceOutcomeUnknown('possible send')) is None
            with pytest.raises(ResourceOutcomeUnknown):origin.refuse_unentered(RuntimeError('not a fact'))
            from kestrel_sovereign.visual_computer.dispatch import NativeActionNotDispatched
            with pytest.raises(ResourceOutcomeUnknown):
                await result.not_dispatched(origin,NativeActionNotDispatched(trace,RuntimeError('forged no-send')))
    assert await store.has_pending_actions(SCOPE,session.session_id)


@pytest.mark.asyncio
async def test_native_database_close_joins_original_result_before_retirement(custody):
    store,_=custody;current,session,action=await prepared(store)
    entered,resume=asyncio.Event(),asyncio.Event()
    async def own():
        with original_dispatch(session,action) as origin:
            async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
                await store.action(SCOPE,session.session_id,action.action_id,
                    dispatch_revision=current.revision,_result_owner=result)
                entered.set();await resume.wait()
                await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no entry')))
    task=asyncio.create_task(own());await asyncio.wait_for(entered.wait(),2)
    closing=asyncio.create_task(store.db.close())
    try:
        await asyncio.sleep(.03);assert not closing.done()
        resume.set();await task;await closing
        assert not store.db.backend._original_result_holds
    finally:
        resume.set();await asyncio.gather(task,closing,return_exceptions=True)


@pytest.mark.asyncio
async def test_exact_owner_rechecked_after_last_native_proof_before_commit(custody,monkeypatch):
    store,_=custody;current,session,action=await prepared(store)
    key=store._key
    if store.db.backend_type=='postgres':
        from kestrel_sovereign.storage.db import postgres
        prove=postgres._prove_committable
        async def retire_after_probe(connection):
            await prove(connection);store._key=b'k'*32
        install=lambda:monkeypatch.setattr(postgres,'_prove_committable',retire_after_probe)
    else:
        backend=store.db.backend;prove=backend._verify_original_signal_outcome
        async def retire_after_probe(connection,journal):
            await prove(connection,journal);store._key=b'k'*32
        install=lambda:monkeypatch.setattr(backend,'_verify_original_signal_outcome',retire_after_probe)
    try:
        with original_dispatch(session,action) as origin:
            async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
                await store.action(SCOPE,session.session_id,action.action_id,
                    dispatch_revision=current.revision,_result_owner=result)
                expected=result._expected
                install()
                # Install the controlled retirement only for terminal proof,
                # after all ordinary admission/CAS transactions completed.
                with pytest.raises(Exception) as refused:
                    await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no entry')))
                assert is_execution_control_error(refused.value)
    finally:store._key=key
    row=await store.db.fetchone('SELECT payload,pending FROM visual_computer_actions')
    assert bytes(row[0])==expected and row[1]==1


@pytest.mark.asyncio
async def test_native_result_schema_and_key_replacement_refuse_no_resurrection(custody):
    store,_=custody;current,session,action=await prepared(store)
    with original_dispatch(session,action) as origin:
        async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
            await store.action(SCOPE,session.session_id,action.action_id,
                dispatch_revision=current.revision,_result_owner=result)
            await store.db.execute('DELETE FROM visual_computer_actions')
            with pytest.raises(BaseException) as refused:
                await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('original no entry')))
            assert is_execution_control_error(refused.value)
    assert (await store.db.fetchone('SELECT COUNT(*) FROM visual_computer_actions'))[0]==0


@pytest.mark.asyncio
async def test_original_result_outermost_native_write_required(custody):
    store,_=custody;current,session,action=await prepared(store)
    with original_dispatch(session,action) as origin:
        async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
            await store.action(SCOPE,session.session_id,action.action_id,
                dispatch_revision=current.revision,_result_owner=result)
            async with store.db.transaction():
                with pytest.raises(ExecutionAuthorityError):
                    await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no entry')))
    assert await store.has_pending_actions(SCOPE,session.session_id)


@pytest.mark.asyncio
async def test_result_writer_is_joined_after_repeated_observer_cancellation(custody,monkeypatch):
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    store,_=custody;current,session,action=await prepared(store)
    committed,resume=asyncio.Event(),asyncio.Event()
    settle=store.db.backend._settle_original_action_result
    async def hold_after_native_commit(journal):
        value=await settle(journal)
        committed.set();await resume.wait();return value
    monkeypatch.setattr(store.db.backend,'_settle_original_action_result',hold_after_native_commit)
    async def original_owner():
        with original_dispatch(session,action) as origin:
            async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
                await store.action(SCOPE,session.session_id,action.action_id,
                    dispatch_revision=current.revision,_result_owner=result)
                await result.not_dispatched(origin,origin.refuse_unentered(asyncio.CancelledError('original effect stop')))
    task=asyncio.create_task(original_owner());await asyncio.wait_for(committed.wait(),3)
    try:
        task.cancel();await asyncio.sleep(.01);task.cancel();await asyncio.sleep(.01)
        assert not task.done() and store.db.backend._original_result_holds
        resume.set()
        with pytest.raises(BaseException) as withdrawn:await task
        assert execution_commit_outcome(withdrawn.value)=='committed'
        assert not store.db.backend._original_result_holds
        assert (await store.action(SCOPE,session.session_id,action.action_id)).receipt.outcome=='denied'
    finally:
        resume.set();await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_sqlite_lost_native_commit_ack_keeps_unknown_and_exact_row(custody,monkeypatch):
    if custody[0].db.backend_type!='sqlite':pytest.skip('SQLite queue ACK boundary; PG native engine covered by signal suite')
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    store,_=custody;current,session,action=await prepared(store)
    connection=store.db.backend._connection;commit=connection.commit
    async def lose_original_ack():
        await commit()
        raise OSError('controlled original native commit ACK loss')
    with original_dispatch(session,action) as origin:
        async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
            await store.action(SCOPE,session.session_id,action.action_id,
                dispatch_revision=current.revision,_result_owner=result)
            monkeypatch.setattr(connection,'commit',lose_original_ack)
            with pytest.raises(BaseException) as unknown:
                await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no effect')))
            assert execution_commit_outcome(unknown.value)=='unknown'
    monkeypatch.setattr(connection,'commit',commit)
    assert (await store.action(SCOPE,session.session_id,action.action_id)).receipt.outcome=='denied'


@pytest.mark.asyncio
async def test_sqlite_external_generation_cannot_borrow_retired_work(custody):
    if custody[0].db.backend_type!='sqlite':pytest.skip('SQLite data_version boundary')
    store,reopen=custody;current,session,action=await prepared(store)
    turn=ExecutionCustody(OriginalTurn('sqlite'))
    other=await reopen()
    try:
        with bind_execution_custody_snapshot((turn,)):
            with original_dispatch(session,action) as origin:
                async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
                    await store.action(SCOPE,session.session_id,action.action_id,
                        dispatch_revision=current.revision,_result_owner=result)
                    # The independent native writer existed BEFORE retirement;
                    # its exact external data generation must remain visible.
                    await other.execute('UPDATE visual_computer_owners SET generation = generation + 1')
                    turn.revoke('original turn retired')
                    with pytest.raises(BaseException) as refused:
                        await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no entry')))
                    assert is_execution_control_error(refused.value)
        assert await store.has_pending_actions(SCOPE,session.session_id)
    finally:await other.close()


@pytest.mark.asyncio
async def test_unissued_owner_and_rejected_native_capture_leave_no_resource_hold(custody):
    from kestrel_sovereign.storage.original_action_result import _OriginalActionResult
    store,_=custody;current,session,action=await prepared(store)
    with original_dispatch(session,action) as origin:
        with pytest.raises(ExecutionAuthorityError):
            await object.__new__(_OriginalActionResult).not_dispatched(origin,object())
        if store.db.backend_type=='sqlite':
            # Ordinary customization is supported, but never an original
            # terminal profile. A failed constructor must not strand close.
            await store.db.backend._connection.create_function('application_hook',0,lambda:0)
            with pytest.raises(ExecutionAuthorityError):
                async with store.original_action_result(SCOPE,session,action,guard=lambda:None):pass
            assert not store._original_results and not store.db.backend._original_result_holds


@pytest.mark.asyncio
async def test_original_result_submission_is_once_only_before_native_entry(custody,monkeypatch):
    store,_=custody;current,session,action=await prepared(store)
    settle=store.db.backend._settle_original_action_result;entries=[]
    async def count(journal):entries.append(journal);return await settle(journal)
    monkeypatch.setattr(store.db.backend,'_settle_original_action_result',count)
    with original_dispatch(session,action) as origin:
        async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
            await store.action(SCOPE,session.session_id,action.action_id,
                dispatch_revision=current.revision,_result_owner=result)
            fact=origin.refuse_unentered(RuntimeError('original no entry'))
            await result.not_dispatched(origin,fact)
            with pytest.raises(BaseException) as denied:await result.not_dispatched(origin,fact)
            assert is_execution_control_error(denied.value)
            assert entries==[result]


@pytest.mark.asyncio
@pytest.mark.parametrize('replacement',['closed','fresh','copy'])
async def test_original_privacy_hold_cannot_be_reacquired_or_copied(custody,replacement):
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate, _DURABLE_GATE_SHARED_HOLDS
    store,_=custody;current,session,action=await prepared(store)
    gate=DurablePersistenceGate();shared=gate.shared();await shared.__aenter__()
    with original_dispatch(session,action) as origin:
        context=store.original_action_result(SCOPE,session,action,guard=lambda:None)
        result=await context.__aenter__()
        try:
            await store.action(SCOPE,session.session_id,action.action_id,
                dispatch_revision=current.revision,_result_owner=result)
            original_hold=result._registration.privacy_holds[0][0]
            clone=copy.copy(original_hold)
            await shared.__aexit__(None,None,None)
            fact=origin.refuse_unentered(RuntimeError('original no input'))
            async def denied():
                with pytest.raises(ExecutionAuthorityError,match='privacy hold retired'):
                    await result.not_dispatched(origin,fact)
            if replacement=='fresh':
                async with gate.shared():await denied()
            elif replacement=='copy':
                # A forged ContextVar frame is test input, never a product
                # operation or an admission fallback.
                token=_DURABLE_GATE_SHARED_HOLDS.set((clone,))
                try:await denied()
                finally:_DURABLE_GATE_SHARED_HOLDS.reset(token)
            else:await denied()
        finally:await context.__aexit__(None,None,None)
    assert await store.has_pending_actions(SCOPE,session.session_id)


@pytest.mark.asyncio
async def test_original_privacy_hold_retirement_at_last_native_probe_rolls_back(custody,monkeypatch):
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
    store,_=custody;current,session,action=await prepared(store)
    entered,resume=asyncio.Event(),asyncio.Event()
    if store.db.backend_type=='postgres':
        from kestrel_sovereign.storage.db import postgres
        prove=postgres._prove_committable
        async def interleave(connection):await prove(connection);entered.set();await resume.wait()
        install=lambda:monkeypatch.setattr(postgres,'_prove_committable',interleave)
    else:
        backend=store.db.backend;prove=backend._verify_original_signal_outcome
        async def interleave(connection,journal):await prove(connection,journal);entered.set();await resume.wait()
        install=lambda:monkeypatch.setattr(backend,'_verify_original_signal_outcome',interleave)
    async def original_owner():
        with original_dispatch(session,action) as origin:
            async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
                await store.action(SCOPE,session.session_id,action.action_id,
                    dispatch_revision=current.revision,_result_owner=result)
                install()
                with pytest.raises(BaseException) as refused:
                    await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no entry')))
                assert is_execution_control_error(refused.value)
    gate=DurablePersistenceGate()
    async with gate.shared():
        task=asyncio.create_task(original_owner())
        await asyncio.wait_for(entered.wait(),3)
    resume.set()
    try:await asyncio.wait_for(task,3)
    finally:
        resume.set();await asyncio.gather(task,return_exceptions=True)
    assert await store.has_pending_actions(SCOPE,session.session_id)
    assert (await store.db.fetchone('SELECT pending FROM visual_computer_actions'))[0]==1
    assert not gate.locked()


@pytest.mark.asyncio
async def test_original_result_uses_canonical_borrowed_postgres_pool_factory(custody):
    store,_=custody
    if store.db.backend_type!='postgres':pytest.skip('native borrowed PostgreSQL pool factory')
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.db.postgres import PostgresBackend
    from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
    current,session,action=await prepared(store)
    native_pool=store.db.backend._pool
    backend=PostgresBackend.from_pool(native_pool)
    async def empty(db):pass
    borrowed=await AsyncDatabase.from_connected_backend(backend,schema_initializer=empty)
    try:
        original=VisualCustodyRepository(borrowed)
        await original.ensure_initialized()
        with original_dispatch(session,action) as origin:
            async with original.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
                await original.action(SCOPE,session.session_id,action.action_id,
                    dispatch_revision=current.revision,_result_owner=result)
                await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('original no input')))
        assert not backend._original_result_holds
    finally:await borrowed.close()
    assert not native_pool.is_closing()  # Borrower cannot close its owner pool.
    assert (await store.action(SCOPE,session.session_id,action.action_id)).receipt.outcome=='denied'


@pytest.mark.asyncio
async def test_replaced_database_refuses_before_application_property_observation(custody):
    store,_=custody;current,session,action=await prepared(store)
    original_db=store.db;observed=[]
    class ForeignDatabase:
        @property
        def backend(self):observed.append('application callback');return original_db.backend
    with original_dispatch(session,action) as origin:
        async with store.original_action_result(SCOPE,session,action,guard=lambda:None) as result:
            await store.action(SCOPE,session.session_id,action.action_id,
                dispatch_revision=current.revision,_result_owner=result)
            store.db=ForeignDatabase()
            try:
                with pytest.raises(ExecutionAuthorityError):
                    await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no entry')))
                assert observed==[]
            finally:store.db=original_db
    assert await store.has_pending_actions(SCOPE,session.session_id)
