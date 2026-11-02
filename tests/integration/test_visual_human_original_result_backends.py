"""Distinct human original results on real SQLite and PostgreSQL."""
import asyncio
import copy
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.computer_use import ActionRequest, ComputerSession, ScreenFrame
from kestrel_sdk.computer_use.human_control import (
    HumanActionRequest, HumanControlRequest, HumanControlLease, HumanScreenFrame,
)
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, bind_execution_custody_snapshot,
    is_execution_control_error,
)
from kestrel_sovereign.visual_computer.dispatch import original_dispatch, original_human_dispatch
from kestrel_sovereign.visual_computer.human import HumanRecord
from tests.integration.test_visual_custody_backends import custody as custody, record, SCOPE
from tests.integration.test_visual_original_result_backends import OriginalTurn


async def prepared_human(store, *, consumer_id=None):
    current = await store.admit(SCOPE, record(), lambda:None)
    session = ComputerSession(session_id=current.session_id, scope=SCOPE, lease_id='lease',
        driver_id='driver', surface='browser', capabilities=frozenset({'screenshot','type','click','takeover'}),
        state='paused', controller='human', control_epoch=1, expires_at=current.request.expires_at)
    current.session, current.phase = session, 'fenced'
    current.human_private, current.human_control_id, current.human_control_epoch = True, 'human', 1
    await store.save(SCOPE, current)
    now = datetime.now(UTC)
    consumer = (store._address('human-consumer', SCOPE.model_dump_json(), session.session_id, 'human')
                if consumer_id is None else consumer_id)
    control = HumanControlRequest(control_id='human', consumer_id=consumer, scope=SCOPE,
        session_id=session.session_id, lease_id=session.lease_id, driver_id=session.driver_id,
        control_epoch=1, issued_at=now, expires_at=now+timedelta(seconds=60),
        capabilities=session.capabilities-{'takeover'}, max_frame_age_seconds=15)
    lease = HumanControlLease(request=control, request_sha256=control.digest(), state='active', accepted_at=now)
    frame = HumanScreenFrame(control_id='human', consumer_id=consumer, frame=ScreenFrame(
        observation_id='private-human-frame', session_id=session.session_id, control_epoch=1,
        observed_at=now, width=100, height=100, image_sha256='b'*64, media_type='image/png'))
    human = HumanRecord(request=control, lease=lease, frame=frame)
    await store.private_record(SCOPE, session.session_id, 'human', 'human', HumanRecord, accept=human)
    request = HumanActionRequest(control_id='human', consumer_id=consumer, action=ActionRequest(
        action_id='original-human-action', session_id=session.session_id, observation_id=frame.frame.observation_id,
        control_epoch=1, not_after=now+timedelta(seconds=10), action={'kind':'type','text':'PRIVATE-ORIGINAL-HUMAN'}))
    await store.action(SCOPE, session.session_id, request.action.action_id, accept=request.action,
        channel='human', control_id='human', prepare=True)
    return current, session, lease, frame, request, human


async def dispatch(store, current, request, result):
    await store.action(SCOPE, current.session_id, request.action.action_id,
        dispatch_revision=current.revision, _result_owner=result)


@pytest.mark.asyncio
async def test_only_exact_human_negative_fact_survives_global_stop_and_private_frame_clear(custody):
    store,reopen = custody
    current,session,lease,frame,request,human = await prepared_human(store)
    turn = ExecutionCustody(OriginalTurn(store.db.backend_type))
    with bind_execution_custody_snapshot((turn,)):
        with original_human_dispatch(session,lease,request,frame) as origin:
            async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None) as result:
                assert not store.db.owns_open_transaction
                await dispatch(store,current,request,result)
                # Genuine current private frame consumption/control retirement
                # cannot erase an already owed historical negative action fact.
                human.frame = None
                human.lease = lease.model_copy(update={'state':'revoked','revoked_at':datetime.now(UTC)})
                await store.private_record(SCOPE,session.session_id,'human','human',HumanRecord,update=human)
                current.session = session.model_copy(update={'state':'ready','controller':'agent','control_epoch':2})
                current.phase = 'active'
                await store.save(SCOPE,current)
                turn.revoke('original human turn stopped')
                with pytest.raises(Exception) as denied:await store.db.fetchone('SELECT 1')
                assert is_execution_control_error(denied.value)
                await result.not_dispatched(origin,origin.refuse_unentered(asyncio.CancelledError('original stop')))
                with pytest.raises(Exception) as denied:await store.db.execute('DELETE FROM visual_computer_actions')
                assert is_execution_control_error(denied.value)
    action = await store.action(SCOPE,session.session_id,request.action.action_id)
    assert action.channel=='human' and action.control_id=='human' and action.request==request.action
    assert action.receipt.outcome=='denied' and action.receipt.reason_code=='host_not_dispatched'
    assert not await store.has_pending_actions(SCOPE,session.session_id)
    assert (await store.get(SCOPE,session_id=session.session_id)).session==current.session
    assert (await store.private_record(SCOPE,session.session_id,'human','human',HumanRecord))==human
    db = await reopen()
    try:
        from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
        assert await VisualCustodyRepository(db).action(SCOPE,session.session_id,request.action.action_id)==action
    finally:await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('wrong',['copy','row','key','origin','private-source','foreign-task','sql','closed','backend','database','subclass'])
async def test_human_original_result_rejects_foreign_or_replaced_owners(custody,wrong):
    store,_ = custody
    current,session,lease,frame,request,_ = await prepared_human(store)
    with original_human_dispatch(session,lease,request,frame) as origin:
        async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None) as result:
            await dispatch(store,current,request,result)
            fact=origin.refuse_unentered(RuntimeError('no driver entry'))
            candidate=result
            if wrong=='copy':candidate=copy.copy(result)
            elif wrong=='row':result._identity=(*result._identity[:2],'foreign-action')
            elif wrong=='key':result._key=b'x'*32
            elif wrong=='origin':
                class ForeignOrigin:
                    @property
                    def binding(self):raise AssertionError('foreign callback ran after Stop')
                result._origin=ForeignOrigin()
            elif wrong=='private-source':result._registration=replace(result._registration,private_source=(result._acceptance.private_identity,b'foreign'))
            elif wrong=='sql':result._query='DELETE FROM visual_computer_actions'
            elif wrong=='closed':result.close()
            elif wrong=='backend':result._backend=object()
            elif wrong=='database':result._database=object()
            elif wrong=='subclass':
                class ExtendedResult(type(result)):pass
                candidate=copy.copy(result)
                candidate.__class__=ExtendedResult
            with pytest.raises(ExecutionAuthorityError):
                if wrong=='foreign-task':await asyncio.create_task(candidate.not_dispatched(origin,fact))
                else:await candidate.not_dispatched(origin,fact)
            if wrong=='backend':result._backend=store.db.backend
        assert await store.has_pending_actions(SCOPE,session.session_id)
        assert not store._original_results and not store.db.backend._original_result_holds


@pytest.mark.asyncio
async def test_human_and_agent_original_profiles_never_convert_channels(custody):
    store,_=custody
    current,session,lease,frame,request,_=await prepared_human(store)
    with original_dispatch(session,request.action) as agent:
        with pytest.raises(ResourceConflict):
            async with store.original_action_result(SCOPE,session,request.action,guard=lambda:None):pass
        with original_human_dispatch(session,lease,request,frame) as human:
            async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None) as result:
                await dispatch(store,current,request,result)
                with pytest.raises(ExecutionAuthorityError):
                    await result.not_dispatched(agent,agent.refuse_unentered(RuntimeError('foreign agent fact')))
                await result.not_dispatched(human,human.refuse_unentered(RuntimeError('original human fact')))
    assert (await store.action(SCOPE,session.session_id,request.action.action_id)).channel=='human'


@pytest.mark.asyncio
@pytest.mark.parametrize('wrong',['consumer','control','frame','lease','channel'])
async def test_human_result_issuance_requires_original_private_binding(custody,wrong):
    store,_=custody
    _,session,lease,frame,request,_=await prepared_human(store)
    if wrong=='consumer':request=request.model_copy(update={'consumer_id':'foreign-consumer'})
    elif wrong=='control':request=request.model_copy(update={'control_id':'foreign-control'})
    elif wrong=='frame':frame=frame.model_copy(update={'frame':frame.frame.model_copy(update={'observation_id':'foreign-frame'})})
    elif wrong=='lease':lease=lease.model_copy(update={'state':'unknown'})
    elif wrong=='channel':
        owner,sk=store._owner(SCOPE),store._address('session',session.session_id)
        row=await store.action(SCOPE,session.session_id,request.action.action_id)
        row.channel,row.control_id='agent',None
        await store.db.execute('UPDATE visual_computer_actions SET payload = ?', (store._seal(owner,f'{sk}:{request.action.action_id}',row),))
    with original_human_dispatch(session,lease,request,frame):
        with pytest.raises((ResourceConflict, ResourceNotFound)):
            async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None):pass
    assert not store._original_results and not store.db.backend._original_result_holds


@pytest.mark.asyncio
async def test_consistent_foreign_consumer_records_never_mint_original_human_owner(custody):
    store,_=custody
    # Admit the deliberately coherent foreign fixture from the outset. The
    # canonical row now correctly refuses rewriting an accepted request.
    _,session,lease,frame,request,_=await prepared_human(store, consumer_id='foreign-consumer')
    with original_human_dispatch(session,lease,request,frame):
        with pytest.raises(ResourceConflict,match='consumer differs'):
            async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None):pass
    assert not store._original_results and not store.db.backend._original_result_holds


@pytest.mark.asyncio
async def test_human_owner_cannot_be_issued_after_driver_entry(custody):
    store,_=custody
    _,session,lease,frame,request,_=await prepared_human(store)
    with original_human_dispatch(session,lease,request,frame) as origin:
        origin.enter_driver()
        with pytest.raises(ResourceConflict):
            async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None):pass
    assert not store._original_results and not store.db.backend._original_result_holds


@pytest.mark.asyncio
async def test_possible_human_partial_input_never_becomes_negative_receipt(custody):
    store,_=custody;current,session,lease,frame,request,_=await prepared_human(store)
    with original_human_dispatch(session,lease,request,frame) as origin:
        async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None) as result:
            await dispatch(store,current,request,result)
            fact=origin.refuse_unentered(ResourceConflict('generic refusal is not proof'))
            origin.trace.input_started=True
            with pytest.raises(ResourceOutcomeUnknown):await result.not_dispatched(origin,fact)
    assert await store.has_pending_actions(SCOPE,session.session_id)
    assert (await store.action(SCOPE,session.session_id,request.action.action_id)).receipt is None


@pytest.mark.asyncio
async def test_human_original_ciphertext_cas_never_overwrites_new_revision(custody):
    store,_=custody;current,session,lease,frame,request,_=await prepared_human(store)
    with original_human_dispatch(session,lease,request,frame) as origin:
        async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None) as result:
            await dispatch(store,current,request,result)
            owner,sk=result._identity[:2]
            row=await store.action(SCOPE,session.session_id,request.action.action_id)
            newer=store._seal(owner,f'{sk}:{request.action.action_id}',row)
            await store.db.execute('UPDATE visual_computer_actions SET payload = ?', (newer,))
            with pytest.raises(BaseException):await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no entry')))
    assert bytes((await store.db.fetchone('SELECT payload FROM visual_computer_actions'))[0])==newer
    assert await store.has_pending_actions(SCOPE,session.session_id)


@pytest.mark.asyncio
async def test_human_original_native_schema_seal_refuses_changed_relation(custody):
    store,_=custody
    current,session,lease,frame,request,_=await prepared_human(store)
    with original_human_dispatch(session,lease,request,frame) as origin:
        async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None) as result:
            await dispatch(store,current,request,result)
            expected=result._expected
            await store.db.execute('ALTER TABLE visual_computer_actions ADD COLUMN human_override TEXT')
            with pytest.raises(BaseException) as refused:
                await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('original no input')))
            assert is_execution_control_error(refused.value)
    row=await store.db.fetchone('SELECT payload,pending FROM visual_computer_actions')
    assert bytes(row[0])==expected and row[1]==1


@pytest.mark.asyncio
@pytest.mark.parametrize('retire',['key','privacy'])
async def test_human_final_native_probe_rechecks_original_key_and_privacy(custody,monkeypatch,retire):
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
    store,_=custody;current,session,lease,frame,request,_=await prepared_human(store)
    entered,resume=asyncio.Event(),asyncio.Event();key=store._key;captured={}
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
        with original_human_dispatch(session,lease,request,frame) as origin:
            async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None) as result:
                await dispatch(store,current,request,result)
                captured['ciphertext']=result._expected
                install()
                with pytest.raises(BaseException) as refused:
                    await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no human input')))
                assert is_execution_control_error(refused.value)
    gate=DurablePersistenceGate();shared=gate.shared();await shared.__aenter__()
    task=asyncio.create_task(original_owner())
    try:
        await asyncio.wait_for(entered.wait(),3)
        if retire=='privacy':await shared.__aexit__(None,None,None)
        else:store._key=b'k'*32
        resume.set();await asyncio.wait_for(task,3)
    finally:
        resume.set();await asyncio.gather(task,return_exceptions=True)
        store._key=key
        if retire!='privacy':await shared.__aexit__(None,None,None)
    row=await store.db.fetchone('SELECT payload,pending FROM visual_computer_actions')
    assert bytes(row[0])==captured['ciphertext'] and row[1]==1
    assert not gate.locked() and not store.db.backend._original_result_holds


@pytest.mark.asyncio
@pytest.mark.parametrize('replacement',['closed','fresh','copy'])
async def test_human_original_privacy_hold_never_becomes_a_fresh_or_copied_hold(custody,replacement):
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate, _DURABLE_GATE_SHARED_HOLDS
    store,_=custody;current,session,lease,frame,request,_=await prepared_human(store)
    gate=DurablePersistenceGate();shared=gate.shared();await shared.__aenter__()
    with original_human_dispatch(session,lease,request,frame) as origin:
        context=store.original_human_action_result(SCOPE,session,request,guard=lambda:None)
        result=await context.__aenter__()
        try:
            await dispatch(store,current,request,result)
            clone=copy.copy(result._registration.privacy_holds[0][0])
            await shared.__aexit__(None,None,None)
            fact=origin.refuse_unentered(RuntimeError('original human no input'))
            async def denied():
                with pytest.raises(ExecutionAuthorityError,match='privacy hold retired'):
                    await result.not_dispatched(origin,fact)
            if replacement=='fresh':
                async with gate.shared():await denied()
            elif replacement=='copy':
                token=_DURABLE_GATE_SHARED_HOLDS.set((clone,))
                try:await denied()
                finally:_DURABLE_GATE_SHARED_HOLDS.reset(token)
            else:await denied()
        finally:await context.__aexit__(None,None,None)
    assert await store.has_pending_actions(SCOPE,session.session_id)


@pytest.mark.asyncio
async def test_human_native_writer_and_database_close_join_after_repeated_cancellation(custody,monkeypatch):
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    store,reopen= custody;current,session,lease,frame,request,_=await prepared_human(store)
    entered,resume=asyncio.Event(),asyncio.Event();settle=store.db.backend._settle_original_action_result
    async def hold_after_commit(journal):
        value=await settle(journal);entered.set();await resume.wait();return value
    monkeypatch.setattr(store.db.backend,'_settle_original_action_result',hold_after_commit)
    async def original_owner():
        with original_human_dispatch(session,lease,request,frame) as origin:
            async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None) as result:
                await dispatch(store,current,request,result)
                await result.not_dispatched(origin,origin.refuse_unentered(asyncio.CancelledError('original human stop')))
    task=asyncio.create_task(original_owner());await asyncio.wait_for(entered.wait(),3)
    closing=asyncio.create_task(store.db.close())
    try:
        task.cancel();await asyncio.sleep(.01);task.cancel();await asyncio.sleep(.01)
        assert not task.done() and not closing.done() and store.db.backend._original_result_holds
        resume.set()
        with pytest.raises(BaseException) as withdrawn:await task
        assert execution_commit_outcome(withdrawn.value)=='committed'
        await asyncio.wait_for(closing,3)
    finally:
        resume.set();await asyncio.gather(task,closing,return_exceptions=True)
    db=await reopen()
    try:
        from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
        stored=await VisualCustodyRepository(db).action(SCOPE,session.session_id,request.action.action_id)
        assert stored.channel=='human' and stored.receipt.outcome=='denied'
    finally:await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('custody',['sqlite'],indirect=True)
async def test_human_lost_native_sqlite_commit_ack_retains_original_unknown(custody,monkeypatch):
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    store,_=custody;current,session,lease,frame,request,_=await prepared_human(store)
    connection=store.db.backend._connection;commit=connection.commit
    async def lose_ack():await commit();raise OSError('controlled human native ACK loss')
    with original_human_dispatch(session,lease,request,frame) as origin:
        async with store.original_human_action_result(SCOPE,session,request,guard=lambda:None) as result:
            await dispatch(store,current,request,result)
            monkeypatch.setattr(connection,'commit',lose_ack)
            with pytest.raises(BaseException) as unknown:
                await result.not_dispatched(origin,origin.refuse_unentered(RuntimeError('no input')))
            assert execution_commit_outcome(unknown.value)=='unknown'
    monkeypatch.setattr(connection,'commit',commit)
    assert (await store.action(SCOPE,session.session_id,request.action.action_id)).receipt.outcome=='denied'
