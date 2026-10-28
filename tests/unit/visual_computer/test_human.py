"""Original host policy/custody with SDK doubles; actual Docker proof is separate."""
import asyncio
from datetime import timedelta
import hashlib
from uuid import uuid4

import pytest
from kestrel_sdk.computer_use import ActionReceipt,ActionRequest,ControlRequest,ScreenFrame,ScreenObservation,TypeText
from kestrel_sdk.computer_use.human_control import (
    HumanActionReceipt,HumanActionRequest,HumanControlLease,HumanControlRevocationTombstone,HumanScreenObservation,
)
from kestrel_sdk.resources import ResourceConflict,ResourceNotFound,ResourceOutcomeUnknown
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sovereign.visual_computer.human import HumanComputerRuntime,HumanRecord
from kestrel_sovereign.visual_computer.docker_human import human_operation_id
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_runtime import opened


class Extension:
    def __init__(self,r):
        self.r=r;self.lease=None;self.frame=None;self.receipts={};self.tombstones={};self.before=None;self.effects=0
    async def _call(self,s,cid,operation,run,aid=None):
        async def dispatch():
            guard=self.r.egress.human_status_operation if operation in {'status','action_status'} else self.r.egress.human_operation
            async with guard(s,human_operation_id(cid,operation,aid)) as check:
                if self.before:await self.before()
                check()
                return run()
        return await asyncio.create_task(dispatch())
    async def open(self,s,request):
        def run():
            self.lease=HumanControlLease(request=request,request_sha256=request.digest(),state='active',accepted_at=self.r._now())
            return self.lease
        return await self._call(s,request.control_id,'open',run)
    async def status(self,s,request):
        return await self._call(s,request.control_id,'status',lambda:self.lease)
    async def observe(self,s,lease):
        def run():
            image=b'private-human-pixels'
            self.frame=ScreenFrame(observation_id='human-'+uuid4().hex,session_id=s.session_id,control_epoch=s.control_epoch,
                observed_at=self.r._now(),width=100,height=100,image_sha256=hashlib.sha256(image).hexdigest(),media_type='image/png')
            return HumanScreenObservation(control_id=lease.request.control_id,consumer_id=lease.request.consumer_id,
                observation=ScreenObservation(frame=self.frame,image=image))
        return await self._call(s,lease.request.control_id,'observe',run)
    async def act(self,s,lease,request):
        def run():
            self.effects+=1
            receipt=HumanActionReceipt(control_id=lease.request.control_id,consumer_id=lease.request.consumer_id,
                receipt=ActionReceipt(action_id=request.action.action_id,session_id=s.session_id,observation_id=request.action.observation_id,
                    control_epoch=s.control_epoch,outcome='completed',recorded_at=self.r._now()))
            self.receipts[request.action.action_id]=receipt
            return receipt
        return await self._call(s,lease.request.control_id,'act',run,request.action.action_id)
    async def action_status(self,s,lease,aid):
        return await self._call(s,lease.request.control_id,'action_status',lambda:self.receipts[aid],aid)
    async def revoke(self,s,request):
        if self.lease is None or request.control_id != self.lease.request.control_id:
            if request.control_id not in self.tombstones:
                self.tombstones[request.control_id]=HumanControlRevocationTombstone(request=request,scope=s.scope,
                    lease_id=s.lease_id,driver_id=s.driver_id,fenced_at=self.r._now())
            return self.tombstones[request.control_id].model_copy(update={'request':request})
        request.require_target(self.lease,s)
        self.lease=self.lease.model_copy(update={'state':'revoked','revoked_at':self.r._now()})
        return self.lease


async def setup(h,ctx):
    s,_=await opened(h,ctx)
    extension=Extension(h.runtime)
    h.adapter.human_control=lambda:extension
    h.runtime.human=HumanComputerRuntime(h.runtime)
    s=await h.runtime.control(ctx,s.session_id,ControlRequest(request_id='takeover',session_id=s.session_id,expected_control_epoch=0,operation='takeover'),action=VisualAction.TAKEOVER)
    lease=await h.runtime.human.open(ctx,s.session_id,'human')
    return s,lease,extension


async def input_request(h,ctx,s,lease):
    observed=await h.runtime.human.observe(ctx,s.session_id,'human')
    return HumanActionRequest(control_id='human',consumer_id=lease.request.consumer_id,
        action=ActionRequest(action_id='manual',session_id=s.session_id,observation_id=observed.observation.frame.observation_id,
            control_epoch=s.control_epoch,not_after=h.runtime._now()+timedelta(seconds=5),action=TypeText(kind='type',text='PRIVATE-HUMAN-TEXT')))


def original_human_rpc(runtime, session):
    """Observe the actual host/SDK owner relation without replacing either."""
    task = asyncio.current_task()
    calls = [call for call in runtime.egress._calls.values()
             if call.sdk_owner is task and call.session == session]
    assert len(calls) == 1
    assert calls[0].origin is not task and not calls[0].origin.done()
    return calls[0]


@pytest.mark.asyncio
async def test_private_human_lane_never_enters_agent_status_and_custody_is_encrypted(harness):
    h=harness;ctx=h.context()
    with h.bind(ctx):
        s,lease,extension=await setup(h,ctx)
        request=await input_request(h,ctx,s,lease)
        receipt=await h.runtime.human.act(ctx,s.session_id,'human',request)
        assert receipt.receipt.outcome=='completed' and extension.effects==1
        assert await h.runtime.human.act(ctx,s.session_id,'human',request)==receipt
        assert await h.runtime.human.action_status(ctx,s.session_id,'human','manual')==receipt
        with pytest.raises(ResourceNotFound):await h.runtime.action_status(ctx,s.session_id,'manual')
        with pytest.raises(ResourceConflict):await h.runtime.observe(ctx,s.session_id)
        status=await h.runtime.status(ctx,s.session_id)
        assert 'human-' not in str(status) and request.action.observation_id not in str(status)
        stored=await h.runtime.human._record(h.runtime.scope(ctx),s.session_id,'human',check=lambda:None)
        assert stored.frame is None
    raw=h.path.read_bytes()
    assert b'PRIVATE-HUMAN-TEXT' not in raw and b'private-human-pixels' not in raw


@pytest.mark.asyncio
async def test_separate_view_and_input_grants_are_current_after_dispatch_wait(harness):
    h=harness;ctx=h.context()
    with h.bind(ctx):
        s,lease,extension=await setup(h,ctx)
        req=await input_request(h,ctx,s,lease)
        async def revoke():h.revoked.add(id(ctx))
        extension.before=revoke
        with pytest.raises(OperatorAuthorizationError):await h.runtime.human.act(ctx,s.session_id,'human',req)
        assert extension.effects==0
    ctx2=h.context(actions={VisualAction.HUMAN_VIEW.value})
    with h.bind(ctx2):
        with pytest.raises(OperatorAuthorizationError):await h.runtime.human.act(ctx2,s.session_id,'human',req)


@pytest.mark.asyncio
async def test_human_callback_cannot_borrow_agent_or_workspace_authority(harness):
    h=harness;ctx=h.context()
    with h.bind(ctx):
        s,lease,_=await setup(h,ctx)
    identifier=human_operation_id('human','observe')
    async def job(callback):
        async with callback(s,identifier) as check:check()
    async with h.runtime.egress.activate_human(s,identifier,lambda:None,deadline=ctx.expires_at):
        await asyncio.create_task(job(h.runtime.egress.human_operation))
        for callback in (h.runtime.egress.operation,h.runtime.egress.transfer_operation,h.runtime.egress.human_status_operation):
            with pytest.raises(OperatorAuthorizationError):await asyncio.create_task(job(callback))
    with pytest.raises(OperatorAuthorizationError):await asyncio.create_task(job(h.runtime.egress.human_operation))


@pytest.mark.asyncio
async def test_delivery_cannot_borrow_native_foreign_consumer_or_replacement_epoch(harness):
    h=harness;ctx=h.context()
    with h.bind(ctx):
        s,lease,_=await setup(h,ctx)
    async def job(session, identifier, *, metadata=False, native=False):
        guard = (h.runtime.egress.human_operation(session,identifier) if native else
            h.runtime.egress.human_delivery_operation(session,identifier,metadata=metadata))
        async with guard as check:check()
    identifier='["human","original-consumer",1,"response"]'
    async with h.runtime.egress.activate_human_delivery(s,identifier,lambda:None,deadline=ctx.expires_at):
        await asyncio.create_task(job(s,identifier))
        for session,key,options in (
            (s,identifier,{'metadata':True}), (s,identifier,{'native':True}),
            (s,identifier.replace('original-consumer','foreign-consumer'),{}),
            (s.model_copy(update={'control_epoch':s.control_epoch+1}),identifier,{}),
            (s.model_copy(update={'lease_id':'replacement'}),identifier,{})):
            with pytest.raises(OperatorAuthorizationError):
                await asyncio.create_task(job(session,key,**options))
    with pytest.raises(OperatorAuthorizationError):await asyncio.create_task(job(s,identifier))


@pytest.mark.asyncio
async def test_handback_requires_share_confirmation_revokes_and_captures_new_agent_frame(harness):
    h=harness;ctx=h.context()
    with h.bind(ctx):
        s,lease,extension=await setup(h,ctx)
        observed=await h.runtime.human.observe(ctx,s.session_id,'human')
        request=ControlRequest(request_id='resume',session_id=s.session_id,expected_control_epoch=1,operation='resume')
        with pytest.raises(ResourceConflict,match='confirm'):await h.runtime.control(ctx,s.session_id,request,action=VisualAction.HANDBACK)
        resumed=await h.runtime.control(ctx,s.session_id,request,action=VisualAction.HANDBACK,share_screen=True)
        assert resumed.control_epoch==2 and extension.lease.state=='revoked'
        record=await h.repository.get(h.runtime.scope(ctx),session_id=s.session_id)
        assert record.frame is not None and record.frame.observation_id!=observed.observation.frame.observation_id
        stored=await h.runtime.human._record(h.runtime.scope(ctx),s.session_id,'human',check=lambda:None)
        assert stored.frame is None and stored.lease.state=='revoked'
        with pytest.raises((ValueError,ResourceConflict)):await h.runtime.human.observe(ctx,s.session_id,'human')


@pytest.mark.asyncio
async def test_preopen_tombstone_blocks_late_open_and_unknown_input_cannot_replay(harness):
    h=harness;ctx=h.context()
    with h.bind(ctx):
        s,lease,extension=await setup(h,ctx)
        tombstone=await h.runtime.human.revoke(ctx,s.session_id,'late','revoke-late',s.control_epoch)
        assert isinstance(tombstone,HumanControlRevocationTombstone)
        with pytest.raises(ResourceConflict):await h.runtime.human.open(ctx,s.session_id,'late')
        req=await input_request(h,ctx,s,lease)
        async def lost(s,lease,request):raise ResourceOutcomeUnknown('lost')
        extension.act=lost
        with pytest.raises(ResourceOutcomeUnknown):await h.runtime.human.act(ctx,s.session_id,'human',req)
        with pytest.raises(ResourceOutcomeUnknown):await h.runtime.human.act(ctx,s.session_id,'human',req)
        request=ControlRequest(request_id='resume',session_id=s.session_id,expected_control_epoch=1,operation='resume')
        with pytest.raises(ResourceOutcomeUnknown):await h.runtime.control(ctx,s.session_id,request,action=VisualAction.HANDBACK,share_screen=True)


@pytest.mark.asyncio
async def test_manual_stop_keeps_workspace_recording_fenced_until_confirmed_handback(harness):
    h=harness;ctx=h.context()
    with h.bind(ctx):
        s,lease,_=await setup(h,ctx)
        stopped=await h.runtime.control(ctx,s.session_id,ControlRequest(request_id='stop',session_id=s.session_id,
            expected_control_epoch=1,operation='pause'),action=VisualAction.STOP)
        assert stopped.controller=='none'
        async with h.runtime.operation(ctx,VisualAction.TRANSFER):
            record=await h.repository.get(h.runtime.scope(ctx),session_id=s.session_id)
            assert record.human_private
            with pytest.raises(ResourceConflict):h.runtime._transfer_ready(record)
        with pytest.raises(ResourceConflict):
            async with h.runtime.session_operation(ctx,s.session_id):
                pytest.fail('private Human state reached workspace transport')


@pytest.mark.asyncio
async def test_wrong_revocation_epoch_does_not_poison_original_host_custody(harness):
    h=harness;ctx=h.context()
    with h.bind(ctx):
        s,lease,_=await setup(h,ctx)
        with pytest.raises((ValueError,ResourceConflict)):
            await h.runtime.human.revoke(ctx,s.session_id,'human','bad-revoke',s.control_epoch+1)
        result=await h.runtime.human.revoke(ctx,s.session_id,'human','good-revoke',s.control_epoch)
        assert result.state=='revoked'


@pytest.mark.asyncio
async def test_new_preopen_close_id_confirms_existing_fence_and_original_remains_reconcilable(harness):
    h=harness;ctx=h.context()
    with h.bind(ctx):
        s,_,_=await setup(h,ctx)
        original=await h.runtime.human.revoke(ctx,s.session_id,'late','first-close',s.control_epoch)
        confirmed=await h.runtime.human.revoke(ctx,s.session_id,'late','second-close',s.control_epoch)
        assert confirmed.request.request_id=='second-close'
        assert confirmed.fenced_at==original.fenced_at
        with pytest.raises(ResourceConflict):await h.runtime.human.open(ctx,s.session_id,'late')
        retried=await h.runtime.human.revoke(ctx,s.session_id,'late','first-close',s.control_epoch)
        assert retried.request==original.request
