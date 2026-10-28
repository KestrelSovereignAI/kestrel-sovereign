"""Manual input keeps its own original no-dispatch fact through global Stop."""
import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionCustody, bind_execution_custody_snapshot, is_execution_control_error,
)
from tests.integration.test_visual_original_result_backends import OriginalTurn
from tests.unit.visual_computer.test_human import setup, input_request


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_original_manual_host_refusal_survives_global_turn_retirement(
    harness, monkeypatch, cancel,
):
    h, entered, resume = harness, asyncio.Event(), asyncio.Event()
    context = h.context()
    action = h.repository.action

    async def after_dispatch_cas(*args, **kwargs):
        result = await action(*args, **kwargs)
        if kwargs.get("dispatch_revision") is not None:
            entered.set()
            await resume.wait()
        return result

    with h.bind(context):
        session, lease, extension = await setup(h, context)
        request = await input_request(h, context, session, lease)
        monkeypatch.setattr(h.repository, "action", after_dispatch_cas)
        turn = ExecutionCustody(OriginalTurn(h.db.backend_type))
        with bind_execution_custody_snapshot((turn,)):
            task = asyncio.create_task(
                h.runtime.human.act(context, session.session_id, "human", request)
            )
            try:
                await asyncio.wait_for(entered.wait(), 2)
                turn.revoke("original manual turn stopped")
                if cancel:
                    task.cancel()
                resume.set()
                with pytest.raises(BaseException) as stopped:
                    await task
                assert isinstance(stopped.value, asyncio.CancelledError) or is_execution_control_error(stopped.value)
            finally:
                resume.set()
                await asyncio.gather(task, return_exceptions=True)
        monkeypatch.setattr(h.repository, "action", action)
        stored = await action(h.runtime.scope(context), session.session_id, request.action.action_id)
        assert extension.effects == 0 and h.runtime._inflight == 0
        assert stored.receipt is not None, "exact original no-dispatch manual result was not stored"
        assert stored.channel == "human" and stored.control_id == "human"
        assert stored.receipt.outcome == "denied" and stored.receipt.reason_code == "host_not_dispatched"
        assert not await h.repository.has_pending_actions(h.runtime.scope(context), session.session_id)


async def native_input(h, context, tmp_path, monkeypatch):
    """Production driver/RPC/owned child with only daemon binding substituted."""
    import sys
    from kestrel_sovereign.visual_computer.docker import DockerComputerDriver, PROVIDER
    h.adapter.driver_id = PROVIDER
    session,lease,_ = await setup(h,context)
    request = await input_request(h,context,session,lease)
    driver = DockerComputerDriver(docker_socket='/tmp/owned-human-result.sock')
    driver.human_guard = h.runtime.egress.human_operation
    driver.human_status_guard = h.runtime.egress.human_status_operation
    driver._connections[session.session_id] = h.adapter.last_connection
    h.runtime.drivers = {PROVIDER:driver}
    async def bound(*args,**kwargs):return {'container_id':'c'*64}
    monkeypatch.setattr(driver,'_bound',bound)
    received = tmp_path/'owned-stdin'
    executable = tmp_path/'owned-native-worker'
    executable.write_text('#!'+sys.executable+' -B\nimport sys\nfrom pathlib import Path\ndata=sys.stdin.buffer.read()\nPath('+repr(str(received))+').write_bytes(data)\nprint("{}")\n')
    executable.chmod(0o700)
    driver.boundary.executable = str(executable)
    return session,lease,request,driver,received


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel',[False,True])
async def test_human_actual_native_launch_revocation_never_writes_input(harness,monkeypatch,tmp_path,cancel):
    h=harness;context=h.context();entered,resume=asyncio.Event(),asyncio.Event();children=[]
    with h.bind(context):
        session,lease,request,driver,received=await native_input(h,context,tmp_path,monkeypatch)
        launch=asyncio.create_subprocess_exec
        async def delayed(*args,**kwargs):
            process=await launch(*args,**kwargs);children.append(process);entered.set();await resume.wait();return process
        monkeypatch.setattr(asyncio,'create_subprocess_exec',delayed)
        turn=ExecutionCustody(OriginalTurn('sqlite'))
        with bind_execution_custody_snapshot((turn,)):
            task=asyncio.create_task(h.runtime.human.act(context,session.session_id,'human',request))
            try:
                await asyncio.wait_for(entered.wait(),3)
                turn.revoke('original turn retired at actual native stdin')
                if cancel:task.cancel()
                resume.set()
                with pytest.raises(BaseException) as stopped:await task
                assert isinstance(stopped.value,asyncio.CancelledError) or is_execution_control_error(stopped.value)
            finally:
                resume.set();await asyncio.gather(task,return_exceptions=True)
        assert children and all(p.returncode is not None for p in children)
        assert not received.exists(), 'retired human operation delivered native stdin'
        saved=await h.repository.action(h.runtime.scope(context),session.session_id,request.action.action_id)
        assert saved.channel=='human' and saved.receipt.outcome=='denied'
        assert saved.receipt.reason_code=='host_not_dispatched'
        assert not await h.repository.has_pending_actions(h.runtime.scope(context),session.session_id)
        assert not h.runtime.egress._calls and not driver._workspace_borrowers


@pytest.mark.asyncio
async def test_human_partial_native_stdin_retains_unknown_original_input(harness,monkeypatch,tmp_path):
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    h=harness;context=h.context();children=[];writes=[]
    with h.bind(context):
        session,lease,request,driver,received=await native_input(h,context,tmp_path,monkeypatch)
        launch=asyncio.create_subprocess_exec
        async def partial(*args,**kwargs):
            process=await launch(*args,**kwargs);children.append(process);stream=process.stdin
            class Partial:
                def write(self,data):stream.write(data[:17]);writes.append(17);raise OSError('owned partial stdin')
                async def drain(self):await stream.drain()
                def close(self):stream.close()
            process.stdin=Partial()
            return process
        monkeypatch.setattr(asyncio,'create_subprocess_exec',partial)
        with pytest.raises(ResourceOutcomeUnknown):await h.runtime.human.act(context,session.session_id,'human',request)
        assert writes==[17] and children and all(p.returncode is not None for p in children)
        saved=await h.repository.action(h.runtime.scope(context),session.session_id,request.action.action_id)
        assert saved.channel=='human' and saved.receipt is None
        assert await h.repository.has_pending_actions(h.runtime.scope(context),session.session_id)
        assert not h.runtime.egress._calls and not driver._workspace_borrowers


@pytest.mark.asyncio
async def test_arbitrary_human_rpc_refusal_is_not_original_native_no_send(harness,monkeypatch,tmp_path):
    from kestrel_sdk.resources import ResourceConflict
    h=harness;context=h.context()
    with h.bind(context):
        session,lease,request,driver,_=await native_input(h,context,tmp_path,monkeypatch)
        async def arbitrary_rpc(*args,**kwargs):
            raise ResourceConflict('arbitrary RPC error has no physical settlement provenance')
        monkeypatch.setattr(driver.boundary,'rpc',arbitrary_rpc)
        with pytest.raises(ResourceConflict):await h.runtime.human.act(context,session.session_id,'human',request)
        saved=await h.repository.action(h.runtime.scope(context),session.session_id,request.action.action_id)
        assert saved.receipt is None and await h.repository.has_pending_actions(h.runtime.scope(context),session.session_id)


@pytest.mark.asyncio
async def test_human_native_refusal_preserves_original_explicit_cause_graph(harness, monkeypatch, tmp_path):
    """Zero-input driver; real Human issuer and original native settlement."""
    from kestrel_sdk.resources import ResourceOutcomeUnknown
    from kestrel_sovereign.visual_computer.docker_human import DockerHumanControl

    h = harness
    context = h.context()
    native = ResourceOutcomeUnknown('original native command unknown')
    child = asyncio.CancelledError('original joined Human borrower')
    child.__cause__ = native
    original = ResourceOutcomeUnknown('original Human driver interruption')
    original.__cause__ = child
    entries = []
    with h.bind(context):
        session, _, request, driver, received = await native_input(h, context, tmp_path, monkeypatch)
        async def zero_input(self, selected, control_id, operation, arguments, action_id=None):
            assert self.driver is driver and selected == session
            assert control_id == 'human' and operation == 'act'
            assert action_id == request.action.action_id
            entries.append(action_id)
            # The production native_human_dispatch issuer accounts for zero
            # commands/input and issues its own exact original negative fact.
            raise original
        monkeypatch.setattr(DockerHumanControl, '_call', zero_input)
        with pytest.raises(ResourceOutcomeUnknown) as caught:
            await h.runtime.human.act(context, session.session_id, 'human', request)
        saved = await h.repository.action(h.runtime.scope(context), session.session_id, request.action.action_id)
        assert saved.channel == 'human' and saved.control_id == 'human'
        assert saved.receipt.outcome == 'denied' and saved.receipt.reason_code == 'host_not_dispatched'
        assert entries == [request.action.action_id] and not received.exists()
        assert h.runtime._inflight == 0 and not h.runtime.egress._calls
        assert not driver._workspace_borrowers and not h.repository._original_results
        assert not h.db.backend._original_result_holds
        assert not await h.repository.has_pending_actions(h.runtime.scope(context), session.session_id)
        assert caught.value is original
        assert original.__cause__ is child and child.__cause__ is native
        replay = await h.runtime.human.act(context, session.session_id, 'human', request)
        assert replay.receipt == saved.receipt and entries == [request.action.action_id]
