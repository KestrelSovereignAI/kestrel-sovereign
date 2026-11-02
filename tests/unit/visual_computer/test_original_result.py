"""Production host paths retain only original facts across global turn Stop."""
import asyncio
import sys

import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, bind_execution_custody_snapshot, is_execution_control_error,
)
from kestrel_sovereign.visual_computer.docker import DockerComputerDriver, PROVIDER
from tests.integration.test_visual_original_result_backends import OriginalTurn
from tests.unit.visual_computer.test_runtime import opened


@pytest.mark.asyncio
@pytest.mark.parametrize('at',['host','native'])
async def test_global_turn_cancellation_settles_original_known_no_dispatch(harness,monkeypatch,tmp_path,at):
    h=harness;ctx=h.context();h.adapter.driver_id=PROVIDER
    entered=asyncio.Event();children=[]
    with h.bind(ctx):
        session,request=await opened(h,ctx)
        driver=DockerComputerDriver(docker_socket='/tmp/owned-original-result.sock',egress_guard=h.runtime.egress.operation)
        driver._connections[session.session_id]=object()
        h.runtime.drivers={PROVIDER:driver}
        if at=='host':
            action=h.repository.action
            async def block(*args,**kwargs):
                value=await action(*args,**kwargs)
                if kwargs.get('dispatch_revision') is not None:
                    entered.set();await asyncio.Event().wait()
                return value
            monkeypatch.setattr(h.repository,'action',block)
        else:
            executable=tmp_path/'owned-native-preflight'
            executable.write_text(f'#!{sys.executable} -B\nimport time\ntime.sleep(60)\n')
            executable.chmod(0o700);driver.boundary.executable=str(executable)
            launch=asyncio.create_subprocess_exec
            async def capture(*args,**kwargs):
                process=await launch(*args,**kwargs);children.append(process);entered.set();return process
            monkeypatch.setattr(asyncio,'create_subprocess_exec',capture)
            async def bound(*args,**kwargs):return await driver.boundary.invoke(['container','inspect','c'*64])
            monkeypatch.setattr(driver,'_bound',bound)
        turn=ExecutionCustody(OriginalTurn('sqlite'))
        with bind_execution_custody_snapshot((turn,)):
            task=asyncio.create_task(h.runtime.act(ctx,request))
            await asyncio.wait_for(entered.wait(),2)
            turn.revoke('global stop retires original turn');task.cancel()
            with pytest.raises(BaseException) as denied:await task
            assert isinstance(denied.value,asyncio.CancelledError) or is_execution_control_error(denied.value)
        assert all(process.returncode is not None for process in children)
        assert children or at=='host'
        # Restore only the test seam; no new authority is given to the old task.
        monkeypatch.undo()
        stored=await h.repository.action(h.runtime.scope(ctx),session.session_id,request.action_id)
        assert stored.receipt.outcome=='denied' and stored.receipt.reason_code=='host_not_dispatched'
        assert not await h.repository.has_pending_actions(h.runtime.scope(ctx),session.session_id)
        assert await h.runtime.act(ctx,request)==stored.receipt
        assert h.adapter.effects==0


@pytest.mark.asyncio
async def test_global_quote_cancellation_needs_fresh_genuine_reconcile_not_old_task(harness,monkeypatch):
    h=harness;ctx=h.context();h.repository.max_requests=1
    quote=h.adapter.quote;entered=asyncio.Event()
    async def blocked(request):entered.set();await asyncio.Event().wait()
    monkeypatch.setattr(h.adapter,'quote',blocked)
    request=h.request(ctx)
    with h.bind(ctx):
        turn=ExecutionCustody(OriginalTurn('sqlite'))
        with bind_execution_custody_snapshot((turn,)):
            task=asyncio.create_task(h.runtime.admit(ctx,request));await entered.wait()
            turn.revoke('global quote cancellation');task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
        record=await h.repository.get(h.runtime.scope(ctx),request_id=request.request_id)
        assert record.phase=='admitted' and record.lease is None and h.adapter.acquires==0
    # A distinct original authenticated endpoint grant performs the normal
    # reconcile. No retired context is rebound or used for the write.
    fresh=h.context()
    with h.bind(fresh):
        result=await h.runtime.reconcile_request(fresh,request.request_id)
        assert result['retired'] and not result['acquired'] and not result['released']
        monkeypatch.setattr(h.adapter,'quote',quote)
        assert (await h.runtime.admit(fresh,h.request(fresh,request_id='fresh'))).session_id!=record.session_id


@pytest.mark.asyncio
async def test_turn_retirement_without_cancellation_fences_actual_native_stdin(harness,monkeypatch,tmp_path):
    from datetime import UTC,datetime
    from kestrel_sdk.computer_use import ActionReceipt
    h=harness;ctx=h.context();h.adapter.driver_id=PROVIDER
    received=tmp_path/'application-stdin'
    executable=tmp_path/'owned-native-rpc'
    executable.write_text(f'#!{sys.executable} -B\nimport sys\nfrom pathlib import Path\ndata=sys.stdin.buffer.read()\nif data:Path({str(received)!r}).write_bytes(data)\nprint("{{}}")\n')
    executable.chmod(0o700)
    with h.bind(ctx):
        session,request=await opened(h,ctx)
        driver=DockerComputerDriver(docker_socket='/tmp/owned-original-result.sock',egress_guard=h.runtime.egress.operation)
        driver.boundary.executable=str(executable)
        h.runtime.drivers={PROVIDER:driver}
        turn=ExecutionCustody(OriginalTurn('sqlite'))
        async def original_native_call(session,request,*,current=None):
            # Bounded native-stdin fixture: the same production driver/owned
            # boundary sends actual OS stdin, without claiming worker admission.
            turn.revoke('original turn revoked without task cancellation')
            await driver.boundary.rpc('c'*64,{'operation':'act','id':'owned-original'},current=current)
            return ActionReceipt(action_id=request.action_id,session_id=session.session_id,
                observation_id=request.observation_id,control_epoch=request.control_epoch,
                outcome='completed',recorded_at=datetime.now(UTC))
        monkeypatch.setattr(driver,'_perform_act',original_native_call)
        with bind_execution_custody_snapshot((turn,)):
            with pytest.raises(BaseException) as retired:await h.runtime.act(ctx,request)
            assert is_execution_control_error(retired.value)
        assert not received.exists()
        receipt=await h.runtime.action_status(ctx,session.session_id,request.action_id)
        assert receipt.outcome=='denied' and receipt.reason_code=='host_not_dispatched'
        assert not await h.repository.has_pending_actions(h.runtime.scope(ctx),session.session_id)
