import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sovereign.visual_computer.egress import ActiveCallEgress
from tests.unit.visual_computer.test_runtime import opened


@pytest.mark.asyncio
async def test_workspace_borrow_is_paused_original_and_cannot_use_action_lane(harness):
    h=harness
    ctx=h.context()
    with h.bind(ctx):
        session, action=await opened(h,ctx)
    egress=ActiveCallEgress()
    paused=session.model_copy(update={"state":"paused","controller":"none"})
    async def job(callback, candidate=paused):
        async with callback(candidate,action.action_id) as current:
            current()
    with pytest.raises(OperatorAuthorizationError):
        async with egress.activate_transfer(session,action.action_id,lambda:None,deadline=action.not_after):
            pass
    async with egress.activate_transfer(paused,action.action_id,lambda:None,deadline=action.not_after):
        with pytest.raises(OperatorAuthorizationError):
            await job(egress.transfer_operation)
        await asyncio.create_task(job(egress.transfer_operation))
        with pytest.raises(OperatorAuthorizationError):
            await asyncio.create_task(job(egress.operation))
        with pytest.raises(OperatorAuthorizationError):
            await asyncio.create_task(job(egress.transfer_operation,session))
    with pytest.raises(OperatorAuthorizationError):
        await asyncio.create_task(job(egress.transfer_operation))
    async with egress.activate(session,action.action_id,lambda:None,deadline=action.not_after):
        with pytest.raises(OperatorAuthorizationError):
            await asyncio.create_task(job(egress.transfer_operation))


@pytest.mark.asyncio
async def test_priority_stop_joins_original_workspace_borrow_before_ack(harness):
    h=harness
    ctx=h.context()
    with h.bind(ctx):
        session, action=await opened(h,ctx)
    paused=session.model_copy(update={"state":"paused","controller":"none"})
    egress=ActiveCallEgress()
    entered,closed=asyncio.Event(),asyncio.Event()
    async def transfer():
        async with egress.transfer_operation(paused,action.action_id):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
    async with egress.activate_transfer(paused,action.action_id,lambda:None,deadline=action.not_after):
        task=asyncio.create_task(transfer())
        await entered.wait()
        await egress.revoke_session(paused.scope,paused.session_id)
        assert closed.is_set() and task.done()
    assert not egress._calls


@pytest.mark.asyncio
async def test_egress_borrows_only_exact_original_active_rpc(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h,ctx)
    egress = ActiveCallEgress()
    checked = []
    def check():
        checked.append(True)
    async def job(candidate=session, aid=action.action_id):
        async with egress.operation(candidate,aid) as current:
            current()
    with pytest.raises(OperatorAuthorizationError):
        await job()
    async with egress.activate(session,action.action_id,check,deadline=action.not_after):
        # A callback is an explicit borrow by an owned gateway task, never a
        # new origin permission or copied task-context authority.
        with pytest.raises(OperatorAuthorizationError):
            await job()
        await asyncio.create_task(job())
        with pytest.raises(OperatorAuthorizationError):
            await asyncio.create_task(job(aid="other-action"))
        with pytest.raises(OperatorAuthorizationError):
            await asyncio.create_task(job(candidate=session.model_copy(update={"control_epoch":1})))
        foreign = session.model_copy(update={"scope":session.scope.model_copy(update={"agent_id":"other"})})
        with pytest.raises(OperatorAuthorizationError):
            await asyncio.create_task(job(candidate=foreign))
    with pytest.raises(OperatorAuthorizationError):
        await asyncio.create_task(job())
    assert checked


@pytest.mark.asyncio
async def test_rpc_exit_revokes_and_physically_joins_socket_job(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h,ctx)
    egress = ActiveCallEgress()
    entered, closed = asyncio.Event(), asyncio.Event()
    async def socket_job():
        async with egress.operation(session,action.action_id):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                # Socket close is part of the authorized job's physical join.
                await asyncio.sleep(0)
                closed.set()
    async with egress.activate(session,action.action_id,lambda:None,deadline=action.not_after):
        task = asyncio.create_task(socket_job())
        await entered.wait()
    assert task.done() and closed.is_set()
    assert not egress._calls


@pytest.mark.asyncio
async def test_priority_revoke_is_owner_exact_and_joins_before_control(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h,ctx)
    egress = ActiveCallEgress()
    entered, closed = asyncio.Event(), asyncio.Event()
    async def socket_job():
        async with egress.operation(session,action.action_id) as current:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
    async with egress.activate(session,action.action_id,lambda:None,deadline=action.not_after):
        task = asyncio.create_task(socket_job())
        await entered.wait()
        await egress.revoke_session(session.scope.model_copy(update={"agent_id":"other"}),session.session_id)
        assert not task.done()
        await egress.revoke_session(session.scope,session.session_id)
        assert task.done() and closed.is_set()
        with pytest.raises(OperatorAuthorizationError):
            await asyncio.create_task(socket_job())


@pytest.mark.asyncio
async def test_original_current_checker_and_deadline_fence_network_effect(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h,ctx)
    egress = ActiveCallEgress()
    revoked = False
    def current():
        if revoked:
            raise OperatorAuthorizationError("current original grant revoked")
    async def job():
        async with egress.operation(session,action.action_id) as check:
            check()
    async with egress.activate(session,action.action_id,current,deadline=action.not_after):
        await asyncio.create_task(job())
        revoked = True
        with pytest.raises(OperatorAuthorizationError):
            await asyncio.create_task(job())
    with pytest.raises(OperatorAuthorizationError):
        async with egress.activate(session,action.action_id,current,deadline=datetime.now(UTC)-timedelta(seconds=1)):
            pass


@pytest.mark.asyncio
async def test_runtime_act_owns_egress_and_stop_revokes_before_driver_control(harness):
    from kestrel_sdk.computer_use import ControlRequest
    from kestrel_sovereign.visual_computer.runtime import VisualAction
    h = harness
    ctx = h.context()
    entered, closed = asyncio.Event(), asyncio.Event()
    async def network():
        async with h.runtime.egress.operation(session,action.action_id):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
    async def before_effect():
        await asyncio.create_task(network())
    with h.bind(ctx):
        session, action = await opened(h,ctx)
        h.adapter.before_effect = before_effect
        active = asyncio.create_task(h.runtime.act(ctx,action))
        await entered.wait()
        original_control = h.adapter.control
        async def priority_control(session,request):
            assert closed.is_set()
            return await original_control(session,request)
        h.adapter.control = priority_control
        stop = ControlRequest(request_id="stop",session_id=session.session_id,expected_control_epoch=0,operation="pause")
        await h.runtime.control(ctx,session.session_id,stop,action=VisualAction.STOP)
        with pytest.raises(asyncio.CancelledError):
            await active
        assert h.adapter.effects == 0
        assert not h.runtime.egress._calls


@pytest.mark.asyncio
async def test_repeated_origin_cancellation_cannot_abandon_gateway_cleanup(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h,ctx)
    egress = ActiveCallEgress()
    entered, closing, finish_close = asyncio.Event(), asyncio.Event(), asyncio.Event()
    jobs=[]
    async def socket_job():
        async with egress.operation(session,action.action_id):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closing.set()
                await finish_close.wait()
    async def origin():
        async with egress.activate(session,action.action_id,lambda:None,deadline=action.not_after):
            jobs.append(asyncio.create_task(socket_job()))
            await asyncio.Event().wait()
    task = asyncio.create_task(origin())
    await entered.wait()
    task.cancel()
    await closing.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not jobs[0].done()
    finish_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert jobs[0].done() and not egress._calls
