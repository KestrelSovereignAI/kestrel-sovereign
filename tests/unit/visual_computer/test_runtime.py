import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from kestrel_sdk.computer_use import ActionRequest, ControlRequest
from kestrel_sdk.operator import OperatorAuthorizationError
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown
from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
from kestrel_sovereign.visual_computer.runtime import VisualAction, VisualComputerRuntime
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.async_database import AsyncDatabase


async def opened(harness, ctx):
    session = await harness.runtime.admit(ctx, harness.request(ctx))
    observation = await harness.runtime.observe(ctx, session.session_id)
    action = ActionRequest(action_id="action", session_id=session.session_id, observation_id=observation.frame.observation_id,
        control_epoch=session.control_epoch, not_after=datetime.now(UTC)+timedelta(seconds=20), action={"kind":"click","x":10,"y":10})
    return session, action


@pytest.mark.asyncio
async def test_real_encrypted_custody_and_original_receipt_survive_restart(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        receipt = await h.runtime.act(ctx, action)
        assert await h.runtime.act(ctx, action) == receipt
        assert h.adapter.effects == 1
        reopened_db = await AsyncDatabase.sqlite(str(h.path), schema_initializer=h.empty)
        restarted = VisualComputerRuntime(VisualCustodyRepository(reopened_db), h.runtime.authority,
            providers=h.runtime.providers, drivers=h.runtime.drivers)
        assert await restarted.action_status(ctx, session.session_id, action.action_id) == receipt
        record = await restarted.repository.get(restarted.scope(ctx), session_id=session.session_id)
        assert record.connection.credential.get_secret_value() == "private-secret"
        assert await restarted.reconcile_session(ctx, session.session_id) == session
        await reopened_db.close()
    raw = h.path.read_bytes()
    assert b"private-secret" not in raw and b"private-endpoint" not in raw and b"private pixels" not in raw


@pytest.mark.asyncio
async def test_owner_agent_and_deserialized_context_never_authorize(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, _ = await opened(h, ctx)
    for other in (h.context(principal="other"), h.context(agent=None), h.context(agent="other")):
        with h.bind(other), pytest.raises(ResourceNotFound):
            await h.runtime.status(other, session.session_id)
    with pytest.raises(OperatorAuthorizationError):
        await h.runtime.status(ctx, session.session_id)
    spoof = h.context()
    with h.bind(spoof), pytest.raises(OperatorAuthorizationError):
        await h.runtime.admit(spoof, h.request(h.context(principal="victim")))


@pytest.mark.asyncio
async def test_expiry_revocation_and_private_modes_fence_operations(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        h.revoked.add(id(ctx))
        with pytest.raises(OperatorAuthorizationError):
            await h.runtime.act(ctx, action)
        assert h.adapter.effects == 0
        h.revoked.clear()
        h.holder.privacy_mode = PrivacyMode.EPHEMERAL
        with pytest.raises(OperatorAuthorizationError):
            await h.runtime.observe(ctx, session.session_id)
        assert await h.runtime.release(ctx, session.session_id) == {"session_id":session.session_id,"released":True}
    now = datetime.now(UTC)
    expired = replace(h.context(), issued_at=now-timedelta(seconds=2), expires_at=now-timedelta(seconds=1))
    with pytest.raises(OperatorAuthorizationError), h.bind(expired):
        await h.runtime.status(expired, session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"observation_id":"stale"}, {"control_epoch":1},
    {"not_after":datetime(2020,1,1,tzinfo=UTC)}, {"action":{"kind":"click","x":100,"y":0}},
])
async def test_stale_frame_epoch_deadline_and_bounds_before_effect(harness, change):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        _, action = await opened(h, ctx)
        bad = ActionRequest.model_validate({**action.model_dump(), **change})
        with pytest.raises(ResourceConflict):
            await h.runtime.act(ctx, bad)
        assert h.adapter.effects == 0


@pytest.mark.asyncio
async def test_lost_action_receipt_never_replays_effect(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        h.adapter.unknown_action = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, action)
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, action)
        assert h.adapter.effects == 1
        receipt = await h.runtime.action_status(ctx, session.session_id, action.action_id)
        assert receipt.outcome == "completed"
        assert await h.runtime.act(ctx, action) == receipt
        changed = action.model_copy(update={"action":action.action.model_copy(update={"x":20})})
        with pytest.raises(ResourceConflict):
            await h.runtime.act(ctx, changed)


@pytest.mark.asyncio
async def test_unknown_acquire_and_release_keep_original_custody(harness):
    h = harness
    ctx = h.context()
    request = h.request(ctx)
    with h.bind(ctx):
        h.adapter.unknown_acquire = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.admit(ctx, request)
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.admit(ctx, request)
        assert h.adapter.acquires == 1
        session = await h.runtime.reconcile_request(ctx, request.request_id)
        assert h.adapter.acquires == 2
        h.adapter.unknown_release = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.release(ctx, session.session_id)
        record = await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
        assert record.lease.state == "ready" and record.connection is not None
        assert record.phase == "fenced"
        h.adapter.unknown_release = False
        assert (await h.runtime.release(ctx, session.session_id))["released"] is True


@pytest.mark.asyncio
async def test_takeover_and_stop_bypass_saturated_action_history(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        h.repository.max_actions = 1
        await h.runtime.act(ctx, action)
        h.runtime._inflight = h.runtime.policy.max_inflight_actions
        takeover = ControlRequest(request_id="take", session_id=session.session_id, expected_control_epoch=0, operation="takeover")
        human = await h.runtime.control(ctx, session.session_id, takeover, action=VisualAction.TAKEOVER)
        assert human.controller == "human" and human.control_epoch == 1
        with pytest.raises(ResourceConflict):
            await h.runtime.observe(ctx, session.session_id)
        handback = ControlRequest(request_id="back", session_id=session.session_id, expected_control_epoch=1, operation="resume")
        returned = await h.runtime.control(ctx, session.session_id, handback, action=VisualAction.HANDBACK)
        assert returned.control_epoch == 2
        with pytest.raises(ResourceConflict):
            await h.runtime.act(ctx, action.model_copy(update={"action_id":"after", "control_epoch":2}))
        stop = ControlRequest(request_id="stop", session_id=session.session_id, expected_control_epoch=2, operation="pause")
        assert (await h.runtime.control(ctx, session.session_id, stop, action=VisualAction.STOP)).controller == "none"


@pytest.mark.asyncio
async def test_restart_does_not_reopen_or_adopt_replacement_worker(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, _ = await opened(h, ctx)
        restarted = VisualComputerRuntime(h.repository, h.runtime.authority, providers=h.runtime.providers, drivers=h.runtime.drivers)
        with pytest.raises(ResourceConflict):
            await restarted.observe(ctx, session.session_id)
        h.adapter.sessions[session.session_id] = session.model_copy(update={"control_epoch":99})
        with pytest.raises(ResourceOutcomeUnknown):
            await restarted.reconcile_session(ctx, session.session_id)
        with pytest.raises(ResourceConflict):
            await restarted.observe(ctx, session.session_id)


@pytest.mark.asyncio
async def test_cancelled_accepted_action_requires_status_and_safety_still_runs(harness):
    h = harness
    ctx = h.context()
    entered, never = asyncio.Event(), asyncio.Event()
    async def blocked():
        entered.set()
        await never.wait()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        h.adapter.before_effect = blocked
        task = asyncio.create_task(h.runtime.act(ctx, action))
        await entered.wait()
        stop = ControlRequest(request_id="stop", session_id=session.session_id, expected_control_epoch=0, operation="pause")
        stopped = await asyncio.wait_for(h.runtime.control(ctx, session.session_id, stop, action=VisualAction.STOP), 1)
        assert stopped.controller == "none"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert h.runtime._inflight == 0
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, action)
        assert h.adapter.effects == 0


@pytest.mark.asyncio
async def test_authority_and_privacy_transition_join_accepted_effect(harness):
    h = harness
    ctx = h.context()
    entered, finish = asyncio.Event(), asyncio.Event()
    async def blocked():
        entered.set()
        await finish.wait()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        h.adapter.before_effect = blocked
        effect = asyncio.create_task(h.runtime.act(ctx, action))
        await entered.wait()
        async def privacy_transition():
            async with h.privacy_gate.exclusive():
                h.holder.privacy_mode = PrivacyMode.EPHEMERAL
        transition = asyncio.create_task(privacy_transition())
        await asyncio.sleep(0)
        assert not transition.done()
        finish.set()
        assert (await effect).outcome == "completed"
        await transition
        with pytest.raises(OperatorAuthorizationError):
            await h.runtime.observe(ctx, session.session_id)


@pytest.mark.asyncio
async def test_lifecycle_withdrawal_fences_driver_and_keeps_physical_custody(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, _ = await opened(h, ctx)
    assert await h.runtime.fence_owned_sessions(h.runtime.scope(ctx)) == []
    record = await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
    assert record.session.controller == "none" and record.phase == "fenced"
    assert record.lease.state == "ready" and record.connection is not None


@pytest.mark.asyncio
async def test_restarted_session_requires_pending_action_status_before_resume(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        h.adapter.unknown_action = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, action)
        restarted = VisualComputerRuntime(h.repository, h.runtime.authority, providers=h.runtime.providers, drivers=h.runtime.drivers)
        with pytest.raises(ResourceOutcomeUnknown):
            await restarted.reconcile_session(ctx, session.session_id)
        await restarted.action_status(ctx, session.session_id, action.action_id)
        await restarted.reconcile_session(ctx, session.session_id)
        record = await h.repository.get(h.runtime.scope(ctx), session_id=session.session_id)
        assert record.frame is None


@pytest.mark.asyncio
async def test_deadline_cannot_outlive_current_grant(harness):
    h = harness
    ctx = h.context(seconds=5)
    with h.bind(ctx):
        _, action = await opened(h, ctx)
        with pytest.raises(ResourceConflict):
            await h.runtime.act(ctx, action)
        assert h.adapter.effects == 0


@pytest.mark.asyncio
async def test_wrong_receipt_retains_original_pending_intent(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        original = h.adapter.act
        async def wrong(session, request):
            return (await original(session, request)).model_copy(update={"observation_id":"other-frame"})
        h.adapter.act = wrong
        with pytest.raises(ResourceConflict):
            await h.runtime.act(ctx, action)
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx, action)
        assert h.adapter.effects == 1
        assert (await h.runtime.action_status(ctx, session.session_id, action.action_id)).outcome == "completed"


@pytest.mark.asyncio
async def test_handback_replay_keeps_current_ready_authority(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, _ = await opened(h, ctx)
        takeover = ControlRequest(request_id="take", session_id=session.session_id, expected_control_epoch=0, operation="takeover")
        await h.runtime.control(ctx,session.session_id,takeover,action=VisualAction.TAKEOVER)
        back = ControlRequest(request_id="back", session_id=session.session_id, expected_control_epoch=1, operation="resume")
        first = await h.runtime.control(ctx,session.session_id,back,action=VisualAction.HANDBACK)
        assert await h.runtime.control(ctx,session.session_id,back,action=VisualAction.HANDBACK) == first
        assert (await h.runtime.observe(ctx,session.session_id)).frame.control_epoch == 2


@pytest.mark.asyncio
async def test_unknown_input_blocks_different_action_until_original_status(harness):
    h = harness
    ctx = h.context()
    with h.bind(ctx):
        session, action = await opened(h, ctx)
        h.adapter.unknown_action = True
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx,action)
        frame = await h.runtime.observe(ctx,session.session_id)
        second = action.model_copy(update={"action_id":"second", "observation_id":frame.frame.observation_id})
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.act(ctx,second)
        assert h.adapter.effects == 1
