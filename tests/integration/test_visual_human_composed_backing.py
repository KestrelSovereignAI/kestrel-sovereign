"""Real original Human and shipped Step ownership on their one native engine."""
import copy
from dataclasses import replace

import pytest

from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.visual_computer.dispatch import original_human_dispatch
from tests.integration.test_visual_custody_backends import custody as custody
from tests.integration.test_visual_custody_backends import SCOPE
from tests.integration.test_visual_human_original_result_backends import prepared_human
from tests.integration.test_visual_step_backends import step_host as step_host


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [False, True], ids=["copied", "replaced-backing"])
async def test_copied_human_owner_close_cannot_retire_original_registration(custody, replacement):
    store, _ = custody
    _, session, lease, frame, request, _ = await prepared_human(store)
    with original_human_dispatch(session, lease, request, frame):
        async with store.original_human_action_result(
            SCOPE, session, request, guard=lambda: None
        ) as original:
            copied = copy.copy(original)
            assert copied._native_backing is original._native_backing
            if replacement:
                copied._native_backing = replace(original._native_backing, owner=copied)
                copied._registration = replace(original._registration, native_backing=copied._native_backing)
            assert store._original_results[original._acceptance] is original
            assert store.db.backend._original_result_holds[original] is original._closed
            with pytest.raises(ExecutionAuthorityError):
                copied.close()
            assert store._original_results[original._acceptance] is original
            assert store.db.backend._original_result_holds[original] is original._closed
            assert not original._closed.done()
    assert original._closed.done()
    assert not store._original_results and not store.db.backend._original_result_holds
    assert await store.has_pending_actions(SCOPE, session.session_id)


@pytest.mark.asyncio
async def test_original_human_close_requires_actual_native_writer_join(custody, monkeypatch):
    from kestrel_sovereign.storage import original_action_result as module
    from tests.integration.test_visual_human_original_result_backends import dispatch

    store, _ = custody
    current, session, lease, frame, request, _ = await prepared_human(store)
    seen = []
    original_await = module.await_owned_task
    with original_human_dispatch(session, lease, request, frame) as origin:
        async with store.original_human_action_result(
            SCOPE, session, request, guard=lambda: None
        ) as result:
            await dispatch(store, current, request, result)

            async def before_join(task):
                # Same original owner task, actual admitted native writer. The
                # observer always joins it, even if the close guard is broken.
                error = None
                try:
                    result.close()
                except ExecutionAuthorityError as refused:
                    error = refused
                outcome = await original_await(task)
                seen.append(outcome)
                assert error is not None
                assert store._original_results[result._acceptance] is result
                assert not result._closed.done()
                return outcome

            monkeypatch.setattr(module, 'await_owned_task', before_join)
            await result.not_dispatched(origin, origin.refuse_unentered(RuntimeError('no driver entered')))
            assert len(seen) == 1 and result.has_acknowledged_commit()
    assert result._closed.done()
    assert not store._original_results and not store.db.backend._original_result_holds
    action = await store.action(SCOPE, session.session_id, request.action.action_id)
    assert action.channel == 'human' and action.receipt.outcome == 'denied'


@pytest.mark.asyncio
async def test_genuine_human_native_writer_cannot_borrow_current_step(custody, monkeypatch):
    from tests.integration.test_visual_human_original_result_backends import dispatch

    store, _ = custody
    current, session, lease, frame, request, _ = await prepared_human(store)
    backend = store.db.backend
    original_native = backend._settle_original_action_result
    calls = []

    async def exact_human_writer(owner):
        with pytest.raises(ExecutionAuthorityError):
            await backend._commit_current_visual_step(owner)
        calls.append(owner)
        return await original_native(owner)

    monkeypatch.setattr(backend, '_settle_original_action_result', exact_human_writer)
    with original_human_dispatch(session, lease, request, frame) as origin:
        async with store.original_human_action_result(
            SCOPE, session, request, guard=lambda: None
        ) as owner:
            await dispatch(store, current, request, owner)
            await owner.not_dispatched(origin, origin.refuse_unentered(RuntimeError('no input')))
            assert calls == [owner] and owner.has_acknowledged_commit()
    stored = await store.action(SCOPE, session.session_id, request.action.action_id)
    assert stored.channel == 'human' and stored.receipt.outcome == 'denied'


@pytest.mark.asyncio
async def test_genuine_current_step_native_writer_cannot_borrow_human_late(step_host, monkeypatch):
    from tests.integration.test_visual_step_backends import invoke
    from kestrel_sovereign.visual_computer.step import PlanningRecord

    h = step_host
    backend = h.store.db.backend
    native = backend._commit_current_visual_step
    calls = []

    async def exact_step_writer(owner):
        with pytest.raises(ExecutionAuthorityError):
            await backend._settle_original_action_result(owner)
        calls.append(owner)
        return await native(owner)

    monkeypatch.setattr(backend, '_commit_current_visual_step', exact_step_writer)
    async def plan(owner):
        return {'action_request': None, 'answer': 'Done'}
    with h.bind():
        result = await invoke(h, plan)
        assert result['state'] == 'finished' and result['receipt'] is None
    assert len(calls) == 2
    stored = await h.store.action(h.runtime.scope(h.ctx), h.session.session_id, 'stable-native')
    assert type(stored) is PlanningRecord and stored.state == 'finished'
    assert not backend._original_result_holds


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["act", "status"])
async def test_native_step_planning_identity_never_becomes_manual_input(step_host, operation):
    from datetime import timedelta

    from kestrel_sdk.computer_use import ActionRequest, ControlRequest, TypeText
    from kestrel_sdk.computer_use.human_control import HumanActionRequest
    from kestrel_sdk.resources import ResourceConflict, ResourceNotFound

    from kestrel_sovereign.visual_computer.human import HumanComputerRuntime
    from kestrel_sovereign.visual_computer.runtime import VisualAction
    from tests.integration.test_visual_step_backends import invoke
    from tests.unit.visual_computer.test_human import Extension

    h = step_host
    async def plan(owner):
        return {"action_request": None, "answer": "Done"}
    with h.bind():
        await invoke(h, plan)
        original = await h.store.db.fetchone("SELECT payload,pending FROM visual_computer_actions")
        extension = Extension(h.runtime)
        h.adapter.human_control = lambda: extension
        h.runtime.human = HumanComputerRuntime(h.runtime)
        session = await h.runtime.control(h.ctx, h.session.session_id, ControlRequest(
            request_id="manual-takeover", session_id=h.session.session_id,
            expected_control_epoch=0, operation="takeover"), action=VisualAction.TAKEOVER)
        lease = await h.runtime.human.open(h.ctx, session.session_id, "manual-control")
        observed = await h.runtime.human.observe(h.ctx, session.session_id, "manual-control")
        frame = observed.observation.frame
        request = HumanActionRequest(control_id="manual-control", consumer_id=lease.request.consumer_id,
            action=ActionRequest(action_id="stable-native", session_id=session.session_id,
                observation_id=frame.observation_id, control_epoch=session.control_epoch,
                not_after=h.runtime._now()+timedelta(seconds=5),
                action=TypeText(kind="type", text="PRIVATE MANUAL INPUT")))
        if operation == "act":
            with pytest.raises(ResourceConflict):
                await h.runtime.human.act(h.ctx, session.session_id, "manual-control", request)
        else:
            with pytest.raises(ResourceNotFound):
                await h.runtime.human.action_status(h.ctx, session.session_id, "manual-control", "stable-native")
        after = await h.store.db.fetchone("SELECT payload,pending FROM visual_computer_actions")
        assert bytes(after[0]) == bytes(original[0]) and after[1] == original[1] == 0
        private = await h.runtime.human._record(h.runtime.scope(h.ctx), session.session_id,
            "manual-control", check=lambda: None)
        assert private.frame == observed.frame_record
        assert extension.effects == h.adapter.effects == h.runtime._inflight == 0
        assert not h.store._original_results and not h.store.db.backend._original_result_holds
