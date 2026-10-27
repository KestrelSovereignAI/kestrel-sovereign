"""The original same-row native claim precedes every planning call."""

import pytest
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown

from kestrel_sovereign.visual_computer.step import PlanningRecord, run_step
from tests.unit.visual_computer.test_inference import (
    inference as inference,  # noqa: PLC0414 - explicit pytest fixture reexport
)


async def step(
    x, ctx, sid, *, identifier="stable-step", prompt="Click once", calls=None
):
    async def plan(owner):
        if calls is not None:
            calls.append(owner)
        return await x.planner.plan(ctx, sid, prompt, _step_owner=owner)

    return await run_step(
        x.h.runtime,
        ctx,
        sid,
        identifier,
        prompt,
        planner=plan,
        configuration={"model": "scripted-unit-only"},
        configuration_current=lambda: None,
    )


@pytest.mark.asyncio
async def test_claim_before_inference_same_id_receipt_no_second_model(inference):
    x = inference
    h = x.h
    ctx = x.ctx
    calls = []
    with h.bind(ctx):
        session = x.session
        result = await step(x, ctx, session.session_id, calls=calls)
        assert result["receipt"]["action_id"] == "stable-step"
        assert result["receipt"]["outcome"] == "completed" and h.adapter.effects == 1
        assert len(calls) == 1 and len(x.posts) == 1
        assert (await step(x, ctx, session.session_id, calls=calls))[
            "receipt"
        ] == result["receipt"]
        assert len(calls) == 1 and len(x.posts) == 1 and h.adapter.effects == 1
        stored = await h.repository.action(
            h.runtime.scope(ctx), session.session_id, "stable-step"
        )
        assert (
            stored.step_intent.prompt == "Click once"
            and stored.request.action_id == "stable-step"
        )
        with pytest.raises(ResourceConflict):
            await step(x, ctx, session.session_id, prompt="Different")
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_no_action_host_terminal_is_not_sdk_receipt(inference):
    x = inference
    h = x.h
    ctx = x.ctx
    x.state.answer = {"action": None, "answer": "Finished"}
    with h.bind(ctx):
        session = x.session
        result = await step(x, ctx, session.session_id)
        assert result == {
            "action_id": "stable-step",
            "state": "finished",
            "answer": "Finished",
            "receipt": None,
        }
        stored = await h.repository.action(
            h.runtime.scope(ctx), session.session_id, "stable-step"
        )
        assert type(stored) is PlanningRecord and not stored.pending()
        assert not await h.repository.has_pending_actions(
            h.runtime.scope(ctx), session.session_id
        )
        assert await step(x, ctx, session.session_id) == result
        assert len(x.posts) == 1 and h.adapter.effects == 0


@pytest.mark.asyncio
async def test_failed_planning_stays_charged_no_replan_or_generic_abandon(inference):
    x = inference
    h = x.h
    ctx = x.ctx
    calls = []

    async def fail(owner):
        calls.append(owner)
        raise RuntimeError("bounded unit model failure")

    with h.bind(ctx):
        session = x.session

        async def invoke():
            return await run_step(
                h.runtime,
                ctx,
                session.session_id,
                "stable-step",
                "Click once",
                planner=fail,
                configuration={},
                configuration_current=lambda: None,
            )

        with pytest.raises(RuntimeError):
            await invoke()
        with pytest.raises(ResourceOutcomeUnknown):
            await invoke()
        assert len(calls) == 1 and h.adapter.effects == 0
        assert await h.repository.has_pending_actions(
            h.runtime.scope(ctx), session.session_id
        )
        with pytest.raises(ResourceOutcomeUnknown):
            await h.repository.action(
                h.runtime.scope(ctx),
                session.session_id,
                "stable-step",
                abandon_prepared=True,
            )
        with pytest.raises(ResourceOutcomeUnknown):
            await h.runtime.action_status(ctx, session.session_id, "stable-step")


@pytest.mark.asyncio
async def test_act_withdrawn_before_native_model_send_admits_no_model(inference):
    from kestrel_sdk.resources import ResourceError

    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.visual_computer.runtime import VisualAction

    x = inference
    h = x.h
    ctx = x.ctx
    withdrawn = []

    async def withdraw():
        withdrawn.append(True)

    x.state.before_request = withdraw
    with h.bind(ctx):
        binding = h.runtime.authority._bindings[id(ctx)]
        original = binding.check_current

        def policy(context, action, target):
            original(context, action, target)
            if withdrawn and action == VisualAction.ACT:
                raise PermissionError("original ACT permission withdrawn")

        binding.check_current = policy
        with pytest.raises((ExecutionAuthorityError, ResourceError, PermissionError)):
            await step(x, ctx, x.session.session_id)
        assert withdrawn and not x.posts and h.adapter.effects == 0


@pytest.mark.asyncio
async def test_configuration_withdrawal_after_dispatch_cas_before_effect(
    inference, monkeypatch
):
    x = inference
    h = x.h
    ctx = x.ctx
    withdrawn = []
    original = h.repository.action

    async def action(*args, **kwargs):
        result = await original(*args, **kwargs)
        if kwargs.get("dispatch_revision") is not None:
            withdrawn.append(True)
        return result

    monkeypatch.setattr(h.repository, "action", action)

    def current():
        if withdrawn:
            raise PermissionError("original visual config withdrawn")

    async def plan(owner):
        return await x.planner.plan(
            ctx, x.session.session_id, "Click once", _step_owner=owner
        )

    with h.bind(ctx):
        with pytest.raises(PermissionError):
            await run_step(
                h.runtime,
                ctx,
                x.session.session_id,
                "config-action",
                "Click once",
                planner=plan,
                configuration={},
                configuration_current=current,
            )
        assert withdrawn and h.adapter.effects == 0


@pytest.mark.asyncio
async def test_copied_step_owner_cannot_close_original_intent(inference):
    import copy

    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    x = inference
    h = x.h
    ctx = x.ctx

    async def plan(owner):
        with pytest.raises(ExecutionAuthorityError):
            copy.copy(owner).close()
        return {"action_request": None, "answer": "Done"}

    with h.bind(ctx):
        assert (
            await run_step(
                h.runtime,
                ctx,
                x.session.session_id,
                "copy-root",
                "Task",
                planner=plan,
                configuration={},
                configuration_current=lambda: None,
            )
        )["state"] == "finished"


@pytest.mark.asyncio
async def test_copied_current_writer_cannot_release_original_physical_hold(
    inference, monkeypatch
):
    import copy

    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.storage.current_visual_step import _CurrentVisualStepMutation

    x = inference
    h = x.h
    ctx = x.ctx
    original = _CurrentVisualStepMutation.commit_step

    async def capture(writer):
        with pytest.raises(ExecutionAuthorityError):
            copy.copy(writer).close()
        assert not writer._closed.done()
        return await original(writer)

    monkeypatch.setattr(_CurrentVisualStepMutation, "commit_step", capture)

    async def plan(owner):
        return {"action_request": None, "answer": "Done"}

    with h.bind(ctx):
        assert (
            await run_step(
                h.runtime,
                ctx,
                x.session.session_id,
                "copy-writer",
                "Task",
                planner=plan,
                configuration={},
                configuration_current=lambda: None,
            )
        )["state"] == "finished"


@pytest.mark.asyncio
@pytest.mark.parametrize("opaque", [False, True])
async def test_failed_original_claim_task_admission_never_arms_model(inference, opaque):
    import asyncio

    from kestrel_sovereign.execution_custody import (
        _explicit_control_chain,
        execution_cleanup_debts,
    )
    from tests.unit.test_host_context_native_cleanup import _actual_operation

    x = inference
    h = x.h
    ctx = x.ctx
    accepted = []
    calls = []
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    refusal = RuntimeError("original claim factory accepted then failed")

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if (
            getattr(getattr(operation, "cr_code", None), "co_name", None)
            == "_commit_current_visual_step"
        ):
            if opaque:

                async def outer():
                    return await coroutine

                actual = outer()
            else:
                actual = coroutine
            accepted.append(asyncio.Task(actual, loop=loop, **kwargs))
            raise refusal
        return asyncio.Task(coroutine, loop=loop, **kwargs)

    async def plan(owner):
        calls.append(owner)
        return {"action_request": None, "answer": "Done"}

    with h.bind(ctx):
        try:
            loop.set_task_factory(factory)
            with pytest.raises((RuntimeError, BaseExceptionGroup)) as denied:
                await run_step(
                    h.runtime,
                    ctx,
                    x.session.session_id,
                    "factory-claim",
                    "Task",
                    planner=plan,
                    configuration={},
                    configuration_current=lambda: None,
                )
            assert refusal in tuple(_explicit_control_chain(denied.value))
            assert execution_cleanup_debts(denied.value)
            assert not calls and not h.adapter.effects
        finally:
            loop.set_task_factory(previous)
            await asyncio.gather(*accepted, return_exceptions=True)
        assert accepted and not h.db.backend._original_result_holds
        assert (await h.db.fetchone("SELECT COUNT(*) FROM visual_computer_actions"))[
            0
        ] == 0
        # A fresh original INSERT/ACK, not the absence read or failed Task,
        # is the ONLY anchor that can now admit one planner.
        assert (
            await run_step(
                h.runtime,
                ctx,
                x.session.session_id,
                "factory-claim",
                "Task",
                planner=plan,
                configuration={},
                configuration_current=lambda: None,
            )
        )["state"] == "finished"
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_terminal_step_uses_original_grants_without_unneeded_status(inference):
    from kestrel_sovereign.visual_computer.runtime import VisualAction

    x = inference
    h = x.h
    with h.bind(x.ctx):
        first = await step(x, x.ctx, x.session.session_id)
    from dataclasses import replace

    ctx = replace(
        x.ctx, granted_actions=x.ctx.granted_actions - {VisualAction.STATUS.value}
    )
    with h.bind(ctx):
        assert await step(x, ctx, x.session.session_id) == first
    assert len(x.posts) == 1 and h.adapter.effects == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["frame", "deadline", "epoch"])
async def test_step_promotes_only_original_current_frame_deadline_epoch(
    inference, changed
):
    from datetime import timedelta

    x = inference
    h = x.h

    async def plan(owner):
        result = await x.planner.plan(
            x.ctx, x.session.session_id, "Click once", _step_owner=owner
        )
        if changed == "frame":
            await h.runtime.observe(x.ctx, x.session.session_id)
        elif changed == "deadline":
            result["action_request"]["not_after"] = (
                h.runtime._now() - timedelta(seconds=1)
            ).isoformat()
        else:
            result["action_request"]["control_epoch"] += 1
        return result

    with h.bind(x.ctx), pytest.raises(ResourceConflict):
        await run_step(
            h.runtime,
            x.ctx,
            x.session.session_id,
            "original-current",
            "Click once",
            planner=plan,
            configuration={},
            configuration_current=lambda: None,
        )
    assert len(x.posts) == 1 and h.adapter.effects == 0
    with h.bind(x.ctx), pytest.raises(ResourceOutcomeUnknown):
        await run_step(
            h.runtime,
            x.ctx,
            x.session.session_id,
            "original-current",
            "Click once",
            planner=plan,
            configuration={},
            configuration_current=lambda: None,
        )
    assert len(x.posts) == 1


@pytest.mark.asyncio
async def test_no_action_answer_bounds_preserve_original_planning_debt(inference):
    from pydantic import ValidationError

    x = inference
    h = x.h

    async def plan(owner):
        return {"action_request": None, "answer": "x" * 2001}

    with h.bind(x.ctx):
        with pytest.raises(ValidationError):
            await run_step(
                h.runtime,
                x.ctx,
                x.session.session_id,
                "oversized-answer",
                "Done?",
                planner=plan,
                configuration={},
                configuration_current=lambda: None,
            )
        record = await h.repository.action(
            h.runtime.scope(x.ctx), x.session.session_id, "oversized-answer"
        )
        assert (
            type(record) is PlanningRecord
            and record.pending()
            and record.answer is None
        )
    assert not x.posts and h.adapter.effects == 0
