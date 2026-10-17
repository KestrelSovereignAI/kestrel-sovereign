"""Original invocation admission retires accepted children before registration.

Actual Agent constructor, lifecycle bookkeeping and SQLite worker are used;
the decorated SQL component is not process_input, Agent READY or real cognition.
"""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.invocation import bind_async_invocation
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from tests.unit.test_cleanup_task_admission import _partial_native_agent
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["allocation", "binding", "callback-before", "callback-after"])
@pytest.mark.parametrize("prior", [False, True])
async def test_invocation_admission_retires_exact_original_child_and_registration(
    tmp_path, monkeypatch, mode, prior,
):
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    assert not agent.llm_service.providers
    await db.execute("CREATE TABLE invocation_admission_effects (id INTEGER)")
    request_id = "original-refused-invocation"
    initial = agent.register_active_request(request_id, nested=False) if prior else None
    refusal = RuntimeError("actual original invocation " + mode + " refused")
    refused, accepted, cleanup_observations = [], [], []
    release = asyncio.Event()
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    actual_bind, actual_cleanup = type(agent).bind_request_operation, type(agent)._cleanup_cancelled_request
    owner = None

    @bind_async_invocation("request_id", track_request_lifecycle=True)
    async def invoke(self, request_id=None):
        await db.execute("INSERT INTO invocation_admission_effects VALUES (1)")
        await release.wait()
        return "unaccepted invocation"

    def binding(self, rid, operation):
        if self is agent and rid == request_id and mode == "binding":
            raise refusal
        return actual_bind(self, rid, operation)

    def cleanup(self, rid, *args, **kwargs):
        if self is agent and rid == request_id:
            cleanup_observations.append(tuple(task.done() for task in accepted))
            assert all(task.done() for task in accepted), "invocation registration retired before accepted child physically joined"
        return actual_cleanup(self, rid, *args, **kwargs)

    class RefusedBindingCallback(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if mode.startswith("callback") and getattr(callback, "__qualname__", "").endswith("bind_request_operation.<locals>.release"):
                if mode == "callback-after":
                    super().add_done_callback(callback, context=context)
                raise refusal
            return super().add_done_callback(callback, context=context)

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname.endswith(".run_isolated_operation"):
            if mode == "allocation":
                refused.append((coroutine, operation))
                raise refusal
            task = RefusedBindingCallback(coroutine, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    try:
        monkeypatch.setattr(type(agent), "bind_request_operation", binding)
        monkeypatch.setattr(type(agent), "_cleanup_cancelled_request", cleanup)
        loop.set_task_factory(factory)
        owner = create_owned_task(invoke(agent, request_id=request_id), retain_originals=True)
        outcome = await await_owned_task(owner)
        loop.set_task_factory(previous)
        assert outcome.error is refusal, "invocation admission failure replaced original refusal"
        assert all(task.done() for task in accepted), "invocation returned with original accepted child still pending"
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "refused invocation leaked unstarted operation"
        assert not any(task in members for members in vars(agent).get("_request_operation_tasks", {}).values() for task in accepted), "failed invocation binding left original operation published"
        assert agent._active_request_generation_counts == ({(request_id, initial): 1} if prior else {})
        assert agent._active_request_counts == ({request_id: 1} if prior else {})
        assert cleanup_observations
        if not prior:
            assert await agent.wait_for_request_completion(request_id, generation=agent._next_request_generation) is RequestCompletionDisposition.ABANDONED
        assert await db.fetchval("SELECT count(*) FROM invocation_admission_effects") == 0
    finally:
        loop.set_task_factory(previous)
        release.set()
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        if owner is not None:
            await await_owned_task(owner)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        monkeypatch.setattr(type(agent), "_cleanup_cancelled_request", actual_cleanup)
        for (rid, generation), count in tuple(agent._active_request_generation_counts.items()):
            for _ in range(count):
                actual_cleanup(agent, rid, disposition=RequestCompletionDisposition.ABANDONED, generation=generation)
        await agent.llm_service.close()
        await db.close()
