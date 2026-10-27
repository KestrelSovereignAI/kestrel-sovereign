"""Original Stop generation retirement on settlement admission/publication refusal.

Actual Agent constructor/lifecycle and SQLite worker are used, not Agent READY,
a routed cognition turn, distributed Stop or a successful durable ACK.
"""
import asyncio
import inspect

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from kestrel_sovereign.signals import OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry
from tests.unit.test_cleanup_task_admission import _partial_native_agent
from tests.unit.test_host_context_native_cleanup import _actual_operation


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["allocation", "binding", "callback", "canonical-before", "canonical-after"])
@pytest.mark.parametrize("prior", [False, True])
async def test_refused_settlement_retires_only_its_original_registration(
    tmp_path, monkeypatch, mode, prior,
):
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    assert not agent.llm_service.providers
    dispatcher = SignalDispatcher(agent=agent, registry=SourceRegistry(),
                                  lock_manager=OrderedLockManager(), store=SignalLogStore(db.backend))
    await db.execute("CREATE TABLE settlement_effects (id INTEGER)")
    request_id = "original-unstarted-settlement"
    initial = agent.register_active_request(request_id, nested=False) if prior else None
    refusal = RuntimeError("original Stop settlement " + mode + " refused")
    refused, accepted, callbacks, generations = [], [], [], []
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    owner = None
    actual_binding = type(agent).bind_request_operation

    def bind(self, rid, operation):
        if self is agent and rid == request_id and mode == "binding":
            raise refusal
        return actual_binding(self, rid, operation)

    class RefusedCompletionPublication(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if mode == "callback" and getattr(callback, "__name__", "") == "release_settlement_owner":
                callbacks.append(callback)
                super().add_done_callback(callback, context=context)
                raise refusal
            return super().add_done_callback(callback, context=context)

    class RefusedCanonicalPublication(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "close_unstarted":
                if mode == "canonical-after":
                    super().add_done_callback(callback, context=context)
                raise refusal
            return super().add_done_callback(callback, context=context)

    async def run_attempts():
        for _ in range(3):
            with pytest.raises(BaseException) as caught:
                guard = dispatcher._hold_durable_cognition_stop_completion(request_id)
                if inspect.isawaitable(guard):
                    guard = await guard
                pytest.fail("refused settlement published an owner")
            if mode == "allocation":
                from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
                assert refusal in _explicit_control_chain(caught.value)
                assert execution_cleanup_debts(caught.value), "opaque settlement admission was falsely retired"
            else:
                assert caught.value is refusal
            generations.append(agent._next_request_generation)
            expected = {(request_id, initial): 1} if prior else {}
            assert agent._active_request_generation_counts == expected, "refused settlement leaked its original active generation"
            assert agent._active_request_counts == ({request_id: 1} if prior else {})
            assert agent._active_request_ids == ({request_id} if prior else set())
            assert all(task.done() for task in accepted), "refused settlement returned before accepted barrier physically retired"
            assert not any(entry[0] is callback for entry in (asyncio.current_task()._callbacks or ()) for callback in callbacks), "refused settlement left its partially published completion callback attached"
            assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair), "refused settlement admission leaked unstarted operation"
            assert await agent.wait_for_request_completion(request_id, generation=generations[-1]) is RequestCompletionDisposition.ABANDONED
        assert len(callbacks if mode == "callback" else refused if mode == "allocation" else accepted) == 3
        assert await db.fetchval("SELECT count(*) FROM settlement_effects") == 0

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is run_attempts.__code__:
            return RefusedCompletionPublication(coroutine, loop=loop, **kwargs)
        selected = code is not None and code.co_qualname.endswith("._hold_durable_cognition_stop_completion.<locals>.await_settlement_release")
        if selected and mode == "allocation":
            refused.append((coroutine, operation))
            raise refusal
        if selected and mode.startswith("canonical"):
            task = RefusedCanonicalPublication(coroutine, loop=loop, **kwargs)
        else:
            task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if selected:
            accepted.append(task)
        return task

    try:
        monkeypatch.setattr(type(agent), "bind_request_operation", bind)
        loop.set_task_factory(factory)
        owner = create_owned_task(run_attempts(), retain_originals=True)
        result = await await_owned_task(owner)
        if result.error is not None:
            raise result.error
    finally:
        loop.set_task_factory(previous)
        if owner is not None:
            await await_owned_task(owner)
        for task in accepted:
            if not task.done():
                task.cancel()
            await await_owned_task(task)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        # Retire only this fixture's exact still-active registrations after
        # a failed BEFORE assertion; no replacement Stop completion is claimed.
        for (rid, generation), count in tuple(agent._active_request_generation_counts.items()):
            for _ in range(count):
                agent._cleanup_cancelled_request(rid, disposition=RequestCompletionDisposition.ABANDONED, generation=generation)
        await agent.llm_service.close()
        await db.close()


@pytest.mark.asyncio
async def test_failed_original_settlement_retirement_keeps_exact_registration_debt(tmp_path, monkeypatch):
    from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts, is_execution_control_error
    monkeypatch.setattr("kestrel_sovereign.entrypoints.discover_entry_point_classes", lambda *_args, **_kwargs: {})
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    dispatcher = SignalDispatcher(agent=agent, registry=SourceRegistry(),
                                  lock_manager=OrderedLockManager(), store=SignalLogStore(db.backend))
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refusal = RuntimeError("original settlement allocation refused before cleanup refusal")
    retirement = RuntimeError("actual original settlement registrar refused retirement")
    actual_cleanup = type(agent)._cleanup_cancelled_request
    refused = []
    request_id = "original-unretired-settlement"
    def refuse(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname.endswith("._hold_durable_cognition_stop_completion.<locals>.await_settlement_release"):
            refused.append((coroutine, operation))
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    def fail_retirement(self, rid, **kwargs):
        if self is agent and rid == request_id:
            raise retirement
        return actual_cleanup(self, rid, **kwargs)
    try:
        monkeypatch.setattr(type(agent), "_cleanup_cancelled_request", fail_retirement)
        loop.set_task_factory(refuse)
        with pytest.raises(BaseException) as caught:
            await dispatcher._hold_durable_cognition_stop_completion(request_id)
        loop.set_task_factory(previous)
        chain = tuple(_explicit_control_chain(caught.value))
        assert refusal in chain and retirement in chain and is_execution_control_error(caught.value)
        generation = agent._active_request_generations[request_id]
        assert agent._active_request_generation_counts == {(request_id, generation): 1}
        assert any(isinstance(resource, tuple) and resource == (agent, request_id, generation) and task is None
                   for resource, task in execution_cleanup_debts(caught.value))
        assert not execution_cleanup_debts(refusal) and not execution_cleanup_debts(retirement)
        assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
    finally:
        loop.set_task_factory(previous)
        monkeypatch.setattr(type(agent), "_cleanup_cancelled_request", actual_cleanup)
        for (rid, generation), count in tuple(agent._active_request_generation_counts.items()):
            for _ in range(count):
                actual_cleanup(agent, rid, disposition=RequestCompletionDisposition.ABANDONED, generation=generation)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await agent.llm_service.close()
        await db.close()
