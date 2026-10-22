"""Native resident re-arm callbacks must not retain a retired HTTP admission.

These finite context tests execute the real dispatcher publishers, task owner,
custody guards and event loop. They do not claim PostgreSQL/product acceptance.
"""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    bind_execution_custody,
    current_execution_custody,
    require_execution_work,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.signals.dispatcher import SignalDispatcher
from tests.unit.test_execution_custody import Authority


def resident_dispatcher():
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.did = "did:test:callback-runtime"
    agent._execution_custody = ExecutionCustody(Authority())
    agent._runtime_publication_ready = asyncio.Event()
    agent._runtime_publication_ready.set()
    agent._background_tasks = set()
    dispatcher = SignalDispatcher.__new__(SignalDispatcher)
    dispatcher._agent = agent
    # This fixture exercises publisher callbacks, not native row ownership;
    # dispatch_signal below is an explicit finite callback seam.
    dispatcher._store = SimpleNamespace(_backend=None)
    dispatcher._durable_shutdown = False
    dispatcher._live_durable_cognition_event_ids = set()
    dispatcher._started_durable_cognition_consumers = {"channel"}
    dispatcher._durable_cognition_drainers = {}
    dispatcher._durable_cognition_drain_rerun = set()
    dispatcher._durable_cognition_drain_timers = {}
    dispatcher._retained_cognition_control_debt = {}
    dispatcher._durable_shutdown_owner_fenced = False
    dispatcher._retained_durable_cognition_tasks = set()
    return agent, dispatcher


async def retire(agent, dispatcher):
    dispatcher._durable_shutdown = True
    for timer in dispatcher._durable_cognition_drain_timers.values():
        timer.cancel()
    tasks = tuple(agent._background_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke_runtime", [False, True])
async def test_ingress_completion_rearms_only_the_original_live_runtime(revoke_runtime):
    agent, dispatcher = resident_dispatcher()
    release, drained = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    failures = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: failures.append(context))

    async def dispatch(*args, **kwargs):
        await release.wait()

    async def drain(consumer_id):
        assert consumer_id == "channel"
        require_execution_work(agent)
        assert current_execution_custody() == (agent._execution_custody,)
        drained.set()

    dispatcher.dispatch_signal = dispatch
    dispatcher._drain_durable_cognition_consumer = drain
    try:
        with bind_execution_custody(Authority()) as ingress:
            handle = await dispatcher.enqueue_durable_cognition(
                SimpleNamespace(id="original-event"),
                source_event_id="telegram:update:1", consumer_id="channel",
            )
        with pytest.raises(ExecutionAuthorityError, match="admission retired"):
            ingress.require_work()
        if revoke_runtime:
            agent._execution_custody.revoke("original runtime lost")
        release.set()
        await handle.task
        if revoke_runtime:
            # Let the finite completion and zero-delay wake become observable.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not drained.is_set()
            with pytest.raises(ExecutionAuthorityError, match="original runtime lost"):
                require_execution_work(agent)
        else:
            try:
                await asyncio.wait_for(drained.wait(), 0.5)
            except TimeoutError:
                pytest.fail(f"resident completion did not re-arm: {failures!r}")
            assert not failures, failures
        assert not dispatcher._live_durable_cognition_event_ids
    finally:
        release.set()
        await retire(agent, dispatcher)
        loop.set_exception_handler(previous)


@pytest.mark.asyncio
async def test_drainer_completion_rerun_uses_original_runtime_not_publisher_scope():
    agent, dispatcher = resident_dispatcher()
    release, drained_twice = asyncio.Event(), asyncio.Event()
    calls = []
    failures = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: failures.append(context))

    async def drain(consumer_id):
        require_execution_work(agent)
        assert current_execution_custody() == (agent._execution_custody,)
        calls.append(consumer_id)
        if len(calls) == 1:
            await release.wait()
        else:
            drained_twice.set()

    dispatcher._drain_durable_cognition_consumer = drain
    try:
        with bind_execution_custody(Authority()):
            dispatcher._start_durable_cognition_drain("channel")
            dispatcher._start_durable_cognition_drain("channel")
        release.set()
        await asyncio.wait_for(drained_twice.wait(), 0.5)
        assert calls == ["channel", "channel"]
        assert not failures, failures
    finally:
        release.set()
        await retire(agent, dispatcher)
        loop.set_exception_handler(previous)
