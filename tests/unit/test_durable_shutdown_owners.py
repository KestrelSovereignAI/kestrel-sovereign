"""Only the dispatcher's actual close owners survive the general task sweep."""

from __future__ import annotations

import asyncio

from tests.unit.test_durable_callback_runtime_context import resident_dispatcher, retire


async def test_shutdown_protects_exact_dispatcher_owners_but_stops_other_runtime_services():
    agent, dispatcher = resident_dispatcher()
    agent.dispatcher = dispatcher
    released, entered = asyncio.Event(), asyncio.Event()

    async def metadata_owner():
        entered.set()
        await released.wait()

    owned = agent._track_runtime_task(metadata_owner(), name="opaque-original-metadata")
    unrelated = agent._track_runtime_task(asyncio.Event().wait(), name="another-runtime-service")
    dispatcher._own_durable_task(owned)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert dispatcher.durable_owner_tasks() == frozenset()
        dispatcher._durable_shutdown = True
        assert dispatcher.durable_owner_tasks() == frozenset({owned})
        await agent._shutdown_background_tasks()
        assert unrelated.cancelled()
        assert owned.cancelling() == 0 and not owned.done()
        assert owned in agent._background_tasks
        released.set()
        await owned
        await asyncio.sleep(0)
        assert not agent._background_tasks
        assert not dispatcher.durable_owner_tasks()
    finally:
        released.set()
        await retire(agent, dispatcher)
