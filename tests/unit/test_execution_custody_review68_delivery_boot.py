"""Delivery rollback must retire the real pre-READY resident barrier."""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.agent.boot import BootPhase, BootPhaseState
from kestrel_sovereign.features.delivery.feature import DeliveryFeature
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_agent_boot_phases import _make_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_phase", [False, True])
async def test_failed_boot_joins_delivery_waiting_at_original_ready_barrier(tmp_path, cancel_phase):
    agent = _make_agent(tmp_path)
    db = await AsyncDatabase.sqlite(str(tmp_path / "native-delivery-boot.db"))
    agent._raw_storage = agent.storage = SimpleNamespace(db=db)
    feature = DeliveryFeature(agent)
    failed = asyncio.Event()
    failure = asyncio.CancelledError("later native boot phase cancelled") if cancel_phase else ValueError("later native boot phase failed")

    async def acquire_delivery(ctx):
        # Preserve production LIFO ordering: the background inventory was
        # registered earlier than feature teardown and cannot rescue its join.
        ctx.on_rollback("background_tasks", agent._shutdown_background_tasks)
        await feature.initialize()
        agent.features[feature.name] = feature
        ctx.on_rollback("features", agent._boot_teardown_features)
        await asyncio.sleep(0)
        assert feature._queue._runtime_task_factory.__self__ is agent
        assert not agent._runtime_publication_ready.is_set()
        assert not feature._queue._task.done()

    async def fail_after_delivery(_ctx):
        failed.set()
        raise failure

    agent._boot_phases = lambda: [BootPhase("native_delivery", acquire_delivery), BootPhase("later_failure", fail_after_delivery)]
    boot = create_owned_task(agent.initialize())
    try:
        await asyncio.wait_for(failed.wait(), 5)
        done, _ = await asyncio.wait({boot}, timeout=2)
        assert done, "boot rollback joined a resident still blocked behind READY"
        outcome = await await_owned_task(boot)
        assert outcome.error is failure
        assert agent._boot_state is BootPhaseState.FAILED
        assert not agent._runtime_publication_ready.is_set()
        assert feature._queue._task.done() and feature._queue._task.cancelled()
        assert agent._background_tasks == set()
        assert agent._boot_context.rollback_errors == []
    finally:
        # Only this never-admitted resident is cancelled to retire a failing
        # before-fix reproduction. It has not entered the delivery loop/SQL.
        if feature._queue is not None and not feature._queue._task.done():
            feature._queue._task.cancel()
            await await_owned_task(feature._queue._task)
        await await_owned_task(boot)
        await agent._shutdown_background_tasks()
        await agent.llm_service.close()
        await db.close()
