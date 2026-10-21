"""Monitored cognition retains every source and the actual finite withdrawal.

The protocol agent is a declared test stand-in; SQLite dispatch and PG source
effects are real. This is not whole-Agent/provider/genesis acceptance.
"""
import asyncio

import pytest
from kestrel_sdk.signals import SignalMode

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_finite_error
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.integration.test_native_finite_outer_consumers import unstarted_native_close
from tests.unit.test_signals_dispatcher import db_path as db_path, dispatcher_components as dispatcher_components
from tests.unit.test_signals_dispatcher import _cognition_reg, _signal
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["worker", "finish"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("ordinary_first", [False, True])
async def test_monitored_cognition_keeps_native_siblings_and_actual_current_c(
    native_pg, dispatcher_components, tmp_path, monkeypatch, phase, source_cancel, ordinary_first,
):
    c = dispatcher_components
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = OriginalFailure.capture(OSError("original ordinary sibling failed before later native finish"))
    resource, independent = await unstarted_native_close()
    template = tmp_path / "finite-monitor.md"
    template.write_text("payload: {payload}")
    c.registry.register(_cognition_reg(template, name="finite_native_monitor"))
    await c.backend.execute("CREATE TABLE effects (id INTEGER PRIMARY KEY)")
    entered, cancelled, release, finishing, finish_release = (asyncio.Event() for _ in range(5))
    withdrawals = []
    owner = None
    actual_join = ownership.await_owned_task
    async def observe(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    async def execute(_prompt):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        await c.backend.execute("INSERT INTO effects VALUES (1)")
        raise ordinary.error if ordinary_first else source.error
    async def monitor(_signal):
        await entered.wait()
        return "original durable source withdrew"
    async def finish(_signal):
        await c.backend.execute("INSERT INTO effects VALUES (2)")
        finishing.set()
        if phase == "finish":
            await finish_release.wait()
        raise execution_finite_error(source.error, independent.error) if ordinary_first else independent.error
    c.agent.process_input = execute
    c.agent.monitor_cognition_signal_execution = monitor
    c.agent.finish_cognition_signal_execution = finish
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    owner = create_owned_task(c.dispatcher.dispatch_signal(_signal("finite_native_monitor", mode=SignalMode.COGNITION)), retain_originals=True)
    try:
        await asyncio.wait_for(cancelled.wait(), 5)
        if phase == "finish":
            release.set()
            await asyncio.wait_for(finishing.wait(), 5)
        owner.cancel("actual monitored " + phase + " observer withdrew")
        await asyncio.sleep(0)
        assert not owner.done()
        release.set()
        finish_release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        if ordinary_first:
            ordinary.assert_retained(result.error)
        independent.assert_retained(result.error)
        assert withdrawals and all(value in tuple(_explicit_control_chain(result.error)) for value in withdrawals)
        assert (await c.backend.fetch_all("SELECT count(*) FROM effects"))[0][0] == 2
        assert all(task.done() for task in c.agent.background_tasks)
    finally:
        release.set()
        finish_release.set()
        await await_owned_task(owner)
        await c.dispatcher.shutdown_durable_delivery()
        await resource.close()
