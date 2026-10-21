"""Unaccepted original gate/persistence coroutines must be closed, not owned."""

import asyncio
import inspect
import sqlite3
from contextlib import closing

import pytest

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.feature_registry import InstalledFeatureRuntime
from kestrel_sovereign.features.health.feature import HealthFeature
from kestrel_sovereign.features.isolated_runtime import ProxyFeature
from tests.unit.test_cleanup_task_admission import (
    _operation_name,
    _partial_native_agent,
)


def original_gate_proxy(agent):
    # Constructor and gate contract only: no child is launched, no runnable
    # metadata/SDK handoff, handshake, genesis or READY is claimed.
    return ProxyFeature(
        agent,
        InstalledFeatureRuntime(
            class_name="NativeGateAdmissionFeature",
            entry_point="native_gate:Feature",
            distribution="native-gate-admission",
            runtime="isolated-venv",
            service="native-gate-service",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["health", "gate"])
async def test_refused_original_gate_or_persist_coroutine_is_closed(
    tmp_path, monkeypatch, surface
):
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    feature = (
        HealthFeature(agent) if surface == "health" else original_gate_proxy(agent)
    )
    if surface == "health":
        await feature.initialize()
    else:
        await db.execute("CREATE TABLE original_gate_admission (id INTEGER)")
        async with feature._traffic_gate.admit():
            await db.execute("INSERT INTO original_gate_admission VALUES (1)")
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refused, original = (
        [],
        RuntimeError("original native lifecycle coroutine admission refused"),
    )

    def factory(loop, coroutine, **kwargs):
        target = (
            "HealthFeature._run_health.<locals>.persist"
            if surface == "health"
            else "_TrafficGate.close_and_drain"
        )
        if _operation_name(coroutine) == target:
            frame = getattr(coroutine, "cr_frame", None)
            operation = frame.f_locals.get("operation") if frame is not None else None
            refused.append(
                (coroutine, operation if inspect.iscoroutine(operation) else coroutine)
            )
            raise original
        return (
            previous(loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=loop, **kwargs)
        )

    try:
        loop.set_task_factory(factory)
        with pytest.raises(RuntimeError) as caught:
            if surface == "health":
                await feature.run_once()
            else:
                await feature._close_traffic_gate()
        loop.set_task_factory(previous)
        assert caught.value is original and refused
        assert all(
            inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
            for pair in refused
            for coro in pair
        ), "fresh unaccepted lifecycle coroutine remained open"
        with closing(sqlite3.connect(db.backend.db_path)) as independent, independent:
            table = "health_log" if surface == "health" else "original_gate_admission"
            assert independent.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[
                0
            ] == (0 if surface == "health" else 1)
        if surface == "health":
            assert not feature._health_persist_tasks
        else:
            assert not feature._traffic_gate.closed and feature._client is None
        assert not agent._runtime_publication_ready.is_set()
    finally:
        loop.set_task_factory(previous)
        for pair in refused:
            for coroutine in pair:
                coroutine.close()
        await feature.shutdown()
        await agent.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_gate_transition_does_not_adopt_actual_foreign_native_task(
    tmp_path, monkeypatch
):
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    feature = original_gate_proxy(agent)
    foreign = create_owned_task(
        db.execute("CREATE TABLE foreign_native_gate_task (id INTEGER)")
    )
    try:
        with pytest.raises(TypeError):
            await feature._run_traffic_gate_operation(foreign, name="foreign")
        assert not foreign.cancelled()
        assert (await await_owned_task(foreign)).error is None
        assert await db.table_exists("foreign_native_gate_task")
        assert not agent._runtime_publication_ready.is_set()
    finally:
        await await_owned_task(foreign)
        await feature.shutdown()
        await agent.shutdown()
        await db.close()


@pytest.mark.asyncio
async def test_canonical_admission_does_not_move_original_work_to_another_loop(
    tmp_path, monkeypatch
):
    agent, db = await _partial_native_agent(tmp_path, monkeypatch)
    other_loop = asyncio.new_event_loop()
    original = db.execute("CREATE TABLE foreign_loop_must_not_enter (id INTEGER)")
    try:
        with pytest.raises(RuntimeError, match="original running loop"):
            create_owned_task(original, loop=other_loop)
        assert inspect.getcoroutinestate(original) == inspect.CORO_CLOSED
        assert not await db.table_exists("foreign_loop_must_not_enter")
        assert not asyncio.all_tasks(other_loop)
    finally:
        original.close()
        other_loop.close()
        await agent.shutdown()
        await db.close()
