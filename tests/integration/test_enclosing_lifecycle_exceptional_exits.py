"""Exceptional lifecycle exits preserve original native owners and finite graphs.

Real PostgreSQL source receipts, SQLite worker retirement and actual Task
publication faults. Component protocols are not full Agent/provider acceptance.
"""
import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.hold import enforcement
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


class UnexpectedControl(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_phoenix_unexpected_source_group_still_retires_independent_proxy(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    from fastapi import FastAPI
    from kestrel_sovereign import server
    _, _, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    group = OriginalFailure.capture(BaseExceptionGroup("original Phoenix startup control", [source.error, UnexpectedControl("original observer sibling")]))
    calls = []
    async def start():
        await db.execute("INSERT INTO host_rollback_effects VALUES (230)")
        raise group.error
    async def close():
        calls.append("close")
        await db.close()
    app = FastAPI()
    original = ownership.create_owned_task(start(), retain_originals=True)
    app.state.phoenix_start_task = original
    app.state.phoenix = SimpleNamespace(aclose=close, stop=lambda: calls.append("stop"))
    owner = ownership.create_owned_task(server._shutdown_phoenix(app), retain_originals=True)
    try:
        result = await ownership.await_owned_task(owner)
        source.assert_retained(result.error)
        group.assert_retained(result.error)
        assert original.done() and calls == ["close", "stop"]
        assert db.backend._connection is None
    finally:
        await ownership.await_owned_task(owner)
        await ownership.await_owned_task(original)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_fallback_join_failure_cannot_skip_independent_standalone_context(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    _, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    opening = OriginalFailure.capture(OSError("original completion waiter publication refused"))
    group = BaseExceptionGroup("original continuation terminal", [source.error, UnexpectedControl("original unexpected control")])
    calls = []
    async def original_tail():
        await db.execute("INSERT INTO host_rollback_effects VALUES (218)")
        raise group
    async def waiter():
        await asyncio.sleep(0)
    async def close(context):
        calls.append(context)
        await db.close()
    continuation = ownership.create_owned_task(original_tail(), retain_originals=True)
    agent._durable_shutdown_continuation = continuation
    monkeypatch.setattr(agent, "wait_for_shutdown_completion", waiter)
    monkeypatch.setattr(enforcement, "close_bound_host_context", close)
    context = SimpleNamespace(db=db)
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        if getattr(operation, "cr_code", None) is waiter.__code__:
            raise opening.error
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
    owner = ownership.create_owned_task(enforcement.complete_bound_agent_shutdown(agent, context), retain_originals=True)
    try:
        loop.set_task_factory(factory)
        result = await ownership.await_owned_task(owner)
        loop.set_task_factory(previous)
        source.assert_retained(result.error)
        opening.assert_retained(result.error)
        assert continuation.done() and calls == [context]
        assert db.backend._connection is None
    finally:
        loop.set_task_factory(previous)
        await ownership.await_owned_task(owner)
        await ownership.await_owned_task(continuation)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_hold_startup_cleanup_does_not_mutate_original_native_source(
    native_pg, onboarding_owner, monkeypatch, source_cancel,
):
    _, _, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    closing = OriginalFailure.capture(OSError("original context close observer failed"))
    async def close(context):
        await db.close()
        raise closing.error
    monkeypatch.setattr(enforcement, "close_bound_host_context", close)
    owner = ownership.create_owned_task(enforcement._close_context_after_startup_failure(
        SimpleNamespace(db=db), source.error, phase="original binding"), retain_originals=True)
    result = await ownership.await_owned_task(owner)
    source.assert_retained(result.error)
    closing.assert_retained(result.error)
    assert db.backend._connection is None


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["features", "scheduler", "agents", "phoenix"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_server_phases_keep_raw_prefix_and_actual_withdrawal_for_native_body(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    from fastapi import FastAPI
    from kestrel_sovereign import server, host_features
    from kestrel_sovereign.execution_custody import _explicit_control_chain
    _, _, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = [OriginalFailure.capture(OSError("original first phase failure")),
                OriginalFailure.capture(ValueError("original second phase failure"))]
    app, calls, observed = FastAPI(), [], []
    entered, release, phase_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    phase_owners = []
    original = None
    actual_join = ownership.await_owned_task
    async def observe(task, *args, **kwargs):
        if task is original and asyncio.current_task().get_name() == "server_shutdown:phoenix":
            phase_entered.set()
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() in phase_owners and result.cancellation is not None:
            observed.append(result.cancellation)
        return result
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    async def nothing(*args, **kwargs):
        pass
    async def first():
        phase_owners.append(asyncio.current_task())
        await db.execute("INSERT INTO host_rollback_effects VALUES (219)")
        entered.set()
        await release.wait()
        if surface == "agents":
            raise ordinary[0].error
    async def effect():
        await db.execute("INSERT INTO host_rollback_effects VALUES (220)")
        calls.append("closed")
    for name in ("_shutdown_host_scheduler", "_shutdown_stop_cleanup", "_shutdown_server_agents",
                 "_shutdown_host_features", "_shutdown_distributed_invocations",
                 "_shutdown_stop_receipts", "_shutdown_shared_agent_postgres_backend", "_shutdown_phoenix"):
        if name != {"features": "_shutdown_host_features", "scheduler": "_shutdown_host_scheduler",
                    "agents": "_shutdown_server_agents", "phoenix": "_shutdown_phoenix"}[surface]:
            monkeypatch.setattr(server, name, nothing)
    originals = []
    if surface == "features":
        app.state.host_context = SimpleNamespace()
        def fail(index):
            def operation(app):
                raise ordinary[index].error
            return operation
        monkeypatch.setattr(host_features, "unmount_host_features", fail(0))
        monkeypatch.setattr(server, "_unmount_feature_routers", fail(1))
        monkeypatch.setattr(server, "_unmount_feature_ui_assets", lambda app: None)
        monkeypatch.setattr(host_features, "close_host_context", nothing)
        originals = ordinary
    elif surface == "scheduler":
        async def stop():
            raise ordinary[0].error
        async def close():
            raise ordinary[1].error
        app.state.host_scheduler_runner = SimpleNamespace(stop=stop)
        app.state.host_scheduler_storage = SimpleNamespace(close=close)
        originals = ordinary
    elif surface == "agents":
        app.state.agent = SimpleNamespace(shutdown=nothing, wait_for_shutdown_completion=first)
        originals = ordinary[:1]
    else:
        original = ownership.create_owned_task(first(), retain_originals=True)
        app.state.phoenix_start_task = original
        app.state.phoenix = SimpleNamespace(aclose=effect, stop=lambda: None)
    owner = ownership.create_owned_task(server._shutdown_server_resources(app, original_failure=source.error), retain_originals=True)
    try:
        if surface in ("agents", "phoenix"):
            await asyncio.wait_for(entered.wait(), 5)
            if surface == "phoenix":
                await asyncio.wait_for(phase_entered.wait(), 5)
            # The completion waiter is another owned Task; cancel the actual
            # enclosing phase, whose shield must still join that same writer.
            if surface == "agents":
                phase = next(task for task in asyncio.all_tasks() if task.get_name() == "server_shutdown:agents")
            else:
                phase = next(task for task in asyncio.all_tasks() if task.get_name() == "server_shutdown:phoenix")
            phase_owners.append(phase)
            phase.cancel("actual original phase observer withdrew")
            await asyncio.sleep(0)
            release.set()
        result = await ownership.await_owned_task(owner)
        assert result.error is None
        terminal = result.result[1]
        source.assert_retained(terminal)
        for original_failure in originals:
            original_failure.assert_retained(terminal)
        if surface in ("agents", "phoenix"):
            assert observed and all(error in tuple(_explicit_control_chain(terminal)) for error in observed)
        if surface == "phoenix":
            assert original.done() and calls == ["closed"]
    finally:
        release.set()
        await ownership.await_owned_task(owner)
        if surface == "phoenix":
            await ownership.await_owned_task(original)
