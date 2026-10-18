"""Enclosing shutdown consumers retain actual withdrawal and native context custody.

Uses native PG sources and real SQLite effects/Hold context workers. Patched
component callbacks and standalone protocol agents are not full-Agent acceptance.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign import _async_ownership as ownership, server, cli, main, hold
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.execution_custody import execution_cleanup_debts
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.multi_agent.config import LocalAgentConfig
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg
from tests.integration.test_host_onboarding_original_admission import onboarding_owner as _onboarding_owner
from tests.integration.test_maintenance_original_retirement import _original_control
from tests.utils.original_failure import OriginalFailure

native_pg = _native_pg
onboarding_owner = _onboarding_owner


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["reaper", "unregistered", "registration", "server-agents", "phoenix"])
@pytest.mark.parametrize("source_cancel", [False, True])
async def test_enclosing_shutdown_keeps_actual_join_withdrawal_before_native_failure(
    native_pg, onboarding_owner, monkeypatch, surface, source_cancel,
):
    manager, agent, db = onboarding_owner
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    ordinary = OriginalFailure.capture(OSError("ordinary original shutdown failed"))
    entered, release, withdrawals = asyncio.Event(), asyncio.Event(), []
    actual_join, owner, child = ownership.await_owned_task, None, None
    async def observe(task, *args, **kwargs):
        result = await actual_join(task, *args, **kwargs)
        if asyncio.current_task() is owner and result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result
    monkeypatch.setattr(ownership, "await_owned_task", observe)
    async def first():
        await db.execute("INSERT INTO host_rollback_effects VALUES (130)")
        entered.set()
        await release.wait()
        if surface in ("reaper", "server-agents"):
            raise ordinary.error
    async def later(*args):
        await db.execute("INSERT INTO host_rollback_effects VALUES (131)")
        raise source.error
    async def nothing(*args):
        pass
    if surface == "reaper":
        child = create_owned_task(first(), retain_originals=True)
        monkeypatch.setattr(agent, "wait_for_shutdown_completion", later)
        owner = agent.handoff_shutdown_to_reaper(child)
    elif surface == "unregistered":
        monkeypatch.setattr(agent, "shutdown", nothing)
        monkeypatch.setattr(agent, "wait_for_shutdown_completion", first)
        monkeypatch.setattr(manager.__class__, "_rollback_host_onboarding", staticmethod(AsyncMock(return_value=(False, source.error))))
        owner = create_owned_task(manager._shutdown_unregistered_agent("component", agent), retain_originals=True)
    elif surface == "registration":
        async def registration(*args):
            await first()
            return later
        manager._scheduler_polling_managed_by_host = True
        manager._scheduler_tenant_registration_hook = registration
        owner = create_owned_task(manager._begin_dynamic_scheduler_tenant_registration(
            "Component", agent.did, LocalAgentConfig(data_dir="component", port=8801)),
            retain_originals=True)
    elif surface == "server-agents":
        app = FastAPI()
        app.state.agent = SimpleNamespace(shutdown=nothing, wait_for_shutdown_completion=first)
        app.state.startup_cleanup_agent = SimpleNamespace(shutdown=later)
        owner = create_owned_task(server._shutdown_server_agents(app), retain_originals=True)
    else:
        app = FastAPI()
        child = create_owned_task(first(), retain_originals=True)
        app.state.phoenix_start_task = child
        app.state.phoenix = SimpleNamespace(aclose=later, stop=lambda: None)
        owner = create_owned_task(server._shutdown_phoenix(app), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel("actual enclosing lifecycle join withdrew")
        await asyncio.sleep(0)
        release.set()
        result = await await_owned_task(owner)
        source.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        if surface in ("reaper", "server-agents"):
            ordinary.assert_retained(result.error)
        assert await db.fetchval("SELECT count(*) FROM host_rollback_effects") == (1 if surface == "unregistered" else 2)
        if child is not None:
            assert child.done()
    finally:
        release.set()
        await await_owned_task(owner)
        if child is not None:
            await await_owned_task(child)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["cli", "main"])
@pytest.mark.parametrize("source_cancel", [False, True])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("failure_phase", [None, "setup", "report"])
async def test_standalone_entrypoint_closes_real_caller_owned_hold_after_settled_native_agent_failure(
    native_pg, monkeypatch, tmp_path, entrypoint, source_cancel, pending, failure_phase,
):
    source = OriginalFailure.capture(await _original_control(native_pg[0], monkeypatch, source_cancel))
    contexts = []
    original_initialize = hold.initialize_with_bound_hold_context
    async def initialize(*args, **kwargs):
        context = await original_initialize(*args, **kwargs)
        contexts.append(context)
        return context
    async def nothing():
        pass
    release, entered = asyncio.Event(), asyncio.Event()
    native_db = await AsyncDatabase.sqlite(str(tmp_path / "original-agent-tail.db"))
    await native_db.execute("CREATE TABLE effects (id INTEGER)")
    async def tail():
        await native_db.execute("INSERT INTO effects VALUES (1)")
        entered.set()
        await release.wait()
        await native_db.execute("INSERT INTO effects VALUES (2)")
    original = create_owned_task(tail(), retain_originals=True) if pending else None
    async def completion():
        if pending:
            debt = RuntimeError("original native Agent tail still pending")
            debt.__cause__ = source.error
            debt._kestrel_native_retirement_debts = ((agent, original),)
            raise debt
        raise source.error
    agent = SimpleNamespace(agent_id="did:test:standalone-component", initialize=nothing,
                            shutdown=nothing, wait_for_shutdown_completion=completion)
    if pending:
        agent._durable_shutdown_continuation = original
        await asyncio.wait_for(entered.wait(), 5)
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path / "host-home"))
    monkeypatch.setenv("KESTREL_HOLD_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setattr(hold, "initialize_with_bound_hold_context", initialize)
    monkeypatch.setattr("builtins.input", lambda prompt: "!quit")
    reporting = OriginalFailure.capture(BrokenPipeError("original standalone output owner failed"))
    actual_print = print
    def output(*args, **kwargs):
        value = " ".join(str(arg) for arg in args)
        if (failure_phase == "setup" and "Kestrel Agent Initialized" in value
                or failure_phase == "report" and "Agent deactivated" in value):
            raise reporting.error
        return actual_print(*args, **kwargs)
    monkeypatch.setattr("builtins.print", output)
    if entrypoint == "main":
        monkeypatch.setattr(main, "KestrelAgent", lambda **kwargs: agent)
        monkeypatch.setattr(main, "LLMService", MagicMock())
        monkeypatch.setattr(main, "get_agent_did_async", AsyncMock(return_value=agent.agent_id))
        monkeypatch.setattr(sys, "argv", ["kestrel-main", str(tmp_path)])
        op = main.main()
    else:
        storage = SimpleNamespace(initialize=AsyncMock(), close=AsyncMock(),
                                  get_nodes_by_type=AsyncMock(return_value=[SimpleNamespace(node_id=agent.agent_id)]))
        monkeypatch.setattr("kestrel_sovereign.storage.AsyncStorage", lambda path: storage)
        monkeypatch.setattr("kestrel_sovereign.kestrel_agent.KestrelAgent", lambda **kwargs: agent)
        monkeypatch.setattr("kestrel_sovereign.llm.service.LLMService", MagicMock())
        op = cli._run_shell(Path(tmp_path), SimpleNamespace(app=None))
    try:
        result = await await_owned_task(create_owned_task(op, retain_originals=True))
        source.assert_retained(result.error)
        if failure_phase is not None:
            reporting.assert_retained(result.error)
        assert len(contexts) == 1
        context = contexts[0]
        if pending:
            assert context.hold_store._db.backend._connection is not None and not original.done()
            assert any(resource is context and task is original for resource, task in execution_cleanup_debts(result.error)), "pending Agent owner lost exact caller-owned context dependency"
        else:
            assert context.hold_store._db.backend._connection is None, "settled Agent failure skipped separately owned Hold context close"
    finally:
        release.set()
        if original is not None:
            await await_owned_task(original)
            assert await native_db.fetchval("SELECT count(*) FROM effects") == 2
        for context in contexts:
            await hold.close_bound_host_context(context)
        await native_db.close()
