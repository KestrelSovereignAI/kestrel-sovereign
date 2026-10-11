"""Original sleep/bootstrap controls and failed-startup recovery, offline."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.agent.sleep import SleepMixin, SleepReport
from kestrel_sovereign.bootstrap.service import BootstrapState
from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.features.bootstrap.feature import BootstrapFeature
from kestrel_sovereign.features.memory.reflection_hook import (
    ReflectionSleepHook,
    RetrievedMemoryCandidate,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
from tests.unit.test_execution_custody_review32 import receipt
from tests.unit.test_execution_custody_review43_native import carries
from tests.unit.test_execution_custody_review45_native import native_process
from tests.unit.test_execution_custody_review46_native import service_for


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["clean", "unknown", "cancel_unknown", "explicit"])
async def test_cancelled_original_startup_is_replaced_only_after_clean_native_retirement(monkeypatch, terminal):
    monkeypatch.setattr("kestrel_sovereign.llm.codex_app_server.resolve_codex_binary", lambda: "offline-no-codex")
    client = CodexAppServerClient(binary="offline-no-codex")
    adapter = CodexAdapter.__new__(CodexAdapter)
    adapter._client = client
    adapter._agent_for_audit = None
    proc = await native_process()
    entered = asyncio.Event()
    original_close = client._close_connection
    failure = receipt(terminal) if terminal in {"unknown", "cancel_unknown"} else None

    async def spawn():
        client._proc = proc

    async def handshake(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    async def close_then_receipt():
        await original_close()
        if failure is not None:
            raise failure

    monkeypatch.setattr(client, "_spawn", spawn)
    monkeypatch.setattr(client, "_handshake", handshake)
    monkeypatch.setattr(client, "_close_connection", close_then_receipt)
    startup = asyncio.create_task(client.ensure_started())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        startup.cancel()
        outcome = await await_owned_task(startup)
        assert proc.returncode is not None and client._proc is None
        if failure is not None:
            assert carries(outcome.error, failure)
        if terminal == "explicit":
            await client.aclose()
        replacement = adapter._app_server()
        if terminal == "clean":
            assert replacement is not client, "cached cancelled startup permanently fenced valid turns"
            async def fresh_spawn():
                pass
            async def fresh_handshake(**kwargs):
                pass
            monkeypatch.setattr(replacement, "_spawn", fresh_spawn)
            monkeypatch.setattr(replacement, "_handshake", fresh_handshake)
            await replacement.ensure_started()
            assert replacement._initialized
            await replacement.aclose()
        else:
            assert replacement is client, "uncertain/explicit retirement granted fresh dispatch authority"
    finally:
        if not startup.done():
            startup.cancel()
        await await_owned_task(startup)
        if client._close_task is not None:
            await await_owned_task(client._close_task)
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["authority", "unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("route", ["state", "skip", "restart", "verify", "reload"])
async def test_bootstrap_actual_command_preserves_original_service_receipt(tmp_path, monkeypatch, kind, route):
    service, db = await service_for(tmp_path)
    await service.set_bootstrap_state(BootstrapState.PENDING)
    original = receipt(kind)
    async def fail(*args, **kwargs):
        raise original
    feature = BootstrapFeature(SimpleNamespace(bootstrap_service=service))
    try:
        if route == "state":
            monkeypatch.setattr(service, "get_bootstrap_state", fail)
            operation = feature.skip_discovery
        elif route == "skip":
            monkeypatch.setattr(service, "skip_discovery", fail)
            operation = feature.skip_discovery
        elif route in {"restart", "verify"}:
            monkeypatch.setattr(service, "restart_discovery", fail if route == "restart" else successful_restart)
            if route == "verify":
                monkeypatch.setattr(service, "_load_discovery_history_strict", fail)
            operation = feature.restart_discovery
        else:
            monkeypatch.setattr(service, "skip_discovery", successful_skip)
            monkeypatch.setattr(feature, "_verify_soul_md_exists", lambda: (True, str(tmp_path / "SOUL.md")))
            feature.agent.context_builder = SimpleNamespace(load_canonical_soul_resource=fail)
            operation = feature.skip_discovery
        with pytest.raises(BaseException) as caught:
            await operation()
        assert caught.value is original
    finally:
        await db.close()


async def successful_restart():
    return SimpleNamespace(history_clear_succeeded=True)


async def successful_skip():
    return "offline complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["authority", "unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("route", ["chat", "sleep_hook", "scheduler"])
async def test_actual_sleep_dispatch_boundaries_preserve_original_control(monkeypatch, kind, route):
    original = receipt(kind)
    async def fail(*args, **kwargs):
        raise original
    if route == "chat":
        hook = ReflectionSleepHook()
        monkeypatch.setattr(hook, "_attest_application", fail)
        operation = lambda: hook._attest_with_chat(SimpleNamespace(), None, [candidate(1)], None)
    elif route == "sleep_hook":
        owner = SleepMixin()
        owner.sleep_hooks = [SimpleNamespace(on_pre_sleep=fail)]
        report = SleepReport(success=True)
        operation = lambda: owner._run_pre_sleep_hooks(report)
    else:
        feature = SchedulerFeature(SimpleNamespace(sleep=fail))
        feature._agent_id = "did:test:sleep-receipt"
        operation = lambda: feature._handle_sleep({"skip_reflection": False})
    with pytest.raises(BaseException) as caught:
        await operation()
    assert caught.value is original


def candidate(identifier):
    return RetrievedMemoryCandidate(identifier, "offline", datetime.now(timezone.utc).isoformat(), None, "user")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "cancel_unknown", "group"])
@pytest.mark.parametrize("cancel_parent", [False, True])
async def test_actual_decision_attestation_joins_children_and_keeps_original_receipt(kind, cancel_parent):
    original = receipt(kind)
    entered = asyncio.Event()
    retired = []
    child_tasks = []
    async def decide(*args, **kwargs):
        task = asyncio.current_task()
        child_tasks.append(task)
        entered.set()
        try:
            if cancel_parent:
                await asyncio.Event().wait()
            raise original
        except asyncio.CancelledError:
            raise original
        finally:
            retired.append(task)
    owner = SimpleNamespace(llm_service=SimpleNamespace(decide=decide))
    hook = ReflectionSleepHook(backend="decision")
    parent = asyncio.create_task(hook._attest_with_decisions(owner, None, [candidate(1), candidate(2)], None))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if cancel_parent:
            parent.cancel()
        outcome = await await_owned_task(parent)
        assert outcome.error is not None and carries(outcome.error, original), "decision gather erased original child receipt"
        assert is_execution_control_error(outcome.error)
        assert child_tasks and all(task.done() for task in child_tasks)
        assert set(retired) == set(child_tasks)
    finally:
        if not parent.done():
            parent.cancel()
        await await_owned_task(parent)
        for task in child_tasks:
            await await_owned_task(task)
