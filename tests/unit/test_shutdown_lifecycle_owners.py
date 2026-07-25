"""Production shutdown callers retain deferred durable cleanup ownership."""

from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign import cli, main, server
from kestrel_sovereign.kestrel_agent import (
    KestrelAgent,
    await_agent_shutdown_completion,
    await_lifecycle_task_completion,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager


class _DeferredShutdownAgent:
    """Small production-shaped agent with a joinable deferred tail."""

    agent_id = "did:test:deferred-shutdown"

    def __init__(self) -> None:
        self._never = asyncio.Event()
        self.completion_entered = asyncio.Event()
        self.allow_completion = asyncio.Event()
        self.shutdown_calls = 0
        self.completion_calls = 0
        # AgentManager treats this as evidence that a timeout handed cleanup
        # to a durable continuation.
        self._durable_shutdown_continuation = object()

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        await self._never.wait()

    async def wait_for_shutdown_completion(self) -> None:
        self.completion_calls += 1
        self.completion_entered.set()
        await self.allow_completion.wait()


class _CancellationTailWithoutContinuation:
    """A real shutdown shape: tail closes storage, then re-raises cancel."""

    agent_id = "did:test:cancelled-without-continuation"

    def __init__(self) -> None:
        self.shutdown_entered = asyncio.Event()
        self.storage_closed = asyncio.Event()

    async def shutdown(self) -> None:
        self.shutdown_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.storage_closed.set()
            raise


class _FailOnceDispatcher:
    def __init__(self, fail_stage: str) -> None:
        self.fail_stage = fail_stage
        self.calls = 0

    async def shutdown_durable_delivery(self) -> None:
        self.calls += 1
        if self.fail_stage == "release" and self.calls == 1:
            raise RuntimeError("owner release failed once")


class _FailOnceStorage:
    def __init__(self, fail_stage: str) -> None:
        self.fail_stage = fail_stage
        self.calls = 0

    async def close(self) -> None:
        self.calls += 1
        if self.fail_stage == "storage" and self.calls == 1:
            raise RuntimeError("storage close failed once")


class _AlwaysFailDispatcher:
    def __init__(self) -> None:
        self.calls = 0

    async def shutdown_durable_delivery(self) -> None:
        self.calls += 1
        raise RuntimeError("owner release failed permanently")


class _TerminalFailureShutdownAgent(_DeferredShutdownAgent):
    """A continuation that has already consumed its one retry and failed."""

    async def wait_for_shutdown_completion(self) -> None:
        self.completion_calls += 1
        self.completion_entered.set()
        raise RuntimeError("durable cleanup failed after retry")


class _PhoenixSupervisor:
    """Server-shaped Phoenix double that records every required teardown step."""

    def __init__(self) -> None:
        self.close_entered = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.stopped = False

    def reconcile_storage_conflict(self) -> None:
        return None

    def prepare_storage(self) -> None:
        return None

    async def aclose(self) -> None:
        self.close_entered.set()
        await self.allow_close.wait()

    def stop(self) -> None:
        self.stopped = True


class _FailingPhoenixSupervisor(_PhoenixSupervisor):
    async def aclose(self) -> None:
        self.close_entered.set()
        raise RuntimeError("proxy close failed")


class _LateStartingPhoenixSupervisor:
    """A start worker that can finish only after supervision is cancelled."""

    def __init__(self) -> None:
        self.start_entered = threading.Event()
        self.allow_start = threading.Event()
        self.started = False
        self.stopped = False
        self.started_when_stopped = False

    def start(self, *, wait_for_health: bool = False) -> bool:
        assert wait_for_health is False
        self.start_entered.set()
        assert self.allow_start.wait(timeout=1.0)
        self.started = True
        return True

    async def is_reachable(self) -> bool:
        return False

    @property
    def running(self) -> bool:
        return self.started

    async def aclose(self) -> None:
        return None

    def stop(self) -> None:
        self.started_when_stopped = self.started
        self.stopped = True


@pytest.mark.asyncio
async def test_completion_join_survives_repeated_cancellation() -> None:
    agent = _DeferredShutdownAgent()
    join = asyncio.create_task(await_agent_shutdown_completion(agent))
    await asyncio.wait_for(agent.completion_entered.wait(), timeout=1.0)
    join.cancel()
    await asyncio.sleep(0)
    join.cancel()
    agent.allow_completion.set()

    assert await asyncio.wait_for(join, timeout=1.0) is True
    assert agent.completion_calls == 1


@pytest.mark.asyncio
async def test_completion_join_classifies_simultaneous_owned_and_caller_cancellation() -> None:
    """The joiner must not swallow its cancellation when both tasks cancel together."""
    entered = asyncio.Event()
    never = asyncio.Event()

    async def owned_work() -> None:
        entered.set()
        await never.wait()

    owned = asyncio.create_task(owned_work())
    join = asyncio.create_task(await_lifecycle_task_completion(owned))
    await asyncio.wait_for(entered.wait(), timeout=1.0)

    # Same event-loop turn: this was previously reported as only owned-task
    # cancellation, allowing lifecycle callers to swallow their own signal.
    owned.cancel()
    join.cancel()

    cancelled, failure = await asyncio.wait_for(join, timeout=1.0)
    assert cancelled is True
    assert isinstance(failure, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_phoenix_cleanup_survives_repeated_cancellation() -> None:
    """The server joins Phoenix work and reaps its child before re-raising cancel."""
    app = SimpleNamespace(state=SimpleNamespace())
    supervisor = _PhoenixSupervisor()
    supervisor_task = asyncio.create_task(asyncio.Event().wait())
    app.state.phoenix = supervisor
    app.state.phoenix_task = supervisor_task

    shutdown = asyncio.create_task(server._shutdown_phoenix(app))
    await asyncio.wait_for(supervisor.close_entered.wait(), timeout=1.0)
    shutdown.cancel()
    await asyncio.sleep(0)
    shutdown.cancel()
    supervisor.allow_close.set()

    assert await asyncio.wait_for(shutdown, timeout=1.0) is True
    assert supervisor_task.cancelled()
    assert supervisor.stopped is True
    assert app.state.phoenix_task is None
    assert app.state.phoenix is None


@pytest.mark.asyncio
async def test_phoenix_cleanup_stops_child_when_proxy_close_fails() -> None:
    """A proxy-client failure cannot leave the supervised child holding its port."""
    app = SimpleNamespace(state=SimpleNamespace())
    supervisor = _FailingPhoenixSupervisor()
    app.state.phoenix = supervisor
    app.state.phoenix_task = None

    assert await server._shutdown_phoenix(app) is False
    assert supervisor.close_entered.is_set()
    assert supervisor.stopped is True
    assert app.state.phoenix is None


@pytest.mark.asyncio
async def test_phoenix_shutdown_joins_late_executor_start_before_stop() -> None:
    """Cancelling supervision cannot let a thread launch Phoenix after stop."""
    app = SimpleNamespace(state=SimpleNamespace())
    supervisor = _LateStartingPhoenixSupervisor()
    app.state.phoenix = supervisor
    app.state.phoenix_start_task = None
    app.state.phoenix_task = asyncio.create_task(
        server._supervise_phoenix_startup(app, supervisor)
    )

    assert await asyncio.wait_for(
        asyncio.to_thread(supervisor.start_entered.wait), timeout=1.0
    )
    shutdown = asyncio.create_task(server._shutdown_phoenix(app))
    await asyncio.sleep(0)
    assert not shutdown.done()
    assert supervisor.stopped is False

    supervisor.allow_start.set()
    assert await asyncio.wait_for(shutdown, timeout=1.0) is False
    assert supervisor.stopped is True
    assert supervisor.started_when_stopped is True
    assert app.state.phoenix_start_task is None


@pytest.mark.asyncio
async def test_lifespan_cancellation_at_yield_runs_every_teardown_phase(
    monkeypatch,
) -> None:
    """A cancellation thrown into the lifespan yield still owns all teardown."""
    supervisor = _PhoenixSupervisor()
    supervisor.allow_close.set()
    entered = asyncio.Event()
    never = asyncio.Event()

    class _Manager:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        async def shutdown_all(self) -> None:
            self.shutdown_calls += 1

    manager = _Manager()

    @asynccontextmanager
    async def fake_startup(app):
        app.state.agent_manager = manager
        app.state.agent = None
        app.state.phoenix = supervisor
        app.state.phoenix_task = None
        app.state.phoenix_start_task = None
        yield

    async def noop_host_shutdown(_app) -> None:
        return None

    monkeypatch.setattr(server, "_lifespan_startup", fake_startup)
    monkeypatch.setattr(server, "_shutdown_host_features", noop_host_shutdown)

    async def run_lifespan() -> None:
        async with server.lifespan(FastAPI()):
            entered.set()
            await never.wait()

    task = asyncio.create_task(run_lifespan())
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
    assert manager.shutdown_calls == 1
    assert supervisor.close_entered.is_set()
    assert supervisor.stopped is True


@pytest.mark.asyncio
async def test_lifespan_host_phase_cancellation_still_reaches_agents_and_phoenix(
    monkeypatch,
) -> None:
    """A cancelled host-feature phase cannot become the fleet's last owner."""
    supervisor = _PhoenixSupervisor()
    supervisor.allow_close.set()

    class _Manager:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        async def shutdown_all(self) -> None:
            self.shutdown_calls += 1

    manager = _Manager()

    @asynccontextmanager
    async def fake_startup(app):
        app.state.agent_manager = manager
        app.state.agent = None
        app.state.phoenix = supervisor
        app.state.phoenix_task = None
        app.state.phoenix_start_task = None
        yield

    async def cancelled_host_shutdown(_app) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(server, "_lifespan_startup", fake_startup)
    monkeypatch.setattr(server, "_shutdown_host_features", cancelled_host_shutdown)

    with pytest.raises(asyncio.CancelledError):
        async with server.lifespan(FastAPI()):
            pass

    assert manager.shutdown_calls == 1
    assert supervisor.close_entered.is_set()
    assert supervisor.stopped is True


@pytest.mark.asyncio
async def test_lifespan_reaps_phoenix_after_agent_manager_cancellation(
    monkeypatch, tmp_path,
) -> None:
    """Agent cancellation cannot bypass the server's Phoenix teardown owner."""
    from kestrel_sovereign import host_features as hf
    from kestrel_sovereign.a2a import did_registry
    from kestrel_sovereign.multi_agent import agent_manager, config as ma_config
    from kestrel_sovereign import phoenix_supervisor as phoenix_module
    from kestrel_sovereign.security import demo_isolation

    config_path = tmp_path / "multi_agent.toml"
    config_path.write_text("[host]\nport = 8888\n")
    fake_config = SimpleNamespace(
        host=SimpleNamespace(bind="127.0.0.1", port=8888), agents={}
    )
    supervisor = _PhoenixSupervisor()
    supervisor.allow_close.set()

    class _CancelledManager:
        init_failures = []

        async def load_from_config(self, config) -> int:
            assert config is fake_config
            return 0

        def list_agents(self):
            return []

        async def shutdown_all(self) -> None:
            raise asyncio.CancelledError()

    async def supervise_forever(_app, _supervisor) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "resolve_multi_agent_path", lambda _env: config_path)
    monkeypatch.setattr(ma_config.MultiAgentConfig, "load", lambda *_a, **_k: fake_config)
    monkeypatch.setattr(agent_manager, "AgentManager", lambda **_k: _CancelledManager())
    monkeypatch.setattr(did_registry, "install_a2a_did_resolver", lambda *_a, **_k: None)
    monkeypatch.setattr(demo_isolation, "classify_server_mode", lambda _agents: False)
    monkeypatch.setattr(phoenix_module, "should_supervise_phoenix", lambda: True)
    monkeypatch.setattr(phoenix_module, "PhoenixSupervisor", lambda: supervisor)
    monkeypatch.setattr(phoenix_module, "autowire_otlp_endpoint", lambda _env: None)
    monkeypatch.setattr(phoenix_module, "autowire_otel_project", lambda _env: None)
    monkeypatch.setattr(server, "_supervise_phoenix_startup", supervise_forever)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "_unmount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_unmount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **_k: [])

    with pytest.raises(asyncio.CancelledError):
        async with server.lifespan(FastAPI()):
            pass

    assert supervisor.close_entered.is_set()
    assert supervisor.stopped is True


@pytest.mark.asyncio
async def test_server_timeout_keeps_lifecycle_owner_until_completion(monkeypatch) -> None:
    agent = _DeferredShutdownAgent()
    monkeypatch.setattr(server, "SHUTDOWN_TIMEOUT", 0.01)

    shutdown = asyncio.create_task(server._shutdown_single_agent(agent))
    await asyncio.wait_for(agent.completion_entered.wait(), timeout=1.0)
    assert not shutdown.done()

    agent.allow_completion.set()
    await asyncio.wait_for(shutdown, timeout=1.0)
    assert agent.shutdown_calls == 1
    assert agent.completion_calls == 1


@pytest.mark.asyncio
async def test_main_timeout_joins_completion_before_returning(tmp_path, monkeypatch) -> None:
    agent = _DeferredShutdownAgent()
    monkeypatch.setattr(main, "KestrelAgent", lambda **_kwargs: agent)
    monkeypatch.setattr(main, "LLMService", MagicMock())
    monkeypatch.setattr(main, "SHUTDOWN_TIMEOUT", 0.01)
    monkeypatch.setattr(main, "get_agent_did_async", AsyncMock(return_value=agent.agent_id))
    monkeypatch.setattr(sys, "argv", ["kestrel-main", str(tmp_path)])
    monkeypatch.setattr("builtins.input", lambda _prompt: "!quit")

    task = asyncio.create_task(main.main())
    await asyncio.wait_for(agent.completion_entered.wait(), timeout=1.0)
    assert not task.done()
    agent.allow_completion.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert agent.completion_calls == 1


@pytest.mark.asyncio
async def test_cli_shell_timeout_joins_completion_before_returning(
    tmp_path, monkeypatch
) -> None:
    agent = _DeferredShutdownAgent()
    storage = MagicMock()
    storage.initialize = AsyncMock()
    storage.close = AsyncMock()
    storage.get_nodes_by_type = AsyncMock(
        return_value=[SimpleNamespace(node_id=agent.agent_id)]
    )

    monkeypatch.setattr("kestrel_sovereign.storage.AsyncStorage", lambda _path: storage)
    monkeypatch.setattr("kestrel_sovereign.kestrel_agent.KestrelAgent", lambda **_kwargs: agent)
    monkeypatch.setattr("kestrel_sovereign.llm.service.LLMService", MagicMock())
    monkeypatch.setattr("kestrel_sovereign.kestrel_config.constants.SHUTDOWN_TIMEOUT", 0.01)
    monkeypatch.setattr("builtins.input", lambda _prompt: "!quit")

    task = asyncio.create_task(
        cli._run_shell(Path(tmp_path), SimpleNamespace(app=None))
    )
    await asyncio.wait_for(agent.completion_entered.wait(), timeout=1.0)
    assert not task.done()
    agent.allow_completion.set()
    assert await asyncio.wait_for(task, timeout=1.0) == 0
    assert agent.completion_calls == 1


@pytest.mark.asyncio
async def test_manager_removal_and_unregistered_cleanup_join_timeout_tails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )

    manager = AgentManager()
    managed = _DeferredShutdownAgent()
    manager._agents["managed"] = managed
    manager._agent_names[managed.agent_id] = "managed"
    removal = asyncio.create_task(manager.remove_agent("managed"))
    await asyncio.wait_for(managed.completion_entered.wait(), timeout=1.0)
    assert not removal.done()
    managed.allow_completion.set()
    assert await asyncio.wait_for(removal, timeout=1.0) is True
    assert managed.completion_calls == 1

    unregistered = _DeferredShutdownAgent()
    cleanup = asyncio.create_task(
        AgentManager._shutdown_unregistered_agent("unregistered", unregistered)
    )
    await asyncio.wait_for(unregistered.completion_entered.wait(), timeout=1.0)
    assert not cleanup.done()
    unregistered.allow_completion.set()
    await asyncio.wait_for(cleanup, timeout=1.0)
    assert unregistered.completion_calls == 1


@pytest.mark.asyncio
async def test_manager_unpublishes_a_cancelled_terminal_shutdown_without_continuation(
    monkeypatch,
) -> None:
    """A task's terminal cancellation, not a continuation attribute, proves cleanup."""
    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )
    manager = AgentManager()
    agent = _CancellationTailWithoutContinuation()
    manager._agents["terminal"] = agent
    manager._agent_names[agent.agent_id] = "terminal"

    removal = asyncio.create_task(manager.remove_agent("terminal"))
    await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)

    assert await asyncio.wait_for(removal, timeout=1.0) is True
    assert agent.storage_closed.is_set()
    assert manager.get_agent("terminal") is None


@pytest.mark.asyncio
async def test_manager_shutdown_all_continues_after_terminal_agent_failure(
    monkeypatch,
) -> None:
    """A failed agent stays published, but cannot abandon later fleet cleanup."""
    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )
    manager = AgentManager()
    failed = _TerminalFailureShutdownAgent()
    healthy = _CancellationTailWithoutContinuation()
    manager._agents = {"failed": failed, "healthy": healthy}
    manager._agent_names = {
        failed.agent_id: "failed",
        healthy.agent_id: "healthy",
    }
    manager._parent_children = {"did:test:parent": ["failed", "healthy"]}
    manager._child_mandates = {"failed": object(), "healthy": object()}

    with pytest.raises(ExceptionGroup, match="fleet agents failed"):
        await manager.shutdown_all()

    assert manager.get_agent("failed") is failed
    assert manager.get_agent("healthy") is None
    assert healthy.storage_closed.is_set()
    assert manager.get_children("did:test:parent") == ["failed"]
    assert manager.get_mandate("failed") is not None
    assert manager.get_mandate("healthy") is None


@pytest.mark.asyncio
async def test_manager_shutdown_all_retains_tracking_for_false_removal(
    monkeypatch,
) -> None:
    """A false removal result is terminal evidence, not permission to forget it."""
    manager = AgentManager()
    stuck = SimpleNamespace(agent_id="did:test:false-removal")
    manager._agents["stuck"] = stuck
    manager._agent_names[stuck.agent_id] = "stuck"
    manager._parent_children = {"did:test:parent": ["stuck"]}
    mandate = object()
    manager._child_mandates = {"stuck": mandate}
    attempted: list[str] = []

    async def return_false(name: str) -> bool:
        attempted.append(name)
        return False

    monkeypatch.setattr(manager, "remove_agent", return_false)

    with pytest.raises(ExceptionGroup, match="fleet agents failed"):
        await manager.shutdown_all()

    assert attempted == ["stuck"]
    assert manager.get_agent("stuck") is stuck
    assert manager.get_children("did:test:parent") == ["stuck"]
    assert manager.get_mandate("stuck") is mandate


@pytest.mark.asyncio
async def test_manager_unpublishes_and_releases_before_propagating_cancellation() -> None:
    """Cancellation cannot leave a closed agent routable or its hold stranded."""
    manager = AgentManager()
    agent = _CancellationTailWithoutContinuation()
    manager._agents["terminal"] = agent
    manager._agent_names[agent.agent_id] = "terminal"

    removal = asyncio.create_task(manager.remove_agent("terminal"))
    await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)
    removal.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(removal, timeout=1.0)
    assert agent.storage_closed.is_set()
    assert manager.get_agent("terminal") is None


@pytest.mark.asyncio
async def test_manager_retries_budget_release_join_before_propagating_cancellation() -> None:
    """A second signal cannot interrupt the final delegated-budget release."""
    manager = AgentManager()
    agent = _CancellationTailWithoutContinuation()
    manager._agents["terminal"] = agent
    manager._agent_names[agent.agent_id] = "terminal"
    release_started = asyncio.Event()
    allow_release = asyncio.Event()
    released: list[str] = []

    async def slow_release(name: str) -> None:
        release_started.set()
        await allow_release.wait()
        released.append(name)

    manager._release_child_budget = slow_release
    removal = asyncio.create_task(manager.remove_agent("terminal"))
    await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)
    removal.cancel()
    await asyncio.wait_for(release_started.wait(), timeout=1.0)
    removal.cancel()
    allow_release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(removal, timeout=1.0)
    assert released == ["terminal"]
    assert manager.get_agent("terminal") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_stage", ["release", "storage"])
async def test_real_agent_retries_failed_durable_continuation_before_exit(
    tmp_path, fail_stage
) -> None:
    """Release and close failures both receive one owned retry before exit."""
    agent = KestrelAgent(
        did=f"did:test:continuation-retry:{fail_stage}",
        storage_path=str(tmp_path / f"{fail_stage}.db"),
    )
    dispatcher = _FailOnceDispatcher(fail_stage)
    storage = _FailOnceStorage(fail_stage)
    agent.dispatcher = dispatcher
    agent.storage = storage

    await agent._ensure_durable_shutdown_continuation(dispatcher)
    assert await await_agent_shutdown_completion(agent) is False
    assert dispatcher.calls == 2
    assert storage.calls == (1 if fail_stage == "release" else 2)


@pytest.mark.asyncio
async def test_real_agent_reports_terminal_continuation_failure_after_retry(
    tmp_path,
) -> None:
    """A persistent failure is surfaced; storage is never closed underneath it."""
    agent = KestrelAgent(
        did="did:test:continuation-terminal-failure",
        storage_path=str(tmp_path / "terminal.db"),
    )
    dispatcher = _AlwaysFailDispatcher()
    storage = _FailOnceStorage("none")
    agent.dispatcher = dispatcher
    agent.storage = storage

    await agent._ensure_durable_shutdown_continuation(dispatcher)
    with pytest.raises(RuntimeError, match="dispatcher release and storage close"):
        await await_agent_shutdown_completion(agent)
    assert dispatcher.calls == 2
    assert storage.calls == 0


@pytest.mark.asyncio
async def test_server_and_unregistered_paths_do_not_swallow_terminal_cleanup_failure(
    monkeypatch,
) -> None:
    """A persistent completion failure is fatal to every lifecycle owner."""
    monkeypatch.setattr(server, "SHUTDOWN_TIMEOUT", 0.01)
    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )

    server_agent = _TerminalFailureShutdownAgent()
    with pytest.raises(RuntimeError, match="durable cleanup failed"):
        await server._shutdown_single_agent(server_agent)

    unregistered = _TerminalFailureShutdownAgent()
    with pytest.raises(RuntimeError, match="durable cleanup failed"):
        await AgentManager._shutdown_unregistered_agent("failed", unregistered)


@pytest.mark.asyncio
async def test_main_and_cli_do_not_swallow_terminal_cleanup_failure(
    tmp_path, monkeypatch,
) -> None:
    """Interactive entry points must not let a live backend disappear on error."""
    main_agent = _TerminalFailureShutdownAgent()
    monkeypatch.setattr(main, "KestrelAgent", lambda **_kwargs: main_agent)
    monkeypatch.setattr(main, "LLMService", MagicMock())
    monkeypatch.setattr(main, "SHUTDOWN_TIMEOUT", 0.01)
    monkeypatch.setattr(main, "get_agent_did_async", AsyncMock(return_value=main_agent.agent_id))
    monkeypatch.setattr(sys, "argv", ["kestrel-main", str(tmp_path)])
    monkeypatch.setattr("builtins.input", lambda _prompt: "!quit")

    with pytest.raises(RuntimeError, match="durable cleanup failed"):
        await main.main()

    cli_agent = _TerminalFailureShutdownAgent()
    storage = MagicMock()
    storage.initialize = AsyncMock()
    storage.close = AsyncMock()
    storage.get_nodes_by_type = AsyncMock(
        return_value=[SimpleNamespace(node_id=cli_agent.agent_id)]
    )
    monkeypatch.setattr("kestrel_sovereign.storage.AsyncStorage", lambda _path: storage)
    monkeypatch.setattr(
        "kestrel_sovereign.kestrel_agent.KestrelAgent", lambda **_kwargs: cli_agent
    )
    monkeypatch.setattr("kestrel_sovereign.llm.service.LLMService", MagicMock())
    monkeypatch.setattr("kestrel_sovereign.kestrel_config.constants.SHUTDOWN_TIMEOUT", 0.01)

    with pytest.raises(RuntimeError, match="durable cleanup failed"):
        await cli._run_shell(Path(tmp_path), SimpleNamespace(app=None))
