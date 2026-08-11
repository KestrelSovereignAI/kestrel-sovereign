"""Production shutdown callers retain deferred durable cleanup ownership."""

from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from starlette.routing import Mount, Route

from kestrel_sovereign import cli, main, server
from kestrel_sovereign.kestrel_agent import (
    KestrelAgent,
    await_agent_shutdown_completion,
    await_lifecycle_task_completion,
)
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.spawn.delegated_wallet import (
    BudgetAllocation,
    BudgetExceededError,
    DelegatedWallet,
)


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


class _CancellationHostileShutdownAgent:
    """Lifecycle double whose active cognition suppresses every cancellation."""

    agent_id = "did:test:cancellation-hostile-cognition"

    def __init__(self) -> None:
        self.shutdown_entered = asyncio.Event()
        self.allow_shutdown_finish = asyncio.Event()
        self.cancellation_attempts = 0

    async def shutdown(self) -> None:
        self.shutdown_entered.set()
        while not self.allow_shutdown_finish.is_set():
            try:
                await self.allow_shutdown_finish.wait()
            except asyncio.CancelledError:
                self.cancellation_attempts += 1

    def handoff_shutdown_to_reaper(self, shutdown_task):
        async def reap() -> None:
            await asyncio.shield(shutdown_task)

        return asyncio.create_task(reap())


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
async def test_completion_join_returns_a_live_owned_task_failure() -> None:
    """A failing owned task is terminal data, not an early shutdown escape."""
    entered = asyncio.Event()
    release_failure = asyncio.Event()

    async def owned_work() -> None:
        entered.set()
        await release_failure.wait()
        raise RuntimeError("owned lifecycle failure")

    owned = asyncio.create_task(owned_work())
    join = asyncio.create_task(await_lifecycle_task_completion(owned))
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    assert not join.done()

    release_failure.set()
    cancelled, failure = await asyncio.wait_for(join, timeout=1.0)

    assert cancelled is False
    assert isinstance(failure, RuntimeError)
    assert str(failure) == "owned lifecycle failure"


@pytest.mark.asyncio
async def test_completion_join_preserves_repeated_cancellation_with_owned_failure() -> None:
    """Caller cancellation is recorded without losing the owned failure."""
    entered = asyncio.Event()
    release_failure = asyncio.Event()

    async def owned_work() -> None:
        entered.set()
        await release_failure.wait()
        raise RuntimeError("owned failure after caller cancellation")

    owned = asyncio.create_task(owned_work())
    join = asyncio.create_task(await_lifecycle_task_completion(owned))
    await asyncio.wait_for(entered.wait(), timeout=1.0)

    join.cancel()
    await asyncio.sleep(0)
    join.cancel()
    release_failure.set()

    cancelled, failure = await asyncio.wait_for(join, timeout=1.0)
    assert cancelled is True
    assert isinstance(failure, RuntimeError)
    assert str(failure) == "owned failure after caller cancellation"


@pytest.mark.asyncio
async def test_server_shutdown_drains_host_agent_and_phoenix_after_failures(
    monkeypatch,
) -> None:
    """Each phase runs in order; terminal errors surface only after the drain."""
    app = FastAPI()
    phases: list[str] = []

    async def fail_host(_app) -> None:
        phases.append("host")
        raise RuntimeError("host failure")

    async def fail_agents(_app) -> None:
        phases.append("agents")
        raise RuntimeError("agent failure")

    async def finish_phoenix(_app) -> bool:
        phases.append("phoenix")
        return False

    monkeypatch.setattr(server, "_shutdown_host_features", fail_host)
    monkeypatch.setattr(server, "_shutdown_server_agents", fail_agents)
    monkeypatch.setattr(server, "_shutdown_phoenix", finish_phoenix)

    cancelled, failure = await server._shutdown_server_resources(app)

    assert phases == ["host", "agents", "phoenix"]
    assert cancelled is False
    assert isinstance(failure, RuntimeError)
    assert str(failure) == "host failure"


@pytest.mark.asyncio
async def test_host_scheduler_drains_cold_onboarding_before_feature_unmount(
    monkeypatch,
):
    """A cold wake cannot remount routes/UI after the host teardown pass."""

    app = FastAPI()
    events: list[str] = []
    onboarding_entered = asyncio.Event()
    release_onboarding = asyncio.Event()
    stale_route = None
    stale_asset = None

    async def cold_onboarding_remount() -> None:
        nonlocal stale_route, stale_asset
        onboarding_entered.set()
        await release_onboarding.wait()
        stale_route = Route("/cold-onboarding-route", endpoint=lambda _request: None)
        stale_asset = Mount("/features/cold/static", app=lambda *_args: None)
        app.routes.extend((stale_route, stale_asset))
        app.state._feature_routes = [stale_route]
        app.state._feature_ui_mounts = [stale_asset]
        app.state._feature_ui_mount_paths = {"/features/cold/static"}
        events.append("cold_onboarding_remounted")

    onboarding = asyncio.create_task(cold_onboarding_remount())
    await asyncio.wait_for(onboarding_entered.wait(), timeout=1.0)

    class _DrainingRunner:
        def __init__(self) -> None:
            self.stop_entered = asyncio.Event()
            self.drained = asyncio.Event()

        async def stop(self) -> None:
            events.append("scheduler_stop")
            self.stop_entered.set()
            await onboarding
            self.drained.set()
            events.append("scheduler_drained")

    class _Storage:
        async def close(self) -> None:
            events.append("scheduler_storage_closed")

    runner = _DrainingRunner()
    app.state.host_scheduler_runner = runner
    app.state.host_scheduler_storage = _Storage()
    app.state._feature_routes = []
    app.state._feature_ui_mounts = []
    app.state._feature_ui_mount_paths = set()

    async def host_feature_teardown(target_app) -> None:
        assert runner.drained.is_set()
        events.append("host_features_unmounted")
        server._unmount_feature_routers(target_app)
        server._unmount_feature_ui_assets(target_app)

    monkeypatch.setattr(server, "_shutdown_host_features", host_feature_teardown)

    shutdown = asyncio.create_task(server._shutdown_server_resources(app))
    await asyncio.wait_for(runner.stop_entered.wait(), timeout=1.0)
    assert "host_features_unmounted" not in events

    release_onboarding.set()
    cancelled, failure = await asyncio.wait_for(shutdown, timeout=1.0)

    assert cancelled is False
    assert failure is None
    assert events.index("scheduler_drained") < events.index("host_features_unmounted")
    assert stale_route not in app.routes
    assert stale_asset not in app.routes
    assert app.state._feature_routes == []
    assert app.state._feature_ui_mounts == []
    assert app.state._feature_ui_mount_paths == set()

    # A next lifespan can mount a new copy without inheriting the cold wake's
    # stale route or static mount.
    fresh_route = Route("/cold-onboarding-route", endpoint=lambda _request: None)
    fresh_asset = Mount("/features/cold/static", app=lambda *_args: None)
    app.routes.extend((fresh_route, fresh_asset))
    assert sum(
        route.path == "/cold-onboarding-route" for route in app.routes
    ) == 1
    assert sum(route.path == "/features/cold/static" for route in app.routes) == 1


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

        def set_agent_registration_hook(self, _hook) -> None:
            return None

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
async def test_host_scheduler_startup_failure_rolls_back_loaded_agents(
    monkeypatch, tmp_path,
) -> None:
    """A failed hosted scheduler cannot orphan already-loaded agents."""
    from kestrel_sovereign import host_features as hf
    from kestrel_sovereign.a2a import did_registry
    from kestrel_sovereign.multi_agent import agent_manager, config as ma_config

    config_path = tmp_path / "multi_agent.toml"
    config_path.write_text("[host]\nport = 8888\n")
    fake_config = SimpleNamespace(
        host=SimpleNamespace(bind="127.0.0.1", port=8888), agents={}
    )
    loaded_agent = SimpleNamespace(shutdown=AsyncMock())

    class _Manager:
        init_failures = []

        def __init__(self) -> None:
            self._agents = {"already-loaded": loaded_agent}
            self.shutdown_calls = 0

        def set_agent_registration_hook(self, _hook) -> None:
            return None

        async def load_from_config(self, config) -> int:
            assert config is fake_config
            return 1

        def list_agents(self):
            return dict(self._agents)

        async def shutdown_all(self) -> None:
            self.shutdown_calls += 1
            await loaded_agent.shutdown()
            self._agents.clear()

    manager = _Manager()
    monkeypatch.setenv("KESTREL_MULTI_AGENT", "1")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    monkeypatch.setattr(server, "resolve_multi_agent_path", lambda _env: config_path)
    monkeypatch.setattr(ma_config.MultiAgentConfig, "load", lambda *_a, **_k: fake_config)
    monkeypatch.setattr(agent_manager, "AgentManager", lambda **_k: manager)
    monkeypatch.setattr(did_registry, "install_a2a_did_resolver", lambda *_a, **_k: None)
    monkeypatch.setattr(
        server, "_start_host_scheduler", AsyncMock(side_effect=RuntimeError("scheduler boot failed"))
    )
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **_k: [])

    app = FastAPI()
    async with server._lifespan_startup(app):
        pass

    assert manager.shutdown_calls == 1
    loaded_agent.shutdown.assert_awaited_once()
    assert manager.list_agents() == {}
    assert app.state.agent_manager is None
    assert app.state.agent is None
    assert "scheduler boot failed" in app.state.startup_error


@pytest.mark.asyncio
async def test_host_scheduler_cancellation_closes_storage_published_before_initialize(
    monkeypatch,
) -> None:
    """Cancellation during initialize cannot strand an invisible DB owner."""
    entered_initialize = asyncio.Event()
    release_initialize = asyncio.Event()
    storage_instances = []

    class _BlockingStorage:
        def __init__(self, *, backend, dsn) -> None:
            assert backend == "postgres"
            assert dsn == "postgresql://scheduler-test"
            self.db = None
            self.closed = False
            storage_instances.append(self)

        async def initialize(self) -> None:
            entered_initialize.set()
            await release_initialize.wait()

        async def close(self) -> None:
            self.closed = True

    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(
            return_value={"did:pkh:warm": ("Warm", object())}
        ),
        cold_scheduler_identity_failures=[],
    )
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", _BlockingStorage
    )

    startup = asyncio.create_task(server._start_host_scheduler(app, manager, object()))
    await asyncio.wait_for(entered_initialize.wait(), timeout=1.0)
    assert app.state.host_scheduler_storage is storage_instances[0]
    assert app.state.host_scheduler_runner is None

    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(startup, timeout=1.0)

    assert storage_instances[0].closed is True
    assert app.state.host_scheduler_storage is None
    assert app.state.host_scheduler_runner is None


@pytest.mark.asyncio
async def test_host_scheduler_start_failure_closes_storage_even_when_runner_stop_fails(
    monkeypatch,
) -> None:
    """The startup cleanup finally block owns storage after a stop failure."""
    storage_instances = []
    runner_instances = []

    class _Storage:
        def __init__(self, *, backend, dsn) -> None:
            assert backend == "postgres"
            assert dsn == "postgresql://scheduler-test"
            self.db = object()
            self.closed = False
            storage_instances.append(self)

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    class _FailingRunner:
        def __init__(self, *args, **kwargs) -> None:
            self.stop_calls = 0
            runner_instances.append(self)

        async def start(self) -> None:
            raise RuntimeError("runner start failed")

        async def stop(self) -> None:
            self.stop_calls += 1
            raise RuntimeError("runner stop failed")

    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(
            return_value={"did:pkh:warm": ("Warm", object())}
        ),
        cold_scheduler_identity_failures=[],
    )
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", _Storage
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner.SchedulerRunner",
        _FailingRunner,
    )

    with pytest.raises(RuntimeError, match="runner start failed"):
        await server._start_host_scheduler(app, manager, object())

    assert runner_instances[0].stop_calls == 1
    assert storage_instances[0].closed is True
    assert app.state.host_scheduler_storage is None
    assert app.state.host_scheduler_runner is None


@pytest.mark.asyncio
async def test_host_scheduler_keeps_healthy_agents_when_cold_identity_is_unavailable(
    monkeypatch,
) -> None:
    """An unresolved configured tenant is omitted without rolling back peers."""
    storage_instances = []
    runner_instances = []

    class _Storage:
        def __init__(self, *, backend, dsn) -> None:
            assert backend == "postgres"
            assert dsn == "postgresql://scheduler-test"
            self.db = object()
            self.closed = False
            storage_instances.append(self)

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    class _Runner:
        def __init__(self, *args, **kwargs) -> None:
            self.authorized_agent_ids = tuple(kwargs["authorized_agent_ids"])
            self.stopped = False
            runner_instances.append(self)

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            self.stopped = True

    missing_identity = RuntimeError("identity database is not initialized")
    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(
            return_value={"did:pkh:warm": ("Warm", object())}
        ),
        cold_scheduler_identity_failures=[("Unincepted", missing_identity)],
    )
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", _Storage
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner.SchedulerRunner", _Runner
    )

    try:
        await server._start_host_scheduler(app, manager, object())

        assert runner_instances[0].authorized_agent_ids == ("did:pkh:warm",)
        assert app.state.scheduler_cold_agent_failures == [
            {
                "agent": "Unincepted",
                "scope": "identity",
                "state": "unavailable",
                "error_code": "scheduler_identity_unavailable",
                "cause_type": "RuntimeError",
            }
        ]
        assert app.state.scheduler_readiness_failures == (
            app.state.scheduler_cold_agent_failures
        )
        assert storage_instances[0].closed is False
    finally:
        await server._shutdown_host_scheduler(app)

    assert runner_instances[0].stopped is True
    assert storage_instances[0].closed is True


@pytest.mark.asyncio
async def test_host_scheduler_latches_runtime_protocol_failure_for_readiness(
    monkeypatch,
) -> None:
    """A post-start protocol outage is observable as a sticky readiness fault."""
    class _Storage:
        def __init__(self, *, backend, dsn) -> None:
            self.db = object()

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class _Runner:
        def __init__(self, *args, **kwargs) -> None:
            self._on_protocol_failure = kwargs["on_protocol_failure"]

        async def start(self) -> None:
            self._on_protocol_failure(RuntimeError("rollout state rejected"))

        async def stop(self) -> None:
            return None

    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(
            return_value={"did:pkh:warm": ("Warm", object())}
        ),
        cold_scheduler_identity_failures=[],
        is_scheduler_agent_authorized=lambda _did: True,
    )
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", _Storage
    )
    monkeypatch.setattr(
        "kestrel_sovereign.features.scheduler.runner.SchedulerRunner", _Runner
    )

    try:
        await server._start_host_scheduler(app, manager, object())
        assert app.state.scheduler_readiness_failures == [
            {
                "scope": "protocol",
                "state": "unavailable",
                "error_code": "scheduler_protocol_unavailable",
                "cause_type": "RuntimeError",
            }
        ]
    finally:
        await server._shutdown_host_scheduler(app)


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
async def test_manager_quarantines_cancellation_hostile_cognition_within_shutdown_bound(
    monkeypatch,
) -> None:
    """A hostile cognition turn cannot indefinitely retain routing or its budget.

    Before the control-plane handoff, ``remove_agent`` cancelled this task and
    then joined it forever.  The retained reaper now owns that join while the
    manager completes removal and the delegated-budget release inside the
    advertised shutdown timeout.
    """
    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )
    manager = AgentManager()
    agent = _CancellationHostileShutdownAgent()
    manager._agents["hostile"] = agent
    manager._agent_names[agent.agent_id] = "hostile"
    released: list[str] = []

    async def release_budget(name: str) -> None:
        released.append(name)

    manager._release_child_budget = release_budget
    removal = asyncio.create_task(manager.remove_agent("hostile"))
    await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)

    assert await asyncio.wait_for(removal, timeout=0.2) is True
    assert manager.get_agent("hostile") is None
    assert released == ["hostile"]
    quarantined = manager.quarantined_shutdowns()
    assert len(quarantined) == 1
    assert next(iter(quarantined.values()))["pending"] is True
    assert agent.cancellation_attempts >= 1

    # Do not leave the deliberately hostile fixture running beyond this test.
    agent.allow_shutdown_finish.set()
    for _ in range(100):
        if not next(iter(manager.quarantined_shutdowns().values()))["pending"]:
            break
        await asyncio.sleep(0)
    assert next(iter(manager.quarantined_shutdowns().values()))["pending"] is False


@pytest.mark.asyncio
async def test_quarantined_removal_fences_blocked_wallet_transfer_then_refunds_once(
    monkeypatch,
) -> None:
    """A blocked child debit cannot pin DELETE or spend after its later refund."""

    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )

    class BlockingChildWallet:
        _balances = {"FIL": {"main": Decimal("10")}}

        def __init__(self) -> None:
            self.transfer_entered = asyncio.Event()
            self.allow_transfer = asyncio.Event()
            self._debit_intents = {}

        def can_afford(self, amount, currency):
            return True

        def get_balance(self, currency, balance_type="main"):
            return self._balances[currency][balance_type]

        async def transfer(self, amount, memo="", currency=None):
            self.transfer_entered.set()
            await self.allow_transfer.wait()
            self._balances[currency]["main"] -= amount
            return True

        async def prepare_debit_intent(self, *, idempotency_key, amount, memo, currency):
            self._debit_intents.setdefault(
                idempotency_key,
                {"amount": amount, "memo": memo, "currency": currency, "outcome": False},
            )
            return idempotency_key

        async def execute_debit_intent(self, intent_id):
            intent = self._debit_intents[intent_id]
            if intent["outcome"] is True:
                return True
            outcome = await self.transfer(intent["amount"], intent["memo"], intent["currency"])
            intent["outcome"] = outcome
            return outcome

        async def resolve_debit_intent(self, intent_id):
            return self._debit_intents[intent_id]["outcome"]

    class ParentWallet:
        _balances = {"FIL": {"main": Decimal("0")}}

        def __init__(self) -> None:
            self.deposits: list[Decimal] = []

        async def deposit(self, amount, currency=None, to_audit=False, memo=""):
            self.deposits.append(amount)
            self._balances[currency]["main"] += amount
            return True

    manager = AgentManager()
    agent = _CancellationHostileShutdownAgent()
    manager._agents["hostile"] = agent
    manager._agent_names[agent.agent_id] = "hostile"
    child_wallet = BlockingChildWallet()
    parent_wallet = ParentWallet()
    delegated = DelegatedWallet(
        child_wallet,
        BudgetAllocation(child_did=agent.agent_id, parent_did="did:parent", amount=Decimal("10")),
    )
    manager._child_budgets["hostile"] = (delegated, parent_wallet)

    spending = asyncio.create_task(delegated.spend(Decimal("3"), "blocked debit", "FIL"))
    await asyncio.wait_for(child_wallet.transfer_entered.wait(), timeout=1.0)
    removal = asyncio.create_task(manager.remove_agent("hostile"))
    await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)

    assert await asyncio.wait_for(removal, timeout=0.2) is True
    assert manager.get_agent("hostile") is None
    assert "hostile" not in manager._child_budgets
    assert delegated.can_spend(Decimal("1")) is False
    assert parent_wallet.deposits == []

    child_wallet.allow_transfer.set()
    assert await asyncio.wait_for(spending, timeout=1.0) is True
    for _ in range(100):
        if parent_wallet.deposits:
            break
        await asyncio.sleep(0)
    assert parent_wallet.deposits == [Decimal("7")]
    with pytest.raises(BudgetExceededError):
        await delegated.spend(Decimal("1"), "post-refund", "FIL")

    agent.allow_shutdown_finish.set()
    for _ in range(100):
        if all(not item["pending"] for item in manager.quarantined_shutdowns().values()):
            break
        await asyncio.sleep(0)
    assert all(not item["pending"] for item in manager.quarantined_shutdowns().values())


@pytest.mark.asyncio
async def test_completed_quarantine_reaper_keeps_only_bounded_metadata() -> None:
    """Finished cleanup does not retain a Task/coroutine traceback forever."""

    manager = AgentManager()

    async def complete() -> None:
        return None

    task = asyncio.create_task(complete())
    reaper_id = manager._retain_quarantined_cleanup(
        name="retired", agent_id="did:test:retired", task=task
    )
    await task
    await asyncio.sleep(0)

    assert manager._quarantined_shutdown_reapers == {}
    assert len(manager._quarantined_shutdown_history) == 1
    assert manager.quarantined_shutdowns()[reaper_id] == {
        "agent_name": "retired",
        "agent_id": "did:test:retired",
        "pending": False,
        "started_monotonic": manager._quarantined_shutdown_history[0].started_monotonic,
        "completed_monotonic": manager._quarantined_shutdown_history[0].completed_monotonic,
        "failure": None,
    }

    async def fail() -> None:
        raise RuntimeError("wallet refund remained unsafe")

    failed_task = asyncio.create_task(fail())
    failed_reaper_id = manager._retain_quarantined_cleanup(
        name="unsafe", agent_id="did:test:unsafe", task=failed_task
    )
    with pytest.raises(RuntimeError, match="remained unsafe"):
        await failed_task
    await asyncio.sleep(0)
    assert manager._quarantined_shutdown_reapers == {}
    assert failed_reaper_id in manager._unsafe_quarantined_shutdown_failures
    assert manager.quarantined_shutdowns()[failed_reaper_id]["failure"] == (
        "RuntimeError: wallet refund remained unsafe"
    )


@pytest.mark.asyncio
async def test_unsafe_quarantine_metadata_is_bounded_and_acknowledgeable() -> None:
    manager = AgentManager()

    async def fail() -> None:
        raise RuntimeError("x" * 10_000)

    reaper_ids = []
    for index in range(130):
        task = asyncio.create_task(fail())
        reaper_ids.append(
            manager._retain_quarantined_cleanup(
                name=f"agent-{index}", agent_id=f"did:test:{index}", task=task
            )
        )
        with pytest.raises(RuntimeError):
            await task
    await asyncio.sleep(0)

    unsafe = manager._unsafe_quarantined_shutdown_failures
    assert len(unsafe) == 128
    assert manager.unsafe_quarantined_shutdown_failure_eviction_count == 2
    assert all(len(record.failure or "") <= 256 for record in unsafe.values())
    latest = reaper_ids[-1]
    assert manager.acknowledge_unsafe_quarantined_shutdown_failure(latest) is True
    assert latest not in manager.quarantined_shutdowns()
    assert manager.acknowledge_unsafe_quarantined_shutdown_failure(latest) is False


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
