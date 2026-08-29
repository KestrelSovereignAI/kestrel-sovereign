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
from fastapi import FastAPI, HTTPException
from starlette.routing import Mount, Route

from kestrel_sovereign import cli, main, server
from kestrel_sovereign.inception_service import generate_secp256k1_keypair
from kestrel_sovereign.kestrel_agent import (
    KestrelAgent,
    await_agent_shutdown_completion,
    await_lifecycle_task_completion,
)
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig
from kestrel_sovereign.multi_agent.agent_manager import (
    AgentManager,
    InflightRuntimeOffboarding,
    RuntimeOffboardingRetainedError,
)
from kestrel_sovereign.spawn.delegated_wallet import (
    BudgetAllocation,
    BudgetExceededError,
    DelegatedWallet,
)
from kestrel_sovereign.spawn.mandate import SpawnMandate


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
async def test_completion_join_propagates_grouped_process_control_after_cancellation() -> None:
    """Caller cancellation cannot turn process-control leaves into failure data."""
    owned = asyncio.get_running_loop().create_future()
    join = asyncio.create_task(await_lifecycle_task_completion(owned))
    await asyncio.sleep(0)

    join.cancel()
    await asyncio.sleep(0)
    owned.set_exception(
        BaseExceptionGroup(
            "process control",
            [asyncio.CancelledError(), GeneratorExit("stop lifecycle")],
        )
    )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(join, timeout=1.0)
    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], GeneratorExit)


@pytest.mark.asyncio
async def test_completion_join_propagates_precompleted_process_control_group() -> None:
    """An already-terminal future receives the same process-control filtering."""
    owned = asyncio.get_running_loop().create_future()
    owned.set_exception(
        BaseExceptionGroup(
            "precompleted process control",
            [asyncio.CancelledError(), GeneratorExit("stop lifecycle")],
        )
    )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await await_lifecycle_task_completion(owned)
    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], GeneratorExit)


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
async def test_lifespan_consumer_preserves_grouped_cancellation_and_failure(
    monkeypatch,
) -> None:
    """Grouped fleet cancellation remains cancellation through teardown."""

    class _GroupedManager:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        async def shutdown_all(self) -> None:
            self.shutdown_calls += 1
            raise BaseExceptionGroup(
                "cancelled fleet failure",
                [
                    asyncio.CancelledError(),
                    RuntimeError("agent cleanup failed"),
                ],
            )

    async def noop(_app) -> None:
        return None

    manager = _GroupedManager()
    app = FastAPI()
    app.state.agent_manager = manager
    app.state.startup_cleanup_agent_manager = None
    app.state.agent = None
    monkeypatch.setattr(server, "_shutdown_host_scheduler", noop)
    monkeypatch.setattr(server, "_shutdown_host_features", noop)
    monkeypatch.setattr(server, "_shutdown_phoenix", noop)

    cancelled, failure = await server._shutdown_server_resources(app)

    assert cancelled is True
    assert isinstance(failure, ExceptionGroup)
    assert len(failure.exceptions) == 1
    assert isinstance(failure.exceptions[0], RuntimeError)
    assert str(failure.exceptions[0]) == "agent cleanup failed"

    with pytest.raises(asyncio.CancelledError):
        async with server._lifespan_teardown_owner(app):
            pass
    assert manager.shutdown_calls == 2


@pytest.mark.asyncio
async def test_startup_rollback_consumer_preserves_grouped_cancellation_and_failure(
    caplog,
) -> None:
    """Startup rollback reports cancellation without dropping its failure leaf."""

    class _GroupedManager:
        async def shutdown_all(self) -> None:
            raise BaseExceptionGroup(
                "cancelled startup rollback",
                [
                    asyncio.CancelledError(),
                    OSError("rollback cleanup failed"),
                ],
            )

    cancelled = await server._rollback_startup_agent_manager(_GroupedManager())

    assert cancelled is True
    warning = next(
        record
        for record in caplog.records
        if "startup rollback did not fully shut down" in record.getMessage()
    )
    assert warning.exc_info is not None
    logged_failure = warning.exc_info[1]
    assert isinstance(logged_failure, ExceptionGroup)
    assert len(logged_failure.exceptions) == 1
    assert isinstance(logged_failure.exceptions[0], OSError)
    assert str(logged_failure.exceptions[0]) == "rollback cleanup failed"


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
    manager._parent_children = {"did:test:parent": ["terminal"]}
    manager._child_mandates = {"terminal": object()}

    removal = asyncio.create_task(manager.remove_agent("terminal"))
    await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)

    assert await asyncio.wait_for(removal, timeout=1.0) is True
    assert agent.storage_closed.is_set()
    assert manager.get_agent("terminal") is None
    assert manager._parent_children == {}
    assert manager._child_mandates == {}

    admission, owns_admission = await manager._admit_agent_operation(
        "terminal", kind="load"
    )
    assert owns_admission
    await manager._release_agent_operation(admission)


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
                name=f"agent-{index}-{'x' * 1_000}",
                agent_id=f"did:test:{index}",
                task=task,
            )
        )
        with pytest.raises(RuntimeError):
            await task
    await asyncio.sleep(0)

    unsafe = manager._unsafe_quarantined_shutdown_failures
    assert len(unsafe) == 128
    assert manager.unsafe_quarantined_shutdown_failure_eviction_count == 2
    assert all(
        len(record.reaper_id) <= 256
        and len(record.agent_name) <= 256
        and len(record.canonical_agent_name) <= 256
        and len(record.agent_id) <= 256
        and len(record.failure or "") <= 256
        for record in unsafe.values()
    )
    latest = reaper_ids[-1]
    assert manager.acknowledge_unsafe_quarantined_shutdown_failure(latest) is True
    assert latest not in manager.quarantined_shutdowns()
    assert manager.acknowledge_unsafe_quarantined_shutdown_failure(latest) is False


@pytest.mark.asyncio
async def test_unsafe_quarantine_name_stays_reserved_until_acknowledgement() -> None:
    """Evicted unsafe cleanup uses a bounded fail-closed overflow reservation."""

    manager = AgentManager()

    async def fail() -> None:
        raise RuntimeError("quarantined cleanup failed")

    # The first name is pushed out of the bounded record map.  No per-name
    # metadata survives eviction: one aggregate reservation blocks both that
    # name and unrelated reuse until the explicit aggregate acknowledgement.
    for index in range(129):
        task = asyncio.create_task(fail())
        manager._retain_quarantined_cleanup(
            name=f"Retired-{index}",
            agent_id=f"did:test:retired:{index}",
            task=task,
        )
        with pytest.raises(RuntimeError, match="quarantined cleanup failed"):
            await task
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="unresolved quarantined cleanup"):
        await manager._admit_agent_operation("retired-0", kind="test")
    with pytest.raises(RuntimeError, match="unresolved quarantined cleanup"):
        await manager._admit_agent_operation("unrelated-new-name", kind="test")

    assert manager.acknowledge_unsafe_quarantined_shutdown_failure_evictions() == 1
    assert manager._unsafe_quarantined_shutdown_failure_overflow_reserved is False
    admission, owns_admission = await manager._admit_agent_operation(
        "retired-0", kind="test"
    )
    assert owns_admission
    await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_terminal_quarantine_drain_reports_evicted_unsafe_failures() -> None:
    """Evicted unsafe outcomes cannot become implicit terminal success."""

    manager = AgentManager()

    async def fail() -> None:
        raise RuntimeError("durable cleanup remained unsafe")

    for index in range(130):
        task = asyncio.create_task(fail())
        manager._retain_quarantined_cleanup(
            name=f"unsafe-{index}",
            agent_id=f"did:test:unsafe:{index}",
            task=task,
        )
        with pytest.raises(RuntimeError, match="remained unsafe"):
            await task
    await asyncio.sleep(0)

    assert manager.unsafe_quarantined_shutdown_failure_eviction_count == 2
    for reaper_id in tuple(manager._unsafe_quarantined_shutdown_failures):
        assert manager.acknowledge_unsafe_quarantined_shutdown_failure(reaper_id)
    assert manager._unsafe_quarantined_shutdown_failures == {}

    with pytest.raises(ExceptionGroup, match="quarantined shutdown reapers") as exc_info:
        await manager.drain_quarantined_shutdowns()

    assert str(exc_info.value.exceptions[0]) == (
        "2 unsafe quarantined shutdown failure record(s) were evicted before "
        "acknowledgement"
    )
    assert (
        manager.acknowledge_unsafe_quarantined_shutdown_failure_evictions() == 2
    )
    assert manager.unsafe_quarantined_shutdown_failure_eviction_count == 0
    assert await manager.drain_quarantined_shutdowns() is False
    assert manager._quarantined_shutdown_handoffs_sealed is False


@pytest.mark.asyncio
async def test_terminal_quarantine_drain_seals_late_bounded_removal_handoffs() -> None:
    """A DELETE racing the terminal seal cannot register a reaper afterwards."""

    manager = AgentManager()
    agent = _CancellationHostileShutdownAgent()
    manager._agents["raced"] = agent
    manager._agent_names[agent.agent_id] = "raced"
    allow_drain_finish = asyncio.Event()

    async def keep_drain_open() -> None:
        await allow_drain_finish.wait()

    manager._retain_quarantined_cleanup(
        name="already-retired",
        agent_id="did:test:already-retired",
        task=asyncio.create_task(keep_drain_open()),
    )

    async def wait_for_seal() -> None:
        while not manager._quarantined_shutdown_handoffs_sealed:
            await asyncio.sleep(0)

    # Keep the drain open after it seals: while it owns that terminal
    # boundary, the removal must not hand off a reaper that this drain could
    # miss.  Once the drain ends, the manager deliberately opens again so a
    # later startup/server retry can proceed.
    drain = asyncio.create_task(manager.drain_quarantined_shutdowns())
    try:
        await asyncio.wait_for(wait_for_seal(), timeout=1.0)
        assert (
            await asyncio.wait_for(manager.remove_agent("raced"), timeout=1.0)
            is False
        )
        assert not agent.shutdown_entered.is_set()
        assert not drain.done()
    finally:
        allow_drain_finish.set()
        if not drain.done():
            assert await asyncio.wait_for(drain, timeout=1.0) is False

    assert manager._quarantined_shutdown_reapers == {}
    assert manager._quarantined_shutdown_handoffs_sealed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_kind", ("authorized", "registered"))
async def test_terminal_drain_seal_atomically_refuses_cold_identity_offboarding(
    monkeypatch,
    tmp_path,
    identity_kind,
) -> None:
    """No identity cleanup task can appear after a drain's empty seal point."""

    from kestrel_sovereign.features import isolated_runtime

    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://host/kestrel")
    manager = AgentManager(base_data_dir=tmp_path)
    did = f"did:test:sealed-{identity_kind}-offboarding"
    name = "Cold"
    if identity_kind == "authorized":
        manager._seed_scheduler_authority(
            {
                did: (
                    name,
                    LocalAgentConfig(
                        data_dir=f"agent_data/{identity_kind}",
                        port=8801,
                        autostart=False,
                    ),
                )
            }
        )
    cleanup = MagicMock(side_effect=AssertionError("post-seal cleanup started"))
    monkeypatch.setattr(
        isolated_runtime,
        "remove_isolated_runtime_namespace",
        cleanup,
    )
    seal_visible = asyncio.Event()
    allow_drain_scan = asyncio.Event()
    real_set_seal = manager._set_quarantined_shutdown_handoffs_sealed

    async def expose_empty_seal(sealed: bool) -> bool:
        cancelled = await real_set_seal(sealed)
        if sealed:
            seal_visible.set()
            await allow_drain_scan.wait()
        return cancelled

    monkeypatch.setattr(
        manager,
        "_set_quarantined_shutdown_handoffs_sealed",
        expose_empty_seal,
    )
    drain = asyncio.create_task(manager.drain_quarantined_shutdowns())
    try:
        await asyncio.wait_for(seal_visible.wait(), timeout=1.0)
        if identity_kind == "registered":
            registered_config = LocalAgentConfig(
                data_dir=f"agent_data/{identity_kind}",
                port=8801,
                autostart=False,
            )
            registered_config.resolve_data_dir(tmp_path).mkdir(parents=True)
            kwargs = {
                "known_agent_id": did,
                "known_agent_config": registered_config,
            }
        else:
            kwargs = {}
        assert await asyncio.wait_for(
            manager.remove_agent(name, offboard_runtime=True, **kwargs),
            timeout=1.0,
        ) is False
        assert manager._inflight_runtime_offboardings == {}
        cleanup.assert_not_called()
        assert did not in manager._scheduler_revoked_dids
        assert name not in manager._scheduler_revoked_names
        if identity_kind == "authorized":
            assert manager.is_scheduler_agent_authorized(did)
            assert manager.scheduler_authority_for(did)[0] == name
        else:
            assert manager.scheduler_authority_for(did) is None
        assert not drain.done()
    finally:
        allow_drain_scan.set()

    assert await asyncio.wait_for(drain, timeout=1.0) is False
    assert manager._quarantined_shutdown_handoffs_sealed is False
    assert manager._inflight_runtime_offboardings == {}
    cleanup.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_type"),
    (
        ("not_hosted", "RuntimeOffboardingNotPerformedError"),
        ("invalid", "TypeError"),
    ),
)
async def test_terminal_drain_validates_inflight_runtime_cleanup_outcome(
    monkeypatch,
    outcome,
    expected_type,
) -> None:
    """Joining an inflight worker cannot bless a non-removal or invalid return."""

    from kestrel_sovereign.features import isolated_runtime

    manager = AgentManager()
    agent = SimpleNamespace(agent_id="did:test:terminal-inflight-outcome")
    started = threading.Event()
    release = threading.Event()

    def delayed_cleanup(_agent):
        started.set()
        release.wait(timeout=5)
        if outcome == "not_hosted":
            return isolated_runtime.RuntimeNamespaceCleanupOutcome.NOT_HOSTED
        return None

    monkeypatch.setattr(
        isolated_runtime,
        "remove_agent_runtime_namespace",
        delayed_cleanup,
    )
    record = manager._start_agent_runtime_offboarding(
        name="Hosted",
        agent=agent,
    )
    try:
        assert await asyncio.to_thread(started.wait, 1)
        drain = asyncio.create_task(manager.drain_quarantined_shutdowns())
        for _ in range(100):
            if manager._quarantined_shutdown_handoffs_sealed:
                break
            await asyncio.sleep(0)
        assert manager._quarantined_shutdown_handoffs_sealed
        release.set()
        with pytest.raises(ExceptionGroup) as raised:
            await drain
    finally:
        release.set()

    assert record.task.done()
    assert any(
        type(error).__name__ == expected_type
        for error in raised.value.exceptions
    )
    assert manager._inflight_runtime_offboardings == {}
    assert manager._quarantined_shutdown_handoffs_sealed is False


@pytest.mark.asyncio
async def test_terminal_quarantine_drain_reports_precompleted_reaper_failure() -> None:
    """Unsafe metadata from a precompleted reaper remains terminal evidence."""

    manager = AgentManager()

    async def fail() -> None:
        raise RuntimeError("late durable release failed")

    task = asyncio.create_task(fail())
    reaper_id = manager._retain_quarantined_cleanup(
        name="unsafe", agent_id="did:test:unsafe", task=task
    )
    with pytest.raises(RuntimeError, match="late durable release failed"):
        await task
    await asyncio.sleep(0)

    with pytest.raises(ExceptionGroup, match="quarantined shutdown reapers") as exc_info:
        await manager.drain_quarantined_shutdowns()

    rendered = str(exc_info.value.exceptions[0])
    assert reaper_id in rendered
    assert "unacknowledged cleanup failure" in rendered
    assert "(RuntimeError)" not in rendered
    assert "late durable release failed" not in rendered
    assert reaper_id in manager._unsafe_quarantined_shutdown_failures


@pytest.mark.asyncio
async def test_quarantined_shutdown_drain_finishes_cleanup_after_cancellation() -> None:
    """A terminal manager owner records cancellation only after its reaper drains."""

    manager = AgentManager()
    cleanup_entered = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def cleanup() -> None:
        cleanup_entered.set()
        await allow_cleanup.wait()

    manager._retain_quarantined_cleanup(
        name="retired",
        agent_id="did:test:retired",
        task=asyncio.create_task(cleanup()),
    )
    drain = asyncio.create_task(manager.drain_quarantined_shutdowns())
    await asyncio.wait_for(cleanup_entered.wait(), timeout=1.0)
    drain.cancel()
    allow_cleanup.set()

    assert await asyncio.wait_for(drain, timeout=1.0) is True
    assert manager._quarantined_shutdown_reapers == {}


@pytest.mark.asyncio
async def test_quarantine_drain_preserves_cancellation_after_reaper_failure() -> None:
    """Caller cancellation and the settled failure are both preserved."""

    manager = AgentManager()
    cleanup_entered = asyncio.Event()
    allow_failure = asyncio.Event()

    async def fail_after_cancellation() -> None:
        cleanup_entered.set()
        await allow_failure.wait()
        raise RuntimeError("durable release failed after caller cancellation")

    reaper_id = manager._retain_quarantined_cleanup(
        name="unsafe",
        agent_id="did:test:unsafe",
        task=asyncio.create_task(fail_after_cancellation()),
    )
    drain = asyncio.create_task(manager.drain_quarantined_shutdowns())
    await asyncio.wait_for(cleanup_entered.wait(), timeout=1.0)
    drain.cancel()
    allow_failure.set()

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(drain, timeout=1.0)
    leaves: list[BaseException] = []

    def collect(error: BaseException) -> None:
        if isinstance(error, BaseExceptionGroup):
            for nested in error.exceptions:
                collect(nested)
        else:
            leaves.append(error)

    collect(exc_info.value)
    assert any(isinstance(error, asyncio.CancelledError) for error in leaves)
    assert any(isinstance(error, RuntimeError) for error in leaves)
    assert manager._quarantined_shutdown_reapers == {}
    assert reaper_id in manager._unsafe_quarantined_shutdown_failures


@pytest.mark.asyncio
async def test_terminal_drain_sanitizes_cancelled_worker_with_prior_failure(
    monkeypatch,
) -> None:
    """A cancellation race cannot discard failures or reflect host paths."""

    manager = AgentManager()
    private_path = "/private/tenant/runtime/credential"

    async def fail_with_private_path() -> None:
        raise OSError(private_path)

    failed_task = asyncio.create_task(fail_with_private_path())
    manager._retain_quarantined_cleanup(
        name="unsafe",
        agent_id="did:test:unsafe",
        task=failed_task,
    )
    with pytest.raises(OSError):
        await failed_task
    await asyncio.sleep(0)

    cleanup_task = asyncio.create_task(asyncio.Event().wait())
    record_key = id(cleanup_task)
    manager._inflight_runtime_offboardings[record_key] = (
        InflightRuntimeOffboarding(
            agent_name="Hosted",
            agent_id="did:test:hosted",
            runtime_path=Path(private_path),
            task=cleanup_task,
        )
    )
    cleanup_task.add_done_callback(
        lambda _task: manager._inflight_runtime_offboardings.pop(record_key, None)
    )

    from kestrel_sovereign.multi_agent import agent_manager as manager_module

    real_join = manager_module.await_lifecycle_task_completion
    cleanup_join_started = asyncio.Event()

    async def observe_join(task):
        if task is cleanup_task:
            cleanup_join_started.set()
        return await real_join(task)

    monkeypatch.setattr(
        manager_module,
        "await_lifecycle_task_completion",
        observe_join,
    )
    drain = asyncio.create_task(manager.drain_quarantined_shutdowns())
    await asyncio.wait_for(cleanup_join_started.wait(), timeout=1.0)
    cleanup_task.cancel()

    with pytest.raises(ExceptionGroup) as exc_info:
        await asyncio.wait_for(drain, timeout=1.0)

    leaves: list[BaseException] = []

    def collect(error: BaseException) -> None:
        if isinstance(error, BaseExceptionGroup):
            for nested in error.exceptions:
                collect(nested)
        else:
            leaves.append(error)

    collect(exc_info.value)
    assert len(leaves) == 2
    assert all(isinstance(error, Exception) for error in leaves)
    assert any("CancelledError" in str(error) for error in leaves)
    rendered = " | ".join(str(error) for error in leaves)
    assert private_path not in rendered
    assert "Cannot nest BaseExceptions" not in rendered
    assert manager._inflight_runtime_offboardings == {}


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
async def test_manager_shutdown_all_retries_retained_agent_and_hold_after_failure() -> None:
    """A failed fleet drain reopens the manager for its next shutdown attempt."""

    class FailOnceShutdownAgent:
        agent_id = "did:test:fail-once-shutdown"

        def __init__(self) -> None:
            self.shutdown_calls = 0

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            if self.shutdown_calls == 1:
                raise RuntimeError("shutdown failed once")

    manager = AgentManager()
    agent = FailOnceShutdownAgent()
    manager._agents["retry"] = agent
    manager._agent_names[agent.agent_id] = "retry"
    manager._parent_children = {"did:test:parent": ["retry"]}
    manager._child_mandates = {"retry": object()}
    manager._child_budgets["retry"] = (object(), object())
    released: list[str] = []

    async def release_child_budget(name: str) -> bool:
        released.append(name)
        manager._child_budgets.pop(name, None)
        return False

    manager._release_child_budget_cancellation_safe = release_child_budget

    with pytest.raises(ExceptionGroup, match="fleet agents failed"):
        await manager.shutdown_all()

    assert agent.shutdown_calls == 1
    assert manager.get_agent("retry") is agent
    assert "retry" in manager._child_budgets
    assert manager._quarantined_shutdown_handoffs_sealed is False

    await manager.shutdown_all()

    assert agent.shutdown_calls == 2
    assert released == ["retry"]
    assert manager.get_agent("retry") is None
    assert "retry" not in manager._child_budgets
    assert manager.get_children("did:test:parent") == []
    assert manager.get_mandate("retry") is None


@pytest.mark.asyncio
async def test_manager_shutdown_all_drains_quarantined_reaper_before_returning(
    monkeypatch,
) -> None:
    """Fleet teardown cannot return while a bounded removal still owns cleanup."""

    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )

    class ObservableHostileShutdownAgent(_CancellationHostileShutdownAgent):
        def __init__(self) -> None:
            super().__init__()
            self.reaper_handed_off = asyncio.Event()

        def handoff_shutdown_to_reaper(self, shutdown_task):
            reaper = super().handoff_shutdown_to_reaper(shutdown_task)
            self.reaper_handed_off.set()
            return reaper

    manager = AgentManager()
    agent = ObservableHostileShutdownAgent()
    manager._agents["hostile"] = agent
    manager._agent_names[agent.agent_id] = "hostile"
    manager._offboard_agent_runtime_namespace = AsyncMock(
        return_value=(False, None)
    )
    removal_completed = asyncio.Event()

    async def release_budget(name: str) -> None:
        assert name == "hostile"
        assert manager.get_agent("hostile") is None
        removal_completed.set()

    manager._release_child_budget = release_budget

    shutdown = asyncio.create_task(manager.shutdown_all())
    try:
        await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)
        await asyncio.wait_for(agent.reaper_handed_off.wait(), timeout=1.0)
        await asyncio.wait_for(removal_completed.wait(), timeout=1.0)

        # The per-agent control plane still unpublishes promptly, but the
        # terminal manager owner remains live until its retained reaper can
        # finish.
        assert manager.get_agent("hostile") is None
        assert not shutdown.done()

        agent.allow_shutdown_finish.set()
        await asyncio.wait_for(shutdown, timeout=1.0)
    finally:
        agent.allow_shutdown_finish.set()
        if not shutdown.done():
            await asyncio.wait_for(shutdown, timeout=1.0)

    assert all(
        not item["pending"] for item in manager.quarantined_shutdowns().values()
    )
    manager._offboard_agent_runtime_namespace.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_offboarding_intent_survives_quarantined_reaper_handoff(
    monkeypatch,
) -> None:
    """A bounded deprovision carries deletion intent into its retained reaper."""

    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )

    class ObservableHostileShutdownAgent(_CancellationHostileShutdownAgent):
        def __init__(self) -> None:
            super().__init__()
            self.reaper_handed_off = asyncio.Event()

        def handoff_shutdown_to_reaper(self, shutdown_task):
            reaper = super().handoff_shutdown_to_reaper(shutdown_task)
            self.reaper_handed_off.set()
            return reaper

    manager = AgentManager()
    agent = ObservableHostileShutdownAgent()
    manager._agents["hostile"] = agent
    manager._agent_names[agent.agent_id] = "hostile"
    manager._offboard_agent_runtime_namespace = AsyncMock(
        return_value=(False, None)
    )

    removal = asyncio.create_task(
        manager.remove_agent("hostile", offboard_runtime=True)
    )
    try:
        await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)
        await asyncio.wait_for(agent.reaper_handed_off.wait(), timeout=1.0)
        with pytest.raises(RuntimeOffboardingRetainedError) as raised:
            await asyncio.wait_for(removal, timeout=1.0)
        assert raised.value.metadata["agent_removed"] is True
        assert raised.value.metadata["runtime_cleanup_pending"] is True
        assert raised.value.metadata["runtime_cleanup_state"] == "pending"
        assert manager.get_agent("hostile") is None
        manager._offboard_agent_runtime_namespace.assert_not_awaited()

        agent.allow_shutdown_finish.set()
        assert await manager.drain_quarantined_shutdowns() is False
    finally:
        agent.allow_shutdown_finish.set()
        if not removal.done():
            await asyncio.wait_for(removal, timeout=1.0)

    manager._offboard_agent_runtime_namespace.assert_awaited_once_with(agent)


@pytest.mark.asyncio
async def test_handed_off_offboarding_failure_is_owned_once_and_remains_visible(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )

    class ObservableHostileShutdownAgent(_CancellationHostileShutdownAgent):
        def __init__(self) -> None:
            super().__init__()
            self.reaper_handed_off = asyncio.Event()

        def handoff_shutdown_to_reaper(self, shutdown_task):
            reaper = super().handoff_shutdown_to_reaper(shutdown_task)
            self.reaper_handed_off.set()
            return reaper

    manager = AgentManager()
    agent = ObservableHostileShutdownAgent()
    manager._agents["hostile"] = agent
    manager._agent_names[agent.agent_id] = "hostile"
    retained_failure = OSError("private retained runtime path")
    manager._offboard_agent_runtime_namespace = AsyncMock(
        return_value=(False, retained_failure)
    )

    removal = asyncio.create_task(
        manager.remove_agent("hostile", offboard_runtime=True)
    )
    try:
        await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)
        await asyncio.wait_for(agent.reaper_handed_off.wait(), timeout=1.0)
        with pytest.raises(RuntimeOffboardingRetainedError) as pending:
            await asyncio.wait_for(removal, timeout=1.0)
        assert pending.value.metadata["runtime_cleanup_state"] == "pending"
        assert manager._inflight_runtime_offboardings == {}

        agent.allow_shutdown_finish.set()
        with pytest.raises(ExceptionGroup, match="quarantined shutdown reapers"):
            await manager.drain_quarantined_shutdowns()
    finally:
        agent.allow_shutdown_finish.set()
        if not removal.done():
            with pytest.raises(RuntimeOffboardingRetainedError):
                await asyncio.wait_for(removal, timeout=1.0)

    manager._offboard_agent_runtime_namespace.assert_awaited_once_with(agent)
    assert manager.get_agent("hostile") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("removed", "not_hosted"))
async def test_quarantined_offboard_preserves_cancellation_on_every_custody_outcome(
    monkeypatch,
    outcome,
) -> None:
    """A successful/no-op worker cannot erase cancellation of its joiner."""

    from kestrel_sovereign.features import isolated_runtime
    from kestrel_sovereign.multi_agent.agent_manager import (
        RuntimeOffboardingNotPerformedError,
    )

    manager = AgentManager()
    agent = SimpleNamespace(agent_id=f"did:test:cancelled-{outcome}")
    manager._agent_names[agent.agent_id] = "Hosted"
    entered = threading.Event()
    release = threading.Event()

    def finish_after_cancellation(_agent):
        entered.set()
        release.wait(timeout=5)
        return (
            isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED
            if outcome == "removed"
            else isolated_runtime.RuntimeNamespaceCleanupOutcome.NOT_HOSTED
        )

    monkeypatch.setattr(
        isolated_runtime,
        "remove_agent_runtime_namespace",
        finish_after_cancellation,
    )
    operation = asyncio.create_task(
        manager._offboard_agent_runtime_namespace(agent)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        operation.cancel()
        await asyncio.sleep(0)
        release.set()
        cancelled, failure = await operation
    finally:
        release.set()

    assert cancelled is True
    if outcome == "removed":
        assert failure is None
    else:
        assert isinstance(failure, RuntimeOffboardingNotPerformedError)
        assert failure.cleanup_state == "not_hosted"


@pytest.mark.asyncio
async def test_cancelled_successful_quarantined_offboard_is_terminally_accounted(
    monkeypatch,
) -> None:
    """Terminal drain sees cancellation even when the deletion itself succeeds."""

    from kestrel_sovereign.features import isolated_runtime

    manager = AgentManager()
    agent = SimpleNamespace(agent_id="did:test:cancelled-successful-offboard")
    entered = threading.Event()
    release = threading.Event()

    def finish_after_cancellation(_agent):
        entered.set()
        release.wait(timeout=5)
        return isolated_runtime.RuntimeNamespaceCleanupOutcome.REMOVED

    monkeypatch.setattr(
        isolated_runtime,
        "remove_agent_runtime_namespace",
        finish_after_cancellation,
    )

    async def retained_owner() -> None:
        cancelled, failure = await manager._offboard_agent_runtime_namespace(agent)
        if failure is not None:
            raise failure
        if cancelled:
            raise asyncio.CancelledError()

    owner = asyncio.create_task(retained_owner())
    manager._retain_quarantined_cleanup(
        name="Hosted",
        agent_id=agent.agent_id,
        task=owner,
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        owner.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(ExceptionGroup, match="quarantined shutdown reapers"):
            await manager.drain_quarantined_shutdowns()
    finally:
        release.set()

    retained = manager.quarantined_shutdowns()
    assert len(retained) == 1
    assert next(iter(retained.values()))["failure"] == (
        "shutdown reaper was cancelled"
    )


@pytest.mark.asyncio
async def test_delete_endpoint_maps_real_shutdown_handoff_to_pending_custody(
    monkeypatch,
    tmp_path,
) -> None:
    from kestrel_sovereign.endpoints.models import delete_agent

    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.SHUTDOWN_TIMEOUT", 0.01
    )
    manager = AgentManager(base_data_dir=tmp_path)
    agent = _CancellationHostileShutdownAgent()
    manager._agents["hostile"] = agent
    manager._agent_names[agent.agent_id] = "hostile"
    manager._offboard_agent_runtime_namespace = AsyncMock(return_value=(False, None))
    config_path = tmp_path / "multi_agent.toml"
    config = MultiAgentConfig(
        agents={
            "hostile": LocalAgentConfig(
                data_dir="agent_data/hostile",
                port=8801,
            )
        }
    )
    config.save(config_path)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                agent_manager=manager,
                multi_agent_config_path=config_path,
                multi_agent_config=config,
            )
        )
    )

    try:
        with pytest.raises(HTTPException) as raised:
            await delete_agent.__wrapped__(
                request,
                "hostile",
                offboard_runtime=True,
            )

        assert raised.value.status_code == 409
        assert raised.value.detail["agent_removed"] is True
        assert raised.value.detail["runtime_cleanup_state"] == "pending"
        assert manager.get_agent("hostile") is None
        assert MultiAgentConfig.from_file(config_path).agents == {}
        manager._offboard_agent_runtime_namespace.assert_not_awaited()

        agent.allow_shutdown_finish.set()
        assert await manager.drain_quarantined_shutdowns() is False
    finally:
        agent.allow_shutdown_finish.set()

    manager._offboard_agent_runtime_namespace.assert_awaited_once_with(agent)


@pytest.mark.asyncio
async def test_manager_shutdown_all_drains_ordinary_release_admitted_before_boundary() -> None:
    """Terminal teardown joins a normal post-unpublish budget release too."""

    class SuccessfulShutdownAgent:
        agent_id = "did:test:ordinary-release"

        async def shutdown(self) -> None:
            return None

    manager = AgentManager()
    agent = SuccessfulShutdownAgent()
    manager._agents["ordinary"] = agent
    manager._agent_names[agent.agent_id] = "ordinary"
    budget_entry = (object(), object())
    manager._child_budgets["ordinary"] = budget_entry
    release_entered = asyncio.Event()
    allow_release = asyncio.Event()

    async def release_budget(name: str) -> None:
        assert name == "ordinary"
        assert manager.get_agent(name) is None
        assert manager._child_budgets.pop(name) is budget_entry
        release_entered.set()
        await allow_release.wait()

    manager._release_child_budget = release_budget
    removal = asyncio.create_task(manager.remove_agent("ordinary"))
    shutdown = None
    try:
        await asyncio.wait_for(release_entered.wait(), timeout=1.0)
        assert manager.get_agent("ordinary") is None
        assert "ordinary" not in manager._child_budgets
        assert manager._quarantined_shutdown_reapers == {}

        # The old drain observed the now-empty routing, hold, and quarantine
        # maps and returned here. The admitted ordinary release keeps this
        # terminal boundary live until its exact task settles.
        shutdown = asyncio.create_task(manager.shutdown_all())
        for _ in range(100):
            if manager._quarantined_shutdown_handoffs_sealed or shutdown.done():
                break
            await asyncio.sleep(0)
        assert manager._quarantined_shutdown_handoffs_sealed is True
        assert not shutdown.done()
    finally:
        allow_release.set()

    assert await asyncio.wait_for(removal, timeout=1.0) is True
    assert shutdown is not None
    await asyncio.wait_for(shutdown, timeout=1.0)
    assert manager._inflight_removal_budget_releases == {}


@pytest.mark.asyncio
async def test_shutdown_all_coalesces_release_already_admitted_by_concurrent_remove() -> None:
    """A fleet drain joins DELETE's pending refund instead of crediting it twice."""

    class SuccessfulShutdownAgent:
        agent_id = "did:test:coalesced-ordinary-release"

        async def shutdown(self) -> None:
            return None

    manager = AgentManager()
    agent = SuccessfulShutdownAgent()
    budget_entry = (object(), object())
    manager._agents["ordinary"] = agent
    manager._agent_names[agent.agent_id] = "ordinary"
    manager._child_budgets["ordinary"] = budget_entry
    release_entered = asyncio.Event()
    allow_release = asyncio.Event()
    calls = 0
    credits = 0

    async def release_budget(name: str) -> bool:
        nonlocal calls, credits
        assert name == "ordinary"
        calls += 1
        release_entered.set()
        await allow_release.wait()
        assert manager._child_budgets.pop(name) is budget_entry
        credits += 1
        return False

    manager._release_child_budget_cancellation_safe = release_budget
    removal = asyncio.create_task(manager.remove_agent("ordinary"))
    shutdown = None
    try:
        await asyncio.wait_for(release_entered.wait(), timeout=1.0)
        assert manager.get_agent("ordinary") is None
        assert "ordinary" in manager._child_budgets

        shutdown = asyncio.create_task(manager.shutdown_all())
        await asyncio.sleep(0)

        # The concurrent fleet owner must receive R1's task, not admit R2
        # while the supported release override still owns the child hold.
        assert calls == 1
        assert len(manager._inflight_removal_budget_releases) == 1
        assert credits == 0
    finally:
        allow_release.set()

    assert await asyncio.wait_for(removal, timeout=1.0) is True
    assert shutdown is not None
    await asyncio.wait_for(shutdown, timeout=1.0)
    assert calls == 1
    assert credits == 1
    assert manager._inflight_removal_budget_releases == {}
    assert manager._inflight_removal_budget_releases_by_child == {}


@pytest.mark.asyncio
async def test_terminal_drain_retains_prelinearization_ordinary_release_failure() -> None:
    """A completed ordinary refund failure remains terminal evidence until acked."""

    manager = AgentManager()
    release_failed = asyncio.Event()

    async def fail_release(name: str) -> bool:
        assert name == "ordinary"
        release_failed.set()
        raise RuntimeError("ordinary refund failed before drain linearization")

    manager._release_child_budget_cancellation_safe = fail_release
    async with manager._lock:
        manager._start_child_budget_release("ordinary")
        drain = asyncio.create_task(manager.drain_quarantined_shutdowns())

        # Keep the manager lock through both task completion and its done
        # callback. The drain is blocked before its sealing linearization
        # point, so this reproduces the former task-discard window exactly.
        await asyncio.wait_for(release_failed.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert not drain.done()
        assert manager._inflight_removal_budget_releases == {}

    with pytest.raises(ExceptionGroup, match="ordinary budget releases") as first:
        await asyncio.wait_for(drain, timeout=1.0)
    rendered_failure = str(first.value.exceptions[0])
    assert "ordinary-budget-release:1" in rendered_failure
    assert "unacknowledged cleanup failure" in rendered_failure
    assert "(RuntimeError)" not in rendered_failure
    assert "ordinary refund failed before drain linearization" not in (
        rendered_failure
    )

    failures = manager.unsafe_removal_budget_release_failures()
    assert len(failures) == 1
    release_id = next(iter(failures))
    assert failures[release_id]["failure"] == (
        "RuntimeError: ordinary refund failed before drain linearization"
    )

    # An earlier drain never acknowledges an unsafe outcome. A second terminal
    # drain must fail too, rather than reporting a false clean shutdown.
    with pytest.raises(ExceptionGroup, match="ordinary budget releases"):
        await manager.drain_quarantined_shutdowns()

    assert manager.acknowledge_unsafe_removal_budget_release_failure(release_id)
    assert await manager.drain_quarantined_shutdowns() is False


@pytest.mark.asyncio
async def test_shutdown_all_reports_failed_ordinary_release_once() -> None:
    """The immediate attempt and its retained evidence do not duplicate one refund."""

    class SuccessfulShutdownAgent:
        agent_id = "did:test:failed-ordinary-release"

        async def shutdown(self) -> None:
            return None

    manager = AgentManager()
    agent = SuccessfulShutdownAgent()
    budget_entry = (object(), object())
    manager._agents["ordinary"] = agent
    manager._agent_names[agent.agent_id] = "ordinary"
    manager._child_budgets["ordinary"] = budget_entry

    async def fail_release(name: str) -> bool:
        assert name == "ordinary"
        assert manager._child_budgets.pop(name) is budget_entry
        raise RuntimeError("ordinary refund failed")

    manager._release_child_budget_cancellation_safe = fail_release

    with pytest.raises(ExceptionGroup) as exc_info:
        await manager.shutdown_all()

    def flatten(error: BaseException) -> list[BaseException]:
        if isinstance(error, BaseExceptionGroup):
            return [item for nested in error.exceptions for item in flatten(nested)]
        return [error]

    failures = flatten(exc_info.value)
    assert [str(failure) for failure in failures] == ["ordinary refund failed"]

    # The failed task's bounded evidence remains unsafe for a separate later
    # terminal drain, even though this one call reports it only once.
    retained = manager.unsafe_removal_budget_release_failures()
    assert len(retained) == 1
    with pytest.raises(ExceptionGroup, match="ordinary budget releases"):
        await manager.drain_quarantined_shutdowns()


@pytest.mark.asyncio
async def test_shutdown_all_reports_grouped_ordinary_release_failure_once() -> None:
    """A grouped refund outcome is not repeated by retained drain evidence."""

    manager = AgentManager()
    budget_entry = (object(), object())
    manager._child_budgets["ordinary"] = budget_entry

    async def fail_release(name: str) -> bool:
        assert manager._child_budgets.pop(name) is budget_entry
        raise BaseExceptionGroup(
            "grouped refund",
            [asyncio.CancelledError(), RuntimeError("grouped ordinary refund failed")],
        )

    manager._release_child_budget_cancellation_safe = fail_release

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await manager.shutdown_all()

    def flatten(error: BaseException) -> list[BaseException]:
        if isinstance(error, BaseExceptionGroup):
            return [item for nested in error.exceptions for item in flatten(nested)]
        return [error]

    failures = flatten(exc_info.value)
    assert sum(
        str(failure) == "grouped ordinary refund failed" for failure in failures
    ) == 1


@pytest.mark.asyncio
async def test_retained_child_tracking_reserves_name_until_drain_reconciliation() -> None:
    """A completed reaper cannot expose its old child name before pruning."""

    manager = AgentManager()
    manager._parent_children = {"did:test:parent": ["Retained"]}
    manager._child_mandates = {"Retained": object()}

    with pytest.raises(RuntimeError, match="retained child lifecycle tracking"):
        await manager._admit_agent_operation("retained", kind="load")

    await manager._prune_all_fully_removed_child_tracking()
    admission, owns_admission = await manager._admit_agent_operation(
        "retained", kind="load"
    )
    assert owns_admission
    await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_direct_child_removal_prunes_tracking_before_name_admission_reopens() -> None:
    """A completed direct DELETE cannot permanently reserve its old child name."""

    class SuccessfulShutdownAgent:
        agent_id = "did:test:direct-child-removal"

        async def shutdown(self) -> None:
            return None

    manager = AgentManager()
    agent = SuccessfulShutdownAgent()
    manager._agents["Retained"] = agent
    manager._agent_names[agent.agent_id] = "Retained"
    manager._parent_children = {"did:test:parent": ["Retained"]}
    manager._child_mandates = {"Retained": object()}

    assert await manager.remove_agent("Retained") is True
    assert manager._parent_children == {}
    assert manager._child_mandates == {}

    admission, owns_admission = await manager._admit_agent_operation(
        "retained", kind="load"
    )
    assert owns_admission
    await manager._release_agent_operation(admission)


@pytest.mark.asyncio
async def test_default_budget_release_failure_remains_terminal_evidence(
    monkeypatch,
) -> None:
    """A failed production refund retains its hold for a safe later retry."""

    class SuccessfulShutdownAgent:
        agent_id = "did:test:default-release-failure"

        async def shutdown(self) -> None:
            return None

    manager = AgentManager()
    agent = SuccessfulShutdownAgent()
    budget_entry = (object(), object())
    manager._agents["ordinary"] = agent
    manager._agent_names[agent.agent_id] = "ordinary"
    manager._child_budgets["ordinary"] = budget_entry
    release_calls: list[tuple[object, object]] = []

    async def fail_release(delegated, parent_wallet) -> Decimal:
        release_calls.append((delegated, parent_wallet))
        if len(release_calls) == 1:
            raise RuntimeError("provider refund failed")
        return Decimal("0")

    # Patch the function imported by AgentManager, but retain its default
    # release methods.  This guards the production path rather than an
    # override that already propagates its own failure.
    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.release_delegated_wallet",
        fail_release,
    )

    with pytest.raises(RuntimeError, match="provider refund failed"):
        await manager.remove_agent("ordinary")

    assert release_calls == [budget_entry]
    # The provider failed before confirming this attempt. The manager retains
    # the exact allocation reference, allowing a durable provider's idempotent
    # retry (or a legacy provider's explicit uncertainty refusal) instead of
    # silently losing the only hold ownership record.
    assert manager._child_budgets["ordinary"] is budget_entry
    failures = manager.unsafe_removal_budget_release_failures()
    assert len(failures) == 1
    failure_id = next(iter(failures))
    assert failures[failure_id]["failure"] == (
        "RuntimeError: provider refund failed"
    )

    assert await manager.remove_agent("ordinary") is True
    assert release_calls == [budget_entry, budget_entry]
    assert manager._child_budgets == {}

    # Retained evidence remains terminal until the operator acknowledges the
    # prior ambiguous failure, even though the later retry completed.
    with pytest.raises(ExceptionGroup, match="ordinary budget releases"):
        await manager.drain_quarantined_shutdowns()
    assert manager.acknowledge_unsafe_removal_budget_release_failure(failure_id)
    assert await manager.drain_quarantined_shutdowns() is False


@pytest.mark.asyncio
async def test_shutdown_all_fences_late_registration_after_empty_fleet_shutdown() -> None:
    """An initializer begun before shutdown cannot publish after it succeeds."""

    class LateAgent:
        agent_id = "did:test:late-registration"

        def __init__(self) -> None:
            self.shutdown_calls = 0

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    manager = AgentManager()
    agent = LateAgent()
    initialize_entered = asyncio.Event()
    allow_initialization = asyncio.Event()

    async def initialize(*_args, **_kwargs):
        initialize_entered.set()
        await allow_initialization.wait()
        return agent

    manager._initialize_agent = initialize
    config = LocalAgentConfig(data_dir="late", port=8801)
    registration = asyncio.create_task(manager.load_agent("late", config))
    await asyncio.wait_for(initialize_entered.wait(), timeout=1.0)

    # Schedule terminal teardown while the fleet is still empty, immediately
    # before the pending initializer reaches its publication critical section.
    await asyncio.wait_for(manager.shutdown_all(), timeout=1.0)
    allow_initialization.set()

    with pytest.raises(RuntimeError, match="manager is shutting down"):
        await asyncio.wait_for(registration, timeout=1.0)
    assert manager.get_agent("late") is None
    assert agent.shutdown_calls == 1


@pytest.mark.asyncio
async def test_shutdown_all_waits_for_a_concurrent_terminal_drain_before_removal() -> None:
    """A pre-existing terminal drain cannot make a live agent look unremovable."""

    class SuccessfulShutdownAgent:
        agent_id = "did:test:drain-before-shutdown-all"

        def __init__(self) -> None:
            self.shutdown_calls = 0

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    manager = AgentManager()
    agent = SuccessfulShutdownAgent()
    manager._agents["live"] = agent
    manager._agent_names[agent.agent_id] = "live"
    allow_drain_finish = asyncio.Event()

    async def keep_drain_open() -> None:
        await allow_drain_finish.wait()

    manager._retain_quarantined_cleanup(
        name="already-retired",
        agent_id="did:test:already-retired",
        task=asyncio.create_task(keep_drain_open()),
    )
    drain = asyncio.create_task(manager.drain_quarantined_shutdowns())
    while not manager._quarantined_shutdown_handoffs_sealed:
        await asyncio.sleep(0)

    shutdown = asyncio.create_task(manager.shutdown_all())
    await asyncio.sleep(0)
    assert agent.shutdown_calls == 0
    assert not shutdown.done()

    allow_drain_finish.set()
    assert await asyncio.wait_for(drain, timeout=1.0) is False
    assert await asyncio.wait_for(shutdown, timeout=1.0) is None
    assert agent.shutdown_calls == 1
    assert manager.get_agent("live") is None


@pytest.mark.asyncio
async def test_simultaneous_shutdown_all_calls_serialize_live_agent_sweeps() -> None:
    """A second fleet owner takes a post-sweep snapshot instead of racing it."""

    class BlockingShutdownAgent:
        def __init__(self, agent_id: str, block: bool = False) -> None:
            self.agent_id = agent_id
            self.block = block
            self.shutdown_calls = 0
            self.shutdown_entered = asyncio.Event()
            self.allow_shutdown = asyncio.Event()

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            self.shutdown_entered.set()
            if self.block:
                await self.allow_shutdown.wait()

    manager = AgentManager()
    first_agent = BlockingShutdownAgent("did:test:first", block=True)
    second_agent = BlockingShutdownAgent("did:test:second")
    manager._agents = {"first": first_agent, "second": second_agent}
    manager._agent_names = {
        first_agent.agent_id: "first",
        second_agent.agent_id: "second",
    }

    first = asyncio.create_task(manager.shutdown_all())
    await asyncio.wait_for(first_agent.shutdown_entered.wait(), timeout=1.0)
    second = asyncio.create_task(manager.shutdown_all())
    await asyncio.sleep(0)
    assert not second.done()

    first_agent.allow_shutdown.set()
    assert await asyncio.wait_for(first, timeout=1.0) is None
    assert await asyncio.wait_for(second, timeout=1.0) is None
    assert first_agent.shutdown_calls == 1
    assert second_agent.shutdown_calls == 1
    assert manager.list_agents() == {}


@pytest.mark.asyncio
async def test_shutdown_all_accepts_false_after_concurrent_removal_fully_removed_target() -> None:
    """A sweep must not fail when a blocked DELETE finished its B target first."""

    class SuccessfulShutdownAgent:
        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            self.shutdown_calls = 0

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    manager = AgentManager()
    first = SuccessfulShutdownAgent("did:test:concurrent-remove-first")
    second = SuccessfulShutdownAgent("did:test:concurrent-remove-second")
    manager._agents = {"A": first, "B": second}
    manager._agent_names = {first.agent_id: "A", second.agent_id: "B"}
    second_lifecycle_lock = manager.scheduler_lifecycle_lock(second.agent_id)
    await second_lifecycle_lock.acquire()

    original_remove_agent = manager.remove_agent
    shutdown_waiting_to_remove_b = asyncio.Event()
    allow_shutdown_to_remove_b = asyncio.Event()

    async def gated_remove_agent(
        name: str,
        *,
        offboard_runtime: bool = False,
    ) -> bool:
        current = asyncio.current_task()
        if name == "B" and current is not None and current.get_name() == "fleet-shutdown":
            shutdown_waiting_to_remove_b.set()
            await allow_shutdown_to_remove_b.wait()
        return await original_remove_agent(
            name,
            offboard_runtime=offboard_runtime,
        )

    manager.remove_agent = gated_remove_agent
    direct_removal = asyncio.create_task(
        manager.remove_agent("B"), name="direct-remove-b"
    )
    fleet_shutdown = asyncio.create_task(
        manager.shutdown_all(), name="fleet-shutdown"
    )
    try:
        await asyncio.wait_for(shutdown_waiting_to_remove_b.wait(), timeout=1.0)
        assert first.shutdown_calls == 1
        assert second.shutdown_calls == 0

        # B's DELETE queued first on the per-DID writer and now fully removes
        # B before this sweep's snapped B attempt is allowed to run.
        second_lifecycle_lock.release()
        assert await asyncio.wait_for(direct_removal, timeout=1.0) is True
        assert second.shutdown_calls == 1
        assert manager.get_agent("B") is None

        allow_shutdown_to_remove_b.set()
        assert await asyncio.wait_for(fleet_shutdown, timeout=1.0) is None
    finally:
        allow_shutdown_to_remove_b.set()
        if second_lifecycle_lock.locked():
            second_lifecycle_lock.release()
        if not direct_removal.done():
            await asyncio.wait_for(direct_removal, timeout=1.0)
        if not fleet_shutdown.done():
            await asyncio.wait_for(fleet_shutdown, timeout=1.0)

    assert first.shutdown_calls == 1
    assert second.shutdown_calls == 1
    assert manager.list_agents() == {}
    assert manager._child_budgets == {}


@pytest.mark.asyncio
async def test_completed_failed_release_override_retires_before_immediate_retry() -> None:
    """A done override Future cannot coalesce an immediate retry to stale work."""

    class SuccessfulShutdownAgent:
        agent_id = "did:test:done-future-release-retry"

        async def shutdown(self) -> None:
            return None

    manager = AgentManager()
    agent = SuccessfulShutdownAgent()
    budget_entry = (object(), object())
    manager._agents["ordinary"] = agent
    manager._agent_names[agent.agent_id] = "ordinary"
    manager._child_budgets["ordinary"] = budget_entry
    failed_release = asyncio.get_running_loop().create_future()
    failed_release.set_exception(RuntimeError("override refund failed"))
    release_attempts: list[str] = []
    credits = 0

    def release_override(name: str):
        nonlocal credits
        release_attempts.append(name)
        if len(release_attempts) == 1:
            return failed_release

        async def retry() -> bool:
            nonlocal credits
            assert manager._child_budgets.pop(name) is budget_entry
            credits += 1
            return False

        return retry()

    manager._release_child_budget_cancellation_safe = release_override

    with pytest.raises(RuntimeError, match="override refund failed"):
        await manager.remove_agent("ordinary")

    # No event-loop yield here: the next removal must not receive the failed
    # Future that the override supplied above.
    assert manager._inflight_removal_budget_releases == {}
    assert manager._inflight_removal_budget_releases_by_child == {}
    failures = manager.unsafe_removal_budget_release_failures()
    assert len(failures) == 1
    failure_id = next(iter(failures))

    assert await manager.remove_agent("ordinary") is True
    assert release_attempts == ["ordinary", "ordinary"]
    assert credits == 1
    assert manager._child_budgets == {}
    assert manager.acknowledge_unsafe_removal_budget_release_failure(failure_id)
    assert await manager.drain_quarantined_shutdowns() is False


@pytest.mark.asyncio
async def test_synchronous_release_override_failure_retains_acknowledgeable_evidence() -> None:
    """A legacy override cannot raise after popping a hold without a record."""

    class SuccessfulShutdownAgent:
        agent_id = "did:test:synchronous-release-override-failure"

        async def shutdown(self) -> None:
            return None

    manager = AgentManager()
    agent = SuccessfulShutdownAgent()
    budget_entry = (object(), object())
    manager._agents["ordinary"] = agent
    manager._agent_names[agent.agent_id] = "ordinary"
    manager._child_budgets["ordinary"] = budget_entry

    def release_override(name: str):
        assert name == "ordinary"
        assert manager._child_budgets.pop(name) is budget_entry
        raise RuntimeError("synchronous override failed after taking hold")

    manager._release_child_budget_cancellation_safe = release_override

    with pytest.raises(RuntimeError, match="synchronous override failed"):
        await manager.remove_agent("ordinary")

    assert manager.get_agent("ordinary") is None
    assert manager._child_budgets == {}
    assert manager._inflight_removal_budget_releases == {}
    assert manager._inflight_removal_budget_releases_by_child == {}
    failures = manager.unsafe_removal_budget_release_failures()
    assert len(failures) == 1
    release_id = next(iter(failures))
    assert failures[release_id]["failure"] == (
        "RuntimeError: synchronous override failed after taking hold"
    )

    with pytest.raises(ExceptionGroup, match="ordinary budget releases"):
        await manager.drain_quarantined_shutdowns()
    assert manager.acknowledge_unsafe_removal_budget_release_failure(release_id)
    assert await manager.drain_quarantined_shutdowns() is False


@pytest.mark.asyncio
async def test_terminal_drain_acknowledges_evicted_ordinary_release_failures() -> None:
    """Acknowledging records and aggregate evictions reopens later drains."""

    manager = AgentManager()

    async def fail_release(name: str) -> bool:
        raise RuntimeError(f"ordinary refund {name} remained unsafe")

    manager._release_child_budget_cancellation_safe = fail_release
    for index in range(130):
        async with manager._lock:
            release = manager._start_child_budget_release(f"ordinary-{index}")
        with pytest.raises(RuntimeError, match="remained unsafe"):
            await release.task
        await asyncio.sleep(0)

    assert manager.unsafe_removal_budget_release_failure_eviction_count == 2
    for release_id in tuple(manager._unsafe_removal_budget_release_failures):
        assert manager.acknowledge_unsafe_removal_budget_release_failure(release_id)
    assert manager._unsafe_removal_budget_release_failures == {}

    with pytest.raises(ExceptionGroup, match="ordinary budget releases") as exc_info:
        await manager.drain_quarantined_shutdowns()

    assert str(exc_info.value.exceptions[0]) == (
        "2 unsafe ordinary budget release failure record(s) were evicted before "
        "acknowledgement"
    )
    assert manager.acknowledge_unsafe_removal_budget_release_failure_evictions() == 2
    assert manager.unsafe_removal_budget_release_failure_eviction_count == 0
    assert await manager.drain_quarantined_shutdowns() is False


@pytest.mark.asyncio
async def test_remove_agent_propagates_completed_release_cancellation() -> None:
    """A release override's completed ``True`` cannot become DELETE success."""

    class SuccessfulShutdownAgent:
        agent_id = "did:test:release-cancelled"

        async def shutdown(self) -> None:
            return None

    manager = AgentManager()
    agent = SuccessfulShutdownAgent()
    budget_entry = (object(), object())
    manager._agents["ordinary"] = agent
    manager._agent_names[agent.agent_id] = "ordinary"
    manager._child_budgets["ordinary"] = budget_entry

    async def release_after_cleanup(name: str) -> bool:
        assert name == "ordinary"
        assert manager._child_budgets.pop(name) is budget_entry
        return True

    manager._release_child_budget_cancellation_safe = release_after_cleanup

    with pytest.raises(asyncio.CancelledError):
        await manager.remove_agent("ordinary")

    assert manager.get_agent("ordinary") is None
    assert manager._child_budgets == {}


@pytest.mark.asyncio
async def test_shutdown_all_propagates_unpublished_hold_release_cancellation() -> None:
    """A legacy release override's completed ``True`` remains cancellation."""

    manager = AgentManager()
    budget_entry = (object(), object())
    manager._child_budgets["unpublished"] = budget_entry
    release_calls: list[str] = []

    async def release_after_cleanup(name: str) -> bool:
        release_calls.append(name)
        assert manager._child_budgets.pop(name) is budget_entry
        # The pre-wrapper override contract: cleanup completed, but the
        # release owner observed caller cancellation and expects propagation.
        return True

    manager._release_child_budget_cancellation_safe = release_after_cleanup

    with pytest.raises(asyncio.CancelledError):
        await manager.shutdown_all()

    assert release_calls == ["unpublished"]
    assert manager._child_budgets == {}


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

    async def return_false(
        name: str,
        *,
        offboard_runtime: bool = False,
    ) -> bool:
        assert offboard_runtime is False
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
    manager._parent_children = {"did:test:parent": ["terminal"]}
    manager._child_mandates = {"terminal": object()}

    removal = asyncio.create_task(manager.remove_agent("terminal"))
    await asyncio.wait_for(agent.shutdown_entered.wait(), timeout=1.0)
    removal.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(removal, timeout=1.0)
    assert agent.storage_closed.is_set()
    assert manager.get_agent("terminal") is None
    assert manager._parent_children == {}
    assert manager._child_mandates == {}

    admission, owns_admission = await manager._admit_agent_operation(
        "terminal", kind="load"
    )
    assert owns_admission
    await manager._release_agent_operation(admission)


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
async def test_shutdown_all_sweeps_after_joined_spawn_cancellation_rollback_group() -> None:
    """A joined spawn group is reported only after later fleet cleanup runs."""

    manager = AgentManager()
    swept = SimpleNamespace(
        agent_id="did:test:spawn-group-swept",
        shutdown=AsyncMock(),
    )
    manager._agents["swept"] = swept
    manager._agent_names[swept.agent_id] = "swept"
    admission, owns_admission = await manager._admit_agent_operation(
        "failed-spawn", kind="spawn"
    )
    assert owns_admission
    allow_failure = asyncio.Event()

    async def failed_spawn() -> None:
        try:
            await allow_failure.wait()
            raise BaseExceptionGroup(
                "spawn cancellation and rollback failure",
                [asyncio.CancelledError(), RuntimeError("rollback refund failed")],
            )
        finally:
            await manager._release_agent_operations([admission])

    spawn = asyncio.create_task(failed_spawn())
    admission.spawn_task = spawn
    shutdown = asyncio.create_task(manager.shutdown_all())
    while not manager._agent_registration_sealed:
        await asyncio.sleep(0)
    allow_failure.set()

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(shutdown, timeout=1.0)

    def leaf_errors(error: BaseException):
        if isinstance(error, BaseExceptionGroup):
            for nested in error.exceptions:
                yield from leaf_errors(nested)
            return
        yield error

    assert any(
        isinstance(error, asyncio.CancelledError)
        for error in leaf_errors(exc_info.value)
    )
    assert any(
        "rollback refund failed" in str(error)
        for error in leaf_errors(exc_info.value)
    )
    swept.shutdown.assert_awaited_once()
    assert manager.get_agent("swept") is None
    assert manager._agent_operations == {}


@pytest.mark.asyncio
async def test_shutdown_all_joins_spawn_before_removing_child_or_budget_commit() -> None:
    """A fenced spawn cannot return a dead child or add state after shutdown."""

    class Child:
        agent_id = "did:test:spawn-fenced-child"

        def __init__(self) -> None:
            self.shutdown_calls = 0

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    manager = AgentManager()
    child = Child()
    parent = SimpleNamespace(
        agent_id="did:test:spawn-fenced-parent",
        _private_key=generate_secp256k1_keypair()[0],
        identity=None,
        features={},
        wallet=None,
        shutdown=AsyncMock(),
    )
    manager._agents["spawn-fenced-parent"] = parent
    manager._agent_names[parent.agent_id] = "spawn-fenced-parent"
    budget_entered = asyncio.Event()
    allow_budget = asyncio.Event()

    async def create_child(name, **_kwargs):
        child._raw_storage = SimpleNamespace(
            graph=SimpleNamespace(add_trusted_cross_agent_edge=AsyncMock())
        )
        admission = manager._agent_operations[manager._canonical_agent_name(name)]
        assert admission.before_publish is not None
        await admission.before_publish(child)
        manager._agents[name] = child
        manager._agent_names[child.agent_id] = name
        return child

    async def paused_budget(*_args, **_kwargs):
        budget_entered.set()
        await allow_budget.wait()

    manager.create_agent = create_child
    manager._apply_delegated_budget = paused_budget
    spawn = asyncio.create_task(
        manager.spawn_agent(
            "spawn-fenced",
            parent,
            SpawnMandate(parent_did=parent.agent_id, purpose="race"),
        )
    )
    await asyncio.wait_for(budget_entered.wait(), timeout=1.0)

    shutdown = asyncio.create_task(manager.shutdown_all())
    while not manager._agent_registration_sealed:
        await asyncio.sleep(0)
    assert not shutdown.done()
    allow_budget.set()

    with pytest.raises(RuntimeError, match="Spawn was fenced"):
        await asyncio.wait_for(spawn, timeout=1.0)
    assert await asyncio.wait_for(shutdown, timeout=1.0) is None
    assert child.shutdown_calls == 1
    assert manager.list_agents() == {}
    assert manager._child_budgets == {}
    assert manager._child_mandates == {}
    assert manager._parent_children == {}
    assert manager._pending_spawns == 0


@pytest.mark.asyncio
async def test_quarantined_refund_failure_restores_exact_hold_for_retry(monkeypatch) -> None:
    """A failed reaper refund retains the fenced allocation and unsafe evidence."""

    class FencedDelegated:
        def __init__(self) -> None:
            self.fence_calls = 0

        def fence_spending(self) -> None:
            self.fence_calls += 1

    manager = AgentManager()
    delegated = FencedDelegated()
    parent_wallet = object()
    entry = (delegated, parent_wallet)
    manager._child_budgets["quarantined"] = entry
    release_calls = 0
    release_started = asyncio.Event()
    allow_first_failure = asyncio.Event()

    async def release(_delegated, _parent_wallet):
        nonlocal release_calls
        release_calls += 1
        if release_calls == 1:
            release_started.set()
            await allow_first_failure.wait()
            raise RuntimeError("quarantined provider refund failed")
        return Decimal("0")

    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.release_delegated_wallet",
        release,
    )
    assert manager._handoff_child_budget_release_to_quarantined_reaper(
        "quarantined", agent_id="did:test:quarantined-refund"
    )
    # The reaper owns this withdrawn name while the hold is temporarily out of
    # the normal map.  A new identity cannot slip into that restoration gap.
    await asyncio.wait_for(release_started.wait(), timeout=1.0)
    with pytest.raises(RuntimeError, match="unresolved quarantined cleanup"):
        await manager.load_agent(
            "quarantined", LocalAgentConfig(data_dir="new", port=8801)
        )
    reaper = next(iter(manager._quarantined_shutdown_reapers.values())).task
    allow_first_failure.set()
    with pytest.raises(RuntimeError, match="provider refund failed"):
        await reaper
    await asyncio.sleep(0)

    assert delegated.fence_calls == 1
    assert manager._child_budgets["quarantined"] is entry
    assert manager._unsafe_quarantined_shutdown_failures

    # The retry uses the ordinary removal owner and the same allocation tuple.
    assert await manager.remove_agent("quarantined") is True
    assert release_calls == 2
    assert manager._child_budgets == {}
    # The first unsafe outcome remains terminal evidence until acknowledgement.
    with pytest.raises(ExceptionGroup, match="quarantined shutdown reapers"):
        await manager.drain_quarantined_shutdowns()
    reaper_id = next(iter(manager._unsafe_quarantined_shutdown_failures))
    assert manager.acknowledge_unsafe_quarantined_shutdown_failure(reaper_id)
    assert await manager.drain_quarantined_shutdowns() is False


@pytest.mark.asyncio
async def test_shutdown_all_preserves_tracking_when_quarantined_refund_restores_hold() -> None:
    """Drain-time refund failure cannot prune the retry relation it restores."""

    manager = AgentManager()
    child_name = "quarantined-child"
    parent_did = "did:test:quarantined-parent"
    child = SimpleNamespace(agent_id="did:test:quarantined-child")
    entry = (object(), object())
    mandate = object()
    manager._agents[child_name] = child
    manager._agent_names[child.agent_id] = child_name
    manager._child_budgets[child_name] = entry
    manager._parent_children[parent_did] = [child_name]
    manager._child_mandates[child_name] = mandate

    async def hand_off_then_restore(
        name: str,
        *,
        offboard_runtime: bool = False,
    ) -> bool:
        assert offboard_runtime is False
        assert name == child_name
        assert manager._agents.pop(name) is child
        assert manager._agent_names.pop(child.agent_id) == name
        assert manager._child_budgets.pop(name) is entry

        async def failed_refund() -> None:
            manager._child_budgets[name] = entry
            raise RuntimeError("quarantined refund failed after withdrawal")

        manager._retain_quarantined_cleanup(
            name=name,
            agent_id=child.agent_id,
            task=asyncio.create_task(failed_refund()),
        )
        return True

    manager.remove_agent = hand_off_then_restore
    with pytest.raises(ExceptionGroup, match="fleet agents failed"):
        await manager.shutdown_all()

    assert manager.get_agent(child_name) is None
    assert manager._child_budgets[child_name] is entry
    assert manager.get_children(parent_did) == [child_name]
    assert manager.get_mandate(child_name) is mandate


@pytest.mark.asyncio
async def test_terminate_child_keeps_tracking_until_quarantined_refund_drains() -> None:
    """A bounded remove keeps its parent edge until its refund reaper succeeds."""

    manager = AgentManager()
    child_name = "quarantined-child"
    parent_did = "did:test:quarantined-parent"
    child = SimpleNamespace(agent_id="did:test:quarantined-child")
    entry = (object(), object())
    mandate = object()
    manager._agents[child_name] = child
    manager._agent_names[child.agent_id] = child_name
    manager._child_budgets[child_name] = entry
    manager._parent_children[parent_did] = [child_name]
    manager._child_mandates[child_name] = mandate
    # This lifecycle-ownership fixture intentionally uses object sentinels,
    # not signed authority receipts. Keep it focused on quarantine tracking.
    manager.get_authoritative_children = AsyncMock(
        side_effect=manager.get_children
    )
    refund_started = asyncio.Event()
    allow_refund = asyncio.Event()

    async def hand_off_pending_refund(
        name: str,
        *,
        offboard_runtime: bool = False,
    ) -> bool:
        assert offboard_runtime is False
        assert name == child_name
        assert manager._agents.pop(name) is child
        assert manager._agent_names.pop(child.agent_id) == name
        assert manager._child_budgets.pop(name) is entry

        async def successful_refund() -> None:
            refund_started.set()
            await allow_refund.wait()

        manager._retain_quarantined_cleanup(
            name=name,
            agent_id=child.agent_id,
            task=asyncio.create_task(successful_refund()),
        )
        return True

    manager.remove_agent = hand_off_pending_refund
    assert await manager.terminate_child(parent_did, child_name) is True
    await asyncio.wait_for(refund_started.wait(), timeout=1.0)
    assert manager.get_children(parent_did) == [child_name]
    assert manager.get_mandate(child_name) is mandate

    allow_refund.set()
    assert await manager.drain_quarantined_shutdowns() is False
    assert manager.get_children(parent_did) == []
    assert manager.get_mandate(child_name) is None


@pytest.mark.asyncio
async def test_terminate_child_keeps_tracking_when_quarantined_refund_restores_hold() -> None:
    """A failed quarantined refund restores the hold without losing retry tracking."""

    manager = AgentManager()
    child_name = "failed-quarantined-child"
    parent_did = "did:test:failed-quarantined-parent"
    child = SimpleNamespace(agent_id="did:test:failed-quarantined-child")
    entry = (object(), object())
    mandate = object()
    manager._agents[child_name] = child
    manager._agent_names[child.agent_id] = child_name
    manager._child_budgets[child_name] = entry
    manager._parent_children[parent_did] = [child_name]
    manager._child_mandates[child_name] = mandate
    # This lifecycle-ownership fixture intentionally uses object sentinels,
    # not signed authority receipts. Keep it focused on quarantine tracking.
    manager.get_authoritative_children = AsyncMock(
        side_effect=manager.get_children
    )
    refund_started = asyncio.Event()
    allow_failure = asyncio.Event()

    async def hand_off_then_restore(
        name: str,
        *,
        offboard_runtime: bool = False,
    ) -> bool:
        assert offboard_runtime is False
        assert name == child_name
        assert manager._agents.pop(name) is child
        assert manager._agent_names.pop(child.agent_id) == name
        assert manager._child_budgets.pop(name) is entry

        async def failed_refund() -> None:
            refund_started.set()
            await allow_failure.wait()
            manager._child_budgets[name] = entry
            raise RuntimeError("quarantined refund failed after withdrawal")

        manager._retain_quarantined_cleanup(
            name=name,
            agent_id=child.agent_id,
            task=asyncio.create_task(failed_refund()),
        )
        return True

    manager.remove_agent = hand_off_then_restore
    assert await manager.terminate_child(parent_did, child_name) is True
    await asyncio.wait_for(refund_started.wait(), timeout=1.0)
    assert manager.get_children(parent_did) == [child_name]
    assert manager.get_mandate(child_name) is mandate

    allow_failure.set()
    with pytest.raises(ExceptionGroup, match="quarantined shutdown reapers"):
        await manager.drain_quarantined_shutdowns()

    assert manager._child_budgets[child_name] is entry
    assert manager.get_children(parent_did) == [child_name]
    assert manager.get_mandate(child_name) is mandate


@pytest.mark.asyncio
async def test_batch_onboarding_cancellation_wins_after_claimed_cleanup_settles() -> None:
    """A failed onboarding cannot hide caller cancellation behind cleanup work."""

    class BlockingCleanupAgent:
        agent_id = "did:test:batch-cancel-cleanup"

        def __init__(self) -> None:
            self.shutdown_calls = 0
            self.cleanup_started = asyncio.Event()
            self.allow_cleanup = asyncio.Event()

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            self.cleanup_started.set()
            await self.allow_cleanup.wait()

    manager = AgentManager()
    agent = BlockingCleanupAgent()
    config = LocalAgentConfig(data_dir="batch-cancel", port=8801)

    async def initialize(*_args, **_kwargs):
        return agent

    async def reject_onboarding(*_args, **_kwargs):
        raise RuntimeError("host onboarding failed")

    manager._initialize_agent = initialize
    manager.set_agent_registration_hook(reject_onboarding)
    batch = asyncio.create_task(manager.load_from_config(MultiAgentConfig(agents={"B": config})))
    await asyncio.wait_for(agent.cleanup_started.wait(), timeout=1.0)
    batch.cancel()
    assert not batch.done()
    agent.allow_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(batch, timeout=1.0)
    assert agent.shutdown_calls == 1
    assert manager.list_agents() == {}


@pytest.mark.asyncio
async def test_batch_claims_failed_onboarding_result_before_cleanup_failure() -> None:
    """One failed cleanup is aggregated once; it never gets a second shutdown."""

    manager = AgentManager()
    agent = SimpleNamespace(agent_id="did:test:batch-cleanup-once")
    config = LocalAgentConfig(data_dir="batch-once", port=8801)
    cleanup_calls = 0

    async def initialize(*_args, **_kwargs):
        return agent

    async def reject_onboarding(*_args, **_kwargs):
        raise RuntimeError("host onboarding failed")

    async def fail_shutdown(_name, _agent):
        nonlocal cleanup_calls
        cleanup_calls += 1
        raise RuntimeError("owned cleanup failed")

    manager._initialize_agent = initialize
    manager._shutdown_unregistered_agent = fail_shutdown
    manager.set_agent_registration_hook(reject_onboarding)

    with pytest.raises(ExceptionGroup, match="claimed cleanup failed"):
        await manager.load_from_config(MultiAgentConfig(agents={"B": config}))
    assert cleanup_calls == 1
    assert manager._agent_operations == {}


@pytest.mark.asyncio
async def test_remove_agent_cancellation_wins_over_settled_refund_failure(
    monkeypatch,
) -> None:
    """Cancellation propagates after refund failure evidence has been retained."""

    class Agent:
        agent_id = "did:test:refund-cancel-wins"

        async def shutdown(self) -> None:
            return None

    manager = AgentManager()
    agent = Agent()
    delegated = object()
    parent_wallet = object()
    entry = (delegated, parent_wallet)
    manager._agents["refund-cancel"] = agent
    manager._agent_names[agent.agent_id] = "refund-cancel"
    manager._child_budgets["refund-cancel"] = entry
    refund_started = asyncio.Event()
    allow_failure = asyncio.Event()

    async def fail_refund(_delegated, _parent_wallet):
        refund_started.set()
        await allow_failure.wait()
        raise RuntimeError("refund failed after caller cancellation")

    monkeypatch.setattr(
        "kestrel_sovereign.multi_agent.agent_manager.release_delegated_wallet",
        fail_refund,
    )
    removal = asyncio.create_task(manager.remove_agent("refund-cancel"))
    await asyncio.wait_for(refund_started.wait(), timeout=1.0)
    removal.cancel()
    allow_failure.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(removal, timeout=1.0)
    await asyncio.sleep(0)
    assert manager.get_agent("refund-cancel") is None
    assert manager._child_budgets["refund-cancel"] is entry
    failures = manager.unsafe_removal_budget_release_failures()
    assert len(failures) == 1
    assert "refund failed after caller cancellation" in next(
        iter(failures.values())
    )["failure"]


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
