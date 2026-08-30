"""Host scheduler protocol bootstrap sequencing regressions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server


@pytest.mark.asyncio
async def test_protocol_preflight_seeds_all_resolved_dids_without_polling(
    monkeypatch,
) -> None:
    """All local DIDs are established before any agent feature runner starts."""
    events: list[str] = []
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
            events.append("storage-initialize")

        async def close(self) -> None:
            self.closed = True
            events.append("storage-close")

    class _Runner:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs
            self.started = False
            runner_instances.append(self)

        async def _ensure_tables(self) -> None:
            events.append("protocol-seeded")

        async def start(self) -> None:  # pragma: no cover - must not be called
            self.started = True
            raise AssertionError("protocol preflight must not start polling")

    agent_configs = {
        "did:pkh:configured-warm": ("Warm", object()),
        "did:pkh:configured-cold": ("Cold", object()),
    }

    async def _resolve(_config):
        events.append("read-only-did-discovery")
        return agent_configs

    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(side_effect=_resolve),
        cold_scheduler_identity_failures=[],
        is_scheduler_agent_authorized=lambda _did: True,
        set_scheduler_polling_managed_by_host=MagicMock(),
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

    await server._prepare_shared_postgres_scheduler_protocol(
        app, manager, object()
    )

    assert events == [
        "read-only-did-discovery",
        "storage-initialize",
        "protocol-seeded",
        "storage-close",
    ]
    assert tuple(runner_instances[0].kwargs["authorized_agent_ids"]) == tuple(
        agent_configs
    )
    assert runner_instances[0].started is False
    assert storage_instances[0].closed is True
    manager.set_scheduler_polling_managed_by_host.assert_called_once_with(True)


@pytest.mark.asyncio
async def test_protocol_preflight_keeps_healthy_dids_when_one_is_unresolved(
    monkeypatch,
) -> None:
    """An unavailable configured identity is latched but cannot block peers."""
    storage_instances = []
    runner_instances = []

    class _Storage:
        def __init__(self, **_kwargs) -> None:
            self.db = object()
            self.closed = False
            storage_instances.append(self)

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    class _Runner:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs
            runner_instances.append(self)

        async def _ensure_tables(self) -> None:
            return None

    missing_identity = RuntimeError("identity database is not initialized")
    manager = SimpleNamespace(
        local_agent_configs_by_did=AsyncMock(
            return_value={"did:pkh:healthy": ("Healthy", object())}
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

    await server._prepare_shared_postgres_scheduler_protocol(
        app, manager, object()
    )

    assert tuple(runner_instances[0].kwargs["authorized_agent_ids"]) == (
        "did:pkh:healthy",
    )
    assert storage_instances[0].closed is True
    assert app.state.scheduler_readiness_failures == [
        {
            "agent": "Unincepted",
            "scope": "identity",
            "state": "unavailable",
            "error_code": "scheduler_identity_unavailable",
            "cause_type": "RuntimeError",
        }
    ]


@pytest.mark.asyncio
async def test_protocol_preflight_cancellation_closes_its_temporary_storage(
    monkeypatch,
) -> None:
    """A cancelled schema-only bootstrap cannot strand an unseen DB pool."""
    entered_initialize = asyncio.Event()
    release_initialize = asyncio.Event()
    storage_instances = []

    class _BlockingStorage:
        def __init__(self, **_kwargs) -> None:
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
            return_value={"did:pkh:healthy": ("Healthy", object())}
        ),
        cold_scheduler_identity_failures=[],
    )
    app = FastAPI()
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setattr(
        "kestrel_sovereign.storage.async_storage.AsyncStorage", _BlockingStorage
    )

    bootstrap = asyncio.create_task(
        server._prepare_shared_postgres_scheduler_protocol(app, manager, object())
    )
    await entered_initialize.wait()
    bootstrap.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bootstrap

    assert storage_instances[0].closed is True


@pytest.mark.asyncio
async def test_lifespan_preflights_before_parallel_agent_initialization(
    monkeypatch,
    tmp_path,
) -> None:
    """The full multi-agent startup order is preflight → load → host runner."""
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
    events: list[str] = []
    removal_resolution_started = asyncio.Event()
    allow_removal_resolution = asyncio.Event()

    class _Manager:
        init_failures = []

        def __init__(self) -> None:
            self.created_agent_persistence_hook = None
            self.created_agent_registration_removal_hook = None

        def set_agent_registration_hook(self, _hook) -> None:
            return None

        def set_created_agent_persistence_hook(self, hook) -> None:
            self.created_agent_persistence_hook = hook

        def set_created_agent_registration_removal_hook(self, hook) -> None:
            self.created_agent_registration_removal_hook = hook

        async def load_from_config(self, config) -> int:
            assert config is fake_config
            events.append("load")
            return 0

        async def resolve_registered_agent_id(self, name, agent_config) -> str:
            assert name == "PersistentChild"
            assert agent_config.port == 8899
            removal_resolution_started.set()
            await allow_removal_resolution.wait()
            return "did:test:persistent-child"

        def list_agents(self):
            return {}

        async def shutdown_all(self) -> None:
            return None

    manager = _Manager()

    async def _preflight(app, supplied_manager, config) -> None:
        assert supplied_manager is manager
        assert config is fake_config
        events.append("preflight")

    async def _start(app, supplied_manager, config) -> None:
        assert supplied_manager is manager
        assert config is fake_config
        events.append("host-start")

    monkeypatch.setenv("KESTREL_MULTI_AGENT", "1")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://scheduler-test")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    monkeypatch.setattr(server, "resolve_multi_agent_path", lambda _env: config_path)
    monkeypatch.setattr(ma_config.MultiAgentConfig, "load", lambda *_a, **_k: fake_config)
    monkeypatch.setattr(agent_manager, "AgentManager", lambda **_k: manager)
    monkeypatch.setattr(
        server, "_prepare_shared_postgres_scheduler_protocol", _preflight
    )
    monkeypatch.setattr(server, "_start_host_scheduler", _start)
    monkeypatch.setattr(did_registry, "install_a2a_did_resolver", lambda *_a, **_k: None)
    monkeypatch.setattr(phoenix_module, "should_supervise_phoenix", lambda: False)
    monkeypatch.setattr(demo_isolation, "classify_server_mode", lambda _agents: False)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **_k: [])

    async with server._lifespan_startup(FastAPI()):
        child_config = ma_config.LocalAgentConfig(
            data_dir=tmp_path / "persistent-child",
            port=8899,
        )
        await manager.created_agent_persistence_hook(
            "PersistentChild",
            child_config,
        )
        removal_task = asyncio.create_task(
            manager.created_agent_registration_removal_hook(
                "PersistentChild",
                "did:test:persistent-child",
            )
        )
        await asyncio.wait_for(removal_resolution_started.wait(), timeout=1)
        external_config = ma_config.LocalAgentConfig(
            data_dir=tmp_path / "external-child",
            port=8901,
        )
        externally_edited = ma_config.MultiAgentConfig.from_file(config_path)
        externally_edited.agents["ExternalChild"] = external_config
        externally_edited.save(config_path)
        concurrent_config = ma_config.LocalAgentConfig(
            data_dir=tmp_path / "concurrent-child",
            port=8900,
        )
        concurrent_persist = asyncio.create_task(
            manager.created_agent_persistence_hook(
                "ConcurrentChild",
                concurrent_config,
            )
        )
        await asyncio.sleep(0)
        assert concurrent_persist.done() is False
        allow_removal_resolution.set()
        rollback = await removal_task
        await concurrent_persist
        assert "PersistentChild" not in ma_config.MultiAgentConfig.from_file(
            config_path
        ).agents
        assert ma_config.MultiAgentConfig.from_file(config_path).agents[
            "ConcurrentChild"
        ] == concurrent_config
        assert ma_config.MultiAgentConfig.from_file(config_path).agents[
            "ExternalChild"
        ] == external_config
        await rollback()
        restored = ma_config.MultiAgentConfig.from_file(config_path).agents
        assert restored["PersistentChild"] == child_config
        assert restored["ConcurrentChild"] == concurrent_config
        assert restored["ExternalChild"] == external_config

    assert events == ["preflight", "load", "host-start"]
    assert callable(manager.created_agent_persistence_hook)
    assert callable(manager.created_agent_registration_removal_hook)
