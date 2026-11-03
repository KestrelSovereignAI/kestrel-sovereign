"""`kestrel doctor` reports the co-host warning host startup logs (#3872, #3878).

Each case builds one project, asks ``diagnose`` about it, then runs the real
multi-agent lifespan on the same files and compares. Startup selects its
registry, reconciles it against durable spawn authority and chooses which
agents to start for real; only agent initialization is replaced. Doctor runs
first, so it is shown not to depend on anything startup writes.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign.a2a.native_publication import NativeHostingConfiguration
from kestrel_sovereign.doctor import diagnose, runtime_env
from kestrel_sovereign.inception_service import generate_secp256k1_keypair
from kestrel_sovereign.multi_agent.config import (
    HostConfig,
    LocalAgentConfig,
    MultiAgentConfig,
)
from kestrel_sovereign.paths import MULTI_AGENT_CONFIG_ENV
from kestrel_sovereign.setup.env_file import write_env
from kestrel_sovereign.spawn.authority_registry import (
    SPAWN_AUTHORITY_REGISTRY_FILENAME,
    SpawnAuthorityRegistry,
)
from kestrel_sovereign.spawn.mandate import SpawnMandate, sign_mandate

HOST_PORT = 8877
UNPUBLISHED = "KESTREL_BASE_URL is unset"
NOT_CHECKED = "Could not check native A2A publication"


def _agent(name: str, port: int, *, autostart: bool = True) -> LocalAgentConfig:
    return LocalAgentConfig(
        data_dir=Path("agent_data") / name.lower(), port=port, autostart=autostart
    )


def _project(tmp_path: Path, agents: dict[str, LocalAgentConfig], *, registry=None):
    """A project whose host loads ``agents`` from ``registry``.

    ``registry`` defaults to the project's own ``multi_agent.toml``. It is
    named through ``KESTREL_MULTI_AGENT_CONFIG`` either way, so startup's
    reconciliation has a file to repair, as on a real host.
    """
    project = tmp_path / "project"
    project.mkdir()
    write_env(
        project / ".env",
        {
            "KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii"),
            "KESTREL_API_KEY": "doctor-native-a2a-test-key",
        },
    )
    registry = project / "multi_agent.toml" if registry is None else registry
    MultiAgentConfig(host=HostConfig(port=HOST_PORT), agents=agents).save(registry)
    return project, registry


def _write_anchor(project: Path, config: LocalAgentConfig, did: str) -> None:
    """An identity database holding one agent root: what an anchor read needs."""
    data_dir = config.resolve_data_dir(project)
    data_dir.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(data_dir / "kestrel_prime.db")) as db:
        db.execute("CREATE TABLE graph_nodes (node_id TEXT, node_type TEXT)")
        db.execute("INSERT INTO graph_nodes VALUES (?, 'agent')", (did,))
        db.commit()


def _signed_child(parent_did: str, child_did: str, **kwargs):
    """A parent double holding the key that signed its child's mandate."""
    private_key, _public_key = generate_secp256k1_keypair()
    mandate = sign_mandate(
        SpawnMandate(
            parent_did=parent_did, child_did=child_did, ttl_seconds=0, **kwargs
        ),
        private_key,
    )
    parent = _initialized_agent(parent_did)
    parent._private_key = private_key
    parent.identity = None
    parent._persisted_spawn_mandate = None
    child = _initialized_agent(child_did)
    child._persisted_spawn_mandate = mandate
    return parent, child, mandate


def _initialized_agent(did: str):
    agent = MagicMock()
    agent.agent_id = did
    agent.initialize = AsyncMock()
    agent.shutdown = AsyncMock()

    async def durable_edges(node_id):
        mandate = vars(agent).get("_persisted_spawn_mandate")
        if not isinstance(mandate, SpawnMandate):
            return []
        return [
            SimpleNamespace(
                label="spawned_by",
                source_id=node_id,
                target_id=mandate.parent_did,
                properties=mandate.to_edge_properties(),
            )
        ]

    agent.storage = SimpleNamespace(get_edges_from=durable_edges)
    return agent


def _isolate(monkeypatch, registry: Path) -> None:
    """The environment ``kestrel start`` would give this host, less the URL."""
    monkeypatch.setenv(MULTI_AGENT_CONFIG_ENV, str(registry))
    monkeypatch.setenv("KESTREL_MULTI_AGENT", "1")
    monkeypatch.setenv("KESTREL_API_KEY", "doctor-native-a2a-test-key")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    for name in (
        "KESTREL_BASE_URL",
        "KESTREL_A2A_DISCOVERY_AGENT",
        "PORT",
        "KESTREL_SERVER_HOST",
        "KESTREL_DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _doctor_findings(report) -> list[str]:
    """The co-host lines of a doctor report: its warning, or why it has none."""
    return [
        message
        for message in report.warn
        if UNPUBLISHED in message or NOT_CHECKED in message
    ]


async def _start_host(monkeypatch, project: Path, caplog, agents=None):
    """Run the real multi-agent lifespan in ``project`` to its yield.

    Returns the names startup chose to start, before any of them initialized,
    and the co-host warnings it logged. ``agents`` maps a registration name to
    the double its initialization returns.
    """
    from kestrel_sovereign import host_features as hf
    from kestrel_sovereign import phoenix_supervisor as phoenix_module
    from kestrel_sovereign.multi_agent import agent_manager
    from kestrel_sovereign.security import demo_isolation

    real_manager = agent_manager.AgentManager
    rosters: list[tuple[str, ...]] = []
    doubles = dict(agents or {})

    async def initialize(name, _config):
        return doubles.get(name) or _initialized_agent(f"did:test:{name.lower()}")

    class _Manager:
        """The lifespan's manager; roster selection runs on a real one."""

        init_failures = []

        def __init__(self, **kwargs):
            self._real = real_manager(**kwargs)
            self._real._initialize_agent = AsyncMock(side_effect=initialize)
            self._real._on_agent_registered = AsyncMock()
            self._real._run_hosted_agent_ready_hooks = AsyncMock()

        def reconcile_spawn_authority_restart_roster(self, config):
            return self._real.reconcile_spawn_authority_restart_roster(config)

        async def load_from_config(self, config, *, before_initialize, **kwargs):
            def record(roster):
                rosters.append(roster)
                before_initialize(roster)

            return await self._real.load_from_config(
                config, before_initialize=record, **kwargs
            )

        def set_agent_registration_hook(self, _hook):
            return None

        def set_created_agent_persistence_hook(self, _hook):
            return None

        def set_created_agent_registration_removal_hook(self, _hook):
            return None

        def bind_hold_store(self, _store):
            return None

        def list_agents(self):
            return {}

    async def _host_context(app, _config):
        app.state.host_context = SimpleNamespace(
            hold_store=object(), feature_contribution_runtime=None
        )
        return app.state.host_context

    async def _noop(*_args):
        return None

    monkeypatch.chdir(project)
    monkeypatch.setattr(agent_manager, "AgentManager", _Manager)
    monkeypatch.setattr(server, "_build_host_control_context", _host_context)
    monkeypatch.setattr(server, "_initialize_stop_receipts", _noop)
    monkeypatch.setattr(server, "_prepare_shared_postgres_scheduler_protocol", _noop)
    monkeypatch.setattr(server, "_start_host_scheduler", _noop)
    monkeypatch.setattr(phoenix_module, "should_supervise_phoenix", lambda: False)
    monkeypatch.setattr(demo_isolation, "classify_server_mode", lambda _agents: False)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **_k: [])
    caplog.set_level(logging.WARNING, logger=server.logger.name)

    app = FastAPI()
    async with server._lifespan_startup(app):
        assert app.state.startup_error is None
    [roster] = rosters
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and UNPUBLISHED in record.getMessage()
    ]
    return roster, warnings


@pytest.mark.parametrize(
    ("autostart", "expected_roster"),
    [
        ((True, True, False), ("Emma", "Kite")),
        ((True, False, False), ("Emma",)),
    ],
    ids=["two-started", "one-started-two-cold"],
)
async def test_doctor_judges_the_registry_named_by_kestrel_multi_agent_config(
    monkeypatch, tmp_path, caplog, autostart, expected_roster
):
    """An external registry: doctor counts what startup starts from it.

    The project's own ``multi_agent.toml`` names one other agent, so a doctor
    that read it would count one agent and never warn.
    """
    external = tmp_path / "fleet" / "fleet.toml"
    external.parent.mkdir()
    names = ("Emma", "Kite", "Wren")
    project, registry = _project(
        tmp_path,
        {
            name: _agent(name, 8801 + index, autostart=started)
            for index, (name, started) in enumerate(zip(names, autostart))
        },
        registry=external,
    )
    MultiAgentConfig(
        host=HostConfig(port=HOST_PORT), agents={"Solo": _agent("Solo", 8809)}
    ).save(project / "multi_agent.toml")
    _isolate(monkeypatch, registry)

    report = diagnose(project)
    assert any("Emma" in line and "registered" in line for line in report.ok)
    doctor = _doctor_findings(report)

    roster, startup = await _start_host(monkeypatch, project, caplog)

    assert roster == expected_roster
    assert doctor == startup
    if len(expected_roster) > 1:
        [warning] = startup
        assert f"none of this host's {len(expected_roster)} agents" in warning


async def test_doctor_counts_the_cold_parent_startup_boots(
    monkeypatch, tmp_path, caplog
):
    """A cold parent an active signed child needs is started, so it counts."""
    parent_did = "did:test:doctor-cold-parent"
    child_did = "did:test:doctor-active-child"
    parent, child, mandate = _signed_child(parent_did, child_did)
    agents = {
        "ColdParent": _agent("ColdParent", 8801, autostart=False),
        "ActiveChild": _agent("ActiveChild", 8802),
        "Dormant": _agent("Dormant", 8803, autostart=False),
    }
    project, registry = _project(tmp_path, agents)
    _write_anchor(project, agents["ColdParent"], parent_did)
    SpawnAuthorityRegistry(project).record_active(
        child_name="ActiveChild",
        child_did=child_did,
        mandate=mandate,
        config=agents["ActiveChild"],
    )
    _isolate(monkeypatch, registry)

    doctor = _doctor_findings(diagnose(project))

    roster, startup = await _start_host(
        monkeypatch,
        project,
        caplog,
        agents={"ColdParent": parent, "ActiveChild": child},
    )

    # One autostart row, but startup boots its cold parent too.
    assert roster == ("ColdParent", "ActiveChild")
    [warning] = startup
    assert "none of this host's 2 agents" in warning
    assert doctor == startup


async def test_doctor_reconciles_the_roster_as_startup_does(
    monkeypatch, tmp_path, caplog
):
    """A child a crash left out of ``multi_agent.toml`` is restored and started.

    Startup repairs the file; doctor reaches the same roster without writing.
    """
    parent_did = "did:test:doctor-roster-parent"
    child_did = "did:test:doctor-registry-only-child"
    parent, child, mandate = _signed_child(parent_did, child_did)
    child_config = _agent("Child", 8802)
    project, registry = _project(tmp_path, {"Parent": _agent("Parent", 8801)})
    SpawnAuthorityRegistry(project).record_active(
        child_name="Child",
        child_did=child_did,
        mandate=mandate,
        config=child_config,
    )
    _isolate(monkeypatch, registry)
    before = registry.read_bytes()

    doctor = _doctor_findings(diagnose(project))
    assert registry.read_bytes() == before

    roster, startup = await _start_host(
        monkeypatch,
        project,
        caplog,
        agents={"Parent": parent, "Child": child},
    )

    assert roster == ("Parent", "Child")
    assert "Child" in MultiAgentConfig.load(registry).agents
    [warning] = startup
    assert "none of this host's 2 agents" in warning
    assert doctor == startup


@pytest.mark.parametrize("pending", ["expired", "uncommitted"])
def test_doctor_does_not_guess_a_roster_startup_must_first_repair(
    monkeypatch, tmp_path, pending
):
    """Retiring an expired or uncommitted spawn takes locks only startup takes.

    Its outcome decides which rows load, so doctor says it could not check,
    and leaves the registry and every agent directory exactly as it found them.
    """
    child_did = f"did:test:doctor-{pending}-child"
    kwargs = (
        {"ttl_seconds": 60}
        if pending == "expired"
        else {"ttl_seconds": 0, "authority_committed": False}
    )
    mandate = SpawnMandate(
        parent_did="did:test:doctor-repair-parent",
        child_did=child_did,
        parent_signature="signed-by-parent",
        created_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        **kwargs,
    )
    agents = {"Emma": _agent("Emma", 8801), "Child": _agent("Child", 8802)}
    project, registry = _project(tmp_path, agents)
    SpawnAuthorityRegistry(project).record_active(
        child_name="Child",
        child_did=child_did,
        mandate=mandate,
        config=agents["Child"],
    )
    _isolate(monkeypatch, registry)
    authority = project / "agent_data" / SPAWN_AUTHORITY_REGISTRY_FILENAME
    before = {
        path: path.read_bytes() for path in (project / "agent_data").rglob("*")
        if path.is_file()
    }

    [finding] = _doctor_findings(diagnose(project))

    assert finding.startswith(NOT_CHECKED)
    assert "next host startup repairs" in finding
    assert {
        path: path.read_bytes() for path in (project / "agent_data").rglob("*")
        if path.is_file()
    } == before
    assert authority in before


def test_doctor_never_takes_the_listen_address_for_a_public_url(
    monkeypatch, tmp_path
):
    """The deliberate refusal from #3872: a bind and port publish nothing."""
    project, registry = _project(
        tmp_path, {"Emma": _agent("Emma", 8801), "Kite": _agent("Kite", 8802)}
    )
    _isolate(monkeypatch, registry)
    monkeypatch.setenv("PORT", str(HOST_PORT))
    monkeypatch.setenv("KESTREL_SERVER_HOST", "127.0.0.1")
    env_before = (project / ".env").read_bytes()

    [warning] = _doctor_findings(diagnose(project))

    assert warning.startswith(UNPUBLISHED)
    # The port is advice for the operator to write down, never a value used.
    assert f"KESTREL_BASE_URL=http://127.0.0.1:{HOST_PORT}" in warning
    assert NativeHostingConfiguration.from_environment(runtime_env(project)) is None
    assert (project / ".env").read_bytes() == env_before

    # Written down, it publishes, and doctor has nothing to say.
    write_env(project / ".env", {"KESTREL_BASE_URL": f"http://127.0.0.1:{HOST_PORT}"})
    assert _doctor_findings(diagnose(project)) == []
