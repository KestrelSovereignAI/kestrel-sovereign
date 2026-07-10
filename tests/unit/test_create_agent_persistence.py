"""POST /api/agents must persist toml-driven registrations (#2358 codex P1).

Startup loads multi_agent.toml whenever it exists — an unpersisted runtime
creation silently vanishes from the fleet on the next restart while the
dialog reported success.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.endpoints.models import create_agent, CreateAgentRequest
from kestrel_sovereign.multi_agent.config import MultiAgentConfig, LocalAgentConfig


def _request_with_state(**state):
    req = MagicMock()
    req.app.state = SimpleNamespace(**state)
    return req


@pytest.mark.asyncio
async def test_create_agent_persists_into_toml_driven_config(tmp_path):
    config_path = tmp_path / "multi_agent.toml"
    config = MultiAgentConfig(agents={
        "Kestrel": LocalAgentConfig(data_dir=Path("agent_data/Kestrel"), port=8801),
    })
    config.save(config_path)

    created_cfg = LocalAgentConfig(data_dir=Path("agent_data/Newbie"), port=8802)
    manager = MagicMock()
    manager.create_agent = AsyncMock(return_value=SimpleNamespace(agent_id="did:x:newbie"))
    manager._created_configs = {"Newbie": created_cfg}

    req = _request_with_state(
        agent_manager=manager,
        multi_agent_config=config,
        multi_agent_config_path=config_path,
    )
    result = await create_agent.__wrapped__(req, CreateAgentRequest(name="Newbie"))

    assert result["success"] is True
    assert result["persisted"] is True
    reloaded = MultiAgentConfig.from_file(config_path)
    assert "Newbie" in reloaded.agents, "created agent survives a restart (present in the toml)"
    assert "Kestrel" in reloaded.agents, "existing registrations untouched"


@pytest.mark.asyncio
async def test_create_agent_skips_persistence_for_auto_discovered_deployments():
    manager = MagicMock()
    manager.create_agent = AsyncMock(return_value=SimpleNamespace(agent_id="did:x:a"))
    manager._created_configs = {"Auto": LocalAgentConfig(data_dir=Path("agent_data/Auto"), port=8803)}

    req = _request_with_state(
        agent_manager=manager,
        multi_agent_config=None,
        multi_agent_config_path=None,   # no toml drives this deployment
    )
    result = await create_agent.__wrapped__(req, CreateAgentRequest(name="Auto"))
    assert result["success"] is True
    assert result["persisted"] is None, "auto-discovered deployments need no write"


def test_save_round_trips_feature_allowlists(tmp_path):
    """codex P1 round 4: save() dropped `features` — rewriting the toml on a
    create would silently LIFT existing agents' feature restrictions."""
    config_path = tmp_path / "multi_agent.toml"
    config = MultiAgentConfig(agents={
        "Restricted": LocalAgentConfig(
            data_dir=Path("agent_data/Restricted"), port=8801,
            features=["MemoryFeature", "SecurityFeature"],
        ),
        "Open": LocalAgentConfig(data_dir=Path("agent_data/Open"), port=8802),
    })
    config.save(config_path)
    reloaded = MultiAgentConfig.from_file(config_path)
    assert reloaded.agents["Restricted"].features == ["MemoryFeature", "SecurityFeature"]
    assert reloaded.agents["Open"].features is None


@pytest.mark.asyncio
async def test_port_allocator_seeds_past_configured_ports(tmp_path):
    """codex P1 round 4: _port_seq started at 8800 regardless of configured
    ports — the first runtime creation collided with an existing agent's port
    and the PERSISTED conflict failed validation on the next boot."""
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    manager = AgentManager(base_data_dir=tmp_path)
    config = MultiAgentConfig(agents={
        "Kestrel": LocalAgentConfig(
            data_dir=Path("agent_data/Kestrel"), port=8801, autostart=False,
        ),
    })
    await manager.load_from_config(config)   # loads nothing (autostart=False) but must reserve
    assert 8801 in manager._reserved_ports, "configured port reserved"
    assert manager._allocate_port() != 8801, "allocation skips the configured port"


@pytest.mark.asyncio
async def test_port_allocator_accounts_for_the_host_port(tmp_path):
    """codex P1 round 5: the HOST's own port must be reserved too — an agent
    persisted onto it fails port-conflict validation on the next boot."""
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.multi_agent.config import HostConfig

    manager = AgentManager(base_data_dir=tmp_path)
    config = MultiAgentConfig(
        host=HostConfig(port=9001),
        agents={
            "Kestrel": LocalAgentConfig(
                data_dir=Path("agent_data/Kestrel"), port=8801, autostart=False,
            ),
        },
    )
    await manager.load_from_config(config)
    assert 9001 in manager._reserved_ports, "host port reserved"
    assert 8801 in manager._reserved_ports, "agent port reserved"
    allocated = manager._allocate_port()
    assert allocated not in (8801, 9001), "allocation avoids host + agent ports"
    assert allocated < 9001, "a high host port must NOT starve the lower free range (codex round 6)"


def test_save_is_atomic_on_write_failure(tmp_path, monkeypatch):
    """codex P2 round 5: a failed rewrite must never truncate the existing
    registry — the original file survives byte-for-byte."""
    import kestrel_sovereign.multi_agent.config as cfg_mod

    config_path = tmp_path / "multi_agent.toml"
    config = MultiAgentConfig(agents={
        "Kestrel": LocalAgentConfig(data_dir=Path("agent_data/Kestrel"), port=8801),
    })
    config.save(config_path)
    original = config_path.read_text()

    def exploding_dump(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(cfg_mod.toml, "dump", exploding_dump)

    config.agents["Newbie"] = LocalAgentConfig(data_dir=Path("agent_data/Newbie"), port=8802)
    with pytest.raises(OSError):
        config.save(config_path)
    assert config_path.read_text() == original, "original registry untouched by the failed write"
    leftovers = list(tmp_path.glob(".multi_agent.toml.*"))
    assert leftovers == [], "no temp-file litter after failure"


@pytest.mark.asyncio
async def test_create_rejects_registered_but_unloaded_names(tmp_path):
    """codex P2 round 7: remote / autostart=false agents aren't in the
    manager's loaded set — creating their name silently replaced the
    registration in the toml."""
    from kestrel_sovereign.multi_agent.config import RemoteAgentConfig

    config = MultiAgentConfig(agents={
        "Remotey": RemoteAgentConfig(url="http://elsewhere:9000"),
    })
    manager = MagicMock()
    manager.create_agent = AsyncMock()
    req = _request_with_state(
        agent_manager=manager,
        multi_agent_config=config,
        multi_agent_config_path=tmp_path / "multi_agent.toml",
    )
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await create_agent.__wrapped__(req, CreateAgentRequest(name="remotey"))
    assert exc.value.status_code == 409
    manager.create_agent.assert_not_awaited()


def test_atomic_save_preserves_file_mode(tmp_path):
    """codex P2 round 7: mkstemp's 0600 must not strip operator-granted
    permissions from an existing config on rewrite."""
    import os
    config_path = tmp_path / "multi_agent.toml"
    config = MultiAgentConfig(agents={
        "Kestrel": LocalAgentConfig(data_dir=Path("agent_data/Kestrel"), port=8801),
    })
    config.save(config_path)
    os.chmod(config_path, 0o664)
    config.agents["Two"] = LocalAgentConfig(data_dir=Path("agent_data/Two"), port=8802)
    config.save(config_path)
    assert (config_path.stat().st_mode & 0o777) == 0o664, "existing mode preserved across atomic replace"


def test_new_config_files_respect_the_process_umask(tmp_path):
    """codex P2 round 8: a hardcoded 0644 on NEW files bypassed restrictive
    umasks (077 deployments keep their configs private, like open('w') did)."""
    import os
    old_umask = os.umask(0o077)
    try:
        config_path = tmp_path / "multi_agent.toml"
        config = MultiAgentConfig(agents={
            "Kestrel": LocalAgentConfig(data_dir=Path("agent_data/Kestrel"), port=8801),
        })
        config.save(config_path)
        assert (config_path.stat().st_mode & 0o777) == 0o600, "umask 077 -> private new config"
    finally:
        os.umask(old_umask)


def test_atomic_save_writes_through_symlinks(tmp_path):
    """codex P2 round 9: a symlinked multi_agent.toml (operator-managed
    config) must stay a symlink — the save updates the TARGET, exactly like
    the in-place open('w') it replaced."""
    real = tmp_path / "real-config.toml"
    link = tmp_path / "multi_agent.toml"
    config = MultiAgentConfig(agents={
        "Kestrel": LocalAgentConfig(data_dir=Path("agent_data/Kestrel"), port=8801),
    })
    config.save(real)
    link.symlink_to(real)

    config.agents["Two"] = LocalAgentConfig(data_dir=Path("agent_data/Two"), port=8802)
    config.save(link)

    assert link.is_symlink(), "the symlink survives the save"
    reloaded = MultiAgentConfig.from_file(real)
    assert "Two" in reloaded.agents, "the TARGET received the update"


@pytest.mark.asyncio
async def test_create_merges_with_the_current_on_disk_config(tmp_path):
    """codex P1 round 10: saving the startup-time snapshot discarded every
    external edit made to the toml after boot — the merge must start from the
    CURRENT file."""
    config_path = tmp_path / "multi_agent.toml"
    startup = MultiAgentConfig(agents={
        "Kestrel": LocalAgentConfig(data_dir=Path("agent_data/Kestrel"), port=8801),
    })
    startup.save(config_path)
    # An operator adds an agent AFTER startup, directly in the file.
    edited = MultiAgentConfig.from_file(config_path)
    edited.agents["HandEdited"] = LocalAgentConfig(data_dir=Path("agent_data/HandEdited"), port=8805)
    edited.save(config_path)

    manager = MagicMock()
    manager.create_agent = AsyncMock(return_value=SimpleNamespace(agent_id="did:x:n"))
    manager._created_configs = {"Newbie": LocalAgentConfig(data_dir=Path("agent_data/Newbie"), port=8806)}
    req = _request_with_state(
        agent_manager=manager,
        multi_agent_config=startup,          # STALE snapshot
        multi_agent_config_path=config_path,
    )
    result = await create_agent.__wrapped__(req, CreateAgentRequest(name="Newbie"))
    assert result["persisted"] is True
    reloaded = MultiAgentConfig.from_file(config_path)
    assert "HandEdited" in reloaded.agents, "external edit survives the create"
    assert "Newbie" in reloaded.agents
    assert "Kestrel" in reloaded.agents


def test_save_falls_back_to_in_place_when_parent_dir_unwritable(tmp_path):
    """codex P2 round 10: group-writable file under an unwritable directory —
    the pre-atomic behavior persisted fine; refuse to regress it."""
    import os
    subdir = tmp_path / "etcish"
    subdir.mkdir()
    config_path = subdir / "multi_agent.toml"
    config = MultiAgentConfig(agents={
        "Kestrel": LocalAgentConfig(data_dir=Path("agent_data/Kestrel"), port=8801),
    })
    config.save(config_path)
    os.chmod(subdir, 0o555)   # dir read/exec only; the FILE stays writable
    try:
        config.agents["Two"] = LocalAgentConfig(data_dir=Path("agent_data/Two"), port=8802)
        config.save(config_path)   # must not raise
        reloaded = MultiAgentConfig.from_file(config_path)
        assert "Two" in reloaded.agents, "in-place fallback persisted the update"
    finally:
        os.chmod(subdir, 0o755)


@pytest.mark.asyncio
async def test_create_reserves_ports_from_the_current_file_and_validates_merge(tmp_path):
    """codex P1 round 11: an agent/host-port added to the toml AFTER startup
    isn't in the boot-time reservations — creation must reserve from the
    CURRENT file, and the merged config must re-validate before saving."""
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    config_path = tmp_path / "multi_agent.toml"
    startup = MultiAgentConfig(agents={
        "Kestrel": LocalAgentConfig(data_dir=Path("agent_data/Kestrel"), port=8801),
    })
    startup.save(config_path)
    manager = AgentManager(base_data_dir=tmp_path)
    await manager.load_from_config(startup)

    # Operator adds an agent on 8802 AFTER startup.
    edited = MultiAgentConfig.from_file(config_path)
    edited.agents["LateAdd"] = LocalAgentConfig(data_dir=Path("agent_data/LateAdd"), port=8802)
    edited.save(config_path)

    async def fake_create(name, **kw):
        port = manager._allocate_port()
        manager._created_configs[name] = LocalAgentConfig(
            data_dir=Path("agent_data") / name, port=port)
        return SimpleNamespace(agent_id=f"did:x:{name}")
    manager.create_agent = fake_create

    req = _request_with_state(
        agent_manager=manager,
        multi_agent_config=startup,
        multi_agent_config_path=config_path,
    )
    result = await create_agent.__wrapped__(req, CreateAgentRequest(name="Newbie"))
    assert result["persisted"] is True
    reloaded = MultiAgentConfig.from_file(config_path)   # from_file validates conflicts
    ports = [c.port for c in reloaded.agents.values() if isinstance(c, LocalAgentConfig)]
    assert len(ports) == len(set(ports)), "no port conflicts persisted"
    assert reloaded.agents["Newbie"].port not in (8801, 8802)
