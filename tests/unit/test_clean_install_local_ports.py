"""Disposable local clean-install ports never touch a running default host."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from kestrel_sovereign.multi_agent.config import (
    HostConfig,
    LocalAgentConfig,
    MultiAgentConfig,
)


_SCRIPT = Path(__file__).parents[2] / "scripts/ci/clean_install_local_ports.py"
_SPEC = importlib.util.spec_from_file_location("clean_install_local_ports", _SCRIPT)
assert _SPEC and _SPEC.loader
local_ports = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(local_ports)


def test_local_rehearsal_uses_distinct_loopback_ports(tmp_path):
    path = tmp_path / "multi_agent.toml"
    original = MultiAgentConfig(
        host=HostConfig(),
        agents={
            "Kestrel": LocalAgentConfig(
                data_dir=Path("agent_data/Kestrel"), port=8801, autostart=True
            )
        },
    )
    original.save(path)

    host_port, agent_port = local_ports.allocate_local_ports(path)
    updated = MultiAgentConfig.load(path)
    assert host_port != agent_port
    assert host_port >= 1024 and agent_port >= 1024
    assert updated.host.port == host_port
    assert updated.host.bind == "127.0.0.1"
    assert updated.agents["Kestrel"].port == agent_port
    assert updated.agents["Kestrel"].data_dir == Path("agent_data/Kestrel")


def test_local_rehearsal_refuses_unexpected_agent_inventory(tmp_path):
    path = tmp_path / "multi_agent.toml"
    original = MultiAgentConfig(
        host=HostConfig(),
        agents={
            "Other": LocalAgentConfig(
                data_dir=Path("agent_data/Other"), port=8801, autostart=True
            )
        },
    )
    original.save(path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="exactly its disposable test agent"):
        local_ports.allocate_local_ports(path)
    assert path.read_bytes() == before
