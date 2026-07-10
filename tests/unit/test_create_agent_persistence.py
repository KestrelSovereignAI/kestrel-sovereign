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
