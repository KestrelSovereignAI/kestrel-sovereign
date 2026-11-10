"""The permission lane must exercise the actual released Docker MCP CLI."""

from pathlib import Path
import re

import yaml


def test_native_gateway_permission_campaign_cannot_drop_its_cli_or_real_cases():
    root = Path(__file__).parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["mcp-gateway-seam"]["steps"]
    install, = [step for step in steps if "MCP_GATEWAY_VERSION" in step.get("env", {})]
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", install["env"]["MCP_GATEWAY_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", install["env"]["MCP_GATEWAY_ARCHIVE_SHA256"])
    command = install["run"]
    assert "set -euo pipefail" in command
    assert "https://github.com/docker/mcp-gateway/releases/download/" in command
    assert command.index("sha256sum --check --strict") < command.index("tar -xzf")
    assert 'MCP_TEST_DOCKER_PLUGIN=$plugin_dir/docker-mcp' in command
    assert "MCP_TEST_DOCKER_HOST=unix:///var/run/docker.sock" in command
    campaign, = [step for step in steps if "tests/integration/test_mcp_gateway_permission_boundary.py" in step.get("run", "")]
    assert steps.index(install) < steps.index(campaign)
    for path in ("test_mcp_gateway_permission_boundary.py", "test_mcp_permission_peer_stdio.py"):
        assert "tests/integration/" + path in campaign["run"]
    assert "--session-timeout=720" in campaign["run"]
    legacy = workflow["jobs"]["mcp-permission-seam"]["steps"]
    legacy_campaign, = [step for step in legacy if "pytest tests/integration/test_mcp_permission_boundary.py" in step.get("run", "")]
    assert "--session-timeout=480" in legacy_campaign["run"]
    assert "test_mcp_gateway_permission_boundary.py" not in legacy_campaign["run"]
