"""The permission lane must exercise the actual released Docker MCP CLI."""

from pathlib import Path
import re
import sys

import pytest
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
    for path in ("test_mcp_gateway_permission_boundary.py", "test_mcp_permission_peer_stdio.py", "test_mcp_gateway_host_restart.py"):
        assert "tests/integration/" + path in campaign["run"]
    assert "--session-timeout=720" in campaign["run"]
    legacy = workflow["jobs"]["mcp-permission-seam"]["steps"]
    legacy_campaign, = [step for step in legacy if "pytest tests/integration/test_mcp_permission_boundary.py" in step.get("run", "")]
    assert "--session-timeout=480" in legacy_campaign["run"]
    assert "test_mcp_gateway_permission_boundary.py" not in legacy_campaign["run"]


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == 'win32', reason='Original POSIX process retirement requires process groups')
@pytest.mark.parametrize('exit_phase', ['zombie', 'group-lookup'])
async def test_original_admitted_gateway_zombie_is_joined_before_receipt_release(tmp_path, monkeypatch, exit_phase):
    """A real exited-but-unreaped process still has an original waiter."""
    import asyncio
    import os
    from pathlib import Path
    import subprocess
    import sys
    import uuid
    import psutil
    from tests.integration.mcp_permission_peers import (
        GatewayTestPeer, retire_original_gateway_process,
    )

    profile = 'kestrel-' + uuid.uuid4().hex
    peer = GatewayTestPeer('gateway', 'unused', gateway_home=tmp_path)
    command = [sys.executable, '-B', '-c',
        'import sys; sys.stdout.write("ready\\n"); sys.stdout.flush(); sys.stdin.read(1)',
        '--profile=' + profile]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          start_new_session=True) as child:
        try:
            assert child.stdout.readline() == b'ready\n'
            original = psutil.Process(child.pid)
            record = dict(pid=child.pid, created=original.create_time(), profile=profile,
                host_created=psutil.Process(os.getpid()).create_time(),
                executables=[str(Path(original.exe()).resolve())])
            assert peer.admitted_processes(record) == [original]
            if exit_phase == 'zombie':
                child.stdin.write(b'x')
                child.stdin.flush()
                async with asyncio.timeout(2):
                    while original.status() != psutil.STATUS_ZOMBIE:
                        await asyncio.sleep(0.01)
                assert peer.admitted_processes(record) == [original], (
                    'Original admitted zombie lost custody before its physical join')
                with pytest.raises(AssertionError, match='Original native Gateway PID was reused'):
                    peer.admitted_processes(dict(record, created=record['created'] - 1))
            else:
                native_getpgid = os.getpgid

                def after_original_exit(pid):
                    if pid == child.pid:
                        # Close the actual race window with a real exit/reap,
                        # then let the original OS call report its own ESRCH.
                        child.stdin.write(b'x')
                        child.stdin.flush()
                        assert child.wait(timeout=2) == 0
                    return native_getpgid(pid)

                with monkeypatch.context() as boundary:
                    boundary.setattr(os, 'getpgid', after_original_exit)
                    assert peer.admitted_processes(record) == [original], (
                        'Native group lookup lost the birth-verified original receipt')
            await retire_original_gateway_process(original.pid, original.wait,
                lambda: not original.is_running())
            assert not original.is_running()
            assert peer.admitted_processes(record) == []
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
            await asyncio.to_thread(child.wait)
