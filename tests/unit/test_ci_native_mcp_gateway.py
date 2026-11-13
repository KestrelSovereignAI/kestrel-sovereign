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
@pytest.mark.parametrize('exit_phase', [
    'zombie', 'group-lookup', 'birth-lookup', 'command-line-withdrawn', 'executable-withdrawn',
])
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
            elif exit_phase == 'group-lookup':
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
            elif exit_phase in ('command-line-withdrawn', 'executable-withdrawn'):
                # Linux reads an exiting process's command line and executable
                # as empty once it has released its memory, before /proc calls
                # it a zombie. No portable exit stops in that window, so the
                # original reads here exactly what /proc reports during it.
                name, withdrawn = (
                    ('cmdline', []) if exit_phase == 'command-line-withdrawn' else ('exe', ''))
                native_read = getattr(psutil.Process, name)

                def exiting(process):
                    return withdrawn if process.pid == child.pid else native_read(process)

                with monkeypatch.context() as boundary:
                    boundary.setattr(psutil.Process, name, exiting)
                    assert peer.admitted_processes(record) == [original], (
                        'An exiting original lost its receipt to withdrawn metadata')
            else:
                native_birth = psutil.Process.create_time

                def after_original_reap(process):
                    if process.pid == child.pid:
                        child.stdin.write(b'x')
                        child.stdin.flush()
                        assert child.wait(timeout=2) == 0
                        # macOS performs a fresh native lookup here. Force the
                        # same real cache miss on platforms that cached it.
                        process._create_time = None
                    return native_birth(process)

                with monkeypatch.context() as boundary:
                    boundary.setattr(psutil.Process, 'create_time', after_original_reap)
                    assert peer.admitted_processes(record) == [], (
                        'Reaped original lost reconciliation during native birth lookup')
            outcome, = await asyncio.gather(retire_original_gateway_process(
                original.pid, original.wait, lambda: not original.is_running()), return_exceptions=True)
            assert not original.is_running(), 'Original native Gateway waiter was not physically joined'
            if isinstance(outcome, BaseException):
                raise outcome
            assert peer.admitted_processes(record) == []
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
            await asyncio.to_thread(child.wait)


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == 'win32', reason='Original POSIX process group requires process groups')
@pytest.mark.parametrize('custody', ['original-waiter', 'private-admission', 'unpublished-intent'])
async def test_original_gateway_child_retires_after_its_leader_has_exited(tmp_path, custody):
    import asyncio
    import os
    import signal
    import subprocess
    import uuid
    import psutil
    from tests.integration.mcp_permission_peers import GatewayTestPeer, retire_original_gateway_process

    profile = '--profile=kestrel-' + uuid.uuid4().hex
    descendant_code = ('import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
        'os.kill(os.getpid(),signal.SIGSTOP); time.sleep(60)')
    leader_code = (
        'import subprocess,sys; '
        'child=subprocess.Popen([sys.executable,"-B","-c",sys.argv[1],sys.argv[2]],'
        'stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
        'print(child.pid,flush=True); sys.stdin.read(1)'
    )
    descendant = None
    with subprocess.Popen([sys.executable, '-B', '-c', leader_code, descendant_code, profile],
            start_new_session=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as leader:
        try:
            descendant = psutil.Process(int(leader.stdout.readline()))
            descendant.create_time()
            original = psutil.Process(leader.pid)
            record = dict(pid=leader.pid, created=original.create_time(),
                host_created=psutil.Process(os.getpid()).create_time(), profile=profile.split('=', 1)[1],
                executables=[str(Path(original.exe()).resolve())])
            async with asyncio.timeout(2):
                while descendant.status() != psutil.STATUS_STOPPED:
                    await asyncio.sleep(0.01)
            assert os.getpgid(descendant.pid) == leader.pid
            leader.stdin.write(b'x')
            leader.stdin.flush()
            assert await asyncio.to_thread(leader.wait, 2) == 0
            if custody == 'unpublished-intent':
                record = {key: value for key, value in record.items() if key not in {'pid', 'created'}}
            expected = psutil.TimeoutExpired if custody == 'unpublished-intent' else TimeoutError
            with pytest.raises(expected):
                if custody == 'original-waiter':
                    await retire_original_gateway_process(leader.pid, leader.wait,
                        lambda: leader.poll() is not None)
                else:
                    peer = GatewayTestPeer('gateway', 'unused', gateway_home=tmp_path)
                    for denied in [dict(record, executables=[]),
                            dict(record, profile='kestrel-' + uuid.uuid4().hex),
                            dict(record, created=descendant.create_time() + 1)]:
                        with pytest.raises(AssertionError, match='group custody is unresolved'):
                            peer.admitted_groups(denied)
                        assert descendant.is_running()
                    groups = peer.admitted_groups(record)
                    assert len(groups) == 1, 'Private admission lost its original group after leader reaping'
                    group, = groups
                    assert group.pid == leader.pid
                    await group.retire()
            assert not descendant.is_running(), 'Original Gateway leader exit abandoned its stopped child'
            if custody == 'unpublished-intent':
                recovered = peer.admitted_groups(record)
                assert recovered == [group] and recovered[0] is group, (
                    'Unpublished admission discarded its original terminal receipt')
                assert group.retired()
        finally:
            if descendant is not None:
                if descendant.is_running():
                    descendant.send_signal(signal.SIGKILL)
                await asyncio.to_thread(descendant.wait, 5)
            leader.stdin.close()
            if leader.poll() is None:
                leader.kill()
            await asyncio.to_thread(leader.wait)


@pytest.mark.skipif(sys.platform == 'win32', reason='Private native admission requires POSIX sessions')
def test_native_launch_intent_refuses_a_foreign_session_with_matching_profile(tmp_path):
    import os
    import subprocess
    import uuid
    import psutil
    from tests.integration.mcp_permission_peers import GatewayTestPeer

    profile = 'kestrel-' + uuid.uuid4().hex
    command = [sys.executable, '-B', '-c',
        'import sys; print("ready",flush=True); sys.stdin.read(1)', '--profile=' + profile]
    peer = GatewayTestPeer('gateway', 'unused', gateway_home=tmp_path)
    with subprocess.Popen(command, process_group=0, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as child:
        try:
            assert child.stdout.readline() == b'ready\n'
            original = psutil.Process(child.pid)
            assert os.getpgid(child.pid) == child.pid
            assert os.getsid(child.pid) == os.getsid(os.getpid()) != child.pid
            record = dict(host_created=psutil.Process(os.getpid()).create_time(), profile=profile,
                executables=[str(Path(original.exe()).resolve())])
            with pytest.raises(AssertionError, match='group custody is unresolved'):
                peer.admitted_groups(record)
            assert original.is_running()
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', ['constructor', 'birth'])
@pytest.mark.skipif(sys.platform == 'win32', reason='Native host admission regression uses a private FIFO')
async def test_native_host_retains_cleanup_before_disappearing_metadata(tmp_path, monkeypatch, metadata):
    import asyncio
    import os
    import psutil
    from cryptography.fernet import Fernet
    from tests.integration.mcp_permission_peers import GatewayTestPeer
    from tests.integration.test_mcp_permission_host_restart import native_host, owned_host_environment

    home = tmp_path / 'host'
    home.mkdir(mode=0o700)
    descriptor = home / 'held-descriptor'
    os.mkfifo(descriptor, mode=0o600)
    peer = GatewayTestPeer('gateway', 'unused', gateway_home=home)
    environment = owned_host_environment(home, api_key='owned-metadata-admission',
        data_key=Fernet.generate_key().decode())
    native_spawn, native_process = asyncio.create_subprocess_exec, psutil.Process
    accepted = {}

    async def exited_before_publication(*args, **kwargs):
        process = await native_spawn(*args, **kwargs)
        accepted['process'] = process
        if metadata == 'birth':
            accepted['native_identity'] = native_process(process.pid)
        process.kill()
        await process.wait()
        if metadata == 'birth':
            accepted['native_identity']._create_time = None
        return process

    def original_identity(pid=None, *args, **kwargs):
        if metadata == 'birth' and pid == accepted['process'].pid:
            return accepted['native_identity']
        return native_process(pid, *args, **kwargs)

    async def attempt():
        async with native_host(home, descriptor, environment, peer, 1):
            raise AssertionError('Exited native host published readiness')

    with monkeypatch.context() as boundary:
        boundary.setattr(asyncio, 'create_subprocess_exec', exited_before_publication)
        boundary.setattr(psutil, 'Process', original_identity)
        outcome, = await asyncio.gather(attempt(), return_exceptions=True)
    assert accepted['process'].returncode is not None
    assert len(peer.host_admissions) == 1, 'Accepted native host lost cleanup before metadata lookup'
    assert isinstance(outcome, AssertionError) and 'Owned host failed' in str(outcome), repr(outcome)
