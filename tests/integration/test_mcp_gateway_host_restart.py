"""Native Docker Gateway keeps governance across actual Core host processes."""
import sys
import pytest
from tests.integration.test_mcp_permission_host_restart import (
    host_campaign as host_campaign, exercise_permission_restart, exercise_approval_restart,
)
from tests.integration.test_mcp_permission_boundary import mcp_boundary as mcp_boundary

pytestmark = [pytest.mark.mcp_integration, pytest.mark.docker_mcp, pytest.mark.timeout(120),
    pytest.mark.skipif(sys.platform == 'win32', reason='Native host restart requires SIGTERM')]

@pytest.mark.parametrize('host_campaign', [{'transport': 'gateway'}], indirect=True)
async def test_native_gateway_host_restores_permission_before_another_effect(host_campaign):
    await exercise_permission_restart(host_campaign)
    host_campaign.peer.assert_restart_observed()

@pytest.mark.parametrize('host_campaign', [{'transport': 'gateway'}], indirect=True)
async def test_native_gateway_host_rejects_old_and_replayed_approvals(host_campaign):
    await exercise_approval_restart(host_campaign)
    host_campaign.peer.assert_restart_observed()


@pytest.mark.parametrize('host_campaign', [{'transport': 'gateway'}], indirect=True)
async def test_native_sigterm_runs_gateway_reconciliation_before_signal_replay(host_campaign):
    import json
    from tests.integration.test_mcp_permission_host_restart import (
        native_host, host_client, configure_mcp,
    )

    c = host_campaign
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 1) as host:
        async with host_client(c, host.state) as client:
            await configure_mcp(client, c, host.state)
    reconciled = c.home / 'host-1-ready.gateway-closed'
    assert reconciled.exists(), 'Native SIGTERM replay bypassed original Gateway reconciliation'
    assert json.loads(reconciled.read_text()) == dict(original_processes=1, all_retired=True)


async def test_cancelled_witness_reader_joins_before_volume_inventory(tmp_path, monkeypatch):
    import asyncio
    import threading
    from tests.integration.mcp_permission_peers import gateway_peer

    async with gateway_peer(tmp_path, monkeypatch) as peer:
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        original = peer.client.api.remove_container

        def delayed_remove(container, *args, **kwargs):
            owned = peer.client.containers.get(container)
            assert owned.attrs['Image'] == peer.image_id
            assert owned.attrs['Config']['Labels']['kestrel.mcp.permission.owner'] == peer.owner
            entered.set()
            try:
                assert release.wait(10), 'Test did not release its actual witness reader'
                return original(container, *args, **kwargs)
            finally:
                finished.set()

        with monkeypatch.context() as reader_owner:
            reader_owner.setattr(peer.client.api, 'remove_container', delayed_remove)
            reader = asyncio.create_task(peer.witness(None, None))
            inventory = None
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                reader.cancel('owned witness observation cancelled')
                # Observe actual Task completion, not a timeout as a negative oracle.
                for _ in range(3):
                    await asyncio.sleep(0)
                joined_before_release = not reader.done()
                inventory = asyncio.create_task(peer.containers())
                for _ in range(3):
                    await asyncio.sleep(0)
                release.set()
                outcome, = await asyncio.gather(reader, return_exceptions=True)
                inventory_outcome, = await asyncio.gather(inventory, return_exceptions=True)
                assert await asyncio.to_thread(finished.wait, 10)

                assert isinstance(outcome, asyncio.CancelledError)
                assert joined_before_release, 'Cancelled witness observation abandoned its original Docker worker'
                assert inventory_outcome == [], 'Volume inventory observed an unjoined original witness reader'
            finally:
                release.set()
                await asyncio.gather(reader, *([inventory] if inventory is not None else []), return_exceptions=True)
                assert await asyncio.to_thread(finished.wait, 10)

@pytest.mark.parametrize('host_campaign', [{'transport': 'gateway'}], indirect=True)
@pytest.mark.parametrize('before_readiness', [False, True], ids=['ready', 'before-ready'])
async def test_parent_retires_original_gateway_after_forced_host_exit(host_campaign, before_readiness):
    import asyncio
    import json
    import os
    import shutil
    import signal
    import psutil
    from tests.integration.test_mcp_permission_host_restart import (
        native_host, host_client, configure_mcp,
    )
    from tests.integration.mcp_permission_peers import (
        _remove_owned_gateway_containers, run_owned_mcp_cli,
    )

    c = host_campaign
    generation = 1
    if before_readiness:
        async with native_host(c.home, c.descriptor, c.environment, c.peer, 1) as first:
            async with host_client(c, first.state) as client:
                await configure_mcp(client, c, first.state)
        generation = 2
        # Block only the test-owned readiness file, after real native restore.
        os.mkfifo(c.home / 'host-2-ready.starting', mode=0o600)
    released, active = asyncio.Event(), asyncio.Event()
    state = {}

    async def run_host():
        async with native_host(c.home, c.descriptor, c.environment, c.peer, generation) as running:
            assert not before_readiness, 'Host crossed its original readiness barrier'
            async with host_client(c, running.state) as client:
                await configure_mcp(client, c, running.state)
            state.update(running.state)
            active.set()
            await released.wait()

    task = asyncio.create_task(run_host())
    original = host = None
    profile = configuration = None
    try:
        if before_readiness:
            ready = str(c.home / 'host-2-ready.json')
            async with asyncio.timeout(30):
                while original is None:
                    assert not task.done(), 'Host failed before actual Gateway admission'
                    admissions = c.home / 'host-2-ready.gateway-admissions'
                    for receipt in admissions.glob('kestrel-*.json'):
                        record = json.loads(receipt.read_text())
                        if 'pid' not in record:
                            continue
                        host = psutil.Process(record['host_pid'])
                        assert host.create_time() == record['host_created']
                        original = psutil.Process(record['pid'])
                        assert original.create_time() == record['created']
                        break
                    await asyncio.sleep(0.01)
        else:
            await asyncio.wait_for(active.wait(), 30)
            host = psutil.Process(state['pid'])
            original = psutil.Process(state['gateway']['pid'])
        assert host.ppid() == os.getpid() and host.pid in {p.pid for p in original.parents()}
        profile, = [value.split('=', 1)[1] for value in original.cmdline() if value.startswith('--profile=')]
        configurations = list((c.peer.gateway_home / '.docker/mcp/owned-gateways').iterdir())
        configuration, = configurations
        assert configuration.name.startswith('kestrel-') and configuration.is_dir()
        assert os.getpgid(original.pid) == original.pid
        host.kill()
        released.set()
        error, = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(error, AssertionError), repr(error)
        assert not original.is_running(), 'Host fixture abandoned its original admitted native Gateway'
        assert not configuration.exists(), 'Host fixture retained its original admitted private configuration'
        import sqlite3
        database = c.peer.gateway_home / '.docker/mcp/mcp-toolkit.db'
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
            assert connection.execute('SELECT id FROM working_set WHERE id=?', (profile,)).fetchone() is None
    finally:
        # Independent test reconciliation is never credited to the fixture.
        # Only the original captured process and its private resources qualify.
        released.set()
        if host is not None and host.is_running():
            host.kill()
        await asyncio.gather(task, return_exceptions=True)
        if original is not None and original.is_running():
            assert os.getpgid(original.pid) == original.pid
            os.killpg(original.pid, signal.SIGTERM)
            await asyncio.to_thread(original.wait, 5)
        await _remove_owned_gateway_containers(c.peer)
        if profile is not None:
            import sqlite3
            database = c.peer.gateway_home / '.docker/mcp/mcp-toolkit.db'
            with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
                retained = connection.execute('SELECT id FROM working_set WHERE id=?', (profile,)).fetchone()
            if retained:
                await run_owned_mcp_cli(['profile', 'remove', profile])
        if configuration is not None and configuration.exists():
            shutil.rmtree(configuration)


@pytest.mark.parametrize('mcp_boundary', ['gateway'], indirect=True)
async def test_fixture_wait_failure_still_joins_its_original_gateway(mcp_boundary):
    import asyncio
    import os
    import signal
    import subprocess
    from tests.integration.mcp_permission_peers import retire_original_gateway_process

    boundary = mcp_boundary
    original = boundary.feature.gateway_manager.gateway.process
    assert original.poll() is None and os.getpgid(original.pid) == original.pid
    try:
        os.killpg(original.pid, signal.SIGSTOP)
        with pytest.raises(subprocess.TimeoutExpired):
            await retire_original_gateway_process(original.pid, original.wait,
                lambda: original.poll() is not None)
        assert original.poll() == -signal.SIGKILL, 'Failed native waiter abandoned its original Gateway'
    finally:
        if original.poll() is None:
            os.killpg(original.pid, signal.SIGCONT)
        for container in await boundary.peer.containers():
            await asyncio.to_thread(container.remove, force=True)
        await boundary.feature.shutdown()
