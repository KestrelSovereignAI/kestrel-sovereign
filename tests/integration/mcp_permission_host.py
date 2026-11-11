"""Launch the real Core app over one private, process-owned HTTP listener."""

from contextlib import AsyncExitStack
import argparse
import asyncio
import faulthandler
import functools
import json
import logging
import math
import re
from pathlib import Path
import socket
import sys
import traceback

import pytest
import uvicorn

from tests.integration.mcp_permission_peers import (
    MCPTestPeer, owned_https_network, owned_provider_network, owned_gateway_network,
)


def gateway_snapshot(agent):
    """Observe an actual MCP feature's original native process and profile."""
    feature = agent.features.get('MCPAgent')
    manager = getattr(feature, 'gateway_manager', None)
    gateway = getattr(manager, 'gateway', None)
    if gateway is None or gateway.process is None:
        return None
    assert gateway.process.poll() is None
    return dict(pid=gateway.process.pid, profile=gateway._profile_id,
        configuration=str(gateway._config_directory), servers=list(gateway.requested_servers))


class ObservedHost:
    """Record actual registry state after requests; never change app behavior."""
    def __init__(self, app, agent_name, receipt, *, shutdown_trace_interval=1.0):
        if not math.isfinite(shutdown_trace_interval) or shutdown_trace_interval <= 0:
            raise ValueError('Shutdown observation interval must be finite and positive')
        self.app, self.agent_name, self.receipt = app, agent_name, receipt
        self.shutdown_trace_interval = shutdown_trace_interval

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            pending_snapshot = None
            interval = self.shutdown_trace_interval

            def snapshot():
                nonlocal pending_snapshot, interval
                self.record_shutdown_tasks()
                # A single early snapshot misses a later SQLite/native close.
                # Back off while preserving the original lifecycle deadline.
                interval *= 2
                pending_snapshot = asyncio.get_running_loop().call_later(interval, snapshot)

            async def observe_lifespan():
                nonlocal pending_snapshot
                event = await receive()
                if event['type'] == 'lifespan.shutdown':
                    pending_snapshot = asyncio.get_running_loop().call_later(
                        interval, snapshot)
                return event

            try:
                await self.app(scope, observe_lifespan, send)
            finally:
                if pending_snapshot is not None:
                    pending_snapshot.cancel()
            return
        await self.app(scope, receive, send)
        if scope['type'] != 'http':
            return
        nonce = dict(scope['headers']).get(b'x-owned-registry-observation')
        if nonce is None:
            return
        nonce = nonce.decode('ascii')
        assert re.fullmatch('[0-9a-f]{32}', nonce)
        # Complete the real app call, including its request-task epilogue,
        # before observing the registry from the host's original context.
        self.record_registry(nonce)

    def record_shutdown_tasks(self):
        """Report original awaits and worker metadata without changing owners."""
        logger = logging.getLogger(__name__)
        logger.warning('Owned shutdown snapshot')
        # Capture normal Python frame references while holding the interpreter
        # lock. The process-wide crash timer belongs to pytest/the launcher;
        # an individual lifespan observer must not replace or cancel it.
        frames = sys._current_frames()
        try:
            for thread_id, frame in frames.items():
                logger.warning('Owned shutdown thread: %s', thread_id)
                traceback.print_stack(frame, file=sys.stderr)
                worker = frame
                while worker is not None:
                    if (worker.f_globals.get('__name__') == 'aiosqlite.core'
                            and worker.f_code.co_name == '_connection_worker_thread'):
                        # Observe only the current function and completion bits,
                        # never SQL, parameters, results, or Future.exception().
                        function = worker.f_locals.get('function')
                        if isinstance(function, functools.partial):
                            function = function.func
                        future = worker.f_locals.get('future')
                        logger.warning(
                            'Owned shutdown SQLite worker: thread=%s function=%s '
                            'future_done=%s future_cancelled=%s', thread_id,
                            getattr(function, '__qualname__', type(function).__name__),
                            future.done() if future is not None else None,
                            future.cancelled() if future is not None else None)
                    worker = worker.f_back
        finally:
            del frames, frame, worker
        for task in asyncio.all_tasks():
            logger.warning('Owned shutdown pending task: %s', task.get_name())
            task.print_stack(file=sys.stderr)
            awaited, seen = task.get_coro(), set()
            while awaited is not None and id(awaited) not in seen:
                seen.add(id(awaited))
                frame = getattr(awaited, 'cr_frame', getattr(awaited, 'gi_frame', None))
                if frame is not None:
                    logger.warning('Owned shutdown await: %s at %s:%s',
                        frame.f_code.co_qualname, frame.f_code.co_filename, frame.f_lineno)
                awaited = getattr(awaited, 'cr_await', getattr(awaited, 'gi_yieldfrom', None))

    def record_registry(self, nonce):
        import os
        agent = self.app.state.agent_manager.get_agent(self.agent_name)
        assert agent is not None
        receipt = self.receipt / (nonce + '.json')
        temporary = receipt.with_suffix('.writing')
        journal = agent.tool_effect_store
        security = agent.features['SecurityFeature']
        temporary.write_text(json.dumps(dict(pid=os.getpid(), did=agent.did,
            privacy_mode=agent.privacy_mode.value,
            journal_initialized=journal is not None and journal.initialized,
            journal_on_primary=journal is not None and journal.database is agent._raw_storage.db,
            mcp_tools={name: owner for name, owner in agent._tool_to_feature.items()
                if owner.startswith('mcp:')}, gateway=gateway_snapshot(agent),
            approvals={request.id: dict(status=request.status.value, scope=request.user_decision)
                for request in security.approval_queue.pending_requests})))
        temporary.replace(receipt)


class OwnedHostServer(uvicorn.Server):
    """Reconcile fixture resources inside Uvicorn's signal-replay boundary."""
    def __init__(self, config, *, network, peer, gateway_settings):
        super().__init__(config)
        self.network, self.peer, self.gateway_settings = network, peer, gateway_settings

    async def _serve(self, sockets=None):
        async with owned_gateway_network(self.network, self.peer, self.gateway_settings) as gateway:
            with owned_https_network(self.network, self.peer, gateway=gateway), owned_provider_network(self.network, self.peer, gateway=gateway):
                await super()._serve(sockets=sockets)


async def serve(descriptor, ready):
    import os

    spec = json.loads(descriptor.read_text())
    home = descriptor.parent
    with socket.socket() as listener, socket.socket() as agent_port:
        listener.bind(("127.0.0.1", 0))
        agent_port.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        native_config = home / "multi_agent.toml"
        native_config.write_text(
            '[host]\nbind="127.0.0.1"\nport=' + str(port) + '\n'
            '[agents.' + spec['agent_name'] + ']\n'
            'data_dir=' + json.dumps(spec['agent_data']) + '\n'
            'port=' + str(agent_port.getsockname()[1]) + '\n'
            'features=["MCPAgent","BootstrapFeature"]\nautostart=true\n'
        )
        assert os.environ['KESTREL_MULTI_AGENT_CONFIG'] == str(native_config)
        from kestrel_sovereign import server
        from kestrel_sovereign.agent.boot import BootPhaseState

        gateway_settings = spec.get('gateway')
        if gateway_settings is not None:
            gateway_settings = dict(gateway_settings, reconciliation_receipt=str(ready.with_suffix('.gateway-closed')),
                admission_directory=str(ready.with_suffix('.gateway-admissions')))
        peer = MCPTestPeer(spec.get('transport', 'remote'), spec['selector'], provider_url=spec['provider_url'],
            tls_certificate=Path(spec['certificate']))
        with pytest.MonkeyPatch.context() as network:
            observations = home / 'observations'
            observations.mkdir(exist_ok=True)
            observed = ObservedHost(server.app, spec['agent_name'], observations)
            runtime = OwnedHostServer(uvicorn.Config(observed,
                host='127.0.0.1', port=port, access_log=False,
                timeout_graceful_shutdown=5, log_config=None), network=network, peer=peer, gateway_settings=gateway_settings)
            serving = asyncio.create_task(runtime.serve(sockets=[listener]))
            try:
                async with asyncio.timeout(30):
                    while not runtime.started:
                        if serving.done():
                            await serving
                            raise RuntimeError('Owned native host failed before readiness')
                        await asyncio.sleep(0.01)
                agent = server.app.state.agent_manager.get_agent(spec['agent_name'])
                assert agent is not None and agent._boot_state is BootPhaseState.READY
                assert not agent._safe_mode
                from kestrel_sovereign import kestrel_agent as agent_module

                prefix_floor = max(float(agent_module.SHUTDOWN_TIMEOUT),
                    agent_module.KESTREL_AGENT_SHUTDOWN_TIMEOUT_S)
                if 'KESTREL_AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S' in os.environ:
                    from kestrel_sovereign.testing.native_shutdown_budget import reserve_native_shutdown_budget
                    from kestrel_sovereign.multi_agent import agent_manager

                    reserve_native_shutdown_budget(agent, network,
                        lifecycle_module=agent_manager)
                close_budget = agent._raw_storage.db.backend.minimum_close_timeout_s
                tail = agent._durable_tail_minimum_budget(
                    close_budget,
                    agent_module._minimum_storage_potential_preclose_timeout(agent.storage))
                prefix, reserved = agent_module._resolve_shutdown_budget(tail)
                journal = getattr(agent, 'tool_effect_store', None)
                if agent.privacy_config.allows_persistent_storage():
                    assert journal is not None, 'Owned host did not initialize its tool effect journal'
                    assert journal.initialized, 'Owned host published readiness before journal initialization'
                    assert journal.database is agent._raw_storage.db
                else:
                    assert journal is None, 'Volatile boot opened durable journal metadata'
                temporary = ready.with_suffix('.starting')
                temporary.write_text(json.dumps(dict(port=port, pid=os.getpid(),
                    did=agent.did, server_source=server.__file__,
                    sqlite_close_timeout_s=close_budget,
                    shutdown_prefix_floor_s=prefix_floor,
                    shutdown_prefix_s=prefix, shutdown_tail_s=reserved,
                    agent_shutdown_timeout_s=agent_module.KESTREL_AGENT_SHUTDOWN_TIMEOUT_S,
                    journal_initialized=journal is not None and journal.initialized,
                    agent_database=str(agent.storage_path), gateway=gateway_snapshot(agent))))
                temporary.replace(ready)
                await serving
            finally:
                if not serving.done():
                    runtime.should_exit = True
                    await asyncio.wait_for(serving, 30)

def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument('descriptor', type=Path)
    parser.add_argument('ready', type=Path)
    args = parser.parse_args()
    asyncio.run(serve(args.descriptor, args.ready))


if __name__ == '__main__':
    main()
