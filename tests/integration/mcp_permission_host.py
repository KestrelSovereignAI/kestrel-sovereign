"""Launch the real Core app over one private, process-owned HTTP listener."""

import argparse
import asyncio
import json
import re
from pathlib import Path
import socket

import pytest
import uvicorn

from tests.integration.mcp_permission_peers import (
    MCPTestPeer, owned_https_network, owned_provider_network,
)


class ObservedHost:
    """Record actual registry state after requests; never change app behavior."""
    def __init__(self, app, agent_name, receipt):
        self.app, self.agent_name, self.receipt = app, agent_name, receipt

    async def __call__(self, scope, receive, send):
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

    def record_registry(self, nonce):
        import os
        agent = self.app.state.agent_manager.get_agent(self.agent_name)
        assert agent is not None
        receipt = self.receipt / (nonce + '.json')
        temporary = receipt.with_suffix('.writing')
        temporary.write_text(json.dumps(dict(pid=os.getpid(), did=agent.did,
            remote_tools={name: owner for name, owner in agent._tool_to_feature.items()
                if owner.startswith('mcp:remote:')})))
        temporary.replace(receipt)


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

        peer = MCPTestPeer('remote', spec['selector'], provider_url=spec['provider_url'],
            tls_certificate=Path(spec['certificate']))
        with pytest.MonkeyPatch.context() as network:
            with owned_https_network(network, peer), owned_provider_network(network, peer):
                observations = home / 'observations'
                observations.mkdir(exist_ok=True)
                observed = ObservedHost(server.app, spec['agent_name'], observations)
                runtime = uvicorn.Server(uvicorn.Config(observed,
                    host='127.0.0.1', port=port, access_log=False,
                    timeout_graceful_shutdown=5, log_config=None))
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
                    temporary = ready.with_suffix('.starting')
                    temporary.write_text(json.dumps(dict(port=port, pid=os.getpid(),
                        did=agent.did, server_source=server.__file__)))
                    temporary.replace(ready)
                    await serving
                finally:
                    if not serving.done():
                        runtime.should_exit = True
                        await asyncio.wait_for(serving, 30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('descriptor', type=Path)
    parser.add_argument('ready', type=Path)
    args = parser.parse_args()
    asyncio.run(serve(args.descriptor, args.ready))


if __name__ == '__main__':
    main()
