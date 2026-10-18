"""Installed MCP package, real Docker peer, and the host's actual security rail.

This proves governed dispatch, not full plugin-isolation bootstrap or vendor
OAuth. The dedicated MCP CI job installs the optional feature explicitly.
"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
import shutil
import uuid

import pytest
import pytest_asyncio

from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
from kestrel_sovereign.kestrel_agent import KestrelAgent


pytestmark = [pytest.mark.docker_mcp, pytest.mark.timeout(120)]


@dataclass
class MCPBoundary:
    agent: KestrelAgent
    feature: object
    store: PermissionStore
    queue: ApprovalQueue
    requests: asyncio.Queue
    calls: list
    container: str
    owner: str
    handle: str
    image: str

    def invoke(self, marker):
        task = asyncio.create_task(self.agent.execute_named_tool(
            self.handle, {'marker': marker}, session_id='mcp-boundary', source='mcp-seam-test',
        ))
        self.calls.append(task)
        return task

    async def witness(self):
        container = self.feature.manager.owned_containers[self.container]['container']
        result = await asyncio.to_thread(container.exec_run, ['python', '-c',
            'from pathlib import Path; p=Path("/tmp/permission-witness"); '
            'print(p.read_text() if p.exists() else "", end="")'])
        assert result.exit_code == 0, result.output
        return result.output.decode().splitlines()


@pytest_asyncio.fixture
async def mcp_boundary(tmp_path):
    # No importorskip: the dedicated job must fail if its installed feature is
    # absent. General integration CI deselects docker_mcp before fixture setup.
    import docker
    import kestrel_feature_mcp
    from kestrel_feature_mcp.feature import MCPAgent

    assert 'site-packages' in str(Path(kestrel_feature_mcp.__file__).resolve())
    build = tmp_path / 'peer'
    build.mkdir()
    shutil.copyfile(Path(__file__).with_name('mcp_permission_peer.py'), build / 'peer.py')
    (build / 'Dockerfile').write_text(
        'FROM python:3.11-slim\nWORKDIR /app\nRUN pip install "mcp>=2.3,<3"\n'
        'COPY peer.py .\nCMD ["python", "peer.py"]\n'
    )
    identity = uuid.uuid4().hex
    image = f'mcp-{identity}:test'
    async with AsyncExitStack() as cleanup:
        client = await asyncio.to_thread(docker.from_env)
        cleanup.push_async_callback(asyncio.to_thread, client.close)
        await asyncio.to_thread(client.images.build, path=str(build), tag=image, rm=True)
        cleanup.push_async_callback(asyncio.to_thread, client.images.remove, image)
        agent = KestrelAgent(did=f'did:test:mcp-seam-{identity}', storage_path=str(tmp_path / 'agent.db'))
        cleanup.push_async_callback(agent.llm_service.close)
        feature = MCPAgent(agent)
        calls = []
        requests = asyncio.Queue()

        async def approval_added(request):
            await requests.put(request)

        store = PermissionStore(str(tmp_path / 'permissions.db'))
        await store.initialize()
        queue = ApprovalQueue(on_request_added=approval_added, permission_store=store, agent=agent)
        agent.hooks_manager.register(SecurityHook(store, queue))
        await feature.initialize()
        cleanup.push_async_callback(feature.shutdown)
        agent.features[feature.name] = feature
        loaded = await feature.load_tool(image)
        assert loaded.status.value == 'ok', loaded.to_dict()
        container = loaded.data['container']
        owner = 'mcp:container:' + container
        handles = [name for name, candidate in agent._tool_to_feature.items() if candidate == owner]
        assert len(handles) == 1
        assert handles[0].endswith('__record_marker'), agent._direct_tool_defs
        assert len(handles[0]) <= agent.MAX_TOOL_NAME_LEN

        async def drain_calls():
            queue.cancel_all()
            await asyncio.gather(*calls, return_exceptions=True)

        cleanup.push_async_callback(drain_calls)
        yield MCPBoundary(agent, feature, store, queue, requests, calls, container, owner, handles[0], image)


async def test_denied_tool_has_no_generic_dispatcher_bypass(mcp_boundary):
    peer = mcp_boundary
    await peer.store.set_permission(peer.owner, peer.handle, PermissionLevel.DENY)
    for generic in ('mcp_call_tool', 'mcp_gateway_call'):
        await peer.store.set_permission('MCPAgent', generic, PermissionLevel.ALLOW)
    result = await peer.invoke('direct-denied')
    assert result['success'] is False and 'Permission denied' in result['error']
    for generic, arguments in [
        ('mcp_call_tool', {'container_name': peer.container, 'tool_name': 'record_marker',
                           'args': {'marker': 'generic-denied'}}),
        ('mcp_gateway_call', {'tool_name': 'record_marker', 'arguments': {'marker': 'gateway-denied'}}),
    ]:
        with pytest.raises(ToolNotRegisteredError):
            await peer.agent.execute_named_tool(generic, arguments, session_id='mcp-boundary')
    assert await peer.witness() == []
    audit = await peer.store.get_audit_log()
    assert any(row['feature'] == peer.owner and row['tool'] == peer.handle
               and row['decision'] == 'auto_denied' for row in audit)


async def test_one_shot_approval_cannot_be_replayed_for_an_external_call(mcp_boundary):
    peer = mcp_boundary
    await peer.store.set_permission(peer.owner, peer.handle, PermissionLevel.ASK)
    first = peer.invoke('approved-once')
    request = await asyncio.wait_for(peer.requests.get(), 5)
    assert (request.feature_name, request.tool_name) == (peer.owner, peer.handle)
    assert request.tool_args == {'marker': 'approved-once'}
    assert await peer.witness() == []
    assert await peer.queue.submit_decision(request.id, True, scope='once')
    assert (await asyncio.wait_for(first, 5))['status'] == 'ok'
    assert await peer.witness() == ['approved-once']
    assert not await peer.queue.submit_decision(request.id, True, scope='once')
    second = peer.invoke('not-approved-again')
    following = await asyncio.wait_for(peer.requests.get(), 5)
    assert following.id != request.id and not second.done()
    assert await peer.witness() == ['approved-once']
    assert await peer.queue.submit_decision(following.id, False)
    assert (await asyncio.wait_for(second, 5))['success'] is False
    assert await peer.witness() == ['approved-once']
    decisions = [row['decision'] for row in await peer.store.get_audit_log()
                 if row['feature'] == peer.owner and row['tool'] == peer.handle]
    assert decisions.count('user_approved') == 1
    assert decisions.count('user_denied') == 1


async def test_pending_approval_cannot_execute_against_a_replacement_connection(mcp_boundary):
    peer = mcp_boundary
    await peer.store.set_permission(peer.owner, peer.handle, PermissionLevel.ASK)
    pending = peer.invoke('approved-for-old-connection')
    request = await asyncio.wait_for(peer.requests.get(), 5)
    original = peer.feature.manager.owned_containers[peer.container]['connection']
    reloaded = await peer.feature.load_tool(peer.image)
    assert reloaded.status.value == 'ok', reloaded.to_dict()
    current = peer.feature.manager.owned_containers[peer.container]['connection']
    assert current is not original and not original.is_connected
    assert await peer.queue.submit_decision(request.id, True, scope='once')
    result = await asyncio.wait_for(pending, 5)
    assert result['status'] == 'error', result
    assert 'changed; rediscover' in result['error']
    assert await peer.witness() == []
    fresh = peer.invoke('approved-for-new-connection')
    following = await asyncio.wait_for(peer.requests.get(), 5)
    assert await peer.queue.submit_decision(following.id, True, scope='once')
    assert (await asyncio.wait_for(fresh, 5))['status'] == 'ok'
    assert await peer.witness() == ['approved-for-new-connection']
