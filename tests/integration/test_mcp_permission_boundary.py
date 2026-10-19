"""Installed MCP package, native peers, and the host's actual security rail.

This proves governed dispatch, not full plugin-isolation bootstrap or vendor
OAuth. The dedicated MCP CI job installs the optional feature explicitly.
"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
import uuid

import pytest
import pytest_asyncio

from tests.integration.mcp_permission_peers import MCPTestPeer, docker_peer, https_peer
from kestrel_sovereign.a2a.stores.unified.feature_enablement_store import FeatureEnablementStore
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
from kestrel_sovereign.kestrel_agent import KestrelAgent


pytestmark = [pytest.mark.mcp_integration, pytest.mark.timeout(120)]


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
    selector: str
    peer: MCPTestPeer

    def invoke(self, marker):
        task = asyncio.create_task(self.agent.execute_named_tool(
            self.handle, {'marker': marker}, session_id='mcp-boundary', source='mcp-seam-test',
        ))
        self.calls.append(task)
        return task

    async def witness(self):
        return await self.peer.witness(self.feature, self.container)

    def current_connection(self):
        return self.peer.connection(self.feature, self.container)

    async def result_with_witness(self, call, expected_witness):
        result = await call
        assert await self.witness() == expected_witness
        return result


@pytest_asyncio.fixture(params=[pytest.param("container", marks=pytest.mark.docker_mcp), "remote"])
async def mcp_boundary(request, tmp_path, monkeypatch):
    # No importorskip: the installed-feature job must fail on a missing package.
    import kestrel_feature_mcp
    from kestrel_feature_mcp.feature import MCPAgent

    assert "site-packages" in str(Path(kestrel_feature_mcp.__file__).resolve())
    identity = uuid.uuid4().hex
    async with AsyncExitStack() as cleanup:
        peer = await cleanup.enter_async_context(
            docker_peer(tmp_path) if request.param == "container" else https_peer(tmp_path, monkeypatch))
        agent = KestrelAgent(did=f'did:test:mcp-seam-{identity}', storage_path=str(tmp_path / 'agent.db'))
        cleanup.push_async_callback(agent.llm_service.close)
        backend = SQLiteBackend(str(tmp_path / "enablement.db"))
        await backend.connect()
        cleanup.push_async_callback(backend.close)
        enablement = FeatureEnablementStore(backend)
        await enablement.initialize()
        agent._feature_enablement_store = enablement
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
        loaded = await feature.load_tool(peer.selector)
        assert loaded.status.value == 'ok', loaded.to_dict()
        container = loaded.data['container' if peer.kind == 'container' else 'connection']
        owner = f'mcp:{peer.kind}:' + container
        handles = [name for name, candidate in agent._tool_to_feature.items() if candidate == owner]
        assert len(handles) == 1
        assert handles[0].endswith('__record_marker'), agent._direct_tool_defs
        assert len(handles[0]) <= agent.MAX_TOOL_NAME_LEN

        async def drain_calls():
            queue.cancel_all()
            await asyncio.gather(*calls, return_exceptions=True)

        cleanup.push_async_callback(drain_calls)
        yield MCPBoundary(agent, feature, store, queue, requests, calls, container, owner, handles[0], peer.selector, peer)


async def test_denied_tool_has_no_generic_dispatcher_bypass(mcp_boundary):
    peer = mcp_boundary
    await peer.store.set_permission(peer.owner, peer.handle, PermissionLevel.DENY)
    for generic in ('mcp_call_tool', 'mcp_gateway_call'):
        await peer.store.set_permission('MCPAgent', generic, PermissionLevel.ALLOW)
    result = await peer.result_with_witness(peer.invoke('direct-denied'), [])
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
    result = await peer.result_with_witness(asyncio.wait_for(first, 5), ['approved-once'])
    assert result['status'] == 'ok'
    assert not await peer.queue.submit_decision(request.id, True, scope='once')
    second = peer.invoke('not-approved-again')
    following = await asyncio.wait_for(peer.requests.get(), 5)
    assert following.id != request.id and not second.done()
    assert await peer.witness() == ['approved-once']
    assert await peer.queue.submit_decision(following.id, False)
    result = await peer.result_with_witness(asyncio.wait_for(second, 5), ['approved-once'])
    assert result['success'] is False
    decisions = [row['decision'] for row in await peer.store.get_audit_log()
                 if row['feature'] == peer.owner and row['tool'] == peer.handle]
    assert decisions.count('user_approved') == 1
    assert decisions.count('user_denied') == 1


async def test_pending_approval_cannot_execute_against_a_replacement_connection(mcp_boundary):
    peer = mcp_boundary
    await peer.store.set_permission(peer.owner, peer.handle, PermissionLevel.ASK)
    pending = peer.invoke('approved-for-old-connection')
    request = await asyncio.wait_for(peer.requests.get(), 5)
    original = peer.current_connection()
    reloaded = await peer.feature.load_tool(peer.selector)
    assert reloaded.status.value == 'ok', reloaded.to_dict()
    current = peer.current_connection()
    assert current is not original and not original.is_connected
    assert await peer.queue.submit_decision(request.id, True, scope='once')
    result = await peer.result_with_witness(asyncio.wait_for(pending, 5), [])
    assert result['status'] == 'error', result
    assert 'changed; rediscover' in result['error']
    fresh = peer.invoke('approved-for-new-connection')
    following = await asyncio.wait_for(peer.requests.get(), 5)
    assert await peer.queue.submit_decision(following.id, True, scope='once')
    result = await peer.result_with_witness(asyncio.wait_for(fresh, 5), ['approved-for-new-connection'])
    assert result['status'] == 'ok'


@pytest.mark.parametrize("mcp_boundary", ["remote"], indirect=True)
async def test_remote_reconnect_refuses_the_real_uninitialized_enablement_service(mcp_boundary):
    peer = mcp_boundary
    peer.agent._feature_enablement_store = None
    result = await peer.feature.load_tool(peer.selector)
    assert await peer.witness() == []
    assert result.status.value == "error"
    assert "did not retain" in result.error
    assert peer.owner not in peer.agent._tool_to_feature.values()
    assert not peer.feature._remote_connections


@pytest.mark.parametrize("mcp_boundary", ["remote"], indirect=True)
async def test_remote_feature_restore_reads_the_real_enablement_rows(mcp_boundary):
    from kestrel_feature_mcp.feature import MCPAgent

    peer = mcp_boundary
    await peer.store.set_permission(peer.owner, peer.handle, PermissionLevel.DENY)
    saved = await peer.agent.get_enablement_deltas("mcp_server")
    assert len(saved) == 1 and saved[0]["metadata"] == {
        "mode": "remote", "url": peer.selector, "auth": "none",
    }
    await peer.feature.shutdown()
    restored = MCPAgent(peer.agent)
    try:
        await restored.initialize()
        peer.agent.features[restored.name] = restored
        await restored.post_all_features_loaded()
        assert set(restored._remote_connections) == {peer.container}
        assert restored.manager is None
        result = await peer.result_with_witness(peer.invoke("denied-after-feature-restore"), [])
        assert result["success"] is False and "Permission denied" in result["error"]
        assert any(row["decision"] == "auto_denied" for row in await peer.store.get_audit_log())
        result = await restored.unload_tool(peer.selector)
        assert result.status.value == "ok"
        assert await peer.agent.get_enablement_deltas("mcp_server") == []
    finally:
        await restored.shutdown()
