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
        initial_manager = None

        async def release_initial_manager():
            # Keep test custody even if an adversarial lifecycle mutation loses
            # the feature's reference to its original container manager.
            if initial_manager is not None:
                async with AsyncExitStack() as resources:
                    resources.push_async_callback(initial_manager.aclose)
                    for name in reversed(list(initial_manager.owned_containers)):
                        resources.push_async_callback(initial_manager.stop_tool, name)

        cleanup.push_async_callback(release_initial_manager)
        await feature.initialize()
        cleanup.push_async_callback(feature.shutdown)
        agent.features[feature.name] = feature
        loaded = await feature.load_tool(peer.selector)
        assert loaded.status.value == 'ok', loaded.to_dict()
        initial_manager = feature.manager
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


async def test_same_instance_disable_enable_restores_the_governed_connection(mcp_boundary):
    peer = mcp_boundary
    await peer.store.set_permission(peer.owner, peer.handle, PermissionLevel.ASK)
    pending = peer.invoke("approved-before-disable")
    request = await asyncio.wait_for(peer.requests.get(), 5)
    original_connection = peer.current_connection()
    original_manager = peer.feature.manager
    await peer.feature.initialize()
    assert peer.feature.manager is original_manager
    assert peer.current_connection() is original_connection

    await peer.agent._unregister_feature_runtime(peer.feature, unload=False)
    assert peer.agent.features[peer.feature.name] is peer.feature
    assert not peer.feature.enabled and not original_connection.is_connected
    assert peer.owner not in peer.agent._tool_to_feature.values()
    await peer.agent._activate_feature_runtime(peer.feature)
    assert peer.agent.features[peer.feature.name] is peer.feature and peer.feature.enabled
    if peer.peer.kind == "container":
        assert peer.feature.manager is not original_manager
    assert peer.current_connection() is not original_connection
    assert await peer.queue.submit_decision(request.id, True, scope="once")
    result = await peer.result_with_witness(asyncio.wait_for(pending, 5), [])
    assert result["status"] == "error" and "changed; rediscover" in result["error"]

    fresh = peer.invoke("approved-after-reenable")
    following = await asyncio.wait_for(peer.requests.get(), 5)
    assert await peer.queue.submit_decision(following.id, True, scope="once")
    result = await peer.result_with_witness(
        asyncio.wait_for(fresh, 5), ["approved-after-reenable"],
    )
    assert result["status"] == "ok"


@pytest.mark.parametrize("mcp_boundary", ["container"], indirect=True)
async def test_disable_cannot_release_custody_ahead_of_an_admitted_docker_acquisition(mcp_boundary, monkeypatch):
    from kestrel_feature_mcp.manager import MCPToolManager

    peer = mcp_boundary
    # Real Core re-enable with no desired server leaves the active feature's
    # lazy manager unacquired, matching a host's first container load.
    unloaded = await peer.feature.unload_tool(peer.selector)
    assert unloaded.status.value == "ok", unloaded.to_dict()
    await peer.agent._unregister_feature_runtime(peer.feature, unload=False)
    await peer.agent._activate_feature_runtime(peer.feature)
    assert peer.feature.manager is None
    assert not await peer.agent.get_enablement_deltas("mcp_server")
    entered, release = asyncio.Event(), asyncio.Event()
    original_create = MCPToolManager.create
    managers = []

    async def create(**kwargs):
        entered.set()
        await release.wait()
        manager = await original_create(**kwargs)
        managers.append(manager)
        return manager

    monkeypatch.setattr(MCPToolManager, "create", create)
    loading = asyncio.create_task(peer.feature.load_tool(peer.selector))
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        stopping = asyncio.create_task(peer.agent._unregister_feature_runtime(peer.feature, unload=False))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not stopping.done(), "Core released feature custody while its Docker acquisition was in flight"
        release.set()
        result = await loading
        await stopping
        assert result.status.value == "error" and "shutting down" in result.error
        assert not peer.feature.enabled
        assert managers and all(not manager.owned_containers and manager.docker_client is None for manager in managers)
        assert not await peer.agent.get_enablement_deltas("mcp_server")
        assert not any(owner.startswith("mcp:") for owner in peer.agent._tool_to_feature.values())
        with pytest.raises(ToolNotRegisteredError):
            await peer.agent.execute_named_tool(peer.handle, {"marker": "after-disable"}, session_id="mcp-boundary")
        refused = await peer.feature.load_tool(peer.selector)
        assert refused.status.value == "error" and "shutting down" in refused.error
        assert len(managers) == 1
    finally:
        release.set()
        await asyncio.gather(*(task for task in (loading, stopping) if task is not None), return_exceptions=True)
        await peer.feature.shutdown()


@pytest.mark.parametrize("mcp_boundary", ["remote"], indirect=True)
async def test_pending_approval_cannot_execute_while_remote_unload_waits_on_storage(mcp_boundary):
    peer = mcp_boundary
    await peer.store.set_permission(peer.owner, peer.handle, PermissionLevel.ASK)
    pending = peer.invoke("approved-during-unload")
    request = await asyncio.wait_for(peer.requests.get(), 5)
    entered, release = asyncio.Event(), asyncio.Event()
    original_clear = peer.agent.clear_feature_enablement

    async def slow_clear(*args, **kwargs):
        entered.set()
        await release.wait()
        await original_clear(*args, **kwargs)

    peer.agent.clear_feature_enablement = slow_clear
    unloading = asyncio.create_task(peer.feature.unload_tool(peer.selector))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert await peer.queue.submit_decision(request.id, True, scope="once")
        result = await peer.result_with_witness(asyncio.wait_for(pending, 5), [])
        assert result["status"] == "error" and "changed; rediscover" in result["error"]
    finally:
        release.set()
        unloaded = await asyncio.wait_for(unloading, 5)
        peer.agent.clear_feature_enablement = original_clear
    assert unloaded.status.value == "ok"
    assert await peer.agent.get_enablement_deltas("mcp_server") == []
    assert not peer.feature._remote_connections


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


@pytest.mark.parametrize("mcp_boundary", ["remote"], indirect=True)
async def test_restore_database_read_failure_does_not_block_another_remote(mcp_boundary, caplog):
    from kestrel_feature_mcp.feature import MCPAgent

    peer = mcp_boundary
    second = await peer.feature.load_tool(peer.selector + "?connection=second")
    assert second.status.value == "ok", second.to_dict()
    rows = await peer.agent.get_enablement_deltas("mcp_server")
    assert len(rows) == 2
    failed_name, healthy_name = rows[0]["name"], rows[1]["name"]
    await peer.feature.shutdown()
    original_reader = peer.agent.get_enablement_deltas
    backend = peer.agent._feature_enablement_store._backend
    reads = 0

    async def fail_one_read(kind):
        nonlocal reads
        reads += 1
        if reads == 2:
            # The actual store executes its unchanged SELECT against a real
            # temporarily unavailable table; the backend raises SDK QueryError.
            await backend.execute("ALTER TABLE feature_enablement RENAME TO enablement_read_fault")
            try:
                return await original_reader(kind)
            finally:
                await backend.execute("ALTER TABLE enablement_read_fault RENAME TO feature_enablement")
        return await original_reader(kind)

    peer.agent.get_enablement_deltas = fail_one_read
    restored = MCPAgent(peer.agent)
    try:
        await restored.initialize()
        peer.agent.features[restored.name] = restored
        await restored.post_all_features_loaded()
        assert set(restored._remote_connections) == {healthy_name}
        assert "Failed to restore remote MCP " + failed_name in caplog.text
        assert "no such table: feature_enablement" in caplog.text
        assert await original_reader("mcp_server") == rows
        owner = "mcp:remote:" + healthy_name
        handles = [name for name, candidate in peer.agent._tool_to_feature.items() if candidate == owner]
        assert len(handles) == 1
        await peer.store.set_permission(owner, handles[0], PermissionLevel.ALLOW)
        result = await peer.agent.execute_named_tool(
            handles[0], {"marker": "after-store-read-failure"}, session_id="mcp-boundary",
        )
        assert await peer.witness() == ["after-store-read-failure"]
        assert result["status"] == "ok"
    finally:
        peer.agent.get_enablement_deltas = original_reader
        await restored.shutdown()


async def test_post_persistence_disable_reports_the_real_saved_connection_and_restore(mcp_boundary, monkeypatch):
    peer = mcp_boundary
    assert (await peer.feature.unload_tool(peer.selector)).status.value == "ok"
    assert await peer.agent.get_enablement_deltas("mcp_server") == []
    entered, release = asyncio.Event(), asyncio.Event()
    original_persist = peer.feature._persist_server

    async def pause_after_committed_persistence(*args, **kwargs):
        outcome = await original_persist(*args, **kwargs)
        entered.set()
        await release.wait()
        return outcome

    monkeypatch.setattr(peer.feature, "_persist_server", pause_after_committed_persistence)
    loading = asyncio.create_task(peer.feature.load_tool(peer.selector))
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 10)
        saved = await peer.agent.get_enablement_deltas("mcp_server")
        assert len(saved) == 1 and saved[0]["state"] == "enabled"
        connection = peer.current_connection()
        stopping = asyncio.create_task(peer.agent._unregister_feature_runtime(peer.feature, unload=False))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not stopping.done(), "Core released custody before the admitted persistence await settled"
        release.set()
        result = await loading
        outcome = await stopping
        assert outcome.name == "RELEASED"
        assert not connection.is_connected and not peer.feature.enabled
        assert peer.owner not in peer.agent._tool_to_feature.values()
        assert await peer.agent.get_enablement_deltas("mcp_server") == saved
        monkeypatch.setattr(peer.feature, "_persist_server", original_persist)
        await peer.agent._activate_feature_runtime(peer.feature)
        assert peer.current_connection().is_connected
        assert peer.owner in peer.agent._tool_to_feature.values()
        fresh = peer.invoke("saved-after-aborted-load")
        approval = await asyncio.wait_for(peer.requests.get(), 5)
        assert await peer.queue.submit_decision(approval.id, True, scope="once")
        restored_result = await peer.result_with_witness(
            asyncio.wait_for(fresh, 5), ["saved-after-aborted-load"],
        )
        assert restored_result["status"] == "ok"
        assert result.status.value == "error"
        assert "remains saved" in result.error, {"result": result.to_dict(), "saved": saved, "restored": restored_result}
        assert "Unload" in result.error
    finally:
        release.set()
        await asyncio.gather(*(task for task in (loading, stopping) if task is not None), return_exceptions=True)
