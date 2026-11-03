"""Real Core HTTP credentials, native MCP effects, approvals and SQLite audit.

No model or operator resources participate. The installed MCP feature is an
explicit dependency of the dedicated MCP CI job, never an import-or-skip.
"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import socket
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx2
from kestrel_sdk.resources import ResourceOutcomeUnknown
import pytest
import pytest_asyncio
import uvicorn

from kestrel_sovereign import server
from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.agent.parts import part_collector
from kestrel_sovereign.auth import caller_context_scope
from kestrel_sovereign.endpoints.auth_oauth import _create_jwt
from kestrel_sovereign.features.privacy import PrivacyAgent
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.security.private_delivery import PrivateDeliveryMiddleware
from kestrel_sovereign.security import private_delivery
from kestrel_sovereign.security.request_auth import (
    AuthenticatedSessionMiddleware, InteractiveAuthenticationLifetimeMiddleware,
    RequestAuthenticationService,
)
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend


pytestmark = [pytest.mark.mcp_integration, pytest.mark.timeout(30)]
API_KEY = "owned-apps-host-api-key"
ALICE, BOB = "alice@apps.test", "bob@apps.test"


class OwnedAuditStore(ObservabilityStore):
    pause = None
    release = None

    async def log_tool_dispatch(self, entry, *, required=False):
        if self.pause is not None:
            self.pause.set()
            await self.release.wait()
        return await super().log_tool_dispatch(entry, required=required)


@dataclass
class Boundary:
    app: object
    client: object
    host: object
    agent: object
    publications: object
    tools: list
    witness: object
    backend: object
    permissions: object
    queue: object
    requests: object
    feature: object

    @property
    def markers(self):
        return self.witness.read_text().splitlines() if self.witness.exists() else []

    def headers(self, principal="key"):
        return ({"X-API-Key": API_KEY} if principal == "key" else
                {"Authorization": "Bearer " + _create_jwt(principal)})

    async def publish(self, *, principal="key", marker="once", session="original-chat"):
        response = await self.client.post("/api/owned-initial", headers=self.headers(principal),
            json={"marker": marker, "session_id": session})
        assert response.status_code == 200, response.text
        value = response.json()
        assert "private-witness" not in response.text
        return next(part["data"]["reference"] for part in value["parts"] if part["type"] == "mcp_app")

    async def open(self, reference, *, principal="key", session="original-chat", **kwargs):
        return await self.client.post(f"/api/mcp/apps/publications/{reference}/views",
            params={"session_id": session}, json={}, headers=self.headers(principal), **kwargs)


@pytest_asyncio.fixture
async def apps_boundary(tmp_path, monkeypatch):
    from kestrel_feature_mcp.apps_execution import RegisteredAppTool
    from kestrel_feature_mcp.apps_publication import AppPublications
    from kestrel_feature_mcp.connection import MCPConnection, bind_tool_caller
    from kestrel_feature_mcp.dynamic_tools import MCPToolWrapper
    from kestrel_feature_mcp.feature import MCPAgent
    from tests.integration.mcp_permission_peer import create_server

    monkeypatch.setenv("KESTREL_API_KEY", API_KEY)
    monkeypatch.setenv("JWT_SECRET_KEY", "owned-apps-jwt-signing-key-with-enough-entropy")
    monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "1")
    monkeypatch.setenv("KESTREL_ALLOWED_EMAILS", ALICE + "," + BOB)
    witness = tmp_path / "native-witness"
    native = create_server(witness, apps=True)
    async with AsyncExitStack() as cleanup:
        listener = socket.socket()
        cleanup.callback(listener.close)
        listener.bind(("127.0.0.1", 0))
        native_app = native.streamable_http_app(stateless_http=True, json_response=True, host="127.0.0.1")
        peer = uvicorn.Server(uvicorn.Config(native_app, log_level="error"))
        peer_task = asyncio.create_task(peer.serve(sockets=[listener]))
        async def close_peer():
            peer.should_exit = True
            await asyncio.wait_for(peer_task, 5)
        cleanup.push_async_callback(close_peer)
        async with asyncio.timeout(5):
            while not peer.started:
                if peer_task.done():
                    await peer_task
                    raise RuntimeError("native MCP peer exited before startup")
                await asyncio.sleep(0.01)
        agent = KestrelAgent(did="did:test:owned-apps-host", storage_path=str(tmp_path / "agent.db"),
                             allowed_features=set(), sync_enabled=False)
        cleanup.push_async_callback(agent.llm_service.close)
        agent.privacy_agent = PrivacyAgent(agent._raw_storage, agent.privacy_mode)
        backend = SQLiteBackend(str(tmp_path / "audit.db"))
        await backend.connect()
        cleanup.push_async_callback(backend.close)
        agent.observability_store = OwnedAuditStore(backend)
        await agent.observability_store.initialize()
        permissions = PermissionStore(str(tmp_path / "permissions.db"))
        await permissions.initialize()
        requests = asyncio.Queue()
        queue = ApprovalQueue(on_request_added=requests.put, permission_store=permissions, agent=agent)
        cleanup.callback(queue.cancel_all)
        agent.hooks_manager.register(SecurityHook(permissions, queue))
        feature = MCPAgent(agent)
        feature.enabled = True
        agent.features[feature.name] = feature
        host = agent.create_mcp_apps_host(feature, configuration={"sandbox_url": "https://sandbox.apps.test/",
                                        "host_origin": "https://host.apps.test"},
                                           is_current=lambda: agent.features.get(feature.name) is feature)
        connection = MCPConnection(f"http://127.0.0.1:{listener.getsockname()[1]}/mcp")
        native_tools = await connection.connect()
        cleanup.push_async_callback(connection.close)
        caller = bind_tool_caller(connection, lambda: True, "owned-apps-peer",
                                  tool_names=[tool.name for tool in native_tools])
        publications = AppPublications(host=agent, agent_id=agent.did, max_pending_calls=4,
            publication_owner=host.publication_owner, require_reader=host.require_reader,
            max_publications=8, max_views=8, max_payload_bytes=8192,
            max_publications_per_owner=4, max_views_per_owner=4, retention_seconds=60)
        host.attach(publications)
        cleanup.push_async_callback(host.close)
        tools = []
        agent.register_dynamic_tools("mcp:owned-apps", [MCPToolWrapper(handle_name="app__" + tool.name,
            tool=tool, caller=caller, app_executor=publications.executor) for tool in native_tools],
            on_registered=lambda tool, binding: tools.append(RegisteredAppTool(tool, binding)))
        publications.register("mcp:owned-apps", tools, is_current=lambda: feature.enabled)
        initial = next(item for item in tools if item.tool.native_tool.name == "open_marker")
        await permissions.set_permission("mcp:owned-apps", initial.binding.name, PermissionLevel.ALLOW)
        app = FastAPI()
        app.state.agent = agent
        app.state.request_authentication = RequestAuthenticationService(
            api_key=lambda: server.get_api_key(generate_if_missing=False))
        app.add_middleware(InteractiveAuthenticationLifetimeMiddleware)
        app.middleware("http")(server.auth_middleware)
        app.add_middleware(AuthenticatedSessionMiddleware, secret_key="owned-apps-session-signing-key",
                           session_cookie="kestrel_session", max_age=600)
        app.add_middleware(PrivateDeliveryMiddleware)
        feature.apps_host = host
        app.include_router(agent.create_mcp_apps_router(feature, resolve_host=lambda: feature.apps_host))

        @app.post("/api/owned-initial")
        async def initial_call(request: Request):
            value = await request.json()
            with caller_context_scope(request.state.caller), part_collector(host=agent, session_id=value["session_id"]):
                try:
                    return await initial.binding.invoke({"marker": value["marker"]},
                        session_id=value["session_id"], source="chat")
                except ResourceOutcomeUnknown:
                    return JSONResponse({"error": "outcome_unknown_do_not_replay"}, status_code=503)

        client = await cleanup.enter_async_context(httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="https://host.apps.test"))
        yield Boundary(app, client, host, agent, publications, tools, witness, backend,
                       permissions, queue, requests, feature)


async def test_initial_result_is_private_and_reopens_without_replaying_native_tool(apps_boundary):
    b = apps_boundary
    reference = await b.publish()
    first = await b.open(reference)
    assert first.status_code == 200, first.text
    value = first.json()
    assert value["result"]["_meta"]["private-witness"] == "once"
    assert "Owned native MCP App" in value["resource"]["html"]
    assert "no-store" in first.headers["cache-control"]
    close = await b.client.delete("/api/mcp/apps/views/" + value["view_id"],
        params={"session_id": "original-chat"}, headers=b.headers())
    assert close.status_code == 200, close.text
    second = await b.open(reference)
    assert second.status_code == 200 and second.json()["view_id"] != value["view_id"]
    assert b.markers == ["once"]
    rows = await b.backend.fetch_all("SELECT session_id,result_status FROM a2a_tool_dispatches")
    assert [tuple(row) for row in rows] == [("original-chat", "success")]


async def test_fixed_routes_resolve_the_current_optional_host_without_retargeting_views(apps_boundary):
    b = apps_boundary
    reference = await b.publish()
    b.feature.apps_host = None
    assert (await b.open(reference)).status_code == 404
    b.feature.apps_host = b.host
    opened = await b.open(reference)
    assert opened.status_code == 200 and opened.json()["result"]["_meta"]["private-witness"] == "once"
    assert b.markers == ["once"]


@pytest.mark.parametrize("owner,reader,session", [("key", ALICE, "original-chat"),
    (ALICE, BOB, "original-chat"), (ALICE, ALICE, "wrong-chat")])
async def test_locator_does_not_grant_another_principal_or_chat_access(apps_boundary, owner, reader, session):
    b = apps_boundary
    reference = await b.publish(principal=owner)
    response = await b.open(reference, principal=reader, session=session)
    assert response.status_code == 404, response.text
    assert "once" not in response.text and b.markers == ["once"]


async def test_app_tool_uses_original_core_one_shot_approval_and_required_audit(apps_boundary):
    b = apps_boundary
    reference = await b.publish()
    opened = await b.open(reference)
    assert opened.status_code == 200, opened.text
    view = opened.json()["view_id"]
    tool = next(item for item in b.tools if item.tool.native_tool.name == "mutate_marker")
    await b.permissions.set_permission("mcp:owned-apps", tool.binding.name, PermissionLevel.ASK)
    url = f"/api/mcp/apps/views/{view}/tools/call?session_id=original-chat"
    call = asyncio.create_task(b.client.post(url, headers=b.headers(),
        json={"name": "mutate_marker", "arguments": {"marker": "approved"}}))
    try:
        approval = await asyncio.wait_for(b.requests.get(), 5)
        assert not call.done() and b.markers == ["once"]
        assert await b.queue.submit_decision(approval.id, True, scope="once")
        response = await asyncio.wait_for(call, 5)
        assert response.status_code == 200, response.text
        assert response.json()["_meta"]["private-witness"] == "approved"
        assert not await b.queue.submit_decision(approval.id, True, scope="once")
        assert b.markers == ["once", "approved"]
        rows = await b.backend.fetch_all("SELECT session_id,result_status FROM a2a_tool_dispatches")
        assert [tuple(row) for row in rows] == [("original-chat", "success"), ("original-chat", "success")]
    finally:
        b.queue.cancel_all()
        await asyncio.gather(call, return_exceptions=True)


async def test_private_routes_refuse_unauthenticated_query_keys_and_request_owner_fields(apps_boundary):
    b = apps_boundary
    reference = await b.publish()
    url = f"/api/mcp/apps/publications/{reference}/views?session_id=original-chat"
    for suffix in ("", "&api_key=" + API_KEY):
        response = await b.client.post(url + suffix, json={})
        assert response.status_code == 401
    response = await b.client.post(url, headers=b.headers(),
        json={"owner": "api_key", "agent_id": b.agent.did})
    assert response.status_code == 400 and b.markers == ["once"]


async def test_private_views_cannot_survive_policy_installation_or_feature_replacement(apps_boundary):
    b = apps_boundary
    reference = await b.publish()
    opened = await b.open(reference)
    assert opened.status_code == 200, opened.text
    b.agent._withdraw_mcp_apps_for_privacy_transition()
    b.agent._privacy_policy_epoch = 1
    response = await b.open(reference)
    assert response.status_code == 404
    await b.host.maintain()
    assert not b.host._views
    fresh = await b.publish(marker="fresh")
    assert (await b.open(fresh)).status_code == 200
    b.agent.features[b.feature.name] = SimpleNamespace(name="MCPAgent", enabled=True)
    assert (await b.open(fresh)).status_code == 403
    assert b.markers == ["once", "fresh"]


async def test_initial_result_staged_before_withdrawal_cannot_publish_under_a_new_epoch(apps_boundary):
    b = apps_boundary
    store = b.agent.observability_store
    store.pause, store.release = asyncio.Event(), asyncio.Event()
    call = asyncio.create_task(b.client.post("/api/owned-initial", headers=b.headers(),
        json={"marker": "completed-before-withdrawal", "session_id": "original-chat"}))
    try:
        await asyncio.wait_for(store.pause.wait(), 5)
        assert b.markers == ["completed-before-withdrawal"] and not call.done()
        b.agent._withdraw_mcp_apps_for_privacy_transition()
        b.agent._privacy_policy_epoch = 2  # same mode, different installation
        store.release.set()
        response = await asyncio.wait_for(call, 5)
        assert response.status_code == 200, response.text
        parts = response.json()["parts"]
        assert not any(part["type"] == "mcp_app" for part in parts)
        assert any(part["type"] == "notice" and "Do not run it again" in part["data"]["body"]
                   for part in parts)
        assert "private-witness" not in response.text
        assert not b.publications._publications and not b.host._views
        rows = await b.backend.fetch_all("SELECT session_id,result_status FROM a2a_tool_dispatches")
        assert [tuple(row) for row in rows] == [("original-chat", "success")]
    finally:
        store.release.set()
        await asyncio.gather(call, return_exceptions=True)


async def test_view_withdrawal_joins_the_actual_outer_send_despite_cancellation_suppression(apps_boundary):
    b = apps_boundary
    reference = await b.publish()
    reached, cancelled, settle = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancellations = []
    async def blocked_transport(scope, receive, send):
        async def physical_send(message):
            if message["type"] == "http.response.body" and b"private-witness" in message.get("body", b""):
                reached.set()
                try:
                    await settle.wait()
                except asyncio.CancelledError:
                    cancellations.append(True)
                    cancelled.set()
                    await settle.wait()
                    raise
            await send(message)
        await b.app(scope, receive, physical_send)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=blocked_transport),
                                  base_url="https://host.apps.test") as client:
        call = asyncio.create_task(client.post(f"/api/mcp/apps/publications/{reference}/views",
            params={"session_id": "original-chat"}, headers=b.headers(), json={}))
        drain = None
        try:
            await asyncio.wait_for(reached.wait(), 5)
            b.host.withdraw_private_data()
            drain = asyncio.create_task(b.host.maintain())
            await asyncio.wait_for(cancelled.wait(), 5)
            b.host.withdraw_private_data()
            await asyncio.sleep(0)
            assert not drain.done() and not call.done() and cancellations == [True]
            settle.set()
            await asyncio.wait_for(drain, 5)
            outcome = await asyncio.gather(call, return_exceptions=True)
            assert isinstance(outcome[0], BaseException)
            assert not b.host._views and b.markers == ["once"]
        finally:
            settle.set()
            await asyncio.gather(call, *([drain] if drain else []), return_exceptions=True)


async def test_cancellation_before_transport_borrow_cancels_the_original_unsent_child(apps_boundary, monkeypatch):
    b = apps_boundary
    reference = await b.publish()
    ready, release = asyncio.Event(), asyncio.Event()
    original = private_delivery.create_owned_task
    children = []
    def delay_send(operation, *, name):
        async def delayed():
            try:
                ready.set()
                await release.wait()
                return await operation
            finally:
                operation.close()
        child = original(delayed(), name=name)
        children.append(child)
        asyncio.current_task().cancel()
        return child
    monkeypatch.setattr(private_delivery, "create_owned_task", delay_send)
    call = asyncio.create_task(b.open(reference))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        # The caller's queued cancellation runs before this event's waiter.
        assert len(children) == 1 and children[0].cancelling() == 1
    finally:
        release.set()
        await asyncio.gather(call, *children, return_exceptions=True)
    assert not b.host._views and b.markers == ["once"]


@pytest.mark.parametrize("configuration", [{}, {"sandbox_url": "javascript:invalid"},
    {"sandbox_url": "https://sandbox.apps.test/", "host_origin": "https://host.apps.test", "max_html_bytes": True},
    {"sandbox_url": "https://host.apps.test/proxy", "host_origin": "https://host.apps.test"},
    {"sandbox_url": "https://host.apps.test:443/proxy", "host_origin": "https://host.apps.test/"}])
async def test_host_configuration_preflight_keeps_the_existing_owner(apps_boundary, configuration):
    b = apps_boundary
    with pytest.raises(ValueError):
        b.agent.validate_mcp_apps_host_configuration(configuration)
    assert b.agent._mcp_apps_hosts == [b.host] and not b.host.retired


async def test_public_origin_is_explicit_and_is_not_inferred_from_a_proxy_backend_url(apps_boundary):
    b = apps_boundary
    reference = await b.publish()
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=b.app), base_url="http://internal-backend") as client:
        path = f"/api/mcp/apps/publications/{reference}/views?session_id=original-chat"
        opened = await client.post(path, json={}, headers={**b.headers(), "Origin": "https://host.apps.test"})
        assert opened.status_code == 200 and opened.json()["host_origin"] == "https://host.apps.test"
        refused = await client.post(path, json={}, headers={**b.headers(), "Origin": "https://foreign.apps.test"})
        assert refused.status_code == 403


async def test_host_configuration_preflight_requires_the_actual_audited_publication_contract(apps_boundary):
    b = apps_boundary
    b.agent.supports_audited_deferred_tool_results = False
    with pytest.raises(TypeError, match="audited deferred results"):
        b.agent.validate_mcp_apps_host_configuration({"sandbox_url": "https://sandbox.apps.test/"})
    assert b.agent._mcp_apps_hosts == [b.host] and not b.host.retired
