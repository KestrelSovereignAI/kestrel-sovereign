"""Publish an actual initialized Core agent through the original native SDK."""

import asyncio
import inspect
import json
from contextlib import asynccontextmanager, contextmanager
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from fastapi import FastAPI, Request

from kestrel_sovereign import server
from kestrel_sovereign._async_ownership import OwnedOperation, await_owned_task
from kestrel_sovereign.auth import (
    AuthMethod,
    CallerContext,
    caller_context_scope,
    current_caller_context,
)
from kestrel_sovereign.endpoints.mcp_server import router
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCustody,
    ProcessRuntimeExecutionFence,
    bind_execution_custody,
    execution_cleanup_debts,
    retire_process_runtime_custody,
)
from kestrel_sovereign.mcp_server.http import NativeMCPHostMiddleware
from kestrel_sovereign.mcp_server.oauth import NativeMCPAuthorizationServer
from kestrel_sovereign.mcp_server.policy import MCPServerPolicy
from kestrel_sovereign.mcp_server.publication import NativeMCPPublication
from kestrel_sovereign.mcp_server.transport import NativeMCPServer
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.private_delivery import PrivateDeliveryMiddleware
from kestrel_sovereign.security.request_auth import (
    InteractiveAuthenticationLifetimeMiddleware,
    RequestAuthenticationService,
    current_request_authentication,
)
from tests.integration.test_native_assistant_write_receipts import (
    assistant_agent as _assistant_agent,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
assistant_agent = _assistant_agent
KEY = "owned-publication-durable-key"


@pytest.fixture
async def publication(assistant_agent, monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", KEY)
    agent = assistant_agent
    assert agent._runtime_publication_ready.is_set()
    assert agent.llm_service.disabled
    app = FastAPI()
    app.state.agent = agent
    app.state.request_authentication = RequestAuthenticationService(api_key=lambda: KEY)
    app.include_router(router)
    app.add_middleware(InteractiveAuthenticationLifetimeMiddleware)
    app.middleware("http")(server.auth_middleware)
    app.add_middleware(NativeMCPHostMiddleware)
    app.add_middleware(PrivateDeliveryMiddleware)
    policy = MCPServerPolicy.model_validate_json(json.dumps({"console_origin": "https://kestrel.example",
        "clients": [{"client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret"}]}))
    p = await NativeMCPPublication.prepare(app, agent, "OwnedNativeMCP", policy, sovereign_key=lambda: KEY)
    async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app),
            base_url=policy.console_origin, follow_redirects=False) as client:
        try:
            yield p, client
        finally:
            await p.close()


async def test_prepared_mcp_routes_are_private_until_the_original_host_commit(publication):
    p, client = publication
    path = urlsplit(p.resource).path
    assert not p.registry.is_current(p.binding)
    with pytest.raises(HostCapabilityAccessDenied, match="publication"):
        p.require_current()
    response = await client.get(path + "/oauth/.missing")
    assert response.status_code in (401, 404)
    p.publish()
    p.require_current()
    response = await client.get("/.well-known/oauth-authorization-server" + path + "/oauth")
    assert response.status_code == 200, response.text
    assert response.json()["issuer"] == p.resource + "/oauth"


async def test_actual_mcp_lane_does_not_promote_an_added_owner_key(publication, monkeypatch):
    p, client = publication
    p.publish()
    path = urlsplit(p.resource).path
    original = p.runtime.authorization.verify_token
    interactive = []

    async def verify(token):
        interactive.append(current_request_authentication())
        return await original(token)

    monkeypatch.setattr(p.runtime.authorization, "verify_token", verify)
    response = await client.post(path, headers={"x-api-key": KEY, "authorization": "Bearer " + KEY,
        "content-type": "application/json",
        "accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert response.status_code == 401, "A Console key bypassed the official MCP Bearer verifier"
    assert "resource_metadata" in response.headers["www-authenticate"]
    assert interactive == [None], "The original native MCP request acquired interactive owner credentials"
    caller = CallerContext.mcp_transport()
    assert caller.auth_method is AuthMethod.MCP_TRANSPORT and not caller.is_sovereign


async def test_published_authorization_and_console_use_the_same_original_sql_grants(publication):
    p, client = publication
    p.publish()
    path = urlsplit(p.resource).path
    response = await client.get(path + "/oauth/authorize", params={"response_type": "code",
        "client_id": "owned-provider", "redirect_uri": "https://provider.example/callback",
        "scope": "memory:read", "resource": p.resource, "state": "original-state",
        "code_challenge": "a" * 43, "code_challenge_method": "S256"})
    assert response.status_code == 302, response.text
    location = response.headers["location"]
    handle = parse_qs(urlsplit(location).query)["request"][0]
    shell = await client.get(location)
    assert shell.status_code == 200
    assert 'data-api-prefix=""' in shell.text
    assert handle not in shell.text, "Opaque consent hint was rendered into HTML"
    description = await client.get("/api/mcp/server/consent/requests/" + handle, headers={"x-api-key": KEY})
    assert description.status_code == 200, description.text
    assert description.json()["resource"] == p.resource
    response = await client.post("/api/mcp/server/consent/requests/" + handle + "/decision",
        headers={"x-api-key": KEY}, json={"decision": "approve", "scopes": ["memory:read"]})
    assert response.status_code == 200, response.text
    assert parse_qs(urlsplit(response.json()["redirect_url"]).query)["state"] == ["original-state"]
    assert await p.runtime.authorization.pending.read(handle) is None


async def test_withdrawal_precedes_sdk_retirement_and_keeps_borrowed_primary_backend(publication):
    p, _client = publication
    p.publish()
    original = p._resident
    assert p.agent._resource_custody().is_held("native_mcp_server")
    p.withdraw()
    assert not p.registry.is_current(p.binding)
    with pytest.raises(HostCapabilityAccessDenied):
        p.require_current()
    await p.close()
    assert original.done() and original.exception() is None, "Original SDK lifetime was not physically joined"
    assert not p.agent._resource_custody().is_held("native_mcp_server")
    assert await p.backend.fetch_val("SELECT 1") == 1, "MCP retirement closed its borrowed primary backend"


async def test_retirement_waits_for_the_original_native_http_reader(publication):
    p, _client = publication
    p.publish()
    entered, release = asyncio.Event(), asyncio.Event()

    async def request():
        async with p.request_lifetime():
            entered.set()
            await release.wait()

    reader = asyncio.create_task(request())
    await entered.wait()
    closer = asyncio.create_task(p.close())
    await asyncio.sleep(0)
    assert not closer.done(), "Native MCP retired while its original HTTP reader was still active"
    release.set()
    await reader
    await closer


async def test_normal_agent_shutdown_physically_joins_the_original_mcp_owner(publication):
    p, _client = publication
    p.publish()
    original = p._resident
    await p.agent.shutdown()
    assert original.done() and original.exception() is None
    assert not p.agent._resource_custody().is_held("native_mcp_server")
    assert not p.registry.is_current(p.binding)


async def test_actual_single_agent_onboarding_is_independent_of_a2a_configuration(publication):
    p, _client = publication
    await p.close()
    environment = {"KESTREL_MCP_SERVER": json.dumps({
        **p.policy.model_dump(mode="json"), "clients": [{**p.policy.clients[0].model_dump(mode="json"),
            "client_secret": "owned-secret"}]})}
    assert "KESTREL_BASE_URL" not in environment
    replacement = await server._onboard_single_agent_mcp(p.app, p.agent, environment)
    try:
        replacement.require_current()
        assert replacement.registry.is_current(replacement.binding)
        assert replacement.runtime.authorization.grants.backend is p.backend
        assert replacement._resident is not p._resident
    finally:
        await replacement.close()


async def test_unconfigured_core_does_not_prepare_native_mcp_or_require_server_keys(publication):
    p, _client = publication
    await p.close()
    assert await server._onboard_single_agent_mcp(p.app, p.agent, {}) is None
    assert vars(p.agent).get("_native_mcp_publication") is None


async def test_actual_manager_commit_and_withdrawal_own_the_same_mcp_generation(publication, tmp_path):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    p, _client = publication
    await p.close()
    manager = AgentManager(base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False)
    p.app.state.agent = None
    p.app.state.agent_manager = manager
    assert "_agent_manager_publication_event" not in vars(p.agent)
    prepared = await NativeMCPPublication.prepare(p.app, p.agent, "OwnedNativeMCP", p.policy,
        sovereign_key=lambda: KEY, manager=manager)
    try:
        async with manager.a2a_lifecycle_lease():
            manager._register_agent("OwnedNativeMCP", p.agent)
        prepared.require_current()
        assert vars(p.agent)["_agent_manager_published"] is True
        assert prepared.registry.is_current(prepared.binding)
        async with manager.a2a_lifecycle_lease():
            manager._withdraw_initialized_agent("OwnedNativeMCP", p.agent)
        assert vars(p.agent)["_agent_manager_published"] is False
        with pytest.raises(HostCapabilityAccessDenied):
            prepared.require_current()
        assert manager.get_agent("OwnedNativeMCP") is None
        assert all(owner is not p.agent for owner in manager._fenced_agent_routes.values()), (
            "Committed private withdrawal retained its original agent route fence"
        )
        assert not prepared.registry.is_current(prepared.binding)
    finally:
        await prepared.close()


async def test_actual_host_onboarding_publishes_and_routes_original_console_tenant(publication, tmp_path, monkeypatch):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager
    from kestrel_sovereign.multi_agent.route_name import encode_agent_route_name

    p, client = publication
    await p.close()
    policy = {**p.policy.model_dump(mode="json"), "clients": [{**p.policy.clients[0].model_dump(mode="json"),
        "client_secret": "owned-secret"}]}
    monkeypatch.setenv("KESTREL_MCP_SERVER", json.dumps(policy))
    monkeypatch.delenv("KESTREL_BASE_URL", raising=False)
    manager = AgentManager(base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False)
    app = p.app
    app.state.agent = None
    app.state.agent_manager = manager
    app.add_middleware(server.MultiAgentAgentRoutingMiddleware)
    route_name = "Owned native / MCP agent"
    async with manager.a2a_lifecycle_lease():
        manager._onboarding_agents[route_name] = p.agent
        rollback = await server._onboard_host_registered_agent(app, manager, route_name, p.agent)
        prepared = p.agent._native_mcp_publication
        assert not prepared.registry.is_current(prepared.binding)
        assert manager.get_agent(route_name) is None
        manager._register_agent(route_name, p.agent)
        manager._onboarding_agents.pop(route_name)
    try:
        prepared.require_current()
        response = await client.get(prepared.resource + "/oauth/authorize", params={"response_type": "code",
            "client_id": "owned-provider", "redirect_uri": "https://provider.example/callback",
            "scope": "memory:read", "resource": prepared.resource, "state": "original-host-state",
            "code_challenge": "a" * 43, "code_challenge_method": "S256"})
        assert response.status_code == 302, response.text
        location = response.headers["location"]
        handle = parse_qs(urlsplit(location).query)["request"][0]
        prefix = "/api/agent-routes/" + encode_agent_route_name(route_name)
        shell = await client.get(location)
        assert shell.status_code == 200
        assert f'data-api-prefix="{prefix}"' in shell.text
        description = await client.get(prefix + "/api/mcp/server/consent/requests/" + handle,
            headers={"x-api-key": KEY})
        assert description.status_code == 200, description.text
        assert description.json()["resource"] == prepared.resource
        assert description.json()["agent_name"] == route_name
        denied = await client.post(prefix + "/api/mcp/server/consent/requests/" + handle + "/decision",
            headers={"x-api-key": KEY}, json={"decision": "deny", "scopes": []})
        assert denied.status_code == 200, denied.text
        assert parse_qs(urlsplit(denied.json()["redirect_url"]).query)["state"] == ["original-host-state"]
    finally:
        async with manager.a2a_lifecycle_lease():
            manager._withdraw_initialized_agent(route_name, p.agent)
            await rollback()
        assert prepared._resident.done()
        assert await prepared.backend.fetch_val("SELECT 1") == 1


async def test_failed_mcp_owner_check_still_revokes_a2a_authority(publication, tmp_path, monkeypatch):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    p, _client = publication
    await p.close()
    policy = {**p.policy.model_dump(mode="json"), "clients": [{**p.policy.clients[0].model_dump(mode="json"),
        "client_secret": "owned-secret"}]}
    monkeypatch.setenv("KESTREL_MCP_SERVER", json.dumps(policy))
    monkeypatch.delenv("KESTREL_BASE_URL", raising=False)
    manager = AgentManager(base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False)
    p.app.state.agent = None
    p.app.state.agent_manager = manager
    async with manager.a2a_lifecycle_lease():
        manager._onboarding_agents["OwnedNativeMCP"] = p.agent
        rollback = await server._onboard_host_registered_agent(p.app, manager, "OwnedNativeMCP", p.agent)
        prepared = p.agent._native_mcp_publication
        manager._register_agent("OwnedNativeMCP", p.agent)
        manager._onboarding_agents.pop("OwnedNativeMCP")
    assert manager.a2a_hosted_policy_for(p.agent) is not None
    try:
        p.agent._native_mcp_publication = object()
        async with manager.a2a_lifecycle_lease():
            with pytest.raises(TypeError, match="original host publication"):
                manager._withdraw_initialized_agent("OwnedNativeMCP", p.agent)
        assert vars(p.agent)["_agent_manager_published"] is False
        assert manager.a2a_hosted_policy_for(p.agent) is None, "A failed MCP owner check left original A2A authority active"
        assert manager.get_agent("OwnedNativeMCP") is None, "Failed protocol retirement left ordinary tenant routing published"
        assert p.agent not in manager.list_agents().values(), "Failed protocol retirement remained publicly discoverable"
        assert manager._get_agent_for_lifecycle("OwnedNativeMCP") is p.agent, "Public fencing discarded original cleanup ownership"
    finally:
        p.agent._native_mcp_publication = prepared
        async with manager.a2a_lifecycle_lease():
            manager._withdraw_initialized_agent("OwnedNativeMCP", p.agent)
            await rollback()


async def test_invalid_mcp_shutdown_owner_does_not_skip_other_original_resources(publication, monkeypatch):
    p, _client = publication
    p.publish()
    original_manager = p.agent.task_manager
    original_close = p.agent._close_native_task_manager
    closed = []

    async def close(manager):
        result = await original_close(manager)
        closed.append(manager)
        return result

    monkeypatch.setattr(p.agent, "_close_native_task_manager", close)
    p.agent._native_mcp_publication = object()
    try:
        try:
            await p.agent.shutdown()
        except (AttributeError, TypeError):
            pass  # The original failed owner must not skip independent retirement.
        first_attempt_closed = original_manager in closed
    finally:
        p.agent._native_mcp_publication = p
        await p.close()
        await p.agent.shutdown()
    assert first_attempt_closed, "An invalid MCP owner skipped original TaskManager retirement before recovery"


async def test_original_sdk_resident_admission_is_retained_before_publication(publication):
    p, _client = publication
    admission = getattr(p, "_resident_operation", None)
    assert admission is not None and admission.task is p._resident, "Original SDK resident admission was not retained by its publication"


@contextmanager
def _refuse_original_mcp_close_admission():
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused = []
    original = RuntimeError("Owned original MCP closer factory acceptance unknown")

    def factory(loop, coroutine, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        inner = frame.f_locals.get("operation") if frame is not None else None
        candidate = inner if inspect.iscoroutine(inner) else coroutine
        code = getattr(candidate, "cr_code", None)
        if code is not None and code.co_qualname == "NativeMCPPublication.close.<locals>.drain":
            refused.append((coroutine, candidate))
            raise original
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    loop.set_task_factory(factory)
    try:
        yield refused
    finally:
        loop.set_task_factory(previous)


async def test_unresolved_native_mcp_close_admission_cannot_be_retried_as_absence(native_tool_effect_agent):
    # This is a private component allocation test. No READY, SDK lifespan,
    # protocol publication, or successful cleanup ACK is fabricated.
    agent = await native_tool_effect_agent(did="did:owned:mcp-close-admission")
    assert not agent._runtime_publication_ready.is_set()
    app = FastAPI()
    app.state.agent = agent
    policy = MCPServerPolicy.model_validate_json(json.dumps({"console_origin": "https://kestrel.example",
        "clients": [{"client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret"}]}))
    owner = NativeMCPPublication(app, agent, "OwnedAdmission", policy, sovereign_key=lambda: KEY)
    agent._native_mcp_publication = owner
    agent._resource_custody().acquire("native_mcp_server")
    with _refuse_original_mcp_close_admission() as refused, pytest.raises(RuntimeError) as opening:
        await owner.close()
    assert len(refused) == 1 and execution_cleanup_debts(opening.value)
    assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
    repeated = None
    try:
        await owner.close()
    except RuntimeError as error:
        repeated = error
    assert repeated is not None and execution_cleanup_debts(repeated), "Unresolved original MCP close admission was replaced by a successful fresh retry"
    assert agent._resource_custody().is_held("native_mcp_server"), "Unresolved MCP admission released primary storage custody"
    assert agent._native_mcp_publication is owner


async def test_missing_resident_task_with_real_admission_debt_retains_primary_custody(native_tool_effect_agent):
    agent = await native_tool_effect_agent(did="did:owned:mcp-resident-admission")
    assert not agent._runtime_publication_ready.is_set()
    app = FastAPI()
    app.state.agent = agent
    policy = MCPServerPolicy.model_validate_json(json.dumps({"console_origin": "https://kestrel.example",
        "clients": [{"client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret"}]}))
    owner = NativeMCPPublication(app, agent, "OwnedResidentAdmission", policy, sovereign_key=lambda: KEY)
    agent._native_mcp_publication = owner
    agent._resource_custody().acquire("native_mcp_server")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    refused = []

    async def unadmitted():
        raise AssertionError("Original unadmitted resident entered its body")

    operation = unadmitted()

    def factory(loop, coroutine, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        inner = frame.f_locals.get("operation") if frame is not None else None
        if inner is operation:
            refused.append(coroutine)
            raise RuntimeError("Owned resident task-factory acceptance is unknown")
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    try:
        loop.set_task_factory(factory)
        owner._resident_operation = OwnedOperation(operation, retirement_resource=owner, retain_originals=True)
    finally:
        loop.set_task_factory(previous)
    assert len(refused) == 1 and owner._resident is None
    assert execution_cleanup_debts(owner._resident_operation.admission_error)
    assert inspect.getcoroutinestate(operation) == inspect.CORO_CLOSED
    with pytest.raises(RuntimeError) as failure:
        await owner.close()
    assert execution_cleanup_debts(failure.value)
    assert agent._resource_custody().is_held("native_mcp_server"), "Missing resident task erased its actual admission debt"
    assert agent._native_mcp_publication is owner


async def test_sdk_resident_keeps_original_agent_runtime_without_borrowing_console_owner(publication, monkeypatch):
    p, client = publication
    await p.close()
    original = NativeMCPServer._lifespan
    witnesses, prepared = [], []

    @asynccontextmanager
    async def observe(runtime, app):
        witnesses.append((current_caller_context(), current_request_authentication()))
        async with original(runtime, app):
            yield

    monkeypatch.setattr(NativeMCPServer, "_lifespan", observe)

    @p.app.post("/api/owned-native-mcp-runtime")
    async def prepare_from_console(request: Request):
        service = p.app.state.request_authentication
        with service.bind(request) as authentication, caller_context_scope(authentication.caller):
            assert authentication.caller.is_sovereign
            owner = await NativeMCPPublication.prepare(p.app, p.agent, "OwnedRuntime", p.policy,
                sovereign_key=lambda: KEY)
            prepared.append(owner)
        return {"prepared": True}

    try:
        response = await client.post("/api/owned-native-mcp-runtime", headers={"x-api-key": KEY})
        assert response.status_code == 200, response.text
        assert witnesses == [(None, None)], "Original SDK resident borrowed its publishing Console credential lifetime"
    finally:
        for owner in prepared:
            await owner.close()


@asynccontextmanager
async def _publication_with_owned_process_custody(publication, monkeypatch):
    previous, client = publication
    await previous.close()
    scope = ExecutionCustody(ProcessRuntimeExecutionFence(previous.agent.did))
    with monkeypatch.context() as patch:
        patch.setattr(previous.agent, "_execution_custody", scope)
        owner = await NativeMCPPublication.prepare(previous.app, previous.agent, previous.routing_key,
            previous.policy, sovereign_key=lambda: KEY)
        owner.publish()
        try:
            yield owner, client, scope
        finally:
            await owner.close()


def _authorization_parameters(owner):
    return {"response_type": "code", "client_id": "owned-provider",
        "redirect_uri": "https://provider.example/callback", "scope": "memory:read",
        "resource": owner.resource, "state": "original-state", "code_challenge": "a" * 43,
        "code_challenge_method": "S256"}


async def test_native_oauth_delivery_rechecks_original_execution_custody(publication, monkeypatch):
    original = NativeMCPAuthorizationServer.authorization_endpoint
    completed = []

    async def retire_after_authorization(authorization, request):
        response = await original(authorization, request)
        assert response.status_code == 302
        completed.append(response.headers["location"])
        retire_process_runtime_custody(scope)
        return response

    monkeypatch.setattr(NativeMCPAuthorizationServer, "authorization_endpoint", retire_after_authorization)
    async with _publication_with_owned_process_custody(publication, monkeypatch) as (owner, client, scope):
        with pytest.raises(ExecutionAuthorityError, match="retired"):
            await client.get(owner.resource + "/oauth/authorize", params=_authorization_parameters(owner))
        assert len(completed) == 1, "The original OAuth operation did not reach the delivery boundary"
        handle = parse_qs(urlsplit(completed[0]).query)["request"][0]
        assert await owner.runtime.authorization.pending.read(handle) is not None


async def test_private_consent_delivery_preserves_execution_control(publication, monkeypatch):
    async with _publication_with_owned_process_custody(publication, monkeypatch) as (owner, client, scope):
        response = await client.get(owner.resource + "/oauth/authorize", params=_authorization_parameters(owner))
        assert response.status_code == 302
        handle = parse_qs(urlsplit(response.headers["location"]).query)["request"][0]
        original = owner.runtime.authorization.grants.read_authorization_request
        completed = []

        async def retire_after_read(value):
            pending = await original(value)
            assert pending is not None
            completed.append(value)
            retire_process_runtime_custody(scope)
            return pending

        monkeypatch.setattr(owner.runtime.authorization.grants, "read_authorization_request", retire_after_read)
        description = await client.get("/api/mcp/server/consent/requests/" + handle, headers={"x-api-key": KEY})
        assert completed == [handle]
        assert description.status_code == 500, "Private consent was delivered after original execution custody ended"
        assert description.json()["error"]["code"] == "execution_control", "Host authority discarded native execution-control evidence"
        assert "client_name" not in description.text


async def test_publication_rejects_replacement_execution_custody(publication, monkeypatch):
    async with _publication_with_owned_process_custody(publication, monkeypatch) as (owner, _client, _scope):
        replacement = ExecutionCustody(ProcessRuntimeExecutionFence(owner.agent.did))
        monkeypatch.setattr(owner.agent, "_execution_custody", replacement)
        with pytest.raises(HostCapabilityAccessDenied, match="runtime"):
            owner.require_current()


async def test_replacement_custody_does_not_hide_original_runtime_control(publication, monkeypatch):
    async with _publication_with_owned_process_custody(publication, monkeypatch) as (owner, _client, scope):
        retire_process_runtime_custody(scope)
        monkeypatch.setattr(owner.agent, "_execution_custody",
            ExecutionCustody(ProcessRuntimeExecutionFence(owner.agent.did)))
        with pytest.raises(ExecutionAuthorityError, match="retired"):
            owner.require_current()


async def test_publication_keeps_ambient_request_execution_restrictions(publication, monkeypatch):
    async with _publication_with_owned_process_custody(publication, monkeypatch) as (owner, _client, _scope):
        with bind_execution_custody(ProcessRuntimeExecutionFence(owner.agent.did)) as request:
            retire_process_runtime_custody(request)
            with pytest.raises(ExecutionAuthorityError, match="retired"):
                owner.require_current()


async def test_public_manager_removal_fences_routes_before_failed_protocol_cleanup(native_tool_effect_agent, tmp_path):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    # Real public manager removal and Agent.shutdown, with privately allocated
    # MCP custody. No READY or successful MCP publication is fabricated.
    agent = await native_tool_effect_agent(did="did:owned:mcp-public-remove")
    manager = AgentManager(base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False)
    async with manager.a2a_lifecycle_lease():
        manager._register_agent("OwnedNativeMCP", agent)
    app = FastAPI()
    app.state.agent_manager = manager
    policy = MCPServerPolicy.model_validate_json(json.dumps({"console_origin": "https://kestrel.example",
        "clients": [{"client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret"}]}))
    owner = NativeMCPPublication(app, agent, "OwnedNativeMCP", policy, sovereign_key=lambda: KEY, manager=manager)
    agent._native_mcp_publication = owner
    agent._resource_custody().acquire("native_mcp_server")
    with _refuse_original_mcp_close_admission() as refused, pytest.raises(RuntimeError) as failure:
        await manager.remove_agent("OwnedNativeMCP")
    assert len(refused) == 1 and execution_cleanup_debts(failure.value)
    assert all(inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED for pair in refused for coro in pair)
    assert agent._resource_custody().is_held("native_mcp_server")
    assert manager.get_agent("OwnedNativeMCP") is None, "Public remove left a retiring agent routable after closer admission failed"
    assert agent not in manager.list_agents().values(), "Public remove left a retiring agent discoverable"
    assert manager._get_agent_for_lifecycle("OwnedNativeMCP") is agent, "Public fencing discarded private cleanup ownership"


async def test_successful_public_removal_releases_original_route_fence(native_tool_effect_agent, tmp_path):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    manager = AgentManager(base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False)
    agent = await native_tool_effect_agent(did="did:owned:mcp-successful-removal")
    assert not agent._runtime_publication_ready.is_set()
    assert vars(agent).get("_native_mcp_publication") is None
    async with manager.a2a_lifecycle_lease():
        manager._register_agent("OwnedRetirement", agent)
    assert await manager.remove_agent("OwnedRetirement") is True
    assert manager.get_agent("OwnedRetirement") is None
    assert manager._get_agent_for_lifecycle("OwnedRetirement") is None
    assert all(value is not agent for value in manager._fenced_agent_routes.values()), (
        "Successful public removal permanently retained its retired agent in the routing fence"
    )


async def test_route_fence_release_requires_withdrawal_and_keeps_replacement_owner(native_tool_effect_agent, tmp_path):
    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    manager = AgentManager(base_data_dir=tmp_path, startup_runtime_env={}, startup_roster_enabled=False)
    original = await native_tool_effect_agent(did="did:owned:mcp-original-fence")
    replacement = await native_tool_effect_agent(did="did:owned:mcp-replacement-fence")
    async with manager.a2a_lifecycle_lease():
        manager._register_agent("OwnedFence", original)
    manager._fence_agent_route("OwnedFence", original)
    with pytest.raises(RuntimeError, match="Withdraw original public routing"):
        manager._clear_withdrawn_agent_route_fence("OwnedFence", original)
    assert manager.get_agent("OwnedFence") is None
    assert manager._get_agent_for_lifecycle("OwnedFence") is original
    assert await manager.remove_agent("OwnedFence") is True
    async with manager.a2a_lifecycle_lease():
        manager._register_agent("OwnedFence", replacement)
    manager._fence_agent_route("OwnedFence", replacement)
    manager._clear_withdrawn_agent_route_fence("OwnedFence", original)
    assert manager.get_agent("OwnedFence") is None, "An old owner cleared the replacement's live route fence"
    assert manager._get_agent_for_lifecycle("OwnedFence") is replacement
    assert await manager.remove_agent("OwnedFence") is True
    assert not manager._fenced_agent_routes


@pytest.mark.parametrize("storage_door", ["facade", "native-backend", "boot"])
async def test_deferred_shutdown_keeps_original_mcp_storage_dependency(native_tool_effect_agent, monkeypatch, storage_door):
    # This drives the real retained continuation and real native SQL owners.
    # The MCP allocation is private, with actual unknown admission debt; no
    # READY, SDK publication, dispatcher completion or release ACK is forged.
    agent = await native_tool_effect_agent(did="did:owned:mcp-deferred-storage")
    app = FastAPI()
    app.state.agent = agent
    policy = MCPServerPolicy.model_validate_json(json.dumps({"console_origin": "https://kestrel.example",
        "clients": [{"client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret"}]}))
    owner = NativeMCPPublication(app, agent, "OwnedDeferredMCP", policy, sovereign_key=lambda: KEY)
    agent._native_mcp_publication = owner
    agent._resource_custody().acquire("native_mcp_server")
    with _refuse_original_mcp_close_admission() as refused, pytest.raises(RuntimeError) as admission:
        await owner.close()
    assert len(refused) == 1 and execution_cleanup_debts(admission.value)
    if storage_door == "native-backend":
        agent.storage = owner.backend
    storage = agent._raw_storage if storage_door == "boot" else agent.storage
    original_close = storage.close
    precloses, closes = [], []

    async def observe_close():
        closes.append(storage)
        return await original_close()

    monkeypatch.setattr(storage, "close", observe_close)
    if storage_door == "facade":
        original_preclose = storage.dispose_cached_sqla_factory

        async def observe_preclose():
            precloses.append(storage)
            return await original_preclose()

        monkeypatch.setattr(storage, "dispose_cached_sqla_factory", observe_preclose)
    continuation = (asyncio.create_task(agent._boot_teardown_storage()) if storage_door == "boot"
        else await agent._ensure_durable_shutdown_continuation(None))
    outcome = await await_owned_task(continuation)
    assert not precloses, "Deferred shutdown disposed borrowed storage before original MCP retirement"
    assert not closes, "Deferred shutdown closed borrowed primary storage before original MCP retirement"
    assert outcome.error is not None and execution_cleanup_debts(outcome.error), (
        "Deferred shutdown discarded original MCP admission debt instead of retaining its dependency"
    )
    assert agent._resource_custody().is_held("native_mcp_server")
    assert agent._native_mcp_publication is owner
    assert await owner.backend.fetch_val("SELECT 1") == 1


async def test_deferred_shutdown_refuses_a_still_held_native_task_manager(native_tool_effect_agent, monkeypatch):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.a2a.task_manager import TaskManager
    from kestrel_sovereign.features.storage_access import resolve_feature_database

    agent = await native_tool_effect_agent(did="did:owned:deferred-task-manager")
    backend = resolve_feature_database(agent).backend
    manager = TaskManager.from_backend(backend, scope=NativeRecipientScope(agent.did), owns_backend=False)
    await manager.initialize()
    agent.task_manager = manager
    agent._resource_custody().acquire("task_manager")
    storage = agent.storage
    original_preclose, original_close = storage.dispose_cached_sqla_factory, storage.close
    precloses, closes = [], []

    async def preclose():
        precloses.append(storage)
        return await original_preclose()

    async def close():
        closes.append(storage)
        return await original_close()

    monkeypatch.setattr(storage, "dispose_cached_sqla_factory", preclose)
    monkeypatch.setattr(storage, "close", close)
    try:
        continuation = await agent._ensure_durable_shutdown_continuation(None)
        outcome = await await_owned_task(continuation)
        assert not precloses and not closes, "Deferred shutdown bypassed another original native storage borrower"
        assert isinstance(outcome.error, RuntimeError)
        assert agent._resource_custody().is_held("task_manager")
        assert await backend.fetch_val("SELECT 1") == 1
    finally:
        await agent._resource_custody().release("task_manager", agent._boot_teardown_task_manager)


async def test_deferred_shutdown_joins_the_original_mcp_close_before_storage(native_tool_effect_agent, monkeypatch):
    agent = await native_tool_effect_agent(did="did:owned:mcp-deferred-join")
    app = FastAPI()
    app.state.agent = agent
    policy = MCPServerPolicy.model_validate_json(json.dumps({"console_origin": "https://kestrel.example",
        "clients": [{"client_id": "owned-provider", "client_name": "Owned provider",
            "redirect_uris": ["https://provider.example/callback"], "client_secret": "owned-secret"}]}))
    owner = NativeMCPPublication(app, agent, "OwnedDeferredJoin", policy, sovereign_key=lambda: KEY)
    agent._native_mcp_publication = owner
    agent._resource_custody().acquire("native_mcp_server")
    original_close = owner.close
    entered, rejoined = asyncio.Event(), asyncio.Event()
    calls = []

    async def observe_close():
        calls.append(owner)
        (entered if len(calls) == 1 else rejoined).set()
        return await original_close()

    monkeypatch.setattr(owner, "close", observe_close)
    async with owner._requests.read():
        first = asyncio.create_task(owner.close())
        await asyncio.wait_for(entered.wait(), 5)
        original_operation = owner._close_operation
        assert original_operation is not None
        continuation = await agent._ensure_durable_shutdown_continuation(None)
        await asyncio.wait_for(rejoined.wait(), 5)
        assert owner._close_operation is original_operation
        assert not continuation.done() and not original_operation.task.done()
        assert await owner.backend.fetch_val("SELECT 1") == 1
    await first
    outcome = await await_owned_task(continuation)
    assert outcome.error is outcome.cancellation is None
    assert not agent._resource_custody().is_held("native_mcp_server")
    assert original_operation.task.done()
