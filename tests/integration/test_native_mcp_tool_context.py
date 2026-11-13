"""Actual official SDK tool callbacks retain their native HTTP/auth context."""

import asyncio
import json
from copy import copy
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.mcpserver import Context
from mcp.shared.exceptions import MCPError
from starlette.requests import Request

from kestrel_sovereign.auth import AuthMethod, current_caller_context
from kestrel_sovereign.mcp_server.policy import MCPServerPolicy
from kestrel_sovereign.mcp_server.publication import NativeMCPPublication
from kestrel_sovereign.mcp_server.tool_caller import (
    _REQUEST_KEY,
    bind_native_mcp_tool_caller,
)
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.request_auth import current_request_authentication
from tests.integration.test_native_mcp_oauth_http import CHALLENGE, VERIFIER
from tests.integration.test_native_mcp_publication import (
    KEY,
)
from tests.integration.test_native_mcp_publication import (
    assistant_agent as _assistant_agent,
)
from tests.integration.test_native_mcp_publication import (
    publication as _publication,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
assistant_agent = _assistant_agent
publication = _publication


async def issue_native_tokens(owner, http, client_id="owned-provider", scope="memory:read"):
    response = await http.get(owner.resource + "/oauth/authorize", params={
        "response_type": "code", "client_id": client_id,
        "redirect_uri": "https://provider.example/callback", "scope": scope,
        "resource": owner.resource, "state": "original-tool-context-state",
        "code_challenge": CHALLENGE, "code_challenge_method": "S256"})
    assert response.status_code == 302, response.text
    handle = parse_qs(urlsplit(response.headers["location"]).query)["request"][0]
    decision = await http.post("/api/mcp/server/consent/requests/" + handle + "/decision",
        headers={"x-api-key": KEY}, json={"decision": "approve", "scopes": scope.split(" ")})
    assert decision.status_code == 200, decision.text
    code = parse_qs(urlsplit(decision.json()["redirect_url"]).query)["code"][0]
    response = await http.post(owner.resource + "/oauth/token", data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": "https://provider.example/callback", "client_id": client_id,
        "client_secret": "owned-secret", "code_verifier": VERIFIER, "resource": owner.resource})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("required_scope,substitution", [
    ("memory:read", None), ("conversation:write", None), ("memory:read", "copied-owner"),
    ("memory:read", "changed-native-token"), ("memory:read", "copied-request"),
])
async def test_original_official_sdk_request_and_bearer_reach_actual_tool_callback(publication, required_scope, substitution):
    owner, http = publication
    owner.publish()
    resource_path = urlsplit(owner.resource).path
    original_requests, calls, bound_callers, refused, children = [], [], [], [], []
    release = asyncio.Event()
    original_owners = []
    initial_turns = []
    delivery_order = []
    marker_key = "owned.native_mcp.request_marker"

    class OriginalRequestMarker:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http" and scope["path"] == resource_path:
                marker = object()
                scope[marker_key] = marker
                original_requests.append((scope, marker))
            async def original_send(message):
                if message["type"] == "http.response.start":
                    delivery_order.append((scope, "response-start"))
                await send(message)

            await self.app(scope, receive, original_send)

    owner.app.add_middleware(OriginalRequestMarker)

    async def original_context(context: Context) -> dict[str, bool]:
        request = context.request_context.request
        token = get_access_token()
        calls.append((request, token, current_caller_context(), current_request_authentication()))
        original_owner = request.scope[_REQUEST_KEY]
        initial_turns.append((owner.agent.get_current_turn_id(), owner.agent._caller_belongs_to_live_turn()))
        original_owners.append(original_owner)
        original_client = token.client_id
        if substitution == "copied-owner":
            request.scope[_REQUEST_KEY] = copy(original_owner)
        elif substitution == "changed-native-token":
            token.client_id = "forged-native-client"
        elif substitution == "copied-request":
            context.request_context.request = Request(dict(request.scope))
        try:
            async with bind_native_mcp_tool_caller(owner, context, required_scope) as caller:
                initial_turns.append((owner.agent.get_current_turn_id(), owner.agent._caller_belongs_to_live_turn()))
                assert current_caller_context() is caller
                bound_callers.append(caller)

                async def copied_caller():
                    await release.wait()
                    return current_caller_context()

                children.append(asyncio.create_task(copied_caller()))
        except HostCapabilityAccessDenied as error:
            refused.append(error)
        finally:
            request.scope[_REQUEST_KEY] = original_owner
            token.client_id = original_client
            context.request_context.request = request
        delivery_order.append((request.scope, "tool-return"))
        return {"original_request": isinstance(request, Request)
            and any(request.scope is scope and request.scope.get(marker_key) is marker
                for scope, marker in original_requests), "native_token": token is not None}

    owner.runtime.server.add_tool(original_context, name="probe_original_native_context")
    tokens = await issue_native_tokens(owner, http)
    secret = tokens["access_token"]
    http.headers.update({"authorization": "Bearer " + secret, "x-api-key": KEY})
    async with Client(streamable_http_client(owner.resource, http_client=http), mode="2026-07-28") as client:
        result = await client.call_tool("probe_original_native_context", {})
        assert not result.is_error
        assert result.structured_content == {"original_request": True, "native_token": True}
    release.set()
    copied = await asyncio.gather(*children)
    if required_scope == "memory:read" and substitution is None:
        assert len(bound_callers) == 1 and not refused
        assert initial_turns == [(None, False), (None, False)], (
            "Native MCP initial caller publication captured or re-presented a prior governed turn")
        bound = bound_callers[0]
        assert bound.auth_method is AuthMethod.MCP_OAUTH and not bound.is_sovereign
        assert bound.identity.startswith("mcp:") and len(bound.identity) == 68
        assert secret not in bound.identity and KEY not in bound.identity
    else:
        assert len(refused) == 1 and not bound_callers, "Unverified or unconsented MCP request acquired a Core caller"
        assert initial_turns == [(None, False)]
    assert copied == ([None] if required_scope == "memory:read" and substitution is None else []), (
        "Native OAuth caller outlived its original HTTP request")
    assert not owner._tool_callers, "Completed HTTP request retained native OAuth caller authority"
    assert len(original_owners) == 1 and not original_owners[0].active
    with pytest.raises(HostCapabilityAccessDenied, match="already bound"):
        await original_owners[0].__aenter__()
    assert len(calls) == 1
    request, token, caller, interactive = calls[0]
    assert token.token == secret and token.client_id == "owned-provider"
    assert token.resource == owner.resource and token.claims["iss"] == owner.resource + "/oauth"
    assert token.scopes == ["memory:read"]
    assert caller is None, "The native SDK resident inherited a publishing Console caller"
    assert interactive is None, "The official native SDK callback acquired Console owner credentials"
    assert any(request.scope is scope for scope, _marker in original_requests)
    assert [event for scope, event in delivery_order if scope is request.scope] == [
        "tool-return", "response-start",
    ], "Native JSON response began delivery before its original tool callback returned"


async def test_native_principal_survives_refresh_and_sdk_restart_but_separates_clients(publication):
    previous, http = publication
    await previous.close()
    configuration = previous.policy.model_dump(mode="json")
    original = configuration["clients"][0]
    configuration["clients"] = [{**original, "client_id": client_id, "client_secret": "owned-secret"}
        for client_id in ("owned-provider", "second-provider")]
    policy = MCPServerPolicy.model_validate_json(json.dumps(configuration))
    callers = []
    owner = await NativeMCPPublication.prepare(previous.app, previous.agent, previous.routing_key,
        policy, sovereign_key=lambda: KEY)

    async def probe_caller(context: Context) -> dict[str, bool]:
        async with bind_native_mcp_tool_caller(owner, context, "memory:read") as caller:
            callers.append(caller)
        return {"bound": True}

    async def call(access):
        async with (
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=owner.app),
                base_url=policy.console_origin,
                headers={"authorization": "Bearer " + access, "x-api-key": KEY}) as transport,
            Client(streamable_http_client(owner.resource, http_client=transport), mode="2026-07-28") as client,
        ):
            result = await client.call_tool("probe_stable_native_caller", {})
            assert not result.is_error and result.structured_content == {"bound": True}

    try:
        owner.runtime.server.add_tool(probe_caller, name="probe_stable_native_caller")
        owner.publish()
        first = await issue_native_tokens(owner, http)
        await call(first["access_token"])
        response = await http.post(owner.resource + "/oauth/token", data={
            "grant_type": "refresh_token", "refresh_token": first["refresh_token"],
            "client_id": "owned-provider", "client_secret": "owned-secret", "resource": owner.resource})
        assert response.status_code == 200, response.text
        refreshed = response.json()
        assert refreshed["access_token"] != first["access_token"]
        await call(refreshed["access_token"])
        second = await issue_native_tokens(owner, http, "second-provider")
        await call(second["access_token"])
        assert callers[0] == callers[1], "OAuth refresh changed the stable governed-tool principal"
        assert callers[0].identity != callers[2].identity, "Distinct registered OAuth clients shared a Core principal"
        await owner.close()
        owner = await NativeMCPPublication.prepare(previous.app, previous.agent, previous.routing_key,
            policy, sovereign_key=lambda: KEY)
        owner.runtime.server.add_tool(probe_caller, name="probe_stable_native_caller")
        owner.publish()
        await call(second["access_token"])
        assert callers[2] == callers[3], "SDK publication restart changed the durable OAuth principal"
    finally:
        await owner.close()


@pytest.mark.parametrize("assistant_agent", [{"features": ["MemoryFeature"]}], indirect=True)
@pytest.mark.parametrize("execution", ["probe", "canonical"])
async def test_native_raw_result_holds_privacy_until_actual_body_delivery(publication, execution):
    owner, http = publication
    private = "owned-private-native-memory-result"
    body_entered, release = asyncio.Event(), asyncio.Event()
    gate = owner.agent._get_durable_persistence_gate()
    callback_finished = []

    class BlockOriginalBody:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            async def physical_send(message):
                if message["type"] == "http.response.body" and private.encode() in message.get("body", b""):
                    body_entered.set()
                    await release.wait()
                await send(message)
            await self.app(scope, receive, physical_send)

    owner.app.add_middleware(BlockOriginalBody)

    async def raw_result(context: Context) -> dict[str, str]:
        async with bind_native_mcp_tool_caller(owner, context, "memory:read"):
            request = context.request_context.request.scope[_REQUEST_KEY]
            async with request.disclosure():
                assert not gate.locked(), "Native disclosure inverted the leaf gate before canonical execution"
                callback_finished.append(request)
                return {"memory": private}

    if execution == "probe":
        owner.runtime.server.add_tool(raw_result, name="probe_private_memory_delivery")
    else:
        await owner.agent.privacy_agent.add_conversation("user", private, session_id="owned-physical-delivery")
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    http.headers.update({"authorization": "Bearer " + token, "x-api-key": KEY})

    async def call():
        async with Client(streamable_http_client(owner.resource, http_client=http), mode="2026-07-28") as client:
            if execution == "canonical":
                return await client.call_tool("recall_recent", {"limit": 5,
                    "operation_id": uuid4().hex, "conversation_id": "owned-physical-delivery"})
            return await client.call_tool("probe_private_memory_delivery", {})

    original_call = asyncio.create_task(call())
    entered = asyncio.create_task(body_entered.wait())
    transition = observer = None
    try:
        await asyncio.wait({entered, original_call}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        assert body_entered.is_set(), "Original native memory result did not reach actual body delivery"
        transition = asyncio.create_task(owner.agent.set_privacy_mode(PrivacyMode.EPHEMERAL))

        async def queued_transition():
            while not gate._lock._waiting_writers:
                await asyncio.sleep(0)

        observer = asyncio.create_task(queued_transition())
        await asyncio.wait({observer, transition}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        protected = gate.locked() and not transition.done() and gate._lock._waiting_writers > 0
        release.set()
        result, transition_result = await asyncio.gather(original_call, transition, return_exceptions=True)
    finally:
        release.set()
        for task in (entered, observer):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(original_call, *([transition] if transition is not None else []), return_exceptions=True)
    assert protected, "Privacy transition landed before original native private bytes were physically delivered"
    assert not isinstance(result, BaseException) and not isinstance(transition_result, BaseException)
    assert not result.is_error and private in result.model_dump_json()
    if execution == "probe":
        assert len(callback_finished) == 1 and result.structured_content == {"memory": private}
    else:
        assert await owner.agent._raw_storage.db.fetchall("SELECT state FROM tool_effect_claims") == [("RETURNED_OK",)]
    assert owner.agent.privacy_mode is PrivacyMode.EPHEMERAL
    assert not gate.locked(), "Original native result retained its privacy lease after physical delivery"


@pytest.mark.parametrize("withdrawal", ["privacy", "privacy-round-trip", "grant", "expiry"])
async def test_native_raw_result_is_refused_after_original_privacy_epoch_changes(publication, monkeypatch, withdrawal):
    owner, http = publication
    private = "owned-private-native-result-before-policy-change"
    produced = []

    async def raw_result(context: Context) -> dict[str, str]:
        async with bind_native_mcp_tool_caller(owner, context, "memory:read"):
            request = context.request_context.request.scope[_REQUEST_KEY]
            async with request.disclosure():
                produced.append(request)
        if withdrawal in {"privacy", "privacy-round-trip"}:
            await owner.agent.set_privacy_mode(PrivacyMode.EPHEMERAL)
            if withdrawal == "privacy-round-trip":
                await owner.agent.set_privacy_mode(PrivacyMode.NORMAL)
                assert owner.agent.privacy_mode is PrivacyMode.NORMAL
        elif withdrawal == "expiry":
            expires = get_access_token().expires_at
            assert expires is not None
            monkeypatch.setattr(owner.runtime.authorization.pending, "clock", lambda: expires + 1)
        else:
            response = await http.post(owner.resource + "/oauth/revoke", data={
                "token": get_access_token().token, "client_id": "owned-provider",
                "client_secret": "owned-secret"})
            assert response.status_code == 200, response.text
        return {"memory": private}

    owner.runtime.server.add_tool(raw_result, name="probe_private_epoch_change")
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    responses = []

    async def observe_original_response(response):
        responses.append((response.status_code, await response.aread()))

    failure = None
    try:
        async with (
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=owner.app, raise_app_exceptions=False),
                base_url=owner.policy.console_origin, headers={"authorization": "Bearer " + token},
                event_hooks={"response": [observe_original_response]}) as transport,
            Client(streamable_http_client(owner.resource, http_client=transport), mode="2026-07-28") as client,
        ):
            await client.call_tool("probe_private_epoch_change", {})
    except (httpx2.HTTPError, MCPError, ExceptionGroup) as error:
        failure = error
    assert len(produced) == 1, "The actual native SDK callback did not produce its original result"
    assert failure is not None and any(status == 500 for status, _body in responses), (
        "A native memory result crossed its original privacy-policy epoch")
    assert all(private.encode() not in body for _status, body in responses), (
        "Original native HTTP error delivery leaked the withdrawn memory result")


async def test_write_only_native_grant_cannot_disclose_memory(publication):
    previous, http = publication
    await previous.close()
    configuration = previous.policy.model_dump(mode="json")
    configuration["clients"][0].update(client_secret="owned-secret", scopes=["conversation:write"])
    policy = MCPServerPolicy.model_validate_json(json.dumps(configuration))
    owner = await NativeMCPPublication.prepare(previous.app, previous.agent, previous.routing_key,
        policy, sovereign_key=lambda: KEY)
    produced = []
    private = "owned-memory-not-authorized-by-write-scope"

    async def raw_result(context: Context) -> dict[str, str]:
        async with bind_native_mcp_tool_caller(owner, context, "conversation:write"):
            request = context.request_context.request.scope[_REQUEST_KEY]
            async with request.disclosure():
                produced.append(request)
                return {"memory": private}

    try:
        owner.runtime.server.add_tool(raw_result, name="probe_write_scope_disclosure")
        owner.publish()
        token = (await issue_native_tokens(owner, http, scope="conversation:write"))["access_token"]
        http.headers.update({"authorization": "Bearer " + token})
        async with Client(streamable_http_client(owner.resource, http_client=http), mode="2026-07-28") as client:
            result = await client.call_tool("probe_write_scope_disclosure", {})
        assert result.is_error and not produced, "A write-only native OAuth grant disclosed stored memory"
        assert private not in result.model_dump_json()
    finally:
        await owner.close()


@pytest.mark.parametrize("publication", [{"operation_seconds": 0.1}], indirect=True)
async def test_native_private_delivery_expires_without_abandoning_its_original_owner(publication):
    owner, http = publication
    private = "owned-native-memory-with-bounded-physical-delivery"
    entered, release = asyncio.Event(), asyncio.Event()
    requests, delivered = [], []
    gate = owner.agent._get_durable_persistence_gate()

    class BlockOriginalBody:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            async def physical_send(message):
                if message["type"] == "http.response.body" and private.encode() in message.get("body", b""):
                    entered.set()
                    await release.wait()
                    delivered.append(message)
                await send(message)

            await self.app(scope, receive, physical_send)

    async def raw_result(context: Context) -> dict[str, str]:
        async with bind_native_mcp_tool_caller(owner, context, "memory:read"):
            request = context.request_context.request.scope[_REQUEST_KEY]
            async with request.disclosure():
                requests.append(request)
                return {"memory": private}

    owner.app.add_middleware(BlockOriginalBody)
    owner.runtime.server.add_tool(raw_result, name="probe_bounded_private_delivery")
    owner.publish()
    token = (await issue_native_tokens(owner, http))["access_token"]
    http.headers.update({"authorization": "Bearer " + token})

    async def call():
        async with Client(streamable_http_client(owner.resource, http_client=http), mode="2026-07-28") as client:
            return await client.call_tool("probe_bounded_private_delivery", {})

    original = asyncio.create_task(call())
    observer = asyncio.create_task(entered.wait())
    try:
        await asyncio.wait({original, observer}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        reached_body = entered.is_set()
        await asyncio.wait({original}, timeout=5)
        completed = original.done()
        retired = len(requests) == 1 and not requests[0].active and not owner._tool_callers
        unlocked = not gate.locked()
        escaped = bool(delivered)
    finally:
        release.set()
        observer.cancel()
        results = await asyncio.gather(original, observer, return_exceptions=True)
    assert reached_body, "Original bounded native result never reached physical body delivery"
    assert completed, "Native private HTTP delivery ignored its configured operation deadline"
    assert retired and unlocked, "Expired native delivery retained its original caller or privacy lease"
    assert not escaped, "Expired native memory bytes escaped the blocked physical send"
    assert isinstance(results[0], BaseException), "Expired native memory delivery reported successful completion"
