"""Real SDK client/HTTP bindings exercise recipient and tenant policy.

These tests cover the native transport boundary, not Core's task lifecycle,
signed cards, or the complete A2A conformance suite.
"""

import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import rfc8785
from a2a.client import ClientConfig, ClientFactory
from a2a.client.client import ClientCallContext
from a2a.client.interceptors import ClientCallInterceptor
from a2a.server.request_handlers import RequestHandler
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.rest_routes import create_rest_routes
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.errors import InvalidParamsError, UnsupportedOperationError
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette
from starlette.requests import Request

from kestrel_sovereign.a2a.native_context import (
    NativeRecipientBinding,
    NativeServerCallContextBuilder,
    bind_native_request,
)
from kestrel_sovereign.a2a.scope import NativeRecipientScope


def _digest(request):
    return hashlib.sha256(rfc8785.dumps(MessageToDict(request))).hexdigest()


class _DigestInterceptor(ClientCallInterceptor):
    async def before(self, args):
        args.context.service_parameters["X-Test-Native-Digest"] = _digest(args.input)

    async def after(self, args):
        pass


class _TenantPeer(RequestHandler):
    def __init__(self, tenant):
        self.tenant = tenant
        self.recipient_id = "did:test:recipient"
        self.routing_key = "registered-agent"
        self.observed = []

    async def _receive(self, params, context):
        before = MessageToDict(params)
        resolved = await bind_native_request(
            params,
            context,
            routing_key=self.routing_key,
            scope=NativeRecipientScope(self.recipient_id, self.tenant),
        )
        # Validation returns a distinct native object; the SDK's parsed input
        # remains unchanged, including REST's transport-carried tenant.
        assert resolved is not params and MessageToDict(params) == before
        expected = context.state["headers"].get("x-test-native-digest")
        if expected is not None:
            assert _digest(resolved) == expected
        self.observed.append(MessageToDict(resolved))
        return Task(
            id="known-task",
            context_id="known-context",
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        )

    async def on_get_task(self, params, context):
        return await self._receive(params, context)

    async def on_message_send(self, params, context):
        return await self._receive(params, context)

    async def _unsupported(self, params, context):
        raise UnsupportedOperationError()

    on_list_tasks = _unsupported
    on_cancel_task = _unsupported
    on_message_send_stream = _unsupported
    on_create_task_push_notification_config = _unsupported
    on_get_task_push_notification_config = _unsupported
    on_subscribe_to_task = _unsupported
    on_list_task_push_notification_configs = _unsupported
    on_delete_task_push_notification_config = _unsupported
    on_get_extended_agent_card = _unsupported


@asynccontextmanager
async def _peer(protocol, tenant="tenant-alpha"):
    receiver = _TenantPeer(tenant)
    binding = NativeRecipientBinding(
        receiver.routing_key, receiver.recipient_id, tenant, protocol
    )
    builder = NativeServerCallContextBuilder(binding)
    routes = (
        create_jsonrpc_routes(
            receiver, rpc_url="/", context_builder=builder, enable_v0_3_compat=False
        )
        if protocol == "JSONRPC"
        else create_rest_routes(
            receiver, context_builder=builder, enable_v0_3_compat=False
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(Starlette(routes=routes)),
        base_url="http://native-a2a.test",
        headers={"A2A-Version": "1.0"},
    ) as http:
        yield SimpleNamespace(
            http=http, receiver=receiver, protocol=protocol, tenant=tenant
        )


@pytest.mark.parametrize("protocol", ["JSONRPC", "HTTP+JSON"])
@pytest.mark.parametrize("tenant", ["", "tenant-alpha"])
async def test_native_clients_and_sdk_routes_bind_complete_requests(protocol, tenant):
    async with _peer(protocol, tenant) as peer:
        card = AgentCard(
            name="Native tenant test",
            description="SDK transport boundary",
            version="1",
            supported_interfaces=[
                AgentInterface(
                    url=str(peer.http.base_url),
                    protocol_binding=protocol,
                    protocol_version="1.0",
                    tenant=tenant,
                )
            ],
            capabilities=AgentCapabilities(streaming=False),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[
                AgentSkill(
                    id="echo",
                    name="echo",
                    description="SDK transport boundary",
                    tags=["test"],
                )
            ],
        )
        client = ClientFactory(
            ClientConfig(
                httpx_client=peer.http,
                streaming=False,
                supported_protocol_bindings=[protocol],
            )
        ).create(card, interceptors=[_DigestInterceptor()])
        request = GetTaskRequest(id="known-task", tenant=tenant, history_length=0)
        task = await client.get_task(
            request, context=ClientCallContext(service_parameters={})
        )
        assert task.id == "known-task" and request.tenant == tenant
        message = SendMessageRequest(
            tenant=tenant,
            message=Message(
                message_id="input-message",
                role=Role.ROLE_USER,
                parts=[Part(text="hello")],
            ),
        )
        events = [
            event
            async for event in client.send_message(
                message, context=ClientCallContext(service_parameters={})
            )
        ]
        assert len(events) == 1 and events[0].task.id == "known-task"
        assert len(peer.receiver.observed) == 2
        assert peer.receiver.observed[0].get("tenant", "") == tenant
        assert peer.receiver.observed[0]["historyLength"] == 0
        assert peer.receiver.observed[1]["message"]["messageId"] == "input-message"


def _send_payload(tenant):
    return {
        "tenant": tenant,
        "message": {
            "messageId": "input-message",
            "role": "ROLE_USER",
            "parts": [{"text": "hello"}],
        },
    }


def _assert_rejected(response, peer, *, diagnostic="tenant"):
    if peer.protocol == "JSONRPC":
        assert response.status_code == 200
        assert response.json()["error"]["code"] == -32602
    else:
        assert response.status_code == 400, response.text
    assert diagnostic in json.dumps(response.json()).lower()
    assert peer.receiver.observed == []


def _direct_context(receive, *, protocol="JSONRPC", path_tenant=None):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [],
        "query_string": b"",
    }
    if path_tenant is not None:
        scope["path_params"] = {"tenant": path_tenant}
    request = Request(scope, receive=receive)
    return NativeServerCallContextBuilder(
        NativeRecipientBinding("registered-agent", "did:test:recipient", "", protocol)
    ).build(request)


@pytest.mark.parametrize("protocol", ["JSONRPC", "HTTP+JSON"])
async def test_foreign_tenant_cannot_select_another_recipient(protocol):
    async with _peer(protocol) as peer:
        params = _send_payload("foreign")
        response = (
            await peer.http.post(
                "/",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "SendMessage",
                    "params": params,
                },
            )
            if protocol == "JSONRPC"
            else await peer.http.post("/foreign/message:send", json=params)
        )
        _assert_rejected(response, peer)


@pytest.mark.parametrize("body_tenant", ["foreign", "", None])
async def test_rest_body_cannot_disagree_with_the_path_tenant(body_tenant):
    async with _peer("HTTP+JSON") as peer:
        response = await peer.http.post(
            "/tenant-alpha/message:send", json=_send_payload(body_tenant)
        )
        _assert_rejected(response, peer)


async def test_rest_bare_route_cannot_use_a_body_tenant_to_select_the_recipient():
    async with _peer("HTTP+JSON") as peer:
        response = await peer.http.post(
            "/message:send", json=_send_payload("tenant-alpha")
        )
        _assert_rejected(response, peer)


async def test_rest_path_supplies_a_tenant_absent_from_the_json_body():
    async with _peer("HTTP+JSON") as peer:
        params = _send_payload("tenant-alpha")
        del params["tenant"]
        response = await peer.http.post("/tenant-alpha/message:send", json=params)
        assert response.status_code == 200, response.text
        assert peer.receiver.observed[0]["tenant"] == "tenant-alpha"


@pytest.mark.parametrize(
    "queries",
    [[("tenant", "foreign")], [("tenant", "tenant-alpha"), ("tenant", "tenant-alpha")]],
)
async def test_rest_read_rejects_foreign_and_repeated_query_tenants(queries):
    async with _peer("HTTP+JSON") as peer:
        response = await peer.http.get("/tenant-alpha/tasks/known-task", params=queries)
        _assert_rejected(response, peer)


@pytest.mark.parametrize("protocol", ["JSONRPC", "HTTP+JSON"])
@pytest.mark.parametrize("identity", ["recipient_id", "routing_key"])
async def test_stale_recipient_binding_fails_before_any_handler_effect(
    protocol, identity
):
    async with _peer(protocol) as peer:
        setattr(peer.receiver, identity, "changed-registration")
        params = _send_payload("tenant-alpha")
        response = (
            await peer.http.post(
                "/",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "SendMessage",
                    "params": params,
                },
            )
            if protocol == "JSONRPC"
            else await peer.http.post("/tenant-alpha/message:send", json=params)
        )
        _assert_rejected(response, peer, diagnostic="recipient")


@pytest.mark.parametrize("raw", [b"{", b"\xff", b"[]"])
async def test_unparsed_rest_read_bodies_cannot_bypass_tenant_validation(raw):
    async with _peer("HTTP+JSON") as peer:
        response = await peer.http.request(
            "GET", "/tenant-alpha/tasks/known-task", content=raw
        )
        _assert_rejected(
            response, peer, diagnostic="tenant" if raw != b"[]" else "object"
        )


@pytest.mark.parametrize("protocol", ["JSONRPC", "HTTP+JSON"])
async def test_protojson_null_tenant_represents_the_advertised_default(protocol):
    async with _peer(protocol, "") as peer:
        params = _send_payload(None)
        response = (
            await peer.http.post(
                "/",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "SendMessage",
                    "params": params,
                },
            )
            if protocol == "JSONRPC"
            else await peer.http.post("/message:send", json=params)
        )
        assert response.status_code == 200, response.text
        assert len(peer.receiver.observed) == 1
        assert "tenant" not in peer.receiver.observed[0]


async def test_invalid_required_native_fields_are_rejected_before_handler_effects():
    async with _peer("JSONRPC") as peer:
        response = await peer.http.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "GetTask",
                "params": {"tenant": "tenant-alpha", "id": ""},
            },
        )
        _assert_rejected(response, peer, diagnostic="validation")


async def test_missing_host_context_cannot_be_reconstructed_from_untrusted_parameters():
    async def forbidden_receive():
        raise AssertionError("unbound request reached HTTP I/O")

    context = _direct_context(forbidden_receive)
    context.state.clear()
    with pytest.raises(RuntimeError, match="host-owned HTTP context"):
        await bind_native_request(
            GetTaskRequest(id="known-task"),
            context,
            routing_key="registered-agent",
            scope=NativeRecipientScope("did:test:recipient"),
        )


@pytest.mark.parametrize("parameters", [object, Message])
async def test_nonnative_request_types_fail_before_reading_the_http_body(parameters):
    async def forbidden_receive():
        raise AssertionError("invalid request type reached HTTP I/O")

    context = _direct_context(forbidden_receive)
    with pytest.raises(TypeError, match="protobuf request|tenant field"):
        await bind_native_request(
            parameters(),
            context,
            routing_key="registered-agent",
            scope=NativeRecipientScope("did:test:recipient"),
        )


async def test_inconsistent_sdk_context_cannot_reach_http_body_or_authorization():
    async def forbidden_receive():
        raise AssertionError("inconsistent SDK tenant reached HTTP I/O")

    context = _direct_context(forbidden_receive)
    context.tenant = "foreign"
    with pytest.raises(InvalidParamsError, match="tenant"):
        await bind_native_request(
            GetTaskRequest(id="known-task"),
            context,
            routing_key="registered-agent",
            scope=NativeRecipientScope("did:test:recipient"),
        )


@pytest.mark.parametrize("source", ["path", "protobuf"])
async def test_rest_context_cannot_hide_a_conflicting_path_or_native_tenant(source):
    async def empty_body():
        return {"type": "http.request", "body": b"", "more_body": False}

    context = _direct_context(
        empty_body,
        protocol="HTTP+JSON",
        path_tenant="foreign" if source == "path" else "",
    )
    params = GetTaskRequest(
        id="known-task", tenant="foreign" if source == "protobuf" else ""
    )
    with pytest.raises(InvalidParamsError, match="tenant"):
        await bind_native_request(
            params,
            context,
            routing_key="registered-agent",
            scope=NativeRecipientScope("did:test:recipient"),
        )


@pytest.mark.parametrize(
    "tenant", [".", "..", "team/other", "team%2Fother", "with spaces", None]
)
def test_nonopaque_tenant_configuration_is_rejected_at_registration(tenant):
    with pytest.raises(ValueError, match="URL-safe opaque"):
        NativeRecipientBinding(
            "registered-agent", "did:test:recipient", tenant, "HTTP+JSON"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("routing_key", ""),
        ("recipient_id", ""),
        ("recipient_id", None),
        ("protocol_binding", "GRPC"),
    ],
)
def test_missing_identity_or_unsupported_binding_fails_at_registration(field, value):
    values = {
        "routing_key": "registered-agent",
        "recipient_id": "did:test:recipient",
        "tenant": "",
        "protocol_binding": "JSONRPC",
    }
    values[field] = value
    with pytest.raises(ValueError):
        NativeRecipientBinding(**values)
