"""Bind official A2A requests to the host-selected recipient and tenant.

The SDK carries REST path tenants in ServerCallContext, while JSON-RPC
carries them in the protobuf request. Validate every supplied carrier before
constructing the complete native request used by sender verification.
"""

import json
from dataclasses import dataclass
from typing import TypeVar

from a2a.server.context import ServerCallContext
from a2a.server.routes.common import DefaultServerCallContextBuilder
from a2a.utils.errors import InvalidParamsError
from a2a.utils.proto_utils import validate_proto_required_fields
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message as ProtoMessage
from starlette.requests import Request

from kestrel_sovereign.a2a.scope import NativeRecipientScope

_CONTEXT_KEY = "kestrel.native_a2a_http_context"
_Request = TypeVar("_Request", bound=ProtoMessage)


@dataclass(frozen=True)
class NativeRecipientBinding:
    """Server-owned registration, never selected by a request's tenant."""

    routing_key: str
    recipient_id: str
    tenant: str
    protocol_binding: str

    def __post_init__(self) -> None:
        if not isinstance(self.routing_key, str) or not self.routing_key:
            raise ValueError("Native A2A routing_key must be a concrete identity")
        NativeRecipientScope(self.recipient_id, self.tenant)
        if self.protocol_binding not in {"JSONRPC", "HTTP+JSON"}:
            raise ValueError("Unsupported native A2A HTTP protocol binding")

    @property
    def scope(self) -> NativeRecipientScope:
        """The same logical recipient is shared by both native HTTP bindings."""
        return NativeRecipientScope(self.recipient_id, self.tenant)


@dataclass(frozen=True)
class _NativeHTTPRequest:
    binding: NativeRecipientBinding
    request: Request


class NativeServerCallContextBuilder(DefaultServerCallContextBuilder):
    """Retain the public Request so asynchronous validation can inspect carriers."""

    def __init__(self, binding: NativeRecipientBinding):
        self._binding = binding

    def build(self, request: Request) -> ServerCallContext:
        context = super().build(request)
        context.state[_CONTEXT_KEY] = _NativeHTTPRequest(self._binding, request)
        return context


def _native_http_request(context: ServerCallContext) -> _NativeHTTPRequest:
    """Require the host's binding once for both validation and proof headers."""
    bound = context.state.get(_CONTEXT_KEY)
    if not isinstance(bound, _NativeHTTPRequest):
        raise RuntimeError("Native A2A request lacks its host-owned HTTP context")  # noqa: TRY004
    return bound


def native_header_values(context: ServerCallContext, name: str) -> list[str]:
    """Retain duplicate HTTP headers for the native proof verifier."""
    return _native_http_request(context).request.headers.getlist(name)


async def wait_native_disconnect(context: ServerCallContext) -> None:
    """Wait for the HTTP peer after its request body has been fully consumed."""
    request = _native_http_request(context).request
    message = await request.receive()
    if message["type"] != "http.disconnect":
        raise RuntimeError("Native disconnect monitoring requires a consumed HTTP request")


def _require_tenant(value: object, expected: str) -> None:
    # ProtoJSON null denotes an unset scalar. An explicitly unset tenant is
    # consistent only with the advertised default/empty tenant.
    if value is None:
        value = ""
    if not isinstance(value, str) or value != expected:
        raise InvalidParamsError(
            message="A2A tenant does not match the selected recipient"
        )


async def bind_native_request(
    params: _Request,
    context: ServerCallContext,
    *,
    routing_key: str,
    scope: NativeRecipientScope,
) -> _Request:
    """Validate routing before authorization, then resolve REST path fields.

    The returned protobuf is a copy. No carrier can change the selected agent,
    and an inconsistent body/query tenant is rejected before path resolution.
    This must precede signature verification and SDK-generated task/context IDs.
    """
    bound = _native_http_request(context)
    binding, request = bound.binding, bound.request
    if binding.routing_key != routing_key or binding.scope != scope:
        raise InvalidParamsError(
            message="A2A request does not match the selected recipient"
        )
    if not isinstance(params, ProtoMessage):
        raise TypeError("Native A2A parameters must be an official protobuf request")
    field = params.DESCRIPTOR.fields_by_name.get("tenant")
    if field is None or field.type != FieldDescriptor.TYPE_STRING:
        raise TypeError("Native A2A request has no tenant field")
    validate_proto_required_fields(params)

    expected = binding.tenant
    _require_tenant(context.tenant, expected)
    queries = request.query_params.getlist("tenant")
    if len(queries) > 1:
        raise InvalidParamsError(
            message="A2A request contains repeated tenant parameters"
        )
    for value in queries:
        _require_tenant(value, expected)

    raw = await request.body()
    if raw:
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise InvalidParamsError(
                message="A2A tenant carrier is not valid JSON"
            ) from error
        if not isinstance(body, dict):
            raise InvalidParamsError(message="A2A request body must be an object")
        if binding.protocol_binding == "JSONRPC":
            body = body.get("params")
            if body is None:
                body = {}
            if not isinstance(body, dict):
                raise InvalidParamsError(
                    message="A2A request parameters must be an object"
                )
        if "tenant" in body:
            _require_tenant(body["tenant"], expected)

    requested_tenant = getattr(params, field.name)
    if binding.protocol_binding == "JSONRPC":
        _require_tenant(requested_tenant, expected)
    else:
        _require_tenant(request.path_params.get("tenant", ""), expected)
        if requested_tenant:
            _require_tenant(requested_tenant, expected)

    result = type(params)()
    result.CopyFrom(params)
    setattr(result, field.name, expected)
    return result
