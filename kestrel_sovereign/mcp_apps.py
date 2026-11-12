"""Authenticated Core ownership of optional MCP Apps and physical Views.

The feature owns native MCP registration and execution. Core owns interactive
credentials, routed-agent membership, privacy leases and actual HTTP delivery.
Neither a chat part nor a request body can supply any of those authorities.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
import secrets
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from kestrel_sdk.operator import OperatorContext
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sdk.tools import ToolPublicationContext
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.responses import JSONResponse

from kestrel_sovereign.agent.parts import current_tool_publication_epoch
from kestrel_sovereign.auth import CallerContext, caller_context_scope, current_caller_context
from kestrel_sovereign.security.capability_authority import (
    AsyncReaderWriterLock, HostCapabilityAccessDenied, HostCapabilityAuthority,
    require_raw_capability_disclosure, require_current_feature_host,
)
from kestrel_sovereign.security.csrf import CSRFError
from kestrel_sovereign.security.private_delivery import (
    AuthorizedCapabilityResponse, PrivateTransportBorrow,
)
from kestrel_sovereign.security.request_auth import (
    RequestAuthenticationService, current_request_authentication,
)


def _endpoint(value):
    url = AnyHttpUrl(value)
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("MCP Apps endpoints must not contain credentials, queries or fragments")
    if url.scheme == "http" and url.host not in {"localhost", "127.0.0.1", "[::1]"}:
        raise ValueError("MCP Apps endpoints require HTTPS outside loopback")
    return url


def _origin(url):
    port = "" if url.port == (443 if url.scheme == "https" else 80) else f":{url.port}"
    return f"{url.scheme}://{url.host}{port}"


class MCPAppsHostPolicy(BaseModel):
    """Host settings, never server metadata or App-selected grants."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    sandbox_url: str
    host_origin: str
    max_html_bytes: int = Field(default=1_048_576, gt=0, le=16_777_216)
    max_message_bytes: int = Field(default=2_097_152, gt=0, le=33_554_432)
    max_request_bytes: int = Field(default=65_536, gt=0, le=1_048_576)
    operation_seconds: float = Field(default=30.0, gt=0, le=300)
    connection_policies: dict[str, dict] = Field(default_factory=dict)

    @field_validator("sandbox_url")
    @classmethod
    def sandbox_endpoint(cls, value):
        return str(_endpoint(value))

    @field_validator("host_origin")
    @classmethod
    def public_host_origin(cls, value):
        url = _endpoint(value)
        if url.path not in (None, "/"):
            raise ValueError("MCP Apps host_origin must contain only an origin")
        return _origin(url)

    @model_validator(mode="after")
    def different_origins(self):
        if _origin(_endpoint(self.sandbox_url)) == self.host_origin:
            raise ValueError("MCP Apps sandbox must use a different origin")
        return self


class _Action(StrEnum):
    READ = "mcp-apps.read"
    CALL = "mcp-apps.call"
    CLOSE = "mcp-apps.close"


@dataclass(frozen=True)
class _Publisher:
    caller: CallerContext
    # Quotas remain per original principal across policy installations.
    privacy_epoch: str = field(compare=False)


@dataclass
class _PhysicalView:
    view: object
    owner: _Publisher
    session_id: str


@dataclass
class _Reader:
    caller: CallerContext
    session_id: str
    authentication: object
    borrowed: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)
    active: bool = True
    baseline_check: Callable[[], None] | None = None
    physical: _PhysicalView | None = None
    opened_id: str | None = None
    delivered: bool = False


@dataclass(frozen=True)
class _OwnedAppCall:
    task: asyncio.Task
    cancel: Callable[[asyncio.Task], object] = field(repr=False)
    active: bool = field(default=True, init=False)

    def _cancel(self, task):
        from kestrel_sovereign.callbacks import require_callback_result

        require_callback_result(self.cancel(task))

    async def wait(self):
        from kestrel_sovereign._async_ownership import await_owned_task, raise_owned_outcome

        if not self.active:
            raise ResourceConflict('Original MCP App call ownership ended')
        outcome = await await_owned_task(self.task, cancel_on_caller_cancellation=self._cancel)
        return raise_owned_outcome(outcome, operation='MCP App call', retain_originals=True)


class _Call(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str = Field(min_length=1, max_length=256)
    arguments: dict
    operation_id: str = Field(min_length=1, max_length=256)


class _Empty(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


def _native(value):
    return value.model_dump(mode="json", by_alias=True, exclude_none=True)


def _error(error, prepared):
    from kestrel_sovereign.api_errors import (
        api_error_response, execution_control_api_error, tool_effect_api_error,
    )

    failure = execution_control_api_error(error)
    if failure is None and not prepared:
        failure = tool_effect_api_error(error)
    if failure is not None:
        return api_error_response(status_code=failure.status_code, code=failure.code,
            message=failure.message, details=failure.details)
    if prepared or isinstance(error, (ResourceOutcomeUnknown, TimeoutError)):
        status, code = 503, "mcp_app_outcome_unknown_do_not_replay"
    elif isinstance(error, (HostCapabilityAccessDenied, CSRFError)):
        status, code = 403, "mcp_app_access_denied"
    elif isinstance(error, ResourceConflict):
        status, code = 404, "mcp_app_unavailable"
    elif isinstance(error, (ValueError, ValidationError)):
        status, code = 400, "invalid_mcp_app_request"
    else:
        status, code = 500, "mcp_app_service_error"
    return JSONResponse({"error": {"code": code}}, status_code=status)


class MCPAppsHost:
    """One feature incarnation; original registrations cannot be retargeted."""

    def __init__(self, agent, feature, *, configuration: dict, is_current: Callable[[], bool]):
        if not callable(is_current) or not isinstance(agent.did, str) or not agent.did.strip():
            raise ValueError("MCP Apps require an original agent and feature incarnation")
        self.agent, self.feature = agent, feature
        self._agent_id, self._is_current = agent.did, is_current
        self._policy = MCPAppsHostPolicy.model_validate(deepcopy(configuration))
        self._reader = ContextVar("mcp_apps_original_reader", default=None)
        self._gate = AsyncReaderWriterLock()
        self.authority = HostCapabilityAuthority(capability="mcp-apps", label="MCP Apps",
            non_disclosing_actions=frozenset({_Action.CLOSE.value}),
            priority_actions=frozenset({_Action.CLOSE.value}))
        self._publications = None
        self._views: dict[str, _PhysicalView] = {}
        self._accepting, self.retired = True, False

    def _require_live(self, request=None):
        require_current_feature_host(self.agent, self.feature, self._agent_id,
            available=self._accepting and not self.retired, is_current=self._is_current,
            label="MCP Apps", request=request)

    def publication_owner(self, destination: ToolPublicationContext):
        """Resolve the real initial HTTP publisher after governed native audit."""
        self._require_live()
        authentication = current_request_authentication()
        if authentication is None:
            return None
        caller = current_caller_context()
        epoch = current_tool_publication_epoch(self.agent, destination.session_id)
        if (caller is not authentication.caller or destination.agent_id != self._agent_id
                or epoch is None or epoch != self.agent._get_privacy_policy_epoch()):
            raise HostCapabilityAccessDenied("original MCP Apps publication lifetime expired")
        authentication.require_current()
        require_raw_capability_disclosure(self.agent, "mcp-apps")
        return _Publisher(caller, epoch)

    def require_reader(self, owner, destination: ToolPublicationContext):
        reader = self._reader.get()
        if (not isinstance(owner, _Publisher) or reader is None or not reader.active
                or reader.baseline_check is None or reader.caller != owner.caller
                or destination.agent_id != self._agent_id or destination.session_id != reader.session_id
                or owner.privacy_epoch != self.agent._get_privacy_policy_epoch()):
            raise HostCapabilityAccessDenied("original MCP Apps reader is unavailable")
        # The baseline does not call View.require_current(), which calls us.
        reader.baseline_check()

    def current_call_session_id(self):
        """The original authenticated App call, never an ambient chat selector."""
        reader = self._reader.get()
        if reader is None:
            return None
        if not reader.active or reader.baseline_check is None or reader.physical is None:
            raise HostCapabilityAccessDenied('Original MCP App call is unavailable')
        reader.baseline_check()
        reader.physical.view.require_current()
        return reader.session_id

    @asynccontextmanager
    async def borrow_input_view(self, view, *, request):
        """Reauthenticate a fresh input continuation against the original View."""
        async with self._resolve_context(request) as context:
            async with self._operation(context, _Action.CALL) as check:
                identifier = next((key for key, physical in self._views.items()
                    if physical.view is view), None)
                if identifier is None:
                    raise ResourceConflict('Original MCP input View is unavailable')
                self._physical(identifier)
                check()
                async with self._transport_operation(context, None, check) as transport:
                    yield transport
                    check()

    @asynccontextmanager
    async def _resolve_context(self, request):
        self._require_live(request)
        session_id = request.query_params.get("session_id")
        if (len(request.query_params.getlist("session_id")) != 1 or not isinstance(session_id, str)
                or not session_id.strip() or len(session_id) > 256):
            raise ValueError("one original chat selector is required")
        service = getattr(request.app.state, "request_authentication", None)
        if not isinstance(service, RequestAuthenticationService):
            raise HostCapabilityAccessDenied("interactive request authentication is unavailable")
        with service.bind(request) as authentication, caller_context_scope(authentication.caller):
            now = self.authority.clock()
            deadline = now + timedelta(seconds=self._policy.operation_seconds)
            if authentication.expires_at is not None:
                deadline = min(deadline, authentication.expires_at)
            context = OperatorContext(tenant_id=self._agent_id, acting_agent_id=self._agent_id,
                principal_id=authentication.caller.identity,
                granted_actions=frozenset(action.value for action in _Action),
                granted_capabilities=frozenset({"mcp-apps"}),
                permitted_boundary_ids=frozenset({self._agent_id}),
                correlation_id=uuid4().hex, issued_at=now, expires_at=deadline)
            reader = _Reader(authentication.caller, session_id, authentication)

            def check_current(_context, _action, target):
                self._require_live(request)
                authentication.require_current()
                if not reader.active or target is not None:
                    raise HostCapabilityAccessDenied("original MCP Apps request expired")

            with self.authority.bind_authenticated(context, privacy_holder=self.agent,
                    check_current=check_current, authority_gate=self._gate):
                token = self._reader.set(reader)
                try:
                    async with reader.borrowed:
                        yield context
                finally:
                    try:
                        if reader.opened_id is not None and not reader.delivered:
                            physical = self._views.get(reader.opened_id)
                            if physical is not None:
                                await self._publications.close_view(physical.view)
                                self._views.pop(reader.opened_id, None)
                    finally:
                        reader.active = False
                        self._reader.reset(token)

    @asynccontextmanager
    async def _operation(self, context, action):
        reader = self._reader.get()
        if action is _Action.CALL:
            await reader.borrowed.enter_async_context(self.agent._get_privacy_transition_lock())
        async with self.authority.operation(context, action, self._agent_id) as baseline:
            reader.baseline_check = baseline
            def check():
                baseline()
                if action is not _Action.CLOSE and reader.physical is not None:
                    token = self._reader.set(reader)
                    try:
                        reader.physical.view.require_current()
                    finally:
                        self._reader.reset(token)
            try:
                yield check
            finally:
                reader.baseline_check = None

    @asynccontextmanager
    async def _transport_operation(self, _context, _value, check):
        reader = self._reader.get()
        physical = reader.physical
        @asynccontextmanager
        async def borrow():
            # The outer physical transport can run outside the producer's
            # ContextVar snapshot. Re-present this exact still-active reader.
            token = self._reader.set(reader)
            try:
                async with physical.view.borrow_delivery() as current:
                    yield current
            finally:
                self._reader.reset(token)
        yield (PrivateTransportBorrow(borrow, physical.view.cancel_delivery)
               if physical is not None and not physical.view.is_closed else None)
        check()
        reader.delivered = True

    def _physical(self, view_id, *, closing=False):
        reader = self._reader.get()
        physical = self._views.get(view_id)
        if (physical is None or physical.owner.caller != reader.caller
                or physical.session_id != reader.session_id):
            raise ResourceConflict("original MCP Apps View is unavailable")
        if not closing and physical.owner.privacy_epoch != self.agent._get_privacy_policy_epoch():
            raise ResourceConflict("original MCP Apps policy expired")
        reader.physical = physical
        if not closing:
            physical.view.require_current()
        return physical

    async def _body(self, request, model):
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise ValueError("JSON request required")
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > self._policy.max_request_bytes:
                raise ValueError("MCP Apps request exceeds byte limit")
            content.extend(chunk)
        return model.model_validate_json(bytes(content))

    def attach(self, publications):
        if self._publications is not None:
            raise ValueError("original MCP Apps publisher already attached")
        if getattr(publications.executor, "supports_owned_delivery", None) is not True:
            raise TypeError("MCP Apps require native owned delivery support")
        self._publications = publications
        self.router = self.router_for(lambda: self)

    @staticmethod
    def router_for(resolve_host):
        router = APIRouter(prefix="/api/mcp/apps", tags=["mcp-apps"])
        def original_host():
            host = resolve_host()
            if host is None:
                raise HTTPException(status_code=404, detail="mcp_app_unavailable")
            if not isinstance(host, MCPAppsHost) or host._publications is None:
                raise TypeError("original attached MCP Apps host required")
            return host

        def respond(host, request, action, build, model=None):
            return AuthorizedCapabilityResponse(request, authority=host.authority,
                resolve_context=host._resolve_context,
                operation=lambda context, _value: host._operation(context, action),
                build=build, error_response=_error, timeout_seconds=host._policy.operation_seconds,
                prepare=(lambda: host._body(request, model)) if model is not None else None,
                transport_operation=host._transport_operation)

        @router.post("/publications/{reference}/views")
        async def open_view(reference: str, request: Request):
            host = original_host()
            async def build(_context, _value):
                browser_origin = request.headers.get("origin")
                if browser_origin is not None and browser_origin != host._policy.host_origin:
                    raise HostCapabilityAccessDenied("original MCP Apps browser origin is required")
                view = host._publications.open_view(reference)
                reader = host._reader.get()
                view_id = secrets.token_urlsafe(32)
                if view_id in host._views:
                    await host._publications.close_view(view)
                    raise ResourceConflict("MCP Apps View identifier collision")
                physical = _PhysicalView(view, _Publisher(reader.caller,
                    host.agent._get_privacy_policy_epoch()), reader.session_id)
                host._views[view_id] = reader.physical = physical
                reader.opened_id = view_id
                resource = await view.read_resource(view.source.tool.resource_uri,
                                                 max_html_bytes=host._policy.max_html_bytes)
                arguments, result = host._publications.initial_result(reference, view)
                return JSONResponse({"view_id": view_id, "arguments": arguments, "result": _native(result),
                    "resource": {"uri": resource.uri, "html": resource.html, "_meta": resource.metadata},
                    "sandbox_url": host._policy.sandbox_url,
                    "host_origin": host._policy.host_origin,
                    "max_message_bytes": host._policy.max_message_bytes,
                    "max_html_bytes": host._policy.max_html_bytes,
                    "policy": deepcopy(host._policy.connection_policies.get(view.caller.name, {}))})
            return respond(host, request, _Action.READ, build, _Empty)

        @router.get("/views/{view_id}/tools")
        async def list_tools(view_id: str, request: Request):
            host = original_host()
            async def build(_context):
                physical = host._physical(view_id)
                return JSONResponse({"tools": [_native(tool) for tool in physical.view.list_tools()]})
            return respond(host, request, _Action.READ, build)

        @router.post("/views/{view_id}/tools/call")
        async def call_tool(view_id: str, request: Request):
            host = original_host()
            async def build(_context, value):
                physical = host._physical(view_id)
                executor = host._publications.executor
                try:
                    result = await executor.invoke(physical.view, value.name,
                        value.arguments, operation_id=value.operation_id)
                except ResourceConflict as error:
                    pending = executor.input_required(error, physical.view)
                    if pending is None:
                        raise
                    return JSONResponse({'input_required': pending}, status_code=202)
                return JSONResponse(_native(result))
            return respond(host, request, _Action.CALL, build, _Call)

        @router.delete("/views/{view_id}")
        async def close_view(view_id: str, request: Request):
            host = original_host()
            async def build(_context):
                physical = host._physical(view_id, closing=True)
                await host._publications.close_view(physical.view)
                host._views.pop(view_id, None)
                return JSONResponse({"closed": True})
            return respond(host, request, _Action.CLOSE, build)
        return router

    def withdraw_private_data(self):
        if self._publications is not None:
            self._publications.withdraw_publications()

    async def maintain(self):
        if self._publications is not None:
            await self._publications.drain_retired()
            self._views = {key: physical for key, physical in self._views.items()
                           if not physical.view.is_closed}

    async def close(self):
        self._accepting = False
        if self._publications is not None:
            await self._publications.close()
        self._views.clear()
        self.retired = True


class MCPAppsHostMixin:
    supports_owned_mcp_app_calls = True

    @asynccontextmanager
    async def mcp_app_call_scope(self, operation, *, name, cancel):
        """Own one View child through its native execution and physical join."""
        from kestrel_sovereign._async_ownership import (
            OwnedCallbackScope, await_owned_task, bind_owned_coordinator,
        )
        from kestrel_sovereign.callbacks import invoke_host_callback
        from kestrel_sovereign.execution_custody import execution_observation_error

        if not callable(cancel):
            raise TypeError('MCP App calls require their original cancellation policy')
        with bind_owned_coordinator(self._get_privacy_transition_lock()):
            scope = OwnedCallbackScope()
            owner = _OwnedAppCall(scope.start(lambda: invoke_host_callback(operation), name=name), cancel)
            pending = None
            try:
                yield owner
            except BaseException as error:
                pending = error
                raise
            finally:
                object.__setattr__(owner, 'active', False)
                scope.close()
                outcome = await await_owned_task(owner.task)
                terminal = execution_observation_error(pending, outcome.error, outcome.cancellation)
                if terminal is not None and terminal is not pending:
                    raise terminal

    supports_mcp_apps_host = True

    def create_mcp_apps_router(self, feature, *, resolve_host):
        """Mount fixed native routes before optional Apps configuration exists."""
        if not callable(resolve_host):
            raise TypeError("original MCP Apps host resolver required")
        def original_host():
            host = resolve_host()
            if host is not None and (not isinstance(host, MCPAppsHost)
                    or host.agent is not self or host.feature is not feature):
                raise HostCapabilityAccessDenied("original MCP Apps host was replaced")
            return host
        return MCPAppsHost.router_for(original_host)

    def validate_mcp_apps_host_configuration(self, configuration):
        """Validate without withdrawing an already usable host incarnation."""
        if any(getattr(self, name, None) is not True for name in (
            "supports_deferred_tool_results", "supports_audited_deferred_tool_results",
            "supports_tool_publication_context",
        )):
            raise TypeError("MCP Apps require audited deferred results and live chat publication")
        return MCPAppsHostPolicy.model_validate(deepcopy(configuration)).model_dump(mode="json")

    def create_mcp_apps_host(self, feature, *, configuration, is_current):
        configuration = self.validate_mcp_apps_host_configuration(configuration)
        hosts = getattr(self, "_mcp_apps_hosts", None)
        if hosts is None:
            hosts = self._mcp_apps_hosts = []
        if any(not host.retired for host in hosts):
            raise ResourceConflict("close the original MCP Apps host before replacement")
        host = MCPAppsHost(self, feature, configuration=configuration, is_current=is_current)
        hosts[:] = [host]
        return host

    def _withdraw_mcp_apps_for_privacy_transition(self):
        for host in getattr(self, "_mcp_apps_hosts", ()):
            host.withdraw_private_data()
