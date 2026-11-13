"""Original Console credentials decide native OAuth consent through delivery."""

import re
from contextlib import asynccontextmanager
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import Request
from kestrel_sdk.operator import OperatorContext
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.responses import HTMLResponse, JSONResponse

from kestrel_sovereign.api_errors import api_error_response, execution_control_api_error
from kestrel_sovereign.auth import caller_context_scope
from kestrel_sovereign.endpoints.mcp_server import create_consent_router
from kestrel_sovereign.multi_agent.route_name import decode_agent_route_name
from kestrel_sovereign.security.browser_shell import OAUTH_BROWSER_HEADERS
from kestrel_sovereign.security.capability_authority import (
    AsyncReaderWriterLock,
    HostCapabilityAccessDenied,
    HostCapabilityAuthority,
    require_current_http_agent,
)
from kestrel_sovereign.security.csrf import CSRFError
from kestrel_sovereign.security.private_delivery import AuthorizedCapabilityResponse
from kestrel_sovereign.security.request_auth import (
    RequestAuthenticationService,
    _bind_request_authentication,
)
from kestrel_sovereign.security.request_body import InvalidJSONRequest, read_json_model

from .grants import (
    MCPAuthorizationCommitRequired,
    MCPAuthorizationDecisionUnavailable,
    MCPAuthorizationGrantQuotaExceeded,
)
from .oauth import NativeMCPAuthorizationServer
from .policy import MemoryScope


class _Action(StrEnum):
    READ = "mcp-server.consent.read"
    DECIDE = "mcp-server.consent.decide"


class _Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["approve", "deny"]
    scopes: list[MemoryScope] = Field(max_length=3)

    @model_validator(mode="after")
    def original_selection(self):
        if (len(set(self.scopes)) != len(self.scopes)
                or (self.decision == "approve") != bool(self.scopes)):
            raise ValueError("Approval requires unique scopes; denial requires none")
        return self


class InvalidMCPConsentRequest(ValueError):
    """The private decision body is invalid before entering its SQL operation."""


def consent_page(api_prefix: str) -> HTMLResponse:
    """Render only a canonical host routing prefix, never request-selected HTML."""
    if api_prefix:
        prefix = "/api/agent-routes/"
        if not api_prefix.startswith(prefix):
            raise ValueError("Original Console routing prefix required")
        decode_agent_route_name(api_prefix[len(prefix):])
    template = (Path(__file__).parents[1] / "static/mcp-server-consent.html").read_text()
    marker = "__KESTREL_MCP_API_PREFIX__"
    if template.count(marker) != 1:
        raise RuntimeError("Original MCP consent shell is invalid")
    return HTMLResponse(template.replace(marker, api_prefix), headers=OAUTH_BROWSER_HEADERS)


class MCPConsentController:
    """One agent's private consent API; the publication owns its lifetime."""

    def __init__(self, agent, authorization: NativeMCPAuthorizationServer, *, require_current):
        grants = authorization.grants
        if agent.did != grants.agent_id or not callable(require_current):
            raise ValueError("Original MCP agent and publication checker required")
        self.agent, self.grants, self._require_current = agent, grants, require_current
        self.authorization = authorization
        self._gate = AsyncReaderWriterLock()
        # Pending OAuth descriptions contain no memory or conversation content.
        # Consent still retains the original durable privacy gate and epoch.
        self.authority = HostCapabilityAuthority(capability="mcp-server-consent", label="MCP consent",
            non_disclosing_actions=frozenset(action.value for action in _Action))
        self.router = create_consent_router(lambda _request: self)

    def _current(self, request):
        self._require_current()
        self.grants.consent_authority.require_current_host()
        require_current_http_agent(self.agent, request, label="MCP consent")

    @asynccontextmanager
    async def _resolve_context(self, request):
        self._current(request)
        service = getattr(request.app.state, "request_authentication", None)
        if not isinstance(service, RequestAuthenticationService):
            raise HostCapabilityAccessDenied("Interactive MCP consent is unavailable")
        # Keep the original freshly reauthenticated HTTP credential, including
        # CSRF, through physical delivery rather than a buffering acknowledgement.
        with (service.bind(request) as authentication, caller_context_scope(authentication.caller),
              _bind_request_authentication(authentication)):
            self.grants.consent_authority.require_sovereign(expected=authentication)
            epoch = self.agent._get_privacy_policy_epoch()
            now = self.authority.clock()
            deadline = now + timedelta(seconds=self.grants.policy.authorization_seconds)
            if authentication.expires_at is not None:
                deadline = min(deadline, authentication.expires_at)
            context = OperatorContext(tenant_id=self.agent.did, acting_agent_id=self.agent.did,
                principal_id=authentication.caller.identity,
                granted_actions=frozenset(action.value for action in _Action),
                granted_capabilities=frozenset({"mcp-server-consent"}),
                permitted_boundary_ids=frozenset({self.agent.did}),
                correlation_id=uuid4().hex, issued_at=now, expires_at=deadline)

            def current(_context, _action, target):
                self._current(request)
                with _bind_request_authentication(authentication):
                    self.grants.consent_authority.require_sovereign(expected=authentication)
                if target is not None or epoch != self.agent._get_privacy_policy_epoch():
                    raise HostCapabilityAccessDenied("Original MCP consent privacy authority ended")

            with self.authority.bind_authenticated(context, privacy_holder=self.agent,
                    check_current=current, authority_gate=self._gate):
                yield context

    async def _body(self, request):
        try:
            return await read_json_model(request, _Decision, max_bytes=self.grants.policy.max_request_bytes)
        except (InvalidJSONRequest, ValidationError):
            raise InvalidMCPConsentRequest("Invalid consent decision") from None

    @staticmethod
    def _error(error, prepared):
        failure = execution_control_api_error(error)
        if failure is not None:
            return api_error_response(status_code=failure.status_code, code=failure.code,
                message=failure.message, details=failure.details)
        if prepared:
            status, code = 503, "mcp_consent_outcome_unknown_do_not_replay"
        elif isinstance(error, (HostCapabilityAccessDenied, CSRFError)):
            status, code = 403, "mcp_consent_access_denied"
        elif isinstance(error, InvalidMCPConsentRequest):
            status, code = 400, "invalid_mcp_consent_request"
        elif isinstance(error, MCPAuthorizationDecisionUnavailable):
            status, code = 404, "mcp_consent_request_unavailable"
        elif isinstance(error, (MCPAuthorizationGrantQuotaExceeded, MCPAuthorizationCommitRequired)):
            status, code = 409, "mcp_consent_decision_unavailable"
        else:
            status, code = 503, "mcp_consent_outcome_unknown_do_not_replay"
        return api_error_response(status_code=status, code=code,
            message="Authorization is unavailable. Reconnect from the connector to start a new request.")

    @staticmethod
    def _handle(handle):
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", handle):
            raise MCPAuthorizationDecisionUnavailable("Original pending request unavailable")

    def _respond(self, request, action, build, *, prepare=None):
        return AuthorizedCapabilityResponse(request, authority=self.authority,
            resolve_context=self._resolve_context,
            operation=lambda context, _value: self.authority.operation(context, action, self.agent.did),
            build=build, error_response=lambda error, prepared: self._error(
                error, prepared and action is _Action.DECIDE), prepare=prepare,
            timeout_seconds=self.grants.policy.authorization_seconds)

    def describe_response(self, handle: str, request: Request):
        async def build(_context):
            self._handle(handle)
            pending = await self.grants.read_authorization_request(handle)
            if pending is None:
                raise MCPAuthorizationDecisionUnavailable("Original pending request unavailable")
            client = self.grants.policy.registered_client(pending.client_id)
            return JSONResponse({"client_name": client.client_name, "agent_name": self.agent.agent_name,
                "resource": self.grants.pending.resource_url,
                "redirect_uri": str(pending.parameters.redirect_uri),
                "expires_at": pending.expires_at, "scopes": pending.parameters.scopes})
        return self._respond(request, _Action.READ, build)

    def decision_response(self, handle: str, request: Request):
        async def build(_context, value):
            self._handle(handle)
            values = {}
            if value.decision == "deny":
                pending = await self.grants.deny_authorization_request(handle)
                if pending is None:
                    raise MCPAuthorizationDecisionUnavailable("Original pending request unavailable")
                values["error"] = "access_denied"
            else:
                pending = await self.grants.read_authorization_request(handle)
                if pending is None:
                    raise MCPAuthorizationDecisionUnavailable("Original pending request unavailable")
                with self.grants.consent_authority.approve(handle, pending, tuple(value.scopes)) as approval:
                    code = await self.grants.issue_code(handle, approval)
                values["code"] = code.code
            if pending.parameters.state is not None:
                values["state"] = pending.parameters.state
            # The same native authorization server owns callback encoding and iss.
            redirect = self.authorization.authorization_response(str(pending.parameters.redirect_uri), values)
            return JSONResponse({"redirect_url": redirect.headers["location"]})
        return self._respond(request, _Action.DECIDE, build, prepare=lambda: self._body(request))
