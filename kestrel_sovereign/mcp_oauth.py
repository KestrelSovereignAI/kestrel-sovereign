"""Host-owned MCP OAuth identity and authenticated connection boundaries.

The feature owns native OAuth and MCP I/O. Core owns operator credentials,
agent membership, private delivery and the public web client identity.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
import hashlib
import ipaddress
import inspect
import json
import hmac
import re
import secrets
from typing import Self
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from kestrel_sdk.operator import OperatorContext
from kestrel_sdk.storage.database import DatabaseBackend
from kestrel_sovereign.auth import CallerContext, caller_context_scope, current_caller_context
from kestrel_sovereign.features.storage_access import resolve_feature_database
from kestrel_sovereign.security.capability_authority import (
    AsyncReaderWriterLock, HostCapabilityAccessDenied, HostCapabilityAuthority,
    require_current_feature_host, require_raw_capability_disclosure,
)
from kestrel_sovereign.security.request_auth import (
    InteractiveRequestAuthentication, RequestAuthenticationService, RetainedCredentialProof,
    current_request_authentication,
)

from fastapi import APIRouter, HTTPException, Request
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.responses import JSONResponse
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.security.csrf import CSRFError
from kestrel_sovereign.security.private_delivery import AuthorizedCapabilityResponse


MCP_OAUTH_CALLBACK_PATH = "/oauth/mcp/callback"
MCP_OAUTH_CLIENT_METADATA_PATH = "/oauth/mcp/client-metadata.json"


class MCPOAuthHostPolicy(BaseModel):
    """Explicit installation identity and finite host budgets; never vendor grants."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False, hide_input_in_errors=True)
    console_origin: str
    client_name: str = "Kestrel Sovereign"
    max_operations: int = Field(default=16, gt=0, le=256)
    max_operations_per_principal: int = Field(default=2, gt=0, le=16)
    authorization_seconds: float = Field(default=600.0, gt=0, le=1800)
    network_seconds: float = Field(default=20.0, gt=0, le=60)
    operation_seconds: float = Field(default=90.0, gt=0, le=300)
    response_max_bytes: int = Field(default=65_536, gt=0, le=1_048_576)
    max_token_bytes: int = Field(default=16_384, gt=0, le=1_048_576)
    max_request_bytes: int = Field(default=16_384, gt=0, le=1_048_576)
    lease_seconds: float = Field(default=900.0, gt=0, le=3600)

    @field_validator("console_origin")
    @classmethod
    def public_console_origin(cls, value: str) -> str:
        if value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("MCP OAuth console origin must be a control-free HTTPS origin")
        url = AnyHttpUrl(value)
        if ("@" in urlsplit(value).netloc or url.scheme != "https"
                or url.username is not None or url.password is not None
                or url.query is not None or url.fragment is not None or url.path not in (None, "/")):
            raise ValueError("MCP OAuth console origin must be an exact credential-free HTTPS origin")
        hostname = url.host.strip("[]")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise ValueError("MCP OAuth web client identity requires a public origin")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("MCP OAuth web client identity requires a public origin")
        return str(url).rstrip("/")

    @field_validator("client_name")
    @classmethod
    def client_display_name(cls, value: str) -> str:
        if (not value.strip() or len(value) > 128
                or any(ord(char) < 32 or ord(char) == 127 for char in value)):
            raise ValueError("MCP OAuth client name must be nonempty and control-free")
        return value

    @model_validator(mode="after")
    def bounded_principal_quota(self) -> Self:
        if self.max_operations_per_principal > self.max_operations:
            raise ValueError("MCP OAuth per-principal quota exceeds the host quota")
        return self

    @property
    def redirect_uri(self) -> str:
        return self.console_origin + MCP_OAUTH_CALLBACK_PATH

    @property
    def client_id(self) -> str:
        return self.console_origin + MCP_OAUTH_CLIENT_METADATA_PATH

    def client_document(self) -> dict:
        """Render one public web CIMD identity, with no per-principal scopes or secrets."""
        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "redirect_uris": [self.redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "application_type": "web",
        }

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> Self | None:
        """Read the explicit public origin and optional JSON policy settings.

        Raises:
            ValueError: Configured identity or budgets are invalid or ambiguous.
        """
        origin = environ.get("KESTREL_MCP_OAUTH_CONSOLE_ORIGIN")
        raw = environ.get("KESTREL_MCP_OAUTH_POLICY")
        if origin is None:
            if raw is not None:
                raise ValueError("MCP OAuth policy requires KESTREL_MCP_OAUTH_CONSOLE_ORIGIN")
            return None
        try:
            settings = {} if raw is None else json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("MCP OAuth policy must be a JSON object") from error
        if not isinstance(settings, dict) or "console_origin" in settings:
            raise ValueError("MCP OAuth policy must contain settings with the origin supplied only by its environment field")
        return cls(console_origin=origin, **settings)


class MCPOAuthAction(StrEnum):
    AUTHORIZE = "mcp-oauth.authorize"
    COMPLETE = "mcp-oauth.complete"
    CONNECT = "mcp-oauth.connect"
    READ = "mcp-oauth.read"
    CANCEL = "mcp-oauth.cancel"
    DISCONNECT = "mcp-oauth.disconnect"


def _principal_id(caller: CallerContext) -> str:
    # A credential lane and role cannot alias a different lane's equal-looking subject.
    value = [caller.role.value, caller.auth_method.value, caller.identity]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class _RequestOwner:
    request: object
    authentication: InteractiveRequestAuthentication
    context: OperatorContext
    active: bool = True
    opened_operation: str | None = None
    delivered: bool = False
    feature_acted: bool = False


@dataclass
class _PendingAuthorization:
    owner: CallerContext
    principal_id: str
    epoch: str
    expires_at: datetime
    handle: object = field(default_factory=object)
    state_hash: bytes | None = None
    consumed: bool = False
    finished: bool = False
    withdrawn: bool = False
    in_flight: bool = True
    retirement_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


class _StrictMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False,
                              hide_input_in_errors=True)


class MCPOAuthAuthorizationRequest(_StrictMessage):
    server_url: str = Field(min_length=1, max_length=4096)
    scopes: list[str] = Field(min_length=1, max_length=32)
    acknowledge_unknown: bool = False

    @field_validator("server_url")
    @classmethod
    def https_endpoint(cls, value: str) -> str:
        return _https_endpoint(value)

    @field_validator("scopes")
    @classmethod
    def explicit_scope_tokens(cls, value: list[str]) -> list[str]:
        if (len(set(value)) != len(value) or any(not scope or len(scope) > 256
                or any(ord(char) < 33 or ord(char) > 126 or char in {'"', "\\"} for char in scope)
                for scope in value)):
            raise ValueError("MCP OAuth scopes require unique explicit scope tokens")
        return value


class MCPOAuthCallback(_StrictMessage):
    state: str = Field(min_length=16, max_length=1024, repr=False)
    code: str | None = Field(default=None, min_length=1, max_length=8192, repr=False)
    error: str | None = Field(default=None, min_length=1, max_length=128)
    iss: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def one_callback_outcome(self):
        if (self.code is None) == (self.error is None):
            raise ValueError("MCP OAuth callback requires one code or denial")
        _https_endpoint(self.iss)
        return self


class MCPOAuthConnectRequest(_StrictMessage):
    refresh: bool = False


class MCPOAuthConnectionStatus(_StrictMessage):
    connection_id: str = Field(min_length=1, max_length=256)
    server_url: str = Field(min_length=1, max_length=4096)
    issuer: str = Field(min_length=1, max_length=4096)
    resource: str = Field(min_length=1, max_length=4096)
    scopes: list[str] = Field(max_length=32)
    state: str
    connected: bool

    @field_validator("state")
    @classmethod
    def known_state(cls, value):
        if value not in {"authorizing", "outcome_unknown", "authorized", "connected", "refresh_unknown",
                         "reauthorization_required", "resource_rejected", "revoked"}:
            raise ValueError("unknown MCP OAuth connection state")
        return value

    @model_validator(mode="after")
    def consistent_connection(self):
        for value in (self.server_url, self.issuer, self.resource):
            _https_endpoint(value)
        MCPOAuthAuthorizationRequest.explicit_scope_tokens(self.scopes)
        if self.connected != (self.state == "connected"):
            raise ValueError("inconsistent MCP OAuth connection disposition")
        return self


class _AuthorizationStarted(_StrictMessage):
    handle: object = Field(exclude=True, repr=False)
    authorization_url: str = Field(min_length=1, max_length=16384, repr=False)
    state: str = Field(min_length=16, max_length=1024, repr=False)
    issuer: str = Field(min_length=1, max_length=4096)
    resource: str = Field(min_length=1, max_length=4096)
    scopes: list[str] = Field(min_length=1, max_length=32)


class _DisconnectStatus(_StrictMessage):
    local_tokens_deleted: bool
    remote_revocation: str

    @model_validator(mode="after")
    def local_revocation_only(self):
        if self.local_tokens_deleted is not True or self.remote_revocation != "not_attempted":
            raise ValueError("MCP OAuth local revocation needs its exact terminal receipt")
        return self


class MCPOAuthRequestError(ValueError):
    """A non-secret host routing or operation-state refusal."""

    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status, self.code = status, code


class MCPOAuthRuntimeError(RuntimeError):
    """The trusted feature returned an invalid contract message, never bad user input."""


def _runtime_message(model, value):
    try:
        return model.model_validate(value)
    except ValidationError:
        raise MCPOAuthRuntimeError("invalid MCP OAuth feature response") from None


def _https_endpoint(value: str) -> str:
    if (not isinstance(value, str) or value != value.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("MCP OAuth requires control-free HTTPS endpoints")
    parsed = urlsplit(value)
    _ = parsed.port
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or "#" in value):
        raise ValueError("MCP OAuth requires credential-free HTTPS endpoints")
    return value


def _response_error(error, prepared):
    if prepared or isinstance(error, (ResourceOutcomeUnknown, TimeoutError)):
        status, code = 503, "mcp_oauth_outcome_unknown_do_not_replay"
    elif isinstance(error, MCPOAuthRequestError):
        status, code = error.status, error.code
    elif isinstance(error, (HostCapabilityAccessDenied, CSRFError)):
        status, code = 403, "mcp_oauth_access_denied"
    elif isinstance(error, ResourceConflict):
        status, code = 409, "mcp_oauth_conflict"
    elif isinstance(error, (ValueError, ValidationError)):
        status, code = 400, "invalid_mcp_oauth_request"
    else:
        status, code = 500, "mcp_oauth_service_error"
    return JSONResponse({"error": {"code": code}}, status_code=status)


class MCPOAuthConnectionLease:
    """Bounded host delegation for one original connection; stored IDs cannot mint it."""

    def __init__(self, host, reader: _RequestOwner, *, connection_id: str, slot_id: str):
        self._host, self._caller = host, reader.authentication.caller
        self._principal_id = reader.context.principal_id
        self._connection_id, self._slot_id = connection_id, slot_id
        self._epoch = host.agent._get_privacy_policy_epoch()
        self._publication = getattr(host.agent, "_host_context_publication_state", None)
        self._revoked = False
        self._proof = self._proof_for(reader.authentication)

    @property
    def agent_id(self) -> str:
        return self._host._agent_id

    @property
    def principal_id(self) -> str:
        return self._principal_id

    @property
    def connection_id(self) -> str:
        return self._connection_id

    @property
    def slot_id(self) -> str:
        return self._slot_id

    @property
    def expires_at(self) -> datetime:
        return self._proof.expires_at

    def _proof_for(self, authentication: InteractiveRequestAuthentication) -> RetainedCredentialProof:
        deadline = self._host.authority.clock() + timedelta(seconds=self._host._policy.lease_seconds)
        return authentication.retain_credential_proof(expires_at=deadline)

    def _renew(self, authentication: InteractiveRequestAuthentication) -> None:
        proof = self._proof_for(authentication)
        previous, self._proof = self._proof, proof
        previous.revoke()

    def _require_owner(self) -> None:
        self._host._require_live()
        if (self._revoked or self not in self._host._leases
                or self._host.authority.clock() >= self.expires_at
                or self._host.agent._get_privacy_policy_epoch() != self._epoch
                or getattr(self._host.agent, "_host_context_publication_state", None)
                is not self._publication):
            raise HostCapabilityAccessDenied("original MCP OAuth connection lease is unavailable")
        require_raw_capability_disclosure(self._host.agent, "mcp-oauth")

    def require_background(self) -> None:
        """Check delegated credentials and owner policy without an ended request grant."""
        self._require_owner()
        self._proof.require_current()

    def require_principal(self) -> None:
        """Require a fresh original interactive principal before credential admission."""
        self._require_owner()
        authentication = current_request_authentication()
        if (authentication is None or current_caller_context() is not authentication.caller
                or _principal_id(authentication.caller) != self._principal_id):
            raise HostCapabilityAccessDenied("original MCP OAuth principal is required")
        authentication.require_current()
        self._renew(authentication)

    @asynccontextmanager
    async def credential_operation(self):
        """Retain current host gates around credential storage and refresh operations."""
        self.require_principal()
        async with self._host._credential_operation(self) as check:
            yield check

    def revoke(self) -> None:
        self._revoked = True
        self._proof.revoke()
        self._host._leases.discard(self)


class MCPOAuthHost:
    """One native feature incarnation and its host-authenticated request lifetimes."""

    def __init__(self, agent, feature, *, is_current: Callable[[], bool]):
        if not callable(is_current) or not isinstance(agent.did, str) or not agent.did.strip():
            raise ValueError("MCP OAuth requires an original agent and feature incarnation")
        self.agent, self.feature = agent, feature
        self._agent_id, self._is_current = agent.did, is_current
        self._accepting, self.retired = True, False
        self._policy = None
        self._runtime = None
        self._reader = ContextVar("mcp_oauth_original_request", default=None)
        self._issuer = ContextVar("mcp_oauth_original_issuer", default=None)
        self._gate = AsyncReaderWriterLock()
        self._leases: set[MCPOAuthConnectionLease] = set()
        self._pending: dict[str, _PendingAuthorization] = {}
        self.authority = HostCapabilityAuthority(capability="mcp-oauth", label="MCP OAuth",
            non_disclosing_actions=frozenset({MCPOAuthAction.READ.value,
                MCPOAuthAction.CANCEL.value, MCPOAuthAction.DISCONNECT.value}),
            priority_actions=frozenset({MCPOAuthAction.CANCEL.value, MCPOAuthAction.DISCONNECT.value}))

    @property
    def database(self) -> DatabaseBackend:
        database = resolve_feature_database(self.agent)
        if not isinstance(database, DatabaseBackend):
            raise TypeError("MCP OAuth requires the original native feature database")
        return database

    def _require_live(self, request=None) -> None:
        require_current_feature_host(self.agent, self.feature, self._agent_id,
            available=self._accepting and not self.retired, is_current=self._is_current,
            label="MCP OAuth", request=request)
        publication_gate = getattr(self.agent, "_host_context_publication_gate", None)
        manager = getattr(self.agent, "_agent_manager", None)
        if (getattr(self.agent, "_agent_manager_published", None) is False
                or (publication_gate is not None and not publication_gate.is_set())
                or (manager is not None
                    and not any(candidate is self.agent for candidate in manager.list_agents().values()))):
            raise HostCapabilityAccessDenied("original MCP OAuth agent was withdrawn")

    def _policy_for(self, request) -> MCPOAuthHostPolicy:
        policy = getattr(request.app.state, "mcp_oauth_policy", None)
        if not isinstance(policy, MCPOAuthHostPolicy):
            raise HostCapabilityAccessDenied("MCP OAuth host identity is not configured")
        if self._policy is not None and self._policy is not policy:
            raise HostCapabilityAccessDenied("original MCP OAuth host identity changed")
        self._policy = policy
        return policy

    def _context_for_authentication(self, authentication: InteractiveRequestAuthentication) -> OperatorContext:
        authentication.require_current()
        if current_caller_context() is not authentication.caller:
            raise HostCapabilityAccessDenied("original MCP OAuth caller scope is required")
        now = self.authority.clock()
        deadline = now + timedelta(seconds=self._policy.operation_seconds)
        if authentication.expires_at is not None:
            deadline = min(deadline, authentication.expires_at)
        return OperatorContext(tenant_id=self._agent_id, acting_agent_id=self._agent_id,
            principal_id=_principal_id(authentication.caller),
            granted_actions=frozenset(action.value for action in MCPOAuthAction),
            granted_capabilities=frozenset({"mcp-oauth"}),
            permitted_boundary_ids=frozenset({self._agent_id}),
            correlation_id=uuid4().hex, issued_at=now, expires_at=deadline)

    @asynccontextmanager
    async def _credential_operation(self, lease: MCPOAuthConnectionLease):
        issuer = self._issuer.get()
        if issuer is not None:
            if (issuer[0] is not asyncio.current_task()
                    or issuer[2] not in {MCPOAuthAction.CONNECT, MCPOAuthAction.COMPLETE}):
                raise HostCapabilityAccessDenied("credential operation needs its original connect task")
            def nested_check():
                issuer[3]()
                lease.require_principal()
            nested_check()
            yield nested_check
            nested_check()
            return
        authentication = current_request_authentication()
        lease.require_principal()
        context = self._context_for_authentication(authentication)

        def check_current(original_context, _action, target):
            if original_context is not context or target is not lease:
                raise HostCapabilityAccessDenied("original credential operation changed")
            lease.require_principal()

        with self.authority.bind_authenticated(context, privacy_holder=self.agent,
                check_current=check_current, authority_gate=self._gate):
            async with self.authority.operation(context, MCPOAuthAction.CONNECT,
                                               self._agent_id, target=lease) as check:
                yield check

    @asynccontextmanager
    async def _resolve_context(self, request):
        self._require_live(request)
        policy = self._policy_for(request)
        service = getattr(request.app.state, "request_authentication", None)
        if not isinstance(service, RequestAuthenticationService):
            raise HostCapabilityAccessDenied("interactive request authentication is unavailable")
        with service.bind(request) as authentication, caller_context_scope(authentication.caller):
            context = self._context_for_authentication(authentication)
            reader = _RequestOwner(request, authentication, context)
            manager = getattr(request.app.state, "agent_manager", None)
            epoch = self.agent._get_privacy_policy_epoch()

            def check_current(_context, _action, target):
                self._require_live(request)
                authentication.require_current()
                if (not reader.active or target is not None or reader.context is not _context
                        or getattr(request.app.state, "agent_manager", None) is not manager
                        or getattr(request.app.state, "mcp_oauth_policy", None) is not policy
                        or self.agent._get_privacy_policy_epoch() != epoch):
                    raise HostCapabilityAccessDenied("original MCP OAuth request expired")

            with self.authority.bind_authenticated(context, privacy_holder=self.agent,
                    check_current=check_current, authority_gate=self._gate):
                token = self._reader.set(reader)
                try:
                    yield context
                finally:
                    try:
                        if reader.opened_operation is not None and not reader.delivered:
                            pending = self._pending.get(reader.opened_operation)
                            if pending is not None and not pending.consumed:
                                await self._retire_pending(reader.opened_operation, pending)
                    finally:
                        reader.active = False
                        self._reader.reset(token)

    @asynccontextmanager
    async def _operation(self, context: OperatorContext, action: MCPOAuthAction):
        async with self.authority.operation(context, action, self._agent_id) as check:
            reader = self._reader.get()
            if reader is None or not reader.active or reader.context is not context:
                raise HostCapabilityAccessDenied("original MCP OAuth request is unavailable")
            token = self._issuer.set((asyncio.current_task(), context, action, check))
            try:
                yield check
            finally:
                self._issuer.reset(token)

    def issue_lease(self, context: OperatorContext, *, connection_id: str,
                    slot_id: str) -> MCPOAuthConnectionLease:
        """Issue only inside this task's original authenticated connect operation."""
        issuer = self._issuer.get()
        reader = self._reader.get()
        if (issuer is None or issuer[0] is not asyncio.current_task() or issuer[1] is not context
                or issuer[2] not in {MCPOAuthAction.CONNECT, MCPOAuthAction.COMPLETE}
                or reader is None or not reader.active or reader.context is not context):
            raise HostCapabilityAccessDenied("MCP OAuth lease requires its original connect operation")
        if not all(isinstance(value, str) and value and len(value) <= 256
                   and not any(ord(char) < 32 or ord(char) == 127 for char in value)
                   for value in (connection_id, slot_id)):
            raise ValueError("MCP OAuth lease requires bounded connection and slot identifiers")
        issuer[3]()
        lease = MCPOAuthConnectionLease(self, reader, connection_id=connection_id, slot_id=slot_id)
        self._leases.add(lease)
        return lease

    def withdraw(self) -> None:
        """Fence existing leases before a host privacy or lifecycle transition."""
        for lease in tuple(self._leases):
            lease.revoke()
        self._leases.clear()
        for pending in self._pending.values():
            pending.withdrawn = True
        if self._runtime is not None:
            result = self._runtime.withdraw_authorizations()
            if result is not None:
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("MCP OAuth withdrawal must be synchronous and return None")

    def attach(self, runtime) -> None:
        """Attach the original feature runtime; its methods receive host authority explicitly."""
        methods = ("begin", "complete", "cancel", "connect", "disconnect", "list",
                   "retire_authorization", "close_authorizations", "withdraw_authorizations")
        if self._runtime is not None or not all(callable(getattr(runtime, name, None)) for name in methods):
            raise TypeError("original complete MCP OAuth runtime must attach exactly once")
        if (inspect.iscoroutinefunction(runtime.withdraw_authorizations)
                or inspect.isasyncgenfunction(runtime.withdraw_authorizations)):
            raise TypeError("MCP OAuth withdrawal must be synchronous")
        self._runtime = runtime

    @asynccontextmanager
    async def _transport_operation(self, context, value, check):
        reader = self._reader.get()
        yield None
        check()
        reader.delivered = True

    async def _body(self, request: Request, model):
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > self._policy.max_request_bytes:
                raise MCPOAuthRequestError(413, "mcp_oauth_request_too_large")
            content.extend(chunk)
        return model.model_validate_json(bytes(content))

    async def _retire_pending(self, identifier: str, pending: _PendingAuthorization) -> None:
        async with pending.retirement_lock:
            current = self._pending.get(identifier)
            if current is None:
                return
            if current is not pending:
                raise HostCapabilityAccessDenied("original MCP OAuth authorization changed")
            pending.withdrawn = True
            await self._runtime.retire_authorization(pending.handle)
            self._pending.pop(identifier, None)

    async def maintain(self) -> None:
        """Retire expired original provider frames without writing credential state."""
        now = self.authority.clock()
        for identifier, pending in tuple(self._pending.items()):
            if not pending.in_flight and (pending.withdrawn or pending.expires_at <= now):
                await self._retire_pending(identifier, pending)

    def _pending_for(self, context, identifier: str) -> _PendingAuthorization:
        pending = self._pending.get(identifier)
        reader = self._reader.get()
        if pending is None:
            raise MCPOAuthRequestError(404, "mcp_oauth_operation_unavailable")
        # An unauthorized caller cannot cancel or consume somebody else's operation.
        if pending.owner != reader.authentication.caller or pending.principal_id != context.principal_id:
            raise HostCapabilityAccessDenied("original MCP OAuth authorization owner is required")
        if pending.withdrawn:
            raise HostCapabilityAccessDenied("original MCP OAuth authorization was withdrawn")
        if pending.expires_at <= self.authority.clock():
            raise MCPOAuthRequestError(410, "mcp_oauth_operation_expired")
        if pending.epoch != self.agent._get_privacy_policy_epoch():
            raise HostCapabilityAccessDenied("original MCP OAuth authorization epoch changed")
        if pending.consumed:
            raise MCPOAuthRequestError(409, "mcp_oauth_operation_consumed")
        if pending.handle is None or pending.state_hash is None:
            raise MCPOAuthRequestError(409, "mcp_oauth_operation_not_ready")
        return pending

    def _validate_started(self, value: _AuthorizationStarted, request: MCPOAuthAuthorizationRequest) -> None:
        _https_endpoint(value.issuer)
        _https_endpoint(value.resource)
        _https_endpoint(value.authorization_url)
        parsed, issuer = urlsplit(value.authorization_url), urlsplit(value.issuer)
        if (parsed.scheme, parsed.hostname, parsed.port or 443) != (issuer.scheme, issuer.hostname, issuer.port or 443):
            raise ValueError("MCP OAuth authorization left its selected issuer")
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=32)
        expected = {"client_id": self._policy.client_id, "redirect_uri": self._policy.redirect_uri,
            "response_type": "code", "code_challenge_method": "S256", "state": value.state,
            "resource": value.resource}
        if any(query.get(name) != [item] for name, item in expected.items()):
            raise ValueError("MCP OAuth authorization changed its host binding")
        scope = query.get("scope", [])
        if (len(scope) != 1 or frozenset(scope[0].split()) != frozenset(value.scopes)
                or not value.scopes or not frozenset(value.scopes) <= frozenset(request.scopes)
                or len(query.get("code_challenge", [])) != 1
                or re.fullmatch(r"[A-Za-z0-9_-]{43}", query["code_challenge"][0]) is None):
            raise ValueError("MCP OAuth authorization changed its scope or PKCE binding")

    async def _begin(self, context, check, value: MCPOAuthAuthorizationRequest):
        await self.maintain()
        active = [pending for pending in self._pending.values() if not pending.finished]
        if (len(active) >= self._policy.max_operations
                or sum(pending.principal_id == context.principal_id for pending in active)
                >= self._policy.max_operations_per_principal):
            raise MCPOAuthRequestError(429, "mcp_oauth_operation_quota")
        # Finished replay markers are bounded independently of live provider frames.
        finished = [key for key, pending in self._pending.items() if pending.finished]
        while len(finished) >= self._policy.max_operations:
            self._pending.pop(finished.pop(0))
        identifier = secrets.token_urlsafe(32)
        if identifier in self._pending:
            raise ResourceConflict("MCP OAuth operation identifier collision")
        reader = self._reader.get()
        pending = _PendingAuthorization(reader.authentication.caller, context.principal_id,
            self.agent._get_privacy_policy_epoch(),
            self.authority.clock() + timedelta(seconds=self._policy.authorization_seconds))
        self._pending[identifier] = pending
        try:
            check()
            raw = await self._runtime.begin(context, check, value.model_dump(mode="json"),
                self._policy.model_dump(mode="json") | {"client_document": self._policy.client_document()},
                authorization_handle=pending.handle)
            started = _runtime_message(_AuthorizationStarted, raw)
            if started.handle is not pending.handle:
                raise MCPOAuthRuntimeError("MCP OAuth authorization changed its original host owner")
            try:
                self._validate_started(started, value)
            except ValueError:
                raise MCPOAuthRuntimeError("invalid MCP OAuth feature authorization binding") from None
            check()
            if pending.withdrawn:
                raise HostCapabilityAccessDenied("original MCP OAuth authorization was withdrawn")
            if pending.expires_at <= self.authority.clock():
                raise MCPOAuthRequestError(410, "mcp_oauth_operation_expired")
            pending.state_hash = hashlib.sha256(started.state.encode("utf-8")).digest()
            reader.opened_operation = identifier
            return JSONResponse({"operation_id": identifier, "authorization_url": started.authorization_url,
                "issuer": started.issuer, "resource": started.resource, "scopes": started.scopes,
                "expires_at": pending.expires_at.isoformat()})
        except BaseException:
            await self._retire_pending(identifier, pending)
            raise
        finally:
            pending.in_flight = False

    async def _complete(self, context, check, identifier: str, value: MCPOAuthCallback):
        pending = self._pending_for(context, identifier)
        if not hmac.compare_digest(pending.state_hash, hashlib.sha256(value.state.encode("utf-8")).digest()):
            await self._retire_pending(identifier, pending)
            raise MCPOAuthRequestError(400, "mcp_oauth_state_mismatch")
        pending.consumed = True
        pending.in_flight = True
        try:
            check()
            self._reader.get().feature_acted = True
            raw = await self._runtime.complete(pending.handle, context, check, value.model_dump(mode="json"))
            status = _runtime_message(MCPOAuthConnectionStatus, raw)
            check()
            pending.finished = True
            return JSONResponse(status.model_dump(mode="json"))
        finally:
            pending.in_flight = False

    async def close(self) -> None:
        """Retire original authorization workers before releasing their host incarnation."""
        self._accepting = False
        self.withdraw()
        if self._runtime is not None:
            await self._runtime.close_authorizations()
        self._pending.clear()
        self.retired = True

    @staticmethod
    def router_for(resolve_host):
        router = APIRouter(prefix="/api/mcp/oauth", tags=["mcp-oauth"])

        def original(request):
            host = resolve_host()
            if host is None:
                raise HTTPException(status_code=404, detail="mcp_oauth_unavailable")
            if not isinstance(host, MCPOAuthHost) or host._runtime is None:
                raise TypeError("original attached MCP OAuth host required")
            try:
                host._policy_for(request)
            except HostCapabilityAccessDenied:
                raise HTTPException(status_code=403, detail="mcp_oauth_host_identity_unavailable") from None
            return host

        def respond(host, request, action, build, model=None):
            original_owner = None
            @asynccontextmanager
            async def resolve(original_request):
                nonlocal original_owner
                async with host._resolve_context(original_request) as context:
                    original_owner = host._reader.get()
                    yield context
            def error_response(error, prepared):
                acted = original_owner is not None and original_owner.feature_acted
                return _response_error(error, prepared or acted)
            return AuthorizedCapabilityResponse(request, authority=host.authority,
                resolve_context=resolve,
                operation=lambda context, _value: host._operation(context, action),
                build=build, error_response=error_response, timeout_seconds=host._policy.operation_seconds,
                prepare=(lambda: host._body(request, model)) if model is not None else None,
                transport_operation=host._transport_operation)

        @router.get("/connections")
        async def connections(request: Request):
            host = original(request)
            async def build(context):
                rows = await host._runtime.list(context)
                values = [_runtime_message(MCPOAuthConnectionStatus, row).model_dump(mode="json") for row in rows]
                return JSONResponse({"connections": values})
            return respond(host, request, MCPOAuthAction.READ, build)

        @router.post("/authorizations")
        async def authorize(request: Request):
            host = original(request)
            async def build(context, value):
                issuer = host._issuer.get()
                return await host._begin(context, issuer[3], value)
            return respond(host, request, MCPOAuthAction.AUTHORIZE, build, MCPOAuthAuthorizationRequest)

        @router.post("/authorizations/{identifier}/callback")
        async def callback(identifier: str, request: Request):
            host = original(request)
            async def build(context, value):
                return await host._complete(context, host._issuer.get()[3], identifier, value)
            return respond(host, request, MCPOAuthAction.COMPLETE, build, MCPOAuthCallback)

        @router.delete("/authorizations/{identifier}")
        async def cancel(identifier: str, request: Request):
            host = original(request)
            async def build(context):
                pending = host._pending_for(context, identifier)
                check = host._issuer.get()[3]
                pending.consumed = True
                pending.in_flight = True
                try:
                    host._reader.get().feature_acted = True
                    await host._runtime.cancel(pending.handle, check)
                    pending.finished = True
                    check()
                    return JSONResponse({"cancelled": True})
                finally:
                    pending.in_flight = False
            return respond(host, request, MCPOAuthAction.CANCEL, build)

        @router.post("/connections/{identifier}/connect")
        async def connect(identifier: str, request: Request):
            host = original(request)
            async def build(context, value):
                host._reader.get().feature_acted = True
                raw = await host._runtime.connect(context, host._issuer.get()[3], identifier, refresh=value.refresh)
                return JSONResponse(_runtime_message(MCPOAuthConnectionStatus, raw).model_dump(mode="json"))
            return respond(host, request, MCPOAuthAction.CONNECT, build, MCPOAuthConnectRequest)

        @router.delete("/connections/{identifier}")
        async def disconnect(identifier: str, request: Request):
            host = original(request)
            async def build(context):
                host._reader.get().feature_acted = True
                raw = await host._runtime.disconnect(context, host._issuer.get()[3], identifier)
                return JSONResponse(_runtime_message(_DisconnectStatus, raw).model_dump(mode="json"))
            return respond(host, request, MCPOAuthAction.DISCONNECT, build)
        return router


class MCPOAuthHostMixin:
    """Core factory for an optional native feature-owned OAuth runtime."""

    supports_mcp_oauth_host = True

    def create_mcp_oauth_host(self, feature, *, is_current):
        hosts = getattr(self, "_mcp_oauth_hosts", ())
        if any(not host.retired for host in hosts):
            raise ResourceConflict("close the original MCP OAuth host before replacement")
        host = MCPOAuthHost(self, feature, is_current=is_current)
        self._mcp_oauth_hosts = [host]
        return host

    def create_mcp_oauth_router(self, feature, *, resolve_host):
        if not callable(resolve_host):
            raise TypeError("original MCP OAuth host resolver required")
        def original():
            host = resolve_host()
            if host is not None and (not isinstance(host, MCPOAuthHost)
                    or host.agent is not self or host.feature is not feature):
                raise HostCapabilityAccessDenied("original MCP OAuth host was replaced")
            return host
        return MCPOAuthHost.router_for(original)

    def _withdraw_mcp_oauth_for_privacy_transition(self):
        for host in getattr(self, "_mcp_oauth_hosts", ()):
            host.withdraw()
