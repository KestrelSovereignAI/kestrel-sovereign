"""Bind verified native OAuth principals to their original HTTP lifetime."""

import asyncio
import hashlib
import hmac
import json
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from urllib.parse import urlsplit

from mcp.server.auth.middleware.auth_context import get_access_token
from starlette.requests import Request

from kestrel_sovereign.auth import (
    AuthMethod,
    CallerContext,
    caller_context_binding_scope,
    caller_context_lifetime,
    capture_caller_context_binding,
)
from kestrel_sovereign.security.capability_authority import (
    HostCapabilityAccessDenied,
    require_raw_capability_disclosure,
)
from kestrel_sovereign.storage.privacy_wrapper import require_durable_persistence_gate

_REQUEST_KEY = "kestrel.native_mcp.tool_caller"


def _principal_fields(token):
    return (token.resource, token.claims.get("iss"), token.client_id, token.subject)


class MCPToolCaller:
    """One exact original request; fields and copied contexts confer no grant."""

    def __init__(self, publication, scope):
        self.publication, self.scope = publication, scope
        self.active = False
        self._entered = False
        self._lifetime = ExitStack()
        self._binding = self._principal = self._verified_access = None
        self._delivery = AsyncExitStack()
        self._privacy_epoch = self._delivery_deadline = None
        self._disclosing = self._disclosure_failed = self._delivery_held = False
        self._sent_body_bytes = 0
        self._grant_reader = None

    def require_current(self):
        self.publication.require_current()
        if (not self.active or self.scope.get(_REQUEST_KEY) is not self
                or self not in self.publication._tool_callers):
            raise HostCapabilityAccessDenied("Original native MCP tool request is unavailable")

    async def __aenter__(self):
        if self._entered or self.active or _REQUEST_KEY in self.scope:
            raise HostCapabilityAccessDenied("Native MCP tool request is already bound")
        self._entered = True
        self.publication._tool_callers.add(self)
        self.scope[_REQUEST_KEY], self.active = self, True
        return self

    async def __aexit__(self, *error):
        self.active = False
        self.publication._tool_callers.discard(self)
        if self.scope.get(_REQUEST_KEY) is self:
            self.scope.pop(_REQUEST_KEY)
        try:
            await self._delivery.__aexit__(*error)
        finally:
            self._grant_reader = None
            self._verified_access = None
            self._lifetime.__exit__(*error)

    def __call__(self):
        self.require_current()

    def _check_operation(self):
        self.require_current()
        if (self._delivery_deadline is not None
                and asyncio.get_running_loop().time() >= self._delivery_deadline):
            raise HostCapabilityAccessDenied("Original native MCP disclosure deadline elapsed")
        if (self._disclosure_failed
                or self.publication.agent._get_privacy_policy_epoch() != self._privacy_epoch):
            raise HostCapabilityAccessDenied("Original native MCP disclosure policy changed")
        require_raw_capability_disclosure(self.publication.agent, "MCP memory")

    def _check_disclosure(self):
        self._check_operation()
        if self._disclosing:
            raise HostCapabilityAccessDenied("Original native MCP disclosure is still executing")

    @asynccontextmanager
    async def disclosure(self, required_scope):
        """Capture policy across canonical execution without inverting its locks."""
        self.require_current()
        if (self._binding is None or capture_caller_context_binding() is not self._binding
                or self._verified_access is None or required_scope not in self._verified_access.scopes):
            raise HostCapabilityAccessDenied("Original verified MCP memory-read scope is required for disclosure")
        require_raw_capability_disclosure(self.publication.agent, "MCP memory")
        epoch = self.publication.agent._get_privacy_policy_epoch()
        if self._privacy_epoch is not None and self._privacy_epoch != epoch:
            raise HostCapabilityAccessDenied("Original native MCP disclosure policy changed")
        if self._disclosing:
            raise HostCapabilityAccessDenied("Original native MCP disclosure is already executing")
        if self._delivery_deadline is None:
            expiry = self._verified_access.expires_at
            remaining = expiry - self.publication.runtime.authorization.pending.now(fractional=True) if expiry is not None else 0
            if remaining <= 0:
                raise HostCapabilityAccessDenied("Original bounded MCP memory-read grant expired")
            self._delivery_deadline = (asyncio.get_running_loop().time()
                + min(self.publication.policy.operation_seconds, remaining))
        self._privacy_epoch, self._disclosing = epoch, True
        try:
            async with asyncio.timeout_at(self._delivery_deadline):
                yield
        except BaseException:
            self._disclosure_failed = True
            raise
        finally:
            self._disclosing = False
        self._check_disclosure()

    @asynccontextmanager
    async def mutation(self, required_scope):
        """Commit the canonical handler under the original OAuth SQL lock.

        The caller enters only after canonical journal admission. The handler
        joins this actual primary transaction; neither credentials nor its
        durable receipt can escape before the top-level commit completes.
        """
        self._check_operation()
        original = self._verified_access
        if original is None or required_scope not in original.scopes:
            raise HostCapabilityAccessDenied("Original verified MCP write scope is required")
        async with self.publication.runtime.authorization.grants.access_lifetime(
                original.token, resource=self.publication.resource, client_id=original.client_id) as current:
            if current().model_dump() != original.model_dump():
                raise HostCapabilityAccessDenied("Original native MCP write grant changed")
            self._check_operation()
            yield
            self._check_operation()
            if current().model_dump() != original.model_dump():
                raise HostCapabilityAccessDenied("Original native MCP write grant changed")
        self._check_operation()

    async def _check_current_grant(self):
        original = self._verified_access
        if original is None or self._grant_reader is None:
            raise HostCapabilityAccessDenied("Original verified MCP grant is unavailable")
        current = self._grant_reader()
        if current is None or current.model_dump() != original.model_dump():
            raise HostCapabilityAccessDenied("Original MCP memory-read grant is no longer current")

    async def send(self, message, send):
        """Acquire the leaf privacy gate only after the tool callback finishes."""
        self.require_current()
        if self._privacy_epoch is None:
            await send(message)
            self.require_current()
            return
        self._check_disclosure()
        if message["type"] == "http.response.body":
            self._sent_body_bytes += len(message.get("body", b""))
            if self._sent_body_bytes > self.publication.policy.max_result_bytes:
                raise HostCapabilityAccessDenied("Native MCP private response exceeds the configured byte limit")
        async with asyncio.timeout_at(self._delivery_deadline):
            if not self._delivery_held:
                gate = require_durable_persistence_gate(self.publication.agent)
                await self._delivery.enter_async_context(gate.shared())
                self._delivery_held = True
            if self._grant_reader is None:
                original = self._verified_access
                self._grant_reader = await self._delivery.enter_async_context(
                    self.publication.runtime.authorization.grants.access_lifetime(original.token,
                        resource=self.publication.resource, client_id=original.client_id))
            self._check_disclosure()
            await self._check_current_grant()
            await send(message)
            await self._check_current_grant()
            self._check_disclosure()

    @asynccontextmanager
    async def bind(self, context, required_scope):
        self.require_current()
        request = context.request_context.request
        if (not isinstance(request, Request) or request.scope is not self.scope
                or self.scope["path"] != urlsplit(self.publication.resource).path):
            raise HostCapabilityAccessDenied("Original native MCP SDK request is required")
        native = get_access_token()
        if native is None:
            raise HostCapabilityAccessDenied("Verified native MCP OAuth token is required")
        current = await self.publication.runtime.authorization.verify_token(native.token)
        self.require_current()
        if (current is None or not hmac.compare_digest(current.token, native.token)
                or current.model_dump() != native.model_dump()
                or required_scope not in current.scopes):
            raise HostCapabilityAccessDenied("Current native MCP OAuth scope is required")
        principal = _principal_fields(current)
        if (principal[0] != self.publication.resource
                or principal[1] != self.publication.resource + "/oauth"
                or any(not isinstance(value, str) or not value for value in principal)):
            raise HostCapabilityAccessDenied("Complete native MCP OAuth principal is required")
        if self._principal is not None and self._principal != principal:
            raise HostCapabilityAccessDenied("Original native MCP OAuth principal changed")
        self._verified_access = current
        if self._binding is None:
            identity = hashlib.sha256(json.dumps(principal, separators=(",", ":")).encode()).hexdigest()
            caller = CallerContext.authenticated("mcp:" + identity, auth_method=AuthMethod.MCP_OAUTH)
            self._binding = self._lifetime.enter_context(caller_context_lifetime(caller))
            self._principal = principal
        with caller_context_binding_scope(self._binding) as caller:
            yield caller
            self.require_current()


@asynccontextmanager
async def bind_native_mcp_tool_caller(publication, context, required_scope, *, disclose=False):
    """Use the SDK's original Request and token, never tool-supplied identity."""
    owner = _native_mcp_owner(publication, context)
    async with owner.bind(context, required_scope) as caller:
        if disclose:
            async with owner.disclosure(required_scope):
                yield caller
        else:
            yield caller


def _native_mcp_owner(publication, context):
    request = context.request_context.request
    if not isinstance(request, Request):
        raise HostCapabilityAccessDenied("Original native MCP SDK request is required")
    owner = request.scope.get(_REQUEST_KEY)
    if (type(owner) is not MCPToolCaller or owner.publication is not publication
            or request.scope is not owner.scope):
        raise HostCapabilityAccessDenied("Original native MCP tool request is required")
    owner.require_current()
    return owner


def native_mcp_write_lifetime(publication, context, required_scope):
    """Capture the original SDK request before awaited hooks or admission."""
    owner = _native_mcp_owner(publication, context)
    original_request = context.request_context.request

    def require_original_request():
        owner.require_current()
        if (context.request_context.request is not original_request
                or original_request.scope is not owner.scope):
            raise HostCapabilityAccessDenied("Original native MCP SDK write request changed")

    @asynccontextmanager
    async def lifetime():
        require_original_request()
        async with owner.mutation(required_scope):
            require_original_request()
            yield
            require_original_request()

    return lifetime
