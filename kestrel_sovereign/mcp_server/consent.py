"""Request-owned approval of one pending request, never a durable OAuth grant."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from kestrel_sovereign.auth import AuthMethod, caller_identity_digest
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.request_auth import (
    InteractiveRequestAuthentication,
    current_request_authentication,
)
from kestrel_sovereign.security.sovereign_key import (
    is_ephemeral_sovereign_key,
    sovereign_key_fingerprint,
)

from .pending import PendingMCPAuthorization
from .policy import MCPServerPolicy
from .records import hash_mcp_secret


def _request_digest(pending: PendingMCPAuthorization) -> str:
    value = [pending.client_id, pending.expires_at, pending.parameters.model_dump(mode="json")]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _approval_fields(approval: MCPConsentApproval) -> tuple:
    return (approval.client_id, approval.resource, approval.scopes,
        approval.subject, approval.credential_fingerprint)


@dataclass(frozen=True)
class MCPConsentApproval:
    """A view of one issued decision; copied fields are not authority."""

    client_id: str
    resource: str
    scopes: tuple[str, ...]
    subject: str
    credential_fingerprint: str = field(repr=False)
    _authority: MCPConsentAuthority = field(repr=False, compare=False)

    def require_current(self) -> None:
        self._authority.require_approval(self)

    def require_pending(self, handle: str, pending: PendingMCPAuthorization) -> None:
        self._authority.require_approval(self, handle=handle, pending=pending)


@dataclass(frozen=True)
class _ApprovalBinding:
    approval: MCPConsentApproval
    fields: tuple
    authentication: InteractiveRequestAuthentication
    pending_hash: str
    request_digest: str
    expires_at: int


class MCPConsentAuthority:
    """Own issued decision identities while borrowing native host authentication.

    The embedding host holds its original agent/privacy authority reader through
    commit. The registry binds an exact approval object and its immutable fields;
    a descriptor or a copied approval cannot recreate that decision.
    """

    def __init__(self, agent_id: str, resource_url: str, policy: MCPServerPolicy, *,
                 require_current: Callable[[], None], sovereign_key: Callable[[], str | None],
                 clock: Callable[[], float] = time.time):
        if not resource_url.startswith(policy.console_origin + "/"):
            raise ValueError("MCP consent must target the configured host")
        self.agent_id, self.resource_url, self.policy = agent_id, resource_url, policy
        self._require_current, self._sovereign_key, self.clock = require_current, sovereign_key, clock
        self._approvals: dict[int, _ApprovalBinding] = {}

    def require_current_host(self) -> None:
        self._require_current()

    def current_sovereign_fingerprint(self) -> str | None:
        self.require_current_host()
        key = self._sovereign_key()
        if not key or is_ephemeral_sovereign_key(key):
            return None
        return sovereign_key_fingerprint(key)

    def require_approval(self, approval: MCPConsentApproval, *, handle: str | None = None,
                         pending: PendingMCPAuthorization | None = None) -> MCPConsentApproval:
        self.require_current_host()
        binding = self._approvals.get(id(approval))
        if (binding is None or binding.approval is not approval or approval._authority is not self
                or binding.fields != _approval_fields(approval)):
            raise HostCapabilityAccessDenied("The original MCP consent request ended or its decision changed")
        if current_request_authentication() is not binding.authentication:
            raise HostCapabilityAccessDenied("The original MCP consent request ended")
        binding.authentication.require_current()
        fingerprint = self.current_sovereign_fingerprint()
        if (fingerprint is None or not hmac.compare_digest(fingerprint, binding.fields[4])):
            raise HostCapabilityAccessDenied("Durable original sovereign credentials are required")
        now = self.clock()
        if not math.isfinite(now) or now < 0 or binding.expires_at <= now:
            raise HostCapabilityAccessDenied("The pending MCP consent request expired")
        if (handle is None) != (pending is None):
            raise TypeError("The original MCP pending handle and request must be checked together")
        if pending is not None and (not hmac.compare_digest(binding.pending_hash, hash_mcp_secret(handle))
                or not hmac.compare_digest(binding.request_digest, _request_digest(pending))):
            raise HostCapabilityAccessDenied("The original MCP consent request changed")
        return binding.approval

    @contextmanager
    def approve(self, handle: str, pending: PendingMCPAuthorization,
                scopes: tuple[str, ...]) -> Iterator[MCPConsentApproval]:
        authentication = current_request_authentication()
        if authentication is None:
            raise HostCapabilityAccessDenied("Interactive MCP consent is required")
        caller = authentication.caller
        if not caller.is_sovereign or caller.auth_method is not AuthMethod.API_KEY:
            raise HostCapabilityAccessDenied("Sovereign MCP consent is required")
        client = self.policy.registered_client(pending.client_id)
        if (client is None or pending.parameters.resource != self.resource_url
                or str(pending.parameters.redirect_uri) not in client.redirect_uris
                or type(scopes) is not tuple or not scopes or len(set(scopes)) != len(scopes)
                or not set(scopes) <= set(pending.parameters.scopes or ()) <= set(client.scopes)):
            raise HostCapabilityAccessDenied("MCP consent requires registered requested scopes and resource")
        approval = MCPConsentApproval(pending.client_id, self.resource_url, scopes,
            caller_identity_digest(caller), caller.credential_fingerprint, self)
        binding = _ApprovalBinding(approval, _approval_fields(approval), authentication,
            hash_mcp_secret(handle), _request_digest(pending), pending.expires_at)
        self._approvals[id(approval)] = binding
        try:
            self.require_approval(approval)
            yield approval
            self.require_approval(approval)
        finally:
            del self._approvals[id(approval)]
