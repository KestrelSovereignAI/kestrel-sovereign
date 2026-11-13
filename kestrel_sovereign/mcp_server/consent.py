"""Request-owned approval of one pending request, never a durable OAuth grant."""

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
from kestrel_sovereign.security.request_auth import current_request_authentication
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


@dataclass(frozen=True)
class MCPConsentApproval:
    """Finite original request authority; serialized fields cannot recreate it."""

    client_id: str
    resource: str
    scopes: tuple[str, ...]
    subject: str
    credential_fingerprint: str = field(repr=False)
    _pending_hash: str = field(repr=False)
    _request_digest: str = field(repr=False)
    _require_current: Callable[[], None] = field(repr=False, compare=False)

    def require_current(self) -> None:
        self._require_current()

    def require_pending(self, handle: str, pending: PendingMCPAuthorization) -> None:
        """Recheck the actual SQL request before the consent owner consumes it."""
        self.require_current()
        if (not hmac.compare_digest(self._pending_hash, hash_mcp_secret(handle))
                or not hmac.compare_digest(self._request_digest, _request_digest(pending))):
            raise HostCapabilityAccessDenied("The original MCP consent request changed")


@contextmanager
def approve_pending_request(handle: str, pending: PendingMCPAuthorization,
        scopes: tuple[str, ...], policy: MCPServerPolicy, *,
        resource_url: str,
        require_current: Callable[[], None], sovereign_key: Callable[[], str | None],
        clock: Callable[[], float] = time.time) -> Iterator[MCPConsentApproval]:
    """Borrow current Console authentication and the target host's authority lease.

    The host must hold its original agent/privacy authority reader through SQL
    commit. Only the existing sovereign API-key lane can delegate durable agent
    access here; ordinary signed-in users are not promoted to sovereigns.
    """
    authentication = current_request_authentication()
    if authentication is None:
        raise HostCapabilityAccessDenied("Interactive MCP consent is required")
    caller = authentication.caller
    if not caller.is_sovereign or caller.auth_method is not AuthMethod.API_KEY:
        raise HostCapabilityAccessDenied("Sovereign MCP consent is required")
    client = policy.registered_client(pending.client_id)
    if (client is None or pending.parameters.resource != resource_url
            or not resource_url.startswith(policy.console_origin + "/")
            or str(pending.parameters.redirect_uri) not in client.redirect_uris
            or type(scopes) is not tuple or not scopes or len(set(scopes)) != len(scopes)
            or not set(scopes) <= set(pending.parameters.scopes or ()) <= set(client.scopes)):
        raise HostCapabilityAccessDenied("MCP consent requires registered requested scopes and resource")
    active = True

    def check() -> None:
        if not active or current_request_authentication() is not authentication:
            raise HostCapabilityAccessDenied("The original MCP consent request ended")
        require_current()
        authentication.require_current()
        key = sovereign_key()
        if (not key or is_ephemeral_sovereign_key(key)
                or not caller.credential_fingerprint
                or not hmac.compare_digest(caller.credential_fingerprint, sovereign_key_fingerprint(key))):
            raise HostCapabilityAccessDenied("Durable original sovereign credentials are required")
        now = clock()
        if not math.isfinite(now) or now < 0 or pending.expires_at <= now:
            raise HostCapabilityAccessDenied("The pending MCP consent request expired")

    check()
    approval = MCPConsentApproval(pending.client_id, pending.parameters.resource, scopes,
        caller_identity_digest(caller), caller.credential_fingerprint,
        hash_mcp_secret(handle), _request_digest(pending), check)
    try:
        yield approval
        check()
    finally:
        active = False
