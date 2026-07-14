"""Identity→tenant resolution at the host auth edge (issue #2444).

The unified observability store needs a ``tenant_id`` per request, but sovereign
has no organization/tenant model — identity today is only a per-agent DID plus an
authenticated user email (``jwt_payload.sub`` / ``user_email``). This module is
the seam that maps the authenticated principal to a stable ``tenant_id``:

* agent DID → the agent's owner tenant
* user email (cookie / JWT session) → that user's personal tenant
* machine API key (e.g. the talon emitter) → the tenant the key is bound to

**Zero-config default (INV-IDENTITY / INV-SOLO).** With no org/identity
infrastructure, every request resolves to a single stable
:data:`DEFAULT_PERSONAL_TENANT_ID`. The solo owner (sovereign API key, an
internal/anonymous caller, or agent traffic they own) always lands on that one
tenant, so the store never returns zero rows for the legitimate owner and never
leaks across tenants. Only a *distinct authenticated user* (a different
OAuth/JWT email) derives a distinct, deterministic ``uuid5`` tenant, so a
downstream store isolates two principals.

This is the seam Castle later swaps to ``Organization → tenant_id`` (INV-TENANT)
with **no observability code change** — it only replaces
:func:`resolve_tenant` / the injected callable.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

# Stable namespace for deriving per-principal tenant ids. A fixed value so the
# same principal maps to the same tenant across restarts and hosts.
TENANT_NAMESPACE = uuid.UUID("6f3c9d2e-1b4a-5c7e-9a2d-3f8b0c1d2e4f")

#: The single default personal tenant for a zero-config solo deployment. Stable
#: across restarts and derived from :data:`TENANT_NAMESPACE`, so it can never
#: collide with a principal-derived tenant.
DEFAULT_PERSONAL_TENANT_ID: uuid.UUID = uuid.uuid5(
    TENANT_NAMESPACE, "default-personal-tenant"
)

# Host config key the fleet observability host feature reads its resolver from
# (fleet v0.4.0 seam contract, issue #2444).
HOST_CONFIG_KEY = "observability_tenant_resolver"

TenantResolver = Callable[[Any], uuid.UUID]


def tenant_id_for_identity(identity: Optional[str]) -> uuid.UUID:
    """Deterministic tenant id for a principal identity string.

    A blank/``None`` identity — no resolvable distinct principal, i.e. the solo
    owner — maps to :data:`DEFAULT_PERSONAL_TENANT_ID`. Any concrete identity
    gets its own stable ``uuid5`` tenant so distinct principals never share a
    tenant.
    """
    if not identity or not identity.strip():
        return DEFAULT_PERSONAL_TENANT_ID
    return uuid.uuid5(TENANT_NAMESPACE, identity.strip().lower())


def _principal_identity(request: Any) -> Optional[str]:
    """Best-effort distinct-principal identity for a request, or ``None``.

    Prefers the :class:`~kestrel_sovereign.auth.CallerContext` the host auth
    middleware attaches at ``request.state.caller``; falls back to the OAuth
    session cookie when no caller is attached. The sovereign API-key caller (the
    solo owner) and an anonymous/internal caller both resolve to ``None`` (→ the
    default tenant), so only a distinct authenticated user (OAuth/JWT email)
    yields a non-default tenant.
    """
    caller = getattr(getattr(request, "state", None), "caller", None)
    if caller is not None:
        role = getattr(caller, "role", None)
        role_value = getattr(role, "value", role)
        identity = getattr(caller, "identity", None)
        # A distinct authenticated user (OAuth/JWT) is the only principal that
        # gets its own tenant. Sovereign API-key / anonymous / internal callers
        # are the solo owner → default tenant.
        if role_value == "authenticated" and identity:
            return identity
        return None

    # No caller attached yet (resolver invoked outside the auth middleware
    # chain) — fall back to the session cookie directly.
    try:
        email = request.session.get("user_email")
    except Exception:  # noqa: BLE001 - no session middleware / not a Request
        email = None
    return email or None


def resolve_tenant(request: Any) -> uuid.UUID:
    """Resolve the authenticated principal of ``request`` to a ``tenant_id``.

    Always returns a concrete :class:`uuid.UUID` (never ``None``) so INV-SOLO
    holds: the legitimate solo owner is guaranteed a stable, non-empty tenant.
    Matches the fleet seam signature ``Callable[[Request], Optional[uuid.UUID]]``
    (a concrete UUID is a valid return; the fleet store only needs ``None`` to
    trigger its own fallback, which this resolver renders unnecessary).
    """
    return tenant_id_for_identity(_principal_identity(request))


def build_tenant_resolver() -> TenantResolver:
    """Return the resolver callable injected into the host feature config.

    A factory (rather than passing :func:`resolve_tenant` directly) so the
    Castle swap to ``Organization → tenant_id`` can bind configuration here
    without touching any call site.
    """
    return resolve_tenant


__all__ = [
    "DEFAULT_PERSONAL_TENANT_ID",
    "HOST_CONFIG_KEY",
    "TENANT_NAMESPACE",
    "TenantResolver",
    "build_tenant_resolver",
    "resolve_tenant",
    "tenant_id_for_identity",
]
