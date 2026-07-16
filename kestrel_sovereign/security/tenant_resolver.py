"""Identity→tenant resolution at the host auth edge (issue #2444).

The unified observability store needs a ``tenant_id`` per request, but sovereign
has no organization/tenant model — identity today is only a per-agent DID plus an
authenticated user email (``jwt_payload.sub`` / ``user_email``). This module is
the seam that maps the authenticated principal to a stable ``tenant_id``:

* agent DID → the agent's owner tenant
* user email (cookie / JWT session) → that user's personal tenant
* machine API key (e.g. the talon emitter) → the tenant the key is bound to

**Zero-config default (INV-IDENTITY / INV-SOLO).** With no org/identity
infrastructure, **every** request — including one carrying an authenticated
OAuth/JWT email — resolves to the single stable
:data:`DEFAULT_PERSONAL_TENANT_ID`. Single-tenant is the default; multi-tenancy
is strictly additive. This keeps the solo owner and the local API-key emitters
(talon, per-agent hook) on the *same* tenant, so the logged-in owner's browser
sees the events those emitters posted instead of a fail-closed empty store
(issue #2554).

Per-principal isolation is **opt-in**, engaged only when multi-tenancy is
actually configured:

* a Castle ``Organization → tenant_id`` provider is bound via
  :func:`bind_org_tenant_provider` (the real multi-tenant path,
  kestrel-castle#18), or
* the env flag :data:`MULTITENANT_ENV_VAR` (``KESTREL_OBSERVABILITY_MULTITENANT``)
  is truthy.

When multi-tenancy is on, a *distinct authenticated user* (a different OAuth/JWT
email) derives a distinct, deterministic ``uuid5`` tenant (or the Castle
provider's tenant), so a downstream store isolates two principals.

This is the seam Castle later swaps to ``Organization → tenant_id`` (INV-TENANT)
with **no observability code change** — it only binds a provider here or
replaces the injected callable.
"""

from __future__ import annotations

import os
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

#: Env flag that opts a deployment into per-principal tenant isolation. Truthy
#: values (``1``/``true``/``yes``/``on``) enable multi-tenant resolution; unset
#: or falsey keeps the single-tenant default (INV-SOLO, issue #2554).
MULTITENANT_ENV_VAR = "KESTREL_OBSERVABILITY_MULTITENANT"

TenantResolver = Callable[[Any], uuid.UUID]

#: Optional Castle-bound ``identity → tenant_id`` provider. When set, it is both
#: the multi-tenant *signal* and the source of truth for a principal's tenant
#: (``Organization → tenant_id``). ``None`` means no org model is bound.
OrgTenantProvider = Callable[[str], Optional[uuid.UUID]]
_org_tenant_provider: Optional[OrgTenantProvider] = None


def bind_org_tenant_provider(provider: Optional[OrgTenantProvider]) -> None:
    """Bind (or clear) the Castle ``Organization → tenant_id`` provider.

    Binding a provider both **enables** multi-tenant resolution and supplies the
    authoritative tenant for a principal. Pass ``None`` to unbind and fall back
    to the single-tenant default (unless :data:`MULTITENANT_ENV_VAR` is set).
    """
    global _org_tenant_provider
    _org_tenant_provider = provider


def _multitenant_enabled() -> bool:
    """True when per-principal isolation is opted into (provider or env flag)."""
    if _org_tenant_provider is not None:
        return True
    return os.environ.get(MULTITENANT_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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

    Single-tenant by default (INV-SOLO, issue #2554): with no multi-tenant
    config every request — including one carrying an authenticated email —
    resolves to :data:`DEFAULT_PERSONAL_TENANT_ID`, so the solo owner shares one
    tenant with the local API-key emitters and sees their events.

    Per-principal isolation is engaged only when multi-tenancy is opted into (a
    bound Castle provider or :data:`MULTITENANT_ENV_VAR`). When on, a distinct
    authenticated principal derives a distinct tenant — from the Castle
    ``Organization → tenant_id`` provider when bound, else a deterministic
    ``uuid5``.

    Always returns a concrete :class:`uuid.UUID` (never ``None``, fail-closed).
    Matches the fleet seam signature ``Callable[[Request], Optional[uuid.UUID]]``.
    """
    if not _multitenant_enabled():
        return DEFAULT_PERSONAL_TENANT_ID

    identity = _principal_identity(request)
    if not identity or not identity.strip():
        return DEFAULT_PERSONAL_TENANT_ID

    if _org_tenant_provider is not None:
        tenant = _org_tenant_provider(identity)
        if tenant is not None:
            return tenant

    return tenant_id_for_identity(identity)


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
    "MULTITENANT_ENV_VAR",
    "TENANT_NAMESPACE",
    "OrgTenantProvider",
    "TenantResolver",
    "bind_org_tenant_provider",
    "build_tenant_resolver",
    "resolve_tenant",
    "tenant_id_for_identity",
]
