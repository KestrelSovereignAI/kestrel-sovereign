"""Authenticated agent-boot resolution for semantic assertion authority."""

from __future__ import annotations

from kestrel_sovereign.identity.runtime_identity import (
    AgentIdentity,
    _loader_verified_identity_binding,
)
from kestrel_sovereign.storage.async_assertion_store import (
    _AssertionTenantCapability,
    _issue_assertion_tenant_capability,
)


def _resolve_authenticated_agent_assertion_capability(
    authenticated_agent_did: str,
    agent_identity: AgentIdentity | None,
) -> _AssertionTenantCapability | None:
    """Bind assertion authority after the agent boot path resolves its DID.

    This private resolver is intentionally separate from storage construction:
    a raw backend or a public ``AsyncStorage(agent_id=...)`` call cannot turn an
    arbitrary string into semantic-tenant authority.  It accepts only the
    loader-attested cryptographic identity that Kestrel's boot sequence
    verified against persisted key material and its DID document.  A
    pre-inception agent has no authority until its identity is established.
    """
    if not isinstance(authenticated_agent_did, str) or not authenticated_agent_did.startswith("did:"):
        raise ValueError("authenticated semantic assertion tenants require a DID")
    if agent_identity is None:
        return None
    binding = _loader_verified_identity_binding(agent_identity)
    if binding is None:
        raise TypeError(
            "semantic assertion authority requires a loader-verified AgentIdentity"
        )
    if authenticated_agent_did not in binding.dids:
        raise ValueError("agent identity is not bound to the semantic assertion tenant")
    return _issue_assertion_tenant_capability(authenticated_agent_did)
