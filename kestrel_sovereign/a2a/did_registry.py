"""Host-level DID resolver for A2A sender verification (#1705).

The signed-envelope verification from #1673 verifies a sender's signature
against the sender's DID document, behind an injectable ``agent.a2a_did_resolver``.
This module provides that resolver for a multi_agent host.

In a multi_agent host the sender and receiver share the host, but agents are
isolated — a receiver agent can't reach a sibling's identity directly. The
HOST holds every loaded agent via the :class:`AgentManager`, so the host builds
the same-host DID registry and injects the resolver into each agent.

**Topology (the #1673/#1705 design): local same-host resolution by default,
federated ``did:web`` optional, NEVER required.** ``_resolve_local`` reads peer
agents' in-memory hybrid verification methods (``RuntimeIdentity`` —
``signing_did`` + ``new_verification_methods``); no network. The optional
``did:web`` network fallback is off unless ``federated_fallback=True``. A DID
that resolves to neither returns ``None``, which the verifier treats as
"unsigned" (back-compat) unless ``KESTREL_A2A_REQUIRE_SIGNED`` is set.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


class HostA2ADidResolver:
    """Resolve a peer agent's DID document for A2A signature verification.

    Resolution is live against the :class:`AgentManager`, so it reflects
    agents added or rotated after startup without a rebuild.
    """

    def __init__(self, manager: Any, *, federated_fallback: bool = False):
        self._manager = manager
        self._federated_fallback = federated_fallback

    def resolve(self, did: str) -> Optional[Mapping[str, Any]]:
        doc = self._resolve_local(did)
        if doc is not None:
            return doc
        if self._federated_fallback:
            return self._resolve_federated(did)
        return None

    def _resolve_local(self, did: str) -> Optional[Mapping[str, Any]]:
        if not did:
            return None
        agents = self._manager.list_agents()
        # AgentManager.list_agents() returns {name: agent}; tolerate a plain
        # iterable of agents too.
        iterable = agents.values() if isinstance(agents, dict) else (agents or [])
        for agent in iterable:
            ident = getattr(agent, "identity", None)
            if ident is None or not getattr(ident, "is_hybrid", False):
                continue
            if getattr(ident, "signing_did", None) != did:
                continue
            vms = getattr(ident, "new_verification_methods", None)
            if vms:
                # Reconstruct the minimal resolvable document the verifier
                # needs: id (for the sender-binding check) + the Multikey
                # verification methods (for the hybrid signature check).
                return {"id": did, "verificationMethod": list(vms)}
        return None

    def _resolve_federated(self, did: str) -> Optional[Mapping[str, Any]]:
        if not str(did).startswith("did:web:"):
            return None
        try:
            from kestrel_sovereign.identity.did_web import resolve as did_web_resolve
            return did_web_resolve(did)
        except Exception as exc:  # noqa: BLE001 - resolution failure is non-fatal
            logger.warning("A2A federated did:web resolution failed for %s: %s", did, exc)
            return None


def install_a2a_did_resolver(manager: Any, *, federated_fallback: bool = False) -> HostA2ADidResolver:
    """Build the host resolver and inject it into every loaded agent.

    Each agent gets ``agent.a2a_did_resolver = resolver.resolve`` — the seam the
    /tasks/send verification (#1673) consumes. Returns the resolver so callers
    can hold a reference (e.g. for tests or later re-injection).
    """
    resolver = HostA2ADidResolver(manager, federated_fallback=federated_fallback)
    agents = manager.list_agents()
    iterable = agents.values() if isinstance(agents, dict) else (agents or [])
    count = 0
    for agent in iterable:
        if agent is not None:
            agent.a2a_did_resolver = resolver.resolve
            count += 1
    logger.info(
        "A2A DID resolver installed on %d agent(s) (federated_fallback=%s)",
        count, federated_fallback,
    )
    return resolver
