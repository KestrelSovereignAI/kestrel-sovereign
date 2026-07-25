"""Verification-document lookup for signed A2A envelopes.

This module answers one question only: which DID document should the envelope
verifier use for a claimed sender DID? Local hybrid identities are resolved
from the live :class:`AgentManager`; optional ``did:web`` lookup is a
per-recipient policy.

Peer authorization is deliberately separate. A valid signature proves control
of a DID, not permission to send work to this recipient. Hosted authorization
is enforced after cryptographic verification by
``a2a.inbound_authorization.RecipientA2ASenderAuthorizer``.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


def local_a2a_verification_document(
    agent: Any,
    did: str,
) -> Optional[Mapping[str, Any]]:
    """Build the minimal verification document for one loaded identity."""
    identity = getattr(agent, "identity", None)
    if identity is None or not getattr(identity, "is_hybrid", False):
        return None
    if getattr(identity, "signing_did", None) != did:
        return None
    verification_methods = getattr(
        identity, "new_verification_methods", None
    )
    if not verification_methods:
        return None
    return {
        "id": did,
        "verificationMethod": list(verification_methods),
    }


class HostA2ADidResolver:
    """Resolve DID documents for one recipient's envelope verifier."""

    def __init__(
        self,
        manager: Any,
        *,
        recipient: Any = None,
        federated_fallback: bool = False,
    ):
        self._manager = manager
        self._recipient = recipient
        self._federated_fallback = federated_fallback

    def resolve(self, did: str) -> Optional[Mapping[str, Any]]:
        """Return one unambiguous local document, then optional ``did:web``."""
        if not isinstance(did, str) or not did:
            return None

        local_documents = [
            document
            for agent in self._agents()
            if (document := self._document_for_agent(agent, did)) is not None
        ]
        if len(local_documents) == 1:
            return local_documents[0]
        if len(local_documents) > 1:
            logger.warning(
                "A2A DID resolution refused for %s: multiple loaded agents "
                "claim the signing DID",
                did,
            )
            return None

        if self._allows_federated_fallback():
            return self._resolve_federated(did)
        return None

    @staticmethod
    def _document_for_agent(
        agent: Any,
        did: str,
    ) -> Optional[Mapping[str, Any]]:
        return local_a2a_verification_document(agent, did)

    def _agents(self) -> tuple[Any, ...]:
        agents = self._manager.list_agents()
        iterable = agents.values() if isinstance(agents, dict) else (agents or [])
        return tuple(agent for agent in iterable if agent is not None)

    def _allows_federated_fallback(self) -> bool:
        """Read federation policy from this resolver's recipient."""
        if self._recipient is None:
            return self._federated_fallback
        configured = getattr(
            self._recipient, "a2a_federated_did_fallback", None
        )
        if configured is None:
            return self._federated_fallback
        return configured is True

    @staticmethod
    def _resolve_federated(did: str) -> Optional[Mapping[str, Any]]:
        if not did.startswith("did:web:"):
            return None
        try:
            from kestrel_sovereign.identity.did_web import (
                resolve as did_web_resolve,
            )

            return did_web_resolve(did)
        except Exception as exc:  # noqa: BLE001 - resolution is a trust boundary
            logger.warning(
                "A2A federated did:web resolution failed for %s: %s",
                did,
                exc,
            )
            return None


def install_a2a_did_resolver(
    manager: Any,
    *,
    recipient: Any = None,
    federated_fallback: bool = False,
) -> HostA2ADidResolver:
    """Install a distinct verification resolver on each selected recipient.

    The host registration hook passes only the newly registered ``recipient``.
    Omitting it retains explicit batch installation for standalone embeddings.
    """
    if recipient is not None:
        recipients = (recipient,)
    else:
        agents = manager.list_agents()
        iterable = agents.values() if isinstance(agents, dict) else (agents or [])
        recipients = tuple(agent for agent in iterable if agent is not None)

    first_resolver: Optional[HostA2ADidResolver] = None
    for current_recipient in recipients:
        resolver = HostA2ADidResolver(
            manager,
            recipient=current_recipient,
            federated_fallback=federated_fallback,
        )
        current_recipient.a2a_did_resolver = resolver.resolve
        if first_resolver is None:
            first_resolver = resolver

    if first_resolver is None:
        first_resolver = HostA2ADidResolver(
            manager,
            federated_fallback=federated_fallback,
        )

    logger.info(
        "A2A verification DID resolver installed on %d recipient(s) "
        "(federated_fallback=%s)",
        len(recipients),
        federated_fallback,
    )
    return first_resolver
