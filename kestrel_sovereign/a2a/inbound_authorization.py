"""Recipient-scoped authorization for cryptographically verified A2A senders.

Cryptographic DID verification and recipient authorization are independent
trust decisions. The endpoint first verifies the signed envelope, then calls
this authorizer with the verified sender DID before it marks the sender
verified or creates a task.

Hosted scope is monotonic for one installed authorizer. Removing or corrupting
the injected router/requester pair is a revocation and cannot restore the
standalone shared-API-key fallback.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Optional

from kestrel_sovereign.features.peers.directory import (
    PeerDirectoryError,
    PeerRequester,
)

logger = logging.getLogger(__name__)

SCOPED_POLICY_MARKER = "_a2a_inbound_scoped_policy_required"


def has_a2a_inbound_scoped_policy(recipient: Any) -> bool:
    """Return authoritative monotonic hosted-policy evidence."""
    return getattr(recipient, SCOPED_POLICY_MARKER, False) is True


def mark_a2a_inbound_scoped_policy(
    recipient: Any,
    *,
    required: bool,
) -> bool:
    """Monotonically record authoritative hosted policy on the recipient.

    Registration/onboarding owns this evidence. Once set, removing every live
    router/requester/authorizer seam cannot make the recipient look standalone.
    Passing ``required=False`` never clears prior evidence.
    """
    if required:
        setattr(recipient, SCOPED_POLICY_MARKER, True)
    return has_a2a_inbound_scoped_policy(recipient)


class RecipientA2ASenderAuthorizer:
    """Authorize verified sender DIDs under one recipient's live peer scope."""

    def __init__(
        self,
        manager: Any,
        *,
        recipient: Any,
        sender_id_resolver: Optional[Callable[[str], Optional[str]]] = None,
    ):
        self._manager = manager
        self._recipient = recipient
        self._sender_id_resolver = sender_id_resolver
        recipient_declares_scope = (
            getattr(recipient, "peer_directory_router", None) is not None
            or getattr(recipient, "peer_requester", None) is not None
        )
        self._scoped_policy_required = mark_a2a_inbound_scoped_policy(
            recipient,
            required=recipient_declares_scope,
        )

    @property
    def requires_verified_sender(self) -> bool:
        """Whether direct unsigned use of this recipient policy is forbidden.

        Hosted requests report ``True`` here.  Their one legacy compatibility
        exception is manager-owned and additionally proves an exact current
        non-hybrid local sender before this authorizer sees its stable id.
        """
        self._observe_live_scope()
        return self._scoped_policy_required

    def has_valid_current_scope(self) -> bool:
        """Whether the recipient's live scoped seam is complete and bound."""
        return self._scoped_context() is not None

    def has_valid_policy_scope(self, router: Any, requester: Any) -> bool:
        """Validate one manager-owned immutable policy context."""
        return self._validate_scoped_context(router, requester) is not None

    async def authorize_with_policy(
        self,
        verified_sender_did: str,
        *,
        router: Any,
        requester: Any,
    ) -> bool:
        """Authorize against manager-owned context, never mutable agent attrs."""
        return (
            await self.authorize_principal_with_policy(
                verified_sender_did,
                router=router,
                requester=requester,
            )
            is not None
        )

    async def authorize_principal_with_policy(
        self,
        verified_sender_did: str,
        *,
        router: Any,
        requester: Any,
    ) -> Optional[str]:
        """Authorize and return the stable principal bound by host policy."""

        if not isinstance(verified_sender_did, str) or not verified_sender_did:
            return None
        context = self._validate_scoped_context(router, requester)
        if context is None:
            return None
        return await self._authorize_principal_with_context(
            verified_sender_did,
            context,
        )

    async def authorize_legacy_local_sender_with_policy(
        self,
        sender_id: str,
        *,
        router: Any,
        requester: Any,
    ) -> bool:
        """Authorize one exact loaded pre-ceremony sender under host policy.

        Unsigned A2A traffic has no signing DID to resolve.  Hosted callers
        may use this narrowly scoped seam only after the manager has proved
        that ``sender_id`` names a current same-host non-hybrid agent.  The
        directory remains the user-scope authority, so this method never
        restores the old global same-host fallback.
        """
        if not isinstance(sender_id, str) or not sender_id:
            return False
        context = self._validate_scoped_context(router, requester)
        if context is None:
            return False
        return await self._authorize_sender_id_with_context(sender_id, context)

    async def authorize(self, verified_sender_did: str) -> bool:
        """Authorize a sender only after its signature has been verified."""
        return await self.authorize_principal(verified_sender_did) is not None

    async def authorize_principal(
        self,
        verified_sender_did: str,
    ) -> Optional[str]:
        """Authorize and return the stable principal for durable task state."""

        if not isinstance(verified_sender_did, str) or not verified_sender_did:
            return None

        context = self._scoped_context()
        if context is None:
            # True standalone compatibility is allowed only when this
            # authorizer has never observed hosted scope.
            return verified_sender_did if not self._scoped_policy_required else None

        return await self._authorize_principal_with_context(
            verified_sender_did,
            context,
        )

    async def _authorize_principal_with_context(
        self,
        verified_sender_did: str,
        context: tuple[Any, PeerRequester],
    ) -> Optional[str]:
        sender_id = self._sender_directory_id(verified_sender_did)
        if sender_id is None:
            return None

        authorized = await self._authorize_sender_id_with_context(sender_id, context)
        return sender_id if authorized else None

    async def _authorize_sender_id_with_context(
        self,
        sender_id: str,
        context: tuple[Any, PeerRequester],
    ) -> bool:
        """Ask the validated directory whether one stable sender is in scope."""
        router, requester = context

        try:
            result = router.authorize_inbound_sender(requester, sender_id)
            if inspect.isawaitable(result):
                result = await result
        except PeerDirectoryError as exc:
            logger.info(
                "Inbound A2A sender %s denied by recipient directory: %s",
                sender_id,
                exc,
            )
            return False
        except Exception:  # noqa: BLE001 - injected provider boundary
            logger.warning(
                "Inbound A2A sender authorization failed for %s",
                sender_id,
                exc_info=True,
            )
            return False
        # Require the explicit singleton, not merely a truthy provider value.
        return result is True

    def _scoped_context(self) -> Optional[tuple[Any, PeerRequester]]:
        self._observe_live_scope()
        router = getattr(self._recipient, "peer_directory_router", None)
        requester = getattr(self._recipient, "peer_requester", None)
        return self._validate_scoped_context(router, requester)

    def _validate_scoped_context(
        self,
        router: Any,
        requester: Any,
    ) -> Optional[tuple[Any, PeerRequester]]:
        if router is None or requester is None:
            return None
        if not isinstance(requester, PeerRequester):
            return None
        if not callable(getattr(router, "authorize_inbound_sender", None)):
            return None

        recipient_id = self._stable_agent_id(self._recipient)
        if recipient_id is None or requester.identity != recipient_id:
            logger.warning(
                "Inbound A2A requester identity %r does not match recipient "
                "stable identity %r",
                requester.identity,
                recipient_id,
            )
            return None
        return router, requester

    def _observe_live_scope(self) -> None:
        recipient_declares_scope = (
            getattr(self._recipient, "peer_directory_router", None) is not None
            or getattr(self._recipient, "peer_requester", None) is not None
        )
        if mark_a2a_inbound_scoped_policy(
            self._recipient,
            required=recipient_declares_scope,
        ):
            self._scoped_policy_required = True

    def _sender_directory_id(self, signing_did: str) -> Optional[str]:
        """Map a loaded signing DID to its stable id; retain external DIDs."""
        if self._sender_id_resolver is not None:
            try:
                sender_id = self._sender_id_resolver(signing_did)
            except Exception:  # noqa: BLE001 - launcher registry boundary
                logger.warning(
                    "Inbound A2A signing-DID mapping failed for %s",
                    signing_did,
                    exc_info=True,
                )
                return None
            if isinstance(sender_id, str) and sender_id:
                return sender_id
            return None
        matches = []
        for agent in self._agents():
            identity = getattr(agent, "identity", None)
            if getattr(identity, "signing_did", None) == signing_did:
                matches.append(agent)
        if len(matches) > 1:
            logger.warning(
                "Inbound A2A authorization refused for %s: multiple loaded "
                "agents claim the signing DID",
                signing_did,
            )
            return None
        if len(matches) == 1:
            return self._stable_agent_id(matches[0])
        # An unloaded/federated sender has no sound local mapping. Its verified
        # signing DID is the stable identity the inbound provider must decide.
        return signing_did

    def _agents(self) -> tuple[Any, ...]:
        agents = self._manager.list_agents()
        iterable = agents.values() if isinstance(agents, dict) else (agents or [])
        return tuple(agent for agent in iterable if agent is not None)

    @staticmethod
    def _stable_agent_id(agent: Any) -> Optional[str]:
        for attribute in ("agent_id", "did"):
            value = getattr(agent, attribute, None)
            if isinstance(value, str) and value:
                return value
        return None


def install_a2a_inbound_sender_authorizer(
    manager: Any,
    *,
    recipient: Any,
    sender_id_resolver: Optional[Callable[[str], Optional[str]]] = None,
) -> RecipientA2ASenderAuthorizer:
    """Install the explicit inbound authorization seam on one recipient."""
    authorizer = RecipientA2ASenderAuthorizer(
        manager,
        recipient=recipient,
        sender_id_resolver=sender_id_resolver,
    )
    recipient.a2a_inbound_sender_authorizer = authorizer
    logger.info(
        "Inbound A2A sender authorizer installed for recipient %r "
        "(scoped=%s)",
        getattr(recipient, "agent_id", None)
        or getattr(recipient, "did", None)
        or "unknown",
        authorizer.requires_verified_sender,
    )
    return authorizer
