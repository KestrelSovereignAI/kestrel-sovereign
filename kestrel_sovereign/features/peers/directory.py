"""Scoped peer-directory and routing boundary for :mod:`peers`.

The Peers feature only accepts an automatic-directory name or slug from a
tool call.  It never turns that value into an address itself: a router first
resolves it in the requester's authorization scope, then every operation is
routed with the resulting stable peer identity.  Hosted implementations must
authorize both resolution/listing *and* the operation that follows it.

``LocalHostPeerDirectory`` discovers the authenticated host roster. All peer
operations use its supplied native A2A session and recipient policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import httpx

from kestrel_sovereign.a2a.transport_auth import A2A_TRANSPORT_KEY_HEADER
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ExecutionCustody,
    execution_work_operation,
    is_execution_control_error,
    owned_native_http_client,
    require_execution_work,
)

PEER_CONNECT_TIMEOUT = 5.0
PEER_READ_TIMEOUT = 300.0


class PeerDirectoryError(RuntimeError):
    """Base error intentionally safe to surface to the requesting agent."""


class PeerDirectoryConfigurationError(PeerDirectoryError):
    """A hosted router was supplied without trusted requester context."""


class PeerNotFoundError(PeerDirectoryError):
    """The peer is absent from the requester's automatic directory."""


class PeerAccessDeniedError(PeerDirectoryError):
    """The router denied an operation in the supplied authorization scope."""


class PeerSelfTargetError(PeerDirectoryError):
    """A resolved automatic peer is the requester itself."""


class PeerUnavailableError(PeerDirectoryError):
    """An authorized peer is currently unavailable."""


class PeerTransportError(PeerDirectoryError):
    """The transport failed without a peer authorization decision."""


class PeerTaskConflictError(PeerDirectoryError):
    """The peer rejected a task operation because its lifecycle already won."""


class PeerProtocolError(PeerDirectoryError):
    """A router or peer returned an invalid protocol payload."""


class PeerSubscriptionUnavailableError(PeerDirectoryError):
    """The peer does not provide task-result subscriptions."""


@dataclass(frozen=True)
class PeerRequester:
    """Trusted context supplied by the embedding host.

    ``authorization_scope`` is intentionally opaque to Kestrel.  A hosted
    runtime creates it after authenticating the request/agent relationship and
    its directory implementation interprets it.  There is no caller-supplied
    user-id parameter anywhere in this contract.
    """

    identity: str
    authorization_scope: object

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise PeerDirectoryConfigurationError(
                "Peer requester identity must be a non-empty stable identity"
            )
        if self.authorization_scope is None:
            raise PeerDirectoryConfigurationError(
                "Peer requester authorization scope is required"
            )


@dataclass(frozen=True)
class PeerIdentity:
    """A scoped directory entry with a stable identity and opaque route key."""

    agent_id: str
    slug: str
    routing_key: str
    name: str = ""
    status: str = "unknown"
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise PeerProtocolError("Peer directory returned an empty agent identity")
        if not isinstance(self.slug, str) or not self.slug.strip():
            raise PeerProtocolError("Peer directory returned an empty peer slug")
        if not isinstance(self.routing_key, str) or not self.routing_key.strip():
            raise PeerProtocolError("Peer directory returned an empty routing key")




class PeerDirectoryRouter(Protocol):
    """Authorize and route operations in one requester's peer directory.

    Implementations MUST authorize ``list_peers``/``resolve_peer`` and MUST
    repeat authorization in every routing operation.  ``PeerIdentity`` values
    can be stale, forged, or retained after an access change; accepting one
    merely because it was previously returned by ``resolve_peer`` is unsafe.
    """

    async def list_peers(
        self, requester: PeerRequester,
    ) -> Sequence[PeerIdentity]:
        """Return only the requester's automatically addressable peers."""

    async def resolve_peer(
        self, requester: PeerRequester, peer_name_or_slug: str,
    ) -> Optional[PeerIdentity]:
        """Resolve an automatic-directory name/slug, or return ``None``."""

    async def resolve_peer_by_agent_id(
        self, requester: PeerRequester, agent_id: str,
    ) -> Optional[PeerIdentity]:
        """Reauthorize one persisted automatic peer by its stable identity.

        This method is intentionally for feature-owned retained state only
        (outbound-task and pending-question records), never a public tool
        input.  It lets a router recover a peer after its display name or slug
        changes while still applying the requester's *current* authorization
        scope before returning a current routing key.
        """

    async def authorize_inbound_sender(
        self,
        requester: PeerRequester,
        sender_id: str,
    ) -> bool:
        """Authorize one cryptographically verified inbound A2A sender.

        This is a distinct trust decision from automatic-peer resolution.
        ``sender_id`` is a loaded sender's stable agent identity, or the
        verified signing DID for an external sender. Implementations must
        evaluate the requester's current scope and return the literal
        singleton ``True`` only when delivery is authorized.
        """


    def open_a2a_session(self, requester: PeerRequester, peer: PeerIdentity):
        """Open an authorized official-SDK session for a stable peer binding."""









class LocalHostPeerDirectory:
    """Discover the local host roster and open its governed native sessions.

    Every operation reauthorizes the stable identity and current route from
    ``/api/agents`` before the host-supplied SDK session can be opened.
    """

    def __init__(
        self,
        host_url: str,
        *,
        transport_key: str = "",
        execution_custody: ExecutionCustody | None = None,
        client_factory: Callable[..., Any] = httpx.AsyncClient,
        native_session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._host_url = host_url.rstrip("/")
        self._execution_custody = execution_custody
        self._transport_key = transport_key
        self._client_factory = client_factory
        # Operational messages use the host-supplied official SDK session.
        self._native_session_factory = native_session_factory

    def bind_native_session_factory(self, factory) -> None:
        """Bind the launcher's session owner once without changing roster origin."""
        if not callable(factory):
            raise PeerDirectoryConfigurationError("Native session factory must be callable")
        if self._native_session_factory is not None:
            raise PeerDirectoryConfigurationError("Native session factory is already bound")
        self._native_session_factory = factory

    @property
    def has_native_session_factory(self) -> bool:
        """Whether the embedding has already supplied its session authority."""
        return self._native_session_factory is not None

    @asynccontextmanager
    async def open_a2a_session(self, requester: PeerRequester, peer: PeerIdentity):
        self._require_requester(requester)
        authorized_peer = await self._authorize_peer(requester, peer)
        if self._native_session_factory is None:
            raise PeerDirectoryConfigurationError("Host has no native A2A session factory")
        async with self._native_session_factory(requester, authorized_peer) as session:
            yield session

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._transport_key:
            headers[A2A_TRANSPORT_KEY_HEADER] = self._transport_key
        return headers

    @staticmethod
    def _require_requester(requester: PeerRequester) -> None:
        # ``PeerRequester`` validates construction.  This defensive check
        # catches duck-typed/legacy callers before any HTTP route is attempted.
        if not isinstance(requester, PeerRequester):
            raise PeerDirectoryConfigurationError(
                "Peer routing requires a trusted requester identity and scope"
            )

    @staticmethod
    def _as_mapping(response: Any, *, action: str) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise PeerProtocolError(
                f"Local peer host returned malformed JSON while {action}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise PeerProtocolError(
                f"Local peer host returned a non-object payload while {action}"
            )
        return payload

    @staticmethod
    def _raise_for_route_status(response: Any, *, action: str) -> None:
        status_code = response.status_code
        if status_code >= 400:
            try:
                payload = response.json()
            except (TypeError, ValueError):
                payload = None
            error = payload.get("error") if isinstance(payload, Mapping) else None
            if isinstance(error, Mapping) and error.get("code") == "execution_control":
                details = error.get("details")
                outcome = (details[0].get("commit_outcome") if isinstance(details, list)
                    and len(details) == 1 and isinstance(details[0], Mapping) else None)
                if isinstance(outcome, str) and outcome in {"unknown", "committed"}:
                    raise ExecutionCommitOutcomeError(outcome)
                raise ExecutionAuthorityError("peer execution stopped; reconcile before continuing")
        if status_code == 404:
            raise PeerNotFoundError("Peer is not in the automatic directory")
        if status_code == 503:
            raise PeerUnavailableError("Peer is unavailable")
        if status_code in (401, 403):
            raise PeerAccessDeniedError("Peer operation is not authorized")
        if status_code == 409:
            raise PeerTaskConflictError(
                "Peer task state conflicts with the requested operation"
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise PeerTransportError(
                f"Local peer host rejected {action} (HTTP {status_code})"
            ) from exc

    @execution_work_operation
    async def _directory_entries(self, requester: PeerRequester) -> list[PeerIdentity]:
        self._require_requester(requester)
        try:
            async with owned_native_http_client(self._client_factory()) as client:
                require_execution_work(self)
                response = await client.get(
                    f"{self._host_url}/api/agents",
                    headers=self._headers(),
                    timeout=PEER_CONNECT_TIMEOUT,
                )
                self._raise_for_route_status(response, action="listing peers")
                raw = response.json()
        except PeerDirectoryError:
            raise
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            if is_execution_control_error(exc):
                raise
            raise PeerTransportError("Could not connect to local peer host") from exc
        except (TypeError, ValueError) as exc:
            raise PeerProtocolError("Local peer host returned malformed peer listing") from exc

        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, Mapping):
            entries = raw.get("agents", [])
        else:
            entries = None
        if not isinstance(entries, list):
            raise PeerProtocolError("Local peer host returned malformed peer listing")

        peers: list[PeerIdentity] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise PeerProtocolError("Local peer host returned an invalid roster entry")
            display_name = entry.get("name")
            routing_key = entry.get("routing_name") or display_name or entry.get("id")
            agent_id = entry.get("id")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (agent_id, routing_key)
            ):
                raise PeerProtocolError("Local peer roster requires an explicit stable identity and routing key")
            slug = str(routing_key)
            peers.append(PeerIdentity(
                agent_id=str(agent_id),
                slug=slug,
                routing_key=str(routing_key),
                name=str(display_name or slug),
                status=str(entry.get("status") or "unknown"),
                description=str(entry.get("description") or ""),
            ))
        return peers

    @execution_work_operation
    async def list_peers(self, requester: PeerRequester) -> Sequence[PeerIdentity]:
        return await self._directory_entries(requester)

    @execution_work_operation
    async def resolve_peer(
        self, requester: PeerRequester, peer_name_or_slug: str,
    ) -> Optional[PeerIdentity]:
        if not isinstance(peer_name_or_slug, str) or not peer_name_or_slug.strip():
            return None
        needle = peer_name_or_slug.casefold()
        peers = await self._directory_entries(requester)
        slug_matches = [peer for peer in peers if peer.slug.casefold() == needle]
        matches = slug_matches or [
            peer for peer in peers if peer.name.casefold() == needle
        ]
        # An ambiguous display name must not select an arbitrary peer.
        return matches[0] if len(matches) == 1 else None

    @execution_work_operation
    async def resolve_peer_by_agent_id(
        self, requester: PeerRequester, agent_id: str,
    ) -> Optional[PeerIdentity]:
        """Resolve persisted identity through the current local directory.

        The caller supplies this value only from feature-owned durable state;
        this adapter still reloads the directory so an access revocation,
        deletion, or routing-key change takes effect before any URL is built.
        """
        if not isinstance(agent_id, str) or not agent_id.strip():
            return None
        matches = [
            peer
            for peer in await self._directory_entries(requester)
            if peer.agent_id == agent_id
        ]
        # A stable identity must remain unique within a scoped directory.
        return matches[0] if len(matches) == 1 else None

    @execution_work_operation
    async def authorize_inbound_sender(
        self,
        requester: PeerRequester,
        sender_id: str,
    ) -> bool:
        """Authorize a verified sender through the live local directory."""
        if not isinstance(sender_id, str) or not sender_id.strip():
            return False
        matches = [
            peer
            for peer in await self._directory_entries(requester)
            if peer.agent_id == sender_id
        ]
        return len(matches) == 1

    @execution_work_operation
    async def _authorize_peer(
        self, requester: PeerRequester, peer: PeerIdentity,
    ) -> PeerIdentity:
        """Re-resolve a peer immediately before an operation.

        The local host has no tenant policy of its own, but it still must
        honour the router contract: a caller can retain or forge a
        ``PeerIdentity`` after a prior lookup.  Fetching the directory again
        makes removal or routing-key changes take effect before the URL is
        constructed, and returning the current entry ensures we never route
        using caller-provided identity fields.
        """
        if not isinstance(peer, PeerIdentity):
            raise PeerProtocolError("Peer route requires a valid peer identity")
        matches = [
            candidate
            for candidate in await self._directory_entries(requester)
            if (
                candidate.agent_id == peer.agent_id
                and candidate.routing_key == peer.routing_key
            )
        ]
        if len(matches) != 1:
            raise PeerNotFoundError("Peer is not in the automatic directory")
        return matches[0]
