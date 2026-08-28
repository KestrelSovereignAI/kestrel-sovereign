"""Scoped peer-directory and routing boundary for :mod:`peers`.

The Peers feature only accepts an automatic-directory name or slug from a
tool call.  It never turns that value into an address itself: a router first
resolves it in the requester's authorization scope, then every operation is
routed with the resulting stable peer identity.  Hosted implementations must
authorize both resolution/listing *and* the operation that follows it.

``LocalHostPeerDirectory`` preserves the existing in-process multi-agent HTTP
host behaviour.  It is deliberately an adapter, not a hosted tenancy policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Mapping, Optional, Protocol, Sequence
from urllib.parse import quote

import httpx


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


@dataclass(frozen=True)
class PeerSubscriptionEvent:
    """One raw A2A SSE frame, normalized for every router implementation."""

    event: str
    data: str


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

    async def invoke(
        self, requester: PeerRequester, peer: PeerIdentity, message: str,
    ) -> Mapping[str, Any]:
        """Authorize and synchronously invoke an already-resolved peer."""

    async def send_a2a_task(
        self,
        requester: PeerRequester,
        peer: PeerIdentity,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Authorize and deliver one signed A2A task envelope exactly once."""

    async def get_a2a_task(
        self, requester: PeerRequester, peer: PeerIdentity, task_id: str,
    ) -> Mapping[str, Any]:
        """Authorize and fetch one routed A2A task result."""

    async def cancel_a2a_task(
        self,
        requester: PeerRequester,
        peer: PeerIdentity,
        task_id: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Authorize and deliver one signed cancellation to its recipient."""

    def subscribe_a2a_task(
        self,
        requester: PeerRequester,
        peer: PeerIdentity,
        task_id: str,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[PeerSubscriptionEvent]:
        """Authorize and stream routed A2A task status events."""


async def iter_sse_events(response: Any) -> AsyncIterator[PeerSubscriptionEvent]:
    """Parse an HTTP SSE response into protocol-neutral subscription events."""
    event_name: Optional[str] = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if event_name is not None or data_lines:
                yield PeerSubscriptionEvent(
                    event=event_name or "message",
                    data="\n".join(data_lines),
                )
            event_name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    if event_name is not None or data_lines:
        yield PeerSubscriptionEvent(
            event=event_name or "message", data="\n".join(data_lines),
        )


class LocalHostPeerDirectory:
    """Default adapter for the existing local multi-agent HTTP host.

    It deliberately resolves a public name/slug from ``/api/agents`` before
    each route rather than interpolating a user-provided target into a URL.
    That preserves local compatibility while exercising the same
    resolve-then-route boundary hosted adapters must enforce.
    """

    def __init__(
        self,
        host_url: str,
        *,
        api_key: str = "",
        client_factory: Callable[..., Any] = httpx.AsyncClient,
    ) -> None:
        self._host_url = host_url.rstrip("/")
        self._api_key = api_key
        self._client_factory = client_factory

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
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

    async def _directory_entries(self, requester: PeerRequester) -> list[PeerIdentity]:
        self._require_requester(requester)
        try:
            async with self._client_factory() as client:
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
                continue
            display_name = entry.get("name")
            routing_key = entry.get("routing_name") or display_name or entry.get("id")
            agent_id = entry.get("id") or routing_key
            if not all(
                isinstance(value, str) and value.strip()
                for value in (agent_id, routing_key)
            ):
                continue
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

    async def list_peers(self, requester: PeerRequester) -> Sequence[PeerIdentity]:
        return await self._directory_entries(requester)

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

    async def invoke(
        self, requester: PeerRequester, peer: PeerIdentity, message: str,
    ) -> Mapping[str, Any]:
        self._require_requester(requester)
        try:
            authorized_peer = await self._authorize_peer(requester, peer)
            async with self._client_factory() as client:
                response = await client.post(
                    self._peer_url(authorized_peer, "api/agent/invoke"),
                    json={"input": message},
                    headers=self._headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_READ_TIMEOUT,
                        write=PEER_READ_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
                self._raise_for_route_status(response, action="invoking peer")
                return self._as_mapping(response, action="invoking peer")
        except PeerDirectoryError:
            raise
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise PeerTransportError("Could not reach local peer host") from exc

    async def send_a2a_task(
        self,
        requester: PeerRequester,
        peer: PeerIdentity,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._require_requester(requester)
        try:
            authorized_peer = await self._authorize_peer(requester, peer)
            async with self._client_factory() as client:
                response = await client.post(
                    self._peer_url(authorized_peer, "api/agent/tasks/send"),
                    json=dict(payload),
                    headers=self._headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_READ_TIMEOUT,
                        write=PEER_READ_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
                self._raise_for_route_status(response, action="sending A2A task")
                return self._as_mapping(response, action="sending A2A task")
        except PeerDirectoryError:
            raise
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise PeerTransportError("Could not reach local peer host") from exc

    async def get_a2a_task(
        self, requester: PeerRequester, peer: PeerIdentity, task_id: str,
    ) -> Mapping[str, Any]:
        self._require_requester(requester)
        try:
            authorized_peer = await self._authorize_peer(requester, peer)
            async with self._client_factory() as client:
                response = await client.get(
                    self._peer_url(
                        authorized_peer,
                        f"api/agent/tasks/{quote(task_id, safe='')}",
                    ),
                    headers=self._headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_CONNECT_TIMEOUT,
                        write=PEER_CONNECT_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
                self._raise_for_route_status(response, action="fetching A2A task")
                return self._as_mapping(response, action="fetching A2A task")
        except PeerDirectoryError:
            raise
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise PeerTransportError("Could not reach local peer host") from exc

    async def cancel_a2a_task(
        self,
        requester: PeerRequester,
        peer: PeerIdentity,
        task_id: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._require_requester(requester)
        try:
            authorized_peer = await self._authorize_peer(requester, peer)
            async with self._client_factory() as client:
                response = await client.post(
                    self._peer_url(
                        authorized_peer,
                        f"api/agent/tasks/{quote(task_id, safe='')}/cancel",
                    ),
                    json=dict(payload),
                    headers=self._headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_READ_TIMEOUT,
                        write=PEER_READ_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
                self._raise_for_route_status(
                    response, action="canceling A2A task"
                )
                return self._as_mapping(response, action="canceling A2A task")
        except PeerDirectoryError:
            raise
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise PeerTransportError("Could not reach local peer host") from exc

    async def subscribe_a2a_task(
        self,
        requester: PeerRequester,
        peer: PeerIdentity,
        task_id: str,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[PeerSubscriptionEvent]:
        self._require_requester(requester)
        authorized_peer = await self._authorize_peer(requester, peer)
        timeout = httpx.Timeout(
            connect=min(PEER_CONNECT_TIMEOUT, timeout_seconds),
            read=timeout_seconds,
            write=min(PEER_CONNECT_TIMEOUT, timeout_seconds),
            pool=min(PEER_CONNECT_TIMEOUT, timeout_seconds),
        )
        try:
            async with self._client_factory(timeout=timeout) as client:
                async with client.stream(
                    "GET",
                    self._peer_url(
                        authorized_peer,
                        f"api/agent/tasks/{quote(task_id, safe='')}/subscribe",
                    ),
                    headers=self._headers(),
                ) as response:
                    if response.status_code == 404:
                        raise PeerSubscriptionUnavailableError(
                            "Peer does not expose A2A task subscriptions"
                        )
                    self._raise_for_route_status(response, action="subscribing to A2A task")
                    async for event in iter_sse_events(response):
                        yield event
        except PeerDirectoryError:
            raise
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            raise PeerTransportError("A2A subscription transport failed") from exc

    def _peer_url(self, peer: PeerIdentity, suffix: str) -> str:
        return (
            f"{self._host_url}/api/agents/"
            f"{quote(peer.routing_key, safe='')}/{suffix}"
        )
