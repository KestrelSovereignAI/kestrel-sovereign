"""Source-host gating for the bootstrap API-key endpoint (#1724).

``/api/auth/key`` hands out the live API key for first-run frontend setup. It
previously trusted the ENTIRE Docker bridge range ``172.16.0.0/12`` (~1M
addresses), so any co-resident container on a shared host could fetch sovereign
credentials. This narrows the trust to loopback + the Docker *gateway* only,
with an explicit operator allowlist for custom network topologies — not the
whole bridge subnet. A request that carries forwarding headers is refused
whatever its client host, since a proxy forwarded it for someone else (#3905).
"""
from __future__ import annotations

import os
from collections.abc import Iterable

# Loopback plus the DEFAULT Docker bridge gateway (172.17.0.1). NOT the whole
# 172.16.0.0/12 range — a sibling container must not be trusted by default.
_DEFAULT_BOOTSTRAP_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "172.17.0.1"})

# Headers an intermediary adds when it forwards a request for another client.
# A browser or CLI on this host sends none of them, so their presence means
# the request did not originate here, whatever the server made of them (#3905).
_FORWARDING_HEADERS = frozenset({
    "forwarded",  # RFC 7239
    "via",  # RFC 9110 §7.6.3
    "x-real-ip",  # nginx
    "x-client-ip",
    "x-cluster-client-ip",
    "x-original-forwarded-for",
    "true-client-ip",  # Akamai, Cloudflare
    "cf-connecting-ip",  # Cloudflare, including cloudflared tunnels
    "fastly-client-ip",
})
_FORWARDING_HEADER_PREFIX = "x-forwarded-"  # -for, -host, -proto, -port, ...


def bootstrap_allowed_hosts() -> set[str]:
    """The exact set of source hosts allowed to fetch the bootstrap key.

    Operators on a non-default Docker network (custom gateway) add their gateway
    address via ``KESTREL_BOOTSTRAP_ALLOWED_HOSTS`` (comma-separated) rather than
    re-opening the whole bridge subnet.
    """
    extra = os.environ.get("KESTREL_BOOTSTRAP_ALLOWED_HOSTS", "")
    return set(_DEFAULT_BOOTSTRAP_HOSTS) | {
        h.strip() for h in extra.split(",") if h.strip()
    }


def is_bootstrap_host_allowed(client_host: str | None) -> bool:
    """Exact-match the request's client host against the allowed set.

    The client host is the server's, not necessarily the socket peer: uvicorn
    replaces a ``127.0.0.1`` peer with the client one ``X-Forwarded-For`` names.
    """
    if not client_host:
        return False
    return client_host in bootstrap_allowed_hosts()


def forwarding_header_names(header_names: Iterable[str]) -> list[str]:
    """The forwarding headers among ``header_names``, lowercased and sorted.

    The client host alone cannot carry this: uvicorn names the forwarded client
    only for one ``X-Forwarded-For`` line from ``127.0.0.1``. Two lines, an RFC
    7239 ``Forwarded`` header, or a proxy connecting from ``::1`` leave the
    loopback peer as the client, so a remote caller would pass the host check.
    """
    found = {
        name.lower()
        for name in header_names
        if name.lower() in _FORWARDING_HEADERS
        or name.lower().startswith(_FORWARDING_HEADER_PREFIX)
    }
    return sorted(found)
