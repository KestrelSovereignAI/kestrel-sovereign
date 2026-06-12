"""SSRF guard for server-side outbound fetches (#1727).

Several endpoints fetch a URL whose host is influenced by a client/peer: the
avatar set-from-URL endpoint and (when ``KESTREL_A2A_FEDERATED_DID`` is on) the
``did:web`` resolver. Without a guard, ``http://169.254.169.254/...`` or an
internal address lets an attacker make the server hit cloud metadata / internal
services. This module resolves the target host and rejects any address that
falls in a private / loopback / link-local / reserved range (which covers the
common metadata endpoints, e.g. 169.254.169.254 and IPv6 ULAs).

It blocks the literal-private-IP and metadata-hostname cases. A determined
DNS-rebinding attacker can still race resolution vs. connection (TOCTOU); pinning
the validated IP into the connection is tracked as a follow-up — this guard
closes the directly-exploitable holes the review flagged.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse


class SSRFError(ValueError):
    """Raised when an outbound URL targets a disallowed (non-public) address."""


def _address_is_blocked(ip: ipaddress._BaseAddress) -> bool:
    """True for any address that must not be reachable from a server-side fetch.

    ``is_private`` already covers RFC1918, loopback, link-local (169.254/16 — the
    cloud-metadata range), unique-local IPv6 (fd00::/8), and more on modern
    Python; we add the remaining non-global categories explicitly for clarity and
    forward-compatibility.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


def _resolve_addresses(host: str, port: int) -> "list[ipaddress._BaseAddress]":
    """Resolve ``host`` to all IPs (blocking). Raises SSRFError on failure."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise SSRFError(f"cannot resolve host {host!r}: {e}") from e
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def validate_outbound_url(url: str, *, allowed_schemes: "tuple[str, ...]" = ("http", "https")) -> None:
    """Validate ``url`` for a server-side fetch; raise :class:`SSRFError` if unsafe.

    Synchronous (uses ``socket.getaddrinfo``). Prefer :func:`assert_safe_url` in
    async code so DNS resolution doesn't block the event loop.
    """
    # Malformed URLs (bad port, invalid IPv6 literal, …) make urlparse/.port
    # raise ValueError; convert to SSRFError so callers get a clean rejection
    # (400) instead of an uncaught 500 (codex r1).
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as e:
        raise SSRFError(f"malformed URL {url!r}: {e}") from e

    if scheme not in allowed_schemes:
        raise SSRFError(
            f"URL scheme {parsed.scheme!r} not allowed (allowed: {allowed_schemes})"
        )
    if not host:
        raise SSRFError(f"URL has no host: {url!r}")

    # Literal IP host: validate directly (no DNS).
    try:
        ip = ipaddress.ip_address(host)
        if _address_is_blocked(ip):
            raise SSRFError(f"URL host {host} is a non-public address")
        return
    except ValueError:
        pass  # not a literal IP — resolve it

    for ip in _resolve_addresses(host, port):
        if _address_is_blocked(ip):
            raise SSRFError(
                f"URL host {host!r} resolves to a non-public address ({ip})"
            )


async def assert_safe_url(url: str, *, allowed_schemes: "tuple[str, ...]" = ("http", "https")) -> None:
    """Async wrapper: validate ``url`` off the event loop (DNS in a thread)."""
    await asyncio.get_running_loop().run_in_executor(
        None, lambda: validate_outbound_url(url, allowed_schemes=allowed_schemes)
    )
