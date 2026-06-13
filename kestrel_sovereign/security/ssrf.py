"""SSRF guard for server-side outbound fetches (#1727).

Several endpoints fetch a URL whose host is influenced by a client/peer: the
avatar set-from-URL endpoint and (when ``KESTREL_A2A_FEDERATED_DID`` is on) the
``did:web`` resolver. Without a guard, ``http://169.254.169.254/...`` or an
internal address lets an attacker make the server hit cloud metadata / internal
services. This module resolves the target host and rejects any address that
falls in a private / loopback / link-local / reserved range (which covers the
common metadata endpoints, e.g. 169.254.169.254 and IPv6 ULAs).

Validation returns the full list of public IPs selected for the outbound connection.
Callers that fetch the URL must use the pinned transport/opener helpers below
so DNS is not repeated after validation (DNS-rebinding TOCTOU, #1746).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import errno
import http.client
import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse
import urllib.request


class SSRFError(ValueError):
    """Raised when an outbound URL targets a disallowed (non-public) address."""


@dataclass(frozen=True, init=False)
class ValidatedOutboundURL:
    """URL validation result with resolved IPs that must be used to connect."""

    url: str
    scheme: str
    host: str
    port: int
    ip_addresses: tuple[ipaddress._BaseAddress, ...]

    def __init__(
        self,
        *,
        url: str,
        scheme: str,
        host: str,
        port: int,
        ip_addresses: tuple[ipaddress._BaseAddress, ...] | None = None,
        ip_address: ipaddress._BaseAddress | None = None,
    ) -> None:
        if ip_addresses is None:
            if ip_address is None:
                raise TypeError("ip_addresses or ip_address is required")
            ip_addresses = (ip_address,)
        elif ip_address is not None:
            raise TypeError("pass either ip_addresses or ip_address, not both")
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "ip_addresses", ip_addresses)

    @property
    def ip_address(self) -> ipaddress._BaseAddress:
        """First validated address, retained for compatibility."""
        return self.ip_addresses[0]


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


def validate_outbound_url(
    url: str, *, allowed_schemes: "tuple[str, ...]" = ("http", "https")
) -> ValidatedOutboundURL:
    """Validate ``url`` for a server-side fetch; raise :class:`SSRFError` if unsafe.

    Synchronous (uses ``socket.getaddrinfo``). Prefer :func:`assert_safe_url` in
    async code so DNS resolution doesn't block the event loop. The returned
    :class:`ValidatedOutboundURL` contains only public IPs that passed validation;
    fetchers must connect to those IPs instead of resolving the hostname again.
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
        return ValidatedOutboundURL(
            url=url, scheme=scheme, host=host, port=port, ip_addresses=(ip,)
        )
    except ValueError:
        pass  # not a literal IP — resolve it

    resolved = _resolve_addresses(host, port)
    if not resolved:
        raise SSRFError(f"cannot resolve host {host!r}: no addresses returned")
    for ip in resolved:
        if _address_is_blocked(ip):
            raise SSRFError(
                f"URL host {host!r} resolves to a non-public address ({ip})"
            )
    return ValidatedOutboundURL(
        url=url, scheme=scheme, host=host, port=port, ip_addresses=tuple(resolved)
    )


async def assert_safe_url(
    url: str, *, allowed_schemes: "tuple[str, ...]" = ("http", "https")
) -> ValidatedOutboundURL:
    """Async wrapper: validate ``url`` off the event loop (DNS in a thread)."""
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: validate_outbound_url(url, allowed_schemes=allowed_schemes)
    )


class _PinnedAsyncNetworkBackend:
    """httpcore network backend that dials validated IPs for one origin."""

    def __init__(self, validated: ValidatedOutboundURL):
        from httpcore._backends.anyio import AnyIOBackend

        self._validated = validated
        self._backend = AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ):
        if host != self._validated.host or port != self._validated.port:
            raise SSRFError(
                f"pinned transport refused connection to unexpected origin {host}:{port}"
            )
        last_error: Exception | None = None
        for ip in self._validated.ip_addresses:
            try:
                return await self._backend.connect_tcp(
                    host=str(ip),
                    port=port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except Exception as e:
                last_error = e
        if last_error is not None:
            raise last_error
        raise SSRFError("pinned transport has no validated addresses")

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ):
        raise SSRFError("pinned transport does not allow Unix-socket connections")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


def pinned_httpx_async_transport(validated: ValidatedOutboundURL):
    """Build an httpx async transport that connects to validated IPs.

    The request URL remains unchanged, so httpcore still uses the original host
    for the HTTP Host header, TLS SNI, and certificate hostname verification.
    """
    import httpcore
    import httpx
    from httpx._config import DEFAULT_LIMITS, create_ssl_context

    class _PinnedAsyncHTTPTransport(httpx.AsyncHTTPTransport):
        def __init__(self) -> None:
            ssl_context = create_ssl_context(verify=True, cert=None, trust_env=True)
            self._pool = httpcore.AsyncConnectionPool(
                ssl_context=ssl_context,
                max_connections=DEFAULT_LIMITS.max_connections,
                max_keepalive_connections=DEFAULT_LIMITS.max_keepalive_connections,
                keepalive_expiry=DEFAULT_LIMITS.keepalive_expiry,
                http1=True,
                http2=False,
                network_backend=_PinnedAsyncNetworkBackend(validated),
            )

    return _PinnedAsyncHTTPTransport()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that dials pinned IPs but verifies the original host."""

    def __init__(
        self,
        host: str,
        *args: Any,
        validated: ValidatedOutboundURL,
        **kwargs: Any,
    ) -> None:
        self._validated = validated
        super().__init__(host, *args, **kwargs)

    def connect(self) -> None:
        if self.host != self._validated.host or self.port != self._validated.port:
            raise OSError(
                f"pinned opener refused connection to unexpected origin "
                f"{self.host}:{self.port}"
            )

        if self._tunnel_host:
            server_hostname = self._tunnel_host
        else:
            server_hostname = self.host

        last_error: Exception | None = None
        for ip in self._validated.ip_addresses:
            sock = None
            try:
                sock = self._create_connection(
                    (str(ip), self.port),
                    self.timeout,
                    self.source_address,
                )
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError as e:
                    if e.errno != errno.ENOPROTOOPT:
                        raise

                self.sock = sock
                if self._tunnel_host:
                    self._tunnel()
                self.sock = self._context.wrap_socket(
                    self.sock, server_hostname=server_hostname
                )
                return
            except Exception as e:
                last_error = e
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                self.sock = None
        if last_error is not None:
            raise last_error
        raise OSError("pinned opener has no validated addresses")


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, validated: ValidatedOutboundURL):
        super().__init__()
        self._validated = validated

    def https_open(self, req):
        def connection_factory(host: str, *args: Any, **kwargs: Any):
            return _PinnedHTTPSConnection(
                host, *args, validated=self._validated, **kwargs
            )

        return self.do_open(connection_factory, req, context=self._context)


def pinned_urllib_https_opener(validated: ValidatedOutboundURL, *handlers: Any):
    """Build a direct urllib opener that pins HTTPS TCP dials to validated IP."""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        *handlers,
        _PinnedHTTPSHandler(validated),
    )
