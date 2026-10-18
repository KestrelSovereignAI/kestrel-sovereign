"""DNS-pinned stdlib HTTPS I/O for the did:web fetcher."""
from __future__ import annotations

import errno
from contextlib import ExitStack
import http.client
import socket
import time
from typing import Any
import urllib.request

from kestrel_sdk.security.outbound import SSRFError, ValidatedOutboundURL


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
        if self.sock is not None:
            raise RuntimeError("pinned HTTPS connection is already connected")
        self._validated.assert_origin(self.host, self.port)
        if self._tunnel_host:
            raise SSRFError("pinned HTTPS connection does not allow proxy tunnels")

        timeout = socket.getdefaulttimeout() if self.timeout is socket._GLOBAL_DEFAULT_TIMEOUT else self.timeout
        deadline = None if timeout is None else time.monotonic() + timeout

        def remaining() -> float | None:
            if deadline is None:
                return None
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise TimeoutError("approved-address connection deadline exceeded")
            return budget

        last_error: OSError | None = None
        for ip in self._validated.ip_addresses:
            dial_timeout = remaining()
            try:
                sock = self._create_connection(
                    (str(ip), self.port), dial_timeout, self.source_address,
                )
            except OSError as e:
                last_error = e
                continue
            break
        else:
            remaining()
            if last_error is not None:
                raise last_error
            raise OSError("pinned opener has no validated addresses")

        # Only TCP connection errors may try another approved address. TLS,
        # certificate and configuration errors must fail on the first peer.
        with ExitStack() as pending:
            pending.callback(sock.close)
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError as e:
                if e.errno != errno.ENOPROTOOPT:
                    raise
            sock.settimeout(remaining())
            wrapped = self._context.wrap_socket(sock, server_hostname=self._validated.host)
            if wrapped is not sock:
                pending.callback(wrapped.close)
            remaining()
            wrapped.settimeout(timeout)
            self.sock = wrapped
            pending.pop_all()


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
