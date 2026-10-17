"""DNS-pinned stdlib HTTPS I/O for the did:web fetcher."""
from __future__ import annotations

import errno
from contextlib import ExitStack
import http.client
import socket
from typing import Any
import urllib.request

from kestrel_sdk.security.outbound import ValidatedOutboundURL


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
        self._validated.assert_origin(self.host, self.port)

        if self._tunnel_host:
            server_hostname = self._tunnel_host
        else:
            server_hostname = self.host

        last_error: OSError | None = None
        for ip in self._validated.ip_addresses:
            connected = False
            try:
                with ExitStack() as pending:
                    sock = self._create_connection(
                        (str(ip), self.port),
                        self.timeout,
                        self.source_address,
                    )
                    pending.callback(sock.close)
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
                    pending.pop_all()
                    connected = True
                    return
            except OSError as e:
                last_error = e
            finally:
                if not connected:
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
