"""Lighthouse's public gateway, answered in-process (#3601).

:meth:`GatewayServer.install` points ``LIGHTHOUSE_GATEWAY_URL`` at the server
and gives every ``LighthouseGateway`` client (``LighthouseRestClient``'s too)
an in-process transport. Everything after the client is the real code: the
URL, the headers, the byte ceiling, the encoding check and the close. Nothing
reaches the network, and a request to any other host fails the test.
"""

from __future__ import annotations

from typing import Callable, Optional

import httpx

from kestrel_sovereign.storage.providers.lighthouse_rest import LighthouseGateway

GATEWAY_URL = "https://gateway.lighthouse.test/ipfs"
_CHUNK_BYTES = 4096


class GatewayServer:
    """Serves ``lookup(cid)`` by CID to anyone, as the public gateway does;
    a CID ``lookup`` answers None for is a 404."""

    def __init__(self, lookup: Callable[[str], Optional[bytes]], *, url: str = GATEWAY_URL):
        self.url = url
        self._lookup = lookup
        self._prefix = httpx.URL(url).path.rstrip("/") + "/"
        self.requests: list[httpx.Request] = []
        # The bytes streamed for each request, which a bounded read may stop.
        self.sent: list[int] = []
        self.clients: list[httpx.AsyncClient] = []
        # Whether a response says how long it is; a gateway need not.
        self.declare_length = True
        # An error status every request is answered with instead.
        self.failure: Optional[int] = None

    @property
    def cids(self) -> list[str]:
        """The CIDs requested, in order."""
        return [request.url.path[len(self._prefix):] for request in self.requests]

    def handle(self, request: httpx.Request) -> httpx.Response:
        gateway = httpx.URL(self.url)
        if request.url.host != gateway.host or not request.url.path.startswith(self._prefix):
            raise AssertionError(f"request outside the Lighthouse gateway: {request.url}")
        self.requests.append(request)
        self.sent.append(0)
        if self.failure is not None:
            return httpx.Response(self.failure)
        body = self._lookup(request.url.path[len(self._prefix):])
        if body is None:
            return httpx.Response(404)
        index = len(self.sent) - 1

        async def chunks():
            for start in range(0, len(body), _CHUNK_BYTES):
                chunk = body[start:start + _CHUNK_BYTES]
                self.sent[index] += len(chunk)
                yield chunk

        headers = {"Content-Length": str(len(body))} if self.declare_length else {}
        return httpx.Response(200, content=chunks(), headers=headers)

    def install(self, monkeypatch) -> "GatewayServer":
        monkeypatch.setenv("LIGHTHOUSE_GATEWAY_URL", self.url)
        transport = httpx.MockTransport(self.handle)
        server = self

        async def _get_client(gateway: LighthouseGateway) -> httpx.AsyncClient:
            if gateway._client is None or gateway._client.is_closed:
                gateway._client = httpx.AsyncClient(
                    timeout=gateway.timeout, transport=transport
                )
                server.clients.append(gateway._client)
            return gateway._client

        monkeypatch.setattr(LighthouseGateway, "_get_client", _get_client)
        return self
