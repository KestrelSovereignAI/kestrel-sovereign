"""Exercise the optional tracing service through the real host proxy."""

import asyncio
import os
from html.parser import HTMLParser
import socket
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request

from kestrel_sovereign.phoenix_supervisor import PhoenixSupervisor, proxy_to_phoenix


class _Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            source = dict(attrs).get("src")
            if source is not None:
                self.sources.append(source)


def _ports():
    with socket.socket() as http, socket.socket() as grpc:
        http.bind(("127.0.0.1", 0))
        grpc.bind(("127.0.0.1", 0))
        return http.getsockname()[1], grpc.getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_live_phoenix_assets_survive_the_same_origin_proxy(tmp_path, monkeypatch):
    pytest.importorskip("phoenix")
    # Only this isolated test process and its child receive these settings.
    for key in list(os.environ):
        if key.startswith(("PHOENIX_", "OTEL_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "1")
    monkeypatch.setenv("PHOENIX_TELEMETRY_ENABLED", "false")
    http_port, grpc_port = _ports()
    supervisor = PhoenixSupervisor(
        working_dir=tmp_path / "phoenix", port=http_port, grpc_port=grpc_port,
    )
    app = FastAPI()

    @app.get("/phoenix/{path:path}")
    async def proxy(request: Request, path: str):
        return await proxy_to_phoenix(request, supervisor, path)

    try:
        assert await asyncio.to_thread(supervisor.start, timeout=45)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://host.test",
        ) as client:
            index = await client.get("/phoenix/")
            assert index.status_code == 200
            assert "text/html" in index.headers["content-type"]
            scripts = _Scripts()
            scripts.feed(index.text)
            assets = [
                source for source in scripts.sources
                if urlsplit(source).path.startswith("/phoenix/assets/")
                and urlsplit(source).path.endswith(".js")
            ]
            assert assets, "Phoenix must advertise its script below the proxy prefix"
            asset = await client.get(assets[0])
            assert asset.status_code == 200
            assert "javascript" in asset.headers["content-type"]
            assert asset.content
    finally:
        try:
            await supervisor.aclose()
        finally:
            await asyncio.to_thread(supervisor.stop)
