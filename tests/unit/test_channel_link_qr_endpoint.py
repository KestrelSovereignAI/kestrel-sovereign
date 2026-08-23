"""``GET /api/agent/channels/{channel_type}/link-qr.png`` — serves the pairing
QR PNG that an isolated channel feature (e.g. WhatsApp) pushed to the host.

The chat's persisted ``channel_link`` card (#2081, ``channelLinkPartRenderer`` in
chat.js) fetches this over http because the in-chat sanitizer strips ``data:``
image URIs. These tests drive the handler against a stub agent whose
``storage_path`` points at a temp data dir.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kestrel_sovereign.endpoints.files as files_endpoint


def _app_with_agent(agent) -> FastAPI:
    app = FastAPI()
    app.include_router(files_endpoint.router)

    @app.middleware("http")
    async def _attach_agent(request, call_next):
        # The agent-routing middleware normally sets request.state.agent;
        # bypass it here since the handler reads get_agent(request) directly.
        request.state.agent = agent
        return await call_next(request)

    return app


def _agent(tmp_path):
    agent = MagicMock()
    agent.storage_path = str(tmp_path / "agent" / "kestrel_prime.db")
    return agent


def test_serves_png_when_present(tmp_path):
    agent = _agent(tmp_path)
    art = tmp_path / "agent" / "channel_link_artifacts"
    art.mkdir(parents=True)
    png = b"\x89PNG\r\n\x1a\n" + b"qr-bytes"
    (art / "whatsapp_link_qr.png").write_bytes(png)

    with TestClient(_app_with_agent(agent)) as client:
        resp = client.get("/api/agent/channels/whatsapp/link-qr.png")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert "no-store" in resp.headers["cache-control"]
    assert resp.content == png


def test_404_when_no_qr(tmp_path):
    agent = _agent(tmp_path)
    with TestClient(_app_with_agent(agent)) as client:
        resp = client.get("/api/agent/channels/whatsapp/link-qr.png")
    assert resp.status_code == 404


@pytest.mark.skipif(os.name != "posix", reason="O_NOFOLLOW contract is POSIX-only")
def test_hosted_qr_refuses_tenant_controlled_symlink(tmp_path):
    runtime_root = tmp_path / "hosted-runtime"
    agent = SimpleNamespace(
        did="did:test:hosted-channel",
        storage_path=None,
        isolated_runtime_root=runtime_root,
        isolated_runtime_namespace="agent-hosted-channel",
        isolated_runtime_hosted=True,
    )
    artifacts = (
        runtime_root / "agent-hosted-channel" / "channel_link_artifacts"
    )
    artifacts.mkdir(parents=True)
    victim = tmp_path / "another-tenant-qr.png"
    victim.write_bytes(b"another tenant's pairing secret")
    (artifacts / "whatsapp_link_qr.png").symlink_to(victim)

    with TestClient(_app_with_agent(agent)) as client:
        resp = client.get("/api/agent/channels/whatsapp/link-qr.png")

    assert resp.status_code == 404
    assert victim.read_bytes() not in resp.content


def test_serves_png_from_explicit_hosted_runtime_namespace(tmp_path):
    runtime_root = tmp_path / "hosted-runtime"
    agent = SimpleNamespace(
        did="did:test:hosted-channel",
        storage_path=None,
        isolated_runtime_root=runtime_root,
        isolated_runtime_namespace="agent-hosted-channel",
        isolated_runtime_hosted=True,
    )
    artifact = (
        runtime_root
        / "agent-hosted-channel"
        / "channel_link_artifacts"
        / "whatsapp_link_qr.png"
    )
    artifact.parent.mkdir(parents=True)
    png = b"\x89PNG\r\n\x1a\nhosted-qr"
    artifact.write_bytes(png)

    with TestClient(_app_with_agent(agent)) as client:
        resp = client.get("/api/agent/channels/whatsapp/link-qr.png")

    assert resp.status_code == 200
    assert resp.content == png
    assert not (
        runtime_root / "agent-hosted-channel" / ".kestrel-runtime-owner"
    ).exists()


def test_hosted_get_does_not_materialize_missing_runtime_namespace(tmp_path):
    runtime_root = tmp_path / "hosted-runtime"
    agent = SimpleNamespace(
        did="did:test:hosted-channel",
        storage_path=None,
        isolated_runtime_root=runtime_root,
        isolated_runtime_namespace="agent-hosted-channel",
        isolated_runtime_hosted=True,
    )

    with TestClient(_app_with_agent(agent)) as client:
        resp = client.get("/api/agent/channels/whatsapp/link-qr.png")

    assert resp.status_code == 404
    assert not runtime_root.exists()


def test_rejects_invalid_channel_type(tmp_path):
    agent = _agent(tmp_path)
    with TestClient(_app_with_agent(agent)) as client:
        # uppercase / punctuation fails the [a-z0-9_] guard (also a traversal guard)
        resp = client.get("/api/agent/channels/..%2Fetc/link-qr.png")
    assert resp.status_code in (400, 404)
