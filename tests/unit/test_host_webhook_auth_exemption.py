"""#2091/P3: an external, keyless webhook caller must be able to reach a
SPECIFIC agent's webhook on a heterogeneous multi-agent host via the documented
``/api/agents/{id}/webhooks/{name}`` path. The host auth middleware previously
only exempted the bare ``/webhooks/`` prefix, so that agent-prefixed path 401'd
and the documented escape hatch was unusable. Webhooks self-authenticate (HMAC),
so the exemption is safe.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from kestrel_sovereign.host import WEBHOOK_PATH_RE, auth_middleware


def test_webhook_path_re_matches_bare_and_agent_prefixed():
    assert WEBHOOK_PATH_RE.match("/webhooks/stripe")
    assert WEBHOOK_PATH_RE.match("/api/agents/nellie/webhooks/stripe")
    assert WEBHOOK_PATH_RE.match("/api/agents/emma/webhooks/rest/webhook")
    # NOT a webhook path — a normal per-agent API route stays protected.
    assert not WEBHOOK_PATH_RE.match("/api/agents/nellie/api/conversations")
    assert not WEBHOOK_PATH_RE.match("/api/agents/nellie/webhookslookalike")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("KESTREL_API_KEY", "host-key")
    a = FastAPI()
    a.middleware("http")(auth_middleware)
    a.add_middleware(SessionMiddleware, secret_key="test-secret",
                     session_cookie="kestrel_session")

    @a.post("/api/agents/{agent_id}/webhooks/{name}")
    async def _wh(agent_id: str, name: str):
        return {"agent": agent_id, "name": name}

    @a.get("/api/agents/{agent_id}/api/conversations")
    async def _protected(agent_id: str):
        return {"ok": True}

    return a


def test_agent_prefixed_webhook_is_reachable_without_host_key(app):
    """The per-agent webhook path is exempt from host API-key auth (external
    callers like Stripe have no key; the webhook's own HMAC is the real gate)."""
    client = TestClient(app)
    resp = client.post("/api/agents/nellie/webhooks/stripe")
    assert resp.status_code == 200, resp.status_code
    assert resp.json() == {"agent": "nellie", "name": "stripe"}


def test_non_webhook_agent_route_still_requires_key(app):
    """The exemption is scoped to webhook paths — a normal per-agent API route is
    still protected."""
    client = TestClient(app)
    assert client.get("/api/agents/nellie/api/conversations").status_code == 401
    assert (
        client.get(
            "/api/agents/nellie/api/conversations",
            headers={"X-API-Key": "host-key"},
        ).status_code
        == 200
    )
