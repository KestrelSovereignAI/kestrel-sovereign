"""`GET /api/auth/vouch`: the host's half of the operator lane (#3233).

Loopback only, unauthenticated, answers a fresh nonce with the HMAC only
the holder of the host's stable sovereign key can produce; refuses when
the host runs on a temporary generated key.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient

from kestrel_sovereign.security.operator_lane import vouch_response


def _prepare_app():
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = None
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


NONCE = "0123456789abcdef" * 2


def test_vouch_answers_the_nonce_under_the_stable_key_over_loopback_only():
    app, original = _prepare_app()
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "stable-key-3233", "KESTREL_REQUIRE_OAUTH": "false"}):
            with TestClient(app, client=("127.0.0.1", 55000)) as client:
                ok = client.get("/api/auth/vouch", params={"nonce": NONCE})
                bad_nonce = client.get("/api/auth/vouch", params={"nonce": "zz"})
                # Public path: no credential required, and none is echoed.
                assert "stable-key-3233" not in ok.text
            with TestClient(app, client=("203.0.113.10", 55000)) as client:
                remote = client.get("/api/auth/vouch", params={"nonce": NONCE})
        assert ok.status_code == 200, ok.text
        assert ok.json() == {"vouch": vouch_response("stable-key-3233", NONCE)}
        assert bad_nonce.status_code == 400
        assert remote.status_code == 403
    finally:
        _restore_app(app, original)


def test_an_ephemeral_key_does_not_vouch():
    """A host that generated its own key has no durable sovereign."""
    from kestrel_sovereign.security.sovereign_key import mark_ephemeral_sovereign_key

    app, original = _prepare_app()
    try:
        mark_ephemeral_sovereign_key("generated-at-boot")
        with patch.dict("os.environ", {"KESTREL_API_KEY": "generated-at-boot", "KESTREL_REQUIRE_OAUTH": "false"}):
            with TestClient(app, client=("127.0.0.1", 55000)) as client:
                response = client.get("/api/auth/vouch", params={"nonce": NONCE})
        assert response.status_code == 404
    finally:
        _restore_app(app, original)
