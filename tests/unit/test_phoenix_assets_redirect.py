"""Root-absolute Phoenix chunk requests redirect into the /phoenix cookie scope.

Phoenix's Vite bundle resolves *dynamically imported* chunks from the
build-time base — root-absolute ``/assets/…`` — which lands outside the
``/phoenix`` embed-cookie scope and 401s (observed live: ``vendor-shiki``,
``rolldown-runtime``). The host answers with an unauthenticated 307 back under
``/phoenix`` where the authenticated proxy serves the real chunk.
"""

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("KESTREL_API_KEY", "test-key-123")

import kestrel_sovereign.server as srv  # noqa: E402


class _FakeSupervisor:
    pass


@pytest.fixture()
def client():
    with TestClient(srv.app, raise_server_exceptions=False) as c:
        yield c


def test_assets_redirects_unauthenticated_into_phoenix_scope(client):
    srv.app.state.phoenix = _FakeSupervisor()
    try:
        r = client.get("/assets/vendor-shiki.js", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/phoenix/assets/vendor-shiki.js"
        # nested chunk paths survive verbatim
        r = client.get("/assets/a/b-c.js", follow_redirects=False)
        assert r.headers["location"] == "/phoenix/assets/a/b-c.js"
    finally:
        srv.app.state.phoenix = None


def test_assets_404_when_phoenix_not_supervised(client):
    srv.app.state.phoenix = None
    assert client.get("/assets/x.js", follow_redirects=False).status_code == 404


def test_assets_post_stays_protected(client):
    srv.app.state.phoenix = _FakeSupervisor()
    try:
        # only GET/HEAD are exempt; anything else hits auth (401), not the route
        assert client.post("/assets/x.js").status_code == 401
    finally:
        srv.app.state.phoenix = None
