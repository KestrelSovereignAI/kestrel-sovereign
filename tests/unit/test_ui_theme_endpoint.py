"""Contract tests for the /api/ui/theme endpoint (epic #986, sub-issue #989)."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.ui import theme_loader


@pytest.fixture
def client():
    """Build a TestClient against the real app with lifespan disabled.

    We don't need agent state for the theme endpoint — it's a pure
    file-reader. Disabling the lifespan avoids booting the full agent.
    """
    from server import app

    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    theme_loader.clear_cache()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.router.lifespan_context = original
        theme_loader.clear_cache()


def test_default_returns_legacy_en(client):
    r = client.get("/api/ui/theme")
    assert r.status_code == 200
    body = r.json()
    assert body["theme"] == "legacy"
    assert body["locale"] == "en"
    assert body["labels"]["tab_identity"] == "Identity"
    assert body["labels"]["sidebar_agents"] == "Agents"
    assert body["fallback_keys"] == []


def test_falconry_theme(client):
    r = client.get("/api/ui/theme", params={"theme": "falconry"})
    assert r.status_code == 200
    body = r.json()
    assert body["theme"] == "falconry"
    assert body["labels"]["sidebar_agents"] == "Mews"
    assert body["labels"]["spawn_title"] == "Hatchery"


def test_plain_theme(client):
    r = client.get("/api/ui/theme", params={"theme": "plain"})
    assert r.status_code == 200
    body = r.json()
    assert body["theme"] == "plain"
    assert body["labels"]["spawn_title"] == "Multi-Agent"
    assert body["labels"]["memories_title"] == "Memories"


def test_explicit_theme_and_locale(client):
    r = client.get("/api/ui/theme", params={"theme": "falconry", "locale": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["theme"] == "falconry"
    assert body["locale"] == "en"


def test_unknown_theme_returns_404(client):
    r = client.get("/api/ui/theme", params={"theme": "doesnotexist"})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "theme_not_found"
    assert "available_themes" in detail
    assert "legacy" in detail["available_themes"]


def test_response_shape_is_consistent(client):
    r = client.get("/api/ui/theme")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"theme", "locale", "labels", "fallback_keys"}
    assert isinstance(body["labels"], dict)
    assert isinstance(body["fallback_keys"], list)


def test_themes_endpoint_lists_available(client):
    r = client.get("/api/ui/themes")
    assert r.status_code == 200
    body = r.json()
    assert "themes" in body
    assert set(body["themes"]) >= {"legacy", "falconry", "plain"}


def test_locale_fallback_to_en_when_locale_missing(client):
    r = client.get("/api/ui/theme", params={"theme": "legacy", "locale": "es"})
    assert r.status_code == 200
    body = r.json()
    # Locale falls back to en, response reports actual locale used
    assert body["locale"] == "en"


def test_invalid_param_lengths_rejected(client):
    # Empty theme name fails Query min_length validation
    r = client.get("/api/ui/theme", params={"theme": ""})
    assert r.status_code == 422
