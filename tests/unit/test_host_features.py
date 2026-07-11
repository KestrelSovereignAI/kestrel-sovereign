"""Unit tests for the host-scoped feature runtime (issue #2293, Phase 1).

Covers discovery + mount at the host root, host lifecycle, the fleet
``HostContext`` + entities session factory, CSRF on state-changing host
endpoints, the host-scoped UI surface, and that per-agent behavior is untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from fastapi import APIRouter, FastAPI

from kestrel_sdk.features.host_base import HostFeature
from kestrel_sdk.features.ui import UIContributions

from kestrel_sovereign import host_features as hf
from kestrel_sovereign.host_features.context import (
    FLEET_TENANT_ID,
    FleetSessionFactory,
    SovereignHostContext,
    build_host_context,
)
from kestrel_sovereign.security import csrf


# ---------------------------------------------------------------------------
# A minimal host feature used across the tests.
# ---------------------------------------------------------------------------


class _DemoHostFeature(HostFeature):
    name = "demo-host"

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.ctx = None

    def get_router(self) -> APIRouter:
        router = APIRouter()

        @router.get("/api/demo-host/ping")
        async def ping():
            return {"ok": True, "scope": "host"}

        @router.post("/api/demo-host/do")
        async def do():
            return {"did": True}

        return router

    def get_ui_contributions(self, static_dir=None):  # signature-tolerant
        return UIContributions(static_dir=None, modules=["/host/features/demo-host/panel.js"])

    async def on_host_start(self, ctx) -> None:
        self.started = True
        self.ctx = ctx

    async def on_host_stop(self, ctx) -> None:
        self.stopped = True


# get_ui_contributions on the SDK ABC takes no args; override cleanly.
class _UIHostFeature(_DemoHostFeature):
    def get_ui_contributions(self):
        return UIContributions(static_dir=None, modules=["/host/features/demo-host/panel.js"])


# ---------------------------------------------------------------------------
# Discovery + manifest
# ---------------------------------------------------------------------------


def test_instantiate_host_features_uses_provided_classes():
    features = hf.instantiate_host_features({"_UIHostFeature": _UIHostFeature})
    assert len(features) == 1
    assert isinstance(features[0], HostFeature)
    assert features[0].name == "demo-host"


def test_manifest_can_disable_host_feature(tmp_path: Path):
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text(
        '[[feature]]\nname = "demo-host"\nhost_scoped = true\nenabled = false\n'
    )
    scoped = hf.read_host_scoped_manifest(manifest)
    assert scoped == {"demo-host": False}

    features = hf.instantiate_host_features(
        {"_UIHostFeature": _UIHostFeature}, manifest_path=manifest
    )
    assert features == []


def test_manifest_enables_host_scoped_entry(tmp_path: Path):
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text('[[feature]]\nname = "demo-host"\nscope = "host"\n')
    features = hf.instantiate_host_features(
        {"_UIHostFeature": _UIHostFeature}, manifest_path=manifest
    )
    assert len(features) == 1


def test_manifest_ignores_non_host_scoped_entries(tmp_path: Path):
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text('[[feature]]\nname = "voice"\n')  # not host-scoped
    assert hf.read_host_scoped_manifest(manifest) == {}


# ---------------------------------------------------------------------------
# Mount at host root — returns 200 with no agent context (AC #1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_host_feature_router_mounts_at_host_root_no_agent_context():
    app = FastAPI()
    feature = _UIHostFeature()
    hf.mount_host_feature_routers(app, [feature])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.get("/api/demo-host/ping")
    # 200, not 503 "Agent not initialized" — there is no get_agent dependency.
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "scope": "host"}


def test_mount_records_host_feature_prefix():
    app = FastAPI()
    hf.mount_host_feature_routers(app, [_UIHostFeature()])
    assert hf.is_host_feature_path(app, "/api/demo-host/do")
    # A per-agent proxy path is NOT a host-feature path.
    assert not hf.is_host_feature_path(app, "/api/agents/claw/api/conversations")


# ---------------------------------------------------------------------------
# Lifecycle (AC #3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_host_lifecycle_start_and_stop_called():
    feature = _UIHostFeature()
    ctx = SovereignHostContext(config={"host_port": 8888})
    await hf.start_host_features([feature], ctx)
    assert feature.started is True
    assert feature.ctx is ctx
    await hf.stop_host_features([feature], ctx)
    assert feature.stopped is True


@pytest.mark.asyncio
async def test_one_feature_start_failure_does_not_abort_others():
    class _Boom(HostFeature):
        name = "boom"

        async def on_host_start(self, ctx):
            raise RuntimeError("nope")

    ok = _UIHostFeature()
    # Boom first — must not prevent ok from starting.
    await hf.start_host_features([_Boom(), ok], SovereignHostContext())
    assert ok.started is True


# ---------------------------------------------------------------------------
# Host context: entities session factory on a host backend under fleet tenancy
# (AC #2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_host_context_provides_fleet_session_factory(tmp_path: Path):
    db_path = str(tmp_path / "kestrel_host.db")
    ctx = await build_host_context(config={"x": 1}, db_path=db_path)
    try:
        assert ctx.fleet_tenant_id == FLEET_TENANT_ID
        assert ctx.db is not None
        assert isinstance(ctx.session_factory, FleetSessionFactory)
        assert ctx.session_factory.tenant_id == FLEET_TENANT_ID
        # The factory yields a working session against the host backend.
        async with ctx.session_factory.read_session() as session:
            assert session is not None
    finally:
        if ctx.session_factory is not None:
            await ctx.session_factory.close()
        if ctx.db is not None and hasattr(ctx.db, "close"):
            await ctx.db.close()


def test_host_context_satisfies_sdk_protocol():
    from kestrel_sdk.features.host_base import HostContext

    ctx = SovereignHostContext(db=object(), backplane=object(), config={})
    assert isinstance(ctx, HostContext)


# ---------------------------------------------------------------------------
# Host-scoped UI surface (AC #5)
# ---------------------------------------------------------------------------


def test_host_ui_manifest_and_mount():
    app = FastAPI()
    hf.mount_host_feature_ui(app, [_UIHostFeature()])
    manifest = app.state.host_ui_manifest
    assert len(manifest) == 1
    assert manifest[0]["feature"] == "demo-host"
    assert manifest[0]["modules"] == ["/host/features/demo-host/panel.js"]


def test_host_ui_manifest_rejects_remote_modules():
    class _Evil(_DemoHostFeature):
        def get_ui_contributions(self):
            return UIContributions(modules=["https://evil.example/x.js"])

    from kestrel_sovereign.host_features.ui import compute_host_ui_manifest

    assert compute_host_ui_manifest([_Evil()]) == []


# ---------------------------------------------------------------------------
# CSRF double-submit helper (AC #4)
# ---------------------------------------------------------------------------


class _FakeReq:
    def __init__(self, method="POST", cookies=None, headers=None):
        self.method = method
        self.cookies = cookies or {}
        self.headers = headers or {}


def test_csrf_exempts_api_key_callers():
    # authed_via_cookie=False → machine caller, never checked.
    req = _FakeReq(method="POST")
    csrf.enforce_csrf(req, authed_via_cookie=False)  # no raise


def test_csrf_exempts_safe_methods():
    req = _FakeReq(method="GET")
    csrf.enforce_csrf(req, authed_via_cookie=True)  # no raise


def test_csrf_requires_token_for_cookie_state_change():
    req = _FakeReq(method="POST")  # no cookie/header
    with pytest.raises(csrf.CSRFError):
        csrf.enforce_csrf(req, authed_via_cookie=True)


def test_csrf_rejects_mismatched_token():
    req = _FakeReq(
        method="POST",
        cookies={csrf.CSRF_COOKIE_NAME: "a"},
        headers={csrf.CSRF_HEADER_NAME: "b"},
    )
    with pytest.raises(csrf.CSRFError):
        csrf.enforce_csrf(req, authed_via_cookie=True)


def test_csrf_accepts_matching_token():
    tok = "matching-token"
    req = _FakeReq(
        method="POST",
        cookies={csrf.CSRF_COOKIE_NAME: tok},
        headers={csrf.CSRF_HEADER_NAME: tok, "host": "h", "origin": "http://h"},
    )
    csrf.enforce_csrf(req, authed_via_cookie=True)  # no raise


# ---------------------------------------------------------------------------
# Host-route integration through server.app auth middleware (AC #1 + #4)
# The host-feature runtime + /api/host/* routes were consolidated onto the
# deployed server:app (issue #2382); the legacy proxy host was retired.
# ---------------------------------------------------------------------------


@pytest.fixture
def host_app_with_feature(monkeypatch):
    """server.app with a demo host feature mounted, bypassing the full lifespan."""
    from kestrel_sovereign import server as host_module

    monkeypatch.setenv("KESTREL_API_KEY", "test-host-key")
    monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "")

    app = host_module.app
    feature = _UIHostFeature()
    # Mount directly (no process manager / agents needed for these assertions).
    hf.unmount_host_features(app)
    hf.mount_host_feature_routers(app, [feature])
    hf.mount_host_feature_ui(app, [feature])
    yield app, host_module
    hf.unmount_host_features(app)


@pytest.mark.asyncio
async def test_host_route_200_with_api_key(host_app_with_feature):
    app, _ = host_app_with_feature
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.get(
            "/api/demo-host/ping", headers={"X-API-Key": "test-host-key"}
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_host_route_401_without_auth(host_app_with_feature):
    app, _ = host_app_with_feature
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.get("/api/demo-host/ping")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_host_post_api_key_exempt_from_csrf(host_app_with_feature):
    app, _ = host_app_with_feature
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.post(
            "/api/demo-host/do", headers={"X-API-Key": "test-host-key"}
        )
    # API-key caller: no CSRF token required.
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_host_ui_contributions_endpoint(host_app_with_feature):
    app, _ = host_app_with_feature
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.get(
            "/api/host/ui/contributions", headers={"X-API-Key": "test-host-key"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["contributions"][0]["feature"] == "demo-host"
