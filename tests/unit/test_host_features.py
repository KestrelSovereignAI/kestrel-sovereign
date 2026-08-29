"""Unit tests for the host-scoped feature runtime (issue #2293, Phase 1).

Covers discovery + mount at the host root, host lifecycle, the fleet
``HostContext`` + entities session factory, CSRF on state-changing host
endpoints, the host-scoped UI surface, and that per-agent behavior is untouched.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from kestrel_sdk.features import ContributionContractError
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
        return UIContributions(
            static_dir=None,
            modules=["/host/features/demo-host/panel.js"],
            css=["/host/features/demo-host/panel.css"],
        )


# ---------------------------------------------------------------------------
# Discovery + manifest
# ---------------------------------------------------------------------------


def test_instantiate_host_features_uses_provided_classes(tmp_path: Path):
    # An explicit path to a manifest that does not exist: the default is
    # otherwise read from CWD, which on a developer machine is the operator's
    # own manifest and would make this assertion depend on their config.
    features = hf.instantiate_host_features(
        {"_UIHostFeature": _UIHostFeature},
        manifest_path=tmp_path / ".kestrel-host-features.toml",
    )
    assert len(features) == 1
    assert isinstance(features[0], HostFeature)
    assert features[0].name == "demo-host"


def test_manifest_can_disable_host_feature(tmp_path: Path):
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text(
        '[[feature]]\nname = "demo-host"\nhost_scoped = true\nenabled = false\n'
    )
    scoped = hf.read_host_scoped_manifest(manifest)
    assert scoped.features == {"demo-host": False}

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
    assert hf.read_host_scoped_manifest(manifest).features == {}


# --- The default for a host feature the manifest never names (#3099) --------


def test_absent_manifest_enables_everything_discovered(tmp_path: Path):
    """The zero-config answer, unchanged: a fresh install runs what it has."""
    missing = tmp_path / ".kestrel-host-features.toml"

    assert hf.read_host_scoped_manifest(missing) == hf.HostScopedManifest()
    assert hf.read_host_scoped_manifest(missing).default_enabled is True
    assert len(
        hf.instantiate_host_features(
            {"_UIHostFeature": _UIHostFeature}, manifest_path=missing
        )
    ) == 1


def test_manifest_default_disables_a_feature_it_never_names(tmp_path: Path):
    """``default_enabled = false`` is a policy, so it also covers what is added
    to the entry-point group later — which is what naming every slug cannot do.
    """
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text(
        f"[{hf.HOST_SCOPE_TABLE}]\n{hf.DEFAULT_ENABLED_KEY} = false\n"
    )

    scoped = hf.read_host_scoped_manifest(manifest)
    assert scoped.default_enabled is False
    assert scoped.features == {}

    features = hf.instantiate_host_features(
        {"_UIHostFeature": _UIHostFeature}, manifest_path=manifest
    )
    assert features == []


def test_an_explicit_entry_beats_the_manifest_default(tmp_path: Path):
    """Otherwise the default could not be used to run a chosen subset."""
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text(
        f"[{hf.HOST_SCOPE_TABLE}]\n{hf.DEFAULT_ENABLED_KEY} = false\n\n"
        '[[feature]]\nname = "demo-host"\nhost_scoped = true\nenabled = true\n'
    )

    features = hf.instantiate_host_features(
        {"_UIHostFeature": _UIHostFeature}, manifest_path=manifest
    )
    assert [f.name for f in features] == ["demo-host"]


def test_a_class_name_entry_also_beats_the_manifest_default(tmp_path: Path):
    """The class name is the second spelling ``instantiate`` accepts."""
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text(
        f"[{hf.HOST_SCOPE_TABLE}]\n{hf.DEFAULT_ENABLED_KEY} = false\n\n"
        '[[feature]]\nname = "_UIHostFeature"\nscope = "host"\n'
    )

    features = hf.instantiate_host_features(
        {"_UIHostFeature": _UIHostFeature}, manifest_path=manifest
    )
    assert len(features) == 1


def test_a_non_boolean_default_is_reported_and_ignored(tmp_path: Path, caplog):
    """``bool("false")`` is ``True``, so a string is never coerced.

    Reading a typo as the opposite of what it says — in the permissive
    direction — is the failure this key exists to remove.
    """
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text(
        f'[{hf.HOST_SCOPE_TABLE}]\n{hf.DEFAULT_ENABLED_KEY} = "false"\n'
    )

    with caplog.at_level("WARNING"):
        scoped = hf.read_host_scoped_manifest(manifest)

    assert scoped.default_enabled is True
    assert hf.DEFAULT_ENABLED_KEY in caplog.text


def test_a_malformed_manifest_keeps_the_documented_default(tmp_path: Path):
    """Unparseable TOML must not break the host, nor invent a policy."""
    manifest = tmp_path / ".kestrel-host-features.toml"
    manifest.write_text("[host_features\nnot toml at all")

    assert hf.read_host_scoped_manifest(manifest) == hf.HostScopedManifest()


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


def test_agent_prefixed_host_path_still_requires_csrf():
    """The public per-agent alias must not bypass the host CSRF boundary."""
    from kestrel_sovereign import server

    app = FastAPI()
    hf.mount_host_feature_routers(app, [_UIHostFeature()])
    path = "/api/agents/Kite/api/demo-host/do"
    request = SimpleNamespace(
        method="POST",
        app=app,
        url=SimpleNamespace(path=path),
        scope={"path": path},
        cookies={},
        headers={},
    )

    response = server._enforce_host_csrf(request)

    assert response is not None
    assert response.status_code == 403
    assert b'"code":"csrf_failed"' in response.body
    assert response.headers["X-Correlation-ID"]


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


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_context", [False, True])
async def test_server_lifespan_wires_and_closes_host_features(
    monkeypatch, tmp_path: Path, reject_context: bool,
):
    """Exercise the deployed ``server:app`` call site, not runtime helpers.

    The host-feature block is the behavior moved by #2382.  This pins startup
    order, the platform ``PORT`` override exposed through ``HostContext``, and
    the shutdown mirror without booting a real agent or database.
    """
    from kestrel_sovereign import server
    from kestrel_sovereign.a2a import did_registry
    from kestrel_sovereign.multi_agent import agent_manager, config as ma_config
    from kestrel_sovereign.security import demo_isolation

    events: list[str] = []
    config_path = tmp_path / "multi_agent.toml"
    config_path.write_text("[host]\nport = 8888\n")
    fake_config = SimpleNamespace(
        host=SimpleNamespace(bind="127.0.0.1", port=8888),
        agents={"Kite": object()},
    )
    fake_agent = SimpleNamespace(is_test_instance=True)

    class FakeManager:
        init_failures = []

        def set_host_context_publication_gate(self, gate) -> None:
            self.host_context_publication_gate = gate

        def set_agent_registration_hook(self, _hook) -> None:
            return None

        async def load_from_config(self, config):
            assert config is fake_config
            assert not self.host_context_publication_gate.is_set()
            events.append("agents-load")
            return 1

        def list_agents(self):
            return ["Kite"]

        def get_agent(self, name):
            assert name == "Kite"
            return fake_agent

        async def shutdown_all(self):
            events.append("agents-stop")

    fake_manager = FakeManager()
    feature = _UIHostFeature()
    host_context_registry = object()
    host_runtime = SimpleNamespace(
        operator_registry=object(),
        wait_registry=object(),
        source_registry=object(),
        permission_defaults_registry=object(),
        setup_step_registry=object(),
        context_clause_registry=host_context_registry,
    )

    class Closeable:
        def __init__(self, event: str):
            self.event = event

        async def close(self):
            events.append(self.event)

    ctx = SovereignHostContext(
        db=Closeable("db-close"),
        config={},
        session_factory=Closeable("session-close"),
    )
    ctx.feature_contribution_runtime = host_runtime

    def validate_host_context(registry):
        assert registry is host_context_registry
        events.append("host-context-validate")
        if reject_context:
            raise ContributionContractError("host/agent context conflict")

    def bind_host_context(registry):
        assert registry is host_context_registry
        assert not fake_manager.host_context_publication_gate.is_set()
        validate_host_context(registry)
        events.append("host-context-bind")

    fake_manager.validate_host_context_clause_registry = validate_host_context
    fake_manager.bind_host_context_clause_registry = bind_host_context

    async def build_context(*, config):
        assert config["host_port"] == 9090
        events.append("context-build")
        return ctx

    async def start_features(features, supplied_ctx):
        assert features == [feature]
        assert supplied_ctx is ctx
        assert not fake_manager.host_context_publication_gate.is_set()
        events.append("host-start")

    async def stop_features(features, supplied_ctx):
        assert features == [feature]
        assert supplied_ctx is ctx
        events.append("host-stop")

    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setattr(server, "resolve_multi_agent_path", lambda env: config_path)
    monkeypatch.setattr(ma_config.MultiAgentConfig, "load", lambda *a, **k: fake_config)
    monkeypatch.setattr(agent_manager, "AgentManager", lambda **k: fake_manager)
    monkeypatch.setattr(did_registry, "install_a2a_did_resolver", lambda *a, **k: None)
    monkeypatch.setattr(demo_isolation, "classify_server_mode", lambda agents: True)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda app: None)
    monkeypatch.setattr(server, "_unmount_feature_ui_assets", lambda app: None)
    monkeypatch.setattr(server, "_unmount_feature_routers", lambda app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **k: [feature])
    monkeypatch.setattr(hf, "build_host_context", build_context)
    monkeypatch.setattr(
        hf,
        "mount_host_feature_routers",
        lambda app, features: events.append("host-router-mount"),
    )
    monkeypatch.setattr(
        hf,
        "mount_host_feature_ui",
        lambda app, features: events.append("host-ui-mount"),
    )
    monkeypatch.setattr(hf, "start_host_features", start_features)
    monkeypatch.setattr(hf, "stop_host_features", stop_features)
    monkeypatch.setattr(
        hf, "unmount_host_features", lambda app: events.append("host-unmount")
    )

    test_app = FastAPI()
    if reject_context:
        with pytest.raises(
            ContributionContractError,
            match="host/agent context conflict",
        ):
            async with server.lifespan(test_app):
                pass

        assert events == [
            "agents-load",
            "context-build",
            "host-start",
            "host-context-validate",
            "host-stop",
            "session-close",
            "db-close",
            "host-unmount",
            "agents-stop",
        ]
        assert not fake_manager.host_context_publication_gate.is_set()
        return

    async with server.lifespan(test_app):
        assert fake_config.host.port == 9090
        assert test_app.state.host_features == [feature]
        assert test_app.state.host_context is ctx
        assert fake_manager.host_context_publication_gate.is_set()
        assert events == [
            "agents-load",
            "context-build",
            "host-start",
            "host-context-validate",
            "host-router-mount",
            "host-ui-mount",
            "host-context-validate",
            "host-context-bind",
        ]

    assert events[-5:] == [
        "host-stop",
        "host-unmount",
        "session-close",
        "db-close",
        "agents-stop",
    ]


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
    assert manifest[0]["css"] == ["/host/features/demo-host/panel.css"]


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
