"""Runtime lifecycle gate for feature-contributed HTTP routes (#2522 P2).

Feature routers (Bridge) and webhook receivers are mounted ONCE at server
startup. Before this fix, disabling or removing a feature at runtime left its
routes serving and its startup-captured webhook receiver dispatching — a
disabled feature's HTTP surface stayed live.

These tests boot the REAL app via ``TestClient`` and drive requests through the
actual server/app path, asserting:

* a feature router (``/api/bridge/*``) returns 200 while its feature is enabled,
  404 the moment the feature is soft-disabled, and 200 again on re-enable — with
  the route staying physically mounted the whole time (no unmount/remount, no
  duplicate routes);
* removing the feature entirely also 404s its route;
* the shared ``/webhooks/{name}`` dispatch router only reaches an ENABLED
  feature's receiver — a disabled feature's webhook stops dispatching (404) and
  resumes on re-enable — proving the receiver set is live, not a stale startup
  snapshot.
"""

import os
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from anyio import ClosedResourceError
from fastapi import APIRouter, Depends, FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from kestrel_sovereign.features.bridge.feature import BridgeFeature
from kestrel_sovereign.features.webhooks.models import WebhookAuthType, WebhookConfig
from kestrel_sovereign.features.webhooks.receiver import WebhookReceiver

pytestmark = pytest.mark.usefixtures("isolated_process_rate_limiter")


API_KEY = "test-route-gate-key"


def _make_bridge_feature():
    """A real BridgeFeature with a stubbed status method (no DB needed)."""
    agent = MagicMock()
    agent.did = "did:test:bridge"
    bridge = BridgeFeature(agent=agent)
    bridge.enabled = True
    bridge.bridge_status = AsyncMock(
        return_value=SimpleNamespace(
            data={
                "uptime_seconds": 1,
                "active_sessions_memory": 0,
                "database_available": False,
            }
        )
    )
    return bridge


class _WebhookFeatureStub:
    """Minimal webhook feature: a real ``WebhookReceiver`` with one webhook and
    a real boolean ``enabled`` flag so the live receiver scan is observable."""

    name = "WebhookFeature"

    def __init__(self, webhook_name: str = "deposit", enabled: bool = True):
        self.enabled = enabled
        self.receiver = WebhookReceiver()
        self.receiver.register_webhook(
            WebhookConfig(name=webhook_name, auth_type=WebhookAuthType.NONE)
        )

    def get_router(self):  # pragma: no cover - never mounted per-feature
        return None


class _WebSocketFeatureStub:
    """Feature router with a real WebSocket endpoint for lifecycle gating."""

    name = "WebSocketFeature"

    def __init__(self):
        self.enabled = True

    def get_router(self):
        router = APIRouter()

        @router.websocket("/test-feature-lifecycle/ws")
        async def lifecycle_socket(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_text("enabled")
            try:
                # Keep the app task alive until the client has consumed the
                # buffered frame and closes its context. Returning here closes
                # AnyIO's receive stream before it drains under CPU pressure.
                await websocket.receive_text()
            except WebSocketDisconnect:
                pass

        return router


class _InstanceBoundRouterFeature:
    """Feature whose endpoint deliberately closes over its own instance."""

    def __init__(self, owner: str):
        self.enabled = True
        self.owner = owner
        self.served = 0

    def get_router(self):
        router = APIRouter()

        @router.get("/test-feature-lifecycle/instance-bound")
        async def instance_bound():
            self.served += 1
            return {"owner": self.owner}

        return router


class _DriftedRouterFeature:
    """Enabled, same feature name, but its router no longer exposes the
    mounted shape: it must not count as a survivor of that route."""

    def __init__(self):
        self.enabled = True
        self.served = 0

    def get_router(self):
        router = APIRouter()

        @router.get("/test-feature-lifecycle/somewhere-else")
        async def elsewhere():
            self.served += 1
            return {"owner": "drifted"}

        return router


class _PartiallyCopiedRouterFeature:
    """A router whose mounted child FastAPI cannot be copied by include_router."""

    def __init__(self):
        self.enabled = True

    def get_router(self):
        router = APIRouter()

        @router.get("/test-feature-lifecycle/partial-normal")
        async def normal_route():
            return {"partial": False}

        router.mount("/test-feature-lifecycle/partial-child", FastAPI())
        return router


async def _overrideable_feature_route_dependency():
    """A stable dependency key used to exercise app-level overrides."""


class _DependencyBoundRouterFeature:
    """Instance-bound router with a per-feature dependency for live dispatch."""

    def __init__(self, owner: str, events: list[str]):
        self.enabled = True
        self.owner = owner
        self.events = events
        self._router = None

    async def _selected_router_dependency(self):
        self.events.append(f"router:{self.owner}")

    def get_router(self):
        if self._router is not None:
            return self._router
        router = APIRouter(
            dependencies=[
                Depends(self._selected_router_dependency),
                Depends(_overrideable_feature_route_dependency),
            ]
        )

        @router.get("/test-feature-lifecycle/dependency-bound")
        async def dependency_bound():
            return {"owner": self.owner}

        self._router = router
        return router


def _boot(features):
    """Boot the real app with a mock single agent exposing ``features``.

    Returns the app plus a restore callable to unwind the patched state.
    """
    from server import app
    from kestrel_sovereign.server import (
        _mount_feature_routers,
        _unmount_feature_routers,
    )

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    agent = MagicMock()
    agent.features = features

    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    _mount_feature_routers(app)

    def restore():
        _unmount_feature_routers(app)
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    return app, agent, restore


def _boot_multi_agent(agents, *, app_dependencies=()):
    """Boot the real app in multi-agent mode with the given ``{name: agent}`` map.

    Installs a fake ``agent_manager`` (``list_agents`` / ``get_agent``) so the
    real ASGI routing middleware resolves ``/api/agents/{name}/...`` to the
    right agent and the shared webhook dispatch router is mounted across all of
    them. Returns the app plus a restore callable.
    """
    from server import app
    from kestrel_sovereign.server import (
        _mount_feature_routers,
        _unmount_feature_routers,
    )

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_dependencies = list(app.router.dependencies)
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    manager = MagicMock()
    # Match AgentManager's production contract: list_agents returns the live
    # name -> agent mapping, not merely a list of names.  Keep it dynamic so
    # removal/reload assertions observe the current test fleet.
    manager.list_agents = MagicMock(side_effect=lambda: dict(agents))
    manager.get_agent = MagicMock(side_effect=lambda name: agents.get(name))

    app.router.lifespan_context = noop_lifespan
    app.router.dependencies = [*original_dependencies, *app_dependencies]
    # Multi-agent mode: no single bound agent, only the manager.
    app.state.agent = None
    app.state.agent_manager = manager
    _mount_feature_routers(app)

    def restore():
        _unmount_feature_routers(app)
        app.router.lifespan_context = original_lifespan
        app.router.dependencies = original_dependencies
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    return app, restore


def _make_agent(features):
    """A mock agent exposing a real ``features`` dict for the live receiver scan."""
    agent = MagicMock()
    agent.features = features
    return agent


def _event_count(stub: "_WebhookFeatureStub") -> int:
    """How many webhook events this feature's receiver has recorded."""
    return len(stub.receiver.event_log)


def _bridge_health_route_count(app) -> int:
    return sum(
        1 for r in app.routes if getattr(r, "path", None) == "/api/bridge/health"
    )


def test_bridge_route_gated_by_live_enabled_state():
    """Enabled → 200, soft-disabled → 404, re-enabled → 200; route never churns."""
    os.environ["KESTREL_API_KEY"] = API_KEY
    bridge = _make_bridge_feature()
    app, _agent, restore = _boot({"BridgeFeature": bridge})
    try:
        # The route is mounted exactly once.
        assert _bridge_health_route_count(app) == 1

        with TestClient(app) as client:
            headers = {"X-API-Key": API_KEY}

            # Enabled → served.
            assert client.get("/api/bridge/health", headers=headers).status_code == 200

            # Soft-disable (endpoint soft-toggle keeps the instance loaded).
            bridge.enabled = False
            resp = client.get("/api/bridge/health", headers=headers)
            assert resp.status_code == 404
            # A disabled POST-only route must be absent from matching too: a
            # wrapper around route.app would let Starlette return 405 before
            # that wrapper ran.
            assert client.get("/api/bridge/invoke", headers=headers).status_code == 404
            # The route is STILL physically mounted — the gate rejected it, not
            # an unmount. Re-enable must not need a remount.
            assert _bridge_health_route_count(app) == 1

            # Re-enable → served again, no duplicate route added.
            bridge.enabled = True
            assert client.get("/api/bridge/health", headers=headers).status_code == 200
            assert _bridge_health_route_count(app) == 1
    finally:
        restore()


def test_bridge_route_404s_when_feature_removed():
    """A full remove (feature dropped from agent.features) 404s the route too."""
    os.environ["KESTREL_API_KEY"] = API_KEY
    bridge = _make_bridge_feature()
    app, agent, restore = _boot({"BridgeFeature": bridge})
    try:
        with TestClient(app) as client:
            headers = {"X-API-Key": API_KEY}
            assert client.get("/api/bridge/health", headers=headers).status_code == 200

            # Runtime remove drops the instance from agent.features entirely.
            agent.features = {}
            assert client.get("/api/bridge/health", headers=headers).status_code == 404
    finally:
        restore()


def test_repeated_feature_mounts_are_deduplicated_and_unmounted():
    """Cold registration does not duplicate routes before outer teardown.

    The host now invokes the mount pass for every scheduler-woken agent.  A
    repeated feature shape must therefore reuse the live-gated route instead
    of inserting a stale duplicate ahead of it; outer teardown still owns the
    one concrete route batch that was mounted.
    """
    from kestrel_sovereign.server import _mount_feature_routers

    bridge = _make_bridge_feature()
    app, _agent, restore = _boot({"BridgeFeature": bridge})
    try:
        assert _bridge_health_route_count(app) == 1
        _mount_feature_routers(app)
        assert _bridge_health_route_count(app) == 1
    finally:
        restore()

    assert _bridge_health_route_count(app) == 0


def test_invalid_feature_router_rolls_back_partially_included_routes():
    """A non-copyable child cannot leave an ungated route or retry duplicate."""

    from kestrel_sovereign import server

    feature = _PartiallyCopiedRouterFeature()
    app = FastAPI()
    app.state.agent = _make_agent({"PartialFeature": feature})
    app.state.agent_manager = None
    normal_path = "/test-feature-lifecycle/partial-normal"

    server._mount_feature_routers(app)
    assert not any(getattr(route, "path", None) == normal_path for route in app.routes)
    assert getattr(app.state, "_feature_routes", []) == []
    assert getattr(app.state, "_feature_router_keys", set()) == set()

    # A retry sees the same invalid shape but cannot accumulate an earlier
    # copied route; this guards both runtime reloads and repeated cold wakes.
    server._mount_feature_routers(app)
    assert not any(getattr(route, "path", None) == normal_path for route in app.routes)
    assert getattr(app.state, "_feature_routes", []) == []
    assert getattr(app.state, "_feature_router_keys", set()) == set()


def test_dynamic_feature_routes_invalidate_openapi_schema_on_mount_and_unmount():
    """A cold feature mount cannot leave a previously-served schema stale."""
    from kestrel_sovereign import server

    bridge = _make_bridge_feature()
    app = FastAPI()
    app.state.agent = _make_agent({"BridgeFeature": bridge})
    app.state.agent_manager = None
    path = "/api/bridge/health"

    # Simulate a client fetching OpenAPI before the scheduler cold-wakes this
    # agent. FastAPI caches that first result on the application instance.
    with TestClient(app) as client:
        initial_schema = client.get("/openapi.json").json()
        assert path not in initial_schema["paths"]
        assert app.openapi_schema is not None

        server._mount_feature_routers(app)
        assert app.openapi_schema is None
        mounted_schema = client.get("/openapi.json").json()
        assert path in mounted_schema["paths"]
        assert app.openapi_schema is not None

        server._unmount_feature_routers(app)
        assert app.openapi_schema is None
        assert path not in client.get("/openapi.json").json()["paths"]


def test_disabled_websocket_feature_route_is_not_matched():
    """A disabled WebSocket feature is excluded before an HTTP response emits."""
    os.environ["KESTREL_API_KEY"] = API_KEY
    feature = _WebSocketFeatureStub()
    app, _agent, restore = _boot({"WebSocketFeature": feature})
    try:
        headers = {"X-API-Key": API_KEY}
        with TestClient(app) as client:
            with client.websocket_connect(
                "/test-feature-lifecycle/ws", headers=headers
            ) as websocket:
                assert websocket.receive_text() == "enabled"

            feature.enabled = False
            with pytest.raises((WebSocketDisconnect, ClosedResourceError)):
                with client.websocket_connect(
                    "/test-feature-lifecycle/ws", headers=headers
                ):
                    pass
    finally:
        restore()


def test_instance_bound_feature_router_dispatches_to_request_agent_and_reload():
    """One shape-mounted route must never retain the first tenant's callable.

    ``include_router`` copies Alice's bound endpoint during startup.  Bob has
    the same route shape, so mounting his router would be deduplicated; the
    physically mounted route must instead resolve Bob's *current* feature on
    each agent-prefixed request.  This also proves an in-place feature reload
    replaces the callable without adding a duplicate route.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY
    alice = _InstanceBoundRouterFeature("alice-v1")
    bob = _InstanceBoundRouterFeature("bob-v1")
    agents = {
        "alice": _make_agent({"ProxyFeature": alice}),
        "bob": _make_agent({"ProxyFeature": bob}),
    }
    app, restore = _boot_multi_agent(agents)
    path = "/test-feature-lifecycle/instance-bound"
    try:
        assert sum(1 for route in app.routes if getattr(route, "path", None) == path) == 1
        headers = {"X-API-Key": API_KEY}
        with TestClient(app) as client:
            assert client.get(f"/api/agents/alice{path}", headers=headers).json() == {
                "owner": "alice-v1"
            }
            assert client.get(f"/api/agents/bob{path}", headers=headers).json() == {
                "owner": "bob-v1"
            }

            # A reload swaps the instance but deliberately keeps the route
            # shape identical.  It must use Bob-v2, not the first mount nor
            # the prior Bob instance.
            agents["bob"].features["ProxyFeature"] = _InstanceBoundRouterFeature(
                "bob-v2"
            )
            assert client.get(f"/api/agents/bob{path}", headers=headers).json() == {
                "owner": "bob-v2"
            }

            agents["bob"].features.pop("ProxyFeature")
            assert client.get(f"/api/agents/bob{path}", headers=headers).status_code == 404
            assert sum(
                1 for route in app.routes if getattr(route, "path", None) == path
            ) == 1
    finally:
        restore()


def test_request_scoped_feature_route_closes_when_target_is_unpublished():
    """A scoped route cannot outlive the manager publication it resolved from."""

    os.environ["KESTREL_API_KEY"] = API_KEY
    target = _make_agent({"ProxyFeature": _InstanceBoundRouterFeature("retired")})
    agents = {"target": target}
    app, restore = _boot_multi_agent(agents)
    manager = app.state.agent_manager

    def resolve_then_unpublish(name):
        candidate = agents.get(name)
        agents.pop(name, None)
        return candidate

    manager.get_agent.side_effect = resolve_then_unpublish
    path = "/test-feature-lifecycle/instance-bound"
    try:
        with TestClient(app) as client:
            response = client.get(
                f"/api/agents/target{path}",
                headers={"X-API-Key": API_KEY},
            )
        assert response.status_code == 404
    finally:
        restore()


def test_current_feature_route_keeps_app_overrides_and_live_dependencies():
    """Current feature dispatch must preserve FastAPI's app-bound execution.

    The mounted route belongs to Alice, but this request selects Bob. The
    router-level dependency must therefore be Bob's, while the app-level
    dependency and dependency override still run through the app-owned route
    copy rather than Bob's source ``current.app``.
    """

    os.environ["KESTREL_API_KEY"] = API_KEY
    events: list[str] = []

    async def host_dependency():
        events.append("host")

    async def dependency_override():
        events.append("override")

    alice = _DependencyBoundRouterFeature("alice", events)
    bob = _DependencyBoundRouterFeature("bob", events)
    agents = {
        "alice": _make_agent({"ProxyFeature": alice}),
        "bob": _make_agent({"ProxyFeature": bob}),
    }
    app, restore = _boot_multi_agent(
        agents,
        app_dependencies=(Depends(host_dependency),),
    )
    previous_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[_overrideable_feature_route_dependency] = (
        dependency_override
    )
    path = "/test-feature-lifecycle/dependency-bound"
    try:
        with TestClient(app) as client:
            response = client.get(
                f"/api/agents/bob{path}",
                headers={"X-API-Key": API_KEY},
            )

        assert response.status_code == 200
        assert response.json() == {"owner": "bob"}
        assert events == ["host", "router:bob", "override"]
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)
        restore()


def test_webhook_dispatch_is_live_and_enabled_filtered():
    """The shared /webhooks/{name} router dispatches only to ENABLED features."""
    os.environ["KESTREL_API_KEY"] = API_KEY
    hook_feature = _WebhookFeatureStub()
    app, _agent, restore = _boot({"WebhookFeature": hook_feature})
    try:
        with TestClient(app) as client:
            # Enabled → the receiver owns "deposit" and (NoAuth) accepts it.
            resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 200
            assert resp.json()["webhook"] == "deposit"

            # Disable the OWNING feature → the live scan drops its receiver, so
            # no receiver owns "deposit" any more → 404 (not still-dispatching).
            hook_feature.enabled = False
            resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 404

            # Re-enable → the receiver reappears in the live scan → 200 again.
            hook_feature.enabled = True
            resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 200
    finally:
        restore()


def test_first_dynamic_webhook_candidate_mounts_shared_dispatch_before_publish():
    """Private onboarding sees the candidate before manager publication."""
    from kestrel_sovereign import server

    hook_feature = _WebhookFeatureStub()
    candidate = _make_agent({"WebhookFeature": hook_feature})
    app = FastAPI()
    app.state.agent = None
    app.state.agent_manager = SimpleNamespace(
        list_agents=lambda: [],
        get_agent=lambda _name: None,
    )

    server._mount_feature_routers(app, agents=(candidate,))

    assert getattr(app.state, "_feature_webhook_dispatch_mounted", False)
    assert any(
        getattr(route, "path", None) == "/webhooks/{webhook_name}"
        for route in app.routes
    )
    # Simulate the later atomic publication commit. The already-mounted live
    # provider must now resolve and dispatch to this receiver.
    app.state.agent = candidate
    with TestClient(app) as client:
        response = client.post("/webhooks/deposit", content=b"{}")
    assert response.status_code == 200


def test_agent_prefixed_webhook_disabled_target_does_not_dispatch_to_peer():
    """#2522: POST /api/agents/A/webhooks/deposit must NOT reach agent B.

    Real ASGI collision: agent A's WebhookFeature is DISABLED and agent B's is
    ENABLED, both registering webhook ``deposit``. Addressing A's disabled
    receiver must 404 with NO dispatch — the aggregate scan previously picked
    B's enabled receiver (first match), so A returned 200 and B logged the
    event. With scope-aware resolution the request sees only A's (empty)
    receiver set → bare 404, and B records nothing.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY
    a_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=False)
    b_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
    agents = {
        "a": _make_agent({"WebhookFeature": a_hook}),
        "b": _make_agent({"WebhookFeature": b_hook}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        with TestClient(app) as client:
            resp = client.post("/api/agents/a/webhooks/deposit", content=b"{}")
            # A's owning feature is disabled → its receiver set is empty →
            # unknown webhook 404, never falling through to B.
            assert resp.status_code == 404, resp.text
            # The disabled target dispatched nothing, and B — a different
            # agent — must not have handled a request addressed to A.
            assert _event_count(b_hook) == 0, "peer agent B must not be dispatched"
            assert _event_count(a_hook) == 0, "disabled A must not be dispatched"
    finally:
        restore()


def test_agent_prefixed_webhook_dispatches_only_to_resolved_agent():
    """#2522: each agent-prefixed request reaches ONLY its own receiver.

    Both A and B are enabled and both own ``deposit``. Posting to A dispatches
    to A's receiver only; posting to B dispatches to B's receiver only — the
    request-scoped agent, not first-match-wins across the aggregate.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY
    a_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
    b_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
    agents = {
        "a": _make_agent({"WebhookFeature": a_hook}),
        "b": _make_agent({"WebhookFeature": b_hook}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        with TestClient(app) as client:
            # Address A → only A handles it.
            resp = client.post("/api/agents/a/webhooks/deposit", content=b"{}")
            assert resp.status_code == 200, resp.text
            assert resp.json()["webhook"] == "deposit"
            assert _event_count(a_hook) == 1
            assert _event_count(b_hook) == 0

            # Address B → only B handles it; A's tally is unchanged.
            resp = client.post("/api/agents/b/webhooks/deposit", content=b"{}")
            assert resp.status_code == 200, resp.text
            assert resp.json()["webhook"] == "deposit"
            assert _event_count(a_hook) == 1
            assert _event_count(b_hook) == 1
    finally:
        restore()


def test_unprefixed_webhook_form_retains_aggregate_lookup():
    """#2522 regression guard: the bare /webhooks/{name} form still aggregates.

    With no agent prefix there is no request-scoped agent, so the dispatch
    router falls back to the aggregate of every enabled receiver — preserving
    the pre-existing cross-agent behavior for keyless external callers that
    address the host, not a specific agent.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY
    # Only B owns "deposit"; the unprefixed form must still find it.
    a_hook = _WebhookFeatureStub(webhook_name="other", enabled=True)
    b_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
    agents = {
        "a": _make_agent({"WebhookFeature": a_hook}),
        "b": _make_agent({"WebhookFeature": b_hook}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        with TestClient(app) as client:
            resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 200, resp.text
            assert resp.json()["webhook"] == "deposit"
            assert _event_count(b_hook) == 1
            assert _event_count(a_hook) == 0
    finally:
        restore()


def _post_unprefixed_and_prefixed_ambiguous(agents, a_hook, b_hook):
    """Boot ``agents`` (an ordered ``{name: agent}`` map whose iteration order
    IS the fleet order) and exercise the ambiguous unprefixed form plus both
    explicit agent-prefixed forms. ``a_hook``/``b_hook`` are agent a's and
    agent b's webhook features, whichever order the fleet lists them in.

    Returns ``(unprefixed_response, unknown_response)`` where the unknown
    response is for a name nobody owns, taken AFTER every ownership assertion
    (the unknown-name 404 is audited on the first receiver, which would skew
    the per-receiver tallies asserted here).
    """
    app, restore = _boot_multi_agent(agents)
    try:
        with TestClient(app) as client:
            resp = client.post("/webhooks/deposit", content=b"{}")
            # Refused, not dispatched: neither owner handled it. Each owner
            # audits exactly one refusal (a 404, unauthenticated) and no
            # receive succeeded anywhere.
            assert resp.status_code == 404, resp.text
            for hook in (a_hook, b_hook):
                events = list(hook.receiver.event_log)
                assert [e.status_code for e in events] == [404], events
                assert not any(e.authenticated for e in events)
                # Refused, so no rate-limit window was opened on either owner.
                assert "deposit" not in hook.receiver._rate_windows

            # The explicit agent-prefixed form is unaffected by the
            # collision: each agent still receives ONLY what is addressed to
            # it, in the same boot.
            resp_a = client.post("/api/agents/a/webhooks/deposit", content=b"{}")
            assert resp_a.status_code == 200, resp_a.text
            assert resp_a.json()["webhook"] == "deposit"
            resp_b = client.post("/api/agents/b/webhooks/deposit", content=b"{}")
            assert resp_b.status_code == 200, resp_b.text
            assert resp_b.json()["webhook"] == "deposit"
            assert [e.status_code for e in a_hook.receiver.event_log] == [404, 200]
            assert [e.status_code for e in b_hook.receiver.event_log] == [404, 200]

            unknown = client.post("/webhooks/ghost", content=b"{}")
            return resp, unknown
    finally:
        restore()


def test_unprefixed_webhook_ambiguous_ownership_is_refused_in_either_fleet_order():
    """#3216: two enabled agents own ``deposit``; the unprefixed form must not
    let fleet iteration order choose the target.

    Before the fix the first receiver in ``list_agents()`` order won: fleet
    order (a, b) dispatched to A and order (b, a) dispatched to B, each a 200
    with the other agent none the wiser. Now BOTH orders refuse without any
    dispatch, with the same public 404 an unregistered name gets — so a
    keyless caller learns nothing about which names collide — while the
    explicit ``/api/agents/{name}/webhooks/deposit`` form keeps working.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY
    for order in (("a", "b"), ("b", "a")):
        a_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
        b_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
        by_name = {
            "a": _make_agent({"WebhookFeature": a_hook}),
            "b": _make_agent({"WebhookFeature": b_hook}),
        }
        agents = {name: by_name[name] for name in order}
        assert list(agents) == list(order)
        resp, unknown = _post_unprefixed_and_prefixed_ambiguous(
            agents, a_hook, b_hook
        )
        # Same safe public response as an unregistered name: status and body
        # shape are identical (only the echoed name differs).
        assert unknown.status_code == 404
        assert resp.status_code == unknown.status_code, order
        assert resp.json() == {"error": "Unknown webhook: deposit"}, order
        assert unknown.json() == {"error": "Unknown webhook: ghost"}, order


def test_unprefixed_webhook_ambiguity_tracks_live_enabled_owners():
    """#3216: only ENABLED owners count, and the count is read live.

    A enabled + B disabled is one owner → the unprefixed form dispatches to
    A. Enabling B makes the name ambiguous → refused, A's tally unchanged.
    Disabling A leaves B the sole owner → dispatches to B. A stale or
    enabled-blind count would fail one of the three legs.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY
    a_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
    b_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=False)
    agents = {
        "a": _make_agent({"WebhookFeature": a_hook}),
        "b": _make_agent({"WebhookFeature": b_hook}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        with TestClient(app) as client:
            resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 200, resp.text
            assert [e.status_code for e in a_hook.receiver.event_log] == [200]
            assert _event_count(b_hook) == 0

            b_hook.enabled = True
            resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 404, resp.text
            assert [e.status_code for e in a_hook.receiver.event_log] == [200, 404]
            assert [e.status_code for e in b_hook.receiver.event_log] == [404]

            a_hook.enabled = False
            resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 200, resp.text
            assert [e.status_code for e in a_hook.receiver.event_log] == [200, 404]
            assert [e.status_code for e in b_hook.receiver.event_log] == [404, 200]
    finally:
        restore()


def test_one_receiver_reachable_through_two_agents_is_one_owner():
    """#3216: dedupe-by-identity is now load-bearing for correctness.

    The aggregate provider dedupes receivers by identity because one
    receiver can be reached through more than one agent. Before the
    ambiguity rule that only saved a redundant scan; now a duplicate entry
    would count as two owners and falsely refuse a single-owner name. The
    same feature instance mounted on two fleet entries must still dispatch
    exactly once.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY
    shared_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
    agents = {
        "a": _make_agent({"WebhookFeature": shared_hook}),
        "b": _make_agent({"WebhookFeature": shared_hook}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        with TestClient(app) as client:
            resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 200, resp.text
            assert [e.status_code for e in shared_hook.receiver.event_log] == [200]
    finally:
        restore()

    # The per-agent scan dedupes too (round 7 coverage gap): one agent whose
    # two feature entries share one receiver object is one owner on the
    # agent-prefixed form, which is the only path where the per-agent scan's
    # result reaches dispatch without the aggregate's own dedupe.
    shared_hook.receiver.event_log.clear()
    agents = {
        "a": _make_agent({"FirstFeature": shared_hook, "SecondFeature": shared_hook}),
    }
    app, restore = _boot_multi_agent(agents)
    try:
        with TestClient(app) as client:
            resp = client.post("/api/agents/a/webhooks/deposit", content=b"{}")
            assert resp.status_code == 200, resp.text
            assert [e.status_code for e in shared_hook.receiver.event_log] == [200]
    finally:
        restore()


@contextmanager
def _receiver_log():
    """Collect the webhook receiver module's log records directly.

    The server's boot reconfigures root logging, so a root-level capture
    (``caplog``) misses records emitted during the first boot in a process;
    a handler on the module logger itself sees them regardless.
    """
    import logging

    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    target = logging.getLogger("kestrel_sovereign.features.webhooks.receiver")
    handler = _Collect(level=logging.WARNING)
    previous = target.level
    target.addHandler(handler)
    if previous == logging.NOTSET or previous > logging.WARNING:
        target.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(previous)


class _MinimalReceiverFeatureStub:
    """A receiver meeting only the host's duck-typed contract: ``webhooks`` +
    ``handle_webhook``. No ``record_refusal``, no ring buffer — the shape an
    out-of-tree feature package can contribute."""

    name = "ThirdPartyWebhookFeature"

    class _Receiver:
        def __init__(self, webhook_name):
            self.webhooks = {webhook_name: object()}
            self.handled = []

        async def handle_webhook(self, name, *, headers, body, source_ip):
            self.handled.append(name)
            return {"status_code": 200, "body": {"status": "received", "webhook": name}}

    def __init__(self, webhook_name="deposit"):
        self.enabled = True
        self.receiver = self._Receiver(webhook_name)

    def get_router(self):  # pragma: no cover - never mounted per-feature
        return None


def test_ambiguity_with_a_contract_minimal_receiver_is_still_the_same_404():
    """#3216 round 3: a receiver that meets only the required contract
    (no ``record_refusal``) must not turn the refusal into a 500 — that would
    be the ownership oracle the shared response denies. Same 404, no
    dispatch to either owner, the core owner still audits, and the host log
    says the other owner could not audit its own refusal.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY
    core_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
    minimal = _MinimalReceiverFeatureStub(webhook_name="deposit")
    for order in (("core", "third"), ("third", "core")):
        by_name = {
            "core": _make_agent({"WebhookFeature": core_hook}),
            "third": _make_agent({"ThirdPartyWebhookFeature": minimal}),
        }
        agents = {name: by_name[name] for name in order}
        core_hook.receiver.event_log.clear()
        minimal.receiver.handled.clear()
        app, restore = _boot_multi_agent(agents)
        try:
            with _receiver_log() as records:
                with TestClient(app) as client:
                    resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 404, (order, resp.text)
            assert resp.json() == {"error": "Unknown webhook: deposit"}, order
            assert minimal.receiver.handled == [], order
            assert [e.status_code for e in core_hook.receiver.event_log] == [404], order
            assert any(
                "cannot audit its own refusal" in r.getMessage() for r in records
            ), order
        finally:
            restore()


def test_two_distinct_receivers_that_compare_equal_are_two_owners():
    """#3216 round 4: the dedupe must be by identity, never ``==``.

    An out-of-tree receiver class with value equality (a dataclass, a
    pydantic model) is admitted on the same two-attribute contract. Two
    distinct instances that compare equal collapsed to one owner under an
    ``in``-based dedupe, so the unprefixed form dispatched again — to
    whichever agent the fleet listed first. Both orders must refuse.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY

    class _ValueEqualReceiver(_MinimalReceiverFeatureStub._Receiver):
        def __eq__(self, other):
            return isinstance(other, _ValueEqualReceiver)

        __hash__ = object.__hash__

    class _ValueEqualFeature(_MinimalReceiverFeatureStub):
        def __init__(self):
            self.enabled = True
            self.receiver = _ValueEqualReceiver("deposit")

    first, second = _ValueEqualFeature(), _ValueEqualFeature()
    assert first.receiver == second.receiver and first.receiver is not second.receiver
    fleets = [
        # Two agents, one value-equal receiver each — both fleet orders.
        {"a": _make_agent({"ThirdPartyWebhookFeature": first}),
         "b": _make_agent({"ThirdPartyWebhookFeature": second})},
        {"b": _make_agent({"ThirdPartyWebhookFeature": second}),
         "a": _make_agent({"ThirdPartyWebhookFeature": first})},
        # One agent, two features with value-equal receivers: the per-agent
        # scan dedupes too, and must also do so by identity.
        {"a": _make_agent({"FirstFeature": first, "SecondFeature": second})},
    ]
    for agents in fleets:
        first.receiver.handled.clear()
        second.receiver.handled.clear()
        app, restore = _boot_multi_agent(agents)
        try:
            with _receiver_log() as records:
                with TestClient(app) as client:
                    resp = client.post("/webhooks/deposit", content=b"{}")
                    assert resp.status_code == 404, (list(agents), resp.text)
                    if len(agents) == 1:
                        # A collision INSIDE one agent: the prefixed form
                        # refuses too, and the log must not send the
                        # operator to an address that also 404s.
                        scoped = client.post(
                            "/api/agents/a/webhooks/deposit", content=b"{}"
                        )
                        assert scoped.status_code == 404, scoped.text
                        assert scoped.json() == {"error": "Unknown webhook: deposit"}
            collisions = [
                r.getMessage() for r in records if "is owned by" in r.getMessage()
            ]
            assert collisions, [r.getMessage() for r in records]
            # An unscoped refusal cannot know whether the owners span agents,
            # so its remedy is conditional: try the prefixed form for the
            # intended agent, and if that refuses too the collision is inside
            # that agent. It must never present the prefixed form as a bare
            # "instead", which 404s in the one-agent fleet.
            unscoped = collisions[0]
            assert "/api/agents/{agent}/webhooks/deposit for the intended agent" in unscoped, unscoped
            assert "if that form refuses too" in unscoped, unscoped
            assert "instead." not in unscoped, unscoped
            if len(agents) == 1:
                # The scoped refusal KNOWS it is the within-agent case.
                assert len(collisions) == 2, collisions
                assert "within the addressed agent" in collisions[1], collisions
                assert "Address it as" not in collisions[1], collisions
            else:
                assert len(collisions) == 1, collisions
            assert first.receiver.handled == [] and second.receiver.handled == [], list(agents)
        finally:
            restore()


@pytest.mark.parametrize(
    "record_refusal,label",
    [
        (lambda self, name, **_: (_ for _ in ()).throw(RuntimeError("audit sink down")), "raises"),
        (lambda self, name, **_: None, "synchronous"),
    ],
)
def test_a_foreign_refusal_audit_that_misbehaves_never_changes_the_404(
    record_refusal, label
):
    """#3216 round 4: the refusal path awaits foreign ``record_refusal``
    implementations. One that raises, or returns nothing awaitable, must not
    turn the indistinguishable 404 into a 500 (the ownership oracle). The
    core owner still audits; the failure is host-logged.
    """
    os.environ["KESTREL_API_KEY"] = API_KEY
    core_hook = _WebhookFeatureStub(webhook_name="deposit", enabled=True)
    Foreign = type(
        "_ForeignReceiver",
        (_MinimalReceiverFeatureStub._Receiver,),
        {"record_refusal": record_refusal},
    )
    foreign = _MinimalReceiverFeatureStub("deposit")
    foreign.receiver = Foreign("deposit")
    for order in (("core", "third"), ("third", "core")):
        by_name = {
            "core": _make_agent({"WebhookFeature": core_hook}),
            "third": _make_agent({"ThirdPartyWebhookFeature": foreign}),
        }
        agents = {name: by_name[name] for name in order}
        core_hook.receiver.event_log.clear()
        app, restore = _boot_multi_agent(agents)
        try:
            with _receiver_log() as records:
                with TestClient(app) as client:
                    resp = client.post("/webhooks/deposit", content=b"{}")
            assert resp.status_code == 404, (label, order, resp.text)
            assert resp.json() == {"error": "Unknown webhook: deposit"}, (label, order)
            assert [e.status_code for e in core_hook.receiver.event_log] == [404], (label, order)
            assert foreign.receiver.handled == [], (label, order)
            failures = [r for r in records if "failed to audit" in r.getMessage()]
            if label == "raises":
                assert failures, order
            else:
                # A synchronous audit is a valid implementation, not a failure.
                assert not failures, (order, [r.getMessage() for r in failures])
        finally:
            restore()


# ---------------------------------------------------------------------------
# #3240: an unprefixed feature route whose mount owner was withdrawn
# ---------------------------------------------------------------------------


def _reorder(agents: dict, *names: str) -> None:
    """Re-key the live fleet mapping in place so ``list_agents()`` iterates in
    the given order; the manager stub reads the same dict on every call."""
    snapshot = {name: agents[name] for name in names}
    agents.clear()
    agents.update(snapshot)


def test_unprefixed_feature_route_with_two_survivors_is_refused_in_either_fleet_order():
    """Once the mount owner is withdrawn, two remaining agents that both serve
    the shape must not be chosen between by ``list_agents()`` order: the
    unprefixed form is refused in both orders and neither endpoint runs, while
    each agent-prefixed form still serves its own agent."""

    os.environ["KESTREL_API_KEY"] = API_KEY
    alice = _InstanceBoundRouterFeature("alice-v1")
    bob = _InstanceBoundRouterFeature("bob-v1")
    carol = _InstanceBoundRouterFeature("carol-v1")
    agents = {
        "alice": _make_agent({"ProxyFeature": alice}),
        "bob": _make_agent({"ProxyFeature": bob}),
        "carol": _make_agent({"ProxyFeature": carol}),
    }
    app, restore = _boot_multi_agent(agents)
    path = "/test-feature-lifecycle/instance-bound"
    headers = {"X-API-Key": API_KEY}
    try:
        with TestClient(app) as client:
            # The mount owner serves the unprefixed form while it is managed.
            assert client.get(path, headers=headers).json() == {"owner": "alice-v1"}
            assert alice.served == 1

            agents.pop("alice")
            for order in (("bob", "carol"), ("carol", "bob")):
                _reorder(agents, *order)
                response = client.get(path, headers=headers)
                assert response.status_code == 404, order
                assert bob.served == 0 and carol.served == 0, order

            assert client.get(f"/api/agents/bob{path}", headers=headers).json() == {
                "owner": "bob-v1"
            }
            assert client.get(f"/api/agents/carol{path}", headers=headers).json() == {
                "owner": "carol-v1"
            }
            assert bob.served == 1 and carol.served == 1
    finally:
        restore()


def test_unprefixed_feature_route_rebinds_to_the_only_survivor():
    """One remaining agent that serves the shape is not an ambiguous choice."""

    os.environ["KESTREL_API_KEY"] = API_KEY
    alice = _InstanceBoundRouterFeature("alice-v1")
    bob = _InstanceBoundRouterFeature("bob-v1")
    agents = {
        "alice": _make_agent({"ProxyFeature": alice}),
        "bob": _make_agent({"ProxyFeature": bob}),
    }
    app, restore = _boot_multi_agent(agents)
    path = "/test-feature-lifecycle/instance-bound"
    headers = {"X-API-Key": API_KEY}
    try:
        with TestClient(app) as client:
            assert client.get(path, headers=headers).json() == {"owner": "alice-v1"}
            agents.pop("alice")
            assert client.get(path, headers=headers).json() == {"owner": "bob-v1"}
            assert alice.served == 1 and bob.served == 1
    finally:
        restore()


def test_unprefixed_feature_route_survivor_count_tracks_live_enabled_serving():
    """A candidate that no longer serves the shape (feature disabled, or the
    feature removed) does not count: with it gone the other one is the only
    survivor and serves; with both live the unprefixed form is refused."""

    os.environ["KESTREL_API_KEY"] = API_KEY
    alice = _InstanceBoundRouterFeature("alice-v1")
    bob = _InstanceBoundRouterFeature("bob-v1")
    carol = _InstanceBoundRouterFeature("carol-v1")
    agents = {
        "alice": _make_agent({"ProxyFeature": alice}),
        "bob": _make_agent({"ProxyFeature": bob}),
        "carol": _make_agent({"ProxyFeature": carol}),
    }
    app, restore = _boot_multi_agent(agents)
    path = "/test-feature-lifecycle/instance-bound"
    headers = {"X-API-Key": API_KEY}
    try:
        with TestClient(app) as client:
            agents.pop("alice")
            assert client.get(path, headers=headers).status_code == 404

            bob.enabled = False
            assert client.get(path, headers=headers).json() == {"owner": "carol-v1"}
            assert bob.served == 0

            bob.enabled = True
            assert client.get(path, headers=headers).status_code == 404

            agents["bob"].features.pop("ProxyFeature")
            assert client.get(path, headers=headers).json() == {"owner": "carol-v1"}
            assert bob.served == 0 and carol.served == 2

            # Enabled and same-named, but the shape is gone from its router:
            # not a survivor of THIS route, so carol is still the only one.
            drifted = _DriftedRouterFeature()
            agents["bob"].features["ProxyFeature"] = drifted
            assert client.get(path, headers=headers).json() == {"owner": "carol-v1"}
            assert drifted.served == 0 and carol.served == 3
    finally:
        restore()


def test_unprefixed_feature_route_follows_the_mount_owner_across_its_reload():
    """Review r1 P2: the mount owner reloaded under a NEW object with the same
    routing name and DID is still the mount owner. Retention by object
    identity made that a permanent 404 while another agent served the shape;
    the DID is the stable identity, so the reloaded owner serves — even with
    two other survivors that would otherwise be an ambiguous pair."""

    os.environ["KESTREL_API_KEY"] = API_KEY
    alice_v1 = _InstanceBoundRouterFeature("alice-v1")
    bob = _InstanceBoundRouterFeature("bob-v1")
    carol = _InstanceBoundRouterFeature("carol-v1")
    agents = {
        "alice": _make_agent({"ProxyFeature": alice_v1}),
        "bob": _make_agent({"ProxyFeature": bob}),
        "carol": _make_agent({"ProxyFeature": carol}),
    }
    agents["alice"].did = "did:test:alice"
    agents["bob"].did = "did:test:bob"
    agents["carol"].did = "did:test:carol"
    app, restore = _boot_multi_agent(agents)
    path = "/test-feature-lifecycle/instance-bound"
    headers = {"X-API-Key": API_KEY}
    try:
        with TestClient(app) as client:
            assert client.get(path, headers=headers).json() == {"owner": "alice-v1"}

            # Withdrawn: two survivors, refused.
            agents.pop("alice")
            assert client.get(path, headers=headers).status_code == 404

            # Re-created under a new object, same name and DID: the owner is
            # back, and it wins over the pair that was ambiguous a moment ago.
            alice_v2 = _InstanceBoundRouterFeature("alice-v2")
            reloaded = _make_agent({"ProxyFeature": alice_v2})
            reloaded.did = "did:test:alice"
            agents["alice"] = reloaded
            assert client.get(path, headers=headers).json() == {"owner": "alice-v2"}
            assert alice_v1.served == 1 and alice_v2.served == 1
            assert bob.served == 0 and carol.served == 0
            assert client.get(f"/api/agents/alice{path}", headers=headers).json() == {
                "owner": "alice-v2"
            }

            # A different agent under the owner's old name is not the owner.
            impostor = _InstanceBoundRouterFeature("impostor")
            agents["alice"] = _make_agent({"ProxyFeature": impostor})
            agents["alice"].did = "did:test:someone-else"
            assert client.get(path, headers=headers).status_code == 404
            assert impostor.served == 0
    finally:
        restore()


class _FeaturesOnceThenGone(dict):
    """A ``features`` mapping whose lookup answers once and then reports the
    feature gone: the change between ``route.matches()`` and dispatch."""

    def __init__(self, name, feature, *, answers: int):
        super().__init__({name: feature})
        self._answers = answers

    def get(self, key, default=None):
        if self._answers <= 0:
            return default
        self._answers -= 1
        return super().get(key, default)


def test_dispatch_refuses_a_feature_that_vanished_after_the_match():
    """Review r1 M12: the match gate admits the request, then the feature is
    gone by dispatch. Dispatch must refuse, never fall back to the mounted
    copy's first-tenant endpoint (which is #3240 itself)."""

    os.environ["KESTREL_API_KEY"] = API_KEY
    alice = _InstanceBoundRouterFeature("alice-v1")
    agent = _make_agent({"ProxyFeature": alice})
    agents = {"alice": agent}
    app, restore = _boot_multi_agent(agents)
    path = "/test-feature-lifecycle/instance-bound"
    headers = {"X-API-Key": API_KEY}
    try:
        with TestClient(app) as client:
            assert client.get(path, headers=headers).json() == {"owner": "alice-v1"}
            assert alice.served == 1
            # One answer for the match gate; dispatch sees no feature.
            agent.features = _FeaturesOnceThenGone("ProxyFeature", alice, answers=1)
            response = client.get(path, headers=headers)
            assert response.status_code == 404
            assert alice.served == 1
    finally:
        restore()


class _FeaturesDisabledAfterFirstLookup(dict):
    """A ``features`` mapping that soft-disables the feature after the first
    lookup: the match gate sees it enabled, dispatch sees it disabled."""

    def __init__(self, name, feature):
        super().__init__({name: feature})
        self._looked_up = 0

    def get(self, key, default=None):
        feature = super().get(key, default)
        self._looked_up += 1
        if self._looked_up > 1 and feature is not None:
            feature.enabled = False
        return feature


def test_dispatch_refuses_a_feature_disabled_after_the_match():
    """Review r2: the match gate admits the request, then the feature is
    soft-disabled before dispatch. Dispatch re-checks ``enabled`` itself and
    refuses; it must not serve on the match gate's stale answer."""

    os.environ["KESTREL_API_KEY"] = API_KEY
    alice = _InstanceBoundRouterFeature("alice-v1")
    agent = _make_agent({"ProxyFeature": alice})
    app, restore = _boot_multi_agent({"alice": agent})
    path = "/test-feature-lifecycle/instance-bound"
    headers = {"X-API-Key": API_KEY}
    try:
        with TestClient(app) as client:
            assert client.get(path, headers=headers).json() == {"owner": "alice-v1"}
            assert alice.served == 1
            agent.features = _FeaturesDisabledAfterFirstLookup("ProxyFeature", alice)
            response = client.get(path, headers=headers)
            assert response.status_code == 404
            assert alice.served == 1
            assert alice.enabled is False
            # Exactly two lookups per request: the match gate, then dispatch.
            # The double disables after the FIRST, so this pins that the 404
            # came from dispatch's own check; a future extra lookup earlier
            # in the request would move the disable into the gate and make
            # this test vacuous — fail loudly instead.
            assert agent.features._looked_up == 2
    finally:
        restore()


def test_a_reloaded_owner_that_does_not_serve_outranks_the_surviving_peer():
    """Review r2: the owner clause precedes the survivor rule. A mount owner
    re-created under its DID with the feature disabled (readiness rollback,
    then retry) is still the owner: the unprefixed form refuses rather than
    quietly rebinding to the one peer that serves, which is exactly the
    order-chosen executor this ticket removes."""

    os.environ["KESTREL_API_KEY"] = API_KEY
    alice_v1 = _InstanceBoundRouterFeature("alice-v1")
    bob = _InstanceBoundRouterFeature("bob-v1")
    agents = {
        "alice": _make_agent({"ProxyFeature": alice_v1}),
        "bob": _make_agent({"ProxyFeature": bob}),
    }
    agents["alice"].did = "did:test:alice"
    agents["bob"].did = "did:test:bob"
    app, restore = _boot_multi_agent(agents)
    path = "/test-feature-lifecycle/instance-bound"
    headers = {"X-API-Key": API_KEY}
    try:
        with TestClient(app) as client:
            assert client.get(path, headers=headers).json() == {"owner": "alice-v1"}

            # Withdrawn: bob is the sole survivor and serves.
            agents.pop("alice")
            assert client.get(path, headers=headers).json() == {"owner": "bob-v1"}
            assert bob.served == 1

            # Back under the same DID, but not serving: the owner is retained
            # and refuses; the surviving peer is not chosen instead.
            alice_v2 = _InstanceBoundRouterFeature("alice-v2")
            alice_v2.enabled = False
            reloaded = _make_agent({"ProxyFeature": alice_v2})
            reloaded.did = "did:test:alice"
            agents["alice"] = reloaded
            assert client.get(path, headers=headers).status_code == 404
            assert bob.served == 1 and alice_v2.served == 0

            # Once it serves again, it serves.
            alice_v2.enabled = True
            assert client.get(path, headers=headers).json() == {"owner": "alice-v2"}
            assert bob.served == 1
    finally:
        restore()
