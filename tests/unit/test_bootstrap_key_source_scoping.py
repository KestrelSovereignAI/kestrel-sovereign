"""Who ``/api/auth/key`` hands the API key to (#3804).

The bootstrap gate admits a request when ``request.client.host`` is in
``security/bootstrap_access.py``'s allowlist. That host is not the socket
peer: uvicorn's default proxy-header handling replaces it with the
``X-Forwarded-For`` client when the peer is ``127.0.0.1``. These tests drive
the real server app wrapped exactly as uvicorn's defaults wrap it, so they pin
what the gate sees on a real launch, not what it would see if it read the
socket.

Two bypasses #3804 reproduced are refused since #3904 and #3905: a fleet host
whose startup failed before it published its registry, and a loopback proxy
whose forwarding headers uvicorn does not read.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from uvicorn.main import main as uvicorn_cli

from kestrel_sovereign import server
from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.security.bootstrap_access import forwarding_header_names

REMOTE = "203.0.113.7"  # TEST-NET-3: a client that is not on this host.
STANDALONE_KEY = "bootstrap-scoping-standalone-key"
FLEET_KEY = "bootstrap-scoping-fleet-key"

pytestmark = pytest.mark.usefixtures("isolated_process_rate_limiter")


def _issued(response) -> bool:
    return response.status_code == 200 and "key" in response.json()


def _assert_refused(response, status: int) -> None:
    assert response.status_code == status
    assert "key" not in response.json()


@pytest.fixture
def served_standalone(monkeypatch, tmp_path):
    """The real standalone server app as ``uvicorn`` serves it by default.

    Every Kestrel launcher leaves uvicorn's proxy-header settings at their
    defaults (``server_command`` and ``server.main`` argv are pinned in
    ``test_process_manager`` and ``test_server_cli``), so wrapping the app in
    ``Config(...).load()`` gives the gate the same ``client`` it sees live.
    """

    @asynccontextmanager
    async def no_lifespan(_app):
        yield

    monkeypatch.setattr(server.app.router, "lifespan_context", no_lifespan)
    for name in ("agent", "agent_manager", "multi_agent_config"):
        monkeypatch.setattr(server.app.state, name, None, raising=False)
    monkeypatch.setenv("KESTREL_API_KEY", STANDALONE_KEY)
    monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "false")
    # Standalone by the lifespan's own test too: no fleet registry exists.
    monkeypatch.setenv(
        "KESTREL_MULTI_AGENT_CONFIG", str(tmp_path / "no-multi-agent.toml")
    )
    for name in (
        "KESTREL_MULTI_AGENT",
        "KESTREL_A2A_TRANSPORT_ONLY",
        "KESTREL_BOOTSTRAP_ALLOWED_HOSTS",
        "KESTREL_KITE_RELEASE_EVIDENCE",
        "FORWARDED_ALLOW_IPS",
    ):
        monkeypatch.delenv(name, raising=False)
    config = uvicorn.Config(server.app, log_config=None)
    config.load()
    return config.loaded_app


def _request_key(served, client_host: str, headers=()):
    with TestClient(served, client=(client_host, 50000)) as client:
        return client.get("/api/auth/key", headers=list(headers))


def test_uvicorn_trusts_only_ipv4_loopback_to_name_the_client(monkeypatch):
    """The default the gate depends on: XFF is honored from 127.0.0.1 only."""
    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    defaults = {param.name: param.default for param in uvicorn_cli.params}

    assert defaults["proxy_headers"] is True
    assert defaults["forwarded_allow_ips"] is None
    config = uvicorn.Config("kestrel_sovereign.server:app")
    assert config.proxy_headers is True
    assert config.forwarded_allow_ips == "127.0.0.1"


def test_a_remote_peer_is_refused(served_standalone):
    _assert_refused(_request_key(served_standalone, REMOTE), 403)


@pytest.mark.parametrize("claimed", ["127.0.0.1", "localhost", "172.17.0.1"])
def test_a_remote_peer_cannot_claim_a_local_address_in_x_forwarded_for(
    served_standalone, claimed
):
    """uvicorn ignores XFF from an untrusted peer, so the claim is never read."""
    response = _request_key(
        served_standalone, REMOTE, [("X-Forwarded-For", claimed)]
    )

    _assert_refused(response, 403)


def test_q1a_a_loopback_proxy_that_sets_no_forwarding_header_is_admitted(
    served_standalone,
):
    """Q1a: indistinguishable from a browser on the host, so it gets the key.

    A same-host reverse proxy or tunnel that forwards a remote caller without
    any forwarding header reaches the app as ``127.0.0.1``. No source-address
    gate can tell the two apart; the risk is documented in
    ``docs/audit/AUTH_SURFACE_MATRIX.md``.
    """
    assert _issued(_request_key(served_standalone, "127.0.0.1"))


@pytest.mark.parametrize(
    "forwarded_for",
    [REMOTE, f"127.0.0.1, {REMOTE}", f"localhost, {REMOTE}"],
    ids=["proxy-sets-client", "client-spoof-then-proxy-append", "localhost-spoof"],
)
def test_q1b_a_loopback_proxy_that_names_a_remote_client_is_refused(
    served_standalone, forwarded_for
):
    """Q1b: one XFF header from 127.0.0.1 makes its last untrusted entry the client."""
    response = _request_key(
        served_standalone, "127.0.0.1", [("X-Forwarded-For", forwarded_for)]
    )

    _assert_refused(response, 403)


@pytest.mark.parametrize(
    ("peer", "headers"),
    [
        (
            "127.0.0.1",
            [("X-Forwarded-For", "127.0.0.1"), ("X-Forwarded-For", REMOTE)],
        ),
        ("127.0.0.1", [("Forwarded", f"for={REMOTE}")]),
        ("::1", [("X-Forwarded-For", REMOTE)]),
    ],
    ids=["two-xff-lines", "rfc7239-forwarded-only", "ipv6-loopback-proxy"],
)
def test_a_loopback_proxy_that_declares_a_remote_client_is_refused(
    served_standalone, peer, headers
):
    """A request carrying forwarding evidence did not originate on this host.

    Two XFF lines: HAProxy's ``option forwardfor`` appends its own line after
    the client's, and uvicorn reads XFF only when exactly one line is present.
    ``Forwarded`` (RFC 7239): uvicorn does not read it. ``::1``: uvicorn's
    default trusted forwarder is ``127.0.0.1`` only.
    """
    _assert_refused(_request_key(served_standalone, peer, headers), 403)


@pytest.mark.parametrize(
    ("peer", "headers"),
    [
        ("127.0.0.1", [("X-Real-IP", REMOTE)]),
        ("127.0.0.1", [("CF-Connecting-IP", REMOTE)]),
        ("127.0.0.1", [("True-Client-IP", REMOTE)]),
        ("127.0.0.1", [("Via", "1.1 edge-proxy")]),
        ("127.0.0.1", [("X-Forwarded-Proto", "https")]),
        ("127.0.0.1", [("X-Forwarded-Host", "kestrel.example")]),
        ("127.0.0.1", [("Forwarded", "for=127.0.0.1")]),
        ("172.17.0.1", [("X-Forwarded-For", REMOTE)]),
    ],
    ids=[
        "x-real-ip",
        "cloudflared-tunnel",
        "true-client-ip",
        "via-only",
        "x-forwarded-proto-only",
        "x-forwarded-host-only",
        "forwarded-claims-loopback",
        "docker-gateway-proxy",
    ],
)
def test_any_forwarding_header_refuses_an_otherwise_admitted_peer(
    served_standalone, peer, headers
):
    """The headers' presence refuses, not the client they name (#3905).

    uvicorn reads none of these from this peer, so the gate sees an allowed
    host; a proxy that says the client is local is refused all the same.
    """
    _assert_refused(_request_key(served_standalone, peer, headers), 403)


def test_ordinary_local_client_headers_are_not_forwarding_evidence():
    browser = [
        "host", "user-agent", "accept", "accept-language", "accept-encoding",
        "cookie", "referer", "origin", "sec-fetch-site", "x-api-key",
        "authorization", "x-requested-with", "x-correlation-id",
    ]

    assert forwarding_header_names(browser) == []
    assert forwarding_header_names(
        ["X-Forwarded-For", "x-forwarded-for", "VIA", "accept"]
    ) == ["via", "x-forwarded-for"]


def test_q1c_the_docker_bridge_gateway_is_admitted_by_default(served_standalone):
    """Q1c: ``172.17.0.1`` is in the default allowlist (#1724)."""
    assert _issued(_request_key(served_standalone, "172.17.0.1"))


@pytest.mark.parametrize("peer", ["172.17.0.2", "172.18.0.1", "10.0.0.5"])
def test_other_private_addresses_are_refused(served_standalone, peer):
    _assert_refused(_request_key(served_standalone, peer), 403)


def test_q2_bootstrap_stays_enabled_after_the_first_issue(served_standalone):
    """Q2: issuing the key does not close the endpoint; it returns the same key."""
    with TestClient(served_standalone, client=("127.0.0.1", 50000)) as client:
        first = client.get("/api/auth/key")
        second = client.get("/api/auth/key")

    assert _issued(first) and _issued(second)
    assert first.json()["key"] == second.json()["key"]


def test_q2_the_only_limit_on_reissue_is_the_per_host_rate_limit(
    served_standalone,
):
    with TestClient(served_standalone, client=("127.0.0.1", 50000)) as client:
        responses = [client.get("/api/auth/key") for _ in range(6)]

    assert all(_issued(response) for response in responses[:5])
    _assert_refused(responses[5], 429)


@pytest.mark.parametrize(
    "variable", ["KESTREL_REQUIRE_OAUTH", "KESTREL_A2A_TRANSPORT_ONLY"]
)
def test_q2_oauth_mode_and_managed_peers_turn_bootstrap_off(
    served_standalone, monkeypatch, variable
):
    monkeypatch.setenv(variable, "true")

    _assert_refused(_request_key(served_standalone, "127.0.0.1"), 404)


def _stub_host_resources(monkeypatch):
    """Replace what the lifespan starts around its fleet/standalone decision."""
    from kestrel_sovereign import host_features as hf
    from kestrel_sovereign import phoenix_supervisor as phoenix_module
    from kestrel_sovereign.security import demo_isolation

    async def _noop(*_args):
        return None

    monkeypatch.setenv("KESTREL_REQUIRE_OAUTH", "false")
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_PHOENIX_ENABLED", "0")
    for name in (
        "PORT",
        "KESTREL_SERVER_HOST",
        "KESTREL_DATABASE_URL",
        "KESTREL_A2A_TRANSPORT_ONLY",
        "KESTREL_BOOTSTRAP_ALLOWED_HOSTS",
        "KESTREL_KITE_RELEASE_EVIDENCE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server, "_initialize_stop_receipts", _noop)
    monkeypatch.setattr(server, "_prepare_shared_postgres_scheduler_protocol", _noop)
    monkeypatch.setattr(server, "_start_host_scheduler", _noop)
    monkeypatch.setattr(phoenix_module, "should_supervise_phoenix", lambda: False)
    monkeypatch.setattr(demo_isolation, "classify_server_mode", lambda _agents: False)
    monkeypatch.setattr(server, "_mount_feature_ui_assets", lambda _app: None)
    monkeypatch.setattr(server, "_mount_feature_routers", lambda _app: None)
    monkeypatch.setattr(server, "setup_tracing", lambda _app: None)
    monkeypatch.setattr(hf, "instantiate_host_features", lambda **_k: [])


def _bootstrap_app() -> FastAPI:
    app = FastAPI()
    app.state.limiter = limiter
    app.middleware("http")(server.auth_middleware)
    app.add_api_route("/api/auth/key", server.get_bootstrap_key, methods=["GET"])
    return app


@asynccontextmanager
async def _fleet_host(
    monkeypatch,
    tmp_path,
    registry_text: str,
    *,
    multi_agent_env: bool = True,
    host_context=None,
):
    """Run the real fleet-host startup on a ``multi_agent.toml`` of ``registry_text``.

    The registry is loaded by the real ``MultiAgentConfig.load``; only the
    resources started after it (Hold custody, agents, scheduler, features)
    are replaced, as in ``test_a2a_native_cohost_startup``. Without
    ``multi_agent_env`` the host is a fleet only because its registry exists.
    ``host_context`` replaces the Hold custody step.
    """
    from kestrel_sovereign.multi_agent import agent_manager

    registry = tmp_path / "multi_agent.toml"
    registry.write_text(registry_text, encoding="utf-8")

    class _Manager:
        init_failures = ()

        def set_agent_registration_hook(self, _hook):
            return None

        def set_created_agent_persistence_hook(self, _hook):
            return None

        def set_created_agent_registration_removal_hook(self, _hook):
            return None

        def reconcile_spawn_authority_restart_roster(self, supplied):
            return supplied

        def bind_hold_store(self, _store):
            return None

        async def load_from_config(
            self, _supplied, *, restart_roster_reconciled, before_initialize
        ):
            return []

        def list_agents(self):
            return {}

        async def shutdown_all(self):
            return None

    async def _host_context(app, _config):
        app.state.host_context = SimpleNamespace(
            hold_store=object(), feature_contribution_runtime=None
        )
        return app.state.host_context

    _stub_host_resources(monkeypatch)
    if multi_agent_env:
        monkeypatch.setenv("KESTREL_MULTI_AGENT", "1")
    else:
        monkeypatch.delenv("KESTREL_MULTI_AGENT", raising=False)
    monkeypatch.setenv("KESTREL_MULTI_AGENT_CONFIG", str(registry))
    monkeypatch.setenv("KESTREL_API_KEY", FLEET_KEY)
    monkeypatch.setattr(agent_manager, "AgentManager", lambda **_k: _Manager())
    monkeypatch.setattr(
        server, "_build_host_control_context", host_context or _host_context
    )

    app = _bootstrap_app()
    async with server._lifespan_startup(app):
        yield app


async def _loopback_get_key(app):
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50000))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/api/auth/key")


async def test_a_fleet_host_that_published_its_registry_never_bootstraps(
    monkeypatch, tmp_path
):
    async with _fleet_host(monkeypatch, tmp_path, '[host]\nbind = "127.0.0.1"\n') as app:
        assert app.state.multi_agent_config is not None
        response = await _loopback_get_key(app)

    _assert_refused(response, 404)


async def test_a_fleet_host_whose_registry_failed_to_load_never_bootstraps(
    monkeypatch, tmp_path
):
    async with _fleet_host(monkeypatch, tmp_path, "[host\nport = = 1\n") as app:
        if not app.state.startup_error:
            pytest.fail("the malformed registry must fail fleet startup")
        if getattr(app.state, "multi_agent_config", None) is not None:
            pytest.fail("startup must fail before the registry is published")
        response = await _loopback_get_key(app)

    _assert_refused(response, 404)


async def _refuse_hold_custody(_app, _config):
    raise RuntimeError(
        "Host Hold control state is unavailable before work admission: refused"
    )


@pytest.mark.parametrize(
    ("registry_text", "multi_agent_env", "host_context"),
    [
        ("[host\nport = = 1\n", False, None),
        ('[host]\nbind = "127.0.0.1"\n', True, _refuse_hold_custody),
    ],
    ids=["found-by-registry-file", "hold-custody-refused"],
)
async def test_every_failed_fleet_startup_keeps_the_fleet_verdict(
    monkeypatch, tmp_path, registry_text, multi_agent_env, host_context
):
    """The verdict is the lifespan's, however the host is a fleet and wherever
    its startup fails before publishing the registry (#3904)."""
    async with _fleet_host(
        monkeypatch,
        tmp_path,
        registry_text,
        multi_agent_env=multi_agent_env,
        host_context=host_context,
    ) as app:
        assert app.state.startup_error
        assert getattr(app.state, "multi_agent_config", None) is None
        assert getattr(app.state, "agent_manager", None) is None
        response = await _loopback_get_key(app)

    _assert_refused(response, 404)


async def test_a_standalone_host_whose_startup_failed_still_bootstraps(
    monkeypatch, tmp_path
):
    """The recorded verdict is the lifespan's decision, not "startup failed"."""

    async def _no_identity(*_args, **_kwargs):
        raise RuntimeError("agent identity is unavailable")

    _stub_host_resources(monkeypatch)
    monkeypatch.delenv("KESTREL_MULTI_AGENT", raising=False)
    monkeypatch.setenv(
        "KESTREL_MULTI_AGENT_CONFIG", str(tmp_path / "no-multi-agent.toml")
    )
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path))
    monkeypatch.setenv("KESTREL_API_KEY", STANDALONE_KEY)
    monkeypatch.setattr(server, "get_agent_did_async", _no_identity)

    app = _bootstrap_app()
    async with server._lifespan_startup(app):
        assert app.state.startup_error
        response = await _loopback_get_key(app)

    assert _issued(response)
