"""Contract tests for the Rasa webhook shim."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def _prepare_app(agent):
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
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


def _api_headers():
    # The Rasa webhook now self-authenticates with a dedicated token (#1729),
    # since /webhooks/* is exempt from the host API-key middleware.
    return {"X-API-Key": "test-key", "X-Webhook-Token": "rasa-token"}


def test_rasa_webhook_does_not_force_hardcoded_model_override():
    agent = MagicMock()
    agent.process_input = AsyncMock(return_value="Take your blood pressure again in 10 minutes.")

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key", "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token"}):
            with TestClient(app) as client:
                response = client.post(
                    "/webhooks/rest/webhook",
                    headers={**_api_headers(), "X-Request-ID": "rasa-retry-2765"},
                    json={"sender": "patient-123", "message": "BP was 140/90"},
                )

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "rasa-retry-2765"
        assert response.json() == [
            {
                "recipient_id": "patient-123",
                "text": "Take your blood pressure again in 10 minutes.",
            }
        ]

        agent.process_input.assert_awaited_once()
        _, kwargs = agent.process_input.await_args
        assert kwargs["session_id"] == "sms:patient-123"
        assert kwargs["include_memories"] is False
        assert kwargs["invocation_id"] == "rasa-retry-2765"
        assert (
            kwargs["invocation_provenance"].source_locator
            == "POST:/webhooks/rest/webhook"
        )
        assert kwargs["invocation_provenance"].actor == "rasa_webhook"
        assert "Patient says: BP was 140/90" in kwargs["user_input"]
        assert "model_override" not in kwargs
    finally:
        _restore_app(app, original)


def test_rasa_webhook_reports_cooperative_stop_as_conflict():
    from kestrel_sovereign.agent.invocation import InvocationCancelledError

    agent = MagicMock()
    agent.process_input = AsyncMock(
        side_effect=InvocationCancelledError("isolated turn stopped")
    )
    app, original = _prepare_app(agent)
    try:
        with patch.dict(
            "os.environ",
            {
                "KESTREL_API_KEY": "test-key",
                "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token",
            },
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/webhooks/rest/webhook",
                    headers={
                        **_api_headers(),
                        "X-Request-ID": "rasa-stopped-turn",
                    },
                    json={"sender": "patient-123", "message": "stop this"},
                )

        assert response.status_code == 409
        assert response.json()["detail"] == "Request stopped during execution."
        assert response.headers["X-Request-ID"] == "rasa-stopped-turn"
    finally:
        _restore_app(app, original)


def _prepare_multi_agent_app(agents, default=None):
    """Boot the real app in multi-agent mode with ``{name: agent}``.

    ``default`` is the host-default agent (``app.state.agent``), ``None`` for
    a host with no default. The real routing middleware resolves
    ``/api/agents/{name}/...`` through the fake manager, exactly as the
    deployed host does.
    """
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    by_did = {getattr(agent, "did", None): name for name, agent in agents.items()}
    manager = MagicMock()
    manager.list_agents = MagicMock(side_effect=lambda: dict(agents))
    manager.get_agent = MagicMock(side_effect=lambda name: agents.get(name))
    manager.get_agent_name = MagicMock(side_effect=lambda did: by_did.get(did))
    app.router.lifespan_context = noop_lifespan
    app.state.agent = default
    app.state.agent_manager = manager
    return app, original


def _rasa_agent(reply, did=None):
    agent = MagicMock()
    agent.did = did or f"did:test:{reply.replace(' ', '-')}"
    agent.process_input = AsyncMock(return_value=reply)
    return agent


# Multi-agent tests enable the Rasa channel for the agents they address
# (#3220 round 1 F1): the prefixed alias is a per-agent opt-in.
_MULTI_ENV = {
    "KESTREL_API_KEY": "test-key",
    "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token",
    "KESTREL_RASA_WEBHOOK_AGENTS": "a, b",
}


def test_prefixed_rasa_alias_invokes_only_the_routed_agent():
    """#3220: ``/api/agents/B/webhooks/rest/webhook`` runs on B, never on A.

    The handler read ``app.state.agent`` (the host default) and ignored the
    agent the routing middleware pinned, so a message explicitly addressed
    to B executed on A. Two agents, no default: each prefixed request must
    reach only its own agent, with the routed agent's reply and a session
    keyed by the sender.
    """
    a = _rasa_agent("reply from a")
    b = _rasa_agent("reply from b")
    app, original = _prepare_multi_agent_app({"a": a, "b": b})
    try:
        with patch.dict("os.environ", _MULTI_ENV):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agents/b/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-1", "message": "hello b"},
                )
                assert response.status_code == 200, response.text
                assert response.json() == [
                    {"recipient_id": "patient-1", "text": "reply from b"}
                ]
                b.process_input.assert_awaited_once()
                a.process_input.assert_not_awaited()
                _, kwargs = b.process_input.await_args
                assert kwargs["session_id"] == "sms:patient-1"
                assert "hello b" in kwargs["user_input"]

                response = client.post(
                    "/api/agents/a/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-2", "message": "hello a"},
                )
                assert response.status_code == 200, response.text
                assert response.json() == [
                    {"recipient_id": "patient-2", "text": "reply from a"}
                ]
                a.process_input.assert_awaited_once()
                b.process_input.assert_awaited_once()
    finally:
        _restore_app(app, original)


def test_prefixed_rasa_alias_beats_the_host_default_agent():
    """Defence-in-depth precedence check over a SYNTHETIC state.

    A real multi-agent boot sets ``app.state.agent = None``, so on a real
    host the pre-fix symptom was a 503 on every prefixed request (see the
    no-default test), never cross-agent execution. Should a default and a
    manager ever coexist, the routed agent must still win and the
    unprefixed form must still reach the default.
    """
    a = _rasa_agent("reply from a")
    b = _rasa_agent("reply from b")
    app, original = _prepare_multi_agent_app({"a": a, "b": b}, default=a)
    try:
        with patch.dict("os.environ", _MULTI_ENV):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agents/b/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-5", "message": "for b"},
                )
                assert response.status_code == 200, response.text
                assert response.json()[0]["text"] == "reply from b"
                b.process_input.assert_awaited_once()
                a.process_input.assert_not_awaited()

                response = client.post(
                    "/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-6", "message": "for the default"},
                )
                assert response.status_code == 200, response.text
                assert response.json()[0]["text"] == "reply from a"
                a.process_input.assert_awaited_once()
                b.process_input.assert_awaited_once()
    finally:
        _restore_app(app, original)


def test_unprefixed_rasa_webhook_on_a_multi_agent_host_without_a_default_refuses():
    """#3220 regression guard: with no host-default agent the unprefixed form
    has no target. It must 503 without invoking anyone — never pick an agent
    from the fleet.
    """
    a = _rasa_agent("reply from a")
    b = _rasa_agent("reply from b")
    app, original = _prepare_multi_agent_app({"a": a, "b": b})
    try:
        with patch.dict(
            "os.environ",
            {"KESTREL_API_KEY": "test-key", "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token"},
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "patient-3", "message": "hello?"},
                )
        assert response.status_code == 503, response.text
        a.process_input.assert_not_awaited()
        b.process_input.assert_not_awaited()
    finally:
        _restore_app(app, original)


def test_prefixed_rasa_alias_still_requires_the_webhook_token():
    """Routing to a named agent does not relax the shim's own auth (#1729):
    the token check runs before any agent is resolved, so an unauthenticated
    prefixed request invokes nobody.
    """
    b = _rasa_agent("reply from b")
    app, original = _prepare_multi_agent_app({"b": b})
    try:
        with patch.dict("os.environ", _MULTI_ENV):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agents/b/webhooks/rest/webhook",
                    headers={"X-API-Key": "test-key"},
                    json={"sender": "patient-4", "message": "no token"},
                )
        assert response.status_code == 401, response.text
        b.process_input.assert_not_awaited()
    finally:
        _restore_app(app, original)


def test_prefixed_rasa_alias_is_a_per_agent_opt_in_that_fails_closed():
    """#3220 round 1 (P2): one host-wide token must not become paid turns on
    every agent by path segment. The prefixed alias is refused — 404, the
    routing middleware's own unknown-agent answer, nobody invoked — unless
    the routed agent is listed in ``KESTREL_RASA_WEBHOOK_AGENTS``. Unset
    refuses everything prefixed; the unprefixed default is untouched by the
    list either way.
    """
    a = _rasa_agent("reply from a")
    b = _rasa_agent("reply from b")
    # "" is the unset case: no agent is enabled for the prefixed alias.
    for env_agents in ("", "a"):
        env = {
            "KESTREL_API_KEY": "test-key",
            "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token",
            "KESTREL_RASA_WEBHOOK_AGENTS": env_agents,
        }
        a.process_input.reset_mock()
        b.process_input.reset_mock()
        app, original = _prepare_multi_agent_app({"a": a, "b": b}, default=a)
        try:
            with patch.dict("os.environ", env):
                with TestClient(app) as client:
                    refused = client.post(
                        "/api/agents/b/webhooks/rest/webhook",
                        headers=_api_headers(),
                        json={"sender": "patient-7", "message": "for b"},
                    )
                    assert refused.status_code == 404, (env_agents, refused.text)
                    b.process_input.assert_not_awaited()
                    a.process_input.assert_not_awaited()

                    default = client.post(
                        "/webhooks/rest/webhook",
                        headers=_api_headers(),
                        json={"sender": "patient-8", "message": "for default"},
                    )
                    assert default.status_code == 200, (env_agents, default.text)
                    a.process_input.assert_awaited_once()

                    if env_agents == "a":
                        listed = client.post(
                            "/api/agents/a/webhooks/rest/webhook",
                            headers=_api_headers(),
                            json={"sender": "patient-9", "message": "for a"},
                        )
                        assert listed.status_code == 200, listed.text
                        assert a.process_input.await_count == 2
                        b.process_input.assert_not_awaited()
        finally:
            _restore_app(app, original)


def test_rasa_rate_limit_bucket_is_per_routed_agent():
    """#3220 round 1 (P2): the middleware strips the prefix before SlowAPI
    keys the request, so the fleet shared one 30/minute bucket per source.
    Exhausting agent RA's bucket must not touch agent RB's.
    """
    ra = _rasa_agent("reply from ra")
    rb = _rasa_agent("reply from rb")
    env = {
        "KESTREL_API_KEY": "test-key",
        "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token",
        "KESTREL_RASA_WEBHOOK_AGENTS": "ra,rb",
    }
    app, original = _prepare_multi_agent_app({"ra": ra, "rb": rb})
    try:
        with patch.dict("os.environ", env):
            with TestClient(app) as client:
                for i in range(30):
                    resp = client.post(
                        "/api/agents/ra/webhooks/rest/webhook",
                        headers=_api_headers(),
                        json={"sender": f"p-{i}", "message": "hi"},
                    )
                    assert resp.status_code == 200, (i, resp.text)
                exhausted = client.post(
                    "/api/agents/ra/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "p-31", "message": "hi"},
                )
                assert exhausted.status_code == 429, exhausted.text
                other = client.post(
                    "/api/agents/rb/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "p-b", "message": "hi"},
                )
                assert other.status_code == 200, other.text
                rb.process_input.assert_awaited_once()
    finally:
        _restore_app(app, original)


def test_token_check_precedes_agent_resolution():
    """The shim's own auth runs first: with NO resolvable agent and a bad
    token the answer is 401, not 503 — the discriminating input the
    ordering claim needs (round 1 coverage note).
    """
    app, original = _prepare_multi_agent_app({})
    try:
        with patch.dict(
            "os.environ",
            {"KESTREL_API_KEY": "test-key", "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token"},
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/webhooks/rest/webhook",
                    headers={"X-API-Key": "test-key", "X-Webhook-Token": "wrong"},
                    json={"sender": "p", "message": "hi"},
                )
        assert response.status_code == 401, response.text
    finally:
        _restore_app(app, original)


def _rasa_shim_log():
    """Collect the shim's own log records via a handler on its logger (the
    app's boot reconfigures root logging, so a root capture can miss them)."""
    import logging
    from contextlib import contextmanager

    @contextmanager
    def capture():
        records = []

        class _Collect(logging.Handler):
            def emit(self, record):
                records.append(record)

        target = logging.getLogger("kestrel_sovereign.endpoints.rasa_shim")
        handler = _Collect(level=logging.WARNING)
        target.addHandler(handler)
        try:
            yield records
        finally:
            target.removeHandler(handler)

    return capture()


def test_unlisted_agent_answers_the_middlewares_own_not_found_envelope_and_logs():
    """#3220 round 2: an unlisted agent answers the same ``agent_not_found``
    envelope the routing middleware answers for an absent one (not a
    secrecy property — the echoed name and the rate bucket still tell them
    apart), and the refusal is host-logged naming the agent and the
    variable to set. The comparison is case-insensitive, as
    ``AgentManager.get_agent`` is: routing key ``Nellie``, list ``nellie``.
    """
    nellie = _rasa_agent("reply from nellie")
    env = {
        "KESTREL_API_KEY": "test-key",
        "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token",
        "KESTREL_RASA_WEBHOOK_AGENTS": "nellie",
    }
    app, original = _prepare_multi_agent_app({"Nellie": nellie, "Emma": _rasa_agent("reply from emma")})
    try:
        with patch.dict("os.environ", env), _rasa_shim_log() as records:
            with TestClient(app) as client:
                listed = client.post(
                    "/api/agents/Nellie/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "p-1", "message": "hi"},
                )
                assert listed.status_code == 200, listed.text
                unlisted = client.post(
                    "/api/agents/Emma/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "p-2", "message": "hi"},
                )
                unknown = client.post(
                    "/api/agents/Nobody/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "p-3", "message": "hi"},
                )
        assert unlisted.status_code == unknown.status_code == 404
        assert unlisted.json()["error"]["code"] == unknown.json()["error"]["code"]
        assert (
            unlisted.json()["error"]["message"].replace("Emma", "X")
            == unknown.json()["error"]["message"].replace("Nobody", "X")
        )
        refusals = [r.getMessage() for r in records if "KESTREL_RASA_WEBHOOK_AGENTS" in r.getMessage()]
        assert refusals and "Emma" in refusals[0], [r.getMessage() for r in records]
    finally:
        _restore_app(app, original)


def test_concurrency_gate_is_per_routed_agent():
    """#3220 round 2: the ten-permit semaphore was one module global. Before
    the routed agent was bound only the host default could be invoked here,
    so the permits belonged to one agent; shared across the fleet, agent A's
    slow turns would stall agent B with no 429 and no timeout. With A's gate
    fully held, B's request must still complete.
    """
    import asyncio

    from kestrel_sovereign.endpoints import rasa_shim

    sa = _rasa_agent("reply from sa")
    sb = _rasa_agent("reply from sb")
    env = {
        "KESTREL_API_KEY": "test-key",
        "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token",
        "KESTREL_RASA_WEBHOOK_AGENTS": "sa,sb",
    }
    gate_a = rasa_shim._agent_semaphore_for("sa")
    gate_b = rasa_shim._agent_semaphore_for("sb")
    assert gate_a is rasa_shim._agent_semaphore_for("sa")
    assert gate_a is not gate_b
    assert rasa_shim._agent_semaphore_for(None) is rasa_shim._agent_semaphore_for("default")

    # The wiring, observed from inside the turn: while B's process_input
    # runs, B's own gate holds exactly one permit and A's is untouched — a
    # postcondition the handler produces, not one the test writes.
    seen = {}

    async def observe(**_kwargs):
        seen["b_permits"] = gate_b._value
        seen["a_locked"] = gate_a.locked()
        return "reply from sb"

    sb.process_input = AsyncMock(side_effect=observe)

    async def drain():
        for _ in range(rasa_shim._AGENT_CONCURRENCY):
            await gate_a.acquire()

    asyncio.run(drain())
    app, original = _prepare_multi_agent_app({"sa": sa, "sb": sb})
    try:
        assert gate_a.locked()
        with patch.dict("os.environ", env):
            with TestClient(app) as client:
                other = client.post(
                    "/api/agents/sb/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "p-b", "message": "hi"},
                )
        assert other.status_code == 200, other.text
        sb.process_input.assert_awaited_once()
        assert seen == {"b_permits": rasa_shim._AGENT_CONCURRENCY - 1, "a_locked": True}, seen
        assert gate_a.locked()  # B's turn never touched A's permits
        assert gate_b._value == rasa_shim._AGENT_CONCURRENCY  # released after the turn
    finally:
        for _ in range(rasa_shim._AGENT_CONCURRENCY):
            gate_a.release()
        _restore_app(app, original)


def test_a_routed_agent_the_registry_can_no_longer_name_fails_closed():
    """#3220 round 3: ``_routed_agent_name`` is ``None`` for the unprefixed
    form AND for a prefixed request whose agent was fenced between routing
    and the opt-in check. The second must not fall into the permissive
    unprefixed branch: refuse, invoke nobody.
    """
    ghost = _rasa_agent("reply from ghost")
    app, original = _prepare_multi_agent_app({"ghost": ghost})
    # Routed (middleware resolves it) but the registry no longer names it.
    app.state.agent_manager.get_agent_name = MagicMock(return_value=None)
    try:
        with patch.dict(
            "os.environ",
            {
                "KESTREL_API_KEY": "test-key",
                "KESTREL_RASA_WEBHOOK_TOKEN": "rasa-token",
                "KESTREL_RASA_WEBHOOK_AGENTS": "ghost",
            },
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/agents/ghost/webhooks/rest/webhook",
                    headers=_api_headers(),
                    json={"sender": "p", "message": "hi"},
                )
        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "agent_not_found"
        ghost.process_input.assert_not_awaited()
    finally:
        _restore_app(app, original)
