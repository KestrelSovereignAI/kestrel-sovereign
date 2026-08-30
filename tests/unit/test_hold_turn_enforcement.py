"""Turn-start Hold is unconditional, typed, and source-independent (#3162)."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from types import MethodType
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.hold import (
    EffectiveHoldState,
    HoldEnforcementUnavailableError,
    HoldScope,
    HoldState,
    HoldTurnRefusal,
    require_context_hold_store,
    require_turn_start_allowed,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent


def _latch(scope: HoldScope, receipt_id: str, *, target: str) -> HoldState:
    return HoldState(
        scope=scope,
        target_id=target,
        reason=f"reason:{receipt_id}",
        actor_id=f"actor:{receipt_id}",
        set_at="2026-08-28T12:00:00+00:00",
        hold_receipt_id=receipt_id,
        revision=7,
    )


class _Store:
    def __init__(self, effective: EffectiveHoldState) -> None:
        self.effective = effective
        self.calls: list[str] = []

    async def get_effective(self, agent_id: str) -> EffectiveHoldState:
        self.calls.append(agent_id)
        return self.effective


def _bare_agent(store: _Store) -> KestrelAgent:
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.did = "did:test:held"
    agent._hold_store = store
    return agent


@pytest.mark.asyncio
async def test_agent_closes_legacy_owned_hold_context_after_shutdown_work() -> None:
    events: list[str] = []

    class _Resource:
        def __init__(self, label: str) -> None:
            self.label = label

        async def close(self) -> None:
            events.append(self.label)

    agent = KestrelAgent.__new__(KestrelAgent)
    agent._standalone_hold_context = SimpleNamespace(
        session_factory=_Resource("factory"),
        db=_Resource("database"),
    )

    cancelled, failure = await agent._close_standalone_hold_context()

    assert cancelled is False
    assert failure is None
    assert events == ["factory", "database"]
    assert agent._standalone_hold_context is None
    assert "await self._close_standalone_hold_context()" in inspect.getsource(
        KestrelAgent.shutdown
    )


@pytest.mark.asyncio
async def test_get_agent_by_did_closes_hold_context_when_initialize_fails(
    monkeypatch,
) -> None:
    import kestrel_sovereign.hold as hold_module
    import kestrel_sovereign.main as main_module

    events: list[str] = []
    context = SimpleNamespace()

    class _Agent:
        def __init__(self, **_kwargs) -> None:
            pass

        async def initialize(self) -> None:
            raise RuntimeError("initialize failed")

    async def build_context(_agent):
        events.append("built")
        return context

    async def close_context(value):
        assert value is context
        events.append("closed")

    monkeypatch.setattr(main_module, "KestrelAgent", _Agent)
    monkeypatch.setattr(main_module, "LLMService", lambda: object())
    monkeypatch.setattr(hold_module, "build_bound_host_context", build_context)
    monkeypatch.setattr(hold_module, "close_bound_host_context", close_context)

    with pytest.raises(RuntimeError, match="initialize failed"):
        await main_module.get_agent_by_did("did:test:legacy")

    assert events == ["built", "closed"]


@pytest.mark.asyncio
async def test_bound_hold_initialization_closes_context_on_failure(monkeypatch) -> None:
    import kestrel_sovereign.hold.enforcement as enforcement

    context = SimpleNamespace()
    close_context = AsyncMock()

    class _Agent:
        async def initialize(self) -> None:
            raise RuntimeError("startup refused")

    async def build_context(_agent, *, config=None):
        assert config == {"mode": "test"}
        return context

    monkeypatch.setattr(enforcement, "build_bound_host_context", build_context)
    monkeypatch.setattr(enforcement, "close_bound_host_context", close_context)

    with pytest.raises(RuntimeError, match="startup refused"):
        await enforcement.initialize_with_bound_hold_context(
            _Agent(),
            config={"mode": "test"},
        )

    close_context.assert_awaited_once_with(context)


@pytest.mark.asyncio
async def test_hold_binding_failure_closes_context_before_refusing_startup(
    monkeypatch,
) -> None:
    """A degraded context never escapes without releasing its live backend."""
    import kestrel_sovereign.host_features.context as context_module
    import kestrel_sovereign.hold.enforcement as enforcement

    closed: list[str] = []

    class _Resource:
        def __init__(self, label: str) -> None:
            self.label = label

        async def close(self) -> None:
            closed.append(self.label)

    context = SimpleNamespace(
        hold_store=None,
        backend_error="Hold schema unavailable",
        session_factory=_Resource("factory"),
        hold_database=None,
        db=_Resource("database"),
    )

    async def build_context(*, config=None):
        assert config == {"mode": "test"}
        return context

    monkeypatch.setattr(context_module, "build_host_context", build_context)

    with pytest.raises(
        HoldEnforcementUnavailableError,
        match="Hold schema unavailable",
    ):
        await enforcement.build_bound_host_context(
            SimpleNamespace(),
            config={"mode": "test"},
        )

    assert closed == ["factory", "database"]


def test_standalone_entrypoints_use_atomic_hold_initialization_wiring() -> None:
    import kestrel_sovereign.cli as cli_module
    import kestrel_sovereign.main as main_module

    cli_source = inspect.getsource(cli_module._run_shell)
    main_source = inspect.getsource(main_module.main)
    assert "initialize_with_bound_hold_context(agent)" in cli_source
    assert "initialize_with_bound_hold_context(agent)" in main_source
    assert "build_bound_host_context(agent)" not in cli_source
    assert "build_bound_host_context(agent)" not in main_source


@pytest.mark.asyncio
async def test_process_input_refuses_at_unconditional_hold_seam() -> None:
    """Mutation tripwire: deleting the non-streaming check must run onward."""

    host = _latch(HoldScope.HOST, "hold:host", target="host")
    agent_hold = _latch(
        HoldScope.AGENT, "hold:agent", target="did:test:held"
    )
    store = _Store(EffectiveHoldState(host=host, agent=agent_hold))
    agent = _bare_agent(store)  # deliberately has no hooks manager or turn state

    with pytest.raises(HoldTurnRefusal) as caught:
        await agent.process_input("do not begin")

    assert store.calls == ["did:test:held"]
    assert caught.value.effective_state.host is host
    assert caught.value.effective_state.agent is agent_hold
    assert caught.value.metadata["host_hold"] is host
    assert caught.value.metadata["agent_hold"] is agent_hold


@pytest.mark.asyncio
async def test_streaming_input_refuses_before_first_yield_or_side_effect() -> None:
    """Mutation tripwire: deleting the streaming check must run onward."""

    agent_hold = _latch(
        HoldScope.AGENT, "hold:stream", target="did:test:held"
    )
    store = _Store(EffectiveHoldState(host=None, agent=agent_hold))
    agent = _bare_agent(store)

    stream = agent.process_input_streaming("do not stream")
    with pytest.raises(HoldTurnRefusal) as caught:
        await anext(stream)

    assert store.calls == ["did:test:held"]
    assert caught.value.host_hold is None
    assert caught.value.agent_hold is agent_hold


@pytest.mark.asyncio
async def test_hold_snapshot_is_the_turn_admission_linearization_point() -> None:
    observed = _latch(
        HoldScope.AGENT, "hold:observed", target="did:test:held"
    )

    class _ReleaseAfterReadStore(_Store):
        async def get_effective(self, agent_id: str) -> EffectiveHoldState:
            result = await super().get_effective(agent_id)
            self.effective = EffectiveHoldState(host=None, agent=None)
            return result

    store = _ReleaseAfterReadStore(EffectiveHoldState(host=None, agent=observed))
    agent = _bare_agent(store)

    with pytest.raises(HoldTurnRefusal) as caught:
        await require_turn_start_allowed(agent)

    # A later release cannot erase the exact latch that refused this turn.
    assert caught.value.agent_hold is observed
    assert store.effective.held is False


@pytest.mark.asyncio
async def test_unheld_snapshot_admits_turn_at_the_same_boundary() -> None:
    store = _Store(EffectiveHoldState(host=None, agent=None))
    agent = _bare_agent(store)

    effective = await require_turn_start_allowed(agent)

    assert effective == EffectiveHoldState(host=None, agent=None)
    assert store.calls == ["did:test:held"]


@pytest.mark.asyncio
async def test_streamed_command_reuses_its_first_hold_admission_snapshot() -> None:
    """A command has one Hold linearization point, not two conflicting reads."""
    admitted = EffectiveHoldState(host=None, agent=None)
    later_hold = _latch(
        HoldScope.AGENT,
        "hold:after-admission",
        target="did:test:held",
    )

    class _ChangingStore(_Store):
        async def get_effective(self, agent_id: str) -> EffectiveHoldState:
            effective = await super().get_effective(agent_id)
            self.effective = EffectiveHoldState(host=None, agent=later_hold)
            return effective

    store = _ChangingStore(admitted)
    agent = _bare_agent(store)
    agent.storage = object()
    agent._safe_mode = False
    agent._constitution_audit_pending = False
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._maybe_audit = AsyncMock()

    async def delegated_process_input(self, *_args, **_kwargs):
        # This is the universal check at the real process_input entrypoint. The
        # streaming wrapper must bind its already-observed snapshot around it.
        await require_turn_start_allowed(self)
        return "command complete"

    agent.process_input = MethodType(delegated_process_input, agent)

    chunks = [chunk async for chunk in agent.process_input_streaming("!status")]

    assert chunks == ["command complete"]
    assert store.calls == ["did:test:held"]


def test_production_context_without_hold_store_fails_closed() -> None:
    context = SimpleNamespace(hold_store=None, backend_error="database locked")

    with pytest.raises(HoldEnforcementUnavailableError, match="database locked"):
        require_context_hold_store(context)


def test_held_refusal_cannot_be_built_from_an_unheld_snapshot() -> None:
    with pytest.raises(ValueError, match="active latch"):
        HoldTurnRefusal(
            agent_id="did:test:held",
            effective_state=EffectiveHoldState(host=None, agent=None),
        )


def _held_refusal() -> HoldTurnRefusal:
    return HoldTurnRefusal(
        agent_id="did:test:held",
        effective_state=EffectiveHoldState(
            host=_latch(HoldScope.HOST, "hold:http", target="host"),
            agent=None,
        ),
    )


def _agent_api_app(agent) -> FastAPI:
    from kestrel_sovereign.api_errors import register_api_error_handlers
    from kestrel_sovereign.endpoints.agent import router
    from kestrel_sovereign.rate_limit import limiter

    app = FastAPI()
    app.state.limiter = limiter
    app.state.agent = agent
    app.include_router(router)
    register_api_error_handlers(app)
    return app


def _transport_app(agent, *routers) -> FastAPI:
    from kestrel_sovereign.api_errors import register_api_error_handlers
    from kestrel_sovereign.rate_limit import limiter

    app = FastAPI()
    app.state.limiter = limiter
    app.state.agent = agent
    for router in routers:
        app.include_router(router)
    register_api_error_handlers(app)
    return app


def test_invoke_maps_hold_to_typed_http_refusal() -> None:
    refusal = _held_refusal()
    agent = MagicMock()
    agent.process_input = AsyncMock(side_effect=refusal)
    agent.register_active_request = MagicMock()
    agent._cleanup_cancelled_request = MagicMock()
    agent.storage.resolve_session_id = AsyncMock(side_effect=lambda value: value)

    response = TestClient(_agent_api_app(agent)).post(
        "/api/agent/invoke",
        json={"input": "do not run", "request_id": "held-http-turn"},
    )

    assert response.status_code == 423
    body = response.json()
    assert body["error"]["code"] == "agent_held"
    evidence = body["error"]["details"][0]
    assert evidence == refusal.wire_payload()
    assert evidence["host_hold"]["hold_receipt_id"] == "hold:http"
    assert evidence["agent_hold"] is None
    agent._cleanup_cancelled_request.assert_called_once_with("held-http-turn")


def test_stream_emits_machine_typed_hold_record_not_agent_prose() -> None:
    refusal = _held_refusal()

    async def held_stream(*_args, **_kwargs):
        raise refusal
        yield  # pragma: no cover - keeps this an async generator

    agent = MagicMock()
    agent.process_input_streaming = held_stream
    agent.register_active_request = MagicMock()
    agent._cleanup_cancelled_request = MagicMock()
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent.storage.resolve_session_id = AsyncMock(side_effect=lambda value: value)

    response = TestClient(_agent_api_app(agent)).post(
        "/api/agent/stream",
        json={"input": "do not stream", "request_id": "held-stream-turn"},
    )

    assert response.status_code == 200
    assert json.loads(response.text) == refusal.wire_payload()
    assert "selected model route" not in response.text
    agent._cleanup_cancelled_request.assert_called_once_with("held-stream-turn")


def test_compatibility_http_surfaces_preserve_typed_hold_refusal(
    monkeypatch,
) -> None:
    """OpenAI, Rasa, and sovereignty adapters cannot collapse Hold to 500."""

    from kestrel_sovereign.endpoints.models import router as models_router
    from kestrel_sovereign.endpoints.rasa_shim import router as rasa_router
    from kestrel_sovereign.endpoints.sovereignty import router as sovereignty_router

    refusal = _held_refusal()
    agent = MagicMock()
    agent.process_input = AsyncMock(side_effect=refusal)
    agent.llm_service.get_active_model_id.return_value = "gpt-test"
    app = _transport_app(
        agent,
        models_router,
        rasa_router,
        sovereignty_router,
    )
    monkeypatch.setenv("KESTREL_RASA_WEBHOOK_TOKEN", "rasa-secret")

    with TestClient(app) as client:
        responses = (
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "messages": [{"role": "user", "content": "held"}],
                },
            ),
            client.post(
                "/webhooks/rest/webhook",
                headers={"X-Webhook-Token": "rasa-secret"},
                json={"sender": "patient-1", "message": "held"},
            ),
            client.post(
                "/api/sovereignty/import",
                json={"cid": "bafyHeldTransport"},
            ),
        )

    for response in responses:
        assert response.status_code == 423, response.text
        body = response.json()["error"]
        assert body["code"] == "agent_held"
        assert body["details"] == [refusal.wire_payload()]


def test_bridge_surfaces_preserve_typed_hold_refusal() -> None:
    """Bridge sync and SSE clients receive the same exact refusal evidence."""

    from kestrel_sovereign.features.bridge.router import get_router

    refusal = _held_refusal()
    bridge = MagicMock()
    bridge.get_or_create_session = AsyncMock(
        return_value=SimpleNamespace(id="held-bridge-session")
    )
    bridge.log_invocation = AsyncMock()
    agent = MagicMock()
    agent.features = {"BridgeFeature": bridge}
    agent.process_input = AsyncMock(side_effect=refusal)

    async def held_stream(*_args, **_kwargs):
        raise refusal
        yield  # pragma: no cover - keeps this an async generator

    agent.process_input_streaming = held_stream
    agent.register_active_request = MagicMock()
    agent._cleanup_cancelled_request = MagicMock()
    app = _transport_app(agent, get_router())

    with TestClient(app) as client:
        invoke = client.post(
            "/api/bridge/invoke",
            json={"message": "held", "channel_type": "api"},
        )
        stream = client.post(
            "/api/bridge/stream",
            json={"message": "held", "channel_type": "api"},
        )

    assert invoke.status_code == 423
    assert invoke.json()["error"]["details"] == [refusal.wire_payload()]
    assert stream.status_code == 200
    assert "event: refusal" in stream.text
    data_line = next(
        line for line in stream.text.splitlines() if line.startswith("data: ")
    )
    assert json.loads(data_line.removeprefix("data: ")) == refusal.wire_payload()
    agent._cleanup_cancelled_request.assert_called_once()
