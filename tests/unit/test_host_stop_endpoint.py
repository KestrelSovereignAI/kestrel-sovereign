"""Host-scope cooperative Stop door (#3154)."""

from dataclasses import replace
import inspect
from unittest.mock import AsyncMock, MagicMock, call

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.endpoints.host_stop import router
from kestrel_sovereign.stop import StopReceipt


class _ReceiptStore:
    def __init__(self, on_load=None):
        self.on_load = on_load
        self.persisted = []
        self.replay = None

    async def load(self, request):
        if self.on_load is not None:
            self.on_load()
        return self.replay

    async def persist(self, request, outcomes):
        receipt_id = f"receipt-{request.correlation_id}"
        receipted = tuple(
            replace(outcome, receipt_id=receipt_id) for outcome in outcomes
        )
        receipt = StopReceipt(
            receipt_id=receipt_id,
            operation_id=request.correlation_id,
            request_fingerprint="test-fingerprint",
            scope=request.scope.value,
            actor_id=request.actor_id,
            requested_target=request.target,
            target_agent_id=request.target_agent_id,
            reason=request.reason,
            cascade=request.cascade,
            occurred_at="2026-08-30T00:00:00+00:00",
            turn_id=request.turn_id,
            span_id=request.span_id,
            trace_id=request.trace_id,
            outcomes=receipted,
        )
        self.persisted.append((request, receipt))
        return receipt


def _agent(agent_id: str, turns=()):
    agent = MagicMock()
    agent.agent_id = agent_id
    agent._active_request_ids = set(turns)
    agent._current_request_id = next(iter(turns), None)
    agent.cancel_current_request = MagicMock(
        side_effect=lambda request_id=None, **_kwargs: request_id in turns
    )
    agent.wait_for_request_completion = AsyncMock(return_value=None)
    agent.terminate = MagicMock()
    agent.shutdown = AsyncMock()
    return agent


def _app(*, agents, caller, receipt_store=True):
    app = FastAPI()
    app.include_router(router)
    manager = MagicMock()
    manager.list_agents.return_value = dict(agents)
    manager.stop_agent = MagicMock()
    manager.terminate_agent = MagicMock()
    app.state.agent_manager = manager
    if receipt_store is True:
        app.state.stop_receipt_store = _ReceiptStore()
    elif receipt_store is not False:
        app.state.stop_receipt_store = receipt_store

    @app.middleware("http")
    async def bind_caller(request: Request, call_next):
        request.state.caller = caller
        return await call_next(request)

    return app, manager


def test_host_stop_fans_out_with_one_receipted_outcome_per_agent():
    alpha = _agent("did:test:alpha", {"turn-alpha"})
    beta = _agent("did:test:beta", {"turn-beta"})
    app, manager = _app(
        agents={"Beta": beta, "Alpha": alpha},
        caller=CallerContext.sovereign(identity="sovereign-key"),
    )

    response = TestClient(app).post(
        "/api/host/stop",
        json={"reason": "andon", "correlation_id": "fleet-stop"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["success"] is True
    assert payload["state"] == "confirmed"
    assert payload["target_count"] == 2
    assert [item["agent_id"] for item in payload["stop_outcomes"]] == [
        "did:test:alpha",
        "did:test:beta",
    ]
    assert all(item["receipt_id"] for item in payload["stop_outcomes"])
    alpha.cancel_current_request.assert_called_once_with(request_id="turn-alpha")
    beta.cancel_current_request.assert_called_once_with(request_id="turn-beta")
    alpha.terminate.assert_not_called()
    beta.shutdown.assert_not_awaited()
    manager.stop_agent.assert_not_called()
    manager.terminate_agent.assert_not_called()


def test_host_stop_requires_sovereign_not_merely_authenticated_identity():
    agent = _agent("did:test:agent", {"turn"})
    app, _manager = _app(
        agents={"Agent": agent},
        caller=CallerContext.authenticated("operator@example.test"),
    )

    response = TestClient(app).post("/api/host/stop")

    assert response.status_code == 403
    agent.cancel_current_request.assert_not_called()


def test_host_stop_preserves_partial_outcomes_and_continues_later_targets():
    broken = _agent("did:test:a-broken", {"broken-turn"})
    healthy = _agent("did:test:z-healthy", {"healthy-turn"})
    broken.wait_for_request_completion.side_effect = RuntimeError("partition")
    app, _manager = _app(
        agents={"Broken": broken, "Healthy": healthy},
        caller=CallerContext.sovereign(),
    )

    response = TestClient(app).post("/api/host/stop")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["state"] == "partial"
    assert [item["disposition"] for item in payload["stop_outcomes"]] == [
        "unreachable",
        "stopped",
    ]
    healthy.cancel_current_request.assert_called_once_with(
        request_id="healthy-turn"
    )


def test_empty_host_inventory_is_distinct_and_never_claimed_successful():
    app, _manager = _app(
        agents={},
        caller=CallerContext.sovereign(),
    )

    response = TestClient(app).post("/api/host/stop")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["state"] == "empty"
    assert payload["target_count"] == 0
    assert payload["stop_outcomes"][0]["disposition"] == "unreachable"


def test_receipt_unavailability_refuses_before_any_cancellation():
    agent = _agent("did:test:agent", {"turn"})
    app, _manager = _app(
        agents={"Agent": agent},
        caller=CallerContext.sovereign(),
        receipt_store=False,
    )

    response = TestClient(app).post("/api/host/stop")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "unconfirmed"
    assert payload["stop_outcomes"][0]["disposition"] == "refused"
    agent.cancel_current_request.assert_not_called()


def test_payload_identity_cannot_select_or_expand_host_targets():
    agent = _agent("did:test:agent", {"turn"})
    app, _manager = _app(
        agents={"Agent": agent},
        caller=CallerContext.sovereign(),
    )

    response = TestClient(app).post(
        "/api/host/stop",
        json={"target": "did:test:forged"},
    )

    assert response.status_code == 422
    agent.cancel_current_request.assert_not_called()


def test_invalid_operation_identity_is_rejected_before_inventory_side_effects():
    agent = _agent("did:test:agent", {"turn"})
    app, _manager = _app(
        agents={"Agent": agent},
        caller=CallerContext.sovereign(),
    )

    response = TestClient(app).post(
        "/api/host/stop",
        json={"correlation_id": "é" * 129},
    )

    assert response.status_code == 422
    agent.cancel_current_request.assert_not_called()


def test_invalid_host_inventory_fails_closed_without_partial_cancellation():
    healthy = _agent("did:test:healthy", {"turn"})
    invalid = _agent("did:test:placeholder")
    invalid.agent_id = None
    app, _manager = _app(
        agents={"Healthy": healthy, "Invalid": invalid},
        caller=CallerContext.sovereign(),
    )

    response = TestClient(app).post("/api/host/stop")

    assert response.status_code == 503
    healthy.cancel_current_request.assert_not_called()


def test_host_stop_rechecks_work_admitted_during_receipt_preflight():
    agent = _agent("did:test:agent", {"turn-a"})

    def admit_turn():
        agent._active_request_ids.add("turn-b")
        agent.cancel_current_request.side_effect = (
            lambda request_id=None, **_kwargs: request_id in {"turn-a", "turn-b"}
        )

    app, _manager = _app(
        agents={"Agent": agent},
        caller=CallerContext.sovereign(),
        receipt_store=_ReceiptStore(on_load=admit_turn),
    )

    response = TestClient(app).post("/api/host/stop")

    assert response.status_code == 200, response.text
    assert agent.cancel_current_request.call_args_list == [
        call(request_id="turn-a"),
        call(request_id="turn-b"),
    ]


def test_host_stop_replay_summary_uses_receipted_not_fresh_inventory():
    original = _agent("did:test:original", {"original-turn"})
    store = _ReceiptStore()
    app, manager = _app(
        agents={"Original": original},
        caller=CallerContext.sovereign(),
        receipt_store=store,
    )
    client = TestClient(app)
    body = {"reason": "andon", "correlation_id": "stable-host-stop"}

    first = client.post("/api/host/stop", json=body)
    assert first.status_code == 200
    store.replay = store.persisted[0][1]
    later = _agent("did:test:later", {"later-turn"})
    manager.list_agents.return_value = {
        "Later": later,
        "Original": original,
    }

    replay = client.post("/api/host/stop", json=body)

    assert replay.status_code == 200
    assert replay.json()["state"] == first.json()["state"]
    assert replay.json()["target_count"] == first.json()["target_count"] == 1
    assert replay.json()["confirmed_count"] == 1
    later.cancel_current_request.assert_not_called()


def test_server_mounts_host_stop_and_implementation_has_no_process_lifecycle():
    from kestrel_sovereign import server
    from kestrel_sovereign.endpoints import host_stop

    paths = {getattr(route, "path", None) for route in server.app.routes}
    assert "/api/host/stop" in paths
    source = inspect.getsource(host_stop)
    assert "process_manager" not in source
    assert ".terminate" not in source
    assert ".shutdown" not in source
