"""Sovereign agent-card Hold and Resume door (#3164)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.hold import (
    EffectiveHoldState,
    HoldAction,
    HoldDisposition,
    HoldMutation,
    HoldReceipt,
    HoldScope,
    HoldState,
)


def _state(
    *,
    target_id: str = "did:test:emma",
    receipt_id: str = "hold-receipt-1",
    reason: str = "operator pause",
    actor_id: str = "sovereign-key",
) -> HoldState:
    return HoldState(
        scope=HoldScope.AGENT,
        target_id=target_id,
        reason=reason,
        actor_id=actor_id,
        set_at="2026-08-30T12:00:00+00:00",
        hold_receipt_id=receipt_id,
        revision=1,
    )


def _mutation(
    *,
    action: HoldAction,
    disposition: HoldDisposition = HoldDisposition.APPLIED,
    current: HoldState | None = None,
    operation_id: str,
    expected_hold_receipt_id: str = "",
) -> HoldMutation:
    prior = current.hold_receipt_id if current is not None else "hold-receipt-1"
    resulting = current.hold_receipt_id if current is not None else ""
    return HoldMutation(
        receipt=HoldReceipt(
            receipt_id=f"receipt-{operation_id}",
            operation_id=operation_id,
            action=action,
            disposition=disposition,
            scope=HoldScope.AGENT,
            target_id="did:test:emma",
            reason=(
                current.reason
                if action is HoldAction.HOLD and current is not None
                else "operator resume"
            ),
            actor_id="sovereign-key",
            occurred_at="2026-08-30T12:00:01+00:00",
            expected_hold_receipt_id=expected_hold_receipt_id,
            prior_hold_receipt_id=prior,
            resulting_hold_receipt_id=resulting,
        ),
        current=current,
    )


def _app(*, caller: CallerContext, store=None, agent_id="did:test:emma"):
    from kestrel_sovereign.api_errors import register_api_error_handlers
    from kestrel_sovereign.endpoints.agent_hold import router

    app = FastAPI()
    register_api_error_handlers(app)
    app.include_router(router)
    agent = SimpleNamespace(agent_id=agent_id)
    manager = MagicMock()
    manager.get_agent.side_effect = lambda name: agent if name == "Emma" else None
    app.state.agent_manager = manager
    app.state.host_context = SimpleNamespace(hold_store=store)

    @app.middleware("http")
    async def bind_caller(request: Request, call_next):
        request.state.caller = caller
        return await call_next(request)

    return app, manager


def test_hold_agent_binds_path_to_expected_did_and_returns_typed_receipt():
    held = _state()
    store = MagicMock()
    store.set_hold = AsyncMock(
        return_value=_mutation(
            action=HoldAction.HOLD,
            current=held,
            operation_id="hold-op-1",
        )
    )
    app, _manager = _app(
        caller=CallerContext.sovereign(identity="sovereign-key"),
        store=store,
    )

    response = TestClient(app).post(
        "/api/host/holds/agents/Emma",
        json={
            "target_agent_id": "did:test:emma",
            "reason": "operator pause",
            "operation_id": "hold-op-1",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["receipt"] == {
        "receipt_id": "receipt-hold-op-1",
        "operation_id": "hold-op-1",
        "action": "hold",
        "disposition": "applied",
        "scope": "agent",
        "target_id": "did:test:emma",
        "reason": "operator pause",
        "actor_id": "sovereign-key",
        "occurred_at": "2026-08-30T12:00:01+00:00",
        "expected_hold_receipt_id": "",
        "prior_hold_receipt_id": "hold-receipt-1",
        "resulting_hold_receipt_id": "hold-receipt-1",
    }
    assert response.json()["current"]["hold_receipt_id"] == "hold-receipt-1"
    store.set_hold.assert_awaited_once_with(
        scope=HoldScope.AGENT,
        target_id="did:test:emma",
        actor_id="sovereign-key",
        reason="operator pause",
        operation_id="hold-op-1",
    )


def test_resume_releases_only_the_observed_agent_latch():
    store = MagicMock()
    store.release_hold = AsyncMock(
        return_value=_mutation(
            action=HoldAction.RELEASE,
            current=None,
            operation_id="resume-op-1",
            expected_hold_receipt_id="hold-receipt-1",
        )
    )
    app, _manager = _app(caller=CallerContext.sovereign(), store=store)

    response = TestClient(app).post(
        "/api/host/holds/agents/Emma/release",
        json={
            "target_agent_id": "did:test:emma",
            "reason": "operator resume",
            "operation_id": "resume-op-1",
            "expected_hold_receipt_id": "hold-receipt-1",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert response.json()["current"] is None
    store.release_hold.assert_awaited_once_with(
        scope=HoldScope.AGENT,
        target_id="did:test:emma",
        actor_id="api_key",
        reason="operator resume",
        operation_id="resume-op-1",
        expected_hold_receipt_id="hold-receipt-1",
    )


def test_stale_resume_is_receipted_and_leaves_replacement_hold_visible():
    replacement = _state(receipt_id="hold-receipt-2", reason="new reason")
    store = MagicMock()
    store.release_hold = AsyncMock(
        return_value=_mutation(
            action=HoldAction.RELEASE,
            disposition=HoldDisposition.REFUSED_STALE,
            current=replacement,
            operation_id="resume-op-stale",
            expected_hold_receipt_id="hold-receipt-1",
        )
    )
    app, _manager = _app(caller=CallerContext.sovereign(), store=store)

    response = TestClient(app).post(
        "/api/host/holds/agents/Emma/release",
        json={
            "target_agent_id": "did:test:emma",
            "reason": "operator resume",
            "operation_id": "resume-op-stale",
            "expected_hold_receipt_id": "hold-receipt-1",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["success"] is False
    assert response.json()["receipt"]["disposition"] == "refused_stale"
    assert response.json()["current"]["hold_receipt_id"] == "hold-receipt-2"


def test_hold_requires_sovereign_authority_before_store_mutation():
    store = MagicMock()
    store.set_hold = AsyncMock()
    app, _manager = _app(
        caller=CallerContext.authenticated("operator@example.test"),
        store=store,
    )

    response = TestClient(app).post(
        "/api/host/holds/agents/Emma",
        json={
            "target_agent_id": "did:test:emma",
            "reason": "operator pause",
            "operation_id": "hold-op-1",
        },
    )

    assert response.status_code == 403
    store.set_hold.assert_not_awaited()


def test_hold_refuses_a_stale_name_to_did_binding():
    store = MagicMock()
    store.set_hold = AsyncMock()
    app, _manager = _app(caller=CallerContext.sovereign(), store=store)

    response = TestClient(app).post(
        "/api/host/holds/agents/Emma",
        json={
            "target_agent_id": "did:test:someone-else",
            "reason": "operator pause",
            "operation_id": "hold-op-1",
        },
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "agent_identity_changed"
    store.set_hold.assert_not_awaited()


def test_missing_hold_store_is_operator_visible_and_fails_without_mutation():
    app, _manager = _app(caller=CallerContext.sovereign(), store=None)

    response = TestClient(app).post(
        "/api/host/holds/agents/Emma",
        json={
            "target_agent_id": "did:test:emma",
            "reason": "operator pause",
            "operation_id": "hold-op-1",
        },
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "hold_state_unavailable"


def test_effective_hold_payload_keeps_host_and_agent_latches_distinct():
    from kestrel_sovereign.endpoints.agent_hold import effective_hold_payload

    host = HoldState(
        scope=HoldScope.HOST,
        target_id="host",
        reason="fleet maintenance",
        actor_id="sovereign-key",
        set_at="2026-08-30T11:00:00+00:00",
        hold_receipt_id="host-hold-1",
        revision=1,
    )
    agent = _state()

    payload = effective_hold_payload(EffectiveHoldState(host=host, agent=agent))

    assert payload["available"] is True
    assert payload["held"] is True
    assert payload["sources"] == ["host", "agent"]
    assert payload["host"]["hold_receipt_id"] == "host-hold-1"
    assert payload["agent"]["hold_receipt_id"] == "hold-receipt-1"


async def test_agent_inventory_reads_durable_effective_hold_for_each_card():
    from kestrel_sovereign.endpoints.models import get_agents

    agent = MagicMock(agent_id="did:test:emma", is_demo=False)
    card = MagicMock()
    card.model_dump.return_value = {"name": "Emma", "description": "agent"}
    agent.get_agent_card = AsyncMock(return_value=card)
    manager = MagicMock()
    manager.list_agents.return_value = {"Emma": agent}
    store = MagicMock()
    store.get_effective = AsyncMock(
        return_value=EffectiveHoldState(host=None, agent=_state())
    )
    request = MagicMock()
    request.app.state.agent_manager = manager
    request.app.state.demo_mode = False
    request.app.state.host_context = SimpleNamespace(hold_store=store)

    result = await get_agents(request)

    assert result["agents"][0]["hold"]["held"] is True
    assert (
        result["agents"][0]["hold"]["agent"]["hold_receipt_id"]
        == "hold-receipt-1"
    )
    store.get_effective.assert_awaited_once_with("did:test:emma")


async def test_agent_inventory_never_reports_missing_hold_state_as_unheld():
    from kestrel_sovereign.endpoints.models import get_agents

    agent = MagicMock(agent_id="did:test:emma", is_demo=False)
    card = MagicMock()
    card.model_dump.return_value = {"name": "Emma"}
    agent.get_agent_card = AsyncMock(return_value=card)
    manager = MagicMock()
    manager.list_agents.return_value = {"Emma": agent}
    request = MagicMock()
    request.app.state.agent_manager = manager
    request.app.state.demo_mode = False
    request.app.state.host_context = SimpleNamespace(hold_store=None)

    result = await get_agents(request)

    assert result["agents"][0]["hold"] == {
        "available": False,
        "held": None,
        "sources": [],
        "host": None,
        "agent": None,
        "error": "hold_state_unavailable",
    }


def test_server_mounts_agent_hold_routes():
    from kestrel_sovereign import server

    paths = {getattr(route, "path", None) for route in server.app.routes}
    assert "/api/host/holds/agents/{agent_name}" in paths
    assert "/api/host/holds/agents/{agent_name}/release" in paths
