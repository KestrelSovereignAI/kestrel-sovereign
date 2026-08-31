"""Typed Stop requests and the one cancellation authority (#3139)."""

import asyncio
import gc
import inspect
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.stop import (
    CancellationAuthority,
    CooperativeStopTarget,
    StopCleanupRegistry,
    StopDisposition,
    StopOutcome,
    StopRequest,
    StopScope,
)


def _authority(target_inventory, **kwargs) -> CancellationAuthority:
    return CancellationAuthority(
        target_inventory,
        cleanup_registry=StopCleanupRegistry(),
        **kwargs,
    )


def test_stop_request_and_outcome_round_trip_exact_wire_values() -> None:
    request = StopRequest(
        scope=StopScope.TURN,
        target="turn-7",
        target_agent_id="did:test:target",
        actor_id="did:test:operator",
        reason="unsafe loop",
        cascade=False,
        correlation_id="stop-7",
        target_is_turn_id=True,
    )
    assert StopRequest.from_dict(request.to_dict()) == request

    resolved_request = StopRequest(
        scope=StopScope.TURN,
        target="request-private",
        target_agent_id="did:test:target",
        actor_id="did:test:operator",
        correlation_id="stop-resolved",
        request_generation=7,
    )
    assert StopRequest.from_dict(resolved_request.to_dict()) == resolved_request

    outcome = StopOutcome(
        scope=request.scope,
        requested_target=request.target,
        resolved_target="agent-a",
        agent_id="did:test:a",
        disposition=StopDisposition.STOPPED,
        correlation_id=request.correlation_id,
        detail="cooperative boundary observed",
        receipt_id="receipt-7",
    )
    assert StopOutcome.from_dict(outcome.to_dict()) == outcome
    assert outcome.to_dict()["disposition"] == "stopped"


@pytest.mark.asyncio
async def test_authority_rejects_unvalidated_request_before_inventory_or_cancel() -> None:
    inventory = MagicMock()
    cancel = AsyncMock(return_value=StopDisposition.STOPPED)
    inventory.return_value = (
        CooperativeStopTarget("did:test:target", "did:test:target", cancel),
    )
    malformed = SimpleNamespace(
        scope=StopScope.AGENT,
        actor_id=" ",
        target="did:test:target",
        target_agent_id=None,
        reason=None,
        cascade=True,
        correlation_id="malformed-request",
    )

    with pytest.raises(TypeError, match="validated StopRequest"):
        await _authority(inventory).stop(malformed)

    inventory.assert_not_called()
    cancel.assert_not_awaited()


@pytest.mark.parametrize("scope", [StopScope.AGENT, StopScope.TURN, StopScope.TOOL_CALL])
def test_addressed_scopes_require_a_target(scope: StopScope) -> None:
    with pytest.raises(ValueError, match="requires a target"):
        StopRequest(scope=scope, actor_id="did:test:operator")


def test_host_scope_rejects_a_target() -> None:
    with pytest.raises(ValueError, match="cannot carry a target"):
        StopRequest(
            scope=StopScope.HOST,
            target="not-a-process-or-agent",
            actor_id="did:test:operator",
        )


@pytest.mark.asyncio
async def test_host_fanout_preserves_partial_per_target_outcomes() -> None:
    async def stopped(_request: StopRequest) -> StopDisposition:
        return StopDisposition.STOPPED

    async def already_complete(_request: StopRequest) -> StopDisposition:
        return StopDisposition.ALREADY_COMPLETE

    async def refused(_request: StopRequest) -> StopDisposition:
        return StopDisposition.REFUSED

    authority = _authority(
        lambda: [
            CooperativeStopTarget("zulu", "did:test:z", refused),
            CooperativeStopTarget("alpha", "did:test:a", stopped),
            CooperativeStopTarget("middle", "did:test:m", already_complete),
        ]
    )

    outcomes = await authority.stop(
        StopRequest(
            scope=StopScope.HOST,
            actor_id="did:test:operator",
            correlation_id="fleet-stop",
        )
    )

    assert [outcome.resolved_target for outcome in outcomes] == [
        "alpha",
        "middle",
        "zulu",
    ]
    assert [outcome.disposition for outcome in outcomes] == [
        StopDisposition.STOPPED,
        StopDisposition.ALREADY_COMPLETE,
        StopDisposition.REFUSED,
    ]


@pytest.mark.asyncio
async def test_host_fanout_converts_target_failure_and_continues() -> None:
    calls: list[str] = []

    async def unreachable(_request: StopRequest) -> StopDisposition:
        calls.append("broken")
        raise ConnectionError("private remote detail")

    async def stopped(_request: StopRequest) -> StopDisposition:
        calls.append("healthy")
        return StopDisposition.STOPPED

    authority = _authority(
        lambda: [
            CooperativeStopTarget("a-broken", "did:test:a", unreachable),
            CooperativeStopTarget("z-healthy", "did:test:z", stopped),
        ]
    )

    outcomes = await authority.stop(
        StopRequest(StopScope.HOST, "did:test:operator")
    )

    assert calls == ["broken", "healthy"]
    assert [outcome.disposition for outcome in outcomes] == [
        StopDisposition.UNREACHABLE,
        StopDisposition.STOPPED,
    ]
    assert "private remote detail" not in outcomes[0].detail


@pytest.mark.asyncio
async def test_host_fanout_times_out_hung_target_without_blocking_healthy_target() -> None:
    healthy_ran = False
    healthy_started = asyncio.Event()
    hung_saw_healthy = False

    async def hung(_request: StopRequest) -> StopDisposition:
        nonlocal hung_saw_healthy
        await healthy_started.wait()
        hung_saw_healthy = True
        await asyncio.Event().wait()
        return StopDisposition.STOPPED

    async def healthy(_request: StopRequest) -> StopDisposition:
        nonlocal healthy_ran
        healthy_ran = True
        healthy_started.set()
        return StopDisposition.STOPPED

    authority = _authority(
        lambda: [
            CooperativeStopTarget("a-hung", "did:test:hung", hung),
            CooperativeStopTarget("z-healthy", "did:test:healthy", healthy),
        ],
        target_timeout_seconds=0.01,
    )

    outcomes = await authority.stop(
        StopRequest(StopScope.HOST, "did:test:operator")
    )

    assert healthy_ran is True
    assert hung_saw_healthy is True
    assert [outcome.disposition for outcome in outcomes] == [
        StopDisposition.UNREACHABLE,
        StopDisposition.STOPPED,
    ]
    assert outcomes[0].detail == "Cooperative Stop target timed out"


@pytest.mark.asyncio
async def test_stop_deadline_detaches_target_that_suppresses_cancellation() -> None:
    cleanup_release = asyncio.Event()
    cancellation_observed = asyncio.Event()

    async def uncooperative(_request: StopRequest) -> StopDisposition:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            await cleanup_release.wait()
        return StopDisposition.STOPPED

    cleanup_registry = StopCleanupRegistry()
    authority = CancellationAuthority(
        lambda: [
            CooperativeStopTarget(
                "uncooperative",
                "did:test:uncooperative",
                uncooperative,
            )
        ],
        cleanup_registry=cleanup_registry,
        target_timeout_seconds=0.01,
    )

    outcomes = await asyncio.wait_for(
        authority.stop(StopRequest(StopScope.HOST, "did:test:operator")),
        timeout=0.1,
    )

    assert outcomes[0].disposition is StopDisposition.UNREACHABLE
    await asyncio.wait_for(cancellation_observed.wait(), timeout=0.1)
    assert len(cleanup_registry._tasks) == 1
    detached = next(iter(cleanup_registry._tasks))
    detached_ref = weakref.ref(detached)
    authority_ref = weakref.ref(authority)
    del detached
    del authority
    gc.collect()
    assert authority_ref() is None
    assert detached_ref() is not None
    cleanup_release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert cleanup_registry._tasks == set()


@pytest.mark.asyncio
async def test_empty_host_inventory_returns_no_agent_outcomes() -> None:
    authority = _authority(list)

    outcomes = await authority.stop(
        StopRequest(StopScope.HOST, "did:test:operator")
    )

    assert outcomes == ()


@pytest.mark.asyncio
async def test_inventory_rejects_duplicate_agent_identities() -> None:
    stop = AsyncMock(return_value=StopDisposition.STOPPED)
    authority = _authority(
        lambda: [
            CooperativeStopTarget("alias-a", "did:test:same", stop),
            CooperativeStopTarget("alias-b", "did:test:same", stop),
        ]
    )

    with pytest.raises(ValueError, match="duplicate agent identities"):
        await authority.stop(
            StopRequest(StopScope.HOST, "did:test:operator")
        )
    stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_inventory_rejects_collision_across_agent_address_namespaces() -> None:
    stop = AsyncMock(return_value=StopDisposition.STOPPED)
    authority = _authority(
        lambda: [
            CooperativeStopTarget("alias-a", "did:test:a", stop),
            CooperativeStopTarget("did:test:a", "did:test:b", stop),
        ]
    )

    with pytest.raises(ValueError, match="ambiguous agent address"):
        await authority.stop(
            StopRequest(
                StopScope.AGENT,
                "did:test:operator",
                "did:test:a",
            )
        )
    stop.assert_not_awaited()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
def test_authority_rejects_non_finite_deadline(timeout: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        _authority(list, target_timeout_seconds=timeout)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"resolved_target": None}, "resolved_target"),
        ({"agent_id": 123}, "agent_id"),
        ({"correlation_id": None}, "correlation_id"),
        ({"detail": 123}, "detail"),
        ({"receipt_id": ""}, "receipt_id"),
    ],
)
def test_stop_outcome_rejects_malformed_wire_fields(
    overrides: dict[str, object], message: str
) -> None:
    payload = StopOutcome(
        scope=StopScope.AGENT,
        requested_target="agent-a",
        resolved_target="agent-a",
        agent_id="did:test:a",
        disposition=StopDisposition.STOPPED,
        correlation_id="stop-7",
    ).to_dict()
    payload.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        StopOutcome.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "scope": "host",
            "requested_target": "agent-a",
            "resolved_target": "agent-a",
            "agent_id": "did:test:a",
            "disposition": "stopped",
            "correlation_id": "stop-7",
        },
        {
            "scope": "agent",
            "requested_target": None,
            "resolved_target": "agent-a",
            "agent_id": "did:test:a",
            "disposition": "stopped",
            "correlation_id": "stop-7",
        },
    ],
)
def test_stop_outcome_rejects_scope_target_contradictions(payload) -> None:
    with pytest.raises(ValueError, match="requested target"):
        StopOutcome.from_dict(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_id": " "},
        {"agent_id": "\t"},
        {"turn_ids": "turn-10"},
        {"turn_request_ids": {"": "request-10"}},
        {"turn_request_ids": {"turn-10": ""}},
        {"turn_request_generations": {"turn-10": 1}},
        {
            "turn_request_ids": {"turn-10": "request-10"},
            "turn_request_generations": {"turn-10": 0},
        },
        {"tool_call_ids": frozenset({""})},
    ],
)
def test_stop_target_rejects_malformed_addresses_before_cancellation(kwargs) -> None:
    values = {
        "target_id": "agent-a",
        "agent_id": "did:test:a",
        "cancel": AsyncMock(return_value=StopDisposition.STOPPED),
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        CooperativeStopTarget(**values)
    values["cancel"].assert_not_awaited()


@pytest.mark.asyncio
async def test_authority_resolves_turn_and_tool_addresses() -> None:
    async def stop(_request: StopRequest) -> StopDisposition:
        return StopDisposition.STOPPED

    authority = _authority(
        lambda: [
            CooperativeStopTarget(
                "agent-a",
                "did:test:a",
                stop,
                turn_ids=frozenset({"turn-a"}),
                tool_call_ids=frozenset({"call-a"}),
            )
        ]
    )

    turn = await authority.stop(
        StopRequest(
            StopScope.TURN,
            "did:test:operator",
            "turn-a",
            target_agent_id="did:test:a",
        )
    )
    tool = await authority.stop(
        StopRequest(
            StopScope.TOOL_CALL,
            "did:test:operator",
            "call-a",
            target_agent_id="did:test:a",
        )
    )
    missing = await authority.stop(
        StopRequest(
            StopScope.TURN,
            "did:test:operator",
            "turn-missing",
            target_agent_id="did:test:a",
        )
    )

    assert turn[0].agent_id == "did:test:a"
    assert tool[0].agent_id == "did:test:a"
    assert missing[0].disposition is StopDisposition.UNREACHABLE


@pytest.mark.asyncio
async def test_work_stop_outcome_preserves_long_agent_target_identity() -> None:
    """Invocation ID bounds never constrain the resolved agent DID."""

    long_agent_id = "did:web:example.test:" + ("agent-" * 50)
    cancel = AsyncMock(return_value=StopDisposition.STOPPED)
    authority = _authority(
        lambda: (
            CooperativeStopTarget(
                long_agent_id,
                long_agent_id,
                cancel,
                turn_ids=frozenset({"turn-long-owner"}),
            ),
        )
    )

    outcomes = await authority.stop(
        StopRequest(
            StopScope.TURN,
            "did:test:operator",
            "turn-long-owner",
            target_agent_id=long_agent_id,
        )
    )

    assert outcomes[0].resolved_target == long_agent_id
    assert outcomes[0].disposition is StopDisposition.STOPPED
    cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_authority_resolves_turn_address_to_live_cancellation_key() -> None:
    observed_targets: list[tuple[str | None, int | None]] = []

    async def stop(request: StopRequest) -> StopDisposition:
        observed_targets.append((request.target, request.request_generation))
        return StopDisposition.STOPPED

    authority = _authority(
        lambda: [
            CooperativeStopTarget(
                "agent-a",
                "did:test:a",
                stop,
                turn_ids=frozenset({"turn-visible"}),
                turn_request_ids={"turn-visible": "request-private"},
                turn_request_generations={"turn-visible": 7},
            )
        ]
    )

    outcomes = await authority.stop(
        StopRequest(
            StopScope.TURN,
            "did:test:operator",
            "turn-visible",
            target_agent_id="did:test:a",
            target_is_turn_id=True,
        )
    )

    assert observed_targets == [("request-private", 7)]
    assert outcomes[0].requested_target == "turn-visible"
    assert outcomes[0].resolved_target == "request-private"


@pytest.mark.asyncio
async def test_request_address_collision_is_not_remapped_as_public_turn() -> None:
    """Address kind, not string membership, selects turn-ID remapping."""

    observed_targets: list[str | None] = []

    async def stop(request: StopRequest) -> StopDisposition:
        observed_targets.append(request.target)
        return StopDisposition.STOPPED

    authority = _authority(
        lambda: [
            CooperativeStopTarget(
                "agent-a",
                "did:test:a",
                stop,
                turn_ids=frozenset({"collision"}),
                turn_request_ids={"collision": "another-request"},
            )
        ]
    )

    outcome = await authority.stop(
        StopRequest(
            StopScope.TURN,
            "did:test:operator",
            "collision",
            target_agent_id="did:test:a",
            target_is_turn_id=False,
        )
    )

    assert observed_targets == ["collision"]
    assert outcome[0].resolved_target == "agent-a"


@pytest.mark.asyncio
async def test_unknown_public_turn_collision_cannot_cancel_legacy_request() -> None:
    """A typed public-turn address never falls through to a request address."""

    cancel = AsyncMock(return_value=StopDisposition.STOPPED)
    authority = _authority(
        lambda: [
            CooperativeStopTarget(
                "agent-a",
                "did:test:a",
                cancel,
                turn_ids=frozenset({"collision"}),
                turn_request_ids={},
            )
        ]
    )

    outcomes = await authority.stop(
        StopRequest(
            StopScope.TURN,
            "did:test:operator",
            "collision",
            target_agent_id="did:test:a",
            target_is_turn_id=True,
        )
    )

    assert outcomes[0].disposition is StopDisposition.UNREACHABLE
    cancel.assert_not_awaited()


def test_turn_stop_preserves_accepted_opaque_whitespace_request_id() -> None:
    request = StopRequest(
        StopScope.TURN,
        "did:test:operator",
        " ",
        target_agent_id="did:test:a",
    )
    outcome = StopOutcome(
        scope=StopScope.TURN,
        requested_target=" ",
        resolved_target=" ",
        agent_id="did:test:a",
        disposition=StopDisposition.STOPPED,
        correlation_id="opaque-whitespace",
    )

    assert StopRequest.from_dict(request.to_dict()) == request
    assert StopOutcome.from_dict(outcome.to_dict()) == outcome


@pytest.mark.asyncio
async def test_turn_address_includes_owner_when_ids_collide() -> None:
    calls: list[str] = []

    async def stop_a(_request: StopRequest) -> StopDisposition:
        calls.append("a")
        return StopDisposition.STOPPED

    async def stop_b(_request: StopRequest) -> StopDisposition:
        calls.append("b")
        return StopDisposition.STOPPED

    authority = _authority(
        lambda: [
            CooperativeStopTarget(
                "agent-a", "did:test:a", stop_a, frozenset({"same-turn"})
            ),
            CooperativeStopTarget(
                "agent-b", "did:test:b", stop_b, frozenset({"same-turn"})
            ),
        ]
    )

    outcomes = await authority.stop(
        StopRequest(
            StopScope.TURN,
            "did:test:operator",
            "same-turn",
            target_agent_id="did:test:b",
        )
    )

    assert calls == ["b"]
    assert outcomes[0].agent_id == "did:test:b"


def test_turn_and_tool_call_addresses_require_owning_agent() -> None:
    for scope in (StopScope.TURN, StopScope.TOOL_CALL):
        with pytest.raises(ValueError, match="owning agent identity"):
            StopRequest(scope, "did:test:operator", "colliding-id")


def test_stop_authority_has_no_process_lifecycle_dependency() -> None:
    source = inspect.getsource(inspect.getmodule(CancellationAuthority))
    assert "ProcessManager" not in source
    assert "terminate_all" not in source


def test_control_panel_uses_termination_wording_for_process_action() -> None:
    panel = Path("control-panel/index.html").read_text()
    function = panel.split("async function terminateAgent", 1)[1].split(
        "// Initial load",
        1,
    )[0]

    assert "Terminating..." in function
    assert "Terminate" in function
    assert "Stopping..." not in function
    assert "textContent = 'Stop'" not in function


def test_live_stop_endpoint_routes_request_through_typed_authority() -> None:
    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:live-agent"
    agent.cancel_current_request = MagicMock(return_value=True)
    app.state.agent = agent
    outcome = StopOutcome(
        scope=StopScope.TURN,
        requested_target="turn-live",
        resolved_target="did:test:live-agent",
        agent_id="did:test:live-agent",
        disposition=StopDisposition.STOPPED,
        correlation_id="authority-wiring",
    )

    with patch(
        "kestrel_sovereign.endpoints.agent.CancellationAuthority.stop",
        new=AsyncMock(return_value=(outcome,)),
    ) as authority_stop:
        response = TestClient(app).post(
            "/api/agent/stop",
            json={"request_id": "turn-live"},
        )

    assert response.status_code == 200
    assert response.json()["cancelled"] is True
    authority_stop.assert_awaited_once()
    routed_request = authority_stop.await_args.args[0]
    assert routed_request.scope is StopScope.TURN
    assert routed_request.target == "turn-live"
    assert routed_request.target_agent_id == "did:test:live-agent"
    agent.cancel_current_request.assert_not_called()


def test_stop_endpoint_preserves_whitespace_only_opaque_turn_id() -> None:
    """Every ID accepted by invoke/stream remains exactly addressable by Stop."""

    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:opaque-stop-id"
    agent._active_request_ids = {" "}
    app.state.agent = agent

    with patch(
        "kestrel_sovereign.endpoints.agent.CancellationAuthority.stop",
        new=AsyncMock(return_value=()),
    ) as authority_stop:
        response = TestClient(app).post(
            "/api/agent/stop",
            json={"request_id": " "},
        )

    assert response.status_code == 200
    routed_request = authority_stop.await_args.args[0]
    assert routed_request.scope is StopScope.TURN
    assert routed_request.target == " "


def test_stop_endpoint_resolves_public_turn_to_whitespace_request_id() -> None:
    """Public turn resolution preserves every accepted opaque request ID."""

    from kestrel_sovereign.agent.request_lifecycle import (
        RequestCompletionDisposition,
    )
    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:opaque-turn-binding"
    agent._active_request_ids = {" "}
    agent.active_turn_request_bindings = MagicMock(
        return_value={"turn-public": (" ", 1)}
    )
    agent.cancel_current_request = MagicMock(return_value=True)
    agent.wait_for_request_completion = AsyncMock(
        return_value=RequestCompletionDisposition.COMPLETED
    )
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"turn_id": "turn-public"},
    )

    assert response.status_code == 200
    assert response.json()["cancelled"] is True
    agent.cancel_current_request.assert_called_once_with(
        request_id=" ",
        generation=1,
    )


def test_stop_types_preserve_whitespace_only_opaque_turn_addresses() -> None:
    async def cancel(_request: StopRequest) -> StopDisposition:
        return StopDisposition.STOPPED

    request = StopRequest(
        scope=StopScope.TURN,
        actor_id="did:test:operator",
        target=" ",
        target_agent_id="did:test:opaque-agent",
    )
    target = CooperativeStopTarget(
        target_id="did:test:opaque-agent",
        agent_id="did:test:opaque-agent",
        cancel=cancel,
        turn_ids=frozenset({" "}),
    )
    outcome = StopOutcome(
        scope=StopScope.TURN,
        requested_target=" ",
        resolved_target=target.target_id,
        agent_id=target.agent_id,
        disposition=StopDisposition.STOPPED,
        correlation_id=request.correlation_id,
    )

    assert request.target in target.turn_ids
    assert StopOutcome.from_dict(outcome.to_dict()) == outcome


@pytest.mark.parametrize(
    "request_kwargs",
    (
        {"json": {"request_id": None}},
        {"json": {"request_id": ""}},
        {"json": {"request_id": 0}},
        {"json": {"request_id": False}},
        {"params": {"request_id": ""}},
    ),
)
def test_explicit_invalid_stop_id_is_rejected_without_widening_scope(
    request_kwargs,
) -> None:
    """A present invalid turn ID must never collapse into agent-wide Stop."""

    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:invalid-stop-id"
    agent.cancel_current_request = MagicMock(return_value=True)
    app.state.agent = agent

    response = TestClient(app).post("/api/agent/stop", **request_kwargs)

    assert response.status_code == 400
    assert "request_id must be a non-empty valid Unicode string" in (
        response.json()["detail"]
    )
    agent.cancel_current_request.assert_not_called()


def test_live_agent_stop_cancels_every_snapshotted_turn() -> None:
    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:live-agent"
    agent._active_request_ids = {"turn-b", "turn-a"}
    agent._current_request_id = "turn-b"
    agent.active_turn_request_ids = MagicMock(
        return_value={"observable-turn": "turn-a"}
    )
    agent.cancel_current_request = MagicMock(return_value=True)
    agent.wait_for_request_completion = AsyncMock(return_value=None)
    app.state.agent = agent

    response = TestClient(app).post("/api/agent/stop")

    assert response.status_code == 200
    assert response.json()["cancelled"] is True
    assert agent.cancel_current_request.call_args_list == [
        call(request_id="turn-a"),
        call(request_id="turn-b"),
    ]
    assert agent.wait_for_request_completion.await_args_list == [
        call("turn-a"),
        call("turn-b"),
    ]


def test_stop_before_registration_fences_the_late_request_generation() -> None:
    """An empty Stop snapshot must still prevent its in-transit turn starting."""

    from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
    from kestrel_sovereign.endpoints.agent import router

    class LiveAgent(RequestLifecycleMixin):
        agent_id = "did:test:registration-race"

        def __init__(self) -> None:
            self._current_request_id = None
            self._active_request_ids = set()
            self._active_request_counts = {}
            self._active_request_started_at = {}
            self._cancelled_requests = set()
            self._request_completion_events = {}

    agent = LiveAgent()
    app = FastAPI()
    app.include_router(router)
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"request_id": "in-transit-turn"},
    )

    assert response.status_code == 200
    assert response.json()["stop_outcomes"][0]["disposition"] == (
        "already_complete"
    )
    assert "in-transit-turn" in agent._pending_request_cancellations

    generation = agent.register_active_request("in-transit-turn")

    assert agent.is_request_cancelled("in-transit-turn") is True
    assert ("in-transit-turn", generation) in agent._cancelled_request_generations
    assert "in-transit-turn" in agent._pending_request_cancellations


def test_expired_pre_registration_stop_does_not_poison_a_future_reuse() -> None:
    from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin

    class LiveAgent(RequestLifecycleMixin):
        def __init__(self) -> None:
            self._current_request_id = None
            self._active_request_ids = set()
            self._active_request_counts = {}
            self._active_request_started_at = {}
            self._cancelled_requests = set()

    agent = LiveAgent()
    agent.reserve_request_cancellation("eventual-reuse")
    agent._pending_request_cancellations["eventual-reuse"] -= 31

    agent.register_active_request("eventual-reuse")

    assert agent.is_request_cancelled("eventual-reuse") is False


def test_live_stop_endpoint_accepts_turn_id_and_resolves_inside_authority() -> None:
    from kestrel_sovereign.agent.request_lifecycle import (
        RequestCompletionDisposition,
    )
    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:live-agent"
    agent._active_request_ids = {"request-private"}
    agent._current_request_id = "request-private"
    agent.active_turn_request_ids = MagicMock(
        return_value={"turn-visible": "request-private"}
    )
    agent.cancel_current_request = MagicMock(return_value=True)
    agent.wait_for_request_completion = AsyncMock(
        return_value=RequestCompletionDisposition.COMPLETED
    )
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"turn_id": "turn-visible"},
    )

    assert response.status_code == 200
    assert response.json()["turn_id"] == "turn-visible"
    assert response.json()["request_id"] is None
    outcome = response.json()["stop_outcomes"][0]
    assert outcome["requested_target"] == "turn-visible"
    assert outcome["resolved_target"] == "request-private"
    agent.cancel_current_request.assert_called_once_with(
        request_id="request-private"
    )
    agent.wait_for_request_completion.assert_awaited_once_with("request-private")


def test_turn_stop_cannot_cancel_a_reused_request_generation() -> None:
    """A stale turn snapshot must not widen onto a fresh same-ID delivery."""

    from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
    from kestrel_sovereign.endpoints.agent import router

    class LiveAgent(RequestLifecycleMixin):
        agent_id = "did:test:generation-race"

        def __init__(self) -> None:
            self._current_request_id = "reused-request"
            self._active_request_ids = {"reused-request"}
            self._active_request_counts = {"reused-request": 1}
            self._active_request_started_at = {
                "reused-request": time.monotonic()
            }
            self._active_request_generations = {"reused-request": 2}
            self._next_request_generation = 2
            self._cancelled_requests = set()
            self._cancelled_request_generations = set()
            self._request_completion_events = {}

        def active_turn_request_bindings(self):
            # Inventory captured turn generation 1 immediately before that
            # turn completed and generation 2 reused its request ID.
            return {"old-public-turn": ("reused-request", 1)}

    agent = LiveAgent()
    app = FastAPI()
    app.include_router(router)
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"turn_id": "old-public-turn"},
    )

    assert response.status_code == 200
    assert response.json()["stop_outcomes"][0]["disposition"] == (
        "already_complete"
    )
    assert ("reused-request", 2) not in agent._cancelled_request_generations
    assert "reused-request" not in getattr(
        agent, "_pending_request_cancellations", {}
    )


def test_live_stop_request_id_collision_does_not_resolve_as_turn_id() -> None:
    from kestrel_sovereign.agent.request_lifecycle import (
        RequestCompletionDisposition,
    )
    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:live-agent"
    agent._active_request_ids = {"collision", "another-request"}
    agent._current_request_id = "another-request"
    agent.active_turn_request_ids = MagicMock(
        return_value={"collision": "another-request"}
    )
    agent.cancel_current_request = MagicMock(return_value=True)
    agent.wait_for_request_completion = AsyncMock(
        return_value=RequestCompletionDisposition.COMPLETED
    )
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"request_id": "collision"},
    )

    assert response.status_code == 200
    agent.cancel_current_request.assert_called_once_with(request_id="collision")
    agent.wait_for_request_completion.assert_awaited_once_with("collision")


def test_unknown_turn_does_not_fall_back_to_agent_wide_stop() -> None:
    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:live-agent"
    agent._active_request_ids = {"unrelated-request"}
    agent._current_request_id = "unrelated-request"
    agent.active_turn_request_ids = MagicMock(return_value={})
    agent.cancel_current_request = MagicMock(return_value=True)
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"turn_id": "already-finished-turn"},
    )

    assert response.status_code == 503
    agent.cancel_current_request.assert_not_called()


def test_stop_endpoint_rejects_request_and_turn_id_together() -> None:
    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    app.state.agent = MagicMock(agent_id="did:test:live-agent")

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"request_id": "request-key", "turn_id": "turn-key"},
    )

    assert response.status_code == 400
    assert "either request_id or turn_id" in response.json()["detail"]


@pytest.mark.parametrize("request_id", ["", None, 0, False])
def test_stop_endpoint_rejects_explicit_falsey_request_id(request_id) -> None:
    """A malformed exact address cannot widen into agent-wide Stop."""

    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:falsey-stop-address"
    agent._active_request_ids = {"unrelated-live-turn"}
    agent.cancel_current_request = MagicMock(return_value=True)
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"request_id": request_id},
    )

    assert response.status_code == 400
    agent.cancel_current_request.assert_not_called()


def test_stop_endpoint_rejects_empty_query_request_id() -> None:
    """Query-field presence is preserved even when its value is empty."""

    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:falsey-query-stop-address"
    agent._active_request_ids = {"unrelated-live-turn"}
    agent.cancel_current_request = MagicMock(return_value=True)
    app.state.agent = agent

    response = TestClient(app).post("/api/agent/stop?request_id=")

    assert response.status_code == 400
    agent.cancel_current_request.assert_not_called()


@pytest.mark.asyncio
async def test_live_stop_reports_stopped_only_after_request_cleanup() -> None:
    """The endpoint must not confuse a cancel marker with completed execution."""
    from kestrel_sovereign.agent.request_lifecycle import (
        RequestCompletionDisposition,
        RequestLifecycleMixin,
    )
    from kestrel_sovereign.endpoints.agent import router

    cancel_seen = asyncio.Event()

    class LiveAgent(RequestLifecycleMixin):
        agent_id = "did:test:live-agent"

        def __init__(self) -> None:
            self._current_request_id = None
            self._active_request_ids = set()
            self._active_request_counts = {}
            self._active_request_started_at = {}
            self._cancelled_requests = set()
            self._request_completion_events = {}

        def cancel_current_request(self, request_id=None):
            cancelled = super().cancel_current_request(request_id)
            if cancelled:
                cancel_seen.set()
            return cancelled

    agent = LiveAgent()
    agent.register_active_request("turn-live")
    app = FastAPI()
    app.include_router(router)
    app.state.agent = agent

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        stop_task = asyncio.create_task(
            client.post("/api/agent/stop", json={"request_id": "turn-live"})
        )
        await asyncio.wait_for(cancel_seen.wait(), timeout=1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(stop_task), timeout=0.05)

        agent._cleanup_cancelled_request("turn-live")
        response = await asyncio.wait_for(stop_task, timeout=1)

    assert response.status_code == 200
    assert response.json()["stop_outcomes"][0]["disposition"] == "stopped"


@pytest.mark.asyncio
async def test_legacy_current_only_stop_waits_for_real_cleanup() -> None:
    """A synthesized generation remains live until its legacy owner exits."""

    from kestrel_sovereign.agent.request_lifecycle import (
        RequestCompletionDisposition,
        RequestLifecycleMixin,
    )

    class LegacyAgent(RequestLifecycleMixin):
        def __init__(self) -> None:
            self._current_request_id = "legacy-current"
            self._active_request_ids = set()
            self._active_request_counts = {}
            self._active_request_started_at = {}
            self._cancelled_requests = set()
            self._request_completion_events = {}

    agent = LegacyAgent()
    assert agent.cancel_current_request("legacy-current")
    waiter = asyncio.create_task(
        agent.wait_for_request_completion("legacy-current")
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.05)

    agent._cleanup_cancelled_request("legacy-current")
    assert await asyncio.wait_for(waiter, timeout=1) is RequestCompletionDisposition.COMPLETED


@pytest.mark.asyncio
async def test_live_stop_stale_prune_is_unreachable_not_stopped() -> None:
    """Age-only bookkeeping abandonment is never execution completion."""

    from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
    from kestrel_sovereign.endpoints.agent import router

    cancel_seen = asyncio.Event()

    class LiveAgent(RequestLifecycleMixin):
        agent_id = "did:test:long-running-agent"

        def __init__(self) -> None:
            self._current_request_id = None
            self._active_request_ids = set()
            self._active_request_counts = {}
            self._active_request_started_at = {}
            self._cancelled_requests = set()
            self._request_completion_events = {}

        def cancel_current_request(self, request_id=None):
            cancelled = super().cancel_current_request(request_id)
            if cancelled:
                cancel_seen.set()
            return cancelled

    agent = LiveAgent()
    agent.register_active_request("long-turn")
    agent._active_request_started_at["long-turn"] -= 1000
    app = FastAPI()
    app.include_router(router)
    app.state.agent = agent

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        stop_task = asyncio.create_task(
            client.post("/api/agent/stop", json={"request_id": "long-turn"})
        )
        await asyncio.wait_for(cancel_seen.wait(), timeout=1)
        assert agent.prune_stale_active_requests(900) == ["long-turn"]
        response = await asyncio.wait_for(stop_task, timeout=1)

        retry = await client.post(
            "/api/agent/stop",
            json={"request_id": "long-turn"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Cooperative Stop could not be confirmed."
    assert retry.status_code == 503
    assert retry.json()["detail"] == "Cooperative Stop could not be confirmed."
    assert "long-turn" in agent._cancelled_requests


def test_agent_wide_stop_includes_pruned_unconfirmed_turns() -> None:
    """Bodyless Stop must pull every abandoned generation's andon cord."""

    from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
    from kestrel_sovereign.endpoints.agent import router

    class LiveAgent(RequestLifecycleMixin):
        agent_id = "did:test:long-running-agent"

        def __init__(self) -> None:
            self._current_request_id = None
            self._active_request_ids = set()
            self._active_request_counts = {}
            self._active_request_started_at = {}
            self._cancelled_requests = set()
            self._request_completion_events = {}

    agent = LiveAgent()
    agent.register_active_request("pruned-turn")
    agent._active_request_started_at["pruned-turn"] -= 1000
    assert agent.prune_stale_active_requests(900) == ["pruned-turn"]

    app = FastAPI()
    app.include_router(router)
    app.state.agent = agent
    response = TestClient(app).post("/api/agent/stop")

    assert response.status_code == 503
    assert response.json()["detail"] == "Cooperative Stop could not be confirmed."
    assert "pruned-turn" in agent._cancelled_requests


@pytest.mark.asyncio
async def test_pruned_legacy_generation_is_reaped_by_eventual_cleanup() -> None:
    """A generation synthesized during prune retains its cleanup owner."""

    from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin

    class LegacyAgent(RequestLifecycleMixin):
        def __init__(self) -> None:
            self._current_request_id = "legacy-pruned"
            self._active_request_ids = {"legacy-pruned"}
            self._active_request_counts = {"legacy-pruned": 1}
            self._active_request_started_at = {
                "legacy-pruned": time.monotonic() - 1000
            }
            self._cancelled_requests = set()
            self._request_completion_events = {}

    agent = LegacyAgent()
    assert agent.cancel_current_request("legacy-pruned")
    generation = agent._active_request_generations["legacy-pruned"]
    assert agent.prune_stale_active_requests(900) == ["legacy-pruned"]
    assert ("legacy-pruned", generation) in agent._abandoned_request_counts

    agent._cleanup_cancelled_request("legacy-pruned")

    assert ("legacy-pruned", generation) not in agent._abandoned_request_counts
    assert generation not in agent._abandoned_request_generations.get(
        "legacy-pruned", set()
    )
    assert "legacy-pruned" not in agent._cancelled_requests


def test_live_stop_endpoint_reports_unreachable_as_http_failure() -> None:
    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:live-agent"
    agent._active_request_ids = {"turn-live"}
    agent.cancel_current_request = MagicMock(side_effect=RuntimeError("still running"))
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"request_id": "turn-live"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Cooperative Stop could not be confirmed."
