"""Host-scope durable Hold door (#3164).

Hold is state and Stop is an event, so this door is deliberately separate
from ``endpoints/host_stop``. What these tests pin is the part a console
cannot re-derive for itself: which identity a latch is keyed by, who may set
one, that a release names the receipt the caller saw, and that a store the
host cannot read is reported as unreadable rather than as "nothing is held".
"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from kestrel_sovereign.api_errors import register_api_error_handlers
from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.endpoints.hold import router
from kestrel_sovereign.hold import (
    HoldAction,
    HoldCorruptStateError,
    HoldDisposition,
    HoldIdempotencyConflict,
    HoldMutation,
    HoldReceipt,
    HoldScope,
    HoldState,
)

ALPHA = "did:test:alpha"
BETA = "did:test:beta"
WHEN = "2026-09-15T10:00:00+00:00"


def _latch(scope, target, receipt="receipt-1", reason="runaway", actor="sovereign-key"):
    return HoldState(
        scope=scope,
        target_id=target,
        reason=reason,
        actor_id=actor,
        set_at=WHEN,
        hold_receipt_id=receipt,
        revision=1,
    )


def _receipt(action, disposition, scope, target, **overrides):
    fields = {
        "receipt_id": "receipt-new",
        "operation_id": "op-1",
        "action": action,
        "disposition": disposition,
        "scope": scope,
        "target_id": target,
        "reason": "runaway",
        "actor_id": "sovereign-key",
        "occurred_at": WHEN,
        "expected_hold_receipt_id": "",
        "prior_hold_receipt_id": "",
        "resulting_hold_receipt_id": "receipt-new",
    }
    fields.update(overrides)
    return HoldReceipt(**fields)


class _Store:
    """A HoldStore stand-in that records the exact calls the door makes."""

    def __init__(self, *, latches=None, set_error=None, release_error=None):
        self.latches = dict(latches or {})
        self.set_error = set_error
        self.release_error = release_error
        self.set_calls = []
        self.release_calls = []
        self.read_error = None

    async def get_hold(self, scope, target_id=None):
        if self.read_error is not None:
            raise self.read_error
        key = (HoldScope(scope), target_id or "host")
        return self.latches.get(key)

    async def set_hold(self, *, scope, actor_id, reason, operation_id, target_id=None):
        self.set_calls.append(
            {
                "scope": scope,
                "target_id": target_id,
                "actor_id": actor_id,
                "reason": reason,
                "operation_id": operation_id,
            }
        )
        if self.set_error is not None:
            raise self.set_error
        latch = _latch(HoldScope(scope), target_id or "host", receipt="receipt-new", reason=reason)
        return HoldMutation(
            receipt=_receipt(
                HoldAction.HOLD,
                HoldDisposition.APPLIED,
                HoldScope(scope),
                target_id or "host",
                operation_id=operation_id,
                actor_id=actor_id,
                reason=reason,
            ),
            current=latch,
        )

    async def release_hold(
        self, *, scope, actor_id, reason, operation_id, expected_hold_receipt_id, target_id=None
    ):
        self.release_calls.append(
            {
                "scope": scope,
                "target_id": target_id,
                "actor_id": actor_id,
                "reason": reason,
                "operation_id": operation_id,
                "expected_hold_receipt_id": expected_hold_receipt_id,
            }
        )
        if self.release_error is not None:
            raise self.release_error
        current = self.latches.get((HoldScope(scope), target_id or "host"))
        stale = current is not None and current.hold_receipt_id != expected_hold_receipt_id
        return HoldMutation(
            receipt=_receipt(
                HoldAction.RELEASE,
                HoldDisposition.REFUSED_STALE if stale else HoldDisposition.APPLIED,
                HoldScope(scope),
                target_id or "host",
                operation_id=operation_id,
                actor_id=actor_id,
                reason=reason,
                expected_hold_receipt_id=expected_hold_receipt_id,
                prior_hold_receipt_id=current.hold_receipt_id if current else "",
                resulting_hold_receipt_id=(
                    current.hold_receipt_id if stale and current else ""
                ),
            ),
            current=current if stale else None,
        )


def _app(*, agents=((("Alpha"), ALPHA),), caller=None, store=None, host_context=True):
    app = FastAPI()
    # The production envelope, so a refusal's public `code` is asserted as
    # the console actually receives it.
    register_api_error_handlers(app)
    app.include_router(router)
    manager = MagicMock()
    manager.list_agents.return_value = {
        name: SimpleNamespace(did=did, agent_id="alias-never-used") for name, did in agents
    }
    app.state.agent_manager = manager
    store = store if store is not None else _Store()
    app.state.host_context = (
        SimpleNamespace(hold_store=store, backend_error="") if host_context else None
    )

    @app.middleware("http")
    async def bind_caller(request: Request, call_next):
        request.state.caller = caller
        return await call_next(request)

    return app, store


def _sovereign():
    return CallerContext.sovereign(identity="sovereign-key")


def _authenticated():
    return CallerContext.authenticated(identity="reader@example.com")


def test_read_composes_the_two_independent_latches_per_agent():
    store = _Store(
        latches={
            (HoldScope.HOST, "host"): _latch(HoldScope.HOST, "host", reason="fleet freeze"),
            (HoldScope.AGENT, ALPHA): _latch(HoldScope.AGENT, ALPHA, receipt="agent-7"),
        }
    )
    app, _ = _app(agents=(("Alpha", ALPHA), ("Beta", BETA)), caller=_sovereign(), store=store)

    payload = TestClient(app).get("/api/host/hold").json()

    assert payload["can_hold"] is True
    assert payload["host_hold"]["reason"] == "fleet freeze"
    alpha, beta = payload["agents"]
    assert alpha["agent_id"] == ALPHA
    assert alpha["sources"] == ["host", "agent"]
    assert alpha["agent_hold"]["hold_receipt_id"] == "agent-7"
    # Beta carries no agent latch of its own and is still held: the two latches
    # compose, they do not shadow one another.
    assert beta["agent_hold"] is None
    assert beta["held"] is True
    assert beta["sources"] == ["host"]


def test_read_reports_authority_without_refusing_the_view():
    app, _ = _app(caller=_authenticated())

    response = TestClient(app).get("/api/host/hold")

    assert response.status_code == 200
    assert response.json()["can_hold"] is False


def test_read_refuses_rather_than_reporting_an_unreadable_store_as_unheld():
    store = _Store()
    store.read_error = HoldCorruptStateError("receipt graph is broken")
    app, _ = _app(caller=_sovereign(), store=store)

    response = TestClient(app).get("/api/host/hold")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "hold_state_corrupt"


def test_hold_latches_the_agents_did_and_names_the_sovereign_actor():
    app, store = _app(caller=_sovereign())

    response = TestClient(app).post(
        "/api/host/hold",
        json={
            "scope": "agent",
            "target_id": ALPHA,
            "reason": "runaway loop",
            "operation_id": "op-1",
        },
    )

    assert response.status_code == 200, response.text
    assert store.set_calls == [
        {
            "scope": HoldScope.AGENT,
            # The DID the turn-start latch reads — never the `agent_id` alias
            # this fixture deliberately sets to something else.
            "target_id": ALPHA,
            "actor_id": "sovereign-key",
            "reason": "runaway loop",
            "operation_id": "op-1",
        }
    ]
    body = response.json()
    assert body["receipt"]["action"] == "hold"
    assert body["receipt"]["disposition"] == "applied"
    assert body["current"]["target_id"] == ALPHA


def test_hold_refuses_a_target_this_host_does_not_host():
    app, store = _app(caller=_sovereign())

    response = TestClient(app).post(
        "/api/host/hold",
        json={"scope": "agent", "target_id": BETA, "reason": "why", "operation_id": "op-1"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "hold_target_unknown"
    assert store.set_calls == [], "no latch is written for an identity nothing reads"


def test_authority_is_checked_before_the_target_so_a_refusal_is_not_a_probe():
    app, store = _app(caller=_authenticated())

    known = TestClient(app).post(
        "/api/host/hold",
        json={"scope": "agent", "target_id": ALPHA, "reason": "why", "operation_id": "op-1"},
    )
    unknown = TestClient(app).post(
        "/api/host/hold",
        json={"scope": "agent", "target_id": BETA, "reason": "why", "operation_id": "op-2"},
    )

    assert known.status_code == 403
    assert unknown.status_code == 403, "a hosted and an unhosted target refuse identically"
    assert known.json()["error"]["code"] == "sovereign_authority_required"
    assert store.set_calls == []


def test_release_carries_the_receipt_the_caller_saw_and_reports_a_stale_refusal():
    store = _Store(
        latches={(HoldScope.AGENT, ALPHA): _latch(HoldScope.AGENT, ALPHA, receipt="agent-9")}
    )
    app, _ = _app(caller=_sovereign(), store=store)

    response = TestClient(app).post(
        "/api/host/hold/release",
        json={
            "scope": "agent",
            "target_id": ALPHA,
            "reason": "audit finished",
            "operation_id": "op-2",
            "expected_hold_receipt_id": "agent-7",
        },
    )

    assert response.status_code == 200, response.text
    assert store.release_calls[0]["expected_hold_receipt_id"] == "agent-7"
    body = response.json()
    # A hold replaced since the caller last looked is refused, not released.
    assert body["receipt"]["disposition"] == "refused_stale"
    assert body["current"]["hold_receipt_id"] == "agent-9"


def test_release_requires_the_observed_receipt():
    app, store = _app(caller=_sovereign())

    response = TestClient(app).post(
        "/api/host/hold/release",
        json={"scope": "agent", "target_id": ALPHA, "reason": "done", "operation_id": "op-3"},
    )

    assert response.status_code == 422
    assert store.release_calls == []


def test_a_reused_operation_id_for_a_different_mutation_is_a_conflict():
    store = _Store(set_error=HoldIdempotencyConflict("operation id reused"))
    app, _ = _app(caller=_sovereign(), store=store)

    response = TestClient(app).post(
        "/api/host/hold",
        json={"scope": "agent", "target_id": ALPHA, "reason": "why", "operation_id": "op-1"},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "hold_operation_conflict"


def test_a_missing_hold_store_refuses_before_any_mutation():
    app, store = _app(caller=_sovereign(), host_context=False)

    read = TestClient(app).get("/api/host/hold")
    write = TestClient(app).post(
        "/api/host/hold",
        json={"scope": "agent", "target_id": ALPHA, "reason": "why", "operation_id": "op-1"},
    )

    assert read.status_code == 503
    assert write.status_code == 503
    assert read.json()["error"]["code"] == "hold_state_unavailable"
    assert store.set_calls == []


def test_host_scope_takes_no_caller_chosen_target():
    app, store = _app(caller=_sovereign())

    response = TestClient(app).post(
        "/api/host/hold",
        json={"scope": "host", "reason": "fleet freeze", "operation_id": "op-1"},
    )

    assert response.status_code == 200, response.text
    assert store.set_calls[0]["scope"] is HoldScope.HOST
    assert store.set_calls[0]["target_id"] is None
    assert response.json()["current"]["target_id"] == "host"


@pytest.mark.parametrize(
    "body",
    [
        {"scope": "agent", "target_id": ALPHA, "reason": "  ", "operation_id": "op"},
        {"scope": "agent", "target_id": ALPHA, "reason": "why", "operation_id": " "},
        {"scope": "agent", "target_id": " ", "reason": "why", "operation_id": "op"},
        {"scope": "turn", "target_id": ALPHA, "reason": "why", "operation_id": "op"},
        {"scope": "agent", "reason": "why", "operation_id": "op", "extra": 1},
    ],
    ids=["blank-reason", "blank-operation", "blank-target", "unknown-scope", "unknown-field"],
)
def test_an_unusable_mutation_never_reaches_the_store(body):
    app, store = _app(caller=_sovereign())

    response = TestClient(app).post("/api/host/hold", json=body)

    assert response.status_code == 422
    assert store.set_calls == []


def test_an_agent_scope_hold_requires_a_target():
    app, store = _app(caller=_sovereign())

    response = TestClient(app).post(
        "/api/host/hold",
        json={"scope": "agent", "reason": "why", "operation_id": "op-1"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "hold_target_required"
    assert store.set_calls == []


def _mutations(client):
    """Every door that reaches the store, so one failure class is asserted at all three."""

    return (
        ("read", client.get("/api/host/hold")),
        (
            "hold",
            client.post(
                "/api/host/hold",
                json={
                    "scope": "agent",
                    "target_id": ALPHA,
                    "reason": "why",
                    "operation_id": "op-1",
                },
            ),
        ),
        (
            "release",
            client.post(
                "/api/host/hold/release",
                json={
                    "scope": "agent",
                    "target_id": ALPHA,
                    "reason": "why",
                    "operation_id": "op-2",
                    "expected_hold_receipt_id": "agent-7",
                },
            ),
        ),
    )


def test_an_unexpected_storage_failure_is_never_reported_as_a_caller_mistake():
    # A driver fault or a programming error is not a bad request, and its text
    # is not the caller's business: it names the store's internals. Answering
    # 400 with that text on the wire says the operator sent something wrong,
    # and tells them what a retriable server failure looked like from inside.
    boom = sqlite3.OperationalError("no such column: hold_actor_secret")
    store = _Store(set_error=boom, release_error=boom)
    store.read_error = boom
    app, _ = _app(caller=_sovereign(), store=store)
    client = TestClient(app, raise_server_exceptions=False)

    for door, response in _mutations(client):
        assert response.status_code >= 500, f"{door} misreported a server fault as a 4xx"
        assert response.json()["error"]["code"] == "internal_error", door
        assert "hold_actor_secret" not in response.text, f"{door} leaked the store's text"
        assert "OperationalError" not in response.text, door


def test_a_programming_error_in_the_store_is_not_a_bad_request_either():
    boom = AttributeError("'NoneType' object has no attribute 'execute'")
    store = _Store(set_error=boom, release_error=boom)
    store.read_error = boom
    app, _ = _app(caller=_sovereign(), store=store)
    client = TestClient(app, raise_server_exceptions=False)

    for door, response in _mutations(client):
        assert response.status_code >= 500, door
        assert "NoneType" not in response.text, f"{door} leaked the raw exception text"


def test_an_unexpected_value_error_in_the_store_is_not_a_bad_request_either():
    # The hole the two tests above would otherwise leave open. `ValueError` is
    # what the store raises to refuse a caller argument, so it is the one
    # server fault that looks like a caller mistake from the outside. This door
    # answers every caller-chosen argument itself, so a `ValueError` arriving
    # from the store means a bug here or below -- never a bad request.
    boom = ValueError("hold latch row shape changed under the reader")
    store = _Store(set_error=boom, release_error=boom)
    store.read_error = boom
    app, _ = _app(caller=_sovereign(), store=store)
    client = TestClient(app, raise_server_exceptions=False)

    for door, response in _mutations(client):
        assert response.status_code >= 500, f"{door} misreported a server fault as a 4xx"
        assert response.json()["error"]["code"] == "internal_error", door
        assert "row shape changed" not in response.text, f"{door} leaked the raw text"


def test_the_host_scope_refuses_a_caller_chosen_target_before_the_store():
    # The other end of the same boundary: this refusal is real, and answering
    # it HERE is what lets the store's own `ValueError` be read as a fault
    # above. A target the store would reject must never reach it.
    app, store = _app(caller=_sovereign())

    response = TestClient(app).post(
        "/api/host/hold",
        json={
            "scope": "host",
            "target_id": ALPHA,
            "reason": "fleet freeze",
            "operation_id": "op-1",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "hold_request_invalid"
    assert store.set_calls == [], "a target the store would refuse still reached it"
    # Our own words, not the store's internals echoed back.
    assert "host control store" not in response.json()["error"]["message"]


def test_the_host_scope_release_refuses_a_caller_chosen_target_too():
    app, store = _app(caller=_sovereign())

    response = TestClient(app).post(
        "/api/host/hold/release",
        json={
            "scope": "host",
            "target_id": ALPHA,
            "reason": "thaw",
            "operation_id": "op-1",
            "expected_hold_receipt_id": "receipt-1",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "hold_request_invalid"
    assert store.release_calls == []


def test_the_host_scope_accepts_its_own_fixed_target_name():
    # `host` is the store's own target for this scope, so naming it explicitly
    # is not a foreign target -- the refusal above must not swallow it.
    app, store = _app(caller=_sovereign())

    response = TestClient(app).post(
        "/api/host/hold",
        json={
            "scope": "host",
            "target_id": "host",
            "reason": "fleet freeze",
            "operation_id": "op-1",
        },
    )

    assert response.status_code == 200, response.text
    assert store.set_calls[0]["target_id"] == "host"


def test_an_agent_without_a_resolvable_identity_fails_the_inventory_closed():
    app, _ = _app(agents=(), caller=_sovereign())
    app.state.agent_manager.list_agents.return_value = {"Ghost": SimpleNamespace(did=None)}

    response = TestClient(app).get("/api/host/hold")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "hold_inventory_unavailable"
