"""The two sovereign receipt-history doors (#3159 R2/R3/R5/R6).

The receipts were complete and unreadable: the only reads were an exact replay,
a retry, and one admission probe, and ``GET /api/host/hold`` returned the
active latches with no history at all. An observability view cannot render a
stop as a stop, or explain a held agent at rest, from evidence nothing exposes.

What these pin is the part a consumer cannot re-derive: who may read the rows,
that a page is bounded and its order total, that an unreadable store is
reported as unreadable rather than as an empty history, that an agent-scope
receipt now names its agent while a pre-fix row still reads as not-recorded,
and that a blinded caller address is never rendered as an identity.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from kestrel_sovereign.api_errors import register_api_error_handlers
from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.endpoints.hold import router as hold_router
from kestrel_sovereign.endpoints.host_stop import router as stop_router
from kestrel_sovereign.endpoints.receipt_feed import (
    MAX_RECEIPT_PAGE_SIZE,
    RECEIPT_FEED_SCHEMA_VERSION,
    disclosable_identity,
)
from kestrel_sovereign.hold import HoldScope, HoldStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.stop import (
    StopDisposition,
    StopOutcome,
    StopReceiptError,
    StopReceiptStore,
    StopRequest,
    StopScope,
)
from kestrel_sovereign.stop.receipt import opaque_stop_identifier

ALPHA = "did:test:alpha"
BETA = "did:test:beta"
OPERATOR = "did:pkh:eip155:1:0xSOVEREIGN"
TRACE = "0123456789abcdef0123456789abcdef"


# --------------------------------------------------------------------------
# Stores
# --------------------------------------------------------------------------


async def _stop_store(db_backend):
    store = StopReceiptStore(AsyncDatabase(db_backend))
    await store.ensure_schema()
    return store


def _agent_stop(target, *, correlation_id, reason="runaway loop", trace_id=None):
    return StopRequest(
        scope=StopScope.AGENT,
        actor_id=OPERATOR,
        target=target,
        reason=reason,
        cascade=False,
        correlation_id=correlation_id,
        trace_id=trace_id,
    )


def _outcomes(request, agent_id=None, disposition=StopDisposition.STOPPED):
    return (
        StopOutcome(
            scope=request.scope,
            requested_target=request.target,
            resolved_target=request.target,
            agent_id=agent_id or request.target,
            disposition=disposition,
            correlation_id=request.correlation_id,
            detail=None,
        ),
    )


async def _record_agent_stop(store, target, correlation_id, **kwargs):
    request = _agent_stop(target, correlation_id=correlation_id, **kwargs)
    await store.persist(request, _outcomes(request))
    return request


# --------------------------------------------------------------------------
# Apps
# --------------------------------------------------------------------------


def _app(router, caller, **state):
    app = FastAPI()
    register_api_error_handlers(app)
    app.include_router(router)
    manager = MagicMock()
    manager.list_agents.return_value = {
        "Alpha": SimpleNamespace(did=ALPHA, agent_id=ALPHA),
    }
    app.state.agent_manager = manager
    for key, value in state.items():
        setattr(app.state, key, value)

    @app.middleware("http")
    async def bind_caller(request: Request, call_next):
        request.state.caller = caller
        return await call_next(request)

    return TestClient(app)


def _sovereign():
    return CallerContext.sovereign(identity=OPERATOR)


def _authenticated():
    return CallerContext.authenticated(identity="reader@example.com")


class _RaisingStopStore:
    def __init__(self, error):
        self._error = error

    async def list_receipts(self, **_kwargs):
        raise self._error


class _RaisingHoldStore:
    def __init__(self, error):
        self._error = error

    async def list_receipts(self, **_kwargs):
        raise self._error

    async def read_boot_state(self):
        return ()


# --------------------------------------------------------------------------
# Authority
# --------------------------------------------------------------------------


class TestOnlyTheSovereignReadsReceipts:
    """R2: the same gate as POST /api/host/stop.

    These rows carry an operator's written reason for an act performed ON
    another agent, and the identity of the operator who performed it.
    """

    @pytest.mark.parametrize(
        "router,path",
        [
            (stop_router, "/api/host/stop/receipts"),
            (hold_router, "/api/host/hold/receipts"),
        ],
    )
    def test_an_authenticated_non_sovereign_is_refused(self, router, path):
        client = _app(
            router,
            _authenticated(),
            stop_receipt_store=_RaisingStopStore(AssertionError("never read")),
            host_context=SimpleNamespace(
                hold_store=_RaisingHoldStore(AssertionError("never read")),
                backend_error="",
            ),
        )

        response = client.get(path)

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "sovereign_authority_required"

    @pytest.mark.parametrize(
        "router,path",
        [
            (stop_router, "/api/host/stop/receipts"),
            (hold_router, "/api/host/hold/receipts"),
        ],
    )
    def test_the_gate_precedes_every_filter(self, router, path):
        """A refusal must not become a probe for which agents exist."""

        client = _app(
            router,
            _authenticated(),
            stop_receipt_store=_RaisingStopStore(AssertionError("never read")),
            host_context=SimpleNamespace(
                hold_store=_RaisingHoldStore(AssertionError("never read")),
                backend_error="",
            ),
        )

        # A malformed window would be a 400 if the filters were read first.
        response = client.get(path, params={"since": "not-a-timestamp"})

        assert response.status_code == 403


# --------------------------------------------------------------------------
# Failure is reported as failure
# --------------------------------------------------------------------------


class TestAnUnreadableHistoryIsNotAnEmptyHistory:
    def test_stop_store_failure_is_503(self):
        client = _app(
            stop_router,
            _sovereign(),
            stop_receipt_store=_RaisingStopStore(
                StopReceiptError("evidence store is closed")
            ),
        )

        response = client.get("/api/host/stop/receipts")

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "stop_receipts_unavailable"

    def test_a_missing_stop_store_is_503_not_an_empty_list(self):
        """No store at all is the same claim: nobody can read the history."""

        client = _app(stop_router, _sovereign(), stop_receipt_store=None)

        response = client.get("/api/host/stop/receipts")

        assert response.status_code == 503

    def test_hold_store_failure_is_503(self):
        from kestrel_sovereign.hold import HoldStateError

        client = _app(
            hold_router,
            _sovereign(),
            host_context=SimpleNamespace(
                hold_store=_RaisingHoldStore(HoldStateError("unreadable")),
                backend_error="",
            ),
        )

        response = client.get("/api/host/hold/receipts")

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "hold_state_unavailable"

    def test_hold_corruption_is_reported_as_corruption(self):
        from kestrel_sovereign.hold import HoldCorruptStateError

        client = _app(
            hold_router,
            _sovereign(),
            host_context=SimpleNamespace(
                hold_store=_RaisingHoldStore(
                    HoldCorruptStateError("history graph is broken")
                ),
                backend_error="",
            ),
        )

        response = client.get("/api/host/hold/receipts")

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "hold_state_corrupt"


# --------------------------------------------------------------------------
# Stop evidence over a real store, on both backends
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_agent_scope_receipt_names_the_agent_it_stopped(db_backend):
    """R3. The header answers "which agent", which no column used to.

    ``requested_target`` is a one-way digest — right for the caller-supplied
    request ids and turn handles it was built for, useless for reading a
    receipt back — so an agent-scope receipt could not say who was stopped.
    """

    store = await _stop_store(db_backend)
    request = await _record_agent_stop(store, ALPHA, "op-alpha")

    page = await store.list_receipts(limit=10)

    assert len(page.receipts) == 1
    record = page.receipts[0]
    assert record.target_agent_id == ALPHA
    assert record.scope == "agent"
    assert record.actor_id == OPERATOR
    assert record.reason == "runaway loop"
    assert record.cascade is False
    assert record.occurred_at
    # The blinded address is still blinded; the DID did not leak through it.
    assert disclosable_identity(
        opaque_stop_identifier("target", request.target)
    ) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_pre_fix_row_reads_back_as_not_recorded(db_backend):
    """R3. A row written before the fix recorded no agent, and says so.

    It is emphatically NOT recovered by re-hashing the live inventory: that
    would guess an identity the receipt never held.
    """

    store = await _stop_store(db_backend)
    await _record_agent_stop(store, ALPHA, "op-legacy")
    # Exactly what the pre-fix writer left behind.
    await store._db.execute(
        "UPDATE stop_receipts SET target_agent_id = NULL WHERE scope = 'agent'"
    )

    page = await store.list_receipts(limit=10)

    assert page.receipts[0].target_agent_id is None
    # ...and the outcome's blinded agent identity is not offered as a fallback.
    assert page.receipts[0].outcomes[0].agent_id is not None
    assert disclosable_identity(page.receipts[0].outcomes[0].agent_id) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_pages_are_bounded_and_totally_ordered(db_backend):
    """R6. Convergence is a property of the read model.

    Immutable rows ordered by ``(occurred_at, receipt_id)`` mean any walk over
    the same window yields the same set in the same order.
    """

    store = await _stop_store(db_backend)
    for index in range(5):
        await _record_agent_stop(store, ALPHA, f"op-{index}")

    first = await store.list_receipts(limit=2)
    assert len(first.receipts) == 2
    assert first.next_key is not None

    second = await store.list_receipts(after=first.next_key, limit=2)
    third = await store.list_receipts(after=second.next_key, limit=2)

    walked = [
        record.receipt_id
        for page in (first, second, third)
        for record in page.receipts
    ]
    whole = await store.list_receipts(limit=50)

    assert walked == [record.receipt_id for record in whole.receipts]
    assert len(walked) == 5
    assert len(set(walked)) == 5, "a page boundary repeated a receipt"
    # Exhausted: the last page issues no cursor.
    assert third.next_key is None
    # And the order is the one the route promises.
    ordering = [(r.occurred_at, r.receipt_id) for r in whole.receipts]
    assert ordering == sorted(ordering)


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_filters_select_by_agent_and_trace(db_backend):
    store = await _stop_store(db_backend)
    await _record_agent_stop(store, ALPHA, "op-a", trace_id=TRACE)
    await _record_agent_stop(store, BETA, "op-b")

    by_agent = await store.list_receipts(agent_id=BETA, limit=10)
    by_trace = await store.list_receipts(trace_id=TRACE, limit=10)

    assert [r.target_agent_id for r in by_agent.receipts] == [BETA]
    assert [r.target_agent_id for r in by_trace.receipts] == [ALPHA]
    assert by_trace.receipts[0].trace_id == TRACE


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_host_fanout_is_findable_by_the_agent_it_reached(db_backend):
    """A host-scope receipt names no single agent in its header.

    Its per-target outcomes do, and those are stored in the clear because a
    host Stop carries no caller-supplied address to blind.
    """

    store = await _stop_store(db_backend)
    request = StopRequest(
        scope=StopScope.HOST,
        actor_id=OPERATOR,
        reason="fleet freeze",
        correlation_id="op-host",
    )
    await store.persist(
        request,
        tuple(
            StopOutcome(
                scope=StopScope.HOST,
                requested_target=None,
                resolved_target=agent,
                agent_id=agent,
                disposition=StopDisposition.STOPPED,
                correlation_id=request.correlation_id,
            )
            for agent in (ALPHA, BETA)
        ),
    )

    page = await store.list_receipts(agent_id=BETA, limit=10)

    assert len(page.receipts) == 1
    record = page.receipts[0]
    assert record.scope == "host"
    assert record.target_agent_id is None
    assert [outcome.agent_id for outcome in record.outcomes] == [ALPHA, BETA]
    assert [outcome.ordinal for outcome in record.outcomes] == [0, 1]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_unreachable_outcomes_are_returned_not_filtered(db_backend):
    """A target Stop could not reach is evidence, not noise."""

    store = await _stop_store(db_backend)
    request = _agent_stop(ALPHA, correlation_id="op-unreachable")
    await store.persist(
        request, _outcomes(request, disposition=StopDisposition.UNREACHABLE)
    )

    page = await store.list_receipts(limit=10)

    assert [o.disposition for o in page.receipts[0].outcomes] == ["unreachable"]


@pytest.mark.asyncio
async def test_the_page_size_is_a_bound_the_store_enforces(tmp_path):
    from kestrel_sovereign.storage.db import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "stop.db"))
    await backend.connect()
    try:
        store = await _stop_store(backend)
        with pytest.raises(ValueError):
            await store.list_receipts(limit=0)
    finally:
        await backend.close()


# --------------------------------------------------------------------------
# Hold history over a real store
# --------------------------------------------------------------------------


@pytest.fixture
async def hold_store(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
    store = HoldStore(db)
    await store.ensure_schema()
    try:
        yield store
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_resume_is_its_own_receipt_and_does_not_erase_the_hold(
    hold_store,
):
    """R5. Append-only, and the ONLY record a released hold leaves.

    The latch row is blanked on release, so without this history a hold that
    was resumed survives nowhere — an agent held on Tuesday and resumed on
    Friday would have no explanation on either day.
    """

    held = await hold_store.set_hold(
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="runaway loop",
        operation_id="op-hold",
    )
    released = await hold_store.release_hold(
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="fixed",
        operation_id="op-release",
        expected_hold_receipt_id=held.receipt.receipt_id,
    )

    page = await hold_store.list_receipts(limit=10)

    assert [r.action.value for r in page.receipts] == ["hold", "release"]
    # The hold event is untouched by the resume...
    assert page.receipts[0].receipt_id == held.receipt.receipt_id
    assert page.receipts[0].reason == "runaway loop"
    # ...and the resume points back at exactly the hold it ended.
    assert page.receipts[1].receipt_id == released.receipt.receipt_id
    assert page.receipts[1].prior_hold_receipt_id == held.receipt.receipt_id
    assert page.receipts[1].disposition.value == "applied"
    # The live latch is gone; only the history explains the agent.
    assert await hold_store.get_hold(HoldScope.AGENT, ALPHA) is None


@pytest.mark.asyncio
async def test_refused_and_already_in_state_are_real_operator_acts(hold_store):
    """R5. They are returned, not filtered: somebody really did ask."""

    held = await hold_store.set_hold(
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="runaway loop",
        operation_id="op-hold",
    )
    # Same actor, same reason: the latch is already exactly what was asked
    # for, so the act is recorded as already-in-state rather than re-applied.
    await hold_store.set_hold(
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="runaway loop",
        operation_id="op-hold-again",
    )
    await hold_store.release_hold(
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="stale",
        operation_id="op-stale",
        expected_hold_receipt_id=f"{held.receipt.receipt_id}-not-current",
    )

    page = await hold_store.list_receipts(limit=10)

    assert [r.disposition.value for r in page.receipts] == [
        "applied",
        "already_in_state",
        "refused_stale",
    ]


@pytest.mark.asyncio
async def test_hold_history_pages_in_a_total_order(hold_store):
    for index in range(4):
        await hold_store.set_hold(
            scope=HoldScope.AGENT,
            target_id=f"{ALPHA}-{index}",
            actor_id=OPERATOR,
            reason="freeze",
            operation_id=f"op-{index}",
        )

    first = await hold_store.list_receipts(limit=2)
    second = await hold_store.list_receipts(after=first.next_key, limit=2)
    whole = await hold_store.list_receipts(limit=10)

    walked = [
        r.receipt_id for page in (first, second) for r in page.receipts
    ]
    assert walked == [r.receipt_id for r in whole.receipts]
    assert len(set(walked)) == 4
    assert second.next_key is None


@pytest.mark.asyncio
async def test_hold_history_filters_by_scope_and_target(hold_store):
    await hold_store.set_hold(
        scope=HoldScope.HOST,
        actor_id=OPERATOR,
        reason="fleet freeze",
        operation_id="op-host",
    )
    await hold_store.set_hold(
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="runaway",
        operation_id="op-alpha",
    )

    host_only = await hold_store.list_receipts(scope=HoldScope.HOST, limit=10)
    alpha_only = await hold_store.list_receipts(
        scope=HoldScope.AGENT, target_id=ALPHA, limit=10
    )

    assert [r.reason for r in host_only.receipts] == ["fleet freeze"]
    assert [r.target_id for r in alpha_only.receipts] == [ALPHA]


# --------------------------------------------------------------------------
# Wire shape
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_stop_route_versions_and_pages_its_payload(tmp_path):
    from kestrel_sovereign.storage.db import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "stop.db"))
    await backend.connect()
    try:
        store = await _stop_store(backend)
        for index in range(3):
            await _record_agent_stop(store, ALPHA, f"op-{index}")

        client = _app(stop_router, _sovereign(), stop_receipt_store=store)
        first = client.get(
            "/api/host/stop/receipts", params={"limit": 2}
        ).json()
        second = client.get(
            "/api/host/stop/receipts",
            params={"limit": 2, "cursor": first["next_cursor"]},
        ).json()
    finally:
        await backend.close()

    assert first["schema_version"] == RECEIPT_FEED_SCHEMA_VERSION
    assert len(first["receipts"]) == 2
    assert first["next_cursor"]
    assert len(second["receipts"]) == 1
    assert second["next_cursor"] is None

    record = first["receipts"][0]
    assert record["target_agent_id"] == ALPHA
    assert record["actor_id"] == OPERATOR
    assert record["reason"] == "runaway loop"
    assert record["cascade"] is False
    assert record["scope"] == "agent"
    assert record["occurred_at"]
    assert record["outcomes"][0]["disposition"] == "stopped"
    # The caller-supplied address was blinded; it is reported absent, never
    # rendered as though it were an identity.
    assert record["outcomes"][0]["agent_id"] is None
    assert record["outcomes"][0]["resolved_target"] is None


@pytest.mark.asyncio
async def test_the_hold_route_versions_its_payload(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
    try:
        store = HoldStore(db)
        await store.ensure_schema()
        held = await store.set_hold(
            scope=HoldScope.AGENT,
            target_id=ALPHA,
            actor_id=OPERATOR,
            reason="runaway loop",
            operation_id="op-hold",
        )
        await store.release_hold(
            scope=HoldScope.AGENT,
            target_id=ALPHA,
            actor_id=OPERATOR,
            reason="resumed",
            operation_id="op-release",
            expected_hold_receipt_id=held.receipt.receipt_id,
        )

        client = _app(
            hold_router,
            _sovereign(),
            host_context=SimpleNamespace(hold_store=store, backend_error=""),
        )
        payload = client.get(
            "/api/host/hold/receipts", params={"agent_id": ALPHA}
        ).json()
    finally:
        await db.close()

    assert payload["schema_version"] == RECEIPT_FEED_SCHEMA_VERSION
    assert [r["action"] for r in payload["receipts"]] == ["hold", "release"]
    assert payload["receipts"][1]["prior_hold_receipt_id"] == (
        payload["receipts"][0]["receipt_id"]
    )
    assert payload["receipts"][0]["actor_id"] == OPERATOR
    assert payload["next_cursor"] is None


class TestTheRouteRefusesNonsenseBeforeReadingAnything:
    def test_a_backwards_window_is_a_400(self):
        client = _app(
            stop_router,
            _sovereign(),
            stop_receipt_store=_RaisingStopStore(
                AssertionError("must not be read")
            ),
        )

        response = client.get(
            "/api/host/stop/receipts",
            params={
                "since": "2026-09-22T10:00:00Z",
                "until": "2026-09-22T09:00:00Z",
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "receipt_window_invalid"

    def test_a_forged_cursor_is_a_400(self):
        client = _app(
            stop_router,
            _sovereign(),
            stop_receipt_store=_RaisingStopStore(
                AssertionError("must not be read")
            ),
        )

        response = client.get(
            "/api/host/stop/receipts", params={"cursor": "not-a-cursor"}
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "receipt_cursor_invalid"

    def test_a_page_past_the_hard_cap_is_refused(self):
        """A read route must never be askable for the whole history."""

        client = _app(
            stop_router,
            _sovereign(),
            stop_receipt_store=_RaisingStopStore(
                AssertionError("must not be read")
            ),
        )

        response = client.get(
            "/api/host/stop/receipts",
            params={"limit": MAX_RECEIPT_PAGE_SIZE + 1},
        )

        assert response.status_code == 422
