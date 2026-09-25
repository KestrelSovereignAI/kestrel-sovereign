"""The two sovereign receipt-history doors (#3159 R2/R3/R5/R6).

The receipts were complete and unreadable: the only reads were an exact replay,
a retry, and one admission probe, and ``GET /api/host/hold`` returned the
active latches with no history at all. An observability view cannot render a
stop as a stop, or explain a held agent at rest, from evidence nothing exposes.

What these pin is the part a consumer cannot re-derive: who may read the rows,
that a page is bounded and its commit order total, that an unreadable store
is reported as unreadable rather than as an empty history, that an agent-scope
receipt names the agent its address RESOLVED to while a pre-fix row still
reads as not-recorded, and that a blinded caller address is never rendered as
an identity.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

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
    resolve_page_size,
)
from kestrel_sovereign.hold import HoldAuthority, HoldScope, HoldStore
from kestrel_sovereign.stop import (
    CancellationAuthority,
    CooperativeStopTarget,
    StopCleanupRegistry,
    StopDisposition,
    StopOutcome,
    StopReceiptError,
    StopReceiptStore,
    StopRequest,
    StopScope,
)
from kestrel_sovereign.stop.receipt import opaque_stop_identifier
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.database_clock import (
    LATEST_TIMESTAMP_BOUND,
    TimestampBoundOutOfRange,
    database_now_sql,
    database_timestamp_bound_text,
)

ALPHA = "did:test:alpha"
BETA = "did:test:beta"
OPERATOR = "did:pkh:eip155:1:0xSOVEREIGN"


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


def _fresh(prefix):
    """A per-test identity: the PostgreSQL leg shares one database across tests,
    so every dual-backend test scopes its reads to rows only it wrote."""

    return f"{prefix}{uuid4().hex}"


def _target(name, agent_id):
    async def cancel(_request):
        return StopDisposition.STOPPED

    return CooperativeStopTarget(target_id=name, agent_id=agent_id, cancel=cancel)


def _authority(store, *targets):
    return CancellationAuthority(
        lambda: targets,
        cleanup_registry=StopCleanupRegistry(),
        receipt_store=store,
    )


async def _record_agent_stop(
    store, target, correlation_id, *, agent_id=None, **kwargs
):
    """Stop ``target`` through the real authority, as every door does.

    ``target`` is the address the caller used; ``agent_id`` (default: the
    target itself) is the DID the inventory resolves it to.
    """

    request = _agent_stop(target, correlation_id=correlation_id, **kwargs)
    outcomes = await _authority(
        store, _target(target, agent_id or target)
    ).stop(request)
    assert [o.disposition for o in outcomes] == [StopDisposition.STOPPED]
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

    @pytest.mark.parametrize(
        "router,path,params",
        [
            # Every value a constrained parameter declaration would reject.
            # FastAPI validates declared parameters BEFORE the handler runs,
            # so each of these answered 422 to a caller the gate refuses.
            (stop_router, "/api/host/stop/receipts", {"limit": "201"}),
            (stop_router, "/api/host/stop/receipts", {"limit": "0"}),
            (stop_router, "/api/host/stop/receipts", {"limit": "many"}),
            (stop_router, "/api/host/stop/receipts", {"cursor": ""}),
            (stop_router, "/api/host/stop/receipts", {"cursor": "x" * 600}),
            (stop_router, "/api/host/stop/receipts", {"agent_id": ""}),
            (stop_router, "/api/host/stop/receipts", {"agent_id": "d" * 600}),
            (stop_router, "/api/host/stop/receipts", {"trace_id": ""}),
            (stop_router, "/api/host/stop/receipts", {"trace_id": "t" * 65}),
            (stop_router, "/api/host/stop/receipts", {"since": "s" * 65}),
            (hold_router, "/api/host/hold/receipts", {"limit": "201"}),
            (hold_router, "/api/host/hold/receipts", {"limit": "-1"}),
            (hold_router, "/api/host/hold/receipts", {"scope": "turn"}),
            (hold_router, "/api/host/hold/receipts", {"cursor": ""}),
            (hold_router, "/api/host/hold/receipts", {"agent_id": ""}),
            (hold_router, "/api/host/hold/receipts", {"agent_id": "d" * 600}),
            (hold_router, "/api/host/hold/receipts", {"until": "u" * 65}),
        ],
    )
    def test_the_gate_precedes_parameter_validation(
        self, router, path, params
    ):
        client = _app(
            router,
            _authenticated(),
            stop_receipt_store=_RaisingStopStore(AssertionError("never read")),
            host_context=SimpleNamespace(
                hold_store=_RaisingHoldStore(AssertionError("never read")),
                backend_error="",
            ),
        )

        response = client.get(path, params=params)

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "sovereign_authority_required"

    @pytest.mark.parametrize(
        "router,path,params",
        [
            (stop_router, "/api/host/stop/receipts", {"limit": "201"}),
            (stop_router, "/api/host/stop/receipts", {"limit": "many"}),
            (stop_router, "/api/host/stop/receipts", {"agent_id": ""}),
            (stop_router, "/api/host/stop/receipts", {"trace_id": "t" * 33}),
            (stop_router, "/api/host/stop/receipts", {"since": "s" * 65}),
            (hold_router, "/api/host/hold/receipts", {"scope": "turn"}),
            (hold_router, "/api/host/hold/receipts", {"limit": "0"}),
            (hold_router, "/api/host/hold/receipts", {"agent_id": "d" * 600}),
        ],
    )
    def test_the_sovereign_still_gets_a_400_for_a_malformed_filter(
        self, router, path, params
    ):
        """Moving validation behind the gate must not delete it."""

        client = _app(
            router,
            _sovereign(),
            stop_receipt_store=_RaisingStopStore(AssertionError("never read")),
            host_context=SimpleNamespace(
                hold_store=_RaisingHoldStore(AssertionError("never read")),
                backend_error="",
            ),
        )

        response = client.get(path, params=params)

        assert response.status_code == 400


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


@pytest.mark.asyncio
async def test_a_closed_real_stop_backend_is_503_not_500(tmp_path):
    """A real backend raises SDK QueryError/ConnectionError, not a domain error.

    The raising double above cannot show that the STORE translates it: this
    closes a real SQLite backend under a real store and goes through the route.
    """

    from kestrel_sovereign.storage.db import SQLiteBackend

    backend = SQLiteBackend(str(tmp_path / "stop.db"))
    await backend.connect()
    store = await _stop_store(backend)
    await backend.close()

    client = _app(stop_router, _sovereign(), stop_receipt_store=store)
    response = client.get("/api/host/stop/receipts")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "stop_receipts_unavailable"


@pytest.mark.asyncio
async def test_a_closed_real_hold_backend_is_503_not_500(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
    store = HoldStore(db)
    await store.ensure_schema()
    await db.close()

    client = _app(
        hold_router,
        _sovereign(),
        host_context=SimpleNamespace(hold_store=store, backend_error=""),
    )
    response = client.get("/api/host/hold/receipts")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "hold_state_unavailable"


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
    agent = _fresh("did:test:")
    request = await _record_agent_stop(store, agent, _fresh("op-"))

    page = await store.list_receipts(agent_id=agent, limit=10)

    assert len(page.receipts) == 1
    record = page.receipts[0]
    assert record.target_agent_id == agent
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
async def test_a_stop_by_routing_name_records_the_resolved_did(db_backend):
    """R3. The receipt names the agent the authority RESOLVED, not the address.

    Agent scope accepts a routing name as well as a DID. Recording the address
    as the agent would file this Stop under ``"alpha"``, and a reader asking
    for the agent's DID would never find it.
    """

    store = await _stop_store(db_backend)
    name, agent = _fresh("alpha-"), _fresh("did:test:")

    await _record_agent_stop(store, name, _fresh("op-"), agent_id=agent)

    by_did = await store.list_receipts(agent_id=agent, limit=10)
    assert [r.target_agent_id for r in by_did.receipts] == [agent]
    assert (await store.list_receipts(agent_id=name, limit=10)).receipts == ()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_caller_supplied_agent_identity_is_not_recorded(db_backend):
    """Only the authority's resolution is evidence; a caller's claim is not."""

    store = await _stop_store(db_backend)
    agent, claimed = _fresh("did:test:"), _fresh("did:test:forged-")
    request = replace(
        _agent_stop(agent, correlation_id=_fresh("op-")),
        target_agent_id=claimed,
    )

    await _authority(store, _target(agent, agent)).stop(request)

    assert (await store.list_receipts(agent_id=claimed, limit=10)).receipts == ()
    found = await store.list_receipts(agent_id=agent, limit=10)
    assert [r.target_agent_id for r in found.receipts] == [agent]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_an_unresolved_agent_stop_records_no_agent(db_backend):
    """An address that resolved to nothing records nothing — never a guess."""

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    trace = uuid4().hex

    await _authority(store).stop(
        _agent_stop(agent, correlation_id=_fresh("op-"), trace_id=trace)
    )

    page = await store.list_receipts(trace_id=trace, limit=10)
    assert [r.target_agent_id for r in page.receipts] == [None]
    assert [o.disposition for o in page.receipts[0].outcomes] == ["unreachable"]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_pre_fix_row_reads_back_as_not_recorded(db_backend):
    """R3. A row written before the fix recorded no agent, and says so.

    It is emphatically NOT recovered by re-hashing the live inventory: that
    would guess an identity the receipt never held.
    """

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    trace = uuid4().hex
    # The pre-fix writer persisted the caller's request unresolved.
    request = _agent_stop(agent, correlation_id=_fresh("op-"), trace_id=trace)
    assert request.target_agent_id is None
    await store.persist(request, _outcomes(request))

    page = await store.list_receipts(trace_id=trace, limit=10)
    assert len(page.receipts) == 1
    # Nothing recorded the agent, so filtering by it cannot find the row.
    assert (await store.list_receipts(agent_id=agent, limit=10)).receipts == ()

    assert page.receipts[0].target_agent_id is None
    # ...and the outcome's blinded agent identity is not offered as a fallback.
    assert page.receipts[0].outcomes[0].agent_id is not None
    assert disclosable_identity(page.receipts[0].outcomes[0].agent_id) is None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_an_exact_retry_replays_after_its_agent_is_gone(db_backend):
    """The resolved DID is evidence, not request semantics.

    A retry of the same correlation id after the agent was unloaded resolves
    to nothing. It is still the same request, so it replays the durable result
    — with the DID the ORIGINAL resolution recorded — instead of conflicting or
    cancelling twice.
    """

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    correlation_id = _fresh("op-")
    first = await _authority(store, _target(agent, agent)).stop(
        _agent_stop(agent, correlation_id=correlation_id)
    )

    replayed = await _authority(store).stop(
        _agent_stop(agent, correlation_id=correlation_id)
    )

    assert [o.receipt_id for o in replayed] == [o.receipt_id for o in first]
    assert [o.disposition.value for o in replayed] == ["stopped"]
    page = await store.list_receipts(agent_id=agent, limit=10)
    assert [r.target_agent_id for r in page.receipts] == [agent]

    # A DIFFERENT request under the same id is still a conflict.
    with pytest.raises(StopReceiptError):
        await store.load(
            _agent_stop(
                agent, correlation_id=correlation_id, reason="another reason"
            )
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_pages_are_bounded_and_totally_ordered(db_backend):
    """R6. Convergence is a property of the read model.

    Immutable rows paged on a commit-ordered ``feed_seq`` mean any walk over
    the same window yields the same set in the same order.
    """

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    for _index in range(5):
        await _record_agent_stop(store, agent, _fresh("op-"))

    first = await store.list_receipts(agent_id=agent, limit=2)
    assert len(first.receipts) == 2
    assert first.next_key is not None

    second = await store.list_receipts(
        agent_id=agent, after=first.next_key, limit=2
    )
    third = await store.list_receipts(
        agent_id=agent, after=second.next_key, limit=2
    )

    walked = [
        record.receipt_id
        for page in (first, second, third)
        for record in page.receipts
    ]
    whole = await store.list_receipts(agent_id=agent, limit=50)

    assert walked == [record.receipt_id for record in whole.receipts]
    assert len(walked) == 5
    assert len(set(walked)) == 5, "a page boundary repeated a receipt"
    # Exhausted: the last page issues no cursor.
    assert third.next_key is None
    # And the order is the one the route promises.
    sequence = [r.feed_seq for r in whole.receipts]
    assert sequence == sorted(set(sequence))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_filters_select_by_agent_and_trace(db_backend):
    store = await _stop_store(db_backend)
    alpha, beta, trace = _fresh("did:test:"), _fresh("did:test:"), uuid4().hex
    await _record_agent_stop(store, alpha, _fresh("op-"), trace_id=trace)
    await _record_agent_stop(store, beta, _fresh("op-"))

    by_agent = await store.list_receipts(agent_id=beta, limit=10)
    by_trace = await store.list_receipts(trace_id=trace, limit=10)

    assert [r.target_agent_id for r in by_agent.receipts] == [beta]
    assert [r.target_agent_id for r in by_trace.receipts] == [alpha]
    assert by_trace.receipts[0].trace_id == trace


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_host_fanout_is_findable_by_the_agent_it_reached(db_backend):
    """A host-scope receipt names no single agent in its header.

    Its per-target outcomes do, and those are stored in the clear because a
    host Stop carries no caller-supplied address to blind.
    """

    store = await _stop_store(db_backend)
    alpha, beta = _fresh("did:test:"), _fresh("did:test:")
    request = StopRequest(
        scope=StopScope.HOST,
        actor_id=OPERATOR,
        reason="fleet freeze",
        correlation_id=_fresh("op-"),
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
            for agent in (alpha, beta)
        ),
    )

    page = await store.list_receipts(agent_id=beta, limit=10)

    assert len(page.receipts) == 1
    record = page.receipts[0]
    assert record.scope == "host"
    assert record.target_agent_id is None
    assert [outcome.agent_id for outcome in record.outcomes] == [alpha, beta]
    assert [outcome.ordinal for outcome in record.outcomes] == [0, 1]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_unreachable_outcomes_are_returned_not_filtered(db_backend):
    """A target Stop could not reach is evidence, not noise."""

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    request = replace(
        _agent_stop(agent, correlation_id=_fresh("op-")),
        target_agent_id=agent,
    )
    await store.persist(
        request, _outcomes(request, disposition=StopDisposition.UNREACHABLE)
    )

    page = await store.list_receipts(agent_id=agent, limit=10)

    assert [o.disposition for o in page.receipts[0].outcomes] == ["unreachable"]


# --------------------------------------------------------------------------
# A cursor must never be overtaken by a later commit (R6)
# --------------------------------------------------------------------------


_FROZEN_CLOCK = {
    "postgres": "2000-01-01T00:00:00.000000+00:00",
    "sqlite": "2000-01-01T00:00:00.000+00:00",
}


def _freeze_database_clock(monkeypatch, backend_type):
    """Pin every receipt writer's statement clock to one instant.

    That is both failure shapes a timestamp key has at once: every receipt
    shares one clock tick (the SQLite millisecond tie) and the clock sits
    behind rows already written (a wall clock that stepped backwards).
    """

    from kestrel_sovereign.hold import state as hold_state
    from kestrel_sovereign.stop import receipt as receipt_module

    frozen = f"'{_FROZEN_CLOCK[backend_type]}'"
    for module in (receipt_module, hold_state):
        monkeypatch.setattr(module, "database_now_sql", lambda _db: frozen)


def _page_rows(page):
    return getattr(page, "receipts", None) or tuple(
        entry.receipt for entry in getattr(page, "entries", ())
    )


def _page_keys(page):
    if hasattr(page, "entries"):
        return [entry.feed_seq for entry in page.entries]
    return [record.feed_seq for record in page.receipts]


async def _walk_from(store, cursor, **filters):
    seen = []
    while True:
        page = await store.list_receipts(after=cursor, limit=1, **filters)
        seen.extend(record.receipt_id for record in _page_rows(page))
        keys = _page_keys(page)
        if page.next_key is None:
            return seen, (keys[-1] if keys else cursor)
        cursor = page.next_key


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_stalled_clock_cannot_sort_a_later_stop_behind_a_cursor(
    db_backend, monkeypatch
):
    """Commit order is feed order even when the clock does not advance.

    Keyed on the clock, the second receipt shares the first's instant and its
    random ``receipt_id`` decides the order: half the time it sorts BEFORE the
    cursor a consumer took after the first commit, and is skipped forever.
    """

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    _freeze_database_clock(monkeypatch, db_backend.backend_type)

    await _record_agent_stop(store, agent, _fresh("op-"))
    seen, cursor = await _walk_from(store, None, agent_id=agent)
    assert len(seen) == 1

    for _index in range(6):
        await _record_agent_stop(store, agent, _fresh("op-"))
        later, cursor = await _walk_from(store, cursor, agent_id=agent)
        assert len(later) == 1, "a later commit sorted behind the cursor"
        seen.extend(later)

    whole = await store.list_receipts(agent_id=agent, limit=50)
    assert [r.receipt_id for r in whole.receipts] == seen
    # The displayed time is exactly what the clock said: ordering was not
    # bought by rewriting it.
    assert {r.occurred_at for r in whole.receipts} == {
        _FROZEN_CLOCK[db_backend.backend_type]
    }


@pytest.mark.asyncio
async def test_a_stalled_clock_cannot_sort_a_later_hold_behind_a_cursor(
    hold_store, monkeypatch
):
    _freeze_database_clock(monkeypatch, "sqlite")

    await hold_store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope=HoldScope.AGENT,
        target_id=f"{ALPHA}-0",
        actor_id=OPERATOR,
        reason="freeze",
        operation_id="op-0",
    )
    seen, cursor = await _walk_from(hold_store, None)
    assert len(seen) == 1
    for index in range(1, 6):
        await hold_store.set_hold(
            authority=HoldAuthority.SOVEREIGN,
            scope=HoldScope.AGENT,
            target_id=f"{ALPHA}-{index}",
            actor_id=OPERATOR,
            reason="freeze",
            operation_id=f"op-{index}",
        )
        later, cursor = await _walk_from(hold_store, cursor)
        assert len(later) == 1, "a later commit sorted behind the cursor"
        seen.extend(later)

    whole = await hold_store.list_receipts(limit=50)
    assert [e.receipt.receipt_id for e in whole.entries] == seen
    assert {e.receipt.occurred_at for e in whole.entries} == {
        _FROZEN_CLOCK["sqlite"]
    }


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_paused_stop_commit_is_still_discoverable_after_a_later_one(
    db_backend, monkeypatch
):
    """The production PostgreSQL race the per-operation lock left open.

    Transaction A allocates its feed position and pauses. Transaction B is a
    DIFFERENT operation, so no per-operation lock orders it. Unserialized, B
    commits, a consumer pages it and holds B's cursor, then A commits BEHIND
    that cursor, forever. The receipt-feed lock makes B wait for A, so
    whatever the consumer paged, walking on from its cursor reaches both.
    """

    import asyncio

    if db_backend.backend_type != "postgres":
        # SQLite's one shared connection cannot hold two open transactions;
        # its BEGIN IMMEDIATE writer slot is the serialization this exercises.
        pytest.skip("concurrent Stop transactions need PostgreSQL")

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    allocated = asyncio.Event()
    release = asyncio.Event()
    real_execute = store._db.execute
    appends = 0

    async def pausing_execute(sql, params=()):
        # Pause A right after its header INSERT: the trigger has allocated
        # A's feed position and holds the feed lock, and A has not committed.
        nonlocal appends
        result = await real_execute(sql, params)
        if sql.startswith("INSERT INTO stop_receipts ("):
            appends += 1
            if appends == 1:
                allocated.set()
                await release.wait()
        return result

    monkeypatch.setattr(store._db, "execute", pausing_execute)

    first = asyncio.create_task(
        _record_agent_stop(store, agent, _fresh("op-a-"))
    )
    second = None
    try:
        await asyncio.wait_for(allocated.wait(), timeout=10)
        second = asyncio.create_task(
            _record_agent_stop(store, agent, _fresh("op-b-"))
        )
        await asyncio.sleep(0.5)
        # The consumer pages while A is paused and B has had time to commit.
        paged, cursor = await _walk_from(store, None, agent_id=agent)
        assert not second.done(), "B committed while A still held the feed slot"
    finally:
        # Always let A finish, or a failed assertion above strands its open
        # transaction and the backend's teardown waits on it forever.
        release.set()
        pending = [task for task in (first, second) if task is not None]
        await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True), timeout=10
        )
    for task in (first, second):
        task.result()

    later, _cursor = await _walk_from(store, cursor, agent_id=agent)
    assert len(paged) + len(later) == 2, "a committed receipt fell behind the cursor"
    whole = await store.list_receipts(agent_id=agent, limit=10)
    assert [r.receipt_id for r in whole.receipts] == paged + later


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_rows_that_predate_the_sequence_are_numbered_after_it(db_backend):
    """Upgrade: pre-existing rows get a position, appended — never interleaved.

    A consumer that paged the numbered history must still find a row numbered
    later, so late numbering has to sort AFTER everything already numbered.
    """

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    for _index in range(2):
        await _record_agent_stop(store, agent, _fresh("op-"))
    before = await store.list_receipts(agent_id=agent, limit=10)
    cursor = before.receipts[-1].feed_seq

    # Model a row the initialization backfill finds unnumbered.
    unnumbered = await _record_agent_stop(store, agent, _fresh("op-legacy-"))
    await store._db.execute(
        "UPDATE stop_receipts SET feed_seq = NULL WHERE operation_id = ?",
        (opaque_stop_identifier("operation", unnumbered.correlation_id),),
    )
    assert (
        await store.list_receipts(agent_id=agent, after=cursor, limit=10)
    ).receipts == ()

    await store.ensure_schema()

    later = await store.list_receipts(agent_id=agent, after=cursor, limit=10)
    assert len(later.receipts) == 1
    assert later.receipts[0].feed_seq > cursor


async def _append_stop_as_pre_sequence_binary(db, *, agent_id):
    """Write one agent-scope Stop receipt exactly as a pre-#3159 binary did.

    That binary names no ``feed_seq`` column and takes no receipt-feed lock —
    it only knows its own per-operation lock. This is the rolling-upgrade
    writer: still running after a newer process added the column.
    """

    receipt_id = str(uuid4())
    operation = opaque_stop_identifier("operation", _fresh("op-old-"))
    now_sql = database_now_sql(db)
    async with db.transaction():
        await db.execute(
            "INSERT INTO stop_receipts ("
            "receipt_id, operation_id, request_fingerprint, scope, "
            "actor_id, requested_target, target_agent_id, reason, "
            "cascade, occurred_at, turn_id, span_id, trace_id"
            f") VALUES (?, ?, ?, 'agent', ?, NULL, ?, ?, 0, {now_sql}, "
            "NULL, NULL, NULL)",
            (receipt_id, operation, "fp-old", OPERATOR, agent_id, "old binary"),
        )
        await db.execute(
            "INSERT INTO stop_receipt_outcomes ("
            "receipt_id, ordinal, resolved_target, agent_id, disposition, "
            "detail) VALUES (?, 0, ?, ?, 'stopped', NULL)",
            (receipt_id, agent_id, agent_id),
        )
    return receipt_id


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_legacy_writer_after_migration_is_visible_without_a_restart(
    db_backend,
):
    """A rolling upgrade's older writer must not create an invisible receipt.

    The newer process has already migrated. The older one keeps appending with
    SQL that has never heard of ``feed_seq``. Nothing re-runs the schema
    backfill afterwards, so if the database did not number that row itself it
    would be excluded from every page for good. No ``ensure_schema()`` call
    follows the legacy append on purpose: that call is exactly what production
    does not get.
    """

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    await _record_agent_stop(store, agent, _fresh("op-new-"))
    before = await store.list_receipts(agent_id=agent, limit=10)
    cursor = before.receipts[-1].feed_seq

    legacy_id = await _append_stop_as_pre_sequence_binary(
        store._db, agent_id=agent
    )
    await _record_agent_stop(store, agent, _fresh("op-newer-"))

    later = await store.list_receipts(agent_id=agent, after=cursor, limit=10)
    assert len(later.receipts) == 2
    assert later.receipts[0].receipt_id == legacy_id
    assert later.receipts[0].target_agent_id == agent
    positions = [record.feed_seq for record in later.receipts]
    assert positions == sorted(positions) and positions[0] > cursor


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_feed_sequence_holds_the_whole_advertised_cursor_range(
    db_backend,
):
    """The cursor contract is 64-bit, so the column must be too.

    A 32-bit column would refuse the insert that numbers the row after
    ``2**31 - 1``, failing every later Stop or Hold receipt write.
    """

    store = await _stop_store(db_backend)
    agent = _fresh("did:test:")
    await _record_agent_stop(store, agent, _fresh("op-wide-"))
    first = (await store.list_receipts(agent_id=agent, limit=10)).receipts[-1]
    # Relative to the table's maximum: a shared Postgres database keeps every
    # earlier run's rows, and the column is uniquely indexed.
    row = await store._db.fetchone("SELECT MAX(feed_seq) FROM stop_receipts")
    wide = max(row[0], 2**31 - 1) + 1
    await store._db.execute(
        "UPDATE stop_receipts SET feed_seq = ? WHERE receipt_id = ?",
        (wide, first.receipt_id),
    )

    await _record_agent_stop(store, agent, _fresh("op-past-32-bit-"))

    later = await store.list_receipts(agent_id=agent, after=wide, limit=10)
    assert [record.feed_seq for record in later.receipts] == [wide + 1]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_database_numbers_every_insert_that_names_no_sequence(
    db_backend,
):
    """The allocator itself, on both engines, for any receipt feed table.

    Hold receipts use the same allocator; their integrity chain makes a
    hand-forged Hold row unreadable by design, so the mechanism is proven here
    on a plain table and end to end on Stop above.
    """


    db = AsyncDatabase(db_backend)
    table = f"feed_probe_{uuid4().hex[:12]}"
    lock_key = "kestrel:test:feed-probe"
    await db.execute(
        f"CREATE TABLE {table} ("
        "receipt_id TEXT NOT NULL PRIMARY KEY, occurred_at TEXT NOT NULL)"
    )
    try:
        await _assert_allocator_numbers_every_insert(db, table, lock_key)
    finally:
        # A shared PostgreSQL test database outlives this test.
        await db.execute(f"DROP TABLE {table}")
        if db_backend.backend_type == "postgres":
            await db.execute(
                f"DROP FUNCTION IF EXISTS {table}_assign_feed_seq_v1()"
            )


async def _assert_allocator_numbers_every_insert(db, table, lock_key):
    from kestrel_sovereign.storage.feed_sequence import ensure_feed_sequence

    # Two rows written before the column existed: backfilled in time order.
    for receipt_id, occurred_at in (("b", "2026-01-02"), ("a", "2026-01-01")):
        await db.execute(
            f"INSERT INTO {table} (receipt_id, occurred_at) VALUES (?, ?)",
            (receipt_id, occurred_at),
        )
    async with db.transaction():
        await ensure_feed_sequence(db, table=table, lock_key=lock_key)
    # Two rows written afterwards by SQL that never names the column. Their
    # displayed time is EARLIER than the backfilled rows; their position is not.
    for receipt_id in ("c", "d"):
        async with db.transaction():
            await db.execute(
                f"INSERT INTO {table} (receipt_id, occurred_at) "
                "VALUES (?, '2025-01-01')",
                (receipt_id,),
            )
    rows = await db.fetchall(
        f"SELECT receipt_id, feed_seq FROM {table} ORDER BY feed_seq"
    )
    assert [tuple(row) for row in rows] == [
        ("a", 1),
        ("b", 2),
        ("c", 3),
        ("d", 4),
    ]
    # Re-running the migration renumbers nothing and installs no second
    # allocator (a second one would skip a number on the next insert).
    async with db.transaction():
        await ensure_feed_sequence(db, table=table, lock_key=lock_key)
    async with db.transaction():
        await db.execute(
            f"INSERT INTO {table} (receipt_id, occurred_at) VALUES ('e', 'x')"
        )
    again = await db.fetchall(
        f"SELECT receipt_id, feed_seq FROM {table} ORDER BY feed_seq"
    )
    assert [tuple(row) for row in again] == [tuple(row) for row in rows] + [
        ("e", 5)
    ]


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
# A sub-millisecond bound on SQLite (#3159 review r3)
#
# SQLite's statement clock stores milliseconds. A bound finer than that must
# round UP to the next stored tick: ``x >= .500001`` excludes a row stored at
# ``.500`` and ``x < .500001`` includes it. Truncating the inclusive lower
# bound to ``.500`` admitted a receipt recorded before the requested instant.
# --------------------------------------------------------------------------


_SUB_MS_STORED = "2000-01-01T00:00:00.500+00:00"
_AT_TICK = datetime(2000, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc)
_JUST_AFTER_TICK = datetime(2000, 1, 1, 0, 0, 0, 500001, tzinfo=timezone.utc)


def _pin_sqlite_clock(monkeypatch, stored):
    from kestrel_sovereign.hold import state as hold_state
    from kestrel_sovereign.stop import receipt as receipt_module

    for module in (receipt_module, hold_state):
        monkeypatch.setattr(
            module, "database_now_sql", lambda _db: f"'{stored}'"
        )


async def _bounded_ids(store, **window):
    page = await store.list_receipts(limit=10, **window)
    return [record.receipt_id for record in _page_rows(page)]


async def _assert_sub_millisecond_bounds(store, receipt_id):
    # Inclusive lower bound: the stored ``.500`` is before ``.500001``.
    assert await _bounded_ids(store, since=_JUST_AFTER_TICK) == []
    assert await _bounded_ids(store, since=_AT_TICK) == [receipt_id]
    # Exclusive upper bound: ``.500`` is before ``.500001`` but not ``.500``.
    assert await _bounded_ids(store, until=_JUST_AFTER_TICK) == [receipt_id]
    assert await _bounded_ids(store, until=_AT_TICK) == []


@pytest.mark.asyncio
async def test_a_sub_millisecond_window_bounds_stop_receipts_exactly(
    tmp_path, monkeypatch
):
    db = await AsyncDatabase.sqlite(str(tmp_path / "stop.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        _pin_sqlite_clock(monkeypatch, _SUB_MS_STORED)
        await _record_agent_stop(store, ALPHA, "op-sub-ms")

        whole = await store.list_receipts(limit=10)
        assert [r.occurred_at for r in whole.receipts] == [_SUB_MS_STORED]
        await _assert_sub_millisecond_bounds(
            store, whole.receipts[0].receipt_id
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_sub_millisecond_window_bounds_hold_receipts_exactly(
    hold_store, monkeypatch
):
    _pin_sqlite_clock(monkeypatch, _SUB_MS_STORED)
    held = await hold_store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="runaway loop",
        operation_id="op-sub-ms",
    )

    assert held.receipt.occurred_at == _SUB_MS_STORED
    await _assert_sub_millisecond_bounds(hold_store, held.receipt.receipt_id)


# --------------------------------------------------------------------------
# A caller-controlled bound is a 400, never a 500
# --------------------------------------------------------------------------

# More digits than ``int()`` will convert (``sys.get_int_max_str_digits``).
_OVERSIZED_LIMIT = "9" * 5000
# Parses, but rounding its sub-millisecond part up to SQLite's stored tick
# leaves the calendar ``datetime`` can represent.
_PAST_THE_LAST_STORABLE_TICK = "9999-12-31T23:59:59.999999Z"
# Parses, but normalizing its offset to UTC leaves the calendar.
_PAST_THE_CALENDAR_IN_UTC = "9999-12-31T23:59:59-01:00"
_BEFORE_THE_CALENDAR_IN_UTC = "0001-01-01T00:00:00+01:00"


class TestAMalformedBoundIsRefusedNotCrashed:
    @pytest.mark.parametrize(
        "router,path,params,code",
        [
            (
                router,
                path,
                params,
                code,
            )
            for router, path in (
                (stop_router, "/api/host/stop/receipts"),
                (hold_router, "/api/host/hold/receipts"),
            )
            for params, code in (
                ({"limit": _OVERSIZED_LIMIT}, "receipt_page_invalid"),
                (
                    {"since": _PAST_THE_LAST_STORABLE_TICK},
                    "receipt_window_invalid",
                ),
                (
                    {"until": _PAST_THE_LAST_STORABLE_TICK},
                    "receipt_window_invalid",
                ),
                (
                    {"until": _PAST_THE_CALENDAR_IN_UTC},
                    "receipt_window_invalid",
                ),
                (
                    {"since": _BEFORE_THE_CALENDAR_IN_UTC},
                    "receipt_window_invalid",
                ),
            )
        ],
    )
    def test_the_sovereign_gets_a_400(self, router, path, params, code):
        client = _app(
            router,
            _sovereign(),
            stop_receipt_store=_RaisingStopStore(AssertionError("never read")),
            host_context=SimpleNamespace(
                hold_store=_RaisingHoldStore(AssertionError("never read")),
                backend_error="",
            ),
        )

        response = client.get(path, params=params)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == code

    def test_leading_zeros_do_not_change_a_page_size(self):
        assert resolve_page_size("050") == 50
        assert resolve_page_size("0" * 5000 + "7") == 7


def test_a_bound_renders_with_a_four_digit_year_on_every_platform():
    """``strftime('%Y')`` renders year 5 as ``5`` under glibc.

    An unpadded year sorts after every four-digit year the clock writes.
    """

    early = datetime(5, 1, 1, tzinfo=timezone.utc)
    for backend_type in ("sqlite", "postgres"):
        text = database_timestamp_bound_text(
            SimpleNamespace(backend_type=backend_type), early
        )
        assert text.startswith("0005-01-01T00:00:00.")


@pytest.mark.parametrize("backend_type", ["sqlite", "postgres"])
def test_a_store_refuses_a_bound_past_the_last_storable_tick(backend_type):
    db = SimpleNamespace(backend_type=backend_type)
    # The last tick itself renders; one microsecond past it is refused as a
    # caller error, never as an OverflowError from rounding.
    assert database_timestamp_bound_text(db, LATEST_TIMESTAMP_BOUND).startswith(
        "9999-12-31T23:59:59.999"
    )
    beyond = LATEST_TIMESTAMP_BOUND.replace(microsecond=999999)
    with pytest.raises(TimestampBoundOutOfRange):
        database_timestamp_bound_text(db, beyond)


@pytest.mark.asyncio
async def test_the_last_storable_tick_is_a_usable_bound_on_real_sqlite_stores(
    tmp_path, hold_store
):
    """The maximum accepted bound reaches both real stores and answers."""

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop.db"))
    try:
        stop_store = StopReceiptStore(db)
        await stop_store.ensure_schema()
        await _record_agent_stop(stop_store, ALPHA, "op-latest-bound")
        await hold_store.set_hold(
            authority=HoldAuthority.SOVEREIGN,
            scope=HoldScope.AGENT,
            target_id=ALPHA,
            actor_id=OPERATOR,
            reason="runaway loop",
            operation_id="op-latest-bound",
        )

        for store in (stop_store, hold_store):
            assert len(
                await _bounded_ids(store, until=LATEST_TIMESTAMP_BOUND)
            ) == 1
            assert await _bounded_ids(store, since=LATEST_TIMESTAMP_BOUND) == []
    finally:
        await db.close()


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
        authority=HoldAuthority.SOVEREIGN,
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="runaway loop",
        operation_id="op-hold",
    )
    released = await hold_store.release_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="fixed",
        operation_id="op-release",
        expected_hold_receipt_id=held.receipt.receipt_id,
    )

    page = await hold_store.list_receipts(limit=10)
    receipts = [entry.receipt for entry in page.entries]

    assert [r.action.value for r in receipts] == ["hold", "release"]
    # The hold event is untouched by the resume...
    assert receipts[0].receipt_id == held.receipt.receipt_id
    assert receipts[0].reason == "runaway loop"
    # ...and the resume points back at exactly the hold it ended.
    assert receipts[1].receipt_id == released.receipt.receipt_id
    assert receipts[1].prior_hold_receipt_id == held.receipt.receipt_id
    assert receipts[1].disposition.value == "applied"
    # The live latch is gone; only the history explains the agent.
    assert await hold_store.get_hold(HoldScope.AGENT, ALPHA) is None


@pytest.mark.asyncio
async def test_a_pre_sequence_hold_database_upgrades_with_its_evidence_intact(
    tmp_path,
):
    """An existing Hold store gains ``feed_seq`` without disturbing custody.

    ``feed_seq`` sits outside every content digest and witness, so numbering
    rows the old schema wrote must leave the history anchor, witnesses, and
    boot validation exactly as satisfied as before.
    """

    path = str(tmp_path / "host.db")
    db = await AsyncDatabase.sqlite(path)
    try:
        store = HoldStore(db)
        await store.ensure_schema()
        for index in range(3):
            await store.set_hold(
                authority=HoldAuthority.SOVEREIGN,
                scope=HoldScope.AGENT,
                target_id=f"{ALPHA}-{index}",
                actor_id=OPERATOR,
                reason="freeze",
                operation_id=f"op-{index}",
            )
        # Return the table to the shape a pre-#3159 release left.
        # The allocator references the column, so SQLite refuses to drop
        # the column while it exists; a pre-#3159 schema had neither.
        await db.execute("DROP TRIGGER trg_hold_receipts_feed_seq_v1")
        await db.execute("DROP INDEX idx_hold_receipts_feed_seq")
        await db.execute("ALTER TABLE hold_receipts DROP COLUMN feed_seq")
        assert not await db.column_exists("hold_receipts", "feed_seq")

        upgraded = HoldStore(db)
        await upgraded.ensure_schema()
        await upgraded.read_boot_state()

        page = await upgraded.list_receipts(limit=10)
        assert [e.receipt.target_id for e in page.entries] == [
            f"{ALPHA}-{index}" for index in range(3)
        ]
        assert [e.feed_seq for e in page.entries] == [1, 2, 3]
        # And a new append continues the sequence.
        await upgraded.set_hold(
            authority=HoldAuthority.SOVEREIGN,
            scope=HoldScope.AGENT,
            target_id=BETA,
            actor_id=OPERATOR,
            reason="freeze",
            operation_id="op-after",
        )
        tail = await upgraded.list_receipts(after=3, limit=10)
        assert [(e.feed_seq, e.receipt.target_id) for e in tail.entries] == [
            (4, BETA)
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_refused_and_already_in_state_are_real_operator_acts(hold_store):
    """R5. They are returned, not filtered: somebody really did ask."""

    held = await hold_store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="runaway loop",
        operation_id="op-hold",
    )
    # Same actor, same reason: the latch is already exactly what was asked
    # for, so the act is recorded as already-in-state rather than re-applied.
    await hold_store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="runaway loop",
        operation_id="op-hold-again",
    )
    await hold_store.release_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope=HoldScope.AGENT,
        target_id=ALPHA,
        actor_id=OPERATOR,
        reason="stale",
        operation_id="op-stale",
        expected_hold_receipt_id=f"{held.receipt.receipt_id}-not-current",
    )

    page = await hold_store.list_receipts(limit=10)

    assert [e.receipt.disposition.value for e in page.entries] == [
        "applied",
        "already_in_state",
        "refused_stale",
    ]


@pytest.mark.asyncio
async def test_hold_history_pages_in_a_total_order(hold_store):
    for index in range(4):
        await hold_store.set_hold(
            authority=HoldAuthority.SOVEREIGN,
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
        e.receipt.receipt_id for page in (first, second) for e in page.entries
    ]
    assert walked == [e.receipt.receipt_id for e in whole.entries]
    assert len(set(walked)) == 4
    assert second.next_key is None
    sequence = [e.feed_seq for e in whole.entries]
    assert sequence == sorted(set(sequence))


@pytest.mark.asyncio
async def test_hold_history_filters_by_scope_and_target(hold_store):
    await hold_store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
        scope=HoldScope.HOST,
        actor_id=OPERATOR,
        reason="fleet freeze",
        operation_id="op-host",
    )
    await hold_store.set_hold(
        authority=HoldAuthority.SOVEREIGN,
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

    assert [e.receipt.reason for e in host_only.entries] == ["fleet freeze"]
    assert [e.receipt.target_id for e in alpha_only.entries] == [ALPHA]


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
    assert isinstance(record["feed_seq"], int)
    assert first["next_cursor"] == str(first["receipts"][-1]["feed_seq"])
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
            authority=HoldAuthority.SOVEREIGN,
            scope=HoldScope.AGENT,
            target_id=ALPHA,
            actor_id=OPERATOR,
            reason="runaway loop",
            operation_id="op-hold",
        )
        await store.release_hold(
            authority=HoldAuthority.SOVEREIGN,
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
    assert (
        payload["receipts"][0]["feed_seq"] < payload["receipts"][1]["feed_seq"]
    )
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

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "receipt_page_invalid"
