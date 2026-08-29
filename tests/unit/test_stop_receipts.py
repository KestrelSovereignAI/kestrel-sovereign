"""Durability and fail-closed evidence gates for cooperative Stop (#3152)."""

import asyncio
import json
from dataclasses import replace
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.stop import (
    CancellationAuthority,
    CooperativeStopTarget,
    StopCleanupRegistry,
    StopDisposition,
    StopOutcome,
    StopReceiptConflict,
    StopReceiptCorruptError,
    StopReceipt,
    StopReceiptStore,
    StopRequest,
    StopScope,
)


class _EndpointReplayStore:
    def __init__(self):
        self.request = None
        self.receipt = None

    async def load(self, request):
        if self.receipt is None:
            return None
        assert request == self.request
        return self.receipt

    async def persist(self, request, outcomes):
        receipt_id = f"receipt-{request.correlation_id}"
        self.request = request
        self.receipt = StopReceipt(
            receipt_id=receipt_id,
            operation_id=request.correlation_id,
            request_fingerprint="test-fingerprint",
            scope=request.scope.value,
            actor_id=request.actor_id,
            requested_target=request.target,
            target_agent_id=request.target_agent_id,
            reason=request.reason,
            cascade=request.cascade,
            occurred_at="2026-08-28T00:00:00+00:00",
            turn_id=request.turn_id,
            span_id=request.span_id,
            trace_id=request.trace_id,
            outcomes=tuple(
                replace(outcome, receipt_id=receipt_id) for outcome in outcomes
            ),
        )
        return self.receipt


def _request(
    *,
    correlation_id: str | None = None,
    reason: str = "unsafe loop",
    span_id: str | None = None,
    trace_id: str | None = None,
):
    return StopRequest(
        scope=StopScope.TURN,
        target="turn-7",
        target_agent_id="did:test:agent",
        actor_id="did:test:operator",
        reason=reason,
        cascade=False,
        correlation_id=correlation_id or f"stop-{uuid4()}",
        span_id=span_id,
        trace_id=trace_id,
    )


def _outcomes(
    request: StopRequest,
    disposition: StopDisposition = StopDisposition.STOPPED,
):
    return (
        StopOutcome(
            scope=request.scope,
            requested_target=request.target,
            resolved_target="agent-runtime-7",
            agent_id="did:test:agent",
            disposition=disposition,
            correlation_id=request.correlation_id,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_receipt_store_roundtrips_exact_evidence_on_available_backends(
    db_backend,
):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    store = StopReceiptStore(AsyncDatabase(db_backend))
    await store.ensure_schema()
    request = _request(
        span_id="0123456789abcdef",
        trace_id="0123456789abcdef0123456789abcdef",
    )

    receipt = await store.persist(request, _outcomes(request))
    replay = await store.load(request)

    assert replay == receipt
    assert receipt.operation_id == request.correlation_id
    assert receipt.actor_id == request.actor_id
    assert receipt.reason == request.reason
    assert receipt.turn_id == request.target
    assert receipt.span_id == request.span_id
    assert receipt.trace_id == request.trace_id
    assert receipt.occurred_at
    assert receipt.outcomes[0].receipt_id == receipt.receipt_id
    assert receipt.outcomes[0].disposition is StopDisposition.STOPPED


@pytest.mark.asyncio
async def test_acknowledged_turn_stop_is_queryable_by_durable_target(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-target.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        acknowledged = _request(correlation_id="acknowledged-turn-stop")
        unreachable = replace(
            _request(correlation_id="unreachable-turn-stop"),
            target="turn-unreachable",
            turn_id="turn-unreachable",
        )
        await store.persist(acknowledged, _outcomes(acknowledged))
        await store.persist(
            unreachable,
            _outcomes(unreachable, StopDisposition.UNREACHABLE),
        )

        assert await store.has_acknowledged_turn_stop(
            "did:test:agent", "turn-7"
        )
        assert not await store.has_acknowledged_turn_stop(
            "did:test:agent", "turn-unreachable"
        )
        assert not await store.has_acknowledged_turn_stop(
            "did:test:other", "turn-7"
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_receipt_survives_sqlite_connection_restart(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    path = tmp_path / "stop-restart.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    first = StopReceiptStore(first_db)
    await first.ensure_schema()
    request = _request()
    written = await first.persist(request, _outcomes(request))
    await first_db.close()

    second_db = await AsyncDatabase.sqlite(str(path))
    try:
        second = StopReceiptStore(second_db)
        await second.ensure_schema()
        assert await second.load(request) == written
    finally:
        await second_db.close()


@pytest.mark.asyncio
async def test_exact_replay_preserves_original_durable_outcome(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-replay.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        request = _request()
        original = await store.persist(request, _outcomes(request))

        replay = await store.persist(
            request,
            _outcomes(request, StopDisposition.ALREADY_COMPLETE),
        )

        assert replay == original
        assert replay.outcomes[0].disposition is StopDisposition.STOPPED
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_exact_replay_does_not_depend_on_changed_live_inventory():
    request = _request()
    store = _EndpointReplayStore()
    receipt = await store.persist(request, _outcomes(request))
    inventory = MagicMock(side_effect=AssertionError("inventory is no longer live"))
    authority = CancellationAuthority(
        inventory,
        cleanup_registry=StopCleanupRegistry(),
        receipt_store=store,
    )

    replay = await authority.stop(request)

    assert replay == receipt.outcomes
    inventory.assert_not_called()


@pytest.mark.asyncio
async def test_exact_retry_preserves_first_transport_trace_evidence(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-trace-retry.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        first = _request(
            correlation_id="same-stop-operation",
            span_id="1111111111111111",
            trace_id="11111111111111111111111111111111",
        )
        written = await store.persist(first, _outcomes(first))
        retry = _request(
            correlation_id=first.correlation_id,
            span_id="2222222222222222",
            trace_id="22222222222222222222222222222222",
        )

        replay = await store.load(retry)

        assert replay == written
        assert replay.span_id == first.span_id
        assert replay.trace_id == first.trace_id
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_operation_reuse_for_different_request_fails_closed(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-conflict.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        operation = f"stop-{uuid4()}"
        first = _request(correlation_id=operation)
        await store.persist(first, _outcomes(first))
        conflicting = _request(
            correlation_id=operation,
            reason="different operation",
        )

        with pytest.raises(StopReceiptConflict):
            await store.load(conflicting)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_request_fingerprint_cannot_bless_a_corrupt_receipt_header(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-corrupt-header.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        request = _request()
        await store.persist(request, _outcomes(request))
        await db.execute(
            "UPDATE stop_receipts SET actor_id = ? WHERE operation_id = ?",
            ("did:test:forged", request.correlation_id),
        )

        with pytest.raises(StopReceiptCorruptError, match="header"):
            await store.load(request)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_initialize_and_shutdown_host_owned_receipt_store(
    tmp_path,
    monkeypatch,
):
    from fastapi import FastAPI

    from kestrel_sovereign import server
    from kestrel_sovereign.host_features import storage

    path = tmp_path / "host-stop-receipts.db"
    monkeypatch.setattr(storage, "prepare_host_database", lambda: path)
    monkeypatch.setattr(storage, "validate_sqlite_family_private", lambda _path: None)
    app = FastAPI()

    await server._initialize_stop_receipts(app)

    assert isinstance(app.state.stop_receipt_store, StopReceiptStore)
    assert app.state.stop_receipt_store_error == ""
    db = app.state.stop_receipt_db
    close = AsyncMock(wraps=db.close)
    db.close = close

    await server._shutdown_stop_receipts(app)

    close.assert_awaited_once()
    assert app.state.stop_receipt_store is None
    assert app.state.stop_receipt_db is None


@pytest.mark.asyncio
async def test_initialize_receipts_uses_configured_shared_postgres(
    tmp_path,
    monkeypatch,
):
    from fastapi import FastAPI

    from kestrel_sovereign import server
    from kestrel_sovereign.host_features import storage
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "postgres-double.db"))
    postgres = AsyncMock(return_value=db)
    prepare = MagicMock(side_effect=AssertionError("SQLite path must not open"))
    monkeypatch.setenv("KESTREL_DB_BACKEND", "postgres")
    monkeypatch.setenv("KESTREL_DATABASE_URL", "postgresql://shared/stop")
    monkeypatch.setattr(AsyncDatabase, "postgres", postgres)
    monkeypatch.setattr(storage, "prepare_host_database", prepare)
    app = FastAPI()

    await server._initialize_stop_receipts(app)

    postgres.assert_awaited_once_with("postgresql://shared/stop")
    prepare.assert_not_called()
    assert isinstance(app.state.stop_receipt_store, StopReceiptStore)
    await server._shutdown_stop_receipts(app)


@pytest.mark.asyncio
async def test_receipt_store_startup_failure_degrades_to_fail_closed(
    monkeypatch,
):
    from fastapi import FastAPI

    from kestrel_sovereign import server
    from kestrel_sovereign.host_features import storage

    def fail_prepare():
        raise RuntimeError("host store unavailable")

    monkeypatch.setattr(storage, "prepare_host_database", fail_prepare)
    app = FastAPI()

    await server._initialize_stop_receipts(app)

    assert app.state.stop_receipt_store is None
    assert app.state.stop_receipt_db is None
    assert app.state.stop_receipt_store_error == "RuntimeError"


@pytest.mark.asyncio
async def test_concurrent_exact_writers_return_one_receipt(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    path = tmp_path / "stop-race.db"
    first_db = await AsyncDatabase.sqlite(str(path))
    second_db = await AsyncDatabase.sqlite(str(path))
    try:
        first = StopReceiptStore(first_db)
        second = StopReceiptStore(second_db)
        await first.ensure_schema()
        request = _request()
        one, two = await asyncio.gather(
            first.persist(request, _outcomes(request)),
            second.persist(request, _outcomes(request)),
        )

        assert one.receipt_id == two.receipt_id
        rows = await first_db.fetchall(
            "SELECT receipt_id FROM stop_receipts WHERE operation_id = ?",
            (request.correlation_id,),
        )
        assert rows == [(one.receipt_id,)]
    finally:
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_partial_host_fanout_persists_every_exact_target_outcome(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-partial-fanout.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        request = StopRequest(
            scope=StopScope.HOST,
            actor_id="did:test:operator",
            reason="fleet loop",
            correlation_id="partial-fanout",
        )
        outcomes = tuple(
            StopOutcome(
                scope=StopScope.HOST,
                requested_target=None,
                resolved_target=target,
                agent_id=f"did:test:{target}",
                disposition=disposition,
                correlation_id=request.correlation_id,
                detail=detail,
            )
            for target, disposition, detail in (
                ("alpha", StopDisposition.STOPPED, None),
                ("bravo", StopDisposition.ALREADY_COMPLETE, "already done"),
                ("charlie", StopDisposition.UNREACHABLE, "timed out"),
                ("delta", StopDisposition.REFUSED, "policy refusal"),
            )
        )

        written = await store.persist(request, outcomes)
        replay = await store.load(request)

        assert replay == written
        assert [outcome.resolved_target for outcome in replay.outcomes] == [
            "alpha",
            "bravo",
            "charlie",
            "delta",
        ]
        assert [outcome.disposition for outcome in replay.outcomes] == [
            StopDisposition.STOPPED,
            StopDisposition.ALREADY_COMPLETE,
            StopDisposition.UNREACHABLE,
            StopDisposition.REFUSED,
        ]
        assert {outcome.receipt_id for outcome in replay.outcomes} == {
            replay.receipt_id
        }
        assert replay.outcomes[2].detail == "timed out"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_host_receipt_rejects_empty_ambiguous_fanout(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-empty-host.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        request = StopRequest(
            scope=StopScope.HOST,
            actor_id="did:test:operator",
            correlation_id="empty-host-fanout",
        )

        with pytest.raises(ValueError, match="at least one target outcome"):
            await store.persist(request, ())
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_host_receipt_reader_rejects_legacy_empty_fanout(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.stop.receipt import _fingerprint

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-empty-host-read.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        request = StopRequest(
            scope=StopScope.HOST,
            actor_id="did:test:operator",
            correlation_id="legacy-empty-host-fanout",
        )
        await db.execute(
            "INSERT INTO stop_receipts ("
            "receipt_id, operation_id, request_fingerprint, scope, actor_id, "
            "requested_target, target_agent_id, reason, cascade, occurred_at, "
            "turn_id, span_id, trace_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-empty-receipt",
                request.correlation_id,
                _fingerprint(request),
                StopScope.HOST.value,
                request.actor_id,
                None,
                None,
                None,
                1,
                "2026-08-28T00:00:00+00:00",
                None,
                None,
                None,
            ),
        )

        with pytest.raises(StopReceiptCorruptError, match="missing its target"):
            await store.load(request)
    finally:
        await db.close()


class _FailingStore:
    def __init__(self, *, fail_load=False, fail_persist=False):
        self.fail_load = fail_load
        self.fail_persist = fail_persist

    async def load(self, _request):
        if self.fail_load:
            raise RuntimeError("store unavailable")
        return None

    async def persist(self, _request, _outcomes):
        if self.fail_persist:
            raise RuntimeError("write unavailable")
        raise AssertionError("test store expected to fail")


@pytest.mark.asyncio
async def test_unreadable_receipt_store_prevents_cancellation():
    cancel = AsyncMock(return_value=StopDisposition.STOPPED)
    request = _request()
    authority = CancellationAuthority(
        lambda: (
            CooperativeStopTarget(
                "agent-runtime-7",
                "did:test:agent",
                cancel,
                turn_ids=frozenset({request.target}),
            ),
        ),
        cleanup_registry=StopCleanupRegistry(),
        receipt_store=_FailingStore(fail_load=True),
    )

    outcomes = await authority.stop(request)

    cancel.assert_not_awaited()
    assert outcomes[0].disposition is StopDisposition.REFUSED
    assert "not attempted" in (outcomes[0].detail or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_disposition",
    [StopDisposition.STOPPED, StopDisposition.ALREADY_COMPLETE],
)
async def test_failed_receipt_write_never_reports_unwitnessed_stop(
    terminal_disposition,
):
    calls = 0

    async def cancel(_request):
        nonlocal calls
        calls += 1
        return terminal_disposition

    request = _request()
    authority = CancellationAuthority(
        lambda: (
            CooperativeStopTarget(
                "agent-runtime-7",
                "did:test:agent",
                cancel,
                turn_ids=frozenset({request.target}),
            ),
        ),
        cleanup_registry=StopCleanupRegistry(),
        receipt_store=_FailingStore(fail_persist=True),
    )

    outcomes = await authority.stop(request)

    assert calls == 1
    assert outcomes[0].disposition is StopDisposition.REFUSED
    assert outcomes[0].receipt_id is None
    assert "may have completed" in (outcomes[0].detail or "")


@pytest.mark.asyncio
async def test_operation_claim_precedes_concurrent_target_side_effect(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-claim-race.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        request = _request(correlation_id="one-effect")
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def cancel(_request):
            nonlocal calls
            calls += 1
            started.set()
            if calls == 1:
                await release.wait()
            return StopDisposition.STOPPED

        def authority():
            return CancellationAuthority(
                lambda: (
                    CooperativeStopTarget(
                        "agent-runtime-7",
                        "did:test:agent",
                        cancel,
                        turn_ids=frozenset({request.target}),
                    ),
                ),
                cleanup_registry=StopCleanupRegistry(),
                receipt_store=store,
            )

        first = asyncio.create_task(authority().stop(request))
        await started.wait()
        overlapping = await authority().stop(request)
        release.set()
        committed = await first
        assert overlapping[0].disposition is StopDisposition.REFUSED
        assert "already in progress" in (overlapping[0].detail or "")
        assert calls == 1

        replay = await authority().stop(request)
        assert committed == replay
        assert calls == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_cancelled_caller_cannot_split_effect_from_receipt(tmp_path):
    from kestrel_sovereign.storage.async_database import AsyncDatabase

    db = await AsyncDatabase.sqlite(str(tmp_path / "stop-owned-receipt.db"))
    try:
        store = StopReceiptStore(db)
        await store.ensure_schema()
        request = _request(correlation_id="caller-cancelled-stop")
        started = asyncio.Event()
        release = asyncio.Event()

        async def cancel(_request):
            started.set()
            await release.wait()
            return StopDisposition.STOPPED

        authority = CancellationAuthority(
            lambda: (
                CooperativeStopTarget(
                    "agent-runtime-7",
                    "did:test:agent",
                    cancel,
                    turn_ids=frozenset({request.target}),
                ),
            ),
            cleanup_registry=StopCleanupRegistry(),
            receipt_store=store,
        )
        operation = asyncio.create_task(authority.stop(request))
        await started.wait()
        operation.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await operation
        receipt = await store.load(request)
        assert receipt is not None
        assert receipt.outcomes[0].disposition is StopDisposition.STOPPED
    finally:
        await db.close()


def test_live_endpoint_without_receipt_store_refuses_before_cancellation():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.agent_id = "did:test:agent"
    agent._active_request_ids = {"turn-7"}
    agent.cancel_current_request = MagicMock(return_value=True)
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"request_id": "turn-7"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Cooperative Stop could not be confirmed."
    agent.cancel_current_request.assert_not_called()


@pytest.mark.parametrize(
    "correlation_id", ["", "   ", 7, "\ud800", "x" * 257]
)
def test_live_endpoint_rejects_invalid_stop_correlation_identity(correlation_id):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)

    # Encode explicitly with JSON escapes so a lone surrogate reaches the
    # application parser. httpx's convenience ``json=`` path encodes with
    # ensure_ascii=False and rejects it in the client before the endpoint runs.
    response = TestClient(app).post(
        "/api/agent/stop",
        content=json.dumps(
            {"correlation_id": correlation_id}, ensure_ascii=True
        ).encode("ascii"),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Stop correlation_id must be a non-empty valid Unicode string."
    )


@pytest.mark.parametrize("request_id", ["", 0, False, None])
def test_live_endpoint_rejects_falsey_explicit_request_id(request_id):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    app.state.stop_receipt_store = _EndpointReplayStore()
    agent = MagicMock()
    agent.agent_id = "did:test:agent"
    agent._active_request_ids = {"turn-a", "turn-b"}
    agent.cancel_current_request = MagicMock(return_value=True)
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"request_id": request_id},
    )

    assert response.status_code == 400
    agent.cancel_current_request.assert_not_called()


def test_live_endpoint_stops_opaque_whitespace_request_id():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kestrel_sovereign.endpoints.agent import router

    app = FastAPI()
    app.include_router(router)
    app.state.stop_receipt_store = _EndpointReplayStore()
    agent = MagicMock()
    agent.agent_id = "did:test:agent"
    agent._active_request_ids = {" "}
    agent.cancel_current_request = MagicMock(return_value=True)
    agent.wait_for_request_completion = AsyncMock(return_value=None)
    app.state.agent = agent

    response = TestClient(app).post(
        "/api/agent/stop",
        json={"request_id": " "},
    )

    assert response.status_code == 200, response.text
    agent.cancel_current_request.assert_called_once_with(request_id=" ")


def test_live_endpoint_replays_client_stop_correlation_without_recancelling():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kestrel_sovereign.endpoints.agent import router
    app = FastAPI()
    app.include_router(router)
    app.state.stop_receipt_store = _EndpointReplayStore()
    agent = MagicMock()
    agent.agent_id = "did:test:agent"
    agent._active_request_ids = {"turn-7"}
    agent.cancel_current_request = MagicMock(return_value=True)
    agent.wait_for_request_completion = AsyncMock(return_value=None)
    app.state.agent = agent
    client = TestClient(app)
    body = {"request_id": "turn-7", "correlation_id": "stop-operation-7"}

    first = client.post("/api/agent/stop", json=body)
    retry = client.post("/api/agent/stop", json=body)

    assert first.status_code == 200
    assert retry.status_code == 200
    assert first.json()["stop_outcomes"] == retry.json()["stop_outcomes"]
    assert first.json()["stop_outcomes"][0]["receipt_id"]
    agent.cancel_current_request.assert_called_once_with(request_id="turn-7")
