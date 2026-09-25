"""Fleet circuit breaker for repeated peer Stop (#3170)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.endpoints.host_stop import router as host_stop_router
from kestrel_sovereign.signals.sources import peer_stop
from kestrel_sovereign.signals.sources.peer_stop import (
    PEER_STOP_CIRCUIT_OPEN,
    PEER_STOP_CIRCUIT_THRESHOLD,
    PEER_STOP_CIRCUIT_THRESHOLD_ENV,
    PEER_STOP_CIRCUIT_UNAVAILABLE,
    PEER_STOP_CIRCUIT_WINDOW_ENV,
    PEER_STOP_CIRCUIT_WINDOW_SECONDS,
    dispatch_peer_stop,
    peer_stop_breaker_refusal,
    peer_stop_request,
    resolve_peer_stop_circuit_policy,
)
from kestrel_sovereign.stop import (
    CancellationAuthority,
    CooperativeStopTarget,
    PeerStopCircuitEventKind,
    PeerStopCircuitPolicy,
    PeerStopCircuitStore,
    StopCleanupRegistry,
    StopDisposition,
    StopDoor,
    StopOutcome,
    StopReceiptStore,
    StopRequest,
    StopScope,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase

# The real dispatcher + receipt-store rail, shared as the ``rail`` fixture.
from tests.unit.test_peer_stop_signals import (  # noqa: F401
    TARGET_DID,
    _intent,
    rail,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)


class _Clock:
    """A settable database clock; ``barrier`` forces two writers to meet."""

    def __init__(self, now: datetime = T0) -> None:
        self.now = now
        self.barrier: asyncio.Barrier | None = None
        self.barrier_timeout = 0.5

    async def __call__(self, _db) -> datetime:
        if self.barrier is not None:
            try:
                await asyncio.wait_for(self.barrier.wait(), self.barrier_timeout)
            except (TimeoutError, asyncio.BrokenBarrierError):
                # Serialized correctly: the other writer is blocked behind this
                # transaction's lock and cannot reach the clock. Break the
                # barrier so it passes straight through once it does.
                await self.barrier.abort()
        return self.now


def _peer_request(target: str, actor: str, correlation: str) -> StopRequest:
    # Operation ids are unique database-wide; binding them to the target keeps
    # reruns against one PostgreSQL database from colliding with old receipts.
    suffix = hashlib.sha256(target.encode()).hexdigest()[:12]
    return peer_stop_request(
        target_agent_id=target,
        actor_id=actor,
        intent=_intent(correlation_id=f"{correlation}-{suffix}"),
    )


def _outcome(request: StopRequest, disposition: StopDisposition) -> StopOutcome:
    return StopOutcome(
        scope=request.scope,
        requested_target=request.target,
        resolved_target=request.target_agent_id,
        agent_id=request.target_agent_id,
        disposition=disposition,
        correlation_id=request.correlation_id,
    )


async def _stores(db, clock, *, threshold=3, window_seconds=60):
    receipts = StopReceiptStore(db)
    await receipts.ensure_schema()
    circuit = PeerStopCircuitStore(
        db,
        policy=PeerStopCircuitPolicy(threshold=threshold, window_seconds=window_seconds),
        clock=clock,
    )
    await circuit.ensure_schema()
    return receipts, circuit


async def _honor(circuit, receipts, target, actor, correlation):
    """Admit one peer Stop and receipt it as having stopped work."""

    request = _peer_request(target, actor, correlation)
    decision = await circuit.admit(request)
    if decision.honored:
        await receipts.persist(
            request,
            (_outcome(request, StopDisposition.STOPPED),),
            door=StopDoor.PEER,
        )
    return decision


def _target() -> str:
    # The PostgreSQL leg shares one database across tests; a fresh target per
    # test keeps each circuit's history its own.
    return f"did:test:circuit-{uuid4().hex}"


# ---------------------------------------------------------------------------
# Store semantics on both backends
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_threshold_opens_refuses_and_is_receipted(db_backend, caplog):
    db = AsyncDatabase(db_backend)
    clock = _Clock()
    receipts, circuit = await _stores(db, clock, threshold=3)
    target = _target()

    with caplog.at_level(logging.WARNING, logger="kestrel_sovereign.stop.circuit"):
        decisions = [
            await _honor(circuit, receipts, target, f"did:test:peer-{i}", f"op-{i}")
            for i in range(4)
        ]

    assert [d.honored for d in decisions] == [True, True, True, False]
    # The circuit opened on the admission that reached the threshold.
    assert decisions[2].opened is not None
    assert decisions[2].opened.admitted_count == 3
    assert decisions[3].opened is None
    opened_logs = [r for r in caplog.records if "circuit OPEN" in r.getMessage()]
    assert len(opened_logs) == 1 and target in opened_logs[0].getMessage()

    (open_circuit,) = [
        c for c in await circuit.open_circuits() if c.target_agent_id == target
    ]
    assert open_circuit.admitted_count == 3
    assert open_circuit.opened_event_id == decisions[2].opened.event_id
    events = await circuit.list_events(target_agent_id=target)
    assert [e.kind for e in events] == [PeerStopCircuitEventKind.OPENED]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_count_survives_a_restart(db_backend):
    db = AsyncDatabase(db_backend)
    clock = _Clock()
    receipts, circuit = await _stores(db, clock, threshold=2)
    target = _target()
    for i in range(2):
        assert (await _honor(circuit, receipts, target, "did:test:peer", f"r{i}")).honored

    # A new worker: fresh store objects, same durable database.
    _receipts, restarted = await _stores(db, clock, threshold=2)
    decision = await restarted.admit(_peer_request(target, "did:test:peer", "r9"))
    assert decision.honored is False
    assert decision.admitted_count == 2


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_window_boundary_closes_the_circuit_automatically(db_backend):
    db = AsyncDatabase(db_backend)
    clock = _Clock()
    receipts, circuit = await _stores(db, clock, threshold=2, window_seconds=60)
    target = _target()
    await _honor(circuit, receipts, target, "did:test:a", "w0")
    clock.now = T0 + timedelta(seconds=10)
    await _honor(circuit, receipts, target, "did:test:b", "w1")

    # One millisecond before the first admission leaves the window: still open.
    clock.now = T0 + timedelta(seconds=60) - timedelta(milliseconds=1)
    assert not (await circuit.admit(_peer_request(target, "did:test:c", "w2"))).honored
    # Exactly W after it: the first admission is out, the count is 1 < 2.
    clock.now = T0 + timedelta(seconds=60)
    decision = await circuit.admit(_peer_request(target, "did:test:c", "w3"))
    assert decision.honored is True
    assert decision.closed is not None
    assert decision.closed.admitted_count == 1
    # Honoring it reached the threshold again: the transition is receipted too.
    assert decision.opened is not None

    kinds = [e.kind for e in await circuit.list_events(target_agent_id=target)]
    assert kinds.count(PeerStopCircuitEventKind.CLOSED) == 1
    assert kinds.count(PeerStopCircuitEventKind.OPENED) == 2


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_status_read_receipts_recovery_nobody_observed(db_backend):
    db = AsyncDatabase(db_backend)
    clock = _Clock()
    receipts, circuit = await _stores(db, clock, threshold=1, window_seconds=30)
    target = _target()
    await _honor(circuit, receipts, target, "did:test:a", "s0")
    assert target in {c.target_agent_id for c in await circuit.open_circuits()}

    clock.now = T0 + timedelta(seconds=31)
    assert target not in {c.target_agent_id for c in await circuit.open_circuits()}
    kinds = [e.kind for e in await circuit.list_events(target_agent_id=target)]
    assert kinds == [PeerStopCircuitEventKind.CLOSED, PeerStopCircuitEventKind.OPENED]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_only_stops_that_stopped_work_count(db_backend):
    db = AsyncDatabase(db_backend)
    clock = _Clock()
    receipts, circuit = await _stores(db, clock, threshold=2)
    target = _target()

    idle = _peer_request(target, "did:test:a", "idle")
    assert (await circuit.admit(idle)).honored
    await receipts.persist(
        idle,
        (_outcome(idle, StopDisposition.ALREADY_COMPLETE),),
        door=StopDoor.PEER,
    )
    # An admitted Stop still in flight (no receipt yet) counts.
    in_flight = _peer_request(target, "did:test:b", "in-flight")
    assert (await circuit.admit(in_flight)).admitted_count == 1
    decision = await circuit.admit(_peer_request(target, "did:test:c", "third"))
    assert decision.honored is True
    assert decision.admitted_count == 2


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_a_retry_of_one_operation_is_counted_once(db_backend):
    db = AsyncDatabase(db_backend)
    clock = _Clock()
    _receipts, circuit = await _stores(db, clock, threshold=2)
    target = _target()
    request = _peer_request(target, "did:test:a", "same")

    first = await circuit.admit(request)
    retry = await circuit.admit(request)
    assert first.honored and retry.honored
    assert retry.admitted_count == 1
    assert (await circuit.admit(_peer_request(target, "did:test:a", "other"))).honored


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_count_binds_the_verified_actor_and_target(db_backend):
    db = AsyncDatabase(db_backend)
    clock = _Clock()
    _receipts, circuit = await _stores(db, clock, threshold=5)
    target, other = _target(), _target()

    await circuit.admit(_peer_request(target, "did:test:verified", "bind"))
    rows = await db.fetchall(
        "SELECT target_agent_id, actor_id FROM stop_circuit_admissions "
        "WHERE target_agent_id IN (?, ?)",
        (target, other),
    )
    assert [tuple(row) for row in rows] == [(target, "did:test:verified")]
    # Per target: another agent's circuit is untouched.
    assert (await circuit.admit(_peer_request(other, "did:test:x", "bind"))).admitted_count == 1


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_sovereign_reset_is_receipted_and_restarts_the_count(db_backend):
    db = AsyncDatabase(db_backend)
    clock = _Clock()
    receipts, circuit = await _stores(db, clock, threshold=1)
    target = _target()
    await _honor(circuit, receipts, target, "did:test:a", "x0")
    assert not (await circuit.admit(_peer_request(target, "did:test:a", "x1"))).honored

    event = await circuit.reset(target, actor_id="sovereign-key", reason="checked")
    assert event.kind is PeerStopCircuitEventKind.RESET
    assert event.actor_id == "sovereign-key" and event.reason == "checked"
    assert event.admitted_count == 1
    assert target not in {c.target_agent_id for c in await circuit.open_circuits()}
    assert (await circuit.admit(_peer_request(target, "did:test:a", "x2"))).honored


async def _second_worker(db_backend) -> tuple[AsyncDatabase, object]:
    """A second, independently connected worker on the same database."""

    if hasattr(db_backend, "db_path"):
        from kestrel_sovereign.storage.db import SQLiteBackend

        backend = SQLiteBackend(db_backend.db_path)
    else:
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        backend = PostgresBackend(
            os.environ.get("TEST_POSTGRES_URL")
            or os.environ.get("KESTREL_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
        )
    await backend.connect()
    return AsyncDatabase(backend), backend


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_two_peers_at_threshold_minus_one_cannot_both_be_honored(db_backend):
    db = AsyncDatabase(db_backend)
    clock = _Clock()
    receipts, circuit = await _stores(db, clock, threshold=3)
    target = _target()
    for i in range(2):
        await _honor(circuit, receipts, target, f"did:test:p{i}", f"c{i}")

    second_db, second_backend = await _second_worker(db_backend)
    try:
        second = PeerStopCircuitStore(second_db, policy=circuit.policy, clock=clock)
        # Both writers must reach the window count together unless the
        # per-target serialization holds one back.
        clock.barrier = asyncio.Barrier(2)
        decisions = await asyncio.gather(
            circuit.admit(_peer_request(target, "did:test:racer-a", "race-a")),
            second.admit(_peer_request(target, "did:test:racer-b", "race-b")),
        )
    finally:
        await second_backend.close()

    assert sorted(d.honored for d in decisions) == [False, True]
    count = await db.fetchval(
        "SELECT COUNT(*) FROM stop_circuit_admissions WHERE target_agent_id = ?",
        (target,),
    )
    assert int(count) == 3


# ---------------------------------------------------------------------------
# The peer rail
# ---------------------------------------------------------------------------


def _small_circuit(rail, threshold: int) -> PeerStopCircuitStore:
    circuit = PeerStopCircuitStore(
        rail.receipt_db,
        policy=PeerStopCircuitPolicy(threshold=threshold, window_seconds=900),
    )
    rail.agent.__dict__[peer_stop._CIRCUIT_ATTRIBUTE] = circuit
    return circuit


@pytest.mark.asyncio
async def test_open_circuit_refuses_peers_fleet_wide_with_an_operation_receipt(rail):
    _small_circuit(rail, threshold=2)
    for i, peer in enumerate(("did:test:peer-a", "did:test:peer-b")):
        rail.agent._active_request_ids.add(f"turn-{i}")
        response = await dispatch_peer_stop(
            rail.agent, actor_id=peer, intent=_intent(correlation_id=f"cb-{i}")
        )
        assert response["stop_outcomes"][0]["disposition"] == "stopped", response

    rail.agent._active_request_ids.add("turn-3")
    refused = await dispatch_peer_stop(
        rail.agent, actor_id="did:test:peer-c", intent=_intent(correlation_id="cb-3")
    )

    (outcome,) = refused["stop_outcomes"]
    assert outcome["disposition"] == "refused"
    assert outcome["detail"] == PEER_STOP_CIRCUIT_OPEN
    # The executor's definitive decision about that operation: operation-keyed.
    assert refused["receipt_kind"] == "operation"
    assert refused["recorded"] is True
    assert "turn-3" in rail.agent._active_request_ids
    page = await rail.receipts.list_receipts(limit=100)
    assert {record.door for record in page.receipts} == {"peer"}


@pytest.mark.asyncio
async def test_operator_stops_are_never_counted(rail):
    circuit = _small_circuit(rail, threshold=1)
    target = peer_stop.peer_stop_target_identity(rail.agent)

    async def cancel(_request):
        return StopDisposition.STOPPED

    authority = CancellationAuthority(
        lambda: (CooperativeStopTarget(target, target, cancel),),
        cleanup_registry=StopCleanupRegistry(),
        receipt_store=rail.receipts,
        door=StopDoor.AGENT,
    )
    for i in range(5):
        outcomes = await authority.stop(
            StopRequest(
                StopScope.AGENT,
                "local-operator:kite",
                target=target,
                cascade=False,
                correlation_id=f"operator-{i}",
            )
        )
        assert outcomes[0].disposition is StopDisposition.STOPPED

    decision = await circuit.admit(_peer_request(target, "did:test:peer", "after"))
    assert decision.honored is True
    assert decision.admitted_count == 1
    page = await rail.receipts.list_receipts(limit=100)
    assert {record.door for record in page.receipts} == {"agent"}


@pytest.mark.asyncio
async def test_breaker_without_a_durable_circuit_refuses_rather_than_honors():
    agent = SimpleNamespace()
    request = _peer_request(TARGET_DID, "did:test:peer", "none")
    assert await peer_stop_breaker_refusal(agent, request) == PEER_STOP_CIRCUIT_UNAVAILABLE


@pytest.mark.asyncio
async def test_breaker_failure_refuses_rather_than_honors(rail):
    circuit = _small_circuit(rail, threshold=5)

    async def broken(_request):
        raise RuntimeError("database gone")

    circuit.admit = broken
    request = _peer_request(TARGET_DID, "did:test:peer", "broken")
    assert await peer_stop_breaker_refusal(rail.agent, request) == PEER_STOP_CIRCUIT_UNAVAILABLE


# ---------------------------------------------------------------------------
# Host doors
# ---------------------------------------------------------------------------


@pytest.fixture
async def host_app(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "host.db"))
    clock = _Clock()
    receipts, circuit = await _stores(db, clock, threshold=1)
    app = FastAPI()
    app.include_router(host_stop_router)
    manager = MagicMock()
    manager.list_agents.return_value = {}
    app.state.agent_manager = manager
    app.state.stop_receipt_store = receipts
    app.state.peer_stop_circuit = circuit
    app.state.caller = CallerContext.sovereign(identity="sovereign-key")

    @app.middleware("http")
    async def bind_caller(request: Request, call_next):
        request.state.caller = request.app.state.caller
        return await call_next(request)

    yield SimpleNamespace(app=app, circuit=circuit, receipts=receipts, clock=clock)
    await db.close()


@pytest.mark.asyncio
async def test_status_shows_open_circuits_to_the_sovereign_only(host_app):
    await _honor(host_app.circuit, host_app.receipts, TARGET_DID, "did:test:a", "h0")
    client = TestClient(host_app.app)

    status = client.get("/api/host/stop/status").json()
    circuit = status["peer_stop_circuit"]
    assert circuit["available"] is True
    assert [c["target_agent_id"] for c in circuit["open"]] == [TARGET_DID]
    assert circuit["open"][0]["admitted_count"] == 1

    host_app.app.state.caller = CallerContext.authenticated("operator@example.test")
    assert "peer_stop_circuit" not in client.get("/api/host/stop/status").json()


@pytest.mark.asyncio
async def test_reset_door_is_sovereign_only_and_receipted(host_app):
    await _honor(host_app.circuit, host_app.receipts, TARGET_DID, "did:test:a", "h1")
    client = TestClient(host_app.app)

    host_app.app.state.caller = CallerContext.authenticated("operator@example.test")
    denied = client.post(
        "/api/host/stop/circuit/reset",
        json={"target": TARGET_DID, "reason": "not mine"},
    )
    assert denied.status_code == 403
    assert client.get("/api/host/stop/circuit/events").status_code == 403

    host_app.app.state.caller = CallerContext.sovereign(identity="sovereign-key")
    reset = client.post(
        "/api/host/stop/circuit/reset",
        json={"target": TARGET_DID, "reason": "reviewed the peers"},
    )
    assert reset.status_code == 200, reset.text
    assert reset.json()["event"]["kind"] == "reset"
    assert reset.json()["event"]["actor_id"] == "sovereign-key"
    assert client.get("/api/host/stop/status").json()["peer_stop_circuit"]["open"] == []
    kinds = [
        e["kind"]
        for e in client.get(
            "/api/host/stop/circuit/events", params={"target": TARGET_DID}
        ).json()["events"]
    ]
    assert kinds == ["reset", "opened"]


# ---------------------------------------------------------------------------
# Configuration, census, and invariants
# ---------------------------------------------------------------------------


def test_policy_defaults_and_environment_overrides() -> None:
    default = resolve_peer_stop_circuit_policy({})
    assert (default.threshold, default.window_seconds) == (
        PEER_STOP_CIRCUIT_THRESHOLD,
        PEER_STOP_CIRCUIT_WINDOW_SECONDS,
    )
    tuned = resolve_peer_stop_circuit_policy(
        {PEER_STOP_CIRCUIT_THRESHOLD_ENV: "3", PEER_STOP_CIRCUIT_WINDOW_ENV: "60"}
    )
    assert (tuned.threshold, tuned.window_seconds) == (3, 60)
    for bad in ("0", "-1", "many"):
        with pytest.raises(ValueError):
            resolve_peer_stop_circuit_policy({PEER_STOP_CIRCUIT_THRESHOLD_ENV: bad})


def _production_sources() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): path.read_text(encoding="utf-8")
        for path in (REPO_ROOT / "kestrel_sovereign").rglob("*.py")
    }


def _calls(source: str, pattern: str) -> list[str]:
    """Every call matching ``pattern`` with its full argument text."""

    calls = []
    for match in re.finditer(pattern, source):
        depth, index = 0, match.end() - 1
        for index in range(match.end() - 1, len(source)):
            if source[index] == "(":
                depth += 1
            elif source[index] == ")":
                depth -= 1
                if depth == 0:
                    break
        calls.append(source[match.start() : index + 1])
    return calls


def test_every_receipt_writer_names_its_door() -> None:
    """Census: no Stop receipt is written without recorded provenance."""

    persist_calls = []
    authorities = []
    for path, source in _production_sources().items():
        if path.endswith("stop/receipt.py"):
            continue
        persist_calls += [(path, c) for c in _calls(source, r"_store\.persist\(")]
        persist_calls += [(path, c) for c in _calls(source, r"receipt_store\.persist\(")]
        authorities += [
            (path, c) for c in _calls(source, r"(?<![\w.])CancellationAuthority\(")
            if "class CancellationAuthority" not in c
        ]
    assert persist_calls and authorities
    for path, call in persist_calls + authorities:
        # The authority forwards the door it was constructed for.
        forwards = path.endswith("stop/authority.py") and "door=self._door" in call
        assert "door=StopDoor." in call or forwards, (path, call)
    doors = {
        re.search(r"door=StopDoor\.(\w+)", call).group(1) for _path, call in authorities
    }
    assert doors == {"AGENT", "HOST", "PEER"}


def test_only_the_peer_rail_admits_and_nothing_here_holds() -> None:
    sources = _production_sources()
    # Anything that can reach the circuit store and call ``admit`` on it.
    admitters = {
        path
        for path, source in sources.items()
        if re.search(r"circuit\.admit\(", source)
        or ("PeerStopCircuitStore" in source and re.search(r"\.admit\(", source))
    }
    assert admitters == {"kestrel_sovereign/signals/sources/peer_stop.py"}
    for path in (
        "kestrel_sovereign/stop/circuit.py",
        "kestrel_sovereign/signals/sources/peer_stop.py",
    ):
        assert "kestrel_sovereign.hold" not in sources[path], path
        assert "set_hold" not in sources[path], path


@pytest.mark.asyncio
async def test_legacy_receipts_gain_a_backfilled_door(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "legacy.db"))
    try:
        receipts = StopReceiptStore(db)
        await receipts.ensure_schema()
        request = StopRequest(
            StopScope.AGENT, "did:test:peer", target="did:test:t", cascade=False
        )
        await receipts.persist(
            request,
            (
                StopOutcome(
                    scope=request.scope,
                    requested_target=request.target,
                    resolved_target=request.target,
                    agent_id=request.target,
                    disposition=StopDisposition.STOPPED,
                    correlation_id=request.correlation_id,
                ),
            ),
            door=StopDoor.PEER,
        )
        # Rebuild the pre-#3170 shape: drop the column, keep the row.
        await db.execute("ALTER TABLE stop_receipts DROP COLUMN door")
        await db.execute(
            "UPDATE stop_receipts SET actor_id = 'api_key', scope = 'agent'"
        )
        await receipts.ensure_schema()
        (record,) = (await receipts.list_receipts(limit=10)).receipts
        assert record.door == "agent"
    finally:
        await db.close()
