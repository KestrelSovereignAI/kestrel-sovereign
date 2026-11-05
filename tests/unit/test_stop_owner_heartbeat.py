"""Stop-owner liveness survives event-loop stalls (#3889).

The owner lease is renewed by a heartbeat on its own thread and connection,
so one agent's synchronous stall on the shared loop no longer lets the lease
lapse and cancel every other agent's turn. These are the architect's three
acceptance checks, plus the heartbeat's own lifecycle.
"""

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI

import kestrel_sovereign
from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError,
    InvocationSelfFencedError,
    bind_async_invocation,
)
from kestrel_sovereign.execution_custody import (
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.stop import (
    DistributedInvocationRegistry,
    DistributedInvocationStore,
    StopDisposition,
)
from kestrel_sovereign.stop.owner_heartbeat import (
    OwnerHeartbeat,
    independent_database_opener,
)
from kestrel_sovereign.stop.receipt import STOP_OWNER_LEASE_SECONDS
from kestrel_sovereign.storage.async_database import AsyncDatabase
from tests.unit.test_distributed_stop_invocations import (
    _HEALTHY_LEASE_SECONDS,
    _SelfFencingReplicaAgent,
    _patch_heartbeat_renewal,
    _wait_for,
)

# Longer than the production lease, as the ticket requires.
_STALL_SECONDS = STOP_OWNER_LEASE_SECONDS + 0.5


async def _no_schema(_database):
    """The owner's connection already created the Stop schema."""


class _StallingAgent(_SelfFencingReplicaAgent):
    """An agent whose turn holds the shared event loop synchronously."""

    def __init__(self, agent_id: str, stall_started: threading.Event | None = None):
        super().__init__(agent_id)
        self.stall_started = stall_started or threading.Event()

    @bind_async_invocation("request_id", track_request_lifecycle=True)
    async def stall_turn(self, request_id: str, seconds: float) -> str:
        self.stall_started.set()
        time.sleep(seconds)
        return request_id


async def _owner_registry(path: Path, **kwargs):
    db = await AsyncDatabase.sqlite(str(path))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    return db, store, DistributedInvocationRegistry(store, **kwargs)


def _generation_of(registry, agent, request_id: str, generation: int = 1) -> str:
    return registry._by_local_generation[(id(agent), request_id, generation)]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_stall_longer_than_the_lease_in_one_turn_spares_another_agents_turn(
    tmp_path,
):
    """Acceptance 1: agent A's stall past the lease cannot cancel B's turn."""

    db, store, registry = await _owner_registry(tmp_path / "stall.db")
    assert registry._owner_lease_seconds == STOP_OWNER_LEASE_SECONDS
    stalling = _StallingAgent("did:test:stalling-agent")
    bystander = _SelfFencingReplicaAgent("did:test:bystander-agent")
    registry.attach(stalling)
    registry.attach(bystander)
    registry.start()
    bystander_turn = asyncio.create_task(bystander.run_turn("bystander-turn"))
    try:
        await asyncio.wait_for(bystander.operation_started.wait(), timeout=10)
        bystander_generation_id = _generation_of(
            registry, bystander, "bystander-turn"
        )

        stalled_at = time.monotonic()
        assert await stalling.stall_turn("stalling-turn", _STALL_SECONDS) == (
            "stalling-turn"
        )
        assert time.monotonic() - stalled_at >= _STALL_SECONDS
        # The heartbeat kept renewing while the loop was blocked.
        assert registry._heartbeat.latest().started > (
            stalled_at + STOP_OWNER_LEASE_SECONDS
        )

        # Let the relay act on everything it can now see.
        await asyncio.sleep(5 * registry._poll_seconds)
        assert not bystander_turn.done()
        assert registry.owner_lifecycle_status == "healthy"
        assert registry.owner_epoch == 1
        inventory = await store.read_owner(
            registry._owner_id, lease_seconds=STOP_OWNER_LEASE_SECONDS
        )
        assert bystander_generation_id in inventory.live_generation_ids

        bystander.release_operation.set()
        assert await bystander_turn == "bystander-turn"
    finally:
        bystander.release_operation.set()
        await asyncio.gather(bystander_turn, return_exceptions=True)
        await registry.close()
        await db.close()


class _HeartbeatKilled(BaseException):
    """Ends the heartbeat thread from inside a renewal, as a crash would."""


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_dead_heartbeat_lets_the_lease_expire_and_fences_within_it(
    tmp_path, monkeypatch
):
    """Acceptance 2: no heartbeat, no lease; then the registry converges."""

    path = tmp_path / "dead-heartbeat.db"
    db, store, registry = await _owner_registry(
        path, poll_seconds=0.01, owner_lease_seconds=_HEALTHY_LEASE_SECONDS
    )
    peer_db = await AsyncDatabase.sqlite(str(path))
    peer_store = DistributedInvocationStore(peer_db)
    agent = _SelfFencingReplicaAgent("did:test:dead-heartbeat")
    registry.attach(agent)
    kill = threading.Event()

    async def killable(original, store, owner_id, **kwargs):
        if kill.is_set():
            kill.clear()
            raise _HeartbeatKilled()
        return await original(store, owner_id, **kwargs)

    _patch_heartbeat_renewal(monkeypatch, killable)
    # Keep the fenced row in place, so a peer can show that it expired.
    original_settle = store.settle
    release_settle = asyncio.Event()

    async def held_settle(*args, **kwargs):
        await release_settle.wait()
        await original_settle(*args, **kwargs)

    store.settle = held_settle
    registry.start()
    turn = asyncio.create_task(agent.run_turn("orphaned-turn"))
    try:
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        generation_id = _generation_of(registry, agent, "orphaned-turn")
        await _wait_for(lambda: registry._heartbeat.latest() is not None)

        kill.set()
        await _wait_for(lambda: not registry._heartbeat.alive)
        last = registry._heartbeat.latest()
        assert last.poll is not None and generation_id in last.poll.live_generation_ids

        await _wait_for(lambda: registry._lease_lost, timeout=10)
        fenced_after = time.monotonic() - last.started
        # Never before the lease could have lapsed; within one relay interval
        # (plus scheduling slack) of it.
        assert fenced_after >= _HEALTHY_LEASE_SECONDS
        assert fenced_after < _HEALTHY_LEASE_SECONDS + 0.25
        assert "the heartbeat thread has exited" in registry.owner_fence_reason
        with pytest.raises(InvocationSelfFencedError):
            await asyncio.wait_for(turn, timeout=10)

        # The durable lease expired too: another replica may reap the row.
        await asyncio.sleep(
            max(0.0, last.finished + _HEALTHY_LEASE_SECONDS + 0.05 - time.monotonic())
        )
        assert await peer_store.reap_expired(
            (generation_id,), lease_seconds=_HEALTHY_LEASE_SECONDS
        ) == (generation_id,)

        # New work restarts the heartbeat under a fresh owner identity.
        later = _SelfFencingReplicaAgent("did:test:dead-heartbeat")
        assert await registry.register(later, "later-turn", 1)
        assert registry._heartbeat.alive
        assert registry.owner_epoch == 2
        assert registry.owner_lifecycle_status == "healthy"
        release_settle.set()
        await _wait_for(lambda: generation_id not in registry._active)
        assert await store.remaining((generation_id,)) == ()
    finally:
        release_settle.set()
        agent.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        await registry.close()
        await peer_db.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_stop_issued_during_one_agents_stall_reaches_another_agent(tmp_path):
    """Acceptance 3: a Stop for B, issued mid-stall by a peer, still lands."""

    path = tmp_path / "stop-during-stall.db"
    db, store, registry = await _owner_registry(path)
    stall_started = threading.Event()
    stalling = _StallingAgent("did:test:stalling-agent", stall_started)
    target = _SelfFencingReplicaAgent("did:test:stop-target")
    registry.attach(stalling)
    registry.attach(target)
    registry.start()
    target_turn = asyncio.create_task(target.run_turn("stopped-turn"))
    issued: dict[str, object] = {}

    def peer_replica() -> None:
        async def issue_stop() -> None:
            peer_db = await AsyncDatabase.sqlite(
                str(path), schema_initializer=_no_schema
            )
            try:
                peer = DistributedInvocationRegistry(
                    DistributedInvocationStore(peer_db)
                )
                if not stall_started.wait(30):
                    raise AssertionError("the stall never started")
                # Past the lease into the stall: a lapsed lease is reapable.
                await asyncio.sleep(STOP_OWNER_LEASE_SECONDS + 0.2)
                ticket = await peer.request_turn(target.agent_id, "stopped-turn")
                issued["at"] = time.monotonic()
                issued["disposition"] = await peer.wait_for_stop(
                    ticket, timeout_seconds=15
                )
            finally:
                await peer_db.close()

        try:
            asyncio.run(issue_stop())
        except BaseException as error:
            issued["error"] = error

    issuer = threading.Thread(target=peer_replica, name="peer-replica", daemon=True)
    try:
        await asyncio.wait_for(target.operation_started.wait(), timeout=10)
        issuer.start()
        stall_seconds = STOP_OWNER_LEASE_SECONDS + 1.5
        stalled_at = time.monotonic()
        assert await stalling.stall_turn("stalling-turn", stall_seconds) == (
            "stalling-turn"
        )
        stall_ended = time.monotonic()

        with pytest.raises(InvocationCancelledError):
            await asyncio.wait_for(target_turn, timeout=15)
        await asyncio.to_thread(issuer.join, 30)
        assert "error" not in issued, issued.get("error")
        assert stalled_at < issued["at"] < stall_ended
        assert issued["disposition"] is StopDisposition.STOPPED
        assert registry.owner_lifecycle_status == "healthy"
        assert registry.owner_epoch == 1
    finally:
        target.release_operation.set()
        await asyncio.gather(target_turn, return_exceptions=True)
        if issuer.is_alive():
            await asyncio.to_thread(issuer.join, 30)
        await registry.close()
        await db.close()


@pytest.mark.asyncio
async def test_the_heartbeat_renews_on_its_own_thread_and_connection(
    tmp_path, monkeypatch
):
    db, _store, registry = await _owner_registry(tmp_path / "own-connection.db")
    loop_thread = threading.get_ident()
    seen = []

    async def record(original, store, owner_id, **kwargs):
        seen.append((threading.get_ident(), store._db))
        return await original(store, owner_id, **kwargs)

    _patch_heartbeat_renewal(monkeypatch, record)
    try:
        registry.start()
        await _wait_for(lambda: len(seen) >= 2)
        assert all(thread != loop_thread for thread, _ in seen)
        assert all(database is not db for _, database in seen)
    finally:
        await registry.close()
        await db.close()


@pytest.mark.asyncio
async def test_close_stops_and_joins_the_heartbeat(tmp_path):
    db, _store, registry = await _owner_registry(tmp_path / "close.db")
    try:
        registry.start()
        await _wait_for(lambda: registry._heartbeat.latest() is not None)
        heartbeat = registry._heartbeat
    finally:
        await registry.close()
        await db.close()
    assert not heartbeat.alive


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_failed_attempt_after_a_stall_keeps_the_renewals_made_during_it(
    tmp_path, monkeypatch
):
    """The renewals made while the loop was blocked hold the lease, even when
    the attempt just before the loop resumed failed."""

    db, store, registry = await _owner_registry(
        tmp_path / "failed-after-stall.db",
        poll_seconds=0.02,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent("did:test:renewed-during-stall")
    registry.attach(agent)
    failing = threading.Event()

    async def failable(original, store, owner_id, **kwargs):
        if failing.is_set():
            raise ConnectionError("injected renewal failure")
        return await original(store, owner_id, **kwargs)

    _patch_heartbeat_renewal(monkeypatch, failable)
    registry.start()
    turn = asyncio.create_task(agent.run_turn("held-turn"))
    try:
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        generation_id = _generation_of(registry, agent, "held-turn")
        heartbeat = registry._heartbeat

        # Block the loop past the lease, then make the next attempts fail
        # before it runs again.
        stalled_at = time.monotonic()
        time.sleep(_HEALTHY_LEASE_SECONDS + 0.2)
        failing.set()
        deadline = time.monotonic() + 10
        while heartbeat.latest().failure is None:
            assert time.monotonic() < deadline, "no renewal attempt failed"
            time.sleep(0.002)
        renewed = heartbeat.latest_renewed()
        # The situation the fix is for: the newest attempt failed, the newest
        # success is well inside the lease, and the lease the loop folded in
        # before the stall is older than a lease.
        assert renewed.started > stalled_at + _HEALTHY_LEASE_SECONDS
        assert time.monotonic() - renewed.started < _HEALTHY_LEASE_SECONDS / 2
        assert generation_id in renewed.poll.live_generation_ids
        assert time.monotonic() - registry._renewed_at[generation_id] >= (
            _HEALTHY_LEASE_SECONDS
        )

        registry._observe_owner_heartbeat()
        assert not registry._lease_lost, registry.owner_fence_reason
        assert registry._renewed_at[generation_id] == renewed.started
        await asyncio.sleep(0.1)
        assert registry.owner_lifecycle_status == "healthy"
        assert not turn.done()

        failing.clear()
        await _wait_for(lambda: heartbeat.latest().failure is None)
        agent.release_operation.set()
        assert await turn == "held-turn"
        assert registry.owner_epoch == 1
    finally:
        failing.clear()
        agent.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        await registry.close()
        await db.close()


def _hold_heartbeat_renewal(monkeypatch):
    """Hold the heartbeat inside its next renewal until ``release`` is set.

    The renewal has written its heartbeat over the heartbeat's own
    connection, which stays open while it is held.
    """

    hold, entered, release = threading.Event(), threading.Event(), threading.Event()

    async def holdable(original, store, owner_id, **kwargs):
        polled = await original(store, owner_id, **kwargs)
        if hold.is_set():
            hold.clear()
            entered.set()
            await asyncio.to_thread(release.wait, 30)
        return polled

    _patch_heartbeat_renewal(monkeypatch, holdable)
    return hold, entered, release


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_close_reports_a_heartbeat_that_did_not_exit_and_keeps_it(
    tmp_path, monkeypatch
):
    from kestrel_sovereign.stop import invocation

    monkeypatch.setattr(invocation, "_DEFAULT_CLOSE_DRAIN_SECONDS", 0.02)
    db, _store, registry = await _owner_registry(tmp_path / "held-heartbeat.db")
    hold, entered, release = _hold_heartbeat_renewal(monkeypatch)
    try:
        registry.start()
        heartbeat = registry._heartbeat
        hold.set()
        assert await asyncio.to_thread(entered.wait, 10)

        with pytest.raises(TimeoutError) as caught:
            await registry.close()
        assert "heartbeat has not retired: its thread has not exited" in str(
            caught.value
        )
        assert is_execution_control_error(caught.value)
        assert any(
            resource is heartbeat and owner is heartbeat._thread
            for resource, owner in execution_cleanup_debts(caught.value)
        )
        assert heartbeat.alive
        assert not await heartbeat.stop(timeout=0.05)

        # Once it has exited and closed its connection, close completes.
        release.set()
        await registry.close()
        assert not heartbeat.alive
        assert heartbeat.retirement_error() is None
    finally:
        release.set()
        try:
            await registry.close()
        finally:
            await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_server_shutdown_keeps_the_stop_database_while_the_heartbeat_runs(
    tmp_path, monkeypatch
):
    """Shutdown neither drops the registry nor closes its database under a
    heartbeat that may still hold a transaction."""

    from kestrel_sovereign import server
    from kestrel_sovereign.stop import invocation

    async def nothing_to_release(_app, **_collectors):
        return None

    monkeypatch.setattr(server, "_shutdown_host_features", nothing_to_release)
    monkeypatch.setattr(server, "_shutdown_phoenix", nothing_to_release)
    monkeypatch.setattr(invocation, "_DEFAULT_CLOSE_DRAIN_SECONDS", 0.02)
    db, _store, registry = await _owner_registry(tmp_path / "server-shutdown.db")
    hold, entered, release = _hold_heartbeat_renewal(monkeypatch)
    app = FastAPI()
    app.state.distributed_invocation_registry = registry
    app.state.stop_receipt_db = db
    try:
        registry.start()
        heartbeat = registry._heartbeat
        hold.set()
        assert await asyncio.to_thread(entered.wait, 10)

        cancelled, failure = await server._shutdown_server_resources(app)
        assert not cancelled
        assert failure is not None and is_execution_control_error(failure)
        assert any(
            resource is heartbeat
            for resource, _owner in execution_cleanup_debts(failure)
        )
        assert heartbeat.alive
        assert app.state.distributed_invocation_registry is registry
        assert app.state.stop_receipt_db is db
        assert [tuple(row) for row in await db.fetchall("SELECT 1")] == [(1,)]

        # A later shutdown, once the heartbeat has exited, retires both.
        release.set()
        await asyncio.to_thread(heartbeat._thread.join, 10)
        cancelled, failure = await server._shutdown_server_resources(app)
        assert not cancelled and failure is None
        assert app.state.distributed_invocation_registry is None
        assert app.state.stop_receipt_db is None
        assert not heartbeat.alive
    finally:
        release.set()
        if app.state.distributed_invocation_registry is not None:
            await registry.close()
        if app.state.stop_receipt_db is not None:
            await db.close()


def test_a_connection_that_did_not_close_stays_owned(tmp_path):
    """A heartbeat whose connection's close failed has not retired."""

    class _CloseFailed(Exception):
        pass

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def opener():
        yield object()
        raise _CloseFailed("connection close failed")

    async def renew(_database, _owner_id):
        return None

    async def run() -> None:
        heartbeat = OwnerHeartbeat(opener, renew, interval_seconds=0.01)
        heartbeat.start("owner-with-an-unclosed-connection")
        await _wait_for(lambda: heartbeat.latest() is not None)
        assert not await heartbeat.stop(timeout=10)
        assert not heartbeat.alive
        error = heartbeat.retirement_error()
        assert error is not None
        assert "its database connection did not close (_CloseFailed)" in str(error)
        assert execution_cleanup_debts(error) == ((heartbeat, heartbeat._thread),)

    asyncio.run(run())


def test_an_in_memory_sqlite_store_has_no_second_connection():
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    database = AsyncDatabase(SQLiteBackend(":memory:"))
    with pytest.raises(ValueError, match="in-memory SQLite"):
        independent_database_opener(database)


def test_postgres_heartbeat_reaches_the_store_by_the_advisory_recipe():
    from kestrel_sovereign.storage.db.postgres import PostgresBackend

    dsn = "postgresql://kestrel:secret@db.example:5432/kestrel"
    backend = PostgresBackend(dsn=dsn)
    assert backend.independent_connection_recipe() == ((dsn,), {})

    by_parameters = PostgresBackend(
        host="db.example", database="kestrel", user="kestrel", password="secret"
    )
    assert by_parameters.independent_connection_recipe() == (
        (),
        {
            "host": "db.example",
            "port": 5432,
            "database": "kestrel",
            "user": "kestrel",
            "password": "secret",
        },
    )


@pytest.mark.asyncio
async def test_a_stopped_heartbeat_is_not_restarted(tmp_path):
    db, _store, registry = await _owner_registry(tmp_path / "stopped.db")
    try:
        registry.start()
        heartbeat: OwnerHeartbeat = registry._heartbeat
        assert await heartbeat.stop(timeout=5)
        heartbeat.ensure_running()
        assert not heartbeat.alive
    finally:
        await registry.close()
        await db.close()


def test_the_heartbeat_stops_once_the_thread_that_started_it_ends(tmp_path):
    """No registry loop, no renewals: its work cannot progress anyway."""

    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    opener = independent_database_opener(
        AsyncDatabase(SQLiteBackend(str(tmp_path / "host-thread.db")))
    )
    renewed = threading.Event()

    async def renew(_database, _owner_id):
        renewed.set()

    heartbeat = OwnerHeartbeat(opener, renew, interval_seconds=0.01)
    started = threading.Event()

    def host() -> None:
        heartbeat.start("owner-of-an-ended-thread")
        started.set()
        assert renewed.wait(10)

    host_thread = threading.Thread(target=host)
    host_thread.start()
    host_thread.join(10)
    assert started.is_set()
    deadline = time.monotonic() + 10
    while heartbeat.alive and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not heartbeat.alive


def test_the_heartbeat_thread_exits_with_its_process(tmp_path):
    """A process that never closes its registry still exits, and its lease
    then expires: the heartbeat thread cannot keep a dead process's work."""

    path = tmp_path / "process-exit.db"
    child = textwrap.dedent(
        f"""
        import asyncio
        from kestrel_sovereign.storage.async_database import AsyncDatabase
        from kestrel_sovereign.stop import (
            DistributedInvocationRegistry, DistributedInvocationStore,
        )

        class Agent:
            agent_id = "did:test:exiting-process"

        async def main():
            db = await AsyncDatabase.sqlite({str(path)!r})
            store = DistributedInvocationStore(db)
            await store.ensure_schema()
            registry = DistributedInvocationRegistry(
                store, poll_seconds=0.01, owner_lease_seconds={_HEALTHY_LEASE_SECONDS}
            )
            registry.start()
            assert await registry.register(Agent(), "abandoned-turn", 1)
            await asyncio.sleep(0.2)
            print(registry._owner_id, flush=True)
            # Never closes the registry, so never stops its heartbeat.
            await db.close()

        asyncio.run(main())
        """
    )
    package_root = str(Path(kestrel_sovereign.__file__).resolve().parents[1])
    environment = {**os.environ, "PYTHONPATH": package_root}
    finished = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
        cwd=str(tmp_path),
    )
    assert finished.returncode == 0, finished.stderr[-2000:]
    owner_id = finished.stdout.strip().splitlines()[-1]

    async def expired() -> tuple[str, ...]:
        db = await AsyncDatabase.sqlite(str(path))
        try:
            store = DistributedInvocationStore(db)
            await asyncio.sleep(_HEALTHY_LEASE_SECONDS + 0.1)
            live = await store.read_owner(
                owner_id, lease_seconds=_HEALTHY_LEASE_SECONDS
            )
            assert live.live_generation_ids == ()
            rows = await db.fetchall(
                "SELECT generation_id FROM stop_active_invocations "
                "WHERE owner_id = ?",
                (owner_id,),
            )
            return await store.reap_expired(
                tuple(str(row[0]) for row in rows),
                lease_seconds=_HEALTHY_LEASE_SECONDS,
            )
        finally:
            await db.close()

    assert len(asyncio.run(expired())) == 1


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.timeout(60)
async def test_the_heartbeat_keeps_the_lease_through_a_stall_on_either_backend(
    db_backend,
):
    """The heartbeat opens its own connection to SQLite and PostgreSQL alike."""

    store = DistributedInvocationStore(AsyncDatabase(db_backend))
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(
        store, poll_seconds=0.02, owner_lease_seconds=_HEALTHY_LEASE_SECONDS
    )
    agent = _SelfFencingReplicaAgent(f"did:test:backend-{uuid4().hex}")
    registry.attach(agent)
    registry.start()
    turn = asyncio.create_task(agent.run_turn("held-turn"))
    try:
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        generation_id = _generation_of(registry, agent, "held-turn")
        time.sleep(_HEALTHY_LEASE_SECONDS + 0.4)
        await asyncio.sleep(0.1)
        assert registry.owner_lifecycle_status == "healthy"
        assert not turn.done()
        inventory = await store.read_owner(
            registry._owner_id, lease_seconds=_HEALTHY_LEASE_SECONDS
        )
        assert generation_id in inventory.live_generation_ids
        agent.release_operation.set()
        assert await turn == "held-turn"
    finally:
        agent.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        await registry.close()
