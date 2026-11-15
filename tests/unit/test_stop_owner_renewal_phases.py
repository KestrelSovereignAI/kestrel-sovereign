"""A Stop owner renewal names where its time went (#4081).

On 2026-11-13 the host's Distributed Stop owner fenced twice with a healthy
event loop: "its renewal has been in flight for 2.0 s". These checks pin the
instrumentation that says what held that renewal back the next time: the
phase the attempt is in, the phases of a slow attempt, and how long the
registry had gone without observing the heartbeat.
"""

import asyncio
import logging
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from kestrel_sovereign.stop import (
    DistributedInvocationRegistry,
    DistributedInvocationStore,
)
from kestrel_sovereign.stop import invocation as invocation_module
from kestrel_sovereign.stop.owner_heartbeat import (
    PHASE_OPEN,
    PHASE_SLOT_WAIT,
    PHASE_WRITE,
    THREAD_NAME,
    OwnerHeartbeat,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_distributed_stop_invocations import (
    _HEALTHY_LEASE_SECONDS,
    _ReplicaAgent,
    _first_owner_fence,
    _wait_for,
)

_POLL_SECONDS = 0.02

# "<flight> s, <seconds> s of it in <phase>": the phase of the attempt in flight.
_IN_PHASE = r"in flight for [0-9.]+ s, ([0-9.]+) s of it in {phase}\b"


async def _registry(path):
    db = await AsyncDatabase.sqlite(str(path))
    store = DistributedInvocationStore(db)
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(
        store, poll_seconds=_POLL_SECONDS, owner_lease_seconds=_HEALTHY_LEASE_SECONDS
    )
    return db, registry


async def _admit_and_start(registry, agent_id: str, turn_id: str) -> str:
    """Admit one generation, start the heartbeat and wait for its first renewal."""

    agent = _ReplicaAgent(agent_id)
    assert await registry.register(agent, turn_id, 1)
    registry.start()
    await _wait_for(lambda: registry._heartbeat.latest_renewed() is not None)
    return registry._by_local_generation[(id(agent), turn_id, 1)]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_fence_reason_names_slot_wait(tmp_path, caplog):
    """A second connection holding SQLite's writer slot past the lease fences
    the owner with its renewal named as waiting in ``slot_wait``."""

    caplog.set_level(logging.WARNING)
    path = tmp_path / "slot-wait.db"
    db, registry = await _registry(path)
    holder = None
    try:
        await _admit_and_start(registry, "did:test:slot-wait", "slot-wait-turn")
        holder = sqlite3.connect(
            str(path), isolation_level=None, timeout=10, check_same_thread=False
        )
        holder.execute("BEGIN IMMEDIATE")

        message = await _first_owner_fence(caplog)
        assert re.search(_IN_PHASE.format(phase=PHASE_SLOT_WAIT), message), message

        # Re-establishment waits a lease for the heartbeat, which is still
        # waiting for the slot: by then it has waited a lease or more.
        await _wait_for(lambda: registry._reacquisition_failure is not None)
        reason = registry.owner_fence_reason
        waited = re.search(
            "re-establishment failed: .*" + _IN_PHASE.format(phase=PHASE_SLOT_WAIT),
            reason,
        )
        assert waited is not None, reason
        assert float(waited.group(1)) >= _HEALTHY_LEASE_SECONDS, reason

        # Once the slot is free the attempt finishes, and logs its phases.
        holder.rollback()
        holder.close()
        holder = None
        phases = re.compile(
            r"Its phases: slot_wait ([0-9.]+) s, write [0-9.]+ s, "
            r"inventory [0-9.]+ s, commit [0-9.]+ s$"
        )

        def waited_a_lease_for_the_slot() -> bool:
            return any(
                (match := phases.search(warning)) is not None
                and float(match.group(1)) >= _HEALTHY_LEASE_SECONDS
                for warning in _slow_renewal_warnings(caplog)
            )

        await _wait_for(waited_a_lease_for_the_slot)
    finally:
        if holder is not None:
            holder.rollback()
            holder.close()
        await registry.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_fence_reason_names_commit(tmp_path, caplog, monkeypatch):
    """A renewal whose COMMIT blocks is named as in ``commit``, not in the
    write or inventory read that finished before it."""

    caplog.set_level(logging.WARNING)
    block, release = threading.Event(), threading.Event()
    original_commit = SQLiteBackend._commit_write

    async def blocking_commit(self, conn, **kwargs):
        if block.is_set() and threading.current_thread().name == THREAD_NAME:
            await asyncio.to_thread(release.wait, 30)
        return await original_commit(self, conn, **kwargs)

    monkeypatch.setattr(SQLiteBackend, "_commit_write", blocking_commit)
    db, registry = await _registry(tmp_path / "commit.db")
    try:
        await _admit_and_start(registry, "did:test:commit", "commit-turn")
        block.set()

        message = await _first_owner_fence(caplog)

        assert re.search(_IN_PHASE.format(phase="commit"), message), message
    finally:
        release.set()
        await registry.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.timeout(60)
async def test_fence_reason_names_write_on_row_lock(db_backend, caplog):
    """On PostgreSQL a row lock on one of the owner's rows holds the renewal's
    UPDATE, which is its ``write`` phase."""

    if db_backend.backend_type != "postgres":
        pytest.skip(
            "a row lock is PostgreSQL's; test_fence_reason_names_slot_wait "
            "is the SQLite case"
        )
    import asyncpg

    caplog.set_level(logging.WARNING)
    store = DistributedInvocationStore(AsyncDatabase(db_backend))
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(
        store, poll_seconds=_POLL_SECONDS, owner_lease_seconds=_HEALTHY_LEASE_SECONDS
    )
    holder = transaction = None
    try:
        generation_id = await _admit_and_start(
            registry, f"did:test:row-lock-{uuid4().hex}", "row-locked-turn"
        )
        args, kwargs = db_backend.independent_connection_recipe()
        holder = await asyncpg.connect(*args, **kwargs)
        transaction = holder.transaction()
        await transaction.start()
        await holder.execute(
            "SELECT 1 FROM stop_active_invocations WHERE generation_id = $1 "
            "FOR UPDATE",
            generation_id,
        )

        message = await _first_owner_fence(caplog)

        assert re.search(_IN_PHASE.format(phase=PHASE_WRITE), message), message
    finally:
        if transaction is not None:
            await transaction.rollback()
        if holder is not None:
            await holder.close()
        await registry.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.timeout(60)
async def test_fence_reason_names_open_while_postgres_connects(
    db_backend, caplog, monkeypatch
):
    """The heartbeat's PostgreSQL pool connects only when its first renewal's
    transaction asks it for a connection. A delay there is ``open``, not the
    ``write`` no statement has reached yet."""

    if db_backend.backend_type != "postgres":
        pytest.skip(
            "SQLite's heartbeat connects in its opener; "
            "test_a_slow_renewal_logs_every_phase_and_a_quick_one_nothing "
            "covers the heartbeat's own open"
        )
    import asyncpg.pool

    caplog.set_level(logging.WARNING)
    block, release = threading.Event(), threading.Event()
    connecting = threading.Event()
    original_connect = asyncpg.pool.Pool._get_new_connection

    async def delayed_connect(pool):
        if block.is_set() and threading.current_thread().name == THREAD_NAME:
            connecting.set()
            await asyncio.to_thread(release.wait, 30)
        return await original_connect(pool)

    monkeypatch.setattr(asyncpg.pool.Pool, "_get_new_connection", delayed_connect)
    store = DistributedInvocationStore(AsyncDatabase(db_backend))
    await store.ensure_schema()
    registry = DistributedInvocationRegistry(
        store, poll_seconds=_POLL_SECONDS, owner_lease_seconds=_HEALTHY_LEASE_SECONDS
    )
    try:
        agent = _ReplicaAgent(f"did:test:pg-connect-{uuid4().hex}")
        assert await registry.register(agent, "connecting-turn", 1)
        block.set()
        registry.start()
        await _wait_for(connecting.is_set)

        message = await _first_owner_fence(caplog)

        in_open = re.search(_IN_PHASE.format(phase=PHASE_OPEN), message)
        assert in_open is not None, message
        assert float(in_open.group(1)) >= _HEALTHY_LEASE_SECONDS, message

        # Once connected, the attempt's phases put the wait in open.
        release.set()
        phases = re.compile(
            r"Its phases: open ([0-9.]+) s, write [0-9.]+ s, "
            r"inventory [0-9.]+ s, commit [0-9.]+ s$"
        )

        def connected_after_a_lease_in_open() -> bool:
            return any(
                (match := phases.search(warning)) is not None
                and float(match.group(1)) >= _HEALTHY_LEASE_SECONDS
                for warning in _slow_renewal_warnings(caplog)
            )

        await _wait_for(connected_after_a_lease_in_open)
    finally:
        release.set()
        await registry.close()


class _Clock:
    """An injected ``time`` for the registry's module: ``monotonic()`` only."""

    def __init__(self, now: float):
        self.now = now

    def monotonic(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_fence_line_reports_observation_gap(tmp_path, caplog, monkeypatch):
    """A lapse the registry noticed late says how long it had not looked."""

    caplog.set_level(logging.ERROR, logger="kestrel_sovereign.stop.invocation")
    db, registry = await _registry(tmp_path / "observation-gap.db")
    agent = _ReplicaAgent("did:test:observation-gap")
    try:
        assert await registry.register(agent, "observed-turn", 1)
        clock = _Clock(time.monotonic())
        with monkeypatch.context() as patched:
            patched.setattr(invocation_module, "time", clock)
            registry._observe_owner_heartbeat()
            assert not registry._lease_lost
            clock.now += 3 * _HEALTHY_LEASE_SECONDS + 0.05
            registry._observe_owner_heartbeat()
        assert registry._lease_lost

        message = await _first_owner_fence(caplog)

        gap = re.search(
            r"The registry last observed the owner heartbeat ([0-9.]+) s earlier",
            message,
        )
        assert gap is not None, message
        assert float(gap.group(1)) >= 3 * _HEALTHY_LEASE_SECONDS, message
    finally:
        await registry.close()
        await db.close()


def _slow_renewal_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Distributed Stop owner renewal was slow")
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_slow_renewal_logs_every_phase_and_a_quick_one_nothing(caplog):
    """An attempt of a quarter lease or more logs one warning with each
    phase's duration; describe() then names the phase it spent longest in."""

    caplog.set_level(logging.WARNING, logger="kestrel_sovereign.stop.owner_heartbeat")
    lease = 0.4

    @asynccontextmanager
    async def opener():
        yield object()

    def heartbeat_renewing(seconds_in_slot_wait: float, *, interval: float, lease: float):
        async def renew(_database, _owner_id):
            heartbeat.enter_phase(PHASE_SLOT_WAIT)
            await asyncio.sleep(seconds_in_slot_wait)
            heartbeat.enter_phase(PHASE_WRITE)

        heartbeat = OwnerHeartbeat(
            opener, renew, interval_seconds=interval, lease_seconds=lease
        )
        return heartbeat

    # Quick against a quarter of a 2 s lease, however loaded the runner.
    quick = heartbeat_renewing(0.0, interval=0.01, lease=2.0)
    quick.start("owner-with-quick-renewals")
    try:
        await _wait_for(lambda: quick.latest() is not None)
        await asyncio.sleep(0.1)
    finally:
        assert await quick.stop(timeout=10)
    assert quick.latest().phases[0][0] == PHASE_SLOT_WAIT
    assert _slow_renewal_warnings(caplog) == []

    # One attempt only: the next waits out an interval longer than the test.
    slow = heartbeat_renewing(lease + 0.05, interval=30, lease=lease)
    slow.start("owner-with-a-slow-renewal")
    try:
        await _wait_for(lambda: slow.latest() is not None)
        description = slow.describe(time.monotonic(), lease)
    finally:
        assert await slow.stop(timeout=10)

    (warning,) = _slow_renewal_warnings(caplog)
    assert "(lease is 0.4 s) and renewed" in warning, warning
    phases = re.search(
        r"Its phases: open [0-9.]+ s, slot_wait ([0-9.]+) s, write [0-9.]+ s$",
        warning,
    )
    assert phases is not None, warning
    assert float(phases.group(1)) >= lease
    assert re.fullmatch(
        r"its last renewal took [0-9.]+ s, ([0-9.]+) s of it in slot_wait",
        description,
    ), description


def test_an_unknown_renewal_phase_is_refused():
    @asynccontextmanager
    async def opener():
        yield object()

    async def renew(_database, _owner_id):
        return None

    heartbeat = OwnerHeartbeat(opener, renew, interval_seconds=0.01)
    with pytest.raises(ValueError, match="unknown owner renewal phase 'sleep'"):
        heartbeat.enter_phase("sleep")
