"""Stop-owner liveness survives event-loop stalls (#3889).

The owner lease is renewed by a heartbeat on its own thread and connection,
so one agent's synchronous stall on the shared loop no longer lets the lease
lapse and cancel every other agent's turn. These are the architect's three
acceptance checks, plus the heartbeat's own lifecycle.
"""

import asyncio
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import aiosqlite
import pytest
from fastapi import FastAPI

import kestrel_sovereign
from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError,
    InvocationSelfFencedError,
    bind_async_invocation,
)
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_cleanup_debts,
    execution_commit_outcome,
    is_execution_control_error,
)
# Imported at collection: importing the server configures logging, which
# would drop a test's caplog handler if it happened inside that test.
from kestrel_sovereign.server import (
    _distributed_invocation_owner_fence_reason,
    _distributed_invocation_owner_status,
    _distributed_invocation_owners_healthy,
)
from kestrel_sovereign.stop import (
    DistributedInvocationRegistry,
    DistributedInvocationStore,
    StopDisposition,
)
from kestrel_sovereign.stop.owner_heartbeat import (
    THREAD_NAME,
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


def _killable_heartbeat(monkeypatch, *, kill_restarted: bool = False):
    """Let a test kill the heartbeat thread from inside its next renewal.

    Returns ``(kill, failing)``. Setting ``kill`` ends the thread that next
    renews; with ``kill_restarted`` it also ends every thread started after
    it. While ``failing`` is set, renewals fail with an ordinary error.
    """

    kill, failing = threading.Event(), threading.Event()

    async def renewal(original, store, owner_id, **kwargs):
        if kill.is_set():
            if not kill_restarted:
                kill.clear()
            raise _HeartbeatKilled()
        if failing.is_set():
            raise ConnectionError("renewal store unavailable")
        return await original(store, owner_id, **kwargs)

    _patch_heartbeat_renewal(monkeypatch, renewal)
    return kill, failing


def _exit_reports(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "kestrel_sovereign.stop.invocation"
        and "owner heartbeat thread has exited" in record.getMessage()
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_heartbeat_that_died_while_idle_is_restarted_before_the_next_turn(
    tmp_path, monkeypatch, caplog
):
    """An idle owner's heartbeat thread dies. The exit is reported at once,
    with nothing in flight to age, and the next turn is admitted only under a
    restarted heartbeat that has renewed: never admitted, then self-fenced."""

    caplog.set_level(logging.ERROR, logger="kestrel_sovereign.stop.invocation")
    db, store, registry = await _owner_registry(
        tmp_path / "idle-death.db",
        poll_seconds=0.01,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent("did:test:idle-death")
    registry.attach(agent)
    kill, _failing = _killable_heartbeat(monkeypatch)
    registry.start()
    heartbeat = registry._heartbeat
    turn = None
    try:
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        kill.set()
        await _wait_for(lambda: not heartbeat.alive)
        died_at = time.monotonic()
        await _wait_for(lambda: _exit_reports(caplog))
        assert time.monotonic() - died_at < _HEALTHY_LEASE_SECONDS
        assert "(it raised _HeartbeatKilled)" in _exit_reports(caplog)[0]
        assert "the heartbeat thread has exited (it raised _HeartbeatKilled)" == (
            heartbeat.describe(time.monotonic(), _HEALTHY_LEASE_SECONDS)
        )

        await asyncio.sleep(_HEALTHY_LEASE_SECONDS + 0.2)
        # Reported once; only new work restarts it, and nothing was held.
        assert len(_exit_reports(caplog)) == 1
        assert not heartbeat.alive
        assert heartbeat.launches == 1
        assert registry.owner_lifecycle_status == "healthy"

        turn = asyncio.create_task(agent.run_turn("next-turn"))
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        assert heartbeat.alive
        assert heartbeat.launches == 2
        assert heartbeat.renewed_since_launch(registry._owner_id)
        generation_id = _generation_of(registry, agent, "next-turn")

        # Past one lease from admission, the turn still holds its lease.
        await asyncio.sleep(_HEALTHY_LEASE_SECONDS + 0.3)
        assert not turn.done()
        assert registry.owner_lifecycle_status == "healthy"
        assert registry.owner_epoch == 1
        inventory = await store.read_owner(
            registry._owner_id, lease_seconds=_HEALTHY_LEASE_SECONDS
        )
        assert generation_id in inventory.live_generation_ids

        agent.release_operation.set()
        assert await turn == "next-turn"
    finally:
        agent.release_operation.set()
        if turn is not None:
            await asyncio.gather(turn, return_exceptions=True)
        await registry.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("restarted", ["cannot_renew", "dies_again"])
async def test_a_restarted_heartbeat_that_does_not_renew_refuses_the_turn_by_name(
    tmp_path, monkeypatch, restarted
):
    """No cognition is admitted under a heartbeat that cannot renew it."""

    db, _store, registry = await _owner_registry(
        tmp_path / f"restart-{restarted}.db",
        poll_seconds=0.01,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent(f"did:test:restart-{restarted}")
    registry.attach(agent)
    kill, failing = _killable_heartbeat(
        monkeypatch, kill_restarted=restarted == "dies_again"
    )
    registry.start()
    heartbeat = registry._heartbeat
    try:
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        kill.set()
        await _wait_for(lambda: not heartbeat.alive)
        if restarted == "cannot_renew":
            failing.set()

        asked_at = time.monotonic()
        with pytest.raises(InvocationSelfFencedError) as refused:
            await agent.run_turn("refused-turn")
        # Within one lease of the restart, and before any cognition ran.
        assert time.monotonic() - asked_at < _HEALTHY_LEASE_SECONDS + 0.25
        assert not agent.operation_started.is_set()
        assert heartbeat.launches == 2
        message = str(refused.value)
        if restarted == "cannot_renew":
            assert heartbeat.alive
            assert (
                "heartbeat was restarted and has not renewed the owner lease"
                in message
            )
        else:
            assert not heartbeat.alive
            assert "the heartbeat thread has exited (it raised _HeartbeatKilled)" in (
                message
            )
        assert registry._active == {}
        rows = await db.fetchall("SELECT generation_id FROM stop_active_invocations")
        assert rows == []
    finally:
        kill.clear()
        failing.clear()
        await registry.close()
        await db.close()


def _server_owner_health(agent) -> tuple[str, str | None, bool]:
    """What server health reads of an agent's Stop owner (#3987)."""

    return (
        _distributed_invocation_owner_status(agent),
        _distributed_invocation_owner_fence_reason(agent),
        _distributed_invocation_owners_healthy(agent, None),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_server_health_reports_a_restart_that_cannot_renew_until_it_does(
    tmp_path, monkeypatch
):
    """An idle heartbeat dies and its restart cannot renew: every turn is
    refused, and server health says so, naming why. Once the restarted
    heartbeat renews the same owner, health is healthy again and turns are
    admitted (#3987)."""

    db, store, registry = await _owner_registry(
        tmp_path / "restart-health.db",
        poll_seconds=0.01,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent("did:test:restart-health")
    registry.attach(agent)
    kill, failing = _killable_heartbeat(monkeypatch)
    registry.start()
    heartbeat = registry._heartbeat
    turn = None
    try:
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        kill.set()
        await _wait_for(lambda: not heartbeat.alive)
        failing.set()
        # Death alone is left to the next turn, which restarts the thread.
        assert _server_owner_health(agent) == ("healthy", None, True)

        for request_id in ("first-refused-turn", "second-refused-turn"):
            with pytest.raises(InvocationSelfFencedError) as refused:
                await agent.run_turn(request_id)
            assert "has not renewed the owner lease" in str(refused.value)
            status, reason, ready = _server_owner_health(agent)
            assert status == "self_fenced"
            assert not ready
            assert reason.startswith(
                "epoch 1: the owner heartbeat was restarted and has not "
                "renewed the owner lease: "
            )
            assert "its last renewal failed (ConnectionError)" in reason
        assert not agent.operation_started.is_set()
        assert heartbeat.alive
        assert heartbeat.launches == 2

        # Nothing is held, so nothing fences, and the relay does not clear it.
        await asyncio.sleep(_HEALTHY_LEASE_SECONDS + 0.1)
        assert not registry._lease_lost
        assert _server_owner_health(agent)[0] == "self_fenced"
        assert heartbeat.launches == 2

        failing.clear()
        await _wait_for(lambda: _server_owner_health(agent)[0] == "healthy")
        assert _server_owner_health(agent) == ("healthy", None, True)
        # The current owner renewed; it was not replaced.
        assert registry.owner_epoch == 1
        assert heartbeat.renewed_since_launch(registry._owner_id)
        assert heartbeat.launches == 2

        turn = asyncio.create_task(agent.run_turn("admitted-turn"))
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        (generation_id,) = (
            generation_id
            for (_, turn_id, _), generation_id in registry._by_local_generation.items()
            if turn_id == "admitted-turn"
        )
        inventory = await store.read_owner(
            registry._owner_id, lease_seconds=_HEALTHY_LEASE_SECONDS
        )
        assert generation_id in inventory.live_generation_ids
        agent.release_operation.set()
        assert await turn == "admitted-turn"
        assert _server_owner_health(agent) == ("healthy", None, True)
    finally:
        kill.clear()
        failing.clear()
        agent.release_operation.set()
        if turn is not None:
            await asyncio.gather(turn, return_exceptions=True)
        await registry.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_restarted_heartbeat_that_died_before_renewing_is_restarted_without_traffic(
    tmp_path, monkeypatch, caplog
):
    """A restart whose thread dies before renewing is reported at once. The
    relay starts another one a lease later, with no turn to do it, so health
    recovers even after a load balancer has stopped sending turns (#3987)."""

    caplog.set_level(logging.ERROR, logger="kestrel_sovereign.stop.invocation")
    db, _store, registry = await _owner_registry(
        tmp_path / "restart-dies-again.db",
        poll_seconds=0.01,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent("did:test:restart-dies-again")
    registry.attach(agent)
    kill, _failing = _killable_heartbeat(monkeypatch, kill_restarted=True)
    registry.start()
    heartbeat = registry._heartbeat
    try:
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        kill.set()
        await _wait_for(lambda: not heartbeat.alive)

        with pytest.raises(InvocationSelfFencedError):
            await agent.run_turn("refused-turn")
        assert heartbeat.launches == 2
        assert not heartbeat.alive
        refused_launch = heartbeat.launched_at
        status, reason, ready = _server_owner_health(agent)
        assert (status, ready) == ("self_fenced", False)
        assert reason == (
            "epoch 1: the owner heartbeat was restarted and has not renewed "
            "the owner lease: the heartbeat thread has exited "
            "(it raised _HeartbeatKilled)"
        )

        # Once a lease has passed since that start, the relay restarts it;
        # this one dies again, so health still reports it.
        await _wait_for(lambda: heartbeat.launches == 3, timeout=10)
        assert heartbeat.launched_at - refused_launch >= _HEALTHY_LEASE_SECONDS
        await _wait_for(lambda: not heartbeat.alive)
        assert _server_owner_health(agent)[0] == "self_fenced"
        # Each thread's exit is reported, once: the relay looks every poll.
        await _wait_for(lambda: len(_exit_reports(caplog)) >= 3)
        assert len(_exit_reports(caplog)) <= heartbeat.launches

        # The next one renews: health recovers, with no turn sent.
        kill.clear()
        await _wait_for(lambda: _server_owner_health(agent)[0] == "healthy")
        assert _server_owner_health(agent) == ("healthy", None, True)
        assert heartbeat.launches >= 4
        assert len(_exit_reports(caplog)) == heartbeat.launches - 1
        assert heartbeat.alive
        assert not registry._lease_lost
        assert registry.owner_epoch == 1
        assert not agent.operation_started.is_set()
    finally:
        kill.clear()
        agent.release_operation.set()
        await registry.close()
        await db.close()


_RESTART_REASON = (
    "epoch 1: the owner heartbeat was restarted and has not renewed the owner "
    "lease: "
)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("preempt_sample", [False, True])
async def test_a_failed_restart_stays_unhealthy_through_the_next_restart_until_a_renewal(
    tmp_path, monkeypatch, preempt_sample,
):
    """A restarted heartbeat dies before renewing; the relay starts another,
    whose renewals keep failing. Through that thread's first lease, while
    admission would still wait for it, health keeps reporting the failure
    rather than reading healthy before anything renewed the owner (#3987)."""

    db, _store, registry = await _owner_registry(
        tmp_path / "restart-retry-still-failing.db",
        poll_seconds=0.01,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent("did:test:restart-retry-still-failing")
    registry.attach(agent)
    kill, failing = _killable_heartbeat(monkeypatch)
    registry.start()
    heartbeat = registry._heartbeat
    try:
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        kill.set()
        await _wait_for(lambda: not heartbeat.alive)
        # The thread the next turn starts dies on its first renewal.
        kill.set()
        with pytest.raises(InvocationSelfFencedError):
            await agent.run_turn("refused-turn")
        assert heartbeat.launches == 2
        assert not heartbeat.alive
        died = (
            "the heartbeat thread has exited (it raised _HeartbeatKilled)"
        )
        assert _server_owner_health(agent) == (
            "self_fenced", _RESTART_REASON + died, False
        )

        # The relay's next thread cannot renew either.
        failing.set()
        await _wait_for(lambda: heartbeat.launches == 3, timeout=10)
        third_launch = heartbeat.launched_at
        samples = []
        preempted = saw_failed_renewal = False

        def observed_health():
            nonlocal preempted, saw_failed_renewal
            if preempt_sample and saw_failed_renewal and not preempted:
                # A real host pause between timestamp and observation can
                # straddle the first lease; the heartbeat thread keeps running.
                remaining = third_launch + _HEALTHY_LEASE_SECONDS + 0.005 - time.monotonic()
                if remaining > 0.005:
                    time.sleep(remaining)
                    preempted = True
            health = _server_owner_health(agent)
            if health[1] and "its last renewal failed (ConnectionError)" in health[1]:
                saw_failed_renewal = True
            return health

        while time.monotonic() < third_launch + _HEALTHY_LEASE_SECONDS + 0.2:
            started = time.monotonic()
            health = observed_health()
            samples.append((started, time.monotonic(), health))
            await asyncio.sleep(0.005)
        assert not preempt_sample or preempted, "Declared lease-boundary preemption did not run"
        assert heartbeat.alive
        assert heartbeat.launches == 3
        assert not heartbeat.renewed_since_launch(registry._owner_id)
        assert all(
            status == "self_fenced" and not ready
            for _, _, (status, _reason, ready) in samples
        )
        within = [
            reason
            for _, completed_at, (_status, reason, _ready) in samples
            if completed_at < third_launch + _HEALTHY_LEASE_SECONDS
        ]
        assert within
        assert all(
            reason.startswith(
                _RESTART_REASON + died + "; the next restart has not renewed "
                "it yet: "
            )
            for reason in within
        ), "Heartbeat health sample was assigned to a lease phase before observation completed"
        assert any("its last renewal failed (ConnectionError)" in r for r in within)
        # Past its first lease the newest thread's own failure is named.
        # Read here, after the loop's deadline, instead of classifying a
        # potentially interrupted last sample as a post-lease observation.
        assert _server_owner_health(agent)[1].startswith(
            _RESTART_REASON + "its last renewal failed (ConnectionError)"
        )
        assert not agent.operation_started.is_set()

        failing.clear()
        await _wait_for(lambda: _server_owner_health(agent)[0] == "healthy")
        assert _server_owner_health(agent) == ("healthy", None, True)
        # The relay's next observation drops the recovered failure.
        await _wait_for(lambda: registry._heartbeat_restart_failure is None)
        assert heartbeat.launches == 3
        assert registry.owner_epoch == 1
        assert not registry._lease_lost
    finally:
        kill.clear()
        failing.clear()
        agent.release_operation.set()
        await registry.close()
        await db.close()


def _refusable_heartbeat_start(monkeypatch) -> threading.Event:
    """While the returned event is set, a heartbeat thread cannot start.

    ``Thread.start()`` raises as CPython does when the OS cannot create a
    thread; every other thread starts as usual.
    """

    refuse = threading.Event()
    original_start = threading.Thread.start

    def start(thread):
        if thread.name == THREAD_NAME and refuse.is_set():
            raise RuntimeError("can't start new thread")
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    return refuse


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_heartbeat_thread_that_cannot_be_started_is_named_until_one_renews(
    tmp_path, monkeypatch
):
    """Restarting the idle owner's dead heartbeat fails in ``Thread.start()``.
    Turns are refused naming that, health names it, through every failed
    restart and while the thread that finally starts has not renewed yet;
    its renewal clears it (#3987)."""

    db, _store, registry = await _owner_registry(
        tmp_path / "restart-cannot-start.db",
        poll_seconds=0.01,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent("did:test:restart-cannot-start")
    registry.attach(agent)
    kill, failing = _killable_heartbeat(monkeypatch)
    refuse_start = _refusable_heartbeat_start(monkeypatch)
    registry.start()
    heartbeat = registry._heartbeat
    try:
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        kill.set()
        await _wait_for(lambda: not heartbeat.alive)
        refuse_start.set()

        cannot_start = (
            "a new heartbeat thread could not be started "
            "(RuntimeError: can't start new thread)"
        )
        for request_id in ("first-refused-turn", "second-refused-turn"):
            with pytest.raises(InvocationSelfFencedError) as refused:
                await agent.run_turn(request_id)
            assert str(refused.value) == (
                "distributed Stop owner heartbeat could not be restarted, so "
                "admission was refused: " + cannot_start
            )
            assert _server_owner_health(agent) == (
                "self_fenced", _RESTART_REASON + cannot_start, False
            )
        assert heartbeat.launches >= 3
        assert not heartbeat.alive
        assert not agent.operation_started.is_set()

        # The relay retries a lease after each failed start; one starts at
        # last, but cannot renew yet: health still names the failed start.
        failed_launches = heartbeat.launches
        failing.set()
        refuse_start.clear()
        await _wait_for(lambda: heartbeat.alive, timeout=10)
        assert heartbeat.launches == failed_launches + 1
        status, reason, ready = _server_owner_health(agent)
        assert (status, ready) == ("self_fenced", False)
        assert reason.startswith(
            _RESTART_REASON + cannot_start + "; the next restart has not "
            "renewed it yet: "
        )

        failing.clear()
        await _wait_for(lambda: _server_owner_health(agent)[0] == "healthy")
        assert _server_owner_health(agent) == ("healthy", None, True)
        assert heartbeat.launches == failed_launches + 1
        assert registry.owner_epoch == 1
        assert not registry._lease_lost
    finally:
        refuse_start.clear()
        kill.clear()
        failing.clear()
        agent.release_operation.set()
        await registry.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_an_admission_never_restarts_the_heartbeat_for_a_lapsed_owner(
    tmp_path, monkeypatch
):
    """With work in flight, a dead heartbeat lets the lease lapse as before.
    An admission that reads the lapse first fences that owner and is refused
    by name; it does not restart the heartbeat under the lapsed owner."""

    db, _store, registry = await _owner_registry(
        tmp_path / "lapsed-owner.db",
        poll_seconds=0.01,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    in_flight = _SelfFencingReplicaAgent("did:test:lapsed-in-flight")
    newcomer = _SelfFencingReplicaAgent("did:test:lapsed-newcomer")
    registry.attach(in_flight)
    registry.attach(newcomer)
    kill, _failing = _killable_heartbeat(monkeypatch)
    registry.start()
    heartbeat = registry._heartbeat
    # Without the relay, the admission is the first to read the lapse.
    relay = registry._relay_task
    relay.cancel()
    await asyncio.gather(relay, return_exceptions=True)
    registry._relay_task = None
    turn = asyncio.create_task(in_flight.run_turn("in-flight-turn"))
    try:
        await asyncio.wait_for(in_flight.operation_started.wait(), timeout=10)
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        kill.set()
        await _wait_for(lambda: not heartbeat.alive)
        await asyncio.sleep(_HEALTHY_LEASE_SECONDS + 0.1)

        with pytest.raises(InvocationSelfFencedError) as refused:
            await newcomer.run_turn("newcomer-turn")
        message = str(refused.value)
        assert "expired before admission" in message
        assert "the heartbeat thread has exited (it raised _HeartbeatKilled)" in (
            message
        )
        assert not newcomer.operation_started.is_set()
        assert heartbeat.launches == 1
        with pytest.raises(InvocationSelfFencedError):
            await asyncio.wait_for(turn, timeout=10)

        # Ownership is then re-established under a restarted heartbeat.
        assert await registry.register(newcomer, "later-turn", 1)
        assert heartbeat.launches == 2
        assert registry.owner_epoch == 2
        assert heartbeat.renewed_since_launch(registry._owner_id)
    finally:
        in_flight.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        await registry.close()
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
@pytest.mark.parametrize("renewal_reply_delay", [0.0, 0.35])
async def test_a_failed_attempt_after_a_stall_keeps_the_renewals_made_during_it(
    tmp_path, monkeypatch, renewal_reply_delay
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
    delay_replies = threading.Event()
    delay_replies.set()

    async def failable(original, store, owner_id, **kwargs):
        if failing.is_set():
            raise ConnectionError("injected renewal failure")
        delayed = delay_replies.is_set()
        poll = await original(store, owner_id, **kwargs)
        if delayed:
            await asyncio.sleep(renewal_reply_delay)
        return poll

    _patch_heartbeat_renewal(monkeypatch, failable)
    registry.start()
    turn = asyncio.create_task(agent.run_turn("held-turn"))
    try:
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        generation_id = _generation_of(registry, agent, "held-turn")
        heartbeat = registry._heartbeat

        # Keep the loop blocked until a real completed renewal began after
        # the lease it last folded in expired. A fixed sleep does not produce
        # that scenario when a genuine renewal's reply is slow.
        stalled_at = time.monotonic()
        deadline = time.monotonic() + 10
        while True:
            renewed = heartbeat.latest_renewed()
            if renewed is not None and renewed.started > stalled_at + _HEALTHY_LEASE_SECONDS:
                break
            assert time.monotonic() < deadline, "no original renewal crossed the stalled lease"
            time.sleep(0.002)
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
        assert time.monotonic() - renewed.started < _HEALTHY_LEASE_SECONDS
        assert generation_id in renewed.poll.live_generation_ids
        assert time.monotonic() - registry._renewed_at[generation_id] >= (
            _HEALTHY_LEASE_SECONDS
        )

        registry._observe_owner_heartbeat()
        assert not registry._lease_lost, registry.owner_fence_reason
        assert registry._renewed_at[generation_id] == renewed.started
        # The failed-attempt observation above is synchronous and complete.
        # End the test-owned reply delay/failure before recovery: a new late
        # reply must not turn the healthy recovery phase into a second expiry.
        delay_replies.clear()
        failing.clear()
        await _wait_for(lambda: heartbeat.latest().failure is None)
        await asyncio.sleep(0.1)
        assert registry.owner_lifecycle_status == "healthy"
        assert not turn.done()
        agent.release_operation.set()
        assert await turn == "held-turn"
        assert registry.owner_epoch == 1
    finally:
        delay_replies.clear()
        failing.clear()
        agent.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        await registry.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_an_admission_whose_reply_is_late_keeps_the_renewal_the_relay_already_read(
    tmp_path, monkeypatch
):
    """An insert's reply can arrive after the relay folded in the renewal that
    renewed the new row. That renewal still holds the row's lease: neither
    the admission nor another agent's turn is fenced."""

    db, store, registry = await _owner_registry(
        tmp_path / "late-admission-reply.db",
        poll_seconds=0.02,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    bystander = _SelfFencingReplicaAgent("did:test:late-reply-bystander")
    late = _SelfFencingReplicaAgent("did:test:late-reply-admission")
    registry.attach(bystander)
    registry.attach(late)
    holding = threading.Event()
    parked = threading.Event()
    released = threading.Event()

    async def holdable(original, store, owner_id, **kwargs):
        if holding.is_set():
            parked.set()
            released.wait(30)
        return await original(store, owner_id, **kwargs)

    _patch_heartbeat_renewal(monkeypatch, holdable)
    original_register = store.register
    late_reply: dict = {}

    async def register_with_a_late_reply(**kwargs):
        sent = time.monotonic()
        admitted = await original_register(**kwargs)
        if kwargs["agent_id"] != late.agent_id:
            return admitted
        # The row has committed; its reply arrives more than a lease later,
        # while the heartbeat renews the row and the relay folds that in.
        await asyncio.sleep(_HEALTHY_LEASE_SECONDS + 0.1)
        # No renewal finishes before the admission resumes, and the relay
        # has already observed the newest one.
        holding.set()
        await _wait_for(parked.is_set)
        renewal = registry._heartbeat.latest_renewed()
        await _wait_for(lambda: registry._observed_renewal is renewal)
        late_reply.update(
            sent=sent, renewal=renewal, generation_id=kwargs["generation_id"]
        )
        return admitted

    store.register = register_with_a_late_reply
    registry.start()
    bystander_turn = asyncio.create_task(bystander.run_turn("bystander-turn"))
    try:
        await asyncio.wait_for(bystander.operation_started.wait(), timeout=10)

        assert await registry.register(late, "late-turn", 1)

        renewal, generation_id = late_reply["renewal"], late_reply["generation_id"]
        # The situation the fix is for: the renewal renewed the new row after
        # it committed, the relay read it before the admission published the
        # row, and the admission began more than a lease ago.
        assert renewal.started > late_reply["sent"]
        assert generation_id in renewal.poll.live_generation_ids
        assert time.monotonic() - late_reply["sent"] >= _HEALTHY_LEASE_SECONDS
        assert registry._renewed_at[generation_id] == renewal.started
        assert not registry._lease_lost, registry.owner_fence_reason
        assert registry.owner_lifecycle_status == "healthy"
        assert registry.owner_epoch == 1
        assert not bystander_turn.done()

        holding.clear()
        released.set()
        # Let the relay act on everything it can now see.
        await asyncio.sleep(10 * registry._poll_seconds)
        assert registry.owner_lifecycle_status == "healthy"
        assert not bystander_turn.done()
        bystander.release_operation.set()
        assert await bystander_turn == "bystander-turn"
        registry.complete_soon(late, "late-turn", 1)
        await _wait_for(lambda: generation_id not in registry._active)
    finally:
        holding.clear()
        released.set()
        bystander.release_operation.set()
        await asyncio.gather(bystander_turn, return_exceptions=True)
        await registry.close()
        await db.close()


def _count_heartbeat_renewals(monkeypatch, failure=None):
    """Record every renewal attempt; raise ``failure()`` while it returns one.

    Returns the list of ``(time.monotonic(), succeeded)`` per attempt.
    """

    attempts: list[tuple[float, bool]] = []

    async def counted(original, store, owner_id, **kwargs):
        started = time.monotonic()
        error = None if failure is None else failure()
        if error is not None:
            attempts.append((started, False))
            raise error
        try:
            polled = await original(store, owner_id, **kwargs)
        except BaseException:
            attempts.append((started, False))
            raise
        attempts.append((started, True))
        return polled

    _patch_heartbeat_renewal(monkeypatch, counted)
    return attempts


_RENEWAL_TASK = "distributed-stop-owner-poll:"


class _LostRenewalAcknowledgement:
    """Once armed, drop the reply to the heartbeat's next renewal COMMIT, on
    its own connection, after SQLite has run the COMMIT.

    The backend cannot tell whether that renewal committed, so it raises
    ``ExecutionCommitOutcomeError("unknown")`` from the error appended to
    ``lost``. ``sparing`` exempts one owner identity's renewals; ``hold``
    keeps the renewal waiting for ``release`` before its reply is lost.
    """

    def __init__(self, monkeypatch):
        self.lost: list[BaseException] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self._armed = False
        self._sparing: str | None = None
        self._hold = False
        native_commit = aiosqlite.Connection.commit

        async def commit(connection):
            await native_commit(connection)
            name = asyncio.current_task().get_name()
            if threading.current_thread().name != THREAD_NAME or not (
                name.startswith(_RENEWAL_TASK)
            ):
                return
            with self._lock:
                if not self._armed or name == f"{_RENEWAL_TASK}{self._sparing}":
                    return
                self._armed = False
                hold = self._hold
            if hold:
                self.entered.set()
                self.release.wait(30)
            self.lost.append(
                ConnectionResetError("renewal COMMIT acknowledgement lost")
            )
            raise self.lost[-1]

        monkeypatch.setattr(aiosqlite.Connection, "commit", commit)

    def arm(self, *, sparing: str | None = None, hold: bool = False) -> None:
        with self._lock:
            self._armed, self._sparing, self._hold = True, sparing, hold


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_lost_renewal_commit_acknowledgement_fences_and_is_never_retried(
    tmp_path, monkeypatch
):
    """A renewal error carrying execution control means what it meant when
    the relay renewed on the server's loop: it is never retried, the owner
    fences, new work is refused with the original error, and close raises
    it. Each outcome asserted here is main's for the same injection."""

    db, _store, registry = await _owner_registry(
        tmp_path / "lost-renewal-ack.db",
        poll_seconds=0.02,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent("did:test:lost-renewal-ack")
    registry.attach(agent)
    attempts = _count_heartbeat_renewals(monkeypatch)
    lost_ack = _LostRenewalAcknowledgement(monkeypatch)
    registry.start()
    heartbeat = registry._heartbeat
    turn = asyncio.create_task(agent.run_turn("in-flight-turn"))
    closed = False
    try:
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        lost_ack.arm()
        await _wait_for(lambda: registry._lease_lost)

        control = heartbeat.latest().control
        assert isinstance(control, ExecutionCommitOutcomeError)
        assert execution_commit_outcome(control) == "unknown"
        assert control.__cause__ is lost_ack.lost[0]
        # The registry holds that exact exception, and fenced on it.
        assert registry._terminal_control is control
        assert registry.owner_lifecycle_status == "self_fenced"
        assert registry.owner_fence_reason == (
            "epoch 1: original execution control retained by Stop settlement"
        )
        with pytest.raises(InvocationSelfFencedError):
            await asyncio.wait_for(turn, timeout=10)

        # Never retried: no further attempt, for that owner or a successor.
        made = len(attempts)
        assert not attempts[-1][1]
        await asyncio.sleep(25 * registry._poll_seconds)
        assert len(attempts) == made
        assert heartbeat.latest().control is control

        later = _SelfFencingReplicaAgent("did:test:lost-renewal-ack")
        with pytest.raises(ExecutionCommitOutcomeError) as refused:
            await registry.register(later, "later-turn", 1)
        assert refused.value is control

        closed = True
        with pytest.raises(ExecutionCommitOutcomeError) as raised:
            await registry.close()
        assert raised.value is control
        assert not heartbeat.alive
        assert len(attempts) == made
    finally:
        agent.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        if not closed:
            try:
                await registry.close()
            except ExecutionCommitOutcomeError:
                pass
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("entry", ["admission", "turn binding"])
async def test_the_call_that_first_reads_a_lost_acknowledgement_is_refused_with_it(
    tmp_path, monkeypatch, entry
):
    """On main the relay held the control the moment the renewal failed, so a
    call that started later was refused with it. Here the call itself may be
    the first to read it from the heartbeat; it is refused with it too."""

    db, _store, registry = await _owner_registry(
        tmp_path / "lost-ack-first-reader.db",
        poll_seconds=0.02,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    lost_ack = _LostRenewalAcknowledgement(monkeypatch)
    agent = _SelfFencingReplicaAgent("did:test:lost-ack-first-reader")
    registry.attach(agent)
    registry.start()
    heartbeat = registry._heartbeat
    # Without the relay, only the call below reads what the heartbeat holds.
    relay = registry._relay_task
    relay.cancel()
    await asyncio.gather(relay, return_exceptions=True)
    registry._relay_task = None
    closed = False
    try:
        if entry == "turn binding":
            assert await registry.register(agent, "private-request", 1)
        rows = await db.fetchall("SELECT generation_id FROM stop_active_invocations")
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        lost_ack.arm()
        await _wait_for(lambda: heartbeat.latest().control is not None)
        control = heartbeat.latest().control
        assert registry._terminal_control is None and not registry._lease_lost

        with pytest.raises(ExecutionCommitOutcomeError) as refused:
            if entry == "admission":
                await registry.register(agent, "refused-request", 1)
            else:
                await registry.bind_public_turn(
                    agent, "public-turn", "private-request", 1
                )
        assert refused.value is control
        assert registry._terminal_control is control
        assert registry.owner_lifecycle_status == "self_fenced"
        # Refused before it wrote anything.
        assert await db.fetchall(
            "SELECT generation_id FROM stop_active_invocations"
        ) == rows
        assert await db.fetchall(
            "SELECT generation_id FROM stop_active_invocations "
            "WHERE public_turn_digest IS NOT NULL"
        ) == []

        closed = True
        with pytest.raises(ExecutionCommitOutcomeError) as raised:
            await registry.close()
        assert raised.value is control
    finally:
        if not closed:
            try:
                await registry.close()
            except ExecutionCommitOutcomeError:
                pass
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_lost_acknowledgement_while_reacquiring_is_held_not_retried(
    tmp_path, monkeypatch
):
    """A fenced registry re-establishing ownership holds a control-bearing
    renewal of its candidate identity; it is not an unavailable store."""

    db, _store, registry = await _owner_registry(
        tmp_path / "lost-ack-reacquiring.db",
        poll_seconds=0.02,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    attempts = _count_heartbeat_renewals(monkeypatch)
    lost_ack = _LostRenewalAcknowledgement(monkeypatch)
    registry.start()
    heartbeat = registry._heartbeat
    closed = False
    try:
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        lost_ack.arm(sparing=registry._owner_id)
        registry._fail_closed_owner("owner fenced for review")
        candidate = registry._candidate_owner_id

        await _wait_for(lambda: registry._terminal_control is not None)
        control = heartbeat.latest().control
        assert heartbeat.latest().owner_id == candidate
        assert control.__cause__ is lost_ack.lost[0]
        assert registry._terminal_control is control
        assert registry._reacquisition_failure is None
        assert registry.owner_epoch == 1
        made = len(attempts)
        await asyncio.sleep(25 * registry._poll_seconds)
        assert len(attempts) == made

        with pytest.raises(ExecutionCommitOutcomeError) as refused:
            await registry.register(
                _SelfFencingReplicaAgent("did:test:lost-ack-reacquiring"),
                "refused-turn",
                1,
            )
        assert refused.value is control
        closed = True
        with pytest.raises(ExecutionCommitOutcomeError) as raised:
            await registry.close()
        assert raised.value is control
    finally:
        if not closed:
            try:
                await registry.close()
            except ExecutionCommitOutcomeError:
                pass
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_close_raises_a_lost_acknowledgement_of_the_renewal_it_waited_for(
    tmp_path, monkeypatch
):
    """A renewal in flight when close begins finishes before the heartbeat
    exits; close raises the control that renewal raised."""

    db, _store, registry = await _owner_registry(
        tmp_path / "lost-ack-at-close.db",
        poll_seconds=0.02,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    lost_ack = _LostRenewalAcknowledgement(monkeypatch)
    registry.start()
    heartbeat = registry._heartbeat
    closing = None
    try:
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        lost_ack.arm(hold=True)
        assert await asyncio.to_thread(lost_ack.entered.wait, 10)

        closing = asyncio.create_task(registry.close())
        done, _ = await asyncio.wait({closing}, timeout=0.1)
        assert not done, "close did not wait for the renewal in flight"
        assert registry._terminal_control is None
        lost_ack.release.set()

        with pytest.raises(ExecutionCommitOutcomeError) as raised:
            await closing
        control = heartbeat.latest().control
        assert raised.value is control
        assert control.__cause__ is lost_ack.lost[0]
        assert registry._terminal_control is control
        assert registry.owner_lifecycle_status == "self_fenced"
        assert not heartbeat.alive
    finally:
        lost_ack.release.set()
        if closing is None:
            await registry.close()
        else:
            await asyncio.gather(closing, return_exceptions=True)
        await db.close()


async def _peer_stop(path: Path, agent_id: str, turn_id: str) -> StopDisposition:
    """A Stop for one turn, issued by another replica over its own connection."""

    peer_db = await AsyncDatabase.sqlite(str(path), schema_initializer=_no_schema)
    try:
        peer = DistributedInvocationRegistry(
            DistributedInvocationStore(peer_db),
            poll_seconds=0.02,
            owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
        )
        ticket = await peer.request_turn(agent_id, turn_id)
        return await peer.wait_for_stop(
            ticket, timeout_seconds=2 * _HEALTHY_LEASE_SECONDS
        )
    finally:
        await peer_db.close()


async def _rows_by_agent(db, table: str) -> list[str]:
    rows = await db.fetchall(f"SELECT agent_id FROM {table} ORDER BY agent_id")
    return [str(row[0]) for row in rows]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("point", ["insert", "confirmation read", "relay"])
async def test_a_lost_renewal_acknowledgement_during_admission_stays_unresolved(
    tmp_path, monkeypatch, point
):
    """Held execution control outranks lease loss wherever admission meets it.

    A renewal loses its COMMIT acknowledgement while an admission awaits its
    insert, or its confirmation read, or its insert while the relay, not the
    admission, first reads the control with another agent's turn running.
    The caller receives that original exception, never a self-fence. No
    generation is recorded completed: each row stays unresolved, a peer's
    Stop for it returns UNREACHABLE, and close raises the original. These
    are main's outcomes when the admission's own renewal lost its
    acknowledgement.
    """

    path = tmp_path / "lost-ack-during-admission.db"
    db, store, registry = await _owner_registry(
        path, poll_seconds=0.02, owner_lease_seconds=_HEALTHY_LEASE_SECONDS
    )
    lost_ack = _LostRenewalAcknowledgement(monkeypatch)
    admitted = _SelfFencingReplicaAgent("did:test:lost-ack-admitted")
    running = _SelfFencingReplicaAgent("did:test:lost-ack-running")
    registry.attach(admitted)
    registry.attach(running)
    registry.start()
    heartbeat = registry._heartbeat
    if point != "relay":
        # Without the relay, the admission is the first to read the control.
        relay = registry._relay_task
        relay.cancel()
        await asyncio.gather(relay, return_exceptions=True)
        registry._relay_task = None

    async def lose_a_renewal_acknowledgement() -> None:
        lost_ack.arm()
        if point == "relay":
            await _wait_for(lambda: registry._terminal_control is not None)
        else:
            await _wait_for(
                lambda: heartbeat.latest() is not None
                and heartbeat.latest().control is not None
            )
            assert registry._terminal_control is None

    if point == "confirmation read":
        original_read = store.read_owner

        async def read_while_a_renewal_loses_its_ack(*args, **kwargs):
            await lose_a_renewal_acknowledgement()
            return await original_read(*args, **kwargs)

        store.read_owner = read_while_a_renewal_loses_its_ack
    else:
        original_register = store.register

        async def insert_while_a_renewal_loses_its_ack(**kwargs):
            inserted = await original_register(**kwargs)
            if kwargs["agent_id"] == admitted.agent_id:
                await lose_a_renewal_acknowledgement()
            return inserted

        store.register = insert_while_a_renewal_loses_its_ack
    running_turn = None
    if point == "relay":
        running_turn = asyncio.create_task(running.run_turn("running-turn"))
        await asyncio.wait_for(running.operation_started.wait(), timeout=10)
    closed = False
    try:
        with pytest.raises(ExecutionCommitOutcomeError) as refused:
            await asyncio.wait_for(admitted.run_turn("admitted-turn"), timeout=10)
        control = heartbeat.latest().control
        assert control.__cause__ is lost_ack.lost[0]
        assert refused.value is control
        assert registry._terminal_control is control
        assert registry.owner_lifecycle_status == "self_fenced"

        unresolved = [admitted.agent_id]
        if running_turn is not None:
            with pytest.raises(InvocationSelfFencedError):
                await asyncio.wait_for(running_turn, timeout=10)
            unresolved.append(running.agent_id)
        await _wait_for(lambda: not registry._active)
        assert await _rows_by_agent(db, "stop_active_invocations") == []
        assert await _rows_by_agent(db, "stop_unresolved_invocations") == sorted(
            unresolved
        )
        assert await _peer_stop(
            path, admitted.agent_id, "admitted-turn"
        ) is StopDisposition.UNREACHABLE
        if running_turn is not None:
            assert await _peer_stop(
                path, running.agent_id, "running-turn"
            ) is StopDisposition.UNREACHABLE

        closed = True
        with pytest.raises(ExecutionCommitOutcomeError) as raised:
            await registry.close()
        assert raised.value is control
        assert await _rows_by_agent(db, "stop_unresolved_invocations") == sorted(
            unresolved
        )
    finally:
        running.release_operation.set()
        admitted.release_operation.set()
        if running_turn is not None:
            await asyncio.gather(running_turn, return_exceptions=True)
        if not closed:
            try:
                await registry.close()
            except ExecutionCommitOutcomeError:
                pass
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "point",
    ["confirmation read fails", "binding waits for its lock", "binding writes the turn"],
)
async def test_held_control_outranks_lease_loss_at_the_other_admission_steps(
    tmp_path, monkeypatch, point
):
    """The remaining places a call can find the owner fenced while a renewal's
    control is held each raise that original exception, not a self-fence."""

    db, store, registry = await _owner_registry(
        tmp_path / "held-control-precedence.db",
        poll_seconds=0.02,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    lost_ack = _LostRenewalAcknowledgement(monkeypatch)
    agent = _SelfFencingReplicaAgent("did:test:held-control-precedence")
    registry.attach(agent)
    registry.start()
    heartbeat = registry._heartbeat
    if point != "binding waits for its lock":
        # Without the relay, the call itself is the first to read the control.
        relay = registry._relay_task
        relay.cancel()
        await asyncio.gather(relay, return_exceptions=True)
        registry._relay_task = None

    async def lose_a_renewal_acknowledgement() -> None:
        lost_ack.arm()
        await _wait_for(
            lambda: heartbeat.latest() is not None
            and heartbeat.latest().control is not None
        )

    closed = False
    try:
        if point == "confirmation read fails":
            original_read = store.read_owner

            async def read_failing_while_a_renewal_loses_its_ack(*args, **kwargs):
                await lose_a_renewal_acknowledgement()
                raise ConnectionError("confirmation read failed")

            store.read_owner = read_failing_while_a_renewal_loses_its_ack
            with pytest.raises(ExecutionCommitOutcomeError) as refused:
                await asyncio.wait_for(agent.run_turn("admitted-turn"), timeout=10)
            store.read_owner = original_read
        else:
            assert await registry.register(agent, "private-request", 1)
            if point == "binding writes the turn":
                original_bind = store.bind_public_turn

                async def bind_while_a_renewal_loses_its_ack(**kwargs):
                    bound = await original_bind(**kwargs)
                    await lose_a_renewal_acknowledgement()
                    return bound

                store.bind_public_turn = bind_while_a_renewal_loses_its_ack
                binding = asyncio.create_task(
                    registry.bind_public_turn(agent, "public-turn", "private-request", 1)
                )
            else:
                async with registry._registration_lock:
                    binding = asyncio.create_task(
                        registry.bind_public_turn(
                            agent, "public-turn", "private-request", 1
                        )
                    )
                    for _ in range(5):
                        await asyncio.sleep(0)
                    assert not binding.done()
                    await lose_a_renewal_acknowledgement()
                    # The relay holds it, and fences, while binding waits.
                    await _wait_for(lambda: registry._terminal_control is not None)
            with pytest.raises(ExecutionCommitOutcomeError) as refused:
                await asyncio.wait_for(binding, timeout=10)

        control = heartbeat.latest().control
        assert control.__cause__ is lost_ack.lost[0]
        assert refused.value is control
        assert registry._terminal_control is control
        closed = True
        with pytest.raises(ExecutionCommitOutcomeError) as raised:
            await registry.close()
        assert raised.value is control
    finally:
        agent.release_operation.set()
        if not closed:
            try:
                await registry.close()
            except ExecutionCommitOutcomeError:
                pass
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_turn_finishing_before_any_observation_of_a_lost_acknowledgement_stays_unresolved(
    tmp_path, monkeypatch
):
    """The relay held a renewal's control the moment it was raised. Here the
    heartbeat holds it until something reads it, and a turn that finishes in
    that window is still not recorded completed."""

    path = tmp_path / "finished-before-observation.db"
    db, _store, registry = await _owner_registry(
        path, poll_seconds=0.02, owner_lease_seconds=_HEALTHY_LEASE_SECONDS
    )
    lost_ack = _LostRenewalAcknowledgement(monkeypatch)
    agent = _SelfFencingReplicaAgent("did:test:finished-before-observation")
    registry.attach(agent)
    registry.start()
    heartbeat = registry._heartbeat
    relay = registry._relay_task
    relay.cancel()
    await asyncio.gather(relay, return_exceptions=True)
    registry._relay_task = None
    turn = asyncio.create_task(agent.run_turn("finished-turn"))
    try:
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        lost_ack.arm()
        await _wait_for(
            lambda: heartbeat.latest() is not None
            and heartbeat.latest().control is not None
        )
        assert registry._terminal_control is None

        agent.release_operation.set()
        assert await turn == "finished-turn"
        await _wait_for(lambda: not registry._active)
        control = heartbeat.latest().control
        assert registry._terminal_control is control
        assert await _rows_by_agent(db, "stop_unresolved_invocations") == [
            agent.agent_id
        ]
        assert await _peer_stop(
            path, agent.agent_id, "finished-turn"
        ) is StopDisposition.UNREACHABLE
    finally:
        agent.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        try:
            await registry.close()
        except ExecutionCommitOutcomeError:
            pass
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_completion_queued_behind_a_writer_when_control_is_held_stays_unresolved(
    tmp_path, monkeypatch
):
    """A completion decided before a renewal raised control, but settled
    after, is not recorded completed.

    The turn finishes while another task holds the server connection's
    writer, so its settlement, already decided ``COMPLETED``, queues behind
    it. A renewal then loses its COMMIT acknowledgement and the registry
    holds the original exception before the writer is released. The
    settlement asks again once it holds the database: the row stays
    unresolved, a peer's Stop for the turn returns UNREACHABLE, and close
    raises the original.
    """

    path = tmp_path / "completion-behind-a-writer.db"
    # A lease long enough that holding the renewal never lapses it here.
    db, store, registry = await _owner_registry(
        path, poll_seconds=0.02, owner_lease_seconds=10 * _HEALTHY_LEASE_SECONDS
    )
    lost_ack = _LostRenewalAcknowledgement(monkeypatch)
    agent = _SelfFencingReplicaAgent("did:test:completion-behind-a-writer")
    registry.attach(agent)
    registry.start()
    heartbeat = registry._heartbeat

    decided: list[RequestCompletionDisposition] = []
    settling = asyncio.Event()
    original_settle = store.settle

    async def recorded_settle(generation_id, owner_id, disposition, **kwargs):
        decided.append(disposition)
        settling.set()
        return await original_settle(generation_id, owner_id, disposition, **kwargs)

    store.settle = recorded_settle
    writer_held = asyncio.Event()
    release_writer = asyncio.Event()

    async def hold_the_writer() -> None:
        async with db.transaction(immediate=True):
            writer_held.set()
            await release_writer.wait()

    turn = asyncio.create_task(agent.run_turn("contended-turn"))
    writer = None
    closed = False
    try:
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        # The renewal commits, then waits before its reply is lost: SQLite's
        # write lock is free, and no control has been raised yet.
        lost_ack.arm(hold=True)
        assert await asyncio.to_thread(lost_ack.entered.wait, 10)
        writer = asyncio.create_task(hold_the_writer())
        await asyncio.wait_for(writer_held.wait(), timeout=10)

        agent.release_operation.set()
        assert await asyncio.wait_for(turn, timeout=10) == "contended-turn"
        await asyncio.wait_for(settling.wait(), timeout=10)
        assert decided == [RequestCompletionDisposition.COMPLETED]
        assert registry._terminal_control is None

        lost_ack.release.set()
        await _wait_for(lambda: registry._terminal_control is not None)
        control = heartbeat.latest().control
        assert control.__cause__ is lost_ack.lost[0]
        assert registry._terminal_control is control
        assert registry._active

        release_writer.set()
        await asyncio.wait_for(writer, timeout=10)
        await _wait_for(lambda: not registry._active)
        assert decided == [RequestCompletionDisposition.COMPLETED]
        assert await _rows_by_agent(db, "stop_active_invocations") == []
        assert await _rows_by_agent(db, "stop_unresolved_invocations") == [
            agent.agent_id
        ]
        assert await _peer_stop(
            path, agent.agent_id, "contended-turn"
        ) is StopDisposition.UNREACHABLE

        closed = True
        with pytest.raises(ExecutionCommitOutcomeError) as raised:
            await registry.close()
        assert raised.value is control
    finally:
        lost_ack.release.set()
        release_writer.set()
        agent.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        if writer is not None:
            await asyncio.gather(writer, return_exceptions=True)
        if not closed:
            try:
                await registry.close()
            except ExecutionCommitOutcomeError:
                pass
        await db.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize("abandon", [True, False])
async def test_a_completed_settlement_asks_again_under_database_custody(
    db_backend, abandon
):
    """``abandon_if`` is asked inside the settlement's transaction, after its
    agent lock, and turns ``COMPLETED`` into ``ABANDONED`` only when true."""

    store = DistributedInvocationStore(AsyncDatabase(db_backend))
    await store.ensure_schema()
    suffix = uuid4().hex
    agent_id = f"did:test:settle-recheck-{suffix}"
    owner_id = f"owner-{suffix}"
    asked: list[bool] = []

    def abandon_if() -> bool:
        asked.append(True)
        return abandon

    async def registered(label: str) -> str:
        generation_id = f"{label}-{suffix}"
        assert await store.register(
            generation_id=generation_id,
            agent_id=agent_id,
            turn_id=f"turn-{label}-{suffix}",
            owner_id=owner_id,
            request_generation=1,
        )
        return generation_id

    async def lifecycle(generation_id: str) -> tuple[bool, bool]:
        active = await store._db.fetchone(
            "SELECT 1 FROM stop_active_invocations WHERE generation_id = ?",
            (generation_id,),
        )
        unresolved = await store._db.fetchone(
            "SELECT 1 FROM stop_unresolved_invocations WHERE generation_id = ?",
            (generation_id,),
        )
        return active is not None, unresolved is not None

    completed = await registered("completed")
    await store.settle(
        completed,
        owner_id,
        RequestCompletionDisposition.COMPLETED,
        abandon_if=abandon_if,
    )
    assert asked == [True]
    assert await lifecycle(completed) == (False, abandon)

    abandoned = await registered("abandoned")
    await store.settle(
        abandoned,
        owner_id,
        RequestCompletionDisposition.ABANDONED,
        abandon_if=abandon_if,
    )
    # An abandonment is already the conservative outcome: nothing to ask.
    assert asked == [True]
    assert await lifecycle(abandoned) == (False, True)

    backend = store._db.backend
    if getattr(backend, "backend_type", "") == "postgres":
        # The native settlement forwards the same recheck to the same SQL.
        native = await registered("native")
        await backend.settle_stop_invocation(
            native,
            owner_id,
            RequestCompletionDisposition.COMPLETED,
            abandon_if=abandon_if,
        )
        assert asked == [True, True]
        assert await lifecycle(native) == (False, abandon)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_transient_renewal_error_then_a_renewal_within_the_lease_changes_nothing(
    tmp_path, monkeypatch
):
    """An ordinary error is retried; the next renewal inside the lease keeps
    the owner and every turn as they were."""

    db, _store, registry = await _owner_registry(
        tmp_path / "transient-renewal-error.db",
        poll_seconds=0.02,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent("did:test:transient-renewal-error")
    registry.attach(agent)
    to_fail = [0]
    to_fail_lock = threading.Lock()

    def transient_failure():
        with to_fail_lock:
            if to_fail[0] == 0:
                return None
            to_fail[0] -= 1
        return ConnectionError("transient renewal failure")

    attempts = _count_heartbeat_renewals(monkeypatch, transient_failure)
    registry.start()
    heartbeat = registry._heartbeat
    turn = asyncio.create_task(agent.run_turn("in-flight-turn"))
    try:
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        with to_fail_lock:
            to_fail[0] = 2

        def failed_indices() -> list[int]:
            return [index for index, (_, ok) in enumerate(attempts) if not ok]

        await _wait_for(
            lambda: len(failed_indices()) == 2
            and len(attempts) > failed_indices()[-1] + 1
        )
        failed = failed_indices()
        assert failed[1] == failed[0] + 1
        before, after = attempts[failed[0] - 1], attempts[failed[1] + 1]
        assert before[1] and after[1]
        assert after[0] - before[0] < _HEALTHY_LEASE_SECONDS

        # Let the relay act on everything it can now see.
        await asyncio.sleep(10 * registry._poll_seconds)
        assert registry.owner_lifecycle_status == "healthy"
        assert registry.owner_epoch == 1
        assert registry._terminal_control is None
        assert heartbeat.latest().control is None
        assert not turn.done()

        agent.release_operation.set()
        assert await turn == "in-flight-turn"
    finally:
        agent.release_operation.set()
        await asyncio.gather(turn, return_exceptions=True)
        await registry.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_ordinary_renewal_errors_past_the_lease_fence_within_it(
    tmp_path, monkeypatch
):
    """Retried errors never extend the lease: it is judged against the last
    successful renewal, so the owner fences once one lease has passed."""

    db, _store, registry = await _owner_registry(
        tmp_path / "renewal-errors-past-lease.db",
        poll_seconds=0.02,
        owner_lease_seconds=_HEALTHY_LEASE_SECONDS,
    )
    agent = _SelfFencingReplicaAgent("did:test:renewal-errors-past-lease")
    registry.attach(agent)
    failing = threading.Event()
    attempts = _count_heartbeat_renewals(
        monkeypatch,
        lambda: (
            ConnectionError("renewal store unavailable") if failing.is_set() else None
        ),
    )
    registry.start()
    heartbeat = registry._heartbeat
    turn = asyncio.create_task(agent.run_turn("in-flight-turn"))
    try:
        await asyncio.wait_for(agent.operation_started.wait(), timeout=10)
        await _wait_for(lambda: heartbeat.latest_renewed() is not None)
        failing.set()
        await _wait_for(lambda: registry._lease_lost, timeout=10)
        fenced_at = time.monotonic()

        last_renewed = heartbeat.latest_renewed()
        failures = [started for started, ok in attempts if not ok]
        # Retried every interval, and none of it renewed the lease.
        assert len(failures) >= 2
        assert min(failures) >= last_renewed.started
        fenced_after = fenced_at - last_renewed.started
        assert fenced_after >= _HEALTHY_LEASE_SECONDS
        assert fenced_after < _HEALTHY_LEASE_SECONDS + 0.25
        reason = registry.owner_fence_reason
        assert "owner lease not renewed for" in reason
        assert "its last renewal failed (ConnectionError)" in reason
        assert registry._terminal_control is None
        with pytest.raises(InvocationSelfFencedError):
            await asyncio.wait_for(turn, timeout=10)

        # The errors were not control: once renewals succeed again, the
        # relay re-establishes ownership and new work is admitted.
        failing.clear()
        await _wait_for(lambda: registry.owner_lifecycle_status == "healthy")
        assert registry.owner_epoch == 2
        later = _SelfFencingReplicaAgent("did:test:renewal-errors-past-lease")
        assert await registry.register(later, "later-turn", 1)
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


@asynccontextmanager
async def _saturated_default_executor():
    """Occupy the only worker of this loop's default executor until exit."""

    loop = asyncio.get_running_loop()
    prior = getattr(loop, "_default_executor", None)
    executor = ThreadPoolExecutor(max_workers=1)
    occupied, release = threading.Event(), threading.Event()

    def occupy() -> None:
        occupied.set()
        release.wait(30)

    loop.set_default_executor(executor)
    blocker = loop.run_in_executor(None, occupy)
    try:
        await _wait_for(occupied.is_set)
        yield
    finally:
        release.set()
        await asyncio.wait_for(blocker, timeout=10)
        if prior is None:
            loop._default_executor = None
        else:
            loop.set_default_executor(prior)
        executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("heartbeat_state", ["exits", "is held"])
async def test_close_waits_for_the_heartbeat_without_a_default_executor_worker(
    tmp_path, monkeypatch, heartbeat_state
):
    """Blocking work occupying the loop's default executor cannot hold close
    past its budget: it waits for the heartbeat thread from the loop."""

    from kestrel_sovereign.stop import invocation

    monkeypatch.setattr(invocation, "_DEFAULT_CLOSE_DRAIN_SECONDS", 0.02)
    db, _store, registry = await _owner_registry(tmp_path / "busy-executor.db")
    hold, entered, release = _hold_heartbeat_renewal(monkeypatch)
    closing = None
    try:
        registry.start()
        heartbeat = registry._heartbeat
        if heartbeat_state == "is held":
            hold.set()
            await _wait_for(entered.is_set)
        else:
            await _wait_for(lambda: heartbeat.latest() is not None)

        async with _saturated_default_executor():
            started = time.monotonic()
            closing = asyncio.create_task(registry.close())
            done, _ = await asyncio.wait({closing}, timeout=5)
            assert closing in done, (
                "registry close waited for a worker of the loop's default executor"
            )
            took = time.monotonic() - started
            if heartbeat_state == "exits":
                await closing
                assert not heartbeat.alive
                assert heartbeat.retirement_error() is None
            else:
                with pytest.raises(TimeoutError) as caught:
                    await closing
                assert "heartbeat has not retired: its thread has not exited" in (
                    str(caught.value)
                )
                assert heartbeat.alive
            # Its own budget: the heartbeat join waits at most about 1 s.
            assert took < 3.0
    finally:
        release.set()
        if closing is not None:
            await asyncio.wait({closing}, timeout=10)
            if closing.done() and not closing.cancelled():
                closing.exception()  # reported by the assertions above
        try:
            await registry.close()
        finally:
            await db.close()


def test_a_connection_that_did_not_close_stays_owned(tmp_path):
    """A heartbeat whose connection's close failed has not retired."""

    class _CloseFailed(Exception):
        pass

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
        assert heartbeat.ensure_running() is False
        assert not heartbeat.alive
    finally:
        await registry.close()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_restart_whose_thread_cannot_start_is_a_named_launch(monkeypatch):
    """``ensure_running()`` reports a thread it could not start, rather than
    raising out of admission; the launch is counted and named until the next
    one, which starts normally (#3987)."""

    @asynccontextmanager
    async def opener():
        yield object()

    attempts = 0

    async def renew(_database, _owner_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _HeartbeatKilled()

    refuse_start = _refusable_heartbeat_start(monkeypatch)
    heartbeat = OwnerHeartbeat(opener, renew, interval_seconds=0.01)
    heartbeat.start("owner-whose-restart-cannot-start")
    try:
        await _wait_for(lambda: not heartbeat.alive)
        refuse_start.set()
        assert heartbeat.ensure_running() is False
        assert heartbeat.launches == 2
        assert not heartbeat.alive
        assert heartbeat.exit_cause() == (
            "a new thread could not be started (RuntimeError: can't start new thread)"
        )
        assert heartbeat.describe(time.monotonic(), _HEALTHY_LEASE_SECONDS) == (
            "a new heartbeat thread could not be started "
            "(RuntimeError: can't start new thread)"
        )

        refuse_start.clear()
        assert heartbeat.ensure_running() is True
        assert heartbeat.launches == 3
        await _wait_for(
            lambda: heartbeat.renewed_since_launch("owner-whose-restart-cannot-start")
        )
        assert heartbeat.exit_cause() is None
    finally:
        refuse_start.clear()
        assert await heartbeat.stop(timeout=10)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_retry_in_flight_still_names_the_failure_it_retries():
    """Health reads describe() whenever it is asked, often mid-retry: the
    failure being retried stays named rather than hidden by the retry."""

    @asynccontextmanager
    async def opener():
        yield object()

    attempts = 0
    retrying, release = threading.Event(), threading.Event()

    async def renew(_database, _owner_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("renewal store unavailable")
        retrying.set()
        await asyncio.to_thread(release.wait, 30)

    heartbeat = OwnerHeartbeat(opener, renew, interval_seconds=0.01)
    heartbeat.start("owner-retrying-a-failure")
    try:
        assert await asyncio.to_thread(retrying.wait, 10)
        description = heartbeat.describe(time.monotonic(), _HEALTHY_LEASE_SECONDS)
        assert description.startswith(
            "its last renewal failed (ConnectionError); its retry has been in "
            "flight for "
        )
    finally:
        release.set()
        assert await heartbeat.stop(timeout=10)


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
    assert heartbeat.exit_cause() == "the thread that started it has ended"


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
