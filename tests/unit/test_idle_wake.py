"""#3664: a drained work queue produces one triage wake.

On 2026-10-15 an agent finished its last item at 23:11Z with nothing queued
and nothing running in Talon. Nothing woke it for about 16 hours: every turn
starts from a wake, and when the last item finishes with nothing new
dispatched, no wake is pending.

These tests run the real idle detector over SQLite, the real wait-ledger,
conversation and restart-request stores, the real dispatcher and the real
``queue.drained`` source. Only the agent body is a double (the #3663 suite's
``_WaitingAgent``: it binds the dispatcher's ``session_id`` as the turn's
session and writes its rows there), plus a Talon provider double that answers
the two read-only calls the detector makes and refuses ``poll()``. Time is a
clock the test advances; each ``_tick`` advances it five minutes, the cron's
cadence. Starting the agent twice over the same databases without stopping it
is two replicas sharing one database.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kestrel_sdk.signals import RateLimit, Visibility
from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign.features.restart_coordinator.store import (
    ensure_restart_requests_table,
)
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.scheduler.runner import SchedulerRunner
from kestrel_sovereign.signals import (
    OrderedLockManager,
    SignalDispatcher,
    SignalLogStore,
    SourceRegistry,
)
from kestrel_sovereign.signals import idle_wake
from kestrel_sovereign.signals.idle_wake import (
    MAX_OBSERVATION_GAP,
    MAX_WAKE_ATTEMPTS,
    OWNER_STALE_AFTER,
    IdleWakeConfig,
    IdleWakeDetector,
    load_idle_wake_config,
    run_idle_wake,
)
from kestrel_sovereign.signals.sources.queue_drained import (
    SOURCE_NAME as QUEUE_DRAINED,
    build_queue_drained_registration,
)
from kestrel_sovereign.signals.sources.scheduler import CRON_TASKS
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.turn_provenance import turn_opener_metadata
from kestrel_sovereign.storage.turn_provenance import wake_reply_metadata
from tests.unit.test_wait_watch_origin_session import AGENT_DID, _WaitingAgent

CONFIG = IdleWakeConfig(enabled=True, idle_minutes=30)
# The chain session the agent's last delivered wake ran in.
CHAIN = "chain-3664"
OPERATOR = "operator-chat-3664"
TERMINAL = {"complete", "failed", "reject", "finished_unknown", "cancelled"}
# The owner of a running wake renews its lease this often. Real time, unlike
# the test clock, so a held turn's lease is renewed at the clock's new time.
LEASE_INTERVAL = timedelta(milliseconds=10)
LEASE_TASK = "idle_wake_lease:"


class _IdleAgent(_WaitingAgent):
    """The #3663 double, plus the reply stamp a real wake turn writes (#3655),
    so a restart can tell a wake that ran from one that did not."""

    async def process_input(
        self, prompt: str, session_id=None, signal_wake=None, wake_reply=None,
        **_kwargs,
    ):
        async with self._turn_lifecycle():
            self._active_session_id = session_id
            self.wake_sessions.append(session_id)
            user_meta = {"sent_form": True, **turn_opener_metadata(signal_wake)}
            await self.conversation.add_conversation(
                "user", prompt, metadata=user_meta, session_id=session_id
            )
            reply_meta = (
                wake_reply_metadata(wake_reply.signal_id) if wake_reply else {}
            )
            await self.conversation.add_conversation(
                "assistant", "nothing actionable", metadata=reply_meta,
                session_id=session_id,
            )
            return "nothing actionable"


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 15, 23, 11, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: float) -> None:
        self.now += timedelta(minutes=minutes)


class _Talon:
    """The two read-only calls the detector makes on Talon's provider."""

    kind = "talon"
    signal = "talon.job_complete"

    def __init__(self) -> None:
        self.jobs: dict = {}
        self.origins: dict = {}

    async def finished_runs(self):
        return {
            "complete": True,
            "runs": [
                {"job_id": job, "repo": "o/r", "issue": "1",
                 "disposition": "completed", "completed_at": "2026-10-15T23:00:00+00:00"}
                for job, status in self.jobs.items()
                if status in TERMINAL
            ],
        }

    async def active_handles(self):
        return list(self.jobs)

    async def origin_session_id(self, handle):
        return self.origins.get(handle)

    async def poll(self, handle):
        raise AssertionError("the idle detector must never poll Talon")


class _CI:
    """A poll-only provider, so a ``ci:`` watch counts as armed."""

    kind = "ci"

    async def poll(self, handle):
        return WaitStatus(Outcome.PENDING, "running")


@pytest.fixture
async def boot(tmp_path):
    """Start the agent over this test's databases. Starting again after
    ``stop()`` is a restart: new process state, the same durable rows."""
    running: list = []
    clock = _Clock()

    async def start():
        db = await AsyncDatabase.sqlite(str(tmp_path / "agent.db"))
        backend = SQLiteBackend(str(tmp_path / "signal_log.db"))
        await backend.connect()
        log = SignalLogStore(backend)
        await log.initialize()
        sources = SourceRegistry()
        # The shipped per-minute cap is real-time; these tests compress hours
        # into milliseconds. The cap itself is pinned separately.
        sources.register(
            dataclasses.replace(
                build_queue_drained_registration(), rate_limit=RateLimit()
            )
        )
        conversation = AsyncConversationStore(db, agent_id=AGENT_DID)
        agent = _IdleAgent(conversation, db)
        agent._raw_storage = SimpleNamespace(db=db, conversation=conversation)
        agent.features = {}
        locks = OrderedLockManager()
        agent._lock_manager = locks
        dispatcher = SignalDispatcher(
            agent=agent, registry=sources, lock_manager=locks, store=log,
        )
        agent.dispatcher = dispatcher
        wakes: list = []
        enqueue = dispatcher.enqueue_signal

        async def capture(signal, **kwargs):
            wakes.append(signal)
            return await enqueue(signal, **kwargs)

        dispatcher.enqueue_signal = capture
        events: list = []

        async def listen(event_type, data):
            events.append((event_type, data))

        agent.add_event_listener(listen)
        detector = IdleWakeDetector(
            agent, clock=clock, heartbeat_interval=LEASE_INTERVAL
        )
        agent._idle_wake_detector = detector
        talon = _Talon()
        agent.wait_registry.register(talon)

        rig = SimpleNamespace(
            agent=agent,
            db=db,
            conversation=conversation,
            detector=detector,
            wakes=wakes,
            events=events,
            talon=talon,
            clock=clock,
            followups=0,
            wait_store=detector._wait_store,
            # Gates holding this rig's wake turns; teardown opens them, so a
            # failing test fails instead of waiting on its held turn.
            gates=[],
        )

        async def stop():
            running.remove(rig)
            await _settle(rig)
            await backend.close()
            await db.close()

        async def crash():
            """The process stops: its running turns and lease tasks die
            with it, as a shutdown cancels them."""
            running.remove(rig)
            alive = [t for t in rig.agent.background_tasks if not t.done()]
            for task in alive:
                task.cancel()
            await asyncio.gather(*alive, return_exceptions=True)
            await backend.close()
            await db.close()

        rig.stop = stop
        rig.crash = crash
        running.append(rig)
        return rig

    yield start

    for rig in running:
        for gate in rig.gates:
            gate.set()
    for rig in list(running):
        await rig.stop()


async def _settle(rig) -> None:
    """Let every enqueued wake's turn run to completion."""
    pending = [t for t in rig.agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _evaluate(
    rig, minutes: float = 5, config: IdleWakeConfig = CONFIG
) -> dict:
    """One cron evaluation, ``minutes`` after the last. A wake it enqueues
    is left running."""
    rig.clock.advance(minutes)

    async def followups():
        return rig.followups

    return await rig.detector.tick(config, pending_followups=followups)


async def _tick(rig, minutes: float = 5, config: IdleWakeConfig = CONFIG) -> dict:
    """One cron evaluation, ``minutes`` after the last, with any wake it
    enqueued run to completion."""
    report = await _evaluate(rig, minutes, config)
    await _settle(rig)
    return report


async def _run_for(rig, minutes: int) -> list:
    reports = []
    for _ in range(minutes // 5):
        reports.append(await _tick(rig))
    return reports


def _drained(rig) -> list:
    return [s for s in rig.wakes if s.source == QUEUE_DRAINED]


async def _say(rig, session_id: str, text: str = "hello", signal_wake=None) -> None:
    """A turn's opening row, as the agent writes it."""
    await rig.conversation.add_conversation(
        "user", text,
        metadata={"sent_form": True, **turn_opener_metadata(signal_wake)},
        session_id=session_id,
    )


async def _deliver(rig, kind: str, handle: str, *, origin=None) -> None:
    """A wait-ledger wake for ``kind:handle`` that was delivered now."""
    if origin is not None:
        await rig.wait_store.start_watch(kind, handle, origin_session_id=origin)
    await rig.wait_store.record_delivery(
        kind, handle,
        delivery_status="ok_queued",
        signaled_outcome="success",
        attempt_at=rig.clock.now,
        delivered_at=rig.clock.now,
    )


async def _fire_once(rig) -> None:
    """Run until the first wake fires. Its owner settles it as its turn
    ends, so the next evaluation finds the stretch spent."""
    for _ in range(12):
        report = await _tick(rig)
        if report.get("fired"):
            break
    else:
        raise AssertionError(f"no wake fired: {report}")
    await _assert_settled(rig, "ok")
    after = await _tick(rig)
    assert "harvested" not in after, after
    assert after["verdict"] == "already_fired", after


async def _assert_settled(rig, status: str) -> None:
    state = await rig.detector._state_store.load()
    assert state.pending_signal_id is None, state
    assert state.pending_owner is None and state.pending_heartbeat_at is None
    assert state.last_wake_status == status, state
    assert state.fired_status == status, state


def _hold_turns(rig) -> asyncio.Event:
    """Hold every wake turn on ``rig`` until the returned gate opens."""
    gate = asyncio.Event()
    rig.gates.append(gate)
    turn = rig.agent.process_input

    async def held(prompt, **kwargs):
        await gate.wait()
        return await turn(prompt, **kwargs)

    rig.agent.process_input = held
    return gate


async def _fire_held(rig) -> dict:
    """Run until a wake fires, leaving its (held) turn running."""
    for _ in range(12):
        report = await _evaluate(rig)
        if report.get("fired"):
            return report
    raise AssertionError(f"no wake fired: {report}")


def _lease_tasks(rig) -> list:
    return [
        t for t in rig.agent.background_tasks
        if t.get_name().startswith(LEASE_TASK) and not t.done()
    ]


async def _lease_renewed(rig) -> None:
    """Wait until the owner's lease task has renewed at the clock's time."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while (
        await rig.detector._state_store.load()
    ).pending_heartbeat_at != rig.clock.now.isoformat():
        assert loop.time() < deadline, "the owner never renewed its lease"
        await asyncio.sleep(0.005)


async def _fire_and_lose_the_outcome(rig) -> None:
    """Fire, and let the wake's turn store its reply, but lose the owner's
    record of how it ended: the lease task dies first, as it does when the
    process stops between the reply and the settle."""
    gate = _hold_turns(rig)
    await _fire_held(rig)
    [lease] = _lease_tasks(rig)
    lease.cancel()
    await asyncio.gather(lease, return_exceptions=True)
    gate.set()
    await _settle(rig)


async def _restart_rows(rig, *rows) -> None:
    rig.agent.features["RestartCoordinatorFeature"] = object()
    await ensure_restart_requests_table(rig.db)
    for request_id, owner, status, wake_delivered in rows:
        await rig.db.execute(
            "INSERT INTO restart_requests "
            "(id, requested_by_agent, reason, requested_at, status, wake_delivered) "
            "VALUES (?, ?, 'test', ?, ?, ?)",
            (request_id, owner, rig.clock.now.isoformat(), status, wake_delivered),
        )


# ---------------------------------------------------------------------------
# A drained queue fires exactly one wake after N minutes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_drained_queue_fires_exactly_one_wake_after_n_minutes(boot):
    rig = await boot()
    rig.talon.jobs = {"job-3659": "complete"}

    first = await _tick(rig, minutes=0)
    assert first["verdict"] == "idle", first
    went_idle = rig.clock.now

    reports = await _run_for(rig, 25)
    assert [r["verdict"] for r in reports] == ["idle"] * 5
    assert _drained(rig) == []

    fired = await _tick(rig)
    assert fired["verdict"] == "fired", fired
    assert rig.clock.now - went_idle == timedelta(minutes=30)
    assert len(_drained(rig)) == 1
    wake = _drained(rig)[0]
    assert wake.payload["idle_minutes"] == 30
    assert wake.payload["attempt"] == 1
    assert len(rig.agent.wake_sessions) == 1, "the wake ran one turn"

    # Its owner settled it when the turn ended.
    await _assert_settled(rig, "ok")

    # Hours more of the same quiet: nothing else.
    later = await _run_for(rig, 180)
    assert {r["verdict"] for r in later} == {"already_fired"}
    assert len(_drained(rig)) == 1
    assert len(rig.agent.wake_sessions) == 1


@pytest.mark.asyncio
async def test_the_wake_reports_the_queue_as_empty_and_dispatches_nothing(boot):
    rig = await boot()
    await _fire_once(rig)

    prompt_rows = await rig.conversation.get_conversation_history(limit=10)
    prompt = next(
        row["content"] for row in prompt_rows if row["role"] == "user"
    )
    assert prompt.startswith("[QUEUE_DRAINED]")
    assert "continuous-work doctrine" in prompt
    assert "dispatched nothing" in prompt
    assert rig.talon.jobs == {}, "the detector starts no work of its own"


# ---------------------------------------------------------------------------
# Work in progress, or a wake on its way, suppresses the wake
# ---------------------------------------------------------------------------


async def _talon_job_in_flight(rig):
    rig.talon.jobs = {"job-1": "running"}


async def _cancel_unrecorded_job(rig):
    rig.talon.jobs = {"job-1": "cancel_unrecorded"}


async def _watch_armed(rig):
    rig.agent.wait_registry.register(_CI())
    await rig.wait_store.start_watch("ci", "o/r#1", origin_session_id=CHAIN)


async def _restart_pending(rig):
    await _restart_rows(rig, ("req-1", AGENT_DID, "pending", 0))


async def _restart_executing(rig):
    await _restart_rows(rig, ("req-1", AGENT_DID, "executing", 0))


async def _restart_wake_undelivered(rig):
    await _restart_rows(rig, ("req-1", AGENT_DID, "completed", 0))


async def _wake_in_flight(rig):
    await rig.wait_store.record_pending(
        "talon", "job-1", signal_id="sig-1", target="success", attempts=1,
    )


async def _wake_parked(rig):
    rig.agent.wait_registry.register(_CI())
    await rig.wait_store.record_deferral(
        "ci", "o/r#1", target="success",
        retry_at=rig.clock.now + timedelta(hours=2),
    )


async def _durable_delivery_pending(rig):
    async def listing(*, consumer_id, statuses, limit):
        return [SimpleNamespace(consumer_id=consumer_id)]

    rig.agent.dispatcher.list_durable_deliveries = listing


async def _self_followup_scheduled(rig):
    rig.followups = 1


async def _turn_in_progress(rig):
    rig.agent._live_turn_id = "turn_live"


SUPPRESSORS = {
    "talon job in flight": (_talon_job_in_flight, "talon_job:job-1"),
    "talon cancel unrecorded": (_cancel_unrecorded_job, "talon_job:job-1"),
    "watch armed": (_watch_armed, "watch_armed:ci:o/r#1"),
    "restart pending": (_restart_pending, "restart_pending:req-1"),
    "restart executing": (_restart_executing, "restart_executing:req-1"),
    "restart wake undelivered": (
        _restart_wake_undelivered, "restart_wake_undelivered:req-1"
    ),
    "wait wake in flight": (_wake_in_flight, "wake_in_flight:talon:job-1"),
    "wait wake parked": (_wake_parked, "wake_parked:ci:o/r#1"),
    "durable delivery pending": (
        _durable_delivery_pending,
        "durable_delivery_pending:core.channel-cognition-v1",
    ),
    "self_followup scheduled": (_self_followup_scheduled, "self_followup_scheduled:1"),
    "turn in progress": (_turn_in_progress, "turn_in_progress"),
}


@pytest.mark.parametrize("name", sorted(SUPPRESSORS))
@pytest.mark.asyncio
async def test_work_or_a_pending_wake_suppresses_the_wake(boot, name):
    rig = await boot()
    arrange, reason = SUPPRESSORS[name]
    await arrange(rig)

    reports = [await _tick(rig, minutes=0)] + await _run_for(rig, 120)

    assert _drained(rig) == [], reports[-1]
    assert {r["verdict"] for r in reports} == {"busy"}
    reasons = reports[-1]["work"] + reports[-1]["deliveries"]
    assert reason in reasons, reports[-1]


@pytest.mark.asyncio
async def test_the_idle_clock_restarts_when_the_suppression_clears(boot):
    rig = await boot()
    rig.talon.jobs = {"job-1": "running"}
    await _run_for(rig, 60)

    rig.talon.jobs = {"job-1": "complete"}
    cleared = rig.clock.now + timedelta(minutes=5)
    reports = await _run_for(rig, 30)
    assert _drained(rig) == [], "the agent has only been idle 25 minutes"
    assert reports[0]["idle_since"] == cleared.isoformat()

    assert (await _tick(rig))["verdict"] == "fired"


@pytest.mark.asyncio
async def test_a_watch_no_provider_can_poll_does_not_hold_the_agent_quiet(boot):
    """A watch on a kind whose feature is not loaded can never fire."""
    rig = await boot()
    await rig.wait_store.start_watch("gone", "h1", origin_session_id=CHAIN)

    await _fire_once(rig)
    assert len(_drained(rig)) == 1


@pytest.mark.asyncio
async def test_another_agents_restart_request_does_not_suppress(boot):
    rig = await boot()
    await _restart_rows(rig, ("req-peer", "did:test:someone-else", "pending", 0))

    await _fire_once(rig)
    assert len(_drained(rig)) == 1


# ---------------------------------------------------------------------------
# No re-fire until state changes
# ---------------------------------------------------------------------------


async def _operator_turn(rig):
    await _say(rig, OPERATOR, "anything new?")


async def _channel_message(rig):
    await _say(rig, OPERATOR, "via telegram", signal_wake={"source": "channel.message"})


async def _peer_turn(rig):
    await _say(rig, "peer-session", "please review", signal_wake={"source": "a2a.task_submitted"})


async def _job_dispatched_and_finished_between_ticks(rig):
    rig.talon.jobs["job-new"] = "complete"


async def _job_dispatched(rig):
    rig.talon.jobs["job-new"] = "running"


async def _job_dispatched_while_talon_unloaded(rig):
    rig.agent.wait_registry.unregister("talon")
    await _tick(rig)
    rig.talon.jobs["job-new"] = "complete"
    rig.agent.wait_registry.register(rig.talon)


async def _restart_filed_and_completed(rig):
    await _restart_rows(rig, ("req-self", AGENT_DID, "completed", 1))


async def _watch_armed_and_fired_between_ticks(rig):
    await _deliver(rig, "ci", "o/r#2", origin=CHAIN)


async def _watch_armed_and_left(rig):
    await _watch_armed(rig)


STATE_CHANGES = {
    "operator turn": _operator_turn,
    "operator channel message": _channel_message,
    "a2a turn": _peer_turn,
    "talon job dispatched and finished": _job_dispatched_and_finished_between_ticks,
    "talon job dispatched": _job_dispatched,
    "talon job dispatched while unloaded": _job_dispatched_while_talon_unloaded,
    "restart filed": _restart_filed_and_completed,
    "watch armed and fired": _watch_armed_and_fired_between_ticks,
    "watch armed": _watch_armed_and_left,
}


@pytest.mark.parametrize("name", sorted(STATE_CHANGES))
@pytest.mark.asyncio
async def test_no_re_fire_until_state_changes(boot, name):
    rig = await boot()
    await _restart_rows(rig)  # the restart table exists from the start
    await _fire_once(rig)
    await _run_for(rig, 120)
    assert len(_drained(rig)) == 1

    await STATE_CHANGES[name](rig)
    await _tick(rig)
    # Whatever started is allowed to finish.
    rig.talon.jobs = {job: "complete" for job in rig.talon.jobs}
    await rig.wait_store.stop_watch("ci", "o/r#1")
    reports = await _run_for(rig, 60)

    assert len(_drained(rig)) == 2, [r["verdict"] for r in reports]
    second = _drained(rig)[1]
    assert second.payload["attempt"] == 1, "a new stretch, not a retry"


async def _autonomous_wake_turn(rig):
    await _say(rig, CHAIN, "[HEARTBEAT]", signal_wake={"source": "heartbeat"})


async def _wake_parked_then_delivered_without_new_work(rig):
    rig.agent.wait_registry.register(_CI())
    await rig.wait_store.record_deferral(
        "ci", "o/r#9", target="success", retry_at=rig.clock.now,
    )
    await _tick(rig)
    await rig.db.execute(
        "UPDATE wait_signal_state SET delivery_deferred_until = NULL "
        "WHERE handle = 'o/r#9'"
    )


async def _operator_signal_notice(rig):
    await rig.conversation.add_conversation(
        "user", "notice", metadata={"operator_signal": True}, session_id=CHAIN,
    )


NOT_STATE_CHANGES = {
    "autonomous wake turn": _autonomous_wake_turn,
    "transient wake delivery": _wake_parked_then_delivered_without_new_work,
    "operator-signal notice": _operator_signal_notice,
    "watch on an unloaded kind": lambda rig: rig.wait_store.start_watch("gone", "h"),
}


@pytest.mark.parametrize("name", sorted(NOT_STATE_CHANGES))
@pytest.mark.asyncio
async def test_what_is_not_new_work_does_not_re_fire(boot, name):
    rig = await boot()
    await _fire_once(rig)

    await NOT_STATE_CHANGES[name](rig)
    await _run_for(rig, 120)

    assert len(_drained(rig)) == 1


async def _talon_job_pruned(rig):
    rig.talon.jobs.pop("job-old")


async def _restart_row_deleted(rig):
    await rig.db.execute("DELETE FROM restart_requests WHERE id = 'req-old'")


async def _newest_delivery_deleted(rig):
    await rig.db.execute("DELETE FROM wait_signal_state WHERE handle = 'o/r#old'")


async def _talon_provider_unloaded_and_reloaded(rig):
    rig.agent.wait_registry.unregister("talon")
    await _tick(rig)
    rig.agent.wait_registry.register(rig.talon)


REMOVALS = {
    "talon job pruned": _talon_job_pruned,
    "restart request deleted": _restart_row_deleted,
    "delivered wake row deleted": _newest_delivery_deleted,
    "talon provider reloaded": _talon_provider_unloaded_and_reloaded,
}


@pytest.mark.parametrize("name", sorted(REMOVALS))
@pytest.mark.asyncio
async def test_evidence_that_disappears_is_not_new_work(boot, name):
    rig = await boot()
    rig.talon.jobs = {"job-old": "complete"}
    await _restart_rows(rig, ("req-old", AGENT_DID, "completed", 1))
    await _deliver(rig, "ci", "o/r#old", origin=CHAIN)
    await _fire_once(rig)

    await REMOVALS[name](rig)
    await _run_for(rig, 120)

    assert len(_drained(rig)) == 1


@pytest.mark.asyncio
async def test_an_operator_turn_before_the_fire_restarts_the_clock(boot):
    rig = await boot()
    await _tick(rig, minutes=0)
    await _run_for(rig, 20)
    await _say(rig, OPERATOR)
    reports = await _run_for(rig, 30)
    # The clock restarts at the first evaluation that sees the turn.
    assert reports[0]["activity"][0].startswith("operator_turn:")
    assert reports[0]["idle_since"] == (rig.clock.now - timedelta(minutes=25)).isoformat()
    assert _drained(rig) == []

    assert (await _tick(rig))["verdict"] == "fired"


@pytest.mark.asyncio
async def test_work_the_triage_turn_starts_ends_the_stretch(boot):
    """The doctrine's happy path: triage dispatches, the job finishes, the
    queue drains again, and the next stretch gets its own wake."""
    rig = await boot()
    await _fire_once(rig)

    rig.talon.jobs["job-triaged"] = "running"
    busy = await _run_for(rig, 60)
    assert {r["verdict"] for r in busy} == {"busy"}

    rig.talon.jobs["job-triaged"] = "complete"
    await _run_for(rig, 35)
    assert len(_drained(rig)) == 2


# ---------------------------------------------------------------------------
# The wake starts a turn in the agent's chain session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_wake_starts_a_turn_in_the_chain_session(boot):
    rig = await boot()
    await _say(rig, OPERATOR, "operator chat that must not be joined")
    await _deliver(rig, "ci", "o/r#3659", origin=CHAIN)

    await _fire_once(rig)

    wake = _drained(rig)[0]
    assert wake.session_id == CHAIN
    assert wake.visibility is Visibility.USER_VISIBLE
    assert rig.agent.wake_sessions == [CHAIN]
    rows = await rig.conversation.get_conversation_history(limit=10)
    wake_rows = [
        r for r in rows
        if (r.get("metadata") or {}).get("signal_wake", {}).get("source") == QUEUE_DRAINED
    ]
    assert [r["metadata"]["session_id"] for r in wake_rows] == [CHAIN]
    surfaced = [d for kind, d in rig.events if kind == "signal_completed"]
    assert surfaced and surfaced[-1]["result_summary"], "the bound wake renders live"


@pytest.mark.asyncio
async def test_the_chain_session_falls_back_to_the_providers_own_record(boot):
    """The same rule the reconciler uses (#3663): the watch row's session,
    else the provider's ``origin_session_id``."""
    rig = await boot()
    rig.talon.jobs = {"job-3659": "complete"}
    rig.talon.origins = {"job-3659": CHAIN}
    await _deliver(rig, "talon", "job-3659")

    await _fire_once(rig)

    assert _drained(rig)[0].session_id == CHAIN
    assert rig.agent.wake_sessions == [CHAIN]


@pytest.mark.asyncio
async def test_the_newest_delivered_wake_names_the_chain(boot):
    rig = await boot()
    await _deliver(rig, "ci", "old", origin="old-chain")
    rig.clock.advance(1)
    await _deliver(rig, "ci", "new", origin=CHAIN)

    await _fire_once(rig)

    assert _drained(rig)[0].session_id == CHAIN


@pytest.mark.asyncio
async def test_with_no_chain_the_wake_gets_a_fresh_session_not_the_operators(boot):
    rig = await boot()
    await _say(rig, OPERATOR, "operator chat that must not be joined")

    await _fire_once(rig)

    wake = _drained(rig)[0]
    assert wake.session_id is None
    assert wake.visibility is Visibility.INTERNAL
    [session] = rig.agent.wake_sessions
    assert session and session != OPERATOR


# ---------------------------------------------------------------------------
# The idle stretch survives a restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_idle_stretch_survives_a_restart_and_does_not_re_fire(boot):
    rig = await boot()
    await _fire_once(rig)
    await rig.stop()

    rig = await boot()
    reports = [await _tick(rig, minutes=2)] + await _run_for(rig, 180)

    assert _drained(rig) == []
    assert {r["verdict"] for r in reports} == {"already_fired"}

    # ...and a new stretch after the restart still gets its wake.
    await _say(rig, OPERATOR)
    await _run_for(rig, 35)
    assert len(_drained(rig)) == 1


@pytest.mark.asyncio
async def test_a_restart_does_not_reset_an_idle_clock_it_kept_watching(boot):
    rig = await boot()
    await _tick(rig, minutes=0)
    await _run_for(rig, 20)
    await rig.stop()

    rig = await boot()
    await _run_for(rig, 10)
    assert len(_drained(rig)) == 1


@pytest.mark.asyncio
async def test_an_unwatched_gap_restarts_the_idle_clock(boot):
    """The detector vouches only for idle time it observed."""
    rig = await boot()
    await _tick(rig, minutes=0)
    await _run_for(rig, 20)

    gap = MAX_OBSERVATION_GAP + timedelta(seconds=1)
    report = await _tick(rig, minutes=gap.total_seconds() / 60)
    assert report["verdict"] == "idle"
    assert report["idle_since"] == rig.clock.now.isoformat()
    assert _drained(rig) == []


# ---------------------------------------------------------------------------
# The observation gap follows the cron's cadence, not the idle threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_one_interval_threshold_fires_although_the_cron_runs_late(boot):
    """``idle_minutes = 5`` is one cron interval. Scheduler delay puts
    evaluations a little more than five minutes apart, and that must not
    restart the clock every time."""
    rig = await boot()
    five = IdleWakeConfig(enabled=True, idle_minutes=5)
    late = 300.001 / 60  # minutes

    first = await _tick(rig, minutes=0, config=five)
    assert first["verdict"] == "idle", first
    went_idle = first["idle_since"]

    fired = await _tick(rig, minutes=late, config=five)
    assert fired["verdict"] == "fired", fired
    assert fired["idle_since"] == went_idle
    assert len(_drained(rig)) == 1


@pytest.mark.asyncio
async def test_a_skipped_occurrence_does_not_restart_the_clock(boot):
    """Two intervals and a little scheduler delay is still a watched stretch."""
    rig = await boot()
    first = await _tick(rig, minutes=0)
    went_idle = first["idle_since"]

    reports = [await _tick(rig, minutes=11.5) for _ in range(2)]
    assert {r["idle_since"] for r in reports} == {went_idle}
    fired = await _tick(rig, minutes=7)
    assert fired["verdict"] == "fired", fired


@pytest.mark.asyncio
async def test_a_long_threshold_does_not_stretch_the_unwatched_gap(boot):
    """With ``idle_minutes = 60`` a 20-minute hole in the evaluations is still
    a hole: the allowance comes from the cadence, not the threshold."""
    rig = await boot()
    hour = IdleWakeConfig(enabled=True, idle_minutes=60)
    await _tick(rig, minutes=0, config=hour)
    await _tick(rig, config=hour)

    report = await _tick(rig, minutes=20, config=hour)
    assert report["idle_since"] == rig.clock.now.isoformat(), report


@pytest.mark.asyncio
async def test_a_wake_that_ran_before_a_restart_is_not_fired_again(boot):
    """The restart lost the harvest, not the turn: its stored reply says the
    wake was delivered, so the stretch is spent."""
    rig = await boot()
    await _fire_and_lose_the_outcome(rig)
    assert len(rig.agent.wake_sessions) == 1
    await rig.stop()

    rig = await boot()
    reports = [await _tick(rig)] + await _run_for(rig, 120)

    assert reports[0]["harvested"]["status"] == "replied_before_restart"
    assert _drained(rig) == []
    assert {r["verdict"] for r in reports} == {"already_fired"}


@pytest.mark.asyncio
async def test_a_wake_the_restart_lost_before_it_replied_is_retried(boot):
    rig = await boot()
    _hold_turns(rig)
    await _fire_held(rig)
    await rig.crash()  # the process went away before the reply was stored

    rig = await boot()
    report = await _tick(rig)
    assert report["harvested"]["status"] == "lost_at_restart"
    assert report["verdict"] == "fired"
    assert _drained(rig)[0].payload["attempt"] == 2


@pytest.mark.asyncio
async def test_an_unreadable_reply_lookup_leaves_the_wake_pending(boot):
    """Not knowing whether the wake ran is not evidence that it did not."""
    rig = await boot()
    await _fire_and_lose_the_outcome(rig)
    await rig.stop()

    rig = await boot()
    lookup = rig.conversation.find_signal_wake_reply_id

    async def locked(signal_id):
        raise RuntimeError("database is locked")

    rig.conversation.find_signal_wake_reply_id = locked
    stuck = await _run_for(rig, 60)
    assert {r["verdict"] for r in stuck} == {"unknown"}
    assert stuck[0]["unknown"][0].startswith("reply_lookup_failed")
    assert _drained(rig) == []

    rig.conversation.find_signal_wake_reply_id = lookup
    report = await _tick(rig)
    assert report["harvested"]["status"] == "replied_before_restart"
    await _run_for(rig, 120)
    assert _drained(rig) == []


@pytest.mark.asyncio
async def test_a_harvest_is_durable_before_the_evaluation_that_follows(boot):
    """The owner could not settle its finished wake, so its next evaluation
    does, from the dispatch it still holds. A failure after that harvest must
    not leave it to read as lost."""
    rig = await boot()
    await _fire_and_lose_the_outcome(rig)
    observe = rig.detector._observe

    async def broken(*args, **kwargs):
        raise RuntimeError("the database went away mid-tick")

    rig.detector._observe = broken
    with pytest.raises(RuntimeError):
        await _tick(rig)
    rig.detector._observe = observe
    await _assert_settled(rig, "ok")

    reports = await _run_for(rig, 120)
    assert "harvested" not in reports[0]
    assert len(_drained(rig)) == 1
    assert {r["verdict"] for r in reports} == {"already_fired"}


# ---------------------------------------------------------------------------
# Replicas sharing one database
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_wake_running_on_another_replica_is_not_dispatched_again(boot):
    """The cron's next occurrence can run on a replica that did not enqueue
    the wake. While the owner's turn runs, that replica sees a live lease,
    not a lost wake, and starts nothing."""
    a = await boot()
    b = await boot()
    gate = _hold_turns(a)
    fired = await _fire_held(a)

    for _ in range(24):  # two hours of evaluations on B
        a.clock.advance(5)
        await _lease_renewed(a)
        report = await _evaluate(b, minutes=0)
        assert report["verdict"] == "wake_in_flight_elsewhere", report
        assert report["signal_id"] == fired["signal_id"]
    assert len(_drained(a)) == 1
    assert _drained(b) == []

    # The turn ends; its owner settles the wake, and B finds the stretch spent.
    gate.set()
    await _settle(a)
    await _assert_settled(a, "ok")
    reports = await _run_for(b, 120)
    assert {r["verdict"] for r in reports} == {"already_fired"}
    assert _drained(b) == []
    assert len(a.agent.wake_sessions) == 1 and b.agent.wake_sessions == []


@pytest.mark.asyncio
async def test_a_replica_recovers_a_wake_only_once_its_owners_lease_lapses(boot):
    a = await boot()
    b = await boot()
    _hold_turns(a)
    await _fire_held(a)
    await a.crash()  # the owner's process stops with the turn unfinished

    stale_after = OWNER_STALE_AFTER.total_seconds() / 60
    within = await _evaluate(b, minutes=stale_after)
    assert within["verdict"] == "wake_in_flight_elsewhere", within

    lapsed = await _tick(b, minutes=1)
    assert lapsed["harvested"]["status"] == "lost_at_restart", lapsed
    # B watched the agent busy with the wake; the retry waits for a fresh
    # idle stretch of its own.
    assert lapsed["verdict"] == "idle", lapsed
    await _run_for(b, 30)

    [retry] = _drained(b)
    assert retry.payload["attempt"] == 2
    assert len(_drained(a)) == 1
    assert len(b.agent.wake_sessions) == 1


@pytest.mark.asyncio
async def test_a_renewal_that_lands_during_a_recovery_wins(boot):
    """A's process stalls past the lease, so B reads it as stale and starts
    recovering the wake. A renews before B writes: B's write is refused and
    nothing is dispatched again."""
    a = await boot()
    b = await boot()
    gate = _hold_turns(a)
    renew = a.detector._state_store.renew
    stalled = asyncio.Event()

    async def stalled_renew(*args):
        await stalled.wait()
        return await renew(*args)

    a.detector._state_store.renew = stalled_renew
    await _fire_held(a)

    lookup = b.detector._reply_outcome

    async def a_wakes_up_mid_recovery(signal_id):
        stalled.set()
        await _lease_renewed(a)
        return await lookup(signal_id)

    b.detector._reply_outcome = a_wakes_up_mid_recovery
    stale_after = OWNER_STALE_AFTER.total_seconds() / 60
    report = await _evaluate(b, minutes=stale_after + 1)
    assert report["verdict"] == "state_conflict", report
    assert _drained(b) == []

    b.detector._reply_outcome = lookup
    assert (await _evaluate(b, minutes=0))["verdict"] == "wake_in_flight_elsewhere"
    gate.set()
    await _settle(a)
    await _assert_settled(a, "ok")
    await _run_for(b, 60)
    assert _drained(b) == [] and len(_drained(a)) == 1


@pytest.mark.asyncio
async def test_two_replicas_evaluating_at_once_claim_one_wake(boot):
    """Both read the row before either claims it. The claim is a
    compare-and-set, so the second claim loses and enqueues nothing."""
    a = await boot()
    b = await boot()
    for _ in range(6):
        await _tick(a)
    raced: dict = {}
    chain_session = a.detector._chain_session

    async def b_evaluates_first():
        # A has read the row and decided to fire; B does the same and
        # claims before A does.
        raced["b"] = await _evaluate(b, minutes=0)
        await _settle(b)
        return await chain_session()

    a.detector._chain_session = b_evaluates_first
    report = await _tick(a)

    assert raced["b"]["verdict"] == "fired", raced["b"]
    assert report["verdict"] == "state_conflict", report
    assert report["superseded"] == "fired"
    assert _drained(a) == [] and len(_drained(b)) == 1
    await _assert_settled(b, "ok")
    assert (await _tick(a))["verdict"] == "already_fired"
    assert len(b.agent.wake_sessions) == 1 and a.agent.wake_sessions == []


@pytest.mark.asyncio
async def test_a_stale_evaluation_cannot_undo_a_settled_wake(boot):
    """B reads the row while A's wake is running, A settles it, then B writes
    what it read: the write is refused, and the stretch stays spent."""
    a = await boot()
    b = await boot()
    gate = _hold_turns(a)
    await _fire_held(a)
    stale = await b.detector._state_store.load()
    assert stale.pending_signal_id is not None

    gate.set()
    await _settle(a)
    await _assert_settled(a, "ok")

    assert await b.detector._state_store.save(stale) is None
    await _assert_settled(a, "ok")
    await _run_for(b, 120)
    assert _drained(b) == []


# ---------------------------------------------------------------------------
# A wake that fails is retried a bounded number of times
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_wake_is_retried_then_given_up(boot):
    rig = await boot()

    async def failing_turn(prompt, **kwargs):
        rig.agent.wake_sessions.append(kwargs.get("session_id"))
        raise RuntimeError("every model route is down")

    rig.agent.process_input = failing_turn
    reports = await _run_for(rig, 120)

    assert len(_drained(rig)) == MAX_WAKE_ATTEMPTS
    assert [w.payload["attempt"] for w in _drained(rig)] == [1, 2, 3]
    assert reports[-1]["verdict"] == "already_fired"
    state = await rig.detector._state_store.load()
    assert state.fired_status == "gave_up:failed"


@pytest.mark.asyncio
async def test_attempts_stay_bounded_when_the_clock_restarts_between_them(boot):
    """A delivery between two failed attempts restarts the clock, not the
    stretch: the bound still holds."""
    rig = await boot()

    async def failing_turn(prompt, **kwargs):
        raise RuntimeError("every model route is down")

    rig.agent.process_input = failing_turn
    for _ in range(6):
        await _run_for(rig, 40)
        rig.agent._live_turn_id = "turn_unrelated"
        await _tick(rig)
        rig.agent._live_turn_id = None

    assert len(_drained(rig)) == MAX_WAKE_ATTEMPTS


# ---------------------------------------------------------------------------
# Unreadable state fails closed
# ---------------------------------------------------------------------------


class _OldTalon(_Talon):
    finished_runs = None  # kestrel-feature-talon < 0.2.10


class _UnreadableTalon(_Talon):
    async def finished_runs(self):
        return {"complete": False, "runs": [], "reason": "jobs.json is not JSON"}


@pytest.mark.parametrize(
    "provider,reason",
    [
        (_OldTalon, "talon_provider_outdated"),
        (_UnreadableTalon, "talon_registry_unreadable: jobs.json is not JSON"),
    ],
)
@pytest.mark.asyncio
async def test_an_unreadable_talon_registry_never_reads_as_idle(boot, provider, reason):
    rig = await boot()
    rig.agent.wait_registry.register(provider(), replace=True)

    reports = await _run_for(rig, 120)

    assert _drained(rig) == []
    assert {r["verdict"] for r in reports} == {"unknown"}
    assert any(u.startswith(reason) for u in reports[-1]["unknown"])


class _ForgetfulTalon(_Talon):
    """``active_handles()`` after a failed reload with nothing in memory."""

    async def active_handles(self):
        return []


@pytest.mark.asyncio
async def test_a_talon_second_read_that_lost_jobs_never_reads_as_idle(boot):
    rig = await boot()
    talon = _ForgetfulTalon()
    talon.jobs = {"job-done": "complete", "job-running": "running"}
    rig.agent.wait_registry.register(talon, replace=True)

    reports = await _run_for(rig, 120)

    assert _drained(rig) == []
    assert "talon_registry_inconsistent" in reports[-1]["unknown"]


@pytest.mark.asyncio
async def test_a_missing_wait_registry_never_reads_as_idle(boot):
    rig = await boot()
    rig.agent.wait_registry = None

    reports = await _run_for(rig, 120)

    assert _drained(rig) == []
    assert "wait_registry_unavailable" in reports[-1]["unknown"]


@pytest.mark.asyncio
async def test_unobservable_durable_deliveries_never_read_as_idle(boot):
    rig = await boot()
    rig.agent.dispatcher.list_durable_deliveries = None

    reports = await _run_for(rig, 120)

    assert _drained(rig) == []
    assert "durable_deliveries_unobservable" in reports[-1]["unknown"]


@pytest.mark.asyncio
async def test_a_volatile_privacy_mode_never_reads_as_idle(boot):
    rig = await boot()
    rig.agent.privacy_config = SimpleNamespace(
        is_ephemeral=lambda: True,
        uses_temp_storage=lambda: False,
        requires_deidentification=lambda: False,
    )

    reports = await _run_for(rig, 120)

    assert _drained(rig) == []
    assert "volatile_privacy_mode" in reports[-1]["unknown"]


@pytest.mark.asyncio
async def test_a_held_agent_is_not_woken(boot):
    rig = await boot()

    class _HoldStore:
        async def get_effective(self, agent_id):
            return SimpleNamespace(held=True)

    rig.agent._hold_store = _HoldStore()
    reports = await _run_for(rig, 120)

    assert _drained(rig) == []
    assert "held" in reports[-1]["unknown"]


# ---------------------------------------------------------------------------
# Off by default, opt-in per agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_the_flag_off_nothing_is_emitted(boot):
    rig = await boot()
    off = IdleWakeConfig()

    for _ in range(40):
        report = await _tick(rig, config=off)
        assert report == {"skipped": "disabled", "fired": False}
    assert await run_idle_wake(rig.agent, config=off) == report

    assert rig.wakes == []
    tables = await rig.db.fetchall(
        "SELECT name FROM sqlite_master WHERE name = 'idle_wake_state'"
    )
    assert tables == [], "a disabled detector reads and writes nothing"


def test_the_flag_is_off_by_default_and_opt_in_per_agent(monkeypatch):
    emma = SimpleNamespace(agent_name="Emma", did="did:test:emma")
    kite = SimpleNamespace(agent_name="Kite", did="did:test:kite")

    monkeypatch.setattr(idle_wake, "load_section", lambda section: {})
    assert load_idle_wake_config(emma) == IdleWakeConfig(enabled=False)

    section = {"enabled": True, "agents": ["Emma"], "idle_minutes": 45}
    monkeypatch.setattr(idle_wake, "load_section", lambda section_name: section)
    assert load_idle_wake_config(emma) == IdleWakeConfig(True, 45)
    assert load_idle_wake_config(kite) == IdleWakeConfig(enabled=False)

    assert load_idle_wake_config(kite, {"enabled": True}).enabled
    assert not load_idle_wake_config(kite, {"enabled": "yes"}).enabled
    assert load_idle_wake_config(
        kite, {"enabled": True, "agents": ["did:test:kite"]}
    ).enabled
    assert load_idle_wake_config(
        emma, {"enabled": True, "idle_minutes": 2}
    ).idle_minutes == idle_wake.DEFAULT_IDLE_MINUTES


# ---------------------------------------------------------------------------
# Scheduler wiring
# ---------------------------------------------------------------------------


def test_the_shipped_wake_is_rate_limited():
    registration = build_queue_drained_registration()
    assert registration.rate_limit == RateLimit(per_minute=1, per_hour=6)
    assert registration.allowed_modes == frozenset({registration.default_mode})


def test_idle_wake_is_a_built_in_action_cron_task():
    from kestrel_sdk.signals import SignalMode

    assert ("idle_wake", SignalMode.ACTION, frozenset()) in CRON_TASKS
    feature = SchedulerFeature(agent=None)
    assert "idle_wake" in feature._builtin_cron_handlers()


async def _scheduler(tmp_path, *, agent_name="Emma"):
    backend = SQLiteBackend(str(tmp_path / "scheduler.db"))
    await backend.connect()
    db = AsyncDatabase(backend)
    agent = SimpleNamespace(
        did=AGENT_DID, agent_id=AGENT_DID, agent_name=agent_name,
        features={}, storage=SimpleNamespace(db=db),
    )
    feature = SchedulerFeature(agent)
    feature._db = db
    feature._agent_id = AGENT_DID
    runner = SchedulerRunner(db, AGENT_DID, None, owner_id="idle-wake-seed")
    await runner._ensure_tables()
    return feature, agent, db


async def _idle_wake_rows(db):
    return await db.fetchall(
        "SELECT cron_expression, idempotency_key FROM scheduled_tasks "
        "WHERE agent_id = ? AND task_name = 'idle_wake' ORDER BY idempotency_key",
        (AGENT_DID,),
    )


@pytest.mark.asyncio
async def test_the_cron_is_seeded_only_for_an_opted_in_agent(tmp_path, monkeypatch):
    feature, agent, db = await _scheduler(tmp_path)
    try:
        monkeypatch.setattr(idle_wake, "load_section", lambda section: {})
        await feature.post_all_features_loaded(agent)
        assert await _idle_wake_rows(db) == []

        monkeypatch.setattr(
            idle_wake, "load_section",
            lambda section: {"enabled": True, "agents": ["Emma"]},
        )
        await feature.post_all_features_loaded(agent)
        assert await _idle_wake_rows(db) == [
            ("*/5 * * * *", "scheduler:builtin:v1:idle_wake")
        ]

        # A user's own idle_wake schedule is theirs; turning the flag off
        # removes only the row core seeded.
        await feature.schedule_add(
            cron_expression="*/10 * * * *", task_name="idle_wake",
            idempotency_key="user:idle",
        )
        monkeypatch.setattr(idle_wake, "load_section", lambda section: {})
        await feature.post_all_features_loaded(agent)
        assert await _idle_wake_rows(db) == [("*/10 * * * *", "user:idle")]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_scheduled_self_followup_is_a_pending_wake_until_it_fires(
    followup_env,
):
    """The count the detector reads follows the follow-up's real lifecycle:
    scheduled from a turn, it is a deferred wake; once fired, it is not."""
    from tests.utils.scheduler_ticks import tick_and_settle

    agent, feature, runner, _db, _backend = followup_env
    assert await feature._count_pending_self_followups() == 0

    async with agent._turn_lifecycle():
        added = await feature.schedule_add_deadline(
            run_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
            task_name="self_followup",
            args_json=json.dumps({"intent": "check CI on PR 3664"}),
        )
    assert added.status.value == "ok", added.error
    assert await feature._count_pending_self_followups() == 1

    await tick_and_settle(runner)
    pending = [t for t in agent.background_tasks if not t.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    assert len(agent.turn_prompts) == 1
    assert await feature._count_pending_self_followups() == 0
