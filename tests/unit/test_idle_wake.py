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
the one read-only call the detector makes and refuses ``poll()`` and
``active_handles()``. Where the host names a Talon dispatcher (#3758), a
co-hosted dispatcher agent holds the lane the detector reads, and one test
runs kestrel-feature-talon's own provider on both agents. Time is a
clock the test advances; each ``_tick`` advances it five minutes, the cron's
cadence. Starting the agent twice over the same databases without stopping it
is two processes sharing one database, as in a rolling restart.
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
from kestrel_sovereign.features.strategic_memory.run_history import (
    UNFINISHED_JOBS_REQUIREMENT,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
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
    OWNER_HEARTBEAT_TASK_PREFIX,
    OWNER_STALE_AFTER,
    TRIAGE_TASK_PREFIX,
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
# The detector's owner renews its lease this often. Real time, unlike the
# test clock, so the lease is renewed at the clock's new time.
LEASE_INTERVAL = timedelta(milliseconds=10)


class _IdleAgent(_WaitingAgent):
    """The #3663 double, plus the reply stamp a real wake turn writes (#3655),
    so a restart can tell a wake that ran from one that did not, and the
    in-span admission a source's pre-turn guard requires (#3310), so a
    scheduled ``self_followup`` can run a turn here."""

    async def process_input(
        self, prompt: str, session_id=None, signal_wake=None, wake_reply=None,
        pre_turn_guard=None, **_kwargs,
    ):
        async with self._turn_lifecycle():
            KestrelAgent._evaluate_pre_turn_guard(pre_turn_guard)
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
    """Talon's provider as kestrel-feature-talon 0.2.14 reports: its
    unfinished jobs come from the same read as its finished ones (the
    ``unfinished`` key #3664 requires), each a record naming its job. The
    real provider is exercised below too (``test_the_real_talon_provider_...``)."""

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
            "unfinished": [
                {"job_id": job, "repo": "o/r", "issue": "1", "status": status,
                 "started_at": "2026-10-15T22:00:00+00:00"}
                for job, status in self.jobs.items()
                if status not in TERMINAL
            ],
        }

    async def active_handles(self):
        raise AssertionError(
            "the idle detector must not take running jobs from active_handles()"
        )

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

    async def start(agent_class=None):
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
        agent = (agent_class or _IdleAgent)(conversation, db)
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
            sources=sources,
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

        async def count_followups():
            return rig.followups

        # What the scheduler supplies: its unfinished self_followup count.
        rig.pending_followups = count_followups

        async def stop():
            """The process stops cleanly: its turns finish, then its lease
            heartbeat goes with it."""
            running.remove(rig)
            await _settle(rig)
            beating = _heartbeats(rig)
            for task in beating:
                task.cancel()
            await asyncio.gather(*beating, return_exceptions=True)
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


def _heartbeats(rig) -> list:
    """The task renewing this rig's lease, while it runs. It runs for the
    life of the process."""
    return [
        t for t in rig.agent.background_tasks
        if t.get_name().startswith(OWNER_HEARTBEAT_TASK_PREFIX) and not t.done()
    ]


async def _settle(rig) -> None:
    """Let every enqueued wake's turn run to completion."""
    pending = [
        t for t in rig.agent.background_tasks
        if not t.done() and t not in _heartbeats(rig)
    ]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _evaluate(
    rig, minutes: float = 5, config: IdleWakeConfig = CONFIG
) -> dict:
    """One cron evaluation, ``minutes`` after the last. A wake it enqueues
    is left running."""
    rig.clock.advance(minutes)
    return await rig.detector.tick(config, pending_followups=rig.pending_followups)


async def _tick(rig, minutes: float = 5, config: IdleWakeConfig = CONFIG) -> dict:
    """One cron evaluation, ``minutes`` after the last, with any wake it
    enqueued run to completion."""
    report = await _evaluate(rig, minutes, config)
    await _settle(rig)
    return report


async def _run_for(rig, minutes: int, config: IdleWakeConfig = CONFIG) -> list:
    reports = []
    for _ in range(minutes // 5):
        reports.append(await _tick(rig, config=config))
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


async def _fire_once(rig, config: IdleWakeConfig = CONFIG) -> None:
    """Run until the first wake fires. Its owner settles it as its turn
    ends, so the next evaluation finds the stretch spent."""
    for _ in range(12):
        report = await _tick(rig, config=config)
        if report.get("fired"):
            break
    else:
        raise AssertionError(f"no wake fired: {report}")
    await _assert_settled(rig, "ok")
    after = await _tick(rig, config=config)
    assert "harvested" not in after, after
    assert after["verdict"] == "already_fired", after


async def _assert_settled(rig, status: str) -> None:
    state = await rig.detector._state_store.load()
    assert state.pending_signal_id is None, state
    assert state.last_wake_status == status, state
    assert state.fired_status == status, state


def _hold_turns(rig) -> asyncio.Event:
    """Hold every wake turn on ``rig`` until the returned gate opens."""
    gate = asyncio.Event()
    rig.gates.append(gate)
    turn = rig.agent.process_input

    async def held(prompt, pre_turn_guard=None, **kwargs):
        await gate.wait()
        return await turn(prompt, pre_turn_guard=pre_turn_guard, **kwargs)

    rig.agent.process_input = held
    return gate


async def _fire_held(rig) -> dict:
    """Run until a wake fires, leaving its (held) turn running."""
    for _ in range(12):
        report = await _evaluate(rig)
        if report.get("fired"):
            return report
    raise AssertionError(f"no wake fired: {report}")


def _triage_tasks(rig) -> list:
    """The task owning each wake this rig fired, while it runs: it hands the
    wake to the dispatcher and settles it when its dispatch ends."""
    return [
        t for t in rig.agent.background_tasks
        if t.get_name().startswith(TRIAGE_TASK_PREFIX) and not t.done()
    ]


async def _lease_renewed(rig) -> None:
    """Wait until the owner's heartbeat has renewed its lease at the clock's
    time."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while (
        await rig.detector._state_store.load()
    ).owner_heartbeat_at != rig.clock.now.isoformat():
        assert loop.time() < deadline, "the owner never renewed its lease"
        await asyncio.sleep(0.005)


async def _fire_and_lose_the_outcome(rig) -> None:
    """Fire, and let the wake's turn store its reply, but lose the owner's
    record of how it ended: the triage task dies first, as it does when the
    process stops between the reply and the settle."""
    gate = _hold_turns(rig)
    await _fire_held(rig)
    [triage] = _triage_tasks(rig)
    triage.cancel()
    await asyncio.gather(triage, return_exceptions=True)
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
    "self_followup scheduled": (_self_followup_scheduled, "self_followup_pending:1"),
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
    await _lease_renewed(rig)
    await rig.stop()

    rig = await boot()
    # The stopped process's lease has not lapsed yet: the new one waits.
    early = await _tick(rig, minutes=1)
    assert early["verdict"] == "not_owner", early
    reports = await _run_for(rig, 180)

    assert reports[0]["took_over"] is True, reports[0]
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
# One process runs the detector
# ---------------------------------------------------------------------------


STALE_MINUTES = OWNER_STALE_AFTER.total_seconds() / 60


def _stall_renewals(rig) -> asyncio.Event:
    """Stall the owner's lease renewals, as a wedged process would, until
    the returned event is set."""
    stalled = asyncio.Event()
    renew = rig.detector._state_store.renew

    async def stalled_renew(*args):
        await stalled.wait()
        return await renew(*args)

    rig.detector._state_store.renew = stalled_renew
    return stalled


@pytest.mark.asyncio
async def test_a_process_without_the_lease_does_not_evaluate(boot):
    """A turn running on one process is invisible to another. A long
    operator turn runs on A, the detector's owner; B evaluates for two
    hours, far past the threshold, and observes, writes and dispatches
    nothing."""
    a = await boot()
    b = await boot()
    assert (await _tick(a, minutes=0))["verdict"] == "idle"

    await _say(a, OPERATOR, "a long piece of work")
    a.agent._live_turn_id = "turn_operator_on_a"
    for _ in range(24):
        a.clock.advance(5)
        await _lease_renewed(a)
        before = await b.detector._state_store.load()
        report = await _evaluate(b, minutes=0)
        assert report == {
            "idle_minutes": 30,
            "fired": False,
            "verdict": "not_owner",
            "owner_heartbeat_at": a.clock.now.isoformat(),
        }, report
        assert await b.detector._state_store.load() == before
    assert _drained(a) == [] and _drained(b) == []
    assert _heartbeats(b) == [], "B never took the lease"

    # A, which runs the detector, sees the turn running there.
    report = await _tick(a, minutes=0)
    assert report["verdict"] == "busy", report
    assert "turn_in_progress" in report["deliveries"]


@pytest.mark.asyncio
async def test_a_wake_running_on_its_owner_is_not_dispatched_again(boot):
    """The cron's next occurrence can run on a process that did not enqueue
    the wake. While the owner lives, that process does not evaluate, so it
    cannot read the running wake as lost."""
    a = await boot()
    b = await boot()
    gate = _hold_turns(a)
    await _fire_held(a)

    for _ in range(24):  # two hours of evaluations on B
        a.clock.advance(5)
        await _lease_renewed(a)
        report = await _evaluate(b, minutes=0)
        assert report["verdict"] == "not_owner", report
    assert len(_drained(a)) == 1
    assert _drained(b) == []

    # The turn ends; its owner settles the wake and finds the stretch spent.
    gate.set()
    await _settle(a)
    await _assert_settled(a, "ok")
    reports = await _run_for(a, 120)
    assert {r["verdict"] for r in reports} == {"already_fired"}
    assert len(a.agent.wake_sessions) == 1 and b.agent.wake_sessions == []


@pytest.mark.asyncio
async def test_a_process_takes_over_only_once_the_owners_lease_lapses(boot):
    a = await boot()
    b = await boot()
    _hold_turns(a)
    await _fire_held(a)
    await _lease_renewed(a)
    await a.crash()  # the owner's process stops with the turn unfinished

    within = await _evaluate(b, minutes=STALE_MINUTES)
    assert within["verdict"] == "not_owner", within

    lapsed = await _tick(b, minutes=1)
    assert lapsed["took_over"] is True, lapsed
    assert lapsed["harvested"]["status"] == "lost_at_restart", lapsed
    # The stretch goes on; its wake never ran, so the retry goes out now.
    assert lapsed["verdict"] == "fired", lapsed
    [retry] = _drained(b)
    assert retry.payload["attempt"] == 2
    assert len(_drained(a)) == 1
    assert len(b.agent.wake_sessions) == 1
    assert len(_heartbeats(b)) == 1


@pytest.mark.asyncio
async def test_a_renewal_that_lands_during_a_takeover_wins(boot):
    """A's process stalls past the lease, so B reads it as lapsed and starts
    taking it over. A renews before B writes: the takeover is refused and B
    goes on not evaluating."""
    a = await boot()
    b = await boot()
    await _tick(a, minutes=0)
    await _lease_renewed(a)
    stalled = _stall_renewals(a)

    take_lease = b.detector._state_store.take_lease

    async def a_wakes_up_mid_takeover(*args):
        stalled.set()
        await _lease_renewed(a)
        return await take_lease(*args)

    b.detector._state_store.take_lease = a_wakes_up_mid_takeover
    report = await _evaluate(b, minutes=STALE_MINUTES + 1)
    assert report["verdict"] == "state_conflict", report
    assert report["superseded"] == "take_lease"

    b.detector._state_store.take_lease = take_lease
    assert (await _evaluate(b, minutes=0))["verdict"] == "not_owner"
    state = await a.detector._state_store.load()
    assert state.owner == a.detector._owner
    assert _heartbeats(b) == []


@pytest.mark.asyncio
async def test_two_processes_taking_a_free_lease_at_once_leave_one_owner(boot):
    """Both read the lease free before either takes it. Taking it is a
    compare-and-set, so the second loses and evaluates nothing."""
    a = await boot()
    b = await boot()
    take_lease = a.detector._state_store.take_lease
    raced: dict = {}

    async def b_takes_it_first(*args):
        raced["b"] = await _evaluate(b, minutes=0)
        return await take_lease(*args)

    a.detector._state_store.take_lease = b_takes_it_first
    report = await _tick(a, minutes=0)
    a.detector._state_store.take_lease = take_lease

    assert raced["b"]["verdict"] == "idle", raced["b"]
    assert report["verdict"] == "state_conflict", report
    assert report["superseded"] == "take_lease"
    assert _heartbeats(a) == [] and len(_heartbeats(b)) == 1

    # From here B runs the detector alone, and the stretch gets one wake.
    for _ in range(12):
        a.clock.advance(5)
        await _lease_renewed(b)
        assert (await _evaluate(a, minutes=0))["verdict"] == "not_owner"
        await _evaluate(b, minutes=0)
        await _settle(b)
    assert _drained(a) == [] and len(_drained(b)) == 1


@pytest.mark.asyncio
async def test_an_owner_whose_lease_was_taken_mid_evaluation_cannot_fire(boot):
    """A reads the row and decides to fire. Before it claims the wake, B,
    whose clock runs ahead of A's, reads A's lease as lapsed, takes it over
    and fires first. By its own clock A confirmed its lease moments ago, so
    only the claim's compare-and-set can stop it: taking the lease advanced
    the revision, A's claim is refused, and the stretch gets one wake."""
    a = await boot()
    b = await boot()
    for _ in range(6):
        await _tick(a)  # idle for 25 minutes
    a.clock.advance(4.5)
    await _lease_renewed(a)
    # From here A's renewals hang, so its lease stays where it is.
    _stall_renewals(a)
    b.detector._clock = lambda: a.clock.now + timedelta(minutes=STALE_MINUTES + 1)
    raced: dict = {}
    chain_session = a.detector._chain_session

    async def b_takes_over_first():
        raced["b"] = await _evaluate(b, minutes=0)
        await _settle(b)
        return await chain_session()

    a.detector._chain_session = b_takes_over_first
    # Thirty minutes idle, half a minute after A's last renewal: A fires.
    report = await _tick(a, minutes=0.5)

    assert raced["b"]["took_over"] is True, raced["b"]
    assert raced["b"]["verdict"] == "fired", raced["b"]
    assert report["verdict"] == "state_conflict", report
    assert report["superseded"] == "fired"
    assert _drained(a) == [] and len(_drained(b)) == 1
    await _assert_settled(b, "ok")

    b.detector._clock = a.clock
    a.clock.advance(5)
    await _lease_renewed(b)
    assert (await _evaluate(a, minutes=0))["verdict"] == "not_owner"
    assert len(b.agent.wake_sessions) == 1 and a.agent.wake_sessions == []


@pytest.mark.asyncio
async def test_an_owner_whose_renewals_hang_does_not_fire(boot):
    """The same race with one clock. A's renewals hang, so by the time B
    reads A's lease as lapsed, A has not confirmed it within
    ``LEASE_CONFIRMED_WITHIN`` either, and A stands down before claiming."""
    a = await boot()
    b = await boot()
    for _ in range(6):
        await _tick(a)
    a.clock.advance(4.5)
    await _lease_renewed(a)
    _stall_renewals(a)
    raced: dict = {}
    chain_session = a.detector._chain_session

    async def b_takes_over_first():
        raced["b"] = await _evaluate(b, minutes=STALE_MINUTES + 1)
        await _settle(b)
        return await chain_session()

    a.detector._chain_session = b_takes_over_first
    report = await _tick(a, minutes=0.5)

    assert raced["b"]["took_over"] is True, raced["b"]
    assert raced["b"]["verdict"] == "fired", raced["b"]
    assert report["verdict"] == "state_conflict", report
    assert report["superseded"] == "lease_unconfirmed"
    assert _drained(a) == [] and len(_drained(b)) == 1
    await _assert_settled(b, "ok")


@pytest.mark.asyncio
async def test_a_process_that_lost_the_lease_does_not_settle_its_wake(boot):
    """A's process stalls with its triage turn running; B takes the lease
    over but cannot yet tell how the wake ended. When A's turn finishes, A
    no longer runs the detector and writes nothing, not even its own wake's
    outcome. B settles it from the reply the turn stored."""
    a = await boot()
    b = await boot()
    gate = _hold_turns(a)
    fired = await _fire_held(a)
    _stall_renewals(a)

    lookup = b.detector._reply_outcome

    async def unreadable(signal_id):
        return None, "reply_lookup_failed: database is locked"

    b.detector._reply_outcome = unreadable
    taken = await _evaluate(b, minutes=STALE_MINUTES + 1)
    assert taken["took_over"] is True and taken["verdict"] == "unknown", taken

    gate.set()
    await _settle(a)
    state = await b.detector._state_store.load()
    assert state.pending_signal_id == fired["signal_id"], state
    assert state.owner == b.detector._owner

    b.detector._reply_outcome = lookup
    report = await _tick(b)
    assert report["harvested"]["status"] == "replied_before_restart", report
    await _run_for(b, 60)
    assert len(_drained(a)) == 1 and _drained(b) == []
    assert len(a.agent.wake_sessions) == 1


@pytest.mark.asyncio
async def test_the_owners_heartbeat_outlives_a_slow_evaluation(boot):
    """Reading Talon's registry can be slow. The heartbeat does not wait for
    the evaluation, so the lease stays live through it, and the renewals do
    not invalidate the evaluation's own write."""
    a = await boot()
    b = await boot()
    await _tick(a, minutes=0)
    reading = asyncio.Event()
    release = asyncio.Event()
    finished_runs = a.talon.finished_runs

    async def slow_finished_runs():
        reading.set()
        await release.wait()
        return await finished_runs()

    a.talon.finished_runs = slow_finished_runs
    evaluation = asyncio.create_task(_evaluate(a))
    await reading.wait()

    a.clock.advance(STALE_MINUTES + 1)
    await _lease_renewed(a)
    assert (await _evaluate(b, minutes=0))["verdict"] == "not_owner"

    release.set()
    report = await evaluation
    assert report["verdict"] == "idle", report


@pytest.mark.asyncio
async def test_a_renewal_with_an_unknown_commit_outcome_ends_the_heartbeat(boot):
    """Execution-custody evidence is not a failed renewal to retry: the
    heartbeat ends with it, and the next evaluation starts a new one."""
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    rig = await boot()
    await _tick(rig, minutes=0)
    [beat] = _heartbeats(rig)

    async def renew(owner, heartbeat_at):
        raise ExecutionCommitOutcomeError("unknown")

    rig.detector._state_store.renew = renew
    done, _ = await asyncio.wait({beat}, timeout=5)
    assert beat in done, "the heartbeat went on renewing"
    assert isinstance(beat.exception(), ExecutionCommitOutcomeError)


@pytest.mark.parametrize("absorbed", ["into an error", "into a return"])
@pytest.mark.asyncio
async def test_a_heartbeat_cancelled_mid_renewal_ends(boot, absorbed):
    """A store that turns the cancellation of a renewal in flight into an
    error, or absorbs it and returns, must not keep the heartbeat alive: a
    shutdown that cancels it waits for it to end (#3664, seen as a teardown
    hang after #3570's custody)."""
    rig = await boot()
    await _tick(rig, minutes=0)
    [beat] = _heartbeats(rig)
    renewing = asyncio.Event()

    async def renew(owner, heartbeat_at):
        renewing.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            if absorbed == "into a return":
                return True
            raise RuntimeError("the renewal's outcome is unknown") from None

    real_renew = rig.detector._state_store.renew
    rig.detector._state_store.renew = renew
    await asyncio.wait_for(renewing.wait(), timeout=5)
    beat.cancel()
    try:
        done, _ = await asyncio.wait({beat}, timeout=5)
        assert beat in done, "the heartbeat swallowed its cancellation"
    finally:
        if not beat.done():
            # A regression fails here rather than hanging the teardown.
            rig.detector._state_store.renew = real_renew
            beat.cancel()
            await asyncio.wait({beat}, timeout=5)


@pytest.mark.asyncio
async def test_a_wake_whose_dispatch_outcome_is_unknown_is_decided_by_its_reply(boot):
    """A dispatch that ends in execution-custody evidence may have run its
    turn. Its stored reply says it did, so the stretch is spent, not retried."""
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    rig = await boot()
    enqueue = rig.agent.dispatcher.enqueue_signal

    async def unknown_outcome(signal, **kwargs):
        handle = await enqueue(signal, **kwargs)

        async def ran_then_lost_its_outcome():
            await handle.task
            raise ExecutionCommitOutcomeError("unknown")

        return SimpleNamespace(
            task=asyncio.create_task(ran_then_lost_its_outcome())
        )

    rig.agent.dispatcher.enqueue_signal = unknown_outcome
    for _ in range(12):
        if (await _tick(rig)).get("fired"):
            break
    assert len(rig.agent.wake_sessions) == 1, "the wake's turn ran"
    state = await rig.detector._state_store.load()
    assert state.pending_signal_id is not None, "not settled as a failure"

    reports = [await _tick(rig)] + await _run_for(rig, 120)

    assert reports[0]["harvested"]["status"] == "replied_before_restart", reports[0]
    assert len(_drained(rig)) == 1
    assert {r["verdict"] for r in reports} == {"already_fired"}


@pytest.mark.asyncio
async def test_a_dispatch_cancelled_by_custody_evidence_is_decided_by_its_reply(boot):
    """A dispatch can end cancelled with an unknown commit outcome as its
    cause. That is custody evidence, not a cancellation to retry: its turn
    may have run. Its stored reply says it did, so the stretch is spent. It
    holds even though another reader took the task's cancellation first, as
    the agent's own completion callback does: asyncio hands the original
    exception to the first reader only."""
    from kestrel_sovereign._async_ownership import create_owned_task
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    rig = await boot()
    enqueue = rig.agent.dispatcher.enqueue_signal

    def read_first(task):
        if task.cancelled():
            with pytest.raises(asyncio.CancelledError):
                task.exception()

    async def cancelled_with_an_unknown_commit(signal, **kwargs):
        handle = await enqueue(signal, **kwargs)

        async def ran_then_cancelled():
            await handle.task
            raise asyncio.CancelledError() from ExecutionCommitOutcomeError(
                "unknown"
            )

        task = create_owned_task(ran_then_cancelled())
        task.add_done_callback(read_first)
        return SimpleNamespace(task=task)

    rig.agent.dispatcher.enqueue_signal = cancelled_with_an_unknown_commit
    for _ in range(12):
        if (await _tick(rig)).get("fired"):
            break
    assert len(rig.agent.wake_sessions) == 1, "the wake's turn ran"
    state = await rig.detector._state_store.load()
    assert state.pending_signal_id is not None, "settled as a cancellation"

    reports = [await _tick(rig)] + await _run_for(rig, 120)

    assert reports[0]["harvested"]["status"] == "replied_before_restart", reports[0]
    assert len(_drained(rig)) == 1
    assert len(rig.agent.wake_sessions) == 1
    assert {r["verdict"] for r in reports} == {"already_fired"}


@dataclasses.dataclass(eq=False)
class _SqliteAdmission:
    """An execution admission SQLite storage accepts (#3570). On PostgreSQL a
    scheduler occurrence holds one of its own, retired as the occurrence
    returns (``bind_execution_custody``). The unit suite has no PostgreSQL,
    so this stands in for it, and for the agent's runtime admission.
    ``ExecutionCustody.revoke`` retires it."""

    backend_type: str = "sqlite"

    def require_work(self) -> None:
        pass

    async def lock_and_validate(self, connection) -> None:
        pass


class _RuntimeAgent(_IdleAgent):
    """The double with the real agent's task tracking (#3570) and a runtime
    admission of its own. ``_track_background_task`` starts a child that
    carries every admission held as it starts; ``_track_runtime_task`` hands
    a task to the runtime admission alone, once the agent is ready."""

    _runtime_owner_context = KestrelAgent._runtime_owner_context
    _finish_background_task = KestrelAgent._finish_background_task
    _remember_shutdown_control = KestrelAgent._remember_shutdown_control

    def __init__(self, conversation, db):
        from kestrel_sovereign.execution_custody import ExecutionCustody

        super().__init__(conversation, db)
        self._background_tasks = set()
        self._runtime_publication_ready = asyncio.Event()
        self._runtime_publication_ready.set()
        self._execution_custody = ExecutionCustody(_SqliteAdmission())

    def _track_background_task(self, coro, *, name, admission=None):
        task = KestrelAgent._track_background_task(
            self, coro, name=name, admission=admission
        )
        self.background_tasks.append(task)
        return task

    def _track_runtime_task(self, coro, *, name):
        task = KestrelAgent._track_runtime_task(self, coro, name=name)
        self.background_tasks.append(task)
        return task


async def _evaluate_in_occurrence(rig, minutes: float = 5, opened=None) -> dict:
    """One evaluation inside a cron occurrence that holds its own execution
    admission and retires it as it returns, as the scheduler's PostgreSQL
    occurrences do. ``opened`` collects each occurrence's admission."""
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody,
        bind_execution_custody_snapshot,
        bind_execution_runtime,
    )

    occurrence = ExecutionCustody(_SqliteAdmission())
    if opened is not None:
        opened.append(occurrence)
    try:
        with bind_execution_runtime(rig.agent), bind_execution_custody_snapshot(
            (occurrence,)
        ):
            return await _evaluate(rig, minutes)
    finally:
        occurrence.revoke("execution admission retired")


@pytest.mark.asyncio
async def test_the_wake_outlives_the_cron_occurrence_that_fired_it(boot):
    """The evaluation fires from inside a cron occurrence, but the triage
    turn, the wake's settlement and the lease heartbeat all run after that
    occurrence has returned and retired its admission. They are the agent
    runtime's work, so none of them is refused."""
    from kestrel_sovereign.execution_custody import current_execution_custody

    rig = await boot(_RuntimeAgent)
    gate = _hold_turns(rig)
    opened: list = []
    for _ in range(12):
        report = await _evaluate_in_occurrence(rig, opened=opened)
        if report.get("fired"):
            break
    else:
        raise AssertionError(f"no wake fired: {report}")

    admissions = []
    turn = rig.agent.process_input

    async def observed(prompt, **kwargs):
        admissions.append(current_execution_custody())
        return await turn(prompt, **kwargs)

    rig.agent.process_input = observed
    gate.set()  # the occurrence that fired the wake has retired by now
    await _settle(rig)

    assert admissions == [(rig.agent._execution_custody,)], (
        "the triage turn ran under the occurrence's admission"
    )
    assert len(rig.agent.wake_sessions) == 1
    await _assert_settled(rig, "ok")
    rig.clock.advance(1)
    await _lease_renewed(rig)
    assert all(
        occurrence not in current for occurrence in opened
        for current in admissions
    )
    reports = await _run_for(rig, 60)
    assert {r["verdict"] for r in reports} == {"already_fired"}, reports
    assert len(_drained(rig)) == 1


@pytest.mark.asyncio
async def test_a_wake_is_not_handed_over_from_an_occurrence_that_retired(boot):
    """The hand-over to the runtime keeps the occurrence's own admission
    check: an occurrence retired between the claim and the hand-over starts
    nothing. Its claim stays recorded, and the next occurrence finds no
    reply and retries it."""
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    rig = await boot(_RuntimeAgent)
    opened: list = []
    save = rig.detector._state_store.save

    async def claim_then_retire(state):
        written = await save(state)
        if written is not None and state.pending_signal_id is not None:
            opened[-1].revoke("execution admission retired")
        return written

    rig.detector._state_store.save = claim_then_retire
    with pytest.raises(ExecutionAuthorityError):
        for _ in range(12):
            await _evaluate_in_occurrence(rig, opened=opened)
            await _settle(rig)
    await _settle(rig)
    assert _drained(rig) == [] and rig.agent.wake_sessions == []
    claimed = await rig.detector._state_store.load()
    assert claimed.pending_signal_id is not None and claimed.attempts == 1

    rig.detector._state_store.save = save
    retried = await _evaluate_in_occurrence(rig)
    await _settle(rig)
    assert retried["harvested"]["status"] == idle_wake.LOST_AT_RESTART, retried
    assert retried["verdict"] == "fired", retried
    assert len(rig.agent.wake_sessions) == 1
    await _assert_settled(rig, "ok")
    assert (await rig.detector._state_store.load()).attempts == 2


@pytest.mark.asyncio
async def test_a_dispatch_accepted_but_not_handed_back_is_settled_not_retried(boot):
    """The agent's task admission can accept the dispatch and then fail to
    hand it back. The dispatch runs; it is this process's wake, settled when
    it ends, and a retry would start a second triage turn."""
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    rig = await boot()
    enqueue = rig.agent.dispatcher.enqueue_signal

    async def accepted_then_raised(signal, **kwargs):
        assert kwargs.get("admission") is not None, "no admission receipt"
        await enqueue(signal, **kwargs)
        raise ExecutionCommitOutcomeError("unknown")

    rig.agent.dispatcher.enqueue_signal = accepted_then_raised
    await _tick(rig, minutes=0)
    # The custody evidence still propagates, once the wake is held.
    with pytest.raises(ExecutionCommitOutcomeError):
        for _ in range(12):
            await _tick(rig)
    await _settle(rig)
    await _assert_settled(rig, "ok")

    reports = await _run_for(rig, 120)
    assert {r["verdict"] for r in reports} == {"already_fired"}
    assert len(rig.agent.wake_sessions) == 1


@pytest.mark.asyncio
async def test_a_dispatch_the_agent_refused_is_retried_once_its_claim_is_decided(boot):
    """Refused with execution-custody evidence, nothing started: the tick
    raises it, the claim stays recorded, and the next evaluation finds no
    reply and retries."""
    from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

    rig = await boot()
    enqueue = rig.agent.dispatcher.enqueue_signal

    async def refused(signal, **kwargs):
        raise ExecutionCommitOutcomeError("unknown")

    rig.agent.dispatcher.enqueue_signal = refused
    await _tick(rig, minutes=0)
    with pytest.raises(ExecutionCommitOutcomeError):
        for _ in range(12):
            await _tick(rig)
    claimed = await rig.detector._state_store.load()
    assert claimed.pending_signal_id is not None and claimed.attempts == 1

    rig.agent.dispatcher.enqueue_signal = enqueue
    retried = await _tick(rig)
    assert retried["harvested"]["status"] == idle_wake.LOST_AT_RESTART, retried
    await _run_for(rig, 60)
    assert len(rig.agent.wake_sessions) == 1, "one turn, from the retry"
    assert (await rig.detector._state_store.load()).attempts == 2


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


class _TalonBefore0214(_Talon):
    """kestrel-feature-talon 0.2.10 to 0.2.13: its ``complete`` report reads
    running jobs and leaves them out, and ``active_handles()`` after a reload
    that failed answers from an empty memory."""

    async def finished_runs(self):
        report = await super().finished_runs()
        del report["unfinished"]
        return report

    async def active_handles(self):
        return []


@pytest.mark.parametrize(
    "jobs",
    [
        pytest.param({"job-running": "running"}, id="only a running job"),
        pytest.param({}, id="an empty registry"),
    ],
)
@pytest.mark.asyncio
async def test_a_talon_that_does_not_list_unfinished_jobs_never_reads_as_idle(
    boot, jobs
):
    """A ``complete`` report that leaves running jobs out cannot say none is
    running, and a second read that lost them cannot either: the detector
    takes running jobs from nowhere else. With only a running job, the
    finished runs are empty and the second read is empty, so nothing it read
    disagrees, and before #3664's review this read as idle."""
    rig = await boot()
    talon = _TalonBefore0214()
    talon.jobs = dict(jobs)
    rig.agent.wait_registry.register(talon, replace=True)

    reports = await _run_for(rig, 120)

    assert _drained(rig) == []
    assert {r["verdict"] for r in reports} == {"unknown"}
    assert any(
        u.startswith("talon_provider_outdated: finished_runs() lists no")
        and u.endswith(f"(needs {UNFINISHED_JOBS_REQUIREMENT})")
        for u in reports[-1]["unknown"]
    )


class _MalformedTalon(_Talon):
    """A ``complete`` report with an entry that names no job."""

    def __init__(self, key: str, entry) -> None:
        super().__init__()
        self.key = key
        self.entry = entry

    async def finished_runs(self):
        report = await super().finished_runs()
        report[self.key].append(self.entry)
        return report


@pytest.mark.parametrize(
    "key,entry",
    [
        pytest.param("unfinished", None, id="unfinished: not a record"),
        pytest.param("unfinished", {"repo": "o/r", "status": "running"},
                     id="unfinished: no job_id"),
        # A bare id is not 0.2.14's shape: a record that is not one says
        # nothing the detector can vouch for.
        pytest.param("unfinished", "job-bare", id="unfinished: a bare id"),
        pytest.param("unfinished", {"job_id": ""}, id="unfinished: empty job_id"),
        pytest.param("unfinished", {"job_id": "  "}, id="unfinished: blank job_id"),
        pytest.param("runs", {"repo": "o/r"}, id="runs: no job_id"),
    ],
)
@pytest.mark.asyncio
async def test_a_talon_entry_that_names_no_job_never_reads_as_idle(boot, key, entry):
    rig = await boot()
    rig.agent.wait_registry.register(_MalformedTalon(key, entry), replace=True)

    reports = await _run_for(rig, 120)

    assert _drained(rig) == []
    assert "talon_registry_unreadable: an entry names no job" in reports[-1]["unknown"]


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


@pytest.mark.asyncio
async def test_disabling_the_detector_stops_its_heartbeat(boot):
    """A detector turned off stops renewing its lease and writes nothing;
    the lease lapses on its own."""
    rig = await boot()
    await _tick(rig, minutes=0)
    [heartbeat] = _heartbeats(rig)
    await _lease_renewed(rig)
    before = await rig.detector._state_store.load()

    report = await _tick(rig, minutes=0, config=IdleWakeConfig())
    assert report == {"skipped": "disabled", "fired": False}
    await asyncio.wait({heartbeat}, timeout=5)
    assert heartbeat.cancelled()
    assert _heartbeats(rig) == []
    assert await rig.detector._state_store.load() == before

    # Enabled again, the same process takes up its lease where it left it.
    assert (await _tick(rig))["verdict"] == "idle"
    assert len(_heartbeats(rig)) == 1


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
# The Talon lane is the dispatcher's registry (#3758)
# ---------------------------------------------------------------------------
#
# A host names one agent as its only Talon dispatcher; every other agent
# sends it work orders, so a selecting agent's own registry stays empty while
# the dispatcher's lane runs. The detector reads the dispatcher's registry,
# found through the host's agent manager, and never the selecting agent's.

DISPATCHER = "Claw"
DISPATCHED = dataclasses.replace(CONFIG, talon_dispatcher=DISPATCHER)


class _SelectorsOwnTalon(_Talon):
    """The selecting agent's own Talon. It holds a job still marked running
    (a stale record from before the dispatcher policy), so a read of it would
    hold the agent busy. Every read of it is counted, and must be none."""

    def __init__(self) -> None:
        super().__init__()
        self.jobs = {"job-selector-stale": "running"}
        self.reads = 0

    async def finished_runs(self):
        self.reads += 1
        return await super().finished_runs()


class _Host:
    """The host's agent manager as an agent holds it (``_agent_manager``):
    ``get_agent(name)`` finds a co-hosted agent by name, case-insensitively,
    as :meth:`AgentManager.get_agent` does, or answers ``None``."""

    def __init__(self, agents: dict) -> None:
        self.agents = agents

    def get_agent(self, name):
        for key, agent in self.agents.items():
            if key.lower() == name.lower():
                return agent
        return None


def _co_hosted(name: str, *providers):
    from kestrel_sovereign.waits.engine import WaitRegistry

    agent = SimpleNamespace(
        agent_name=name, did=f"did:test:{name.lower()}", wait_registry=WaitRegistry()
    )
    for provider in providers:
        agent.wait_registry.register(provider)
    return agent


def _with_dispatcher(rig, *, lane=None):
    """Make ``rig``'s agent a selecting agent whose Talon work runs on the
    co-hosted :data:`DISPATCHER`. Returns the dispatcher's Talon provider."""
    rig.own_talon = _SelectorsOwnTalon()
    rig.agent.wait_registry.register(rig.own_talon, replace=True)
    lane = _Talon() if lane is None else lane
    rig.agent._agent_manager = _Host({DISPATCHER: _co_hosted(DISPATCHER, lane)})
    return lane


@pytest.mark.asyncio
async def test_a_job_running_on_the_dispatcher_suppresses_the_wake(boot):
    rig = await boot()
    lane = _with_dispatcher(rig)
    lane.jobs = {"job-claw": "running"}

    reports = [await _tick(rig, 0, DISPATCHED)] + await _run_for(rig, 120, DISPATCHED)

    assert _drained(rig) == [], reports[-1]
    assert {r["verdict"] for r in reports} == {"busy"}
    assert "talon_job:job-claw" in reports[-1]["work"]
    assert reports[-1]["talon_registry"] == DISPATCHER
    assert rig.own_talon.reads == 0


@pytest.mark.asyncio
async def test_the_wake_fires_once_the_dispatchers_lane_drains(boot):
    """The selecting agent's own registry still lists a running job; only the
    dispatcher's lane decides."""
    rig = await boot()
    lane = _with_dispatcher(rig)
    lane.jobs = {"job-claw": "running"}
    await _run_for(rig, 60, DISPATCHED)
    assert _drained(rig) == []

    lane.jobs = {"job-claw": "complete"}
    first = await _tick(rig, config=DISPATCHED)
    assert first["verdict"] == "idle", first
    went_idle = rig.clock.now
    await _run_for(rig, 25, DISPATCHED)
    assert _drained(rig) == []

    fired = await _tick(rig, config=DISPATCHED)
    assert fired["verdict"] == "fired", fired
    assert rig.clock.now - went_idle == timedelta(minutes=30)
    later = await _run_for(rig, 180, DISPATCHED)
    assert {r["verdict"] for r in later} == {"already_fired"}
    assert len(_drained(rig)) == 1
    assert rig.own_talon.reads == 0


@pytest.mark.asyncio
async def test_a_job_the_dispatcher_starts_for_anyone_starts_a_new_stretch(boot):
    """The dispatcher's registry holds every run its Talon started, whoever
    ordered it. A new one is the host's Talon lane going busy again."""
    rig = await boot()
    lane = _with_dispatcher(rig)
    lane.jobs = {"job-claw-1": "complete"}
    await _fire_once(rig, DISPATCHED)
    await _run_for(rig, 120, DISPATCHED)
    assert len(_drained(rig)) == 1

    lane.jobs["job-claw-2"] = "running"
    busy = await _tick(rig, config=DISPATCHED)
    assert busy["verdict"] == "busy" and "talon_job_dispatched" in busy["activity"]
    lane.jobs["job-claw-2"] = "complete"
    await _run_for(rig, 60, DISPATCHED)

    assert len(_drained(rig)) == 2
    assert _drained(rig)[1].payload["attempt"] == 1, "a new stretch, not a retry"
    assert rig.own_talon.reads == 0


def _no_host(rig):
    del rig.agent._agent_manager


def _dispatcher_not_running(rig):
    rig.agent._agent_manager = _Host({})


def _dispatcher_without_talon(rig):
    rig.agent._agent_manager = _Host({DISPATCHER: _co_hosted(DISPATCHER)})


def _host_lookup_fails(rig):
    def get_agent(name):
        raise RuntimeError("agent manager is shutting down")

    rig.agent._agent_manager = SimpleNamespace(get_agent=get_agent)


UNREACHABLE_DISPATCHERS = {
    "no host agent manager": (
        _no_host, "talon_dispatcher_unreachable: Claw is not an agent of this host"
    ),
    "dispatcher not running here": (
        _dispatcher_not_running,
        "talon_dispatcher_unreachable: Claw is not running on this host",
    ),
    "dispatcher loads no talon": (
        _dispatcher_without_talon, "talon_dispatcher_has_no_talon: Claw"
    ),
    "host lookup raises": (
        _host_lookup_fails,
        "talon_dispatcher_unreachable: Claw: RuntimeError: agent manager is "
        "shutting down",
    ),
}


@pytest.mark.parametrize("name", sorted(UNREACHABLE_DISPATCHERS))
@pytest.mark.asyncio
async def test_a_dispatcher_lane_that_cannot_be_read_never_reads_as_idle(boot, name):
    """Never the selecting agent's own (empty, or stale) registry instead."""
    rig = await boot()
    _with_dispatcher(rig)
    arrange, reason = UNREACHABLE_DISPATCHERS[name]
    arrange(rig)

    reports = await _run_for(rig, 120, DISPATCHED)

    assert _drained(rig) == []
    assert {r["verdict"] for r in reports} == {"unknown"}
    assert any(u.startswith(reason) for u in reports[-1]["unknown"]), reports[-1]
    assert rig.own_talon.reads == 0


@pytest.mark.asyncio
async def test_a_dispatcher_naming_the_agent_itself_reads_its_own_registry(boot):
    """The dispatcher's own detector: its registry is the lane."""
    rig = await boot()
    rig.talon.jobs = {"job-1": "running"}
    config = dataclasses.replace(CONFIG, talon_dispatcher=rig.agent.agent_name.upper())

    reports = await _run_for(rig, 60, config)

    assert {r["verdict"] for r in reports} == {"busy"}
    assert "talon_job:job-1" in reports[-1]["work"]
    assert reports[-1]["talon_registry"] == idle_wake.OWN_TALON_REGISTRY


@pytest.mark.asyncio
async def test_moving_the_lane_to_the_dispatcher_is_not_new_work(boot):
    """Pointing the detector at the dispatcher's registry finds jobs it never
    read before. They are a baseline, not a dispatch; the next one is."""
    rig = await boot()
    rig.talon.jobs = {"job-own": "complete"}
    await _fire_once(rig)

    lane = _with_dispatcher(rig)
    lane.jobs = {"job-claw-old": "complete", "job-claw-older": "complete"}
    reports = await _run_for(rig, 120, DISPATCHED)
    assert {r["verdict"] for r in reports} == {"already_fired"}
    assert len(_drained(rig)) == 1

    lane.jobs["job-claw-new"] = "complete"
    await _run_for(rig, 60, DISPATCHED)
    assert len(_drained(rig)) == 2


def _policy(dispatcher) -> dict:
    """kestrel.toml's ``[talon]`` table naming ``dispatcher``."""
    return {"preference": {"default_backend": "claude"},
            "policy": {"dispatcher": dispatcher}}


def test_talon_dispatcher_names_the_agent_that_runs_its_talon_work():
    emma = SimpleNamespace(agent_name="Emma", did="did:test:emma")
    on = {"enabled": True}

    def load(section, talon=None):
        return load_idle_wake_config(emma, {**on, **section}, talon)

    named = load({"talon_dispatcher": " Claw "})
    assert (named.talon_dispatcher, named.config_error) == ("Claw", None)
    assert load({}).talon_dispatcher is None
    assert load({}, {"preference": {}}).talon_dispatcher is None
    assert load({}, "not a table").talon_dispatcher is None
    not_a_table = load({}, {"policy": "Claw"})
    assert not_a_table.talon_dispatcher is None
    assert "[talon] policy" in not_a_table.config_error
    # The host's Talon policy names it when [idle_wake] does not.
    assert load({}, _policy("Claw")).talon_dispatcher == "Claw"
    # Named in both, they must agree.
    agreed = load({"talon_dispatcher": "claw"}, _policy("Claw"))
    assert (agreed.talon_dispatcher, agreed.config_error) == ("claw", None)
    disagree = load({"talon_dispatcher": "Meridian"}, _policy("Claw"))
    assert disagree.talon_dispatcher is None
    assert "[talon.policy] dispatcher = 'Claw'" in disagree.config_error
    for bad in ("", "  ", 5, ["Claw"], True):
        refused = load({"talon_dispatcher": bad})
        assert refused.enabled and refused.talon_dispatcher is None, bad
        assert refused.config_error and "talon_dispatcher" in refused.config_error
        policy_refused = load({}, _policy(bad))
        assert policy_refused.talon_dispatcher is None, bad
        assert "[talon.policy] dispatcher" in policy_refused.config_error


def test_the_dispatcher_is_read_from_the_hosts_talon_policy(monkeypatch):
    """With only ``[idle_wake]`` turned on, a fleet host whose kestrel.toml
    names Claw as its Talon dispatcher watches Claw's lane: the default never
    reads the selecting agent's own registry."""
    emma = SimpleNamespace(agent_name="Emma", did="did:test:emma")
    sections = {
        "idle_wake": {"enabled": True, "agents": ["Emma"]},
        "talon": _policy("Claw"),
    }
    monkeypatch.setattr(idle_wake, "load_section", lambda name: sections.get(name, {}))

    config = load_idle_wake_config(emma)

    assert (config.talon_dispatcher, config.config_error) == ("Claw", None)


@pytest.mark.asyncio
async def test_a_malformed_talon_dispatcher_never_reads_as_idle(boot):
    """An operator who wrote the key meant another agent's lane. Reading the
    agent's own registry in its place could report Talon idle mid-run."""
    rig = await boot()
    config = load_idle_wake_config(
        rig.agent, {"enabled": True, "talon_dispatcher": ""}
    )

    reports = await _run_for(rig, 120, config)

    assert _drained(rig) == []
    assert {r["verdict"] for r in reports} == {"unknown"}
    assert any(
        u.startswith("idle_wake_misconfigured: talon_dispatcher")
        for u in reports[-1]["unknown"]
    )


def test_the_host_finds_a_co_hosted_agent_by_name():
    """The lookup ``_Host`` stands in for: the agent manager binds itself to
    every agent it prepares and finds an agent by name."""
    import inspect

    from kestrel_sovereign.multi_agent.agent_manager import AgentManager

    assert list(inspect.signature(AgentManager.get_agent).parameters) == [
        "self", "name"
    ]
    assert "agent._agent_manager = self" in inspect.getsource(AgentManager)


def _real_talon_lane(root, name: str):
    """kestrel-feature-talon's own provider over ``root/agent_data/<name>``,
    as a fleet host builds it, and that agent's job directory."""
    from kestrel_feature_talon.feature import TalonCoordinatorFeature
    from kestrel_feature_talon.wait_provider import TalonWaitable

    agent_dir = root / "agent_data" / name
    agent_dir.mkdir(parents=True)
    feature = TalonCoordinatorFeature(
        SimpleNamespace(
            storage_path=str(agent_dir / "kestrel_prime.db"),
            did=f"did:test:{name.lower()}",
            agent_name=name,
        )
    )
    jobs = agent_dir / "talon_jobs"
    jobs.mkdir(mode=0o700, exist_ok=True)
    return TalonWaitable(feature), jobs


def _write_jobs(jobs, records: dict) -> tuple:
    """Persist ``records`` as Talon does (``jobs.json`` plus each job's log
    beside it) and return the registry's bytes and mtime, to show that only
    this test writes it."""
    registry = {}
    for job_id, status in records.items():
        log = jobs / f"{job_id}.log"
        log.write_text("processing\n")
        registry[job_id] = {
            "method": "cli_background", "status": status, "repo": "o/r",
            "issue": "3664", "log_path": str(log),
            "started_at": "2026-10-20T22:14:00+00:00",
            **({"completed_at": "2026-10-20T23:10:00+00:00"}
               if status in TERMINAL else {}),
        }
    path = jobs / "jobs.json"
    path.write_text(json.dumps(registry))
    return path.read_bytes(), path.stat().st_mtime_ns


def _unchanged(jobs, written: tuple) -> bool:
    path = jobs / "jobs.json"
    return (path.read_bytes(), path.stat().st_mtime_ns) == written


@pytest.mark.asyncio
async def test_the_real_talon_provider_on_the_dispatcher_drives_the_wake(
    boot, tmp_path
):
    """kestrel-feature-talon's own ``TalonWaitable``, not a double, on both
    agents. The dispatcher's running job holds the selecting agent busy; once
    it finishes, one wake fires, although the selecting agent's own registry
    still lists a job running. The detector writes neither registry."""
    pytest.importorskip("kestrel_feature_talon.wait_provider")
    from importlib.metadata import version

    from packaging.requirements import Requirement

    installed = version("kestrel-feature-talon")
    if not Requirement(UNFINISHED_JOBS_REQUIREMENT).specifier.contains(installed):
        pytest.skip(f"kestrel-feature-talon {installed} predates the idle detector's floor")

    rig = await boot()
    own, own_jobs = _real_talon_lane(tmp_path / "selector", "Emma")
    own_written = _write_jobs(own_jobs, {"job-emma-stale": "running"})
    rig.agent.wait_registry.register(own, replace=True)
    lane, lane_jobs = _real_talon_lane(tmp_path / "host", DISPATCHER)
    rig.agent._agent_manager = _Host({DISPATCHER: _co_hosted(DISPATCHER, lane)})

    written = _write_jobs(lane_jobs, {"job-claw": "running"})
    reports = [await _tick(rig, 0, DISPATCHED)] + await _run_for(rig, 60, DISPATCHED)
    assert {r["verdict"] for r in reports} == {"busy"}, reports[-1]
    assert "talon_job:job-claw" in reports[-1]["work"]
    assert reports[-1]["talon_registry"] == DISPATCHER
    assert _unchanged(lane_jobs, written)

    written = _write_jobs(lane_jobs, {"job-claw": "complete"})
    first = await _tick(rig, config=DISPATCHED)
    assert first["verdict"] == "idle", first
    await _run_for(rig, 25, DISPATCHED)
    assert _drained(rig) == []
    fired = await _tick(rig, config=DISPATCHED)
    assert fired["verdict"] == "fired", fired
    await _run_for(rig, 60, DISPATCHED)
    assert len(_drained(rig)) == 1
    assert _unchanged(lane_jobs, written)
    assert _unchanged(own_jobs, own_written)


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


# ---------------------------------------------------------------------------
# A self_followup is a pending wake until the scheduler has finished it
# ---------------------------------------------------------------------------


async def _insert_followup_row(db, *, task_id, agent_id, task_name, flags):
    enabled, claim_fenced, rollout_fenced = flags
    await db.execute(
        "INSERT INTO scheduled_tasks "
        "(id, agent_id, task_name, cron_expression, enabled, created_at, "
        " scheduler_claim_fenced, scheduler_rollout_fenced) "
        "VALUES (?, ?, ?, '', ?, ?, ?, ?)",
        (
            task_id, agent_id, task_name, enabled,
            datetime.now(timezone.utc).isoformat(), claim_fenced, rollout_fenced,
        ),
    )


@pytest.mark.parametrize(
    "flags,pending",
    [
        # (enabled, scheduler_claim_fenced, scheduler_rollout_fenced)
        ((1, 0, 0), True),  # waiting for its deadline
        ((0, 1, 0), True),  # claimed: its turn has not run, or its runner died
        ((0, 0, 1), True),  # fenced for a protocol rollout; activation re-enables it
        ((None, 0, 0), True),  # an unreadable state is not a finished one
        ((2, 0, 0), True),
        ((0, 2, 0), True),
        ((0, 0, 0), False),  # terminal, or paused
    ],
)
@pytest.mark.asyncio
async def test_a_self_followup_counts_until_the_scheduler_finishes_it(
    followup_env, flags, pending,
):
    agent, feature, _runner, db, _backend = followup_env
    await _insert_followup_row(
        db, task_id="mine", agent_id=agent.did, task_name="self_followup",
        flags=flags,
    )
    # Neither another agent's follow-up nor another task of this agent counts.
    await _insert_followup_row(
        db, task_id="theirs", agent_id="did:test:someone-else",
        task_name="self_followup", flags=(0, 1, 0),
    )
    await _insert_followup_row(
        db, task_id="other-task", agent_id=agent.did, task_name="memory_consolidate",
        flags=(0, 1, 0),
    )

    assert await feature._count_pending_self_followups() == (1 if pending else 0)


async def _attach_scheduler(rig):
    """The real scheduler over this rig's database, delivering through its
    dispatcher, and the detector reading the scheduler's own count."""
    from kestrel_sovereign.signals.sources.scheduler import (
        build_cron_registrations,
    )

    async def no_tool(name, args):
        raise AssertionError(f"unexpected tool lookup for {name}")

    for registration in build_cron_registrations(
        tool_lookup=no_tool,
        reason_codes_lookup=lambda _name: frozenset(),
        agent=rig.agent,
    ):
        rig.sources.register(registration)
    rig.agent.signal_registry = rig.sources
    feature = SchedulerFeature(rig.agent)
    feature._db = rig.db
    feature._agent_id = AGENT_DID
    runner = SchedulerRunner(rig.db, AGENT_DID, feature._dispatch_scheduled_task)
    await runner._ensure_tables()
    rig.pending_followups = feature._count_pending_self_followups
    return feature, runner


async def _followup_flags(rig):
    return await rig.db.fetchall(
        "SELECT enabled, scheduler_claim_fenced FROM scheduled_tasks "
        "WHERE agent_id = ? AND task_name = 'self_followup'",
        (AGENT_DID,),
    )


@pytest.mark.asyncio
async def test_a_claimed_self_followup_holds_the_wake_until_its_turn_has_run(boot):
    """The scheduler claims a due follow-up (``enabled = 0``) before its turn
    starts. Held there, past the threshold, it still suppresses the wake;
    once its turn has run, exactly one wake fires after the threshold."""
    from tests.utils.scheduler_ticks import settle_occurrences

    rig = await boot()
    feature, runner = await _attach_scheduler(rig)
    async with rig.agent._turn_lifecycle():
        added = await feature.schedule_add_deadline(
            run_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
            task_name="self_followup",
            args_json=json.dumps({"intent": "check CI on PR 3664"}),
        )
    assert added.status.value == "ok", added.error

    gate = _hold_turns(rig)  # the follow-up's turn waits at its entry
    await runner._tick()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while await _followup_flags(rig) != [(0, 1)]:
        assert loop.time() < deadline, await _followup_flags(rig)
        await asyncio.sleep(0.01)
    assert rig.agent._live_turn_id is None, "the turn has not started"

    # Not settled between evaluations: a wake fired here would wait at the
    # same gate, and must fail the test rather than hang it.
    reports = [await _evaluate(rig, minutes=0)]
    reports += [await _evaluate(rig) for _ in range(12)]
    assert _drained(rig) == [], reports[-1]
    assert {r["verdict"] for r in reports} == {"busy"}
    assert "self_followup_pending:1" in reports[-1]["deliveries"]

    gate.set()
    await settle_occurrences(runner)
    assert len(rig.agent.wake_sessions) == 1, "the follow-up's turn ran"
    assert await _followup_flags(rig) == [(0, 0)]

    first = await _tick(rig)
    assert first["verdict"] == "idle", first
    went_idle = rig.clock.now
    await _run_for(rig, 25)
    assert _drained(rig) == []
    fired = await _tick(rig)
    assert fired["verdict"] == "fired", fired
    assert rig.clock.now - went_idle == timedelta(minutes=30)
    await _run_for(rig, 60)
    assert len(_drained(rig)) == 1
