"""One slow occurrence must not stop the scheduler claiming other due work.

#3465: ``SchedulerRunner._tick`` used to await every due occurrence of a poll
together, so the next poll waited for the slowest. Two per-minute tasks that
blocked for 20 minutes kept Emma's 08:00 ``morning_signal`` and
``backup_snapshot`` from being claimed until they had passed their misfire
grace. Each occurrence now runs on its own runner-owned task, and the runner
records the ones still in flight so it never admits a row twice.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SchedulerRunner,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.utils.async_waits import wait_until
from tests.utils.scheduler_ticks import settle_occurrences

# A hang guard, not a coordination window: every wait below is event-driven.
_HANG_GUARD_SECONDS = 30


async def _database(path) -> AsyncDatabase:
    backend = SQLiteBackend(str(path))
    await backend.connect()
    return AsyncDatabase(backend)


async def _seed_due(
    db,
    task_id: str,
    task_name: str,
    *,
    agent_id: str = "agent-1",
    late_seconds: float = 2,
    misfire_grace_seconds: Optional[int] = None,
) -> None:
    due = (datetime.now(timezone.utc) - timedelta(seconds=late_seconds)).isoformat()
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at, misfire_policy, misfire_grace_seconds,
             idempotency_key, scheduler_protocol_version)
        VALUES (?, ?, ?, '* * * * *', '{}', 1, ?, ?, 'skip', ?, ?, ?)
        """,
        (
            task_id,
            agent_id,
            task_name,
            due,
            due,
            misfire_grace_seconds,
            f"effect-{task_id}",
            SCHEDULER_PROTOCOL_VERSION,
        ),
    )


async def _log_statuses(db, task_id: str) -> list:
    return [
        row[0]
        for row in await db.fetchall(
            "SELECT status FROM task_execution_log WHERE task_id = ?", (task_id,)
        )
    ]


class _SlowExecutor:
    """Blocks the ``slow`` task until released; every other task returns."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.slow_started = asyncio.Event()
        self.slow_cancelled = asyncio.Event()
        self.release_slow = asyncio.Event()
        self.ran: dict[str, asyncio.Event] = {}

    def ran_event(self, task_name: str) -> asyncio.Event:
        return self.ran.setdefault(task_name, asyncio.Event())

    async def __call__(self, task_name, _args):
        self.calls.append(task_name)
        if task_name == "slow":
            self.slow_started.set()
            try:
                await self.release_slow.wait()
            except asyncio.CancelledError:
                self.slow_cancelled.set()
                raise
            return "slow done"
        self.ran_event(task_name).set()
        return f"{task_name} done"


@pytest.mark.asyncio
async def test_slow_occurrence_does_not_stop_the_loop_claiming_later_due_work(
    tmp_path,
):
    """A row that comes due while another occurrence blocks still runs."""

    db = await _database(tmp_path / "slow-occurrence.db")
    executor = _SlowExecutor()
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="slow-occurrence", poll_interval=0.01
    )
    try:
        await runner.start(polling=False)
        await _seed_due(db, "slow-task", "slow")
        await runner.arm()
        await asyncio.wait_for(
            executor.slow_started.wait(), timeout=_HANG_GUARD_SECONDS
        )

        # Due only after the slow occurrence is already blocked, like the
        # 08:00 crons behind the stuck per-minute tasks.
        await _seed_due(db, "later-task", "morning_signal", late_seconds=0)
        await asyncio.wait_for(
            executor.ran_event("morning_signal").wait(),
            timeout=_HANG_GUARD_SECONDS,
        )
        await wait_until(
            lambda: "later-task" not in runner._occurrences,
            timeout=_HANG_GUARD_SECONDS,
            interval=0.01,
            message="later occurrence never finished",
        )

        assert not executor.release_slow.is_set()
        assert await _log_statuses(db, "later-task") == ["success"]
        assert await _log_statuses(db, "slow-task") == ["claimed"]
        # The loop kept polling past the slow row without re-admitting it.
        assert executor.calls.count("slow") == 1
        assert runner.readiness_failure is None
    finally:
        executor.release_slow.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_tick_never_admits_a_second_occurrence_for_an_in_flight_row(tmp_path):
    """A row still waiting for its claim is not claimed again by later ticks."""

    db = await _database(tmp_path / "in-flight-registry.db")
    executor = _SlowExecutor()
    # One slot: the queued row stays due, and selected, behind the slow one.
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="in-flight", max_concurrent_tasks=1
    )
    claim_attempts: list[str] = []
    original_claim = runner._claim

    async def counting_claim(task, polled_at):
        claim_attempts.append(task.id)
        return await original_claim(task, polled_at)

    runner._claim = counting_claim
    try:
        await runner._ensure_tables()
        await _seed_due(db, "slow-task", "slow", late_seconds=3)
        await _seed_due(db, "queued-task", "queued")

        await runner._tick()
        await asyncio.wait_for(
            executor.slow_started.wait(), timeout=_HANG_GUARD_SECONDS
        )
        admitted = {
            schedule_id: occurrence.task
            for schedule_id, occurrence in runner._occurrences.items()
        }
        assert set(admitted) == {"slow-task", "queued-task"}
        assert runner._occurrences["slow-task"].awaiting_claim is False
        assert runner._occurrences["queued-task"].awaiting_claim is True

        for _ in range(3):
            await runner._tick()
        assert {
            schedule_id: occurrence.task
            for schedule_id, occurrence in runner._occurrences.items()
        } == admitted

        executor.release_slow.set()
        await asyncio.wait_for(
            settle_occurrences(runner), timeout=_HANG_GUARD_SECONDS
        )
        assert claim_attempts == ["slow-task", "queued-task"]
        assert executor.calls == ["slow", "queued"]
        assert runner._occurrences == {}
        assert runner._oldest_unclaimed_admission_monotonic is None
    finally:
        executor.release_slow.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_occurrence_that_waited_for_a_slot_still_misfires_when_late(tmp_path):
    """Lateness is still measured at the claim, so a queued row can misfire."""

    db = await _database(tmp_path / "queued-misfire.db")
    executor = _SlowExecutor()
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="queued-misfire", max_concurrent_tasks=1
    )
    try:
        await runner._ensure_tables()
        await _seed_due(db, "slow-task", "slow", late_seconds=3)
        await _seed_due(db, "queued-task", "queued", misfire_grace_seconds=3)

        await runner._tick()
        await asyncio.wait_for(
            executor.slow_started.wait(), timeout=_HANG_GUARD_SECONDS
        )
        # Queued at 2s late with 3s of grace: it is genuinely late once the
        # slot frees, and lateness only grows from here.
        await asyncio.sleep(1.5)
        executor.release_slow.set()
        await asyncio.wait_for(
            settle_occurrences(runner), timeout=_HANG_GUARD_SECONDS
        )

        assert await _log_statuses(db, "slow-task") == ["success"]
        assert await _log_statuses(db, "queued-task") == ["skipped_misfire"]
        assert executor.calls == ["slow"]
    finally:
        executor.release_slow.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_paged_occurrence_keeps_its_page_after_the_cursor_moves(tmp_path):
    """An in-flight occurrence keeps renewing after the poll reaches page two."""

    db = await _database(tmp_path / "paged-in-flight.db")
    executor = _SlowExecutor()
    authorized = ("agent-1", "agent-2")

    def page_provider(after, limit):
        return tuple(
            agent_id for agent_id in authorized if after is None or agent_id > after
        )[:limit]

    runner = SchedulerRunner(
        db,
        None,
        executor,
        owner_id="paged-in-flight",
        lease_seconds=1,
        authorized_agent_ids=(),
        authorized_agent_ids_page_provider=page_provider,
        authorized_agent_ids_page_size=1,
        is_agent_authorized=lambda agent_id: agent_id in authorized,
    )
    page_moved = False
    renewals_after_move: list[bool] = []
    renewed_after_move = asyncio.Event()
    original_renew = runner._renew_live_claim_once

    async def recording_renew(task):
        renewed = await original_renew(task)
        if page_moved:
            renewals_after_move.append(renewed)
            renewed_after_move.set()
        return renewed

    runner._renew_live_claim_once = recording_renew
    try:
        await runner._ensure_tables()
        await _seed_due(db, "slow-task", "slow")

        await runner._tick()
        assert runner._authorized_agent_ids_page == ("agent-1",)
        await asyncio.wait_for(
            executor.slow_started.wait(), timeout=_HANG_GUARD_SECONDS
        )
        await runner._tick()
        assert runner._authorized_agent_ids_page == ("agent-2",)
        page_moved = True

        await asyncio.wait_for(
            renewed_after_move.wait(), timeout=_HANG_GUARD_SECONDS
        )
        assert renewals_after_move[0] is True
        assert not executor.slow_cancelled.is_set()
        executor.release_slow.set()
        await asyncio.wait_for(
            settle_occurrences(runner), timeout=_HANG_GUARD_SECONDS
        )
        assert await _log_statuses(db, "slow-task") == ["success"]
    finally:
        executor.release_slow.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_paged_occurrence_is_still_revoked_by_live_authority(tmp_path):
    """Keeping its page never outlives the live ``is_agent_authorized`` check."""

    db = await _database(tmp_path / "paged-revoked.db")
    executor = _SlowExecutor()
    fleet = ("agent-1", "agent-2")
    authorized = set(fleet)

    def page_provider(after, limit):
        return tuple(
            agent_id for agent_id in fleet if after is None or agent_id > after
        )[:limit]

    runner = SchedulerRunner(
        db,
        None,
        executor,
        owner_id="paged-revoked",
        lease_seconds=1,
        authorized_agent_ids=(),
        authorized_agent_ids_page_provider=page_provider,
        authorized_agent_ids_page_size=1,
        is_agent_authorized=lambda agent_id: agent_id in authorized,
    )
    try:
        await runner._ensure_tables()
        await _seed_due(db, "slow-task", "slow")
        await runner._tick()
        await asyncio.wait_for(
            executor.slow_started.wait(), timeout=_HANG_GUARD_SECONDS
        )
        await runner._tick()
        assert runner._authorized_agent_ids_page == ("agent-2",)

        authorized.discard("agent-1")
        await asyncio.wait_for(
            executor.slow_cancelled.wait(), timeout=_HANG_GUARD_SECONDS
        )
        await asyncio.wait_for(
            settle_occurrences(runner), timeout=_HANG_GUARD_SECONDS
        )
        assert await _log_statuses(db, "slow-task") == ["claimed"]
    finally:
        executor.release_slow.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_unpaged_scope_still_drops_a_removed_tenant_mid_execution(tmp_path):
    """Only paged hosts keep an admitting page; a live scope stays current."""

    db = await _database(tmp_path / "unpaged-removed.db")
    executor = _SlowExecutor()
    scope = {"agent-1"}
    runner = SchedulerRunner(
        db,
        None,
        executor,
        owner_id="unpaged-removed",
        lease_seconds=1,
        authorized_agent_ids=(),
        authorized_agent_ids_provider=lambda: tuple(scope),
    )
    try:
        await runner._ensure_tables()
        await _seed_due(db, "slow-task", "slow")
        await runner._tick()
        await asyncio.wait_for(
            executor.slow_started.wait(), timeout=_HANG_GUARD_SECONDS
        )

        scope.clear()
        await asyncio.wait_for(
            executor.slow_cancelled.wait(), timeout=_HANG_GUARD_SECONDS
        )
        await asyncio.wait_for(
            settle_occurrences(runner), timeout=_HANG_GUARD_SECONDS
        )
        assert await _log_statuses(db, "slow-task") == ["claimed"]
    finally:
        executor.release_slow.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_stop_cancels_and_joins_in_flight_occurrences(tmp_path):
    """Occurrences outlive their tick, so ``stop`` owns their cancellation."""

    db = await _database(tmp_path / "stop-in-flight.db")
    executor = _SlowExecutor()
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="stop-in-flight", poll_interval=0.01
    )
    try:
        await runner.start(polling=False)
        await _seed_due(db, "slow-task", "slow")
        await runner.arm()
        await asyncio.wait_for(
            executor.slow_started.wait(), timeout=_HANG_GUARD_SECONDS
        )
        occurrence = runner._occurrences["slow-task"].task

        await asyncio.wait_for(runner.stop(), timeout=_HANG_GUARD_SECONDS)

        assert executor.slow_cancelled.is_set()
        assert occurrence.cancelled()
        assert runner._occurrences == {}
        assert await _log_statuses(db, "slow-task") == ["claimed"]
    finally:
        executor.release_slow.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_hung_claim_is_a_stalled_tick_after_the_poll_returned(tmp_path):
    """An admission stuck before its claim still trips the stall watchdog."""

    db = await _database(tmp_path / "hung-claim.db")
    claim_entered = asyncio.Event()
    release_claim = asyncio.Event()
    runner = SchedulerRunner(db, "agent-1", _SlowExecutor(), owner_id="hung-claim")
    original_claim = runner._claim

    async def hung_claim(task, polled_at):
        claim_entered.set()
        await release_claim.wait()
        return await original_claim(task, polled_at)

    runner._claim = hung_claim
    runner._tick_in_progress_limit_seconds = 0.05
    try:
        await runner._ensure_tables()
        await _seed_due(db, "hung-task", "quick")
        # Model an armed runner without a polling loop re-stamping its poll.
        runner._running = True

        await runner._tick()
        await asyncio.wait_for(claim_entered.wait(), timeout=_HANG_GUARD_SECONDS)
        await asyncio.sleep(0.1)
        assert runner._tick_started_monotonic is None
        assert runner._active_claimed_execution_count == 0
        assert runner.tick_stalled is True

        release_claim.set()
        await asyncio.wait_for(
            settle_occurrences(runner), timeout=_HANG_GUARD_SECONDS
        )
        assert runner.tick_stalled is False
        assert await _log_statuses(db, "hung-task") == ["success"]
    finally:
        release_claim.set()
        runner._running = False
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_rows_waiting_for_a_slot_keep_the_reported_tick_open(tmp_path):
    """A queued admission keeps health's in-progress exemption, as before.

    Status health exempts overdue rows while a reported tick is unfinished,
    because that batch owns rows still waiting behind the concurrency cap.
    The poll returns at once now, so the batch has to stay open until its
    queued rows have claimed.
    """

    db = await _database(tmp_path / "open-batch.db")
    executor = _SlowExecutor()
    runner = SchedulerRunner(
        db,
        "agent-1",
        executor,
        owner_id="open-batch",
        poll_interval=0.01,
        max_concurrent_tasks=1,
    )
    try:
        await runner.start(polling=False)
        await _seed_due(db, "slow-task", "slow", late_seconds=3)
        await _seed_due(db, "queued-task", "queued")
        await runner.arm()
        await asyncio.wait_for(
            executor.slow_started.wait(), timeout=_HANG_GUARD_SECONDS
        )
        queued = runner._occurrences["queued-task"]
        assert queued.awaiting_slot is True
        assert queued.admitted_at is not None

        # Let later polls complete while the queued row still waits.
        polls_seen = 0
        previous_poll: Optional[float] = None

        def later_polls_completed() -> bool:
            nonlocal polls_seen, previous_poll
            current = runner._tick_started_monotonic
            if current is not None and current != previous_poll:
                previous_poll = current
                polls_seen += 1
            return polls_seen >= 3 and runner._tick_started_monotonic is None

        await wait_until(
            later_polls_completed,
            timeout=_HANG_GUARD_SECONDS,
            interval=0.001,
            message="polling loop stopped ticking",
        )
        assert runner._last_tick_started_at == queued.admitted_at
        assert runner._last_tick_completed_at is None or (
            datetime.fromisoformat(runner._last_tick_completed_at)
            < datetime.fromisoformat(queued.admitted_at)
        )

        executor.release_slow.set()
        await wait_until(
            lambda: "queued-task" not in runner._occurrences,
            timeout=_HANG_GUARD_SECONDS,
            interval=0.01,
            message="queued occurrence never finished",
        )
        await wait_until(
            lambda: runner._last_tick_completed_at is not None
            and datetime.fromisoformat(runner._last_tick_completed_at)
            >= datetime.fromisoformat(runner._last_tick_started_at),
            timeout=_HANG_GUARD_SECONDS,
            interval=0.01,
            message="reported tick never closed after the queue drained",
        )
        assert executor.calls == ["slow", "queued"]
    finally:
        executor.release_slow.set()
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_row_with_a_free_slot_closes_its_batch_in_the_same_poll(tmp_path):
    """Only rows queued behind the cap keep the reported tick open.

    A row that got a free slot is usually still mid-claim when its poll
    ends. If that held the batch open until the next poll, a runner that
    admits work on every poll would never report a completed tick, and the
    health overdue exemption would stay on for good.
    """

    db = await _database(tmp_path / "free-slot-batch.db")
    executor = _SlowExecutor()
    # One poll only: a batch left open here would never be closed by a later
    # poll.
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="free-slot-batch", poll_interval=3600
    )
    try:
        await runner.start(polling=False)
        await _seed_due(db, "quick-task", "quick")
        await runner.arm()
        await asyncio.wait_for(
            executor.ran_event("quick").wait(), timeout=_HANG_GUARD_SECONDS
        )
        await wait_until(
            lambda: runner._tick_started_monotonic is None
            and runner._last_tick_started_at is not None,
            timeout=_HANG_GUARD_SECONDS,
            interval=0.01,
            message="first poll never finished",
        )

        assert runner._last_tick_completed_at is not None
        assert datetime.fromisoformat(
            runner._last_tick_completed_at
        ) >= datetime.fromisoformat(runner._last_tick_started_at)
    finally:
        await runner.stop()
        await db.close()
