"""A scheduler readiness latch that one occurrence caused can clear (#3782).

Before #3782 nothing cleared ``SchedulerRunner._readiness_failure``, and the
host copied every runner latch into its own list for the process lifetime.
Two kinds of occurrence failure therefore took host ``/health`` to 503 until
a restart:

- a storage error in one occurrence's own read, such as the snapshot read in
  ``_claim_token_is_live`` whose connection close overran its budget, and
- an unknown commit outcome from a cron run fenced mid-commit.

The first no longer latches: the occurrence's lease expires and a later poll
recovers its row. The second still latches, but only until the claim it ended
no longer holds its schedule row.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    execution_evidence_carriers,
)
from kestrel_sovereign.features.scheduler.runner import (
    SCHEDULER_PROTOCOL_VERSION,
    SchedulerRunner,
    _is_occurrence_storage_failure,
    _is_reconcilable_occurrence_failure,
    _ReadinessReconciliation,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import QueryError
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.utils.async_waits import wait_until
from tests.utils.scheduler_ticks import settle_occurrences, tick_and_settle


async def _database(path) -> AsyncDatabase:
    backend = SQLiteBackend(str(path))
    await backend.connect()
    return AsyncDatabase(backend)


async def _seed_due(db, *, task_id: str = "task-1") -> str:
    due = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    await db.execute(
        """
        INSERT INTO scheduled_tasks
            (id, agent_id, task_name, cron_expression, args_json, enabled,
             next_run_at, created_at, schedule_kind, run_at, timezone_name,
             misfire_policy, idempotency_key, scheduler_protocol_version)
        VALUES (?, 'agent-1', 'wait_reconcile', '* * * * *', '{}', 1, ?, ?,
                'cron', NULL, 'UTC', 'skip', 'stable-effect', ?)
        """,
        (task_id, due, due, SCHEDULER_PROTOCOL_VERSION),
    )
    return due


async def _expire_lease(db, task_id: str = "task-1") -> None:
    """Let the claim's lease elapse without waiting for it."""
    await db.execute(
        "UPDATE scheduled_tasks SET lease_expires_at = ? WHERE id = ?",
        ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), task_id),
    )


async def _held_claim(db, task_id: str = "task-1") -> tuple[str, str] | None:
    """The ``(execution_id, claim_token)`` that holds the schedule row now."""
    row = await db.fetchone(
        "SELECT claim_execution_id, claim_token FROM scheduled_tasks "
        "WHERE id = ? AND scheduler_claim_fenced = 1",
        (task_id,),
    )
    return None if row is None else (row[0], row[1])


async def _stop(runner) -> BaseException | None:
    """Stop ``runner``; return the control evidence ``stop()`` re-raises."""
    try:
        await runner.stop()
    except ExecutionCommitOutcomeError as error:
        return error
    return None


async def _renew_until_stopped(_task):
    # The native renewal monitor stops cooperatively; keep its stop boundary
    # without renewing, so the test decides when the lease elapses.
    await asyncio.current_task()._kestrel_renewal_state.stop_requested.wait()


def test_storage_failure_classification():
    """Only a storage error with no control evidence ends just one occurrence."""
    query_failure = QueryError("Query failed: close did not complete")
    # Aggregation stamps an unresolved outcome on an ordinary carrier.
    commit_carrier = QueryError("Query failed during commit")
    commit_carrier._kestrel_unresolved_commit_outcome = "unknown"

    assert _is_occurrence_storage_failure(query_failure)
    assert not _is_occurrence_storage_failure(commit_carrier)
    assert not _is_occurrence_storage_failure(RuntimeError("claim-token read"))
    assert not _is_occurrence_storage_failure(ExecutionCommitOutcomeError("unknown"))


def test_reconcilable_failure_classification():
    """Durable rows settle commit outcomes and ordinary failures, not denials."""
    debt = ExecutionCommitOutcomeError("unknown")
    debt._kestrel_native_retirement_debts = ((object(), None),)

    for already_latched in (False, True):
        assert _is_reconcilable_occurrence_failure(
            ExecutionCommitOutcomeError("unknown"), already_latched=already_latched
        )
        assert _is_reconcilable_occurrence_failure(
            ExecutionCommitOutcomeError("committed"), already_latched=already_latched
        )
        assert not _is_reconcilable_occurrence_failure(
            ExecutionAuthorityError("runtime authority revoked"),
            already_latched=already_latched,
        )
        assert not _is_reconcilable_occurrence_failure(
            debt, already_latched=already_latched
        )
    # An ordinary failure is an occurrence outcome only inside a latch; one
    # that would start a latch is scheduler infrastructure.
    assert _is_reconcilable_occurrence_failure(
        RuntimeError("target failed while latched"), already_latched=True
    )
    assert not _is_reconcilable_occurrence_failure(
        RuntimeError("scheduler infrastructure failed"), already_latched=False
    )


@pytest.mark.asyncio
async def test_storage_error_in_final_admission_read_leaves_readiness_healthy(tmp_path):
    """The #3782 snapshot-read failure ends one occurrence, not the runtime."""

    db = await _database(tmp_path / "admission-storage-error.db")
    executor = AsyncMock(return_value="reconciled")
    runner = SchedulerRunner(db, "agent-1", executor, owner_id="admission-storage")
    runner._renew_lease = _renew_until_stopped
    original_claim_token_is_live = runner._claim_token_is_live
    token_reads = 0

    async def read_final_claim_token(task):
        nonlocal token_reads
        token_reads += 1
        if token_reads == 2:
            raise QueryError(
                "Query failed: SQLite connection close did not complete within 1.01s"
            )
        return await original_claim_token_is_live(task)

    try:
        await runner._ensure_tables()
        due = await _seed_due(db)
        # The initial renewal verifies the token once; fail the second
        # verification, at the final pre-effect admission boundary.
        runner._claim_token_is_live = read_final_claim_token

        await tick_and_settle(runner)

        assert token_reads == 2
        executor.assert_not_awaited()
        assert runner.readiness_failure is None
        claim = await db.fetchone(
            "SELECT enabled, scheduler_claim_fenced, claim_execution_id, next_run_at "
            "FROM scheduled_tasks WHERE id = ?",
            ("task-1",),
        )
        assert claim[:2] == (0, 1)
        execution_id = claim[2]
        assert claim[3] == due
        assert await db.fetchall(
            "SELECT id, status FROM task_execution_log WHERE task_id = ?",
            ("task-1",),
        ) == [(execution_id, "claimed")]

        # No effect started, so the expired claim is claimed again and runs.
        await _expire_lease(db)
        await tick_and_settle(runner)

        executor.assert_awaited_once()
        assert runner.readiness_failure is None
        assert await db.fetchall(
            "SELECT id, status, attempt_count FROM task_execution_log WHERE task_id = ?",
            ("task-1",),
        ) == [(execution_id, "success", 2)]
    finally:
        await runner.stop()
        await db.close()


@pytest.mark.asyncio
async def test_unknown_commit_outcome_latch_clears_once_its_claim_is_settled(tmp_path):
    """The latch holds while the claim does, then clears without a restart."""

    db = await _database(tmp_path / "commit-outcome.db")
    commit_outcome = ExecutionCommitOutcomeError("unknown")
    executor = AsyncMock(side_effect=[commit_outcome, "reconciled"])
    notifications = []
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="commit-outcome",
        on_protocol_failure=notifications.append,
    )
    runner._renew_lease = _renew_until_stopped
    stopped = False
    try:
        await runner._ensure_tables()
        await _seed_due(db)

        await tick_and_settle(runner)

        executor.assert_awaited_once()
        assert runner.readiness_failure is not None
        assert runner.readiness_failure_reconcilable
        # The host reads the live latch; nothing is recorded for its lifetime.
        assert notifications == []
        [claim] = runner._readiness_reconciliations
        execution_id = claim.execution_id
        # SQLite persists no effect-start marker, so recovery may claim this
        # occurrence again; PostgreSQL would hold it as ``executing``.
        assert await db.fetchone(
            "SELECT status FROM task_execution_log WHERE id = ?", (execution_id,)
        ) == ("claimed",)

        # The claim still holds the row until its lease elapses.
        await runner._reconcile_readiness_latch()
        assert runner.readiness_failure is not None

        await _expire_lease(db)
        await tick_and_settle(runner)
        await runner._reconcile_readiness_latch()

        assert runner.readiness_failure is None
        assert not runner.readiness_failure_reconcilable
        assert runner._readiness_reconciliations == set()
        assert executor.await_count == 2
        assert await db.fetchone(
            "SELECT status, attempt_count FROM task_execution_log WHERE id = ?",
            (execution_id,),
        ) == ("success", 2)
        # Readiness is restored; the run's control evidence is not erased.
        stopped = True
        assert await _stop(runner) is not None
    finally:
        if not stopped:
            await _stop(runner)
        await db.close()


@pytest.mark.asyncio
async def test_the_polling_loop_clears_a_settled_latch(tmp_path):
    """No caller has to ask: the armed runner restores its own readiness."""

    db = await _database(tmp_path / "polling-reconciliation.db")
    executor = AsyncMock(side_effect=[ExecutionCommitOutcomeError("unknown"), "reconciled"])
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="polling-reconciliation", poll_interval=0.01
    )
    runner._renew_lease = _renew_until_stopped
    try:
        await runner.start(polling=False)
        await _seed_due(db)
        await runner.arm()
        await wait_until(
            lambda: runner.readiness_failure is not None,
            timeout=30, interval=0.01, message="the commit outcome never latched",
        )
        await _expire_lease(db)
        await wait_until(
            lambda: runner.readiness_failure is None,
            timeout=30, interval=0.01, message="the polling loop never cleared the latch",
        )
        assert executor.await_count == 2
    finally:
        await _stop(runner)
        await db.close()


@pytest.mark.asyncio
async def test_settled_claims_never_clear_a_latch_another_failure_joined(tmp_path):
    """A protocol failure without a claim keeps the latch for the runner's life."""

    db = await _database(tmp_path / "joined-latch.db")
    executor = AsyncMock(side_effect=ExecutionCommitOutcomeError("unknown"))
    notifications = []
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="joined-latch",
        on_protocol_failure=notifications.append,
    )
    runner._renew_lease = _renew_until_stopped
    protocol_failure = RuntimeError("scheduler protocol row is unknown")
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        await tick_and_settle(runner)
        first = runner.readiness_failure
        assert runner.readiness_failure_reconcilable

        runner._latch_protocol_failure(protocol_failure)

        assert not runner.readiness_failure_reconcilable
        assert notifications == [protocol_failure]
        await _expire_lease(db)
        await tick_and_settle(runner)
        await runner._reconcile_readiness_latch()
        assert runner.readiness_failure is first
    finally:
        await _stop(runner)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_renewal",
    [
        # The renewal ``_execute_claim`` makes before the monitor starts; its
        # failure reaches ``_run_occurrence`` with no other observer.
        pytest.param(1, id="renewal-before-the-monitor"),
        # The real renewal monitor: its stop, the claim's retirement and the
        # occurrence each observe the same failure again.
        pytest.param(2, id="real-renewal-monitor"),
    ],
)
async def test_renewal_commit_outcome_latch_clears_once_its_claim_is_settled(
    tmp_path, failing_renewal
):
    """A renewal whose commit landed but whose outcome was lost ends one claim.

    Before the fix every observer after the first latched it without the
    claim, so the latch became permanent and the host recorded it for its
    lifetime even after recovery finalized the execution.
    """

    db = await _database(tmp_path / "renewal-commit-outcome.db")
    outcome_lost = asyncio.Event()
    effects = []

    async def execute(_task_name, _args):
        if not outcome_lost.is_set():
            # Hold the effect until the lost lease cancels it.
            await asyncio.Event().wait()
        effects.append("reconciled")
        return "reconciled"

    notifications = []
    runner = SchedulerRunner(
        db, "agent-1", execute, owner_id=f"renewal-outcome-{failing_renewal}",
        lease_seconds=3, on_protocol_failure=notifications.append,
    )
    real_renew_lease_once = runner._renew_lease_once
    renewals = 0

    async def renew_then_lose_the_outcome(task):
        nonlocal renewals
        renewals += 1
        renewed = await real_renew_lease_once(task)
        if renewals == failing_renewal:
            # The real SQLite renewal committed; its acknowledgement did not.
            outcome_lost.set()
            raise ExecutionCommitOutcomeError("unknown")
        return renewed

    runner._renew_lease_once = renew_then_lose_the_outcome
    stopped = False
    try:
        await runner._ensure_tables()
        await _seed_due(db)

        await tick_and_settle(runner)

        assert renewals == failing_renewal
        assert effects == []
        assert runner.readiness_failure is not None
        assert runner.readiness_failure_reconcilable
        assert notifications == []
        [claim] = runner._readiness_reconciliations
        assert await _held_claim(db) == (claim.execution_id, claim.claim_token)

        # The renewal extended the lease, so the claim still holds the row.
        await runner._reconcile_readiness_latch()
        assert runner.readiness_failure is not None

        await _expire_lease(db)
        await tick_and_settle(runner)
        await runner._reconcile_readiness_latch()

        assert effects == ["reconciled"]
        assert runner.readiness_failure is None
        assert notifications == []
        assert await db.fetchone(
            "SELECT status, attempt_count FROM task_execution_log WHERE id = ?",
            (claim.execution_id,),
        ) == ("success", 2)
        stopped = True
        assert await _stop(runner) is not None
    finally:
        if not stopped:
            await _stop(runner)
        await db.close()


@pytest.mark.asyncio
async def test_a_recovered_attempt_with_its_own_unknown_outcome_keeps_the_latch(tmp_path):
    """SQLite recovery reuses the execution ID under a new claim token.

    Keyed by execution ID alone, the second claim was dropped: the next check
    saw the first token gone and restored readiness while the recovered
    attempt's own unresolved claim still held the row.
    """

    db = await _database(tmp_path / "consecutive-outcomes.db")
    executor = AsyncMock(side_effect=[
        ExecutionCommitOutcomeError("unknown"),
        ExecutionCommitOutcomeError("unknown"),
        "reconciled",
    ])
    runner = SchedulerRunner(db, "agent-1", executor, owner_id="consecutive-outcomes")
    runner._renew_lease = _renew_until_stopped
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        await tick_and_settle(runner)
        [first] = runner._readiness_reconciliations

        await _expire_lease(db)
        await tick_and_settle(runner)

        assert executor.await_count == 2
        execution_id, token = await _held_claim(db)
        assert execution_id == first.execution_id
        assert token != first.claim_token
        second = _ReadinessReconciliation(
            schedule_id="task-1", agent_id="agent-1",
            execution_id=execution_id, claim_token=token,
        )
        assert runner._readiness_reconciliations == {first, second}

        await runner._reconcile_readiness_latch()

        # The first claim is settled; the recovered one still holds the row.
        assert runner._readiness_reconciliations == {second}
        assert runner.readiness_failure is not None
        assert runner.readiness_failure_reconcilable

        await _expire_lease(db)
        await tick_and_settle(runner)
        await runner._reconcile_readiness_latch()

        assert runner.readiness_failure is None
        assert executor.await_count == 3
        assert await db.fetchone(
            "SELECT status, attempt_count FROM task_execution_log WHERE id = ?",
            (execution_id,),
        ) == ("success", 3)
    finally:
        await _stop(runner)
        await db.close()


@pytest.mark.asyncio
async def test_the_polling_loop_keeps_a_latch_while_a_recovered_claim_is_unsettled(tmp_path):
    """Consecutive unknown outcomes through the armed loop, end to end."""

    db = await _database(tmp_path / "polling-consecutive-outcomes.db")
    executor = AsyncMock(side_effect=[
        ExecutionCommitOutcomeError("unknown"),
        ExecutionCommitOutcomeError("unknown"),
        "reconciled",
    ])
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="polling-consecutive", poll_interval=0.01
    )
    runner._renew_lease = _renew_until_stopped
    real_reconcile = runner._reconcile_readiness_latch
    passes = 0

    async def counted_reconcile():
        nonlocal passes
        await real_reconcile()
        passes += 1

    runner._reconcile_readiness_latch = counted_reconcile
    try:
        await runner.start(polling=False)
        await _seed_due(db)
        await runner.arm()
        await wait_until(
            lambda: executor.await_count == 1 and runner.readiness_failure is not None,
            timeout=30, interval=0.01, message="the first outcome never latched",
        )
        await _expire_lease(db)
        await wait_until(
            lambda: executor.await_count == 2,
            timeout=30, interval=0.01, message="the claim was never recovered",
        )
        await settle_occurrences(runner)
        held = await _held_claim(db)
        settled_at = passes
        await wait_until(
            lambda: passes >= settled_at + 3,
            timeout=30, interval=0.01, message="the polling loop stopped reconciling",
        )

        assert await _held_claim(db) == held
        assert runner.readiness_failure is not None
        assert runner.readiness_failure_reconcilable

        await _expire_lease(db)
        await wait_until(
            lambda: runner.readiness_failure is None,
            timeout=30, interval=0.01, message="the polling loop never cleared the latch",
        )
        assert executor.await_count == 3
    finally:
        await _stop(runner)
        await db.close()


@pytest.mark.asyncio
async def test_relatching_a_reconcilable_failure_without_its_claim_keeps_it(tmp_path):
    """The done callback and ``stop()`` observe a failure without its claim."""

    db = await _database(tmp_path / "relatch.db")
    executor = AsyncMock(side_effect=ExecutionCommitOutcomeError("unknown"))
    notifications = []
    runner = SchedulerRunner(
        db, "agent-1", executor, owner_id="relatch",
        on_protocol_failure=notifications.append,
    )
    runner._renew_lease = _renew_until_stopped
    try:
        await runner._ensure_tables()
        await _seed_due(db)
        await tick_and_settle(runner)
        failure = runner.readiness_failure
        [claim] = runner._readiness_reconciliations

        runner._latch_protocol_failure(failure)
        # An aggregate that carries it is the same evidence.
        runner._latch_protocol_failure(
            BaseExceptionGroup("observed again", [failure, asyncio.CancelledError()])
        )

        assert runner.readiness_failure is failure
        assert runner.readiness_failure_reconcilable
        assert runner._readiness_reconciliations == {claim}
        assert notifications == []

        # The finding belongs to the runner that made it.
        other = SchedulerRunner(db, "agent-1", executor, owner_id="relatch-other")
        other._latch_protocol_failure(failure)
        assert not other.readiness_failure_reconcilable
    finally:
        await _stop(runner)
        await db.close()


@pytest.mark.asyncio
async def test_an_ordinary_failure_starts_a_latch_only_as_infrastructure(tmp_path):
    """A claim does not make scheduler infrastructure an occurrence outcome."""

    db = await _database(tmp_path / "ordinary-latch.db")
    notifications = []
    runner = SchedulerRunner(
        db, "agent-1", AsyncMock(), owner_id="ordinary-latch",
        on_protocol_failure=notifications.append,
    )
    first = _ReadinessReconciliation("task-1", "agent-1", "execution-1", "token-1")
    second = _ReadinessReconciliation("task-2", "agent-1", "execution-2", "token-2")
    try:
        infrastructure = RuntimeError("claim admission read failed")
        runner._latch_protocol_failure(infrastructure, reconciliation=first)
        assert not runner.readiness_failure_reconcilable
        assert notifications == [infrastructure]

        latched = SchedulerRunner(db, "agent-1", AsyncMock(), owner_id="ordinary-joins")
        latched._latch_protocol_failure(
            ExecutionCommitOutcomeError("unknown"), reconciliation=first
        )
        # Inside a reconcilable latch, an ordinary failure of another claim
        # is that claim's outcome.
        latched._latch_protocol_failure(
            RuntimeError("target failed while latched"), reconciliation=second
        )
        assert latched.readiness_failure_reconcilable
        assert latched._readiness_reconciliations == {first, second}
    finally:
        await db.close()


def test_evidence_carriers_follow_causes_and_groups_not_context():
    """The graph the latch searches for a runner's earlier finding."""
    cause = ExecutionCommitOutcomeError("unknown")
    carrier = RuntimeError("renewal retired")
    carrier.__cause__ = cause
    member = ValueError("sibling")
    group = BaseExceptionGroup("terminal", [carrier, member])
    unrelated = KeyError("handling context")
    group.__context__ = unrelated

    carriers = execution_evidence_carriers(group)

    assert {id(error) for error in carriers} == {
        id(group), id(carrier), id(cause), id(member)
    }
    assert execution_evidence_carriers(None) == ()


def _runner_agent(runner):
    return SimpleNamespace(
        features={"SchedulerFeature": SimpleNamespace(_runner=runner)}
    )


def _fake_app():
    return SimpleNamespace(
        state=SimpleNamespace(
            host_scheduler_runner=None,
            scheduler_readiness_failures=[],
            scheduler_reconciling_failures=[],
        )
    )


def test_host_reports_a_reconciling_runner_only_while_it_is_latched():
    """A runner latch that clears itself is not kept for the host's lifetime."""
    from kestrel_sovereign.server import (
        _latch_active_scheduler_runner_failures,
        _scheduler_readiness_failures,
    )

    runner = SimpleNamespace(
        readiness_latch=(ExecutionCommitOutcomeError("unknown"), True)
    )
    manager = MagicMock()
    manager.list_agents.return_value = {"Emma": _runner_agent(runner)}
    app = _fake_app()
    record = {
        "scope": "runtime",
        "state": "unavailable",
        "error_code": "scheduler_runtime_unavailable",
        "cause_type": "ExecutionCommitOutcomeError",
        "agent": "Emma",
    }

    _latch_active_scheduler_runner_failures(app, None, manager)
    assert _scheduler_readiness_failures(app) == [record]
    assert app.state.scheduler_readiness_failures == []

    # An inventory that cannot be read is not evidence of recovery.
    manager.list_agents.side_effect = RuntimeError("inventory unavailable")
    _latch_active_scheduler_runner_failures(app, None, manager)
    assert _scheduler_readiness_failures(app) == [record]

    manager.list_agents.side_effect = None
    runner.readiness_latch = None
    _latch_active_scheduler_runner_failures(app, None, manager)
    assert _scheduler_readiness_failures(app) == []


def test_host_keeps_a_runner_latch_that_cannot_clear_itself():
    """Every other runner latch stays recorded after the runner forgets it."""
    from kestrel_sovereign.server import (
        _latch_active_scheduler_runner_failures,
        _scheduler_readiness_failures,
    )

    runner = SimpleNamespace(
        readiness_latch=(RuntimeError("scheduler protocol row is unknown"), False)
    )
    manager = MagicMock()
    manager.list_agents.return_value = {"Emma": _runner_agent(runner)}
    app = _fake_app()

    _latch_active_scheduler_runner_failures(app, None, manager)
    runner.readiness_latch = None
    _latch_active_scheduler_runner_failures(app, None, manager)

    assert _scheduler_readiness_failures(app) == [
        {
            "scope": "runtime",
            "state": "unavailable",
            "error_code": "scheduler_runtime_unavailable",
            "cause_type": "RuntimeError",
            "agent": "Emma",
        }
    ]


def test_host_reads_a_runner_latch_in_one_step():
    """A probe never pairs a failure with a flag read at another moment.

    Health runs on a threadpool thread while the runner changes on the loop.
    A runner that publishes ``readiness_latch`` is judged by that pair alone,
    so a failure attribute still visible mid-change cannot become a lifetime
    record.
    """
    from kestrel_sovereign.server import (
        _latch_active_scheduler_runner_failures,
        _scheduler_readiness_failures,
    )

    class _ClearedRunner:
        readiness_latch = None
        readiness_failure = ExecutionCommitOutcomeError("unknown")

    manager = MagicMock()
    manager.list_agents.return_value = {"Emma": _runner_agent(_ClearedRunner())}
    app = _fake_app()

    _latch_active_scheduler_runner_failures(app, None, manager)

    assert _scheduler_readiness_failures(app) == []
