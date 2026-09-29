"""A2A cognition dispatches are durably owned and harvested (#2532, AC4).

The event-manager A2A callbacks run only after their TaskStore transition is
persisted.  Their selected durable cognition consumers preserve the one-shot
wake through Hold/restart; the callback task still owns and harvests the live
dispatch so failures remain observable.

"Detached" still carries two obligations, and these tests pin both:

  - **Owned** — the dispatch runs inside a task registered with the
    agent's tracker, so shutdown drains it instead of it vanishing with a
    "coroutine was never awaited" warning.
  - **Harvested** — the terminal ``SignalResult`` is awaited, so a wake
    the dispatcher accepted and then FAILED or dropped is observable in
    the log rather than silently lost.

Awaiting *acceptance* alone would satisfy neither: ``enqueue_signal``
returns its ``SignalHandle`` the moment the signal is queued.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sdk.signals import SignalHandle, SignalMode, SignalResult, Status

from kestrel_sovereign.a2a.types import TaskState
from kestrel_sovereign.agent.event_manager import EventManagerMixin
from kestrel_sovereign.signals import DurableAdmissionDisposition
from kestrel_sovereign.signals.dispatcher import (
    DurableAdmissionResult,
    SignalDispatchHandle,
)
from kestrel_sovereign.signals.registry import SourceRegistry
from kestrel_sovereign.signals.sources.a2a import (
    build_a2a_task_complete_registration,
)
from kestrel_sovereign.signals.sources.a2a_task_submitted import (
    build_a2a_task_submitted_registration,
)


def _signal_handle(status: Status, *, error: str | None = None) -> SignalHandle:
    """A real handle whose task resolves to a real terminal result."""

    async def _terminal() -> SignalResult:
        return SignalResult(
            signal_id="sig-test",
            status=status,
            mode=SignalMode.COGNITION,
            duration_ms=1,
            error=error,
        )

    return SignalHandle(
        signal_id="sig-test", task=asyncio.ensure_future(_terminal()),
    )


class _Agent(EventManagerMixin):
    """Minimal agent exposing just what the two callbacks touch."""

    def __init__(self):
        self.did = "did:test:agent"
        self._pending_task_notifications = []
        self.signal_registry = SourceRegistry()
        self.signal_registry.register(build_a2a_task_submitted_registration())
        self.signal_registry.register(build_a2a_task_complete_registration())
        self.dispatcher = MagicMock()
        # These callbacks exercise the durable path, which a real dispatcher
        # takes only once the consumer is registered.
        self.dispatcher.has_durable_consumer = AsyncMock(return_value=True)
        # No earlier submission wake exists unless a test says otherwise.
        self.dispatcher.get_durable_delivery_for_source_event = AsyncMock(
            return_value=None
        )
        self.task_manager = MagicMock()
        self.task_manager.get_task = AsyncMock(
            side_effect=lambda task_id: _task(task_id, state=TaskState.SUBMITTED)
        )
        self.task_manager.get_task_cancellation_snapshot = AsyncMock(
            side_effect=lambda _task_id, *, recipient_agent_id: MagicMock(
                state="submitted",
                actor_agent_id=None,
                recipient_agent_id=recipient_agent_id,
            )
        )
        self.tracked: list = []

    def _track_background_task(self, coro, *, name: str) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self.tracked.append((task, name))
        return task


def _task(task_id: str = "task-1", *, state=TaskState.COMPLETED):
    task = MagicMock()
    task.id = task_id
    task.status.state = state
    task.status.message = None
    task.metadata = {"agent_id": "Meridian", "skill": "research"}
    return task


async def _drain(agent) -> None:
    for task, _name in agent.tracked:
        await asyncio.wait_for(task, timeout=5)


def _durable_signal_handle(
    disposition: DurableAdmissionDisposition,
    *,
    status: Status = Status.OK,
) -> SignalDispatchHandle:
    handle = _signal_handle(status)
    admission = asyncio.get_running_loop().create_future()
    admission.set_result(DurableAdmissionResult(disposition, "sig-test"))
    return SignalDispatchHandle(
        signal_id=handle.signal_id,
        task=handle.task,
        durable_admission=admission,
    )


@pytest.mark.asyncio
async def test_boot_reconciliation_recreates_missing_a2a_outbox_rows():
    """TaskStore, not the one-shot callback, is the crash-safe wake authority."""

    agent = _Agent()
    submitted = _task("submitted-gap", state=TaskState.SUBMITTED)
    working = _task("working-gap", state=TaskState.WORKING)
    completed = _task("completed-gap", state=TaskState.COMPLETED)
    agent.task_manager.list_cognition_wake_candidates = AsyncMock(
        return_value=[
            SimpleNamespace(task=submitted, lifecycle_revision=0),
            SimpleNamespace(task=working, lifecycle_revision=7),
            SimpleNamespace(task=completed, lifecycle_revision=8),
        ]
    )
    agent.dispatcher.enqueue_durable_cognition = AsyncMock(
        side_effect=lambda *args, **kwargs: _durable_signal_handle(
            DurableAdmissionDisposition.COMMITTED
        )
    )

    before = datetime.now(timezone.utc)
    await agent.reconcile_a2a_cognition_wakes()
    after = datetime.now(timezone.utc)

    calls = agent.dispatcher.enqueue_durable_cognition.await_args_list
    assert [call.kwargs["source_event_id"] for call in calls] == [
        "submitted-gap",
        "working-gap:working:7",
        "completed-gap",
    ]
    assert [call.kwargs["consumer_id"] for call in calls] == [
        "core.a2a-task-submitted-cognition-v1",
        "core.a2a-task-submitted-cognition-v1",
        "core.a2a-task-complete-cognition-v1",
    ]
    # Each window is the owning source's registered retention, not a copy.
    window = agent.task_manager.list_cognition_wake_candidates.await_args.kwargs
    for key, registration in (
        ("live_changed_since", build_a2a_task_submitted_registration()),
        ("terminal_changed_since", build_a2a_task_complete_registration()),
    ):
        retention = timedelta(days=registration.retention_days)
        assert before - retention <= window[key] <= after - retention


@pytest.mark.asyncio
async def test_boot_reconciliation_follows_a_changed_source_retention():
    """The window moves with the registration, so it cannot drift from it."""

    agent = _Agent()
    agent.signal_registry = SourceRegistry()
    agent.signal_registry.register(
        replace(build_a2a_task_submitted_registration(), retention_days=3)
    )
    agent.signal_registry.register(
        replace(build_a2a_task_complete_registration(), retention_days=40)
    )
    agent.task_manager.list_cognition_wake_candidates = AsyncMock(return_value=[])

    before = datetime.now(timezone.utc)
    await agent.reconcile_a2a_cognition_wakes()

    window = agent.task_manager.list_cognition_wake_candidates.await_args.kwargs
    assert before - timedelta(days=3, seconds=5) <= window["live_changed_since"]
    assert window["live_changed_since"] <= before - timedelta(days=3) + timedelta(
        seconds=5
    )
    assert before - timedelta(days=40, seconds=5) <= window[
        "terminal_changed_since"
    ] <= before - timedelta(days=40) + timedelta(seconds=5)


@pytest.mark.asyncio
async def test_boot_reconciliation_refuses_without_a_registered_a2a_source():
    agent = _Agent()
    agent.signal_registry = SourceRegistry()
    agent.signal_registry.register(build_a2a_task_complete_registration())
    agent.task_manager.list_cognition_wake_candidates = AsyncMock(return_value=[])

    with pytest.raises(RuntimeError, match="a2a.task_submitted"):
        await agent.reconcile_a2a_cognition_wakes()
    agent.task_manager.list_cognition_wake_candidates.assert_not_awaited()


@pytest.mark.asyncio
async def test_boot_reconciliation_fails_closed_when_wake_is_not_durable():
    """One unadmitted wake refuses boot, but only after every candidate ran."""

    agent = _Agent()
    agent.task_manager.list_cognition_wake_candidates = AsyncMock(
        return_value=[
            SimpleNamespace(
                task=_task("missing-outbox", state=TaskState.SUBMITTED),
                lifecycle_revision=0,
            ),
            SimpleNamespace(
                task=_task("behind-it", state=TaskState.COMPLETED),
                lifecycle_revision=4,
            ),
        ]
    )
    dispositions = iter(
        (
            DurableAdmissionDisposition.NOT_ADMITTED,
            DurableAdmissionDisposition.COMMITTED,
        )
    )
    agent.dispatcher.enqueue_durable_cognition = AsyncMock(
        side_effect=lambda *args, **kwargs: _durable_signal_handle(
            next(dispositions),
            status=Status.FAILED,
        )
    )

    with pytest.raises(RuntimeError, match="could not durably admit.*missing-outbox"):
        await agent.reconcile_a2a_cognition_wakes()
    assert [
        call.kwargs["source_event_id"]
        for call in agent.dispatcher.enqueue_durable_cognition.await_args_list
    ] == ["missing-outbox", "behind-it"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original,mints",
    [
        (None, True),
        (SimpleNamespace(is_terminal=True), True),
        (SimpleNamespace(is_terminal=False), False),
    ],
    ids=["no-original", "original-finished", "original-still-live"],
)
async def test_boot_reconciliation_mints_working_only_after_the_submission_ends(
    original, mints
):
    """A live submission wake resumes its own work; a second would duplicate it."""

    agent = _Agent()
    agent.task_manager.list_cognition_wake_candidates = AsyncMock(
        return_value=[
            SimpleNamespace(
                task=_task("mid-turn", state=TaskState.WORKING),
                lifecycle_revision=1,
            )
        ]
    )
    agent.dispatcher.get_durable_delivery_for_source_event = AsyncMock(
        return_value=original
    )
    agent.dispatcher.enqueue_durable_cognition = AsyncMock(
        side_effect=lambda *args, **kwargs: _durable_signal_handle(
            DurableAdmissionDisposition.COMMITTED
        )
    )

    await agent.reconcile_a2a_cognition_wakes()

    agent.dispatcher.get_durable_delivery_for_source_event.assert_awaited_once_with(
        consumer_id="core.a2a-task-submitted-cognition-v1",
        source="a2a.task_submitted",
        source_event_id="mid-turn",
    )
    minted = [
        call.kwargs["source_event_id"]
        for call in agent.dispatcher.enqueue_durable_cognition.await_args_list
    ]
    assert minted == (["mid-turn:working:1"] if mints else [])


@pytest.mark.asyncio
async def test_a2a_wake_retries_a_failed_durable_consumer_read(monkeypatch, caplog):
    """The read that picks the path retries like the handoff it gates."""

    monkeypatch.setattr(
        "kestrel_sovereign.agent.event_manager.A2A_WAKE_RETRY_INITIAL_SECONDS",
        0,
    )
    agent = _Agent()
    agent.dispatcher.has_durable_consumer = AsyncMock(
        side_effect=[RuntimeError("database is locked"), True]
    )
    agent.dispatcher.enqueue_durable_cognition = AsyncMock(
        side_effect=lambda *args, **kwargs: _durable_signal_handle(
            DurableAdmissionDisposition.COMMITTED
        )
    )

    with caplog.at_level(logging.WARNING):
        agent._on_task_submitted(_task(state=TaskState.SUBMITTED))
        await _drain(agent)

    assert agent.dispatcher.has_durable_consumer.await_count == 2
    agent.dispatcher.enqueue_durable_cognition.assert_awaited_once()
    assert "durable-consumer read failed; retrying" in caplog.text


@pytest.mark.asyncio
async def test_boot_reconciliation_leaves_a_terminally_failed_wake_failed(caplog):
    """A wake whose delivery already failed is not lost and cannot wedge boot."""

    agent = _Agent()
    agent.task_manager.list_cognition_wake_candidates = AsyncMock(
        return_value=[
            SimpleNamespace(
                task=_task("exhausted", state=TaskState.SUBMITTED),
                lifecycle_revision=0,
            ),
            SimpleNamespace(
                task=_task("behind-it", state=TaskState.COMPLETED),
                lifecycle_revision=4,
            ),
        ]
    )
    dispositions = iter(
        (
            DurableAdmissionDisposition.DELIVERY_FAILED,
            DurableAdmissionDisposition.COMMITTED,
        )
    )
    agent.dispatcher.enqueue_durable_cognition = AsyncMock(
        side_effect=lambda *args, **kwargs: _durable_signal_handle(
            next(dispositions),
            status=Status.FAILED,
        )
    )

    with caplog.at_level(logging.WARNING):
        await agent.reconcile_a2a_cognition_wakes()

    assert [
        call.kwargs["source_event_id"]
        for call in agent.dispatcher.enqueue_durable_cognition.await_args_list
    ] == ["exhausted", "behind-it"]
    assert "exhausted already failed terminally" in caplog.text


@pytest.mark.asyncio
async def test_completion_wake_does_not_clear_submission_self_decline():
    """Only the submission-wake owner may retire its cancellation exemption."""

    agent = _Agent()
    agent._a2a_self_declining_task_ids = {"shared-task"}
    agent.dispatcher.enqueue_durable_cognition = AsyncMock(
        return_value=_durable_signal_handle(DurableAdmissionDisposition.COMMITTED)
    )

    await agent._deliver_a2a_wake_until_durable(
        signal_factory=lambda: SimpleNamespace(payload={}),
        task_id="shared-task",
        consumer_id="core.a2a-task-complete-cognition-v1",
        label="a2a.task_complete[shared-task]",
        cancellation_aware=False,
    )

    assert agent._a2a_self_declining_task_ids == {"shared-task"}


@pytest.mark.asyncio
async def test_privacy_elided_a2a_replay_rehydrates_from_task_store():
    """A marker-only signal row regains content from the scoped task ledger."""

    agent = _Agent()
    task = _task("private-task", state=TaskState.WORKING)
    task.sessionId = "private-session"
    task.history = []
    agent.task_manager.get_task_for_recipient = AsyncMock(return_value=task)
    event = SimpleNamespace(
        source="a2a.task_submitted",
        source_event_id="private-task:working:7",
        dedupe_key=task.id,
        target_agent=agent.did,
    )
    dispatch_signal = SimpleNamespace(id="retry-signal", arrived_at=MagicMock())

    recovered = await agent.rehydrate_durable_cognition_signal(
        event,
        dispatch_signal=dispatch_signal,
    )

    assert recovered.id == "retry-signal"
    assert recovered.payload["task_id"] == task.id
    assert recovered.payload["session_id"] == "private-session"
    agent.task_manager.get_task_for_recipient.assert_awaited_once_with(
        task.id,
        agent.did,
    )


@pytest.mark.asyncio
async def test_working_a2a_task_remains_executable_after_crash():
    """SUBMITTED->WORKING is progress, not proof the cognition wake finished."""

    agent = _Agent()
    agent.task_manager.get_task_cancellation_snapshot = AsyncMock(
        return_value=SimpleNamespace(state="working", actor_agent_id=None)
    )
    signal = SimpleNamespace(
        source="a2a.task_submitted",
        payload={"task_id": "working-after-crash"},
    )

    assert await agent.validate_cognition_signal_execution(signal) is None


@pytest.mark.asyncio
async def test_a2a_persistence_retry_rebuilds_a_fresh_signal(monkeypatch):
    """A failed dispatcher attempt must not donate its mutations to retry."""

    monkeypatch.setattr(
        "kestrel_sovereign.agent.event_manager.A2A_WAKE_RETRY_INITIAL_SECONDS",
        0,
    )
    agent = _Agent()
    seen = []

    async def enqueue(signal, **_kwargs):
        seen.append(signal)
        if len(seen) == 1:
            signal.payload["failed_attempt_mutation"] = True
            return _durable_signal_handle(
                DurableAdmissionDisposition.NOT_ADMITTED,
                status=Status.FAILED,
            )
        assert "failed_attempt_mutation" not in signal.payload
        return _durable_signal_handle(DurableAdmissionDisposition.COMMITTED)

    agent.dispatcher.enqueue_durable_cognition = AsyncMock(side_effect=enqueue)

    agent._on_task_submitted(_task(state=TaskState.SUBMITTED))
    await _drain(agent)

    assert len(seen) == 2
    assert seen[0] is not seen[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fire,state",
    [
        (lambda a, t: a._on_background_task_complete(t), TaskState.COMPLETED),
        (lambda a, t: a._on_task_submitted(t), TaskState.SUBMITTED),
    ],
    ids=["task_complete", "task_submitted"],
)
async def test_a2a_wake_does_not_retry_a_terminally_failed_delivery(
    monkeypatch, caplog, fire, state
):
    """A retry loop ends on a condition retrying cannot change (#3163)."""

    monkeypatch.setattr(
        "kestrel_sovereign.agent.event_manager.A2A_WAKE_RETRY_INITIAL_SECONDS",
        0,
    )
    agent = _Agent()
    agent.dispatcher.enqueue_durable_cognition = AsyncMock(
        side_effect=lambda *a, **k: _durable_signal_handle(
            DurableAdmissionDisposition.DELIVERY_FAILED,
            status=Status.FAILED,
        )
    )

    with caplog.at_level(logging.WARNING):
        fire(agent, _task(state=state))
        await _drain(agent)

    agent.dispatcher.enqueue_durable_cognition.assert_awaited_once()
    assert "already failed terminally" in caplog.text


@pytest.mark.parametrize(
    "fire,prefix",
    [
        (lambda a, t: a._on_background_task_complete(t), "a2a_complete:"),
        (lambda a, t: a._on_task_submitted(t), "a2a_submitted:"),
    ],
    ids=["task_complete", "task_submitted"],
)
class TestDetachedDeliveryIsOwnedAndHarvested:
    @pytest.mark.asyncio
    async def test_dispatch_runs_in_a_tracked_task(self, fire, prefix):
        """The enqueue must not be fire-and-forget: an untracked coroutine
        is invisible to shutdown, which is how #2660 lost rows."""
        agent = _Agent()
        agent.dispatcher.enqueue_durable_cognition = AsyncMock(
            side_effect=lambda *a, **k: _signal_handle(Status.OK)
        )

        fire(agent, _task())

        assert [n for _t, n in agent.tracked if n.startswith(prefix)], (
            f"dispatch was not registered with the agent tracker ({prefix})"
        )
        await _drain(agent)
        agent.dispatcher.enqueue_durable_cognition.assert_awaited_once()
        call = agent.dispatcher.enqueue_durable_cognition.await_args
        assert call.kwargs["source_event_id"] == "task-1"
        assert call.kwargs["consumer_id"] in {
            "core.a2a-task-complete-cognition-v1",
            "core.a2a-task-submitted-cognition-v1",
        }

    @pytest.mark.asyncio
    async def test_terminal_result_is_harvested(self, fire, prefix):
        """The handle must actually be awaited to a terminal state — not
        merely accepted.

        Asserting ``handle.task.done()`` would be vacuous: the dispatch
        task resolves on its own whether or not anyone harvests it. The
        load-bearing observation is that ``wait()`` was *called*.
        """
        waited = []

        class _RecordingHandle:
            async def wait(self):
                waited.append(True)
                return SignalResult(
                    signal_id="sig-test",
                    status=Status.OK,
                    mode=SignalMode.COGNITION,
                    duration_ms=1,
                )

        agent = _Agent()
        agent.dispatcher.enqueue_durable_cognition = AsyncMock(
            side_effect=lambda *a, **k: _RecordingHandle()
        )

        fire(agent, _task())
        await _drain(agent)

        assert waited, (
            "handle.wait() was never called — the dispatch was only "
            "accepted, so a failed wake could never be observed"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [
        Status.FAILED,
        Status.DROPPED_RATE_LIMIT,
        Status.DROPPED_CYCLE,
        Status.DROPPED_VALIDATION,
    ])
    async def test_undelivered_wake_is_logged_not_silent(
        self, fire, prefix, status, caplog
    ):
        """A wake the dispatcher accepted and then failed/dropped must
        surface. Silence here is the exact defect #2532 was filed for."""
        agent = _Agent()
        agent.dispatcher.enqueue_durable_cognition = AsyncMock(
            side_effect=lambda *a, **k: _signal_handle(status, error="boom")
        )

        with caplog.at_level(logging.WARNING):
            fire(agent, _task())
            await _drain(agent)

        assert any(
            "never delivered" in r.getMessage() for r in caplog.records
        ), f"{status} was swallowed silently"

    @pytest.mark.asyncio
    async def test_coalesced_is_not_reported_as_failure(
        self, fire, prefix, caplog
    ):
        """COALESCED means an equivalent wake was already dispatched —
        the intended outcome for a detached callback, so it must not be
        logged as a failure. (It is NOT checkpoint-grade for the durable
        producers; that asymmetry is deliberate.)"""
        agent = _Agent()
        agent.dispatcher.enqueue_durable_cognition = AsyncMock(
            side_effect=lambda *a, **k: _signal_handle(Status.COALESCED)
        )

        with caplog.at_level(logging.WARNING):
            fire(agent, _task())
            await _drain(agent)

        assert not any("never delivered" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_enqueue_failure_retries_without_escaping_the_callback(
        self, fire, prefix, monkeypatch
    ):
        """These are sync callbacks on the task-persistence path; a
        dispatcher failure must never break task completion/creation or lose
        the already-committed wake."""
        monkeypatch.setattr(
            "kestrel_sovereign.agent.event_manager.A2A_WAKE_RETRY_INITIAL_SECONDS",
            0,
        )
        agent = _Agent()
        agent.dispatcher.enqueue_durable_cognition = AsyncMock(
            side_effect=[
                RuntimeError("dispatcher down"),
                _durable_signal_handle(DurableAdmissionDisposition.COMMITTED),
            ]
        )

        fire(agent, _task())
        await _drain(agent)  # harvest task must not raise
        assert agent.dispatcher.enqueue_durable_cognition.await_count == 2


# ---------------------------------------------------------------------------
# supervise_terminal_delivery — the shape producers use when they cannot await
# inline. Its cancellation path had no coverage, which is where codex found a
# P1 during the #2532 review.
# ---------------------------------------------------------------------------


class _OwnedTaskFeature:
    """Minimal stand-in for the Feature background-task ownership contract."""

    def __init__(self):
        self.tasks: list[asyncio.Task] = []

    def _track_owned_background_task(self, coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        return task


def _handle_that_never_settles() -> SignalHandle:
    """A dispatch that is still in flight — the state shutdown interrupts."""
    handle = MagicMock(spec=SignalHandle)
    handle.wait = AsyncMock(side_effect=asyncio.Event().wait)
    return handle


@pytest.mark.asyncio
async def test_supervisor_cancellation_restores_optimistically_retired_state():
    """Shutdown must not strand a durable row this producer already retired.

    ``await_terminal_delivery`` re-raises caller cancellation on the reasoning
    that a checkpoint which never advanced is already correct. That holds for
    a producer which retains by NOT advancing (the watchers) and is false for
    one which retires up front: A2A question completion claims its row
    ``RESOLVED``/``EXPIRED`` before dispatching. Cancelling the supervisor —
    feature shutdown, soft disable, boot rollback — also cancels the dispatch,
    so the wake never lands. Without the restore the row stays terminal
    forever and startup replay never resumes the asker.
    """
    from kestrel_sovereign.signals.delivery import (
        STATUS_SUPERVISOR_CANCELLED,
        supervise_terminal_delivery,
    )

    feature = _OwnedTaskFeature()
    restored: list = []
    settled: list = []

    async def _restore(outcome):
        restored.append(outcome)

    task = supervise_terminal_delivery(
        feature,
        _handle_that_never_settles(),
        label="a2a_question[q-1]",
        task_name="a2a_question_delivery:q-1",
        on_undelivered=_restore,
        on_settled=lambda: settled.append(True),
    )

    await asyncio.sleep(0)  # let the supervisor reach its await
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(restored) == 1, (
        "the durable row this producer retired before dispatch must be "
        "restored when shutdown cancels the wake"
    )
    assert restored[0].status == STATUS_SUPERVISOR_CANCELLED
    assert restored[0].delivered is False, (
        "a cancelled supervisor never observed a delivery"
    )
    assert settled == [True], "in-flight bookkeeping still runs on every path"


@pytest.mark.asyncio
async def test_supervisor_advances_the_checkpoint_only_on_terminal_ok():
    """The rule itself, through the supervisor path."""
    from kestrel_sovereign.signals.delivery import supervise_terminal_delivery

    for status, expect_advance in (
        (Status.OK, True),
        (Status.FAILED, False),
        (Status.COALESCED, False),
    ):
        feature = _OwnedTaskFeature()
        advanced: list = []
        retained: list = []
        handle = MagicMock(spec=SignalHandle)
        handle.wait = AsyncMock(
            return_value=SignalResult(
                signal_id="sig-supervised",
                status=status,
                mode=SignalMode.COGNITION,
                duration_ms=1,
            )
        )

        task = supervise_terminal_delivery(
            feature, handle,
            label="watch", task_name="watch:1",
            on_delivered=lambda: advanced.append(True) or asyncio.sleep(0),
            on_undelivered=lambda outcome: retained.append(outcome) or asyncio.sleep(0),
        )
        await task

        assert bool(advanced) is expect_advance, (
            f"{status} must {'advance' if expect_advance else 'NOT advance'} "
            f"the checkpoint"
        )
        assert bool(retained) is (not expect_advance)
