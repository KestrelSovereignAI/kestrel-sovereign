"""Detached signal dispatches are owned and harvested (#2532, AC4).

The event-manager A2A callbacks are *intentional detached dispatch*: the
TaskStore row is already persisted before the callback runs, so nothing
durable advances on delivery and retrying is not the callback's job. That
makes them the opposite classification from the watcher/question sites,
which gate a durable checkpoint on terminal delivery.

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
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sdk.signals import SignalHandle, SignalMode, SignalResult, Status

from kestrel_sovereign.a2a.types import TaskState
from kestrel_sovereign.agent.event_manager import EventManagerMixin


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
        self.dispatcher = MagicMock()
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
        agent.dispatcher.enqueue_signal = AsyncMock(
            side_effect=lambda *a, **k: _signal_handle(Status.OK)
        )

        fire(agent, _task())

        assert [n for _t, n in agent.tracked if n.startswith(prefix)], (
            f"dispatch was not registered with the agent tracker ({prefix})"
        )
        await _drain(agent)
        agent.dispatcher.enqueue_signal.assert_awaited_once()

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
        agent.dispatcher.enqueue_signal = AsyncMock(
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
        agent.dispatcher.enqueue_signal = AsyncMock(
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
        agent.dispatcher.enqueue_signal = AsyncMock(
            side_effect=lambda *a, **k: _signal_handle(Status.COALESCED)
        )

        with caplog.at_level(logging.WARNING):
            fire(agent, _task())
            await _drain(agent)

        assert not any("never delivered" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_enqueue_failure_does_not_escape_the_callback(
        self, fire, prefix
    ):
        """These are sync callbacks on the task-persistence path; a
        dispatcher failure must never break task completion/creation."""
        agent = _Agent()
        agent.dispatcher.enqueue_signal = AsyncMock(
            side_effect=RuntimeError("dispatcher down")
        )

        fire(agent, _task())
        await _drain(agent)  # harvest task must not raise
