"""Startup-replay + hourly expiry sweep for in-flight
``send_a2a_question`` rows (#1444 step 6).

Pins:

  - Startup replay: each WAITING row triggers either a fresh supervisor
    (within-deadline) or a synthetic ``state='expired'`` signal
    (past-deadline). No row gets both — that would double-resume.
  - Unparseable deadline strings are treated as expired (safer than
    spawning a forever-running supervisor).
  - Hourly sweep: expired rows get the synthetic signal exactly once,
    and a transient store failure does NOT kill the loop (it must
    survive the agent's lifetime as the deadline backstop).
  - Already-terminal rows (raced by another resumption path) drop
    silently rather than firing a duplicate signal — the WAITING-only
    semantics of ``mark_expired`` carry this guarantee.
  - Terminal delivery, not enqueue acceptance, is what retires the durable
    WAITING row (#2532). Every dispatch here therefore hands back a REAL
    ``SignalHandle`` resolving to a REAL ``SignalResult`` — a truthy handle
    mock is always "successful" and would only prove the happy path can
    happen, never that failure is handled.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from kestrel_sdk.signals import SignalHandle, SignalMode, SignalResult, Status

from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.storage.async_pending_a2a_question_store import (
    PendingA2AQuestion,
)


def _signal_handle(
    status: Status = Status.OK, *, error: str | None = None,
) -> SignalHandle:
    """A real ``SignalHandle`` whose task resolves to a real ``SignalResult``.

    The dispatcher hands back the handle at *acceptance*; the terminal
    ``Status`` only arrives via ``await handle.wait()``. Tests must drive the
    real object so a non-``OK`` terminal state is actually exercised (#2532).
    """

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


def _cancelled_signal_handle() -> SignalHandle:
    """A handle whose dispatch task was cancelled out from under the caller."""

    async def _never() -> SignalResult:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    task = asyncio.ensure_future(_never())
    task.cancel()
    return SignalHandle(signal_id="sig-test", task=task)


def _make_feature_for_replay(*, run_tracked_tasks: bool = False):
    """Build a PeersFeature wired enough to exercise
    ``post_all_features_loaded`` / replay / sweep without a real DB or
    httpx client. ``_supervise_a2a_question`` is replaced with an
    AsyncMock so we can assert it was scheduled without driving the
    SSE loop.

    ``enqueue_signal`` returns a real ``SignalHandle`` that settles ``OK``, so
    the default posture is "the wake actually landed". Tests that need a
    different terminal state override ``side_effect`` with another
    :func:`_signal_handle`.

    ``run_tracked_tasks=True`` makes the tracker start real asyncio tasks
    (still recorded in ``tracked``) instead of closing the coroutine, for the
    boot-path tests that must drive the delivery supervisor to completion.
    """
    feature = PeersFeature.__new__(PeersFeature)
    feature._host_url = "http://host:8888"
    feature._transport_key = ""
    feature._own_name = "Sender"

    agent = MagicMock()
    agent.did = "did:test:sender"
    agent.dispatcher = MagicMock()

    async def _enqueue_ok(_signal):
        return _signal_handle(Status.OK)
    agent.dispatcher.enqueue_signal = AsyncMock(side_effect=_enqueue_ok)
    tracked: list = []

    def _track_bg(coro, *, name=""):
        if run_tracked_tasks:
            task = asyncio.ensure_future(coro)
            tracked.append((task, name))
            return task
        tracked.append((coro, name))
        # Close coroutines so they don't leak; tests assert on `tracked`.
        coro.close()
        return MagicMock()
    agent._track_background_task = _track_bg
    feature.agent = agent
    return feature, agent, tracked


async def _drain(tracked, prefix: str) -> None:
    """Await every started task whose name begins with ``prefix``."""
    for task, name in tracked:
        if name.startswith(prefix) and isinstance(task, asyncio.Task):
            await asyncio.wait_for(task, timeout=5)


def _started_tasks(tracked, prefix: str) -> list:
    return [
        obj for obj, name in tracked
        if name.startswith(prefix) and isinstance(obj, asyncio.Task)
    ]


def _row(
    task_id: str,
    *,
    deadline: datetime,
    recipient: str = "Meridian",
    status: str = "WAITING",
) -> PendingA2AQuestion:
    return PendingA2AQuestion(
        task_id=task_id,
        recipient=recipient,
        original_question=f"q-for-{task_id}",
        origin_turn_id=None,
        origin_session_id="sess-1",
        deadline=deadline.isoformat(),
        status=status,
        created_at=datetime.now(timezone.utc).isoformat(),
        resolved_at=None,
    )


# ---------------------------------------------------------------------------
# post_all_features_loaded — wiring + skip semantics
# ---------------------------------------------------------------------------

class TestPostAllFeaturesLoadedSkips:
    @pytest.mark.asyncio
    async def test_skips_when_pending_store_unwired(self):
        """In standalone (non-multi-agent) mode, ``pending_a2a_questions``
        won't be wired. Skip silently — not an error."""
        feature, agent, tracked = _make_feature_for_replay()
        agent.pending_a2a_questions = None
        await feature.post_all_features_loaded(agent)
        assert tracked == [], (
            "No background tasks must be spawned when the store is "
            "absent — the hourly sweep would AttributeError on every "
            "tick."
        )

    @pytest.mark.asyncio
    async def test_skips_when_no_host_url(self):
        """Agent is not in a multi_agent — there's nothing to
        subscribe to on a replay, and no peer to ask. Skip."""
        feature, agent, tracked = _make_feature_for_replay()
        feature._host_url = None
        agent.pending_a2a_questions = MagicMock()
        agent.pending_a2a_questions.list_waiting = AsyncMock(return_value=[])
        await feature.post_all_features_loaded(agent)
        assert tracked == []


# ---------------------------------------------------------------------------
# Startup replay — within-deadline → supervisor; past-deadline → expired
# ---------------------------------------------------------------------------

class TestStartupReplay:
    @pytest.mark.asyncio
    async def test_within_deadline_row_spawns_supervisor(self):
        feature, agent, tracked = _make_feature_for_replay()
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        row = _row("task-fresh", deadline=future)

        store = MagicMock()
        store.list_waiting = AsyncMock(return_value=[row])
        store.mark_expired = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store

        await feature.post_all_features_loaded(agent)

        store.mark_expired.assert_not_awaited()
        # 1 supervisor + 1 hourly sweep loop.
        spawn_names = [n for _, n in tracked]
        assert any(
            "a2a_question_supervisor:replay" in n and "task-fresh" in n
            for n in spawn_names
        ), (
            f"Expected a replay supervisor for task-fresh, got "
            f"{spawn_names}."
        )
        assert "a2a_question_expiry_sweep" in spawn_names

    @pytest.mark.asyncio
    async def test_replay_passes_persisted_stable_recipient_identity(self):
        """A restart must route by the dispatched peer, not an old display name."""
        feature, _agent, _tracked = _make_feature_for_replay()
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        row = PendingA2AQuestion(
            task_id="task-stable-recipient",
            recipient="Companion",
            recipient_agent_id="did:test:original-companion",
            original_question="q",
            origin_turn_id=None,
            origin_session_id="sess-1",
            deadline=future.isoformat(),
            status="WAITING",
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
        )
        store = MagicMock()
        store.list_waiting = AsyncMock(return_value=[row])
        feature._supervise_a2a_question = AsyncMock()

        await feature._replay_pending_a2a_questions(store)

        assert feature._supervise_a2a_question.call_args.kwargs[
            "recipient_agent_id"
        ] == "did:test:original-companion"

    @pytest.mark.asyncio
    async def test_retry_payload_row_refires_captured_answer(self):
        feature, agent, tracked = _make_feature_for_replay()
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        row = PendingA2AQuestion(
            task_id="task-retry-answer",
            recipient="Meridian",
            original_question="q",
            origin_turn_id=None,
            origin_session_id="sess-1",
            deadline=future.isoformat(),
            status="WAITING",
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            retry_state="completed",
            retry_reply_text="Captured reply",
        )

        store = MagicMock()
        store.list_waiting = AsyncMock(return_value=[row])
        store.mark_resolved = AsyncMock(return_value=True)
        store.mark_expired = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store

        await feature.post_all_features_loaded(agent)

        store.mark_resolved.assert_awaited_once_with("task-retry-answer")
        store.mark_expired.assert_not_awaited()
        agent.dispatcher.enqueue_signal.assert_awaited_once()
        sig = agent.dispatcher.enqueue_signal.await_args.args[0]
        assert sig.payload["state"] == "completed"
        assert sig.payload["reply_text"] == "Captured reply"
        spawn_names = [n for _, n in tracked]
        assert not any(
            "a2a_question_supervisor" in n and "task-retry-answer" in n
            for n in spawn_names
        ), (
            "A retry-payload row carries an already-observed terminal answer "
            f"— it must never get an SSE supervisor. Got {spawn_names}."
        )
        # It DOES get a delivery supervisor: boot replay cannot await the
        # resumed cognition turn inline, so the wait that decides whether the
        # WAITING row stays retired is handed to a feature-owned task (#2532).
        assert (
            "a2a_question_answered_delivery:task-retry-answer" in spawn_names
        ), spawn_names

    @pytest.mark.asyncio
    async def test_past_deadline_row_fires_expired_signal(self):
        feature, agent, tracked = _make_feature_for_replay()
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        row = _row("task-stale", deadline=past)

        store = MagicMock()
        store.list_waiting = AsyncMock(return_value=[row])
        store.mark_expired = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store

        await feature.post_all_features_loaded(agent)

        store.mark_expired.assert_awaited_once_with("task-stale")
        # NO replay supervisor for stale rows (just the sweep loop).
        replay_supervisors = [
            n for _, n in tracked
            if "a2a_question_supervisor:replay" in n
        ]
        assert replay_supervisors == [], (
            "Past-deadline rows must NOT spawn a supervisor — that's "
            "what the expired signal is for."
        )
        agent.dispatcher.enqueue_signal.assert_awaited_once()
        sig = agent.dispatcher.enqueue_signal.await_args.args[0]
        assert sig.payload["state"] == "expired"
        assert sig.payload["task_id"] == "task-stale"

    @pytest.mark.asyncio
    async def test_unparseable_deadline_treated_as_expired(self):
        """Unparseable deadline string is safer to expire than to
        spawn a supervisor that might run forever (no terminal cap)."""
        feature, agent, tracked = _make_feature_for_replay()
        bad_row = PendingA2AQuestion(
            task_id="task-bad",
            recipient="Meridian",
            original_question="x",
            origin_turn_id=None,
            origin_session_id=None,
            deadline="not-an-isoformat-string",
            status="WAITING",
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
        )
        store = MagicMock()
        store.list_waiting = AsyncMock(return_value=[bad_row])
        store.mark_expired = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store

        await feature.post_all_features_loaded(agent)

        store.mark_expired.assert_awaited_once_with("task-bad")
        agent.dispatcher.enqueue_signal.assert_awaited_once()
        assert (
            agent.dispatcher.enqueue_signal.await_args.args[0]
            .payload["state"] == "expired"
        )

    @pytest.mark.asyncio
    async def test_mixed_rows_route_independently(self):
        """A boot snapshot with both fresh and stale rows must route
        each correctly — no batch decision masks the mixed state."""
        feature, agent, tracked = _make_feature_for_replay()
        rows = [
            _row("fresh-1",
                 deadline=datetime.now(timezone.utc) + timedelta(minutes=5)),
            _row("stale-1",
                 deadline=datetime.now(timezone.utc) - timedelta(hours=1)),
            _row("fresh-2",
                 deadline=datetime.now(timezone.utc) + timedelta(minutes=15)),
        ]
        store = MagicMock()
        store.list_waiting = AsyncMock(return_value=rows)
        store.mark_expired = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store

        await feature.post_all_features_loaded(agent)

        # Two supervisor spawns, one expired signal, one sweep loop.
        replay_supervisors = [
            n for _, n in tracked
            if "a2a_question_supervisor:replay" in n
        ]
        assert sorted(replay_supervisors) == sorted([
            "a2a_question_supervisor:replay:Meridian:fresh-1",
            "a2a_question_supervisor:replay:Meridian:fresh-2",
        ])
        store.mark_expired.assert_awaited_once_with("stale-1")
        # Exactly one expired signal.
        assert agent.dispatcher.enqueue_signal.await_count == 1


# ---------------------------------------------------------------------------
# Hourly sweep — fires expired signal, survives transient failure
# ---------------------------------------------------------------------------

class TestHourlySweep:
    @pytest.mark.asyncio
    async def test_sweep_fires_expired_signal_for_each_past_row(
        self, monkeypatch,
    ):
        """Cut the interval to ~0 so the test doesn't sleep an hour.
        First tick returns 2 expired rows + a CancelledError to stop."""
        feature, agent, tracked = _make_feature_for_replay()
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        rows = [_row("e1", deadline=past), _row("e2", deadline=past)]

        store = MagicMock()
        store.mark_expired = AsyncMock(return_value=True)
        # First call returns the expired rows; second raises Cancelled
        # to break the loop deterministically.
        store.list_waiting_past_deadline = AsyncMock(
            side_effect=[rows, asyncio.CancelledError()],
        )
        agent.pending_a2a_questions = store
        feature.EXPIRY_SWEEP_INTERVAL_SECONDS = 0  # tick immediately

        with pytest.raises(asyncio.CancelledError):
            await feature._hourly_expiry_sweep_loop(store)

        assert store.mark_expired.await_count == 2
        assert agent.dispatcher.enqueue_signal.await_count == 2
        states = [
            agent.dispatcher.enqueue_signal.await_args_list[i].args[0]
            .payload["state"]
            for i in range(2)
        ]
        assert states == ["expired", "expired"]

    @pytest.mark.asyncio
    async def test_sweep_survives_transient_store_failure(self):
        """A failing ``list_waiting_past_deadline`` call must NOT kill
        the loop — it's the deadline backstop and runs as long as the
        agent. Logs + backoff + continues."""
        feature, agent, tracked = _make_feature_for_replay()
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        rows = [_row("e-after-fail", deadline=past)]

        # First tick: store explodes. Second tick: returns one expired
        # row. Third tick: CancelledError to stop the loop.
        store = MagicMock()
        store.mark_expired = AsyncMock(return_value=True)
        store.list_waiting_past_deadline = AsyncMock(side_effect=[
            RuntimeError("transient DB blip"),
            rows,
            asyncio.CancelledError(),
        ])
        agent.pending_a2a_questions = store
        feature.EXPIRY_SWEEP_INTERVAL_SECONDS = 0  # tick immediately

        # The 60s post-failure backoff would normally make this test
        # slow — short-circuit asyncio.sleep so we don't actually wait.
        async def _instant_sleep(_secs):
            return None
        import kestrel_sovereign.features.peers.feature as peers_mod
        # The supervisor module imports asyncio lazily inside the
        # function; patch the module attribute the loop references.
        original_sleep = peers_mod.__dict__.get("asyncio", asyncio).sleep
        try:
            asyncio.sleep_orig = original_sleep  # not strictly needed
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(asyncio, "sleep", _instant_sleep)
                with pytest.raises(asyncio.CancelledError):
                    await feature._hourly_expiry_sweep_loop(store)
        finally:
            pass

        # The sweep loop should have called the store 3 times: blow up,
        # recover with rows, then get cancelled.
        assert store.list_waiting_past_deadline.await_count == 3
        # The recover-tick must have processed the expired row.
        assert store.mark_expired.await_count == 1
        agent.dispatcher.enqueue_signal.assert_awaited()


# ---------------------------------------------------------------------------
# _handle_expired_row — idempotency
# ---------------------------------------------------------------------------

class TestHandleExpiredRow:
    @pytest.mark.asyncio
    async def test_already_terminal_row_does_not_double_fire(self):
        """If ``mark_expired`` returns False (someone else's resumption
        path got there first), the synthetic expired signal must NOT
        fire — that would double-resume the asking turn."""
        feature, agent, _ = _make_feature_for_replay()
        store = MagicMock()
        store.mark_expired = AsyncMock(return_value=False)
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        row = _row("already-resolved", deadline=past)

        await feature._handle_expired_row(store, row)

        store.mark_expired.assert_awaited_once_with("already-resolved")
        agent.dispatcher.enqueue_signal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retry_payload_past_deadline_preserves_captured_answer(self):
        feature, agent, _ = _make_feature_for_replay()
        store = MagicMock()
        store.mark_resolved = AsyncMock(return_value=True)
        store.mark_expired = AsyncMock(return_value=True)
        store.mark_waiting_for_retry = AsyncMock(return_value=True)
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        row = PendingA2AQuestion(
            task_id="retry-past-deadline",
            recipient="Meridian",
            original_question="q",
            origin_turn_id=None,
            origin_session_id="sess-1",
            deadline=past.isoformat(),
            status="WAITING",
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            retry_state="completed",
            retry_reply_text="Late but real",
        )

        await feature._handle_expired_row(store, row)

        store.mark_resolved.assert_awaited_once_with("retry-past-deadline")
        store.mark_expired.assert_not_awaited()
        agent.dispatcher.enqueue_signal.assert_awaited_once()
        sig = agent.dispatcher.enqueue_signal.await_args.args[0]
        assert sig.payload["state"] == "completed"
        assert sig.payload["reply_text"] == "Late but real"

    @pytest.mark.asyncio
    async def test_expired_retry_payload_uses_expired_terminal_transition(self):
        feature, agent, _ = _make_feature_for_replay()
        store = MagicMock()
        store.mark_resolved = AsyncMock(return_value=True)
        store.mark_expired = AsyncMock(return_value=True)
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        row = PendingA2AQuestion(
            task_id="retry-expired",
            recipient="Meridian",
            original_question="q",
            origin_turn_id=None,
            origin_session_id="sess-1",
            deadline=past.isoformat(),
            status="WAITING",
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            retry_state="expired",
            retry_reply_text="",
        )

        await feature._handle_expired_row(store, row)

        store.mark_expired.assert_awaited_once_with("retry-expired")
        store.mark_resolved.assert_not_awaited()
        agent.dispatcher.enqueue_signal.assert_awaited_once()
        sig = agent.dispatcher.enqueue_signal.await_args.args[0]
        assert sig.payload["state"] == "expired"
        assert sig.payload["reply_text"] == ""

    @pytest.mark.asyncio
    async def test_retry_payload_enqueue_failure_restores_payload_again(self):
        feature, agent, _ = _make_feature_for_replay()
        agent.dispatcher.enqueue_signal.side_effect = RuntimeError("still down")
        store = MagicMock()
        store.mark_resolved = AsyncMock(return_value=True)
        store.mark_waiting_for_retry = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        row = PendingA2AQuestion(
            task_id="retry-still-down",
            recipient="Meridian",
            original_question="q",
            origin_turn_id=None,
            origin_session_id="sess-1",
            deadline=past.isoformat(),
            status="WAITING",
            created_at=datetime.now(timezone.utc).isoformat(),
            resolved_at=None,
            retry_state="completed",
            retry_reply_text="Keep carrying this",
        )

        await feature._handle_expired_row(store, row)

        store.mark_waiting_for_retry.assert_awaited_once_with(
            "retry-still-down",
            state="completed",
            reply_text="Keep carrying this",
        )

    @pytest.mark.asyncio
    async def test_near_term_retry_refires_restored_payload(self, monkeypatch):
        feature, agent, _ = _make_feature_for_replay()
        store = MagicMock()
        store.mark_resolved = AsyncMock(side_effect=[True, True])
        store.mark_waiting_for_retry = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store
        agent.dispatcher.enqueue_signal.side_effect = [
            RuntimeError("dispatcher blip"),
            _signal_handle(Status.OK),
        ]

        async def _instant_sleep(_secs):
            return None
        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

        await feature._retry_restored_question_answered_signal(
            task_id="retry-soon",
            recipient="Meridian",
            original_question="q",
            sess_id="sess-1",
            state="completed",
            reply_text="Recovered answer",
            causation_chain=None,
        )

        assert store.mark_resolved.await_count == 2
        store.mark_waiting_for_retry.assert_awaited_once_with(
            "retry-soon",
            state="completed",
            reply_text="Recovered answer",
        )
        assert agent.dispatcher.enqueue_signal.await_count == 2
        assert (
            agent.dispatcher.enqueue_signal.await_args_list[1]
            .args[0].payload["reply_text"] == "Recovered answer"
        )

    @pytest.mark.asyncio
    async def test_near_term_expired_retry_uses_mark_expired(self, monkeypatch):
        feature, agent, _ = _make_feature_for_replay()
        store = MagicMock()
        store.mark_expired = AsyncMock(return_value=True)
        store.mark_resolved = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store

        async def _instant_sleep(_secs):
            return None
        monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

        await feature._retry_restored_question_answered_signal(
            task_id="retry-expired-soon",
            recipient="Meridian",
            original_question="q",
            sess_id="sess-1",
            state="expired",
            reply_text="",
            causation_chain=None,
        )

        store.mark_expired.assert_awaited_once_with("retry-expired-soon")
        store.mark_resolved.assert_not_awaited()
        agent.dispatcher.enqueue_signal.assert_awaited_once()
        assert (
            agent.dispatcher.enqueue_signal.await_args.args[0]
            .payload["state"] == "expired"
        )


# ---------------------------------------------------------------------------
# Terminal delivery gates the durable WAITING row (#2532)
#
# The caller retires the row BEFORE firing (claim-first, so a racing sweep
# can't double-fire). That makes terminal delivery the only thing entitled to
# keep it retired: a wake the dispatcher accepted and then failed or dropped
# leaves the asker blocked forever with a terminal row and nothing to replay.
# ---------------------------------------------------------------------------

class TestTerminalDeliveryGatesWaitingRow:
    async def _expire_with(self, handle_factory):
        """Run one sweep-path expiry whose wake settles per the factory."""
        feature, agent, tracked = _make_feature_for_replay()

        async def _enqueue(_sig):
            return handle_factory()
        agent.dispatcher.enqueue_signal = AsyncMock(side_effect=_enqueue)
        store = MagicMock()
        store.mark_expired = AsyncMock(return_value=True)
        store.mark_waiting_for_retry = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store
        past = datetime.now(timezone.utc) - timedelta(hours=2)

        await feature._handle_expired_row(store, _row("gate-1", deadline=past))
        return feature, agent, store, tracked

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [
        Status.FAILED,
        Status.DROPPED_RATE_LIMIT,
        Status.DROPPED_QUIET_HOURS,
        Status.DROPPED_CYCLE,
        Status.DROPPED_VALIDATION,
        Status.COALESCED,
    ])
    async def test_non_ok_terminal_state_restores_waiting_row(self, status):
        _f, _a, store, _t = await self._expire_with(
            lambda: _signal_handle(status)
        )
        store.mark_waiting_for_retry.assert_awaited_once_with(
            "gate-1", state="expired", reply_text="",
        )

    @pytest.mark.asyncio
    async def test_ok_terminal_state_keeps_row_retired(self):
        _f, _a, store, _t = await self._expire_with(
            lambda: _signal_handle(Status.OK)
        )
        store.mark_waiting_for_retry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancelled_dispatch_restores_waiting_row(self):
        """The dispatch task was reaped. The asker was never woken, so the
        row must go back to WAITING — this caller is healthy."""
        _f, _a, store, _t = await self._expire_with(_cancelled_signal_handle)
        store.mark_waiting_for_retry.assert_awaited_once_with(
            "gate-1", state="expired", reply_text="",
        )

    @pytest.mark.asyncio
    async def test_unobservable_handle_restores_waiting_row(self):
        """No awaitable ``wait`` means delivery can never be observed.
        Unobservable is undelivered, never success."""
        _f, _a, store, _t = await self._expire_with(lambda: object())
        store.mark_waiting_for_retry.assert_awaited_once_with(
            "gate-1", state="expired", reply_text="",
        )

    @pytest.mark.asyncio
    async def test_wait_error_restores_waiting_row(self):
        class _ExplodingHandle:
            async def wait(self):
                raise RuntimeError("dispatcher internals blew up")

        _f, _a, store, _t = await self._expire_with(_ExplodingHandle)
        store.mark_waiting_for_retry.assert_awaited_once_with(
            "gate-1", state="expired", reply_text="",
        )

    @pytest.mark.asyncio
    async def test_caller_cancelled_mid_delivery_still_restores_row(self):
        """If this caller is torn down while waiting, its own ``if not fired``
        restore never runs — CancelledError propagates past it. The obligation
        is therefore the fire path's: restore under a shield, then re-raise."""
        feature, agent, _tracked = _make_feature_for_replay()

        started = asyncio.Event()

        async def _never_settles():
            started.set()
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

        async def _enqueue(_sig):
            return SignalHandle(
                signal_id="sig-test",
                task=asyncio.ensure_future(_never_settles()),
            )
        agent.dispatcher.enqueue_signal = AsyncMock(side_effect=_enqueue)

        store = MagicMock()
        store.mark_expired = AsyncMock(return_value=True)
        store.mark_waiting_for_retry = AsyncMock(return_value=True)
        agent.pending_a2a_questions = store
        past = datetime.now(timezone.utc) - timedelta(hours=2)

        task = asyncio.ensure_future(
            feature._handle_expired_row(store, _row("gate-cancel", deadline=past))
        )
        await started.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        store.mark_waiting_for_retry.assert_awaited_once_with(
            "gate-cancel", state="expired", reply_text="",
        )


# ---------------------------------------------------------------------------
# Boot-path ownership boundary (#2532)
#
# Startup replay runs inline inside feature init: the cognition turn it is
# dispatching cannot start until boot finishes, so awaiting terminal delivery
# there would deadlock. The wait is handed to a feature-owned supervisor that
# owns the restore instead — and that supervisor must actually run.
# ---------------------------------------------------------------------------

class TestBootReplayDeliverySupervisor:
    async def _replay_with(self, handle_factory):
        feature, agent, tracked = _make_feature_for_replay(
            run_tracked_tasks=True,
        )

        async def _enqueue(_sig):
            return handle_factory()
        agent.dispatcher.enqueue_signal = AsyncMock(side_effect=_enqueue)
        store = MagicMock()
        store.mark_expired = AsyncMock(return_value=True)
        store.mark_waiting_for_retry = AsyncMock(return_value=True)
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        store.list_waiting = AsyncMock(
            return_value=[_row("boot-1", deadline=past)]
        )
        agent.pending_a2a_questions = store

        await feature._replay_pending_a2a_questions(store)
        await _drain(tracked, "a2a_question_answered_delivery:")
        return feature, agent, store, tracked

    @pytest.mark.asyncio
    async def test_boot_replay_does_not_block_on_delivery(self):
        """The replay call itself must return while the wake is still
        unsettled — blocking here would deadlock boot against the very agent
        that has not finished initializing."""
        feature, agent, tracked = _make_feature_for_replay(
            run_tracked_tasks=True,
        )
        gate = asyncio.Event()

        async def _slow():
            await gate.wait()
            return SignalResult(
                signal_id="sig-test",
                status=Status.OK,
                mode=SignalMode.COGNITION,
                duration_ms=1,
            )

        async def _enqueue(_sig):
            return SignalHandle(
                signal_id="sig-test", task=asyncio.ensure_future(_slow()),
            )
        agent.dispatcher.enqueue_signal = AsyncMock(side_effect=_enqueue)

        store = MagicMock()
        store.mark_expired = AsyncMock(return_value=True)
        store.mark_waiting_for_retry = AsyncMock(return_value=True)
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        store.list_waiting = AsyncMock(
            return_value=[_row("boot-nonblock", deadline=past)]
        )
        agent.pending_a2a_questions = store

        # Returns even though nothing has settled.
        await asyncio.wait_for(
            feature._replay_pending_a2a_questions(store), timeout=5,
        )
        supervisors = _started_tasks(tracked, "a2a_question_answered_delivery:")
        assert len(supervisors) == 1
        assert not supervisors[0].done()

        gate.set()
        await _drain(tracked, "a2a_question_answered_delivery:")
        store.mark_waiting_for_retry.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_supervisor_restores_row_on_failed_delivery(self):
        _f, _a, store, _t = await self._replay_with(
            lambda: _signal_handle(Status.FAILED, error="turn blew up")
        )
        store.mark_waiting_for_retry.assert_awaited_once_with(
            "boot-1", state="expired", reply_text="",
        )

    @pytest.mark.asyncio
    async def test_supervisor_restores_row_on_dropped_delivery(self):
        _f, _a, store, _t = await self._replay_with(
            lambda: _signal_handle(Status.DROPPED_CYCLE)
        )
        store.mark_waiting_for_retry.assert_awaited_once_with(
            "boot-1", state="expired", reply_text="",
        )

    @pytest.mark.asyncio
    async def test_supervisor_keeps_row_retired_on_ok_delivery(self):
        _f, _a, store, _t = await self._replay_with(
            lambda: _signal_handle(Status.OK)
        )
        store.mark_waiting_for_retry.assert_not_awaited()


# ---------------------------------------------------------------------------
# #2532 round-2: restoring the row on teardown must NOT also start a retry.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_teardown_restores_the_row_without_starting_a_retry(monkeypatch):
    """Restore always. Retry only when the feature is still alive.

    ``supervise_terminal_delivery`` calls ``on_undelivered`` on supervisor
    cancellation so an optimistically-retired row goes back. But cancellation
    means the feature is tearing down, and ``_cancel_owned_background_tasks``
    has ALREADY captured its task list — so a retry started from here escapes
    teardown entirely and can emit an A2A wake after Peers is disabled.

    The restored WAITING row is the retry: the next boot's replay picks it up.
    """
    from kestrel_sovereign.signals import delivery as delivery_mod
    from kestrel_sovereign.signals.delivery import (
        STATUS_SUPERVISOR_CANCELLED,
        DeliveryOutcome,
    )

    feat = PeersFeature.__new__(PeersFeature)
    captured: dict = {}
    restored: list = []

    def _capture_supervisor(feature, handle, **kwargs):
        captured["on_undelivered"] = kwargs["on_undelivered"]
        return MagicMock()

    async def _fake_restore(task_id, **kwargs):
        restored.append(kwargs["schedule_retry"])

    # The consumer imports this inside the method, so patch it at the source.
    monkeypatch.setattr(
        delivery_mod, "supervise_terminal_delivery", _capture_supervisor
    )
    monkeypatch.setattr(
        feat, "_restore_pending_question_waiting", _fake_restore, raising=False
    )
    monkeypatch.setattr(
        feat, "_build_question_answered_signal",
        lambda **kw: MagicMock(), raising=False,
    )
    feat.agent = MagicMock()
    feat.agent.dispatcher = MagicMock()
    feat.agent.dispatcher.enqueue_signal = MagicMock(return_value=MagicMock())

    await feat._fire_question_answered_signal(
        task_id="q-1", recipient="did:peer:x", original_question="?",
        sess_id="s", state="completed", reply_text="hi",
        causation_chain=None, await_delivery=False, schedule_retry=True,
    )

    on_undelivered = captured.get("on_undelivered")
    assert on_undelivered is not None, "supervisor was never armed"

    # A genuine delivery failure while the feature is alive → retry is fine.
    await on_undelivered(DeliveryOutcome(status="failed", delivered=False))
    # Teardown → restore the row, start nothing.
    await on_undelivered(
        DeliveryOutcome(status=STATUS_SUPERVISOR_CANCELLED, delivered=False)
    )

    assert restored == [True, False], (
        "a live-feature failure may schedule a retry; a cancelled supervisor "
        f"must restore only. got {restored}"
    )
