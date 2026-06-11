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
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.storage.async_pending_a2a_question_store import (
    PendingA2AQuestion,
)


def _make_feature_for_replay():
    """Build a PeersFeature wired enough to exercise
    ``post_all_features_loaded`` / replay / sweep without a real DB or
    httpx client. ``_supervise_a2a_question`` is replaced with an
    AsyncMock so we can assert it was scheduled without driving the
    SSE loop."""
    feature = PeersFeature.__new__(PeersFeature)
    feature._host_url = "http://host:8888"
    feature._api_key = ""
    feature._own_name = "Sender"

    agent = MagicMock()
    agent.did = "did:test:sender"
    agent.dispatcher = MagicMock()
    agent.dispatcher.enqueue_signal = AsyncMock()
    tracked: list = []

    def _track_bg(coro, *, name=""):
        tracked.append((coro, name))
        # Close coroutines so they don't leak; tests assert on `tracked`.
        coro.close()
        return MagicMock()
    agent._track_background_task = _track_bg
    feature.agent = agent
    return feature, agent, tracked


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
        assert not any("task-retry-answer" in n for n in spawn_names)

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
            None,
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
