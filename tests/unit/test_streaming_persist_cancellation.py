"""
Cancellation-safe persistence: when the streaming generator is
cancelled (client disconnect, browser nav, agent switch) AFTER the
last chunk is yielded but BEFORE the assistant row is inserted, the
insert must complete in the background — not be cancelled along with
the generator.

Without this, the user sees the response stream complete in their
chat pane but the next turn's history loader can't find it. Surfaced
by Meridian's "I don't see my own quantum response" transcript.

asyncio.shield(...) is the standard pattern. These tests pin the
behavior so a future refactor can't silently revert it.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.streaming import StreamingMixin


def _make_agent_with_persist(persist_impl):
    """Build a minimal mock agent that exposes the helper under test."""
    agent = MagicMock()
    agent.did = "test-did"
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.add_conversation = persist_impl
    agent.observability_store = MagicMock()
    agent.observability_store.log_metric = AsyncMock()
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    return agent


@pytest.mark.asyncio
async def test_persist_completes_when_outer_task_is_cancelled_mid_insert():
    """The classic Bug A scenario: client disconnects after last yield
    but the assistant row is still being written. Without shield(), the
    add_conversation coroutine would be cancelled too and the row would
    be lost. With shield(), the insert completes regardless."""
    persist_started = asyncio.Event()
    persist_completed = asyncio.Event()

    async def slow_persist(role, content, **kw):
        persist_started.set()
        # Hold long enough that the outer cancel definitely fires
        # while we're suspended here.
        await asyncio.sleep(0.05)
        persist_completed.set()

    agent = _make_agent_with_persist(slow_persist)

    async def outer():
        await agent._persist_assistant_turn_safely(
            "answer text", metadata=None, session_id="s1"
        )

    task = asyncio.create_task(outer())

    # Wait until the persist coroutine has started, then cancel the
    # outer task. shield() is supposed to let the inner coroutine
    # keep running.
    await persist_started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # The shielded inner coroutine should still complete.
    await asyncio.wait_for(persist_completed.wait(), timeout=0.5)
    assert persist_completed.is_set(), (
        "asyncio.shield must keep add_conversation alive past outer cancel"
    )


@pytest.mark.asyncio
async def test_persist_failure_logs_metric_and_swallows_exception():
    """If add_conversation raises (DB error, privacy violation, etc.),
    the helper must NOT propagate the exception — the response is
    already streamed to the client. It must log an
    ``assistant_turn_persist_failed`` metric so production can see
    this happening."""
    async def failing_persist(role, content, **kw):
        raise RuntimeError("DB unavailable")

    agent = _make_agent_with_persist(failing_persist)

    # Should NOT raise.
    await agent._persist_assistant_turn_safely(
        "answer", metadata={"x": 1}, session_id="s1"
    )

    agent.observability_store.log_metric.assert_awaited_once()
    args = agent.observability_store.log_metric.await_args
    assert args.kwargs["metric_name"] == "assistant_turn_persist_failed"
    assert args.kwargs["metric_value"] == 1.0
    assert args.kwargs["metadata"]["session_id"] == "s1"
    assert args.kwargs["metadata"]["error_type"] == "RuntimeError"
    assert "DB unavailable" in args.kwargs["metadata"]["error_msg"]


@pytest.mark.asyncio
async def test_persist_failure_with_broken_telemetry_does_not_raise():
    """Last-line-of-defense: if add_conversation fails AND
    log_metric also fails, the helper must still not raise. The error
    log line is the only signal we get."""
    async def failing_persist(role, content, **kw):
        raise ValueError("storage layer panic")

    agent = _make_agent_with_persist(failing_persist)
    agent.observability_store.log_metric = AsyncMock(
        side_effect=RuntimeError("metrics also down")
    )

    # MUST NOT raise either error.
    await agent._persist_assistant_turn_safely(
        "answer", metadata=None, session_id="s2"
    )


@pytest.mark.asyncio
async def test_persist_passes_metadata_and_session_id_through():
    """The helper must forward metadata and session_id to
    add_conversation untouched — no silent dropping."""
    captured = []

    async def capture(role, content, metadata=None, session_id=None):
        captured.append({
            "role": role, "content": content,
            "metadata": metadata, "session_id": session_id,
        })

    agent = _make_agent_with_persist(capture)

    await agent._persist_assistant_turn_safely(
        "answer", metadata={"tool_events": [{"name": "x"}]}, session_id="abc"
    )

    assert captured[0]["role"] == "assistant"
    assert captured[0]["content"] == "answer"
    assert captured[0]["metadata"] == {"tool_events": [{"name": "x"}]}
    assert captured[0]["session_id"] == "abc"


@pytest.mark.asyncio
async def test_outer_cancellation_propagates_after_shielded_completes():
    """If the outer task was cancelled while the shielded persist ran,
    the helper should NOT swallow the CancelledError — it must
    re-raise so the caller's cancellation semantics stay intact."""
    async def fast_persist(role, content, **kw):
        return None

    agent = _make_agent_with_persist(fast_persist)

    # Simulate the outer task cancelling itself between scheduling the
    # shielded persist and re-raising. asyncio.shield won't propagate
    # the inner cancel here because the inner coroutine completes
    # cleanly first; the helper's `except CancelledError: raise`
    # branch is what we want for the case where the outer task IS
    # cancelled mid-await. We simulate by running inside a cancelled
    # task.
    async def runner():
        # Cancel ourselves mid-helper. The shielded persist still
        # completes; CancelledError propagates after.
        task = asyncio.current_task()
        await asyncio.sleep(0)
        task.cancel()
        await agent._persist_assistant_turn_safely(
            "answer", metadata=None, session_id=None
        )

    task = asyncio.create_task(runner())
    with pytest.raises(asyncio.CancelledError):
        await task
