"""Unit tests for the sender-side subscription supervisor (#1444 step 3+5+7).

Pins:

  - Happy path: a terminal SSE ``status`` frame triggers
    ``store.mark_resolved`` + ``dispatcher.enqueue_signal`` exactly once,
    with the reply text extracted from the canonical A2A
    ``status.message.parts[].text`` shape.
  - Flattened-shape parsing: kestrel's own ``GET /tasks/{id}`` returns
    ``{"status": "completed", "message": "Rome"}`` (string status, top-
    level message). The SSE producer emits the canonical shape, but the
    parser handles both — same dual-shape support the legacy polling
    path carried (#1366 P1).
  - 404 hard-cut: a recipient that lacks ``/subscribe`` (legacy build)
    must NOT burn the whole deadline reconnecting. Supervisor enqueues
    one ``state='failed'`` signal with a clear "upgrade them" message
    and exits.
  - Dedup: if the pending row is already terminal (``mark_resolved``
    returns False, racing the startup-replay sweep or hourly expiry),
    the supervisor drops its own signal rather than double-firing.
  - SSE parser ignores ``keepalive`` / ``ping`` frames, ignores
    pre-terminal status updates, and survives malformed JSON.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from kestrel_sovereign.features.peers.feature import PeersFeature


# ---------------------------------------------------------------------------
# Fakes — we deliberately avoid touching a real DB or a real httpx server
# here. The integration test in tests/peers/ covers the wire end-to-end;
# this file only pins per-method behavior.
# ---------------------------------------------------------------------------

class _FakeStreamResponse:
    """Stand-in for an httpx streaming response. ``aiter_lines`` replays
    a pre-recorded list of SSE-shaped lines so we can drive the
    supervisor deterministically."""

    def __init__(self, lines: List[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient. ``stream`` returns the next
    queued response. Use a list so a single test can simulate a
    reconnect (first call → drop, second call → terminal)."""

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method: str, url: str, headers: Optional[dict] = None):
        self.calls.append((method, url))
        if not self._responses:
            raise httpx.RequestError("no more responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _make_feature(
    *,
    sse_responses: List[Any],
    mark_resolved_returns: bool = True,
    monkeypatch=None,
) -> tuple[PeersFeature, MagicMock, AsyncMock, _FakeAsyncClient]:
    """Build a PeersFeature with a stub agent + fake httpx client.

    Returns ``(feature, agent_mock, dispatcher_enqueue_mock,
    fake_client)`` so individual tests can assert on the signal fired
    and the URL hit.
    """
    feature = PeersFeature.__new__(PeersFeature)
    feature._host_url = "http://host:8888"
    feature._api_key = ""
    feature._own_name = "Sender"

    fake_client = _FakeAsyncClient(sse_responses)

    agent = MagicMock()
    agent.did = "did:test:sender"
    agent.pending_a2a_questions = MagicMock()
    agent.pending_a2a_questions.mark_resolved = AsyncMock(
        return_value=mark_resolved_returns,
    )
    dispatcher_enqueue = AsyncMock()
    agent.dispatcher = MagicMock()
    agent.dispatcher.enqueue_signal = dispatcher_enqueue
    feature.agent = agent

    if monkeypatch is not None:
        monkeypatch.setattr(
            "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
            lambda *a, **kw: fake_client,
        )

    return feature, agent, dispatcher_enqueue, fake_client


# ---------------------------------------------------------------------------
# SSE parser — _iter_sse_events + _parse_sse_status_data
# ---------------------------------------------------------------------------

class TestSSEParsing:
    @pytest.mark.asyncio
    async def test_iter_sse_events_groups_lines_by_blank_separator(self):
        feature = PeersFeature.__new__(PeersFeature)
        resp = _FakeStreamResponse([
            "event: status",
            'data: {"a": 1}',
            "",
            "event: keepalive",
            "data: ",
            "",
            ": comment-heartbeat",
            "event: status",
            'data: {"a": 2}',
            "",
        ])
        events = []
        async for ev in feature._iter_sse_events(resp):
            events.append(ev)
        assert events == [
            {"event": "status", "data": '{"a": 1}'},
            {"event": "keepalive", "data": ""},
            {"event": "status", "data": '{"a": 2}'},
        ], (
            "SSE parser must group lines by blank-line separator, "
            "strip the ``event:`` / ``data:`` prefixes, and drop "
            "comment lines silently."
        )

    def test_parse_status_data_canonical_shape(self):
        feature = PeersFeature.__new__(PeersFeature)
        result = feature._parse_sse_status_data(json.dumps({
            "id": "t-1",
            "status": {
                "state": "completed",
                "message": {
                    "parts": [{"type": "text", "text": "Rome"}],
                },
            },
            "final": True,
        }))
        assert result == ("completed", "Rome")

    def test_parse_status_data_flattened_kestrel_shape(self):
        """Some kestrel endpoints flatten the envelope —
        ``{"status": "completed", "message": "Rome"}``. The parser
        must handle this for backwards-compat with non-spec
        receivers."""
        feature = PeersFeature.__new__(PeersFeature)
        result = feature._parse_sse_status_data(json.dumps({
            "id": "t-1",
            "status": "completed",
            "message": "Rome",
        }))
        assert result == ("completed", "Rome")

    def test_parse_status_data_returns_none_on_malformed_json(self):
        feature = PeersFeature.__new__(PeersFeature)
        assert feature._parse_sse_status_data("not json") is None
        assert feature._parse_sse_status_data("") is None
        assert feature._parse_sse_status_data("null") is None

    def test_parse_status_data_pre_terminal_state(self):
        feature = PeersFeature.__new__(PeersFeature)
        result = feature._parse_sse_status_data(json.dumps({
            "id": "t-1",
            "status": {"state": "working"},
        }))
        # Pre-terminal frames still return a parsed tuple; the supervisor
        # decides what to do with them. ``working`` is not in
        # ``terminal_states`` so the supervisor keeps reading.
        assert result == ("working", "")


# ---------------------------------------------------------------------------
# Supervisor — happy path, 404 hard-cut, dedup
# ---------------------------------------------------------------------------

class TestSupervisorHappyPath:
    @pytest.mark.asyncio
    async def test_terminal_status_fires_signal_with_extracted_reply(
        self, monkeypatch,
    ):
        terminal_frame = json.dumps({
            "id": "t-1",
            "status": {
                "state": "completed",
                "message": {
                    "parts": [{"type": "text", "text": "Rome is the capital of Italy."}],
                },
            },
            "final": True,
        })
        sse_response = _FakeStreamResponse([
            f"event: status",
            f"data: {terminal_frame}",
            "",
        ])
        feature, agent, enqueue, fake_client = _make_feature(
            sse_responses=[sse_response], monkeypatch=monkeypatch,
        )

        await feature._supervise_a2a_question(
            task_id="t-1",
            recipient="Meridian",
            original_question="What is the capital of Italy?",
            sess_id="sess-1",
            deadline_utc=datetime.now(timezone.utc) + timedelta(minutes=5),
            causation_chain=None,
        )

        agent.pending_a2a_questions.mark_resolved.assert_awaited_once_with("t-1")
        enqueue.assert_awaited_once()
        # The signal carries the reply text inline.
        sent_signal = enqueue.await_args.args[0]
        assert sent_signal.source == "a2a.question_answered"
        assert sent_signal.payload["task_id"] == "t-1"
        assert sent_signal.payload["state"] == "completed"
        assert sent_signal.payload["reply_text"] == "Rome is the capital of Italy."
        assert sent_signal.payload["recipient"] == "Meridian"
        assert sent_signal.payload["original_question"] == (
            "What is the capital of Italy?"
        )
        assert sent_signal.target_agent == "did:test:sender"

    @pytest.mark.asyncio
    async def test_pre_terminal_then_terminal_only_fires_once(
        self, monkeypatch,
    ):
        """The receiver may emit several ``status`` frames before the
        terminal one (e.g. ``submitted`` → ``working`` → ``completed``).
        Only the terminal frame should trigger the signal fire."""
        pre_terminal = json.dumps({
            "id": "t-2",
            "status": {"state": "working"},
        })
        terminal = json.dumps({
            "id": "t-2",
            "status": {
                "state": "completed",
                "message": {
                    "parts": [{"type": "text", "text": "42"}],
                },
            },
            "final": True,
        })
        sse_response = _FakeStreamResponse([
            "event: status",
            f"data: {pre_terminal}",
            "",
            "event: keepalive",
            "data: ",
            "",
            "event: status",
            f"data: {terminal}",
            "",
        ])
        feature, agent, enqueue, _ = _make_feature(
            sse_responses=[sse_response], monkeypatch=monkeypatch,
        )

        await feature._supervise_a2a_question(
            task_id="t-2",
            recipient="Meridian",
            original_question="?",
            sess_id="s-2",
            deadline_utc=datetime.now(timezone.utc) + timedelta(minutes=5),
            causation_chain=None,
        )

        agent.pending_a2a_questions.mark_resolved.assert_awaited_once()
        enqueue.assert_awaited_once()
        assert enqueue.await_args.args[0].payload["reply_text"] == "42"


class TestSupervisor404HardCut:
    @pytest.mark.asyncio
    async def test_404_subscribe_short_circuits_with_failed_signal(
        self, monkeypatch,
    ):
        """Recipient missing ``/subscribe`` means a legacy build that
        cannot satisfy the fire-and-resume contract. Supervisor must
        NOT burn the whole deadline reconnecting; it fires a single
        ``state='failed'`` signal with a clear "upgrade them" message
        and exits."""
        not_found = _FakeStreamResponse([], status_code=404)
        feature, agent, enqueue, fake_client = _make_feature(
            sse_responses=[not_found], monkeypatch=monkeypatch,
        )

        await feature._supervise_a2a_question(
            task_id="t-404",
            recipient="LegacyAgent",
            original_question="?",
            sess_id="s-404",
            deadline_utc=datetime.now(timezone.utc) + timedelta(minutes=5),
            causation_chain=None,
        )

        # Single connect attempt — no reconnect storm on 404.
        assert len(fake_client.calls) == 1
        enqueue.assert_awaited_once()
        sig = enqueue.await_args.args[0]
        assert sig.payload["state"] == "failed"
        assert "subscribe" in sig.payload["reply_text"].lower(), (
            "Failed-state reply must explain that the recipient lacks "
            "/subscribe so the resumed turn can diagnose."
        )
        assert "1444" in sig.payload["reply_text"], (
            "Failed-state reply must cite #1444 so the operator can "
            "find the upgrade ticket."
        )


class TestSupervisorDedupSignal:
    @pytest.mark.asyncio
    async def test_already_terminal_pending_row_drops_signal(
        self, monkeypatch,
    ):
        """If the startup-replay sweep beat the supervisor to the same
        terminal frame, ``mark_resolved`` returns False. The supervisor
        must NOT enqueue a duplicate signal — that's the whole point
        of the WAITING-only transition contract."""
        terminal = json.dumps({
            "id": "t-3",
            "status": {
                "state": "completed",
                "message": {"parts": [{"type": "text", "text": "x"}]},
            },
            "final": True,
        })
        sse_response = _FakeStreamResponse([
            "event: status",
            f"data: {terminal}",
            "",
        ])
        feature, agent, enqueue, _ = _make_feature(
            sse_responses=[sse_response],
            mark_resolved_returns=False,  # someone got there first
            monkeypatch=monkeypatch,
        )

        await feature._supervise_a2a_question(
            task_id="t-3",
            recipient="Meridian",
            original_question="?",
            sess_id="s-3",
            deadline_utc=datetime.now(timezone.utc) + timedelta(minutes=5),
            causation_chain=None,
        )

        agent.pending_a2a_questions.mark_resolved.assert_awaited_once()
        enqueue.assert_not_awaited(), (
            "Supervisor must drop its signal when mark_resolved "
            "returns False — duplicate signal fires defeat the dedupe "
            "contract."
        )
