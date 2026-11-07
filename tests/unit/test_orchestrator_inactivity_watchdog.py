"""#3300: the orchestrator's per-call watchdog measures inactivity, not
total elapsed time.

The watchdog exists to turn a hung upstream into a visible timeout marker
instead of silent dead air. It used to bound the WHOLE follow-up generation
at ``ORCHESTRATOR_TURN_TIMEOUT_SECS`` (180s) — harmless while every Claude
response was capped at 4,096 tokens, but once the output ceiling is the
model's own (up to 128,000 tokens), a response that is still streaming would
be cut off mid-answer. It now re-arms on every streamed item: a progressing
stream is never killed, a silent one still is.

Time is virtual — an event loop whose clock the fake stream advances — so
these tests cover 400 "seconds" of streaming without sleeping.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple

from kestrel_sovereign.agent.streaming import _parse_stream_sentinels
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from tests.utils.virtual_clock import VirtualClockLoop as _VirtualClockLoop
from tests.utils.native_tool_effects import drive_native


async def _agent_with_followup(create, followup):
    """Create the real partial Agent on the same loop as native SQL and input."""
    from kestrel_sovereign.features.base import Feature

    class TestTool(Feature):
        @property
        def tool_description(self):
            return "One owned successful tool before the observed generation"

        async def initialize(self):
            pass

        async def execute_as_subagent(self, **kwargs):
            return {"success": True}

    agent = await create(did="did:test:watchdog")
    feature = TestTool(agent)
    agent.features[feature.name] = feature
    agent.llm_service.stream_with_tool_detection = followup
    return agent


def _run_turn(loop: _VirtualClockLoop, followup, *, tmp_path, monkeypatch) -> Tuple[List[Any], Optional[BaseException]]:
    """Observe the actual follow-up and retire its native fixture on this loop."""
    from kestrel_sovereign.agent.orchestrator_engine import LLMCallInactivityTimeout
    from kestrel_sovereign.testing.fixtures import native_tool_effect_agent

    chunks: List[Any] = []

    async def run():
        lifetime = native_tool_effect_agent.__wrapped__(tmp_path, monkeypatch)
        try:
            create = await anext(lifetime)
            agent = await _agent_with_followup(create, followup)

            async def drive():
                first = LLMResponse(content="", tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})])
                async for chunk in agent._handle_orchestrator_response_streaming(
                    response=first, feature_tools=[], system_prompt="sys",
                    force_local_only=False, effective_model="m", user_message="hi",
                ):
                    chunks.append(chunk)

            await drive_native(agent, drive)
        finally:
            await lifetime.aclose()

    try:
        loop.run_until_complete(run())
    except LLMCallInactivityTimeout as exc:
        return chunks, exc
    return chunks, None


def _timeout_markers(chunks: List[Any]) -> List[dict]:
    text = "".join(c for c in chunks if isinstance(c, str))
    _clean, parts, _ = _parse_stream_sentinels(text)
    return [p for p in parts if p["phase"] == "error" and "timeout" in (p.get("detail") or "")]


def test_a_stream_that_keeps_progressing_past_the_watchdog_is_not_cut(tmp_path, monkeypatch):
    """400 virtual seconds of output, a chunk every 10 — more than twice the
    180s watchdog — completes with every chunk and no timeout marker."""
    from kestrel_sovereign.agent.orchestrator_engine import (
        ORCHESTRATOR_TURN_TIMEOUT_SECS,
    )

    loop = _VirtualClockLoop()
    words = [f"w{i} " for i in range(40)]

    async def long_followup(**_kw):
        for word in words:
            loop.advance(10)
            await asyncio.sleep(0)
            yield word
        yield LLMResponse(content="".join(words), tool_calls=[])

    try:
        chunks, failure = _run_turn(loop, long_followup, tmp_path=tmp_path, monkeypatch=monkeypatch)
    finally:
        loop.close()

    assert loop.time() > 2 * ORCHESTRATOR_TURN_TIMEOUT_SECS
    assert failure is None
    assert _timeout_markers(chunks) == []
    text = "".join(c for c in chunks if isinstance(c, str))
    assert "".join(words) in text


def test_a_stream_that_goes_silent_is_still_cut(tmp_path, monkeypatch):
    """One chunk, then 300 virtual seconds with nothing: the watchdog trips
    180 seconds after the last item and the turn shows a timeout marker."""
    from kestrel_sovereign.agent.orchestrator_engine import (
        ORCHESTRATOR_TURN_TIMEOUT_SECS,
    )

    loop = _VirtualClockLoop()
    silent_from: List[float] = []

    async def stalling_followup(**_kw):
        loop.advance(50)
        yield "partial "
        silent_from.append(loop.time())
        for _ in range(30):
            loop.advance(10)
            await asyncio.sleep(0)
        yield "this never arrives"

    try:
        chunks, failure = _run_turn(loop, stalling_followup, tmp_path=tmp_path, monkeypatch=monkeypatch)
    finally:
        loop.close()

    assert failure is not None
    assert len(_timeout_markers(chunks)) == 1
    assert "this never arrives" not in "".join(c for c in chunks if isinstance(c, str))
    # Tripped by silence measured from the last item, not from the start:
    # within one 10s step of the fake stream (the step that crosses the
    # deadline may complete before the cancellation lands).
    silence = loop.time() - silent_from[0]
    assert ORCHESTRATOR_TURN_TIMEOUT_SECS <= silence <= ORCHESTRATOR_TURN_TIMEOUT_SECS + 20


def test_a_stream_that_never_starts_is_still_cut(tmp_path, monkeypatch):
    """No item at all — the hung-upstream case the watchdog exists for."""
    loop = _VirtualClockLoop()

    async def hung_followup(**_kw):
        for _ in range(30):
            loop.advance(10)
            await asyncio.sleep(0)
        yield "this never arrives"

    try:
        chunks, failure = _run_turn(loop, hung_followup, tmp_path=tmp_path, monkeypatch=monkeypatch)
    finally:
        loop.close()

    assert failure is not None
    assert len(_timeout_markers(chunks)) == 1
