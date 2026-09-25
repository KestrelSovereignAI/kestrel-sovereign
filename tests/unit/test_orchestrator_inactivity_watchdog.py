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
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.agent.streaming import _parse_stream_sentinels
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall


class _VirtualClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose ``time()`` only moves when a test advances it, so
    ``asyncio.timeout`` deadlines are measured against virtual seconds."""

    def __init__(self) -> None:
        super().__init__()
        self._now = 0.0

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


_BOUND_METHODS = (
    "_handle_orchestrator_response_streaming",
    "_execute_tool_batch_at_stop_boundary",
    "_execute_tool_with_hooks", "_execute_tool_batch",
    "_partition_tool_calls", "_dispatch_tool_call",
    "_dispatch_feature_tool", "_dispatch_direct_tool",
    "_get_denied_tools", "_handle_feature_error",
    "_prune_orchestrator_messages", "_build_all_tools",
    "_build_feature_tools", "_visible_features_by_tool_name",
    "_visible_known_tool_names", "_known_tool_names",
    "_registered_tool_names", "_registered_features_by_tool_name",
    "_feature_supports_subagent_dispatch", "_hidden_context_features",
    "_hidden_context_tools", "_feature_hidden_from_context",
    "_direct_tool_hidden_from_context",
)


def _agent_with_followup(followup) -> MagicMock:
    """A KestrelAgent whose orchestrator methods are real and whose one tool
    succeeds, with ``followup`` as the post-tool ``stream_with_tool_detection``."""
    from kestrel_sovereign.hooks import HooksManager
    from kestrel_sovereign.kestrel_agent import KestrelAgent

    feature = MagicMock()
    feature.tool_name = "test_tool"
    feature.name = "test_feature"
    feature.execute_as_subagent = AsyncMock(return_value={"success": True})
    feature.to_orchestrator_tool.return_value = {
        "type": "function",
        "function": {"name": "test_tool", "description": "", "parameters": {}},
    }
    agent = MagicMock()
    agent.did = "did:test"
    agent.features = {"test_feature": feature}
    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="e1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()
    agent._direct_tools = {}
    agent._tool_to_feature = {}
    agent.hooks_manager = HooksManager()
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._explored_features = {}
    agent._direct_tool_defs = []
    agent._register_explored_feature_tools = MagicMock()
    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = followup
    for name in _BOUND_METHODS:
        setattr(agent, name, getattr(KestrelAgent, name).__get__(agent))
    agent._build_tool_calls_msg = KestrelAgent._build_tool_calls_msg
    return agent


def _run_turn(loop: _VirtualClockLoop, agent: MagicMock) -> List[Any]:
    async def drive() -> List[Any]:
        first = LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="test_tool", arguments={})],
        )
        return [chunk async for chunk in agent._handle_orchestrator_response_streaming(
            response=first, feature_tools=[], system_prompt="sys",
            force_local_only=False, effective_model="m", user_message="hi",
        )]

    return loop.run_until_complete(drive())


def _timeout_markers(chunks: List[Any]) -> List[dict]:
    text = "".join(c for c in chunks if isinstance(c, str))
    _clean, parts, _ = _parse_stream_sentinels(text)
    return [p for p in parts if p["phase"] == "error" and "timeout" in (p.get("detail") or "")]


def test_a_stream_that_keeps_progressing_past_the_watchdog_is_not_cut():
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
        chunks = _run_turn(loop, _agent_with_followup(long_followup))
    finally:
        loop.close()

    assert loop.time() > 2 * ORCHESTRATOR_TURN_TIMEOUT_SECS
    assert _timeout_markers(chunks) == []
    text = "".join(c for c in chunks if isinstance(c, str))
    assert "".join(words) in text


def test_a_stream_that_goes_silent_is_still_cut():
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
        chunks = _run_turn(loop, _agent_with_followup(stalling_followup))
    finally:
        loop.close()

    assert len(_timeout_markers(chunks)) == 1
    assert "this never arrives" not in "".join(c for c in chunks if isinstance(c, str))
    # Tripped by silence measured from the last item, not from the start:
    # within one 10s step of the fake stream (the step that crosses the
    # deadline may complete before the cancellation lands).
    silence = loop.time() - silent_from[0]
    assert ORCHESTRATOR_TURN_TIMEOUT_SECS <= silence <= ORCHESTRATOR_TURN_TIMEOUT_SECS + 20


def test_a_stream_that_never_starts_is_still_cut():
    """No item at all — the hung-upstream case the watchdog exists for."""
    loop = _VirtualClockLoop()

    async def hung_followup(**_kw):
        for _ in range(30):
            loop.advance(10)
            await asyncio.sleep(0)
        yield "this never arrives"

    try:
        chunks = _run_turn(loop, _agent_with_followup(hung_followup))
    finally:
        loop.close()

    assert len(_timeout_markers(chunks)) == 1
