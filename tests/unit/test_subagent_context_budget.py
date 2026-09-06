"""The subagent tool-call loop must not send a request it knows will be rejected.

The loop appends a full tool result each iteration (up to
KESTREL_MAX_TOOL_ITERATIONS, default 50) and resends the whole array. With no
ceiling it grew until the provider refused: Emma, 2026-08-24/25/26,
`prompt is too long: 1,104,046 tokens > 1,000,000 maximum`.

It REFUSES rather than trims. Trimming is model-visible pruning, and
salvage.py forbids that without a durable artifact or lossless pointer --
which tool results do not have: they are never written to
conversation_history, and _log_tool_dispatch keeps only result_status and
result_size_bytes. Silently dropping them would trade a loud failure for a
quiet loss.
"""

import pytest
from unittest.mock import MagicMock

from kestrel_sovereign.features.base import Feature


class _StubFeature(Feature):
    def __init__(self, agent=None):
        self.agent = agent
        self.disabled_skills = set()

    @property
    def name(self):
        return "StubFeature"

    @property
    def tool_description(self) -> str:
        return "stub"

    async def initialize(self) -> bool:
        return True

    def get_tools(self):
        return []


@pytest.fixture
def feature():
    return _StubFeature(agent=MagicMock())


def _messages(total_chars):
    return [{"role": "tool", "tool_call_id": "t", "content": "x" * total_chars}]


def test_a_small_conversation_is_not_refused(feature):
    """Control: the ceiling must not fire on ordinary work, or the refusal
    below proves nothing."""
    assert feature._subagent_context_overflow(_messages(1000), "claude-opus-5") is None


def test_an_oversized_conversation_is_refused(feature):
    # claude-opus-5 is recorded at 1,000,000 tokens; 0.85 of that is the
    # ceiling. ~1.2M tokens of content clears it comfortably.
    reason = feature._subagent_context_overflow(_messages(5_000_000), "claude-opus-5")
    assert reason is not None
    assert "context budget" in reason


def test_the_refusal_names_the_numbers_and_the_cause(feature):
    """An operator-facing message has to say what was exceeded and why, not
    just that something failed."""
    reason = feature._subagent_context_overflow(_messages(5_000_000), "claude-opus-5")
    assert "claude-opus-5" in reason
    assert "tool result" in reason
    assert any(c.isdigit() for c in reason)


def test_budget_is_a_fraction_of_the_window_not_the_whole_thing(feature):
    """Filling the window exactly still leaves no room for the response.

    Compared against the limit resolved AT RUNTIME, not a hardcoded number:
    the unit suite isolates host runtime paths, so the discovered 1,000,000
    limit for this model is not loaded and a hardcoded bound is satisfied by
    any fraction, including 1.0.
    """
    from kestrel_sovereign.agent.token_counter import get_token_counter

    limit = get_token_counter("claude-opus-5").get_context_limit()
    budget = feature._subagent_context_budget("claude-opus-5")
    assert budget is not None
    assert 0 < budget < limit, (
        f"budget {budget} leaves no headroom under a {limit}-token window"
    )


def test_an_unknown_model_does_not_enforce(feature):
    """An unrecognised model must not kill a working subagent; the provider's
    own limit still backstops it."""
    fake = "not-a-real-model-xyz"
    # Either no limit is known (None) or a default applies; in the former case
    # enforcement is skipped entirely.
    budget = feature._subagent_context_budget(fake)
    if budget is None:
        assert feature._subagent_context_overflow(_messages(5_000_000), fake) is None


def test_measurement_failure_does_not_refuse(feature, monkeypatch):
    """A broken measurement must fail open. Refusing on our own inability to
    count would turn a metrology bug into an outage.

    Only ``count_messages`` is broken, not ``get_token_counter``: breaking the
    factory makes the BUDGET lookup fail first and return None, so the loop
    short-circuits and the measurement failure path is never reached — the
    test would pass without ever exercising what it names.
    """
    from kestrel_sovereign.agent.token_counter import TokenCounter

    def boom(self, messages):
        raise RuntimeError("counter exploded")

    monkeypatch.setattr(TokenCounter, "count_messages", boom)
    assert feature._subagent_context_budget("claude-opus-5") is not None, (
        "budget lookup must still work, or this test proves nothing"
    )
    assert feature._subagent_context_overflow(_messages(5_000_000), "claude-opus-5") is None


# ---------------------------------------------------------------------------
# The wiring, not just the helper
# ---------------------------------------------------------------------------

class _Call:
    def __init__(self, name, id="c1", arguments=None):
        self.name = name
        self.id = id
        self.arguments = arguments or {}


class _Resp:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


@pytest.mark.asyncio
async def test_loop_refuses_before_calling_the_provider(feature, monkeypatch):
    """A helper nothing calls is not a fix: assert the loop consults it and
    stops BEFORE generate_with_messages."""
    called = {"provider": 0}

    async def _never(*a, **k):
        called["provider"] += 1
        return _Resp(content="should not happen")

    async def _huge(*, tool_name, args, tools_by_name, **kw):
        return {"payload": "x" * 5_000_000}

    feature.agent.llm_service.generate_with_messages = _never
    feature._execute_subagent_tool = _huge

    tool = MagicMock(); tool.name = "big_tool"
    result = await feature._handle_feature_tool_calls(
        _Resp(tool_calls=[_Call("big_tool")]),
        tools=[], system_prompt="sp", max_iterations=3,
        model_override="claude-opus-5",
        runtime_tools=[tool],
    )

    assert called["provider"] == 0, "loop sent a request it knew would be rejected"
    assert "context budget" in result


@pytest.mark.asyncio
async def test_loop_still_continues_when_it_fits(feature):
    """Control: proves the refusal above is caused by size, not by the
    harness simply never reaching the provider."""
    called = {"provider": 0}

    async def _ok(*a, **k):
        called["provider"] += 1
        return _Resp(content="done")

    async def _small(*, tool_name, args, tools_by_name, **kw):
        return {"payload": "x" * 100}

    feature.agent.llm_service.generate_with_messages = _ok
    feature._execute_subagent_tool = _small

    tool = MagicMock(); tool.name = "small_tool"
    result = await feature._handle_feature_tool_calls(
        _Resp(tool_calls=[_Call("small_tool")]),
        tools=[], system_prompt="sp", max_iterations=3,
        model_override="claude-opus-5",
        runtime_tools=[tool],
    )

    assert called["provider"] == 1
    assert result == "done"
