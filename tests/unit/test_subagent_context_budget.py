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

from kestrel_sovereign.features.base import (
    Feature,
    SubagentContextBudgetExceeded,
)


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


KNOWN_MODEL = "test-model-1m"
KNOWN_LIMIT = 1_000_000
SMALL_MODEL = "test-model-32k"
SMALL_LIMIT = 32_768


@pytest.fixture(autouse=True)
def known_limits(monkeypatch):
    """Pin the context limits under test.

    The unit suite isolates host runtime paths, so the discovered-limits file
    is not loaded and no real model resolves -- tests that lean on ambient
    discovery therefore read different numbers on a developer box than in CI,
    and can pass for reasons unrelated to what they name. These two models
    exist only here.
    """
    from kestrel_sovereign.agent.token_counter import TokenCounter

    real = TokenCounter.resolved_context_limit

    def fake(self):
        return {KNOWN_MODEL: KNOWN_LIMIT, SMALL_MODEL: SMALL_LIMIT}.get(
            self.model, real(self)
        )

    monkeypatch.setattr(TokenCounter, "resolved_context_limit", fake)


@pytest.fixture
def feature():
    return _StubFeature(agent=MagicMock())


def _messages(total_chars):
    return [{"role": "tool", "tool_call_id": "t", "content": "x" * total_chars}]


def test_a_small_conversation_is_not_refused(feature):
    """Control: the ceiling must not fire on ordinary work, or the refusal
    below proves nothing."""
    assert feature._subagent_context_overflow(_messages(1000), SMALL_MODEL) is None


def test_an_oversized_conversation_is_refused(feature):
    # claude-opus-5 is recorded at 1,000,000 tokens; 0.85 of that is the
    # ceiling. ~1.2M tokens of content clears it comfortably.
    reason = feature._subagent_context_overflow(_messages(5_000_000), SMALL_MODEL)
    assert reason is not None
    assert "context budget" in reason


def test_the_refusal_names_the_numbers_and_the_cause(feature):
    """An operator-facing message has to say what was exceeded and why, not
    just that something failed."""
    reason = feature._subagent_context_overflow(_messages(5_000_000), SMALL_MODEL)
    assert SMALL_MODEL in reason
    assert "tool result" in reason
    assert any(c.isdigit() for c in reason)


def test_budget_is_a_fraction_of_the_window_not_the_whole_thing(feature):
    """Filling the window exactly still leaves no room for the response.

    Compared against the limit resolved AT RUNTIME, not a hardcoded number:
    the unit suite isolates host runtime paths, so the discovered 1,000,000
    limit for this model is not loaded and a hardcoded bound is satisfied by
    any fraction, including 1.0.
    """
    limit = KNOWN_LIMIT
    budget = feature._subagent_context_budget(KNOWN_MODEL)
    assert budget is not None
    assert 0 < budget < limit, (
        f"budget {budget} leaves no headroom under a {limit}-token window"
    )
    # A fraction close enough to 1.0 leaves no usable headroom either.
    assert budget <= limit * 0.95


def test_an_unknown_model_does_not_enforce(feature):
    """An unrecognised model must not kill a working subagent.

    Asserted unconditionally. The previous version guarded the assertion on
    `budget is None` and was vacuous, because get_context_limit() never
    returns None -- it substitutes 32768 -- so an unknown model DID enforce,
    at a ceiling ~30x below a large window.
    """
    fake = "not-a-real-model-xyz"
    assert feature._subagent_context_budget(fake) is None
    assert feature._subagent_context_overflow(_messages(5_000_000), fake) is None


def test_no_model_does_not_enforce(feature):
    """Neither execute_as_subagent call site passes model_override, so the
    loop learns the model from the response. Until it has one, enforcing
    would be guessing."""
    assert feature._subagent_context_budget(None) is None
    assert feature._subagent_context_overflow(_messages(5_000_000), None) is None


def test_budget_is_derived_from_the_named_model(feature):
    """Kills the mutant that ignores the model and always resolves "auto".

    That mutant is the live bug this fix exists to avoid: "auto" resolves to
    the 32768 default, so a subagent on a 1,000,000-token window would be
    refused at ~28k.
    """
    big = feature._subagent_context_budget(KNOWN_MODEL)
    small = feature._subagent_context_budget(SMALL_MODEL)
    assert big is not None and small is not None
    assert big > small, (
        "budget does not vary with the model — a subagent on a large window "
        "would be refused at the small model's ceiling"
    )
    assert big < KNOWN_LIMIT


def test_measurement_failure_does_not_refuse(feature, monkeypatch):
    """A broken measurement must fail open. Refusing on our own inability to
    count would turn a metrology bug into an outage.

    Only the token count is broken, not the limit lookup: breaking the
    factory (or the limit) makes the BUDGET resolve to None first, so the
    check short-circuits and the measurement path it names is never reached
    -- the test would pass without exercising anything. The assertion below
    that the budget still resolves is what keeps that honest.
    """
    from kestrel_sovereign.agent.token_counter import TokenCounter

    def boom(self, text):
        raise RuntimeError("counter exploded")

    monkeypatch.setattr(TokenCounter, "count", boom)
    assert feature._subagent_context_budget(SMALL_MODEL) is not None, (
        "budget lookup must still work, or this test proves nothing"
    )
    assert feature._subagent_context_overflow(_messages(5_000_000), SMALL_MODEL) is None


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
    with pytest.raises(SubagentContextBudgetExceeded) as excinfo:
        await feature._handle_feature_tool_calls(
            _Resp(tool_calls=[_Call("big_tool")]),
            tools=[], system_prompt="sp", max_iterations=3,
            model_override=SMALL_MODEL,
            runtime_tools=[tool],
        )

    assert called["provider"] == 0, "loop sent a request it knew would be rejected"
    assert "context budget" in str(excinfo.value)


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
        model_override=SMALL_MODEL,
        runtime_tools=[tool],
    )

    assert called["provider"] == 1
    assert result == "done"


@pytest.mark.asyncio
async def test_loop_learns_the_model_from_the_response(feature):
    """Neither execute_as_subagent call site passes model_override, so if the
    loop only trusted that parameter the budget would never resolve and the
    gate would be dead in production -- which is exactly how it shipped the
    first time."""
    called = {"provider": 0}

    async def _never(*a, **k):
        called["provider"] += 1
        return _Resp(content="should not happen")

    async def _huge(*, tool_name, args, tools_by_name, **kw):
        return {"payload": "x" * 5_000_000}

    feature.agent.llm_service.generate_with_messages = _never
    feature._execute_subagent_tool = _huge

    tool = MagicMock(); tool.name = "big_tool"
    first = _Resp(tool_calls=[_Call("big_tool")])
    first.model = SMALL_MODEL          # the provider says what it served

    with pytest.raises(SubagentContextBudgetExceeded):
        await feature._handle_feature_tool_calls(
            first, tools=[], system_prompt="sp", max_iterations=3,
            model_override=None,       # as every real caller does
            runtime_tools=[tool],
        )
    assert called["provider"] == 0


def test_tool_schemas_count_against_the_budget(feature):
    """Tool definitions are resent with every request. A conversation that
    fits on its own can still overflow once the schemas are added, so leaving
    them out of the measurement lets the gate pass a doomed request."""
    small_messages = _messages(1000)
    assert feature._subagent_context_overflow(small_messages, SMALL_MODEL) is None

    fat_tools = [
        {"type": "function",
         "function": {"name": f"t{i}", "description": "d" * 20_000,
                      "parameters": {"type": "object", "properties": {}}}}
        for i in range(40)
    ]
    reason = feature._subagent_context_overflow(
        small_messages, SMALL_MODEL, fat_tools
    )
    assert reason is not None, "tool schemas were not counted"
