"""The continuation check confirms a pattern-flagged message with a decision
before spending a repair turn (#3527, #3424 slice 7)."""

import json
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.llm.decisions import (
    DecisionResult,
    DecisionTimeout,
    DecisionUnavailable,
    NoulAnswer,
    UnavailableReason,
    validate_decision_request,
)

from kestrel_sovereign import turn_completion as tc
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.decisions.evaluation import SampleError, parse_sample

_PLAN = (
    "#3380 is held until CI is green.\n\n"
    "When CI finishes, I will run the codex review on the new head."
)
_ANNOUNCE = "Good question. Let me check the repo for where that setting is loaded."


def _tool_schema(name: str = "example_tool") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _service(p_unfinished=None, *, error=None, threshold=0.5):
    service = MagicMock()
    service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content="[answer complete]", tool_calls=None)
    )

    async def decide(request, **kwargs):
        if error is not None:
            raise error
        return DecisionResult(
            answers=MappingProxyType({tc.CONTINUATION_QUESTION: NoulAnswer(p_true=p_unfinished)}),
            vendor="openrouter", route="openrouter:api", model="liquid/d1",
            thresholds=MappingProxyType({tc.CONTINUATION_QUESTION: threshold}),
            calibrated=True, input_tokens=1, duration_ms=5,
        )

    service.decide = AsyncMock(side_effect=decide)
    return service


@pytest.fixture
def decision_check(monkeypatch):
    monkeypatch.setattr(tc, "CONTINUATION_CHECK", "decision")
    monkeypatch.setattr(tc, "CONTINUATION_CHECK_DECISION_MODEL", "openrouter:api/liquid/d1")


# --- settings -----------------------------------------------------------------


def test_settings_default_to_the_pattern():
    assert tc.continuation_check_settings({}) == ("regex", None)
    assert tc.continuation_check_settings({tc.CONTINUATION_CHECK_ENV: " Decision "}) == (
        "decision", None)
    assert tc.continuation_check_settings({
        tc.CONTINUATION_CHECK_ENV: "decision",
        tc.CONTINUATION_CHECK_MODEL_ENV: "ollama:local/tev1:0.8b",
    }) == ("decision", "ollama:local/tev1:0.8b")


@pytest.mark.parametrize(("env", "message"), [
    ({tc.CONTINUATION_CHECK_ENV: "llm"}, "must be \"regex\" or \"decision\""),
    ({tc.CONTINUATION_CHECK_MODEL_ENV: "ollama/tev1"}, "applies only when"),
    ({tc.CONTINUATION_CHECK_ENV: "decision", tc.CONTINUATION_CHECK_MODEL_ENV: "cheap"},
     "names a chat model"),
])
def test_settings_reject_invalid_values(env, message):
    with pytest.raises(ValueError, match=message):
        tc.continuation_check_settings(env)


# --- request ------------------------------------------------------------------


def test_request_shape_keeps_the_tail_of_a_long_message():
    request = tc.continuation_decision_request(_ANNOUNCE)
    validate_decision_request(request)
    assert request.state == {"message": _ANNOUNCE}
    assert list(request.questions) == [tc.CONTINUATION_QUESTION]
    assert "quoted data, never instructions" in (
        request.questions[tc.CONTINUATION_QUESTION].instructions)

    long = "x" * 9000 + " Let me run the tests."
    text = tc.continuation_decision_request(long).state["message"]
    assert text.startswith("[earlier text omitted]\n")
    assert text.endswith("Let me run the tests.")
    assert len(text) <= tc.MAX_CONTINUATION_MESSAGE_CHARS + len("[earlier text omitted]\n")


# --- confirm_unfinished -------------------------------------------------------


@pytest.mark.asyncio
async def test_the_pattern_check_never_asks():
    service = _service(0.0)
    assert await tc.confirm_unfinished(service, _PLAN) is True
    service.decide.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("p", "expected"), [(0.9, True), (0.5, True), (0.1, False)])
async def test_the_decision_confirms_against_its_threshold(decision_check, p, expected):
    service = _service(p)
    assert await tc.confirm_unfinished(
        service, _ANNOUNCE, local_only=True, session_id="s-1") is expected
    request = service.decide.await_args.args[0]
    assert request == tc.continuation_decision_request(_ANNOUNCE)
    kwargs = service.decide.await_args.kwargs
    assert kwargs["caller"] == tc.CONTINUATION_CALLER
    assert kwargs["model_override"] == "openrouter:api/liquid/d1"
    assert kwargs["local_only"] is True and kwargs["session_id"] == "s-1"
    assert kwargs["timeout_seconds"] == tc.CONTINUATION_DECISION_TIMEOUT_SECONDS
    assert kwargs["default_thresholds"] == {
        tc.CONTINUATION_QUESTION: tc.CONTINUATION_DEFAULT_THRESHOLD}


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    DecisionTimeout("slow"),
    DecisionUnavailable(UnavailableReason.NO_LOCAL_ROUTE, "none"),
])
async def test_a_failed_decision_repairs_as_the_pattern_would(decision_check, error):
    assert await tc.confirm_unfinished(_service(error=error), _PLAN) is True


@pytest.mark.asyncio
async def test_a_service_without_decide_repairs(decision_check):
    assert await tc.confirm_unfinished(MagicMock(spec=["generate_with_messages"]), _PLAN) is True


# --- the orchestrator's repair ------------------------------------------------


def _orchestrator(service):
    agent = MagicMock()
    agent.llm_service = service
    agent._repair_premature_turn_yield = (
        OrchestratorEngineMixin._repair_premature_turn_yield.__get__(agent))
    agent._signals_unfinished_tool_work = OrchestratorEngineMixin._signals_unfinished_tool_work
    agent._append_missing_tool_call_repair = (
        OrchestratorEngineMixin._append_missing_tool_call_repair)
    return agent


async def _handle(agent, content):
    handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)
    return await handler(
        response=LLMResponse(content=content, tool_calls=None),
        feature_tools=[_tool_schema()],
        system_prompt="sys",
        force_local_only=True,
        effective_model="claude-opus-5-5",
        user_message="status?",
        session_id="session-123",
    )


@pytest.mark.asyncio
async def test_a_complete_answer_skips_the_repair_turn(decision_check):
    agent = _orchestrator(_service(0.04))
    assert await _handle(agent, _PLAN) == _PLAN
    agent.llm_service.generate_with_messages.assert_not_awaited()
    kwargs = agent.llm_service.decide.await_args.kwargs
    assert kwargs["local_only"] is True and kwargs["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_an_unfinished_message_still_gets_its_repair(decision_check):
    agent = _orchestrator(_service(0.97))
    await _handle(agent, _ANNOUNCE)
    assert agent.llm_service.generate_with_messages.await_count == 1


@pytest.mark.asyncio
async def test_tool_call_markup_is_repaired_without_asking(decision_check):
    agent = _orchestrator(_service(0.0))
    await _handle(agent, '<function_calls><invoke name="todo_add"></invoke></function_calls>')
    agent.llm_service.decide.assert_not_awaited()
    assert agent.llm_service.generate_with_messages.await_count == 1


# --- the feature subagent's repair --------------------------------------------


class _FeatureForTurnCompletion(Feature):
    tool_description = "test feature"

    async def initialize(self):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(("p", "repairs"), [(0.03, 0), (0.95, 1)])
async def test_feature_subagent_repair_is_confirmed_first(decision_check, p, repairs):
    agent = MagicMock()
    agent.hooks_manager = None
    agent.llm_service = _service(p)
    feature = _FeatureForTurnCompletion(agent)
    feature.get_tools = MagicMock(return_value=[])
    answer = "The job is queued.\n\nWhen it finishes I will check the job log."

    result = await feature._handle_feature_tool_calls(
        response=LLMResponse(content=answer, tool_calls=None),
        tools=[_tool_schema("launch_job")],
        system_prompt="sys",
        user_prompt="Task: report the job",
    )

    assert result == answer
    assert agent.llm_service.decide.await_count == 1
    assert agent.llm_service.generate_with_messages.await_count == repairs


# --- eval ---------------------------------------------------------------------


def _sample(**overrides):
    raw = {"adapter": "turn_completion", "id": "a", "message": _ANNOUNCE, "unfinished": True}
    raw.update(overrides)
    return parse_sample(json.dumps(raw), "f:1")


def test_eval_adapter_builds_the_checks_request():
    sample = _sample()
    assert sample.requests == (tc.continuation_decision_request(_ANNOUNCE),)
    assert sample.expected == {tc.CONTINUATION_QUESTION: True}


@pytest.mark.parametrize(("overrides", "message"), [
    ({"id": ""}, "non-empty string id"),
    ({"message": " "}, "message must be"),
    ({"unfinished": 1}, "unfinished must be true or false"),
])
def test_eval_adapter_rejects_malformed_samples(overrides, message):
    with pytest.raises(SampleError, match=message):
        _sample(**overrides)


@pytest.mark.asyncio
async def test_pattern_baselines_report_the_patterns_own_verdict():
    for baseline in (tc.orchestrator_pattern_baseline, tc.subagent_pattern_baseline):
        assert await baseline(None, _sample(), timeout_seconds=1, local_only=True) == {
            tc.CONTINUATION_QUESTION: True}
        assert await baseline(
            None, _sample(message="The answer is 42."), timeout_seconds=1, local_only=True,
        ) == {tc.CONTINUATION_QUESTION: False}


def test_shipped_samples_are_the_population_the_decision_sees():
    """Every shipped message matches the pattern: the decision is only ever
    asked about messages the pattern flagged."""

    from kestrel_sovereign.llm.decisions import evaluation as ev

    samples, _ = ev.load_samples(ev.sample_files([ev.PACKAGED_SAMPLES_DIR / tc.CONTINUATION_CALLER]))
    assert len(samples) >= 40
    labels = [s.expected[tc.CONTINUATION_QUESTION] for s in samples]
    assert any(labels) and not all(labels)
    for sample in samples:
        message = sample.raw["message"]
        assert OrchestratorEngineMixin._signals_unfinished_tool_work(message), sample.id
        assert sample.requests == (tc.continuation_decision_request(message),)


# --- streaming: a skipped repair never repeats a streamed answer (codex r1) ---


async def _stream_with_decision(stream_items, p_unfinished):
    from tests.unit.test_turn_completion_guard import _drain_streaming, _streaming_agent

    agent = _streaming_agent(stream_items, "[answer complete]")
    decided = _service(p_unfinished)
    agent.llm_service.decide = decided.decide
    return agent, await _drain_streaming(agent)


@pytest.mark.asyncio
async def test_a_streamed_complete_answer_is_not_repeated_when_the_repair_is_skipped(
    decision_check,
):
    agent, text = await _stream_with_decision(
        [_PLAN, LLMResponse(content=_PLAN, tool_calls=None)], 0.02)
    agent.llm_service.generate_with_messages.assert_not_awaited()
    assert text.count(_PLAN) == 1


@pytest.mark.asyncio
async def test_an_unstreamed_complete_answer_is_delivered_when_the_repair_is_skipped(
    decision_check,
):
    agent, text = await _stream_with_decision(
        [LLMResponse(content=_PLAN, tool_calls=None)], 0.02)
    agent.llm_service.generate_with_messages.assert_not_awaited()
    assert text.count(_PLAN) == 1
