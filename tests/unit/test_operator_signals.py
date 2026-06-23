import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "kestrel_sovereign"
    / "agent"
    / "operator_signals.py"
)
_SPEC = importlib.util.spec_from_file_location("operator_signals_under_test", _MODULE_PATH)
operator_signals = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(operator_signals)

OperatorSignalProducer = operator_signals.OperatorSignalProducer
SOURCE_AUTO_MODE = operator_signals.SOURCE_AUTO_MODE
SOURCE_GOVERNANCE_DELTA = operator_signals.SOURCE_GOVERNANCE_DELTA
SOURCE_TOKEN_BUDGET = operator_signals.SOURCE_TOKEN_BUDGET
supports_inline_system_for_next_route = (
    operator_signals.supports_inline_system_for_next_route
)


class _InlineModelAdapter:
    @staticmethod
    def _model_supports_inline_system(model):
        normalized = (model or "").lower().replace("_", "-")
        return "claude-opus-4-8" in normalized


class _LLM:
    def __init__(self, provider):
        self._provider = provider

    def resolve_provider_routing(self, *, model_override=None, force_local_only=False):
        return [self._provider], model_override


@pytest.mark.asyncio
async def test_auto_mode_uses_inline_system_when_route_and_model_support_it():
    agent = SimpleNamespace(did="agent-1")
    producer = OperatorSignalProducer(agent)
    producer.enqueue_auto_mode(True)
    llm = _LLM(
        {
            "name": "anthropic:api",
            "model": "claude-opus-4-8-20260601",
            "adapter": _InlineModelAdapter(),
            "capabilities": {"supports_inline_system": True},
        }
    )

    batch = await producer.collect_for_turn(
        session_id="s1",
        llm_service=llm,
        model_override=None,
        force_local_only=False,
        budget_summary=None,
        state_of_mind=None,
    )

    assert batch.role == "system"
    assert batch.keep_trailing_system is True
    assert batch.fallback is False
    assert [event.source for event in batch.events] == [SOURCE_AUTO_MODE]
    assert "auto-mode is now enabled" in batch.content


@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_visible_user_notice_on_unsupported_route():
    agent = SimpleNamespace(did="agent-1")
    producer = OperatorSignalProducer(agent)
    producer.enqueue_auto_mode(False)
    llm = _LLM(
        {
            "name": "openai:api",
            "model": "gpt-5",
            "adapter": object(),
            "capabilities": {},
        }
    )

    batch = await producer.collect_for_turn(
        session_id="s1",
        llm_service=llm,
        model_override=None,
        force_local_only=False,
        budget_summary=None,
        state_of_mind=None,
    )

    assert batch.role == "user"
    assert batch.keep_trailing_system is False
    assert batch.fallback is True
    assert batch.content.startswith("<operator_notice>")
    assert "auto-mode is now disabled" in batch.content


@pytest.mark.asyncio
async def test_budget_notice_emits_only_when_crossing_low_threshold():
    producer = OperatorSignalProducer(SimpleNamespace(did="agent-1"))
    llm = _LLM({"name": "openai:api", "model": "gpt-5", "capabilities": {}})
    high = {"total_budget": 10000, "total_used": 7000}
    low = {"total_budget": 10000, "total_used": 8500}

    first = await producer.collect_for_turn(
        session_id="s1",
        llm_service=llm,
        model_override=None,
        force_local_only=False,
        budget_summary=high,
        state_of_mind=None,
    )
    second = await producer.collect_for_turn(
        session_id="s1",
        llm_service=llm,
        model_override=None,
        force_local_only=False,
        budget_summary=low,
        state_of_mind=None,
    )
    third = await producer.collect_for_turn(
        session_id="s1",
        llm_service=llm,
        model_override=None,
        force_local_only=False,
        budget_summary=low,
        state_of_mind=None,
    )

    assert not first.has_events
    assert [event.source for event in second.events] == [SOURCE_TOKEN_BUDGET]
    assert "remaining token budget" in second.content
    assert not third.has_events


@pytest.mark.asyncio
async def test_governance_delta_emits_initial_state_and_dedupes_unchanged_state():
    producer = OperatorSignalProducer(SimpleNamespace(did="agent-1"))
    llm = _LLM({"name": "openai:api", "model": "gpt-5", "capabilities": {}})
    state = SimpleNamespace(
        provider="anthropic",
        model="claude-opus-4-8-20260601",
        governance_mode="complementary",
        transparency="published",
        delegated_principles=["honesty"],
        active_conflicts=[],
    )

    first = await producer.collect_for_turn(
        session_id="s1",
        llm_service=llm,
        model_override=None,
        force_local_only=False,
        budget_summary=None,
        state_of_mind=state,
    )
    second = await producer.collect_for_turn(
        session_id="s1",
        llm_service=llm,
        model_override=None,
        force_local_only=False,
        budget_summary=None,
        state_of_mind=state,
    )

    assert [event.source for event in first.events] == [SOURCE_GOVERNANCE_DELTA]
    assert "constitutional governance state has changed" in first.content
    assert not second.has_events


def test_supports_inline_system_requires_route_and_model_capability():
    llm = _LLM(
        {
            "name": "anthropic:api",
            "model": "claude-sonnet-4-5-20250929",
            "adapter": _InlineModelAdapter(),
            "capabilities": {"supports_inline_system": True},
        }
    )

    supported, route = supports_inline_system_for_next_route(
        llm, model_override=None, force_local_only=False
    )

    assert supported is False
    assert route == "anthropic:api/claude-sonnet-4-5-20250929"
