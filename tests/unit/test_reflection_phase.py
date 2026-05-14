"""Regression coverage for the per-turn reflection phase (#1238).

Acceptance criterion from the ticket:
> in a test conversation where the agent learns three structural facts
> (e.g., a renaming, a package location, a bug observation), all three
> are persisted without the user explicitly asking.

Each of the tests below isolates the reflection phase via a MagicMock
agent — mirroring the pattern in test_turn_completion_guard.py — so we
can drive the LLM response deterministically.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.reflection import (
    REFLECTION_SYSTEM_PROMPT,
    RESERVED_FACT_TOOL_NAMES,
    filter_reflection_tools,
    format_turn_transcript,
    reflection_disabled,
)
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall


def _fact_tool_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"test reflection tool: {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _bind_reflection_helpers(agent):
    """Bind the reflection methods onto a MagicMock agent for testing."""
    agent._run_reflection_phase = (
        OrchestratorEngineMixin._run_reflection_phase.__get__(agent)
    )
    agent._log_reflection_call = (
        OrchestratorEngineMixin._log_reflection_call.__get__(agent)
    )
    agent._finalize_turn = OrchestratorEngineMixin._finalize_turn.__get__(agent)
    # _build_tool_calls_msg is the static helper used when assembling messages.
    agent._build_tool_calls_msg = OrchestratorEngineMixin._build_tool_calls_msg


def _three_fact_response() -> LLMResponse:
    """LLM reflection response that saves three structural facts."""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id="r_1",
                name="save_fact",
                arguments={
                    "subject": "project",
                    "predicate": "renamed_from",
                    "value": "CodeEditFeature -> CodeFeature",
                    "confidence": 1.0,
                },
            ),
            ToolCall(
                id="r_2",
                name="save_fact",
                arguments={
                    "subject": "project",
                    "predicate": "audit_module_path",
                    "value": "kestrel_sovereign/features/response_audit/",
                    "confidence": 1.0,
                },
            ),
            ToolCall(
                id="r_3",
                name="strategy_add_pattern",
                arguments={
                    "pattern": "Tests using bare pip install break in CI",
                    "implication": "always use uv pip install in CI scripts",
                },
            ),
        ],
    )


@pytest.mark.asyncio
async def test_reflection_persists_three_structural_facts_unprompted(monkeypatch):
    """Acceptance criterion: 3 facts in a turn -> 3 tool calls executed."""
    monkeypatch.delenv("KESTREL_REFLECTION_DISABLED", raising=False)

    agent = MagicMock()
    agent.did = "did:web:test-agent"
    _bind_reflection_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(return_value=_three_fact_response())
    agent._build_all_tools = MagicMock(
        return_value=[
            _fact_tool_schema("save_fact"),
            _fact_tool_schema("strategy_add_pattern"),
            _fact_tool_schema("strategy_add_blocker"),
            _fact_tool_schema("github_issue_view"),  # non-fact tool, must be filtered
        ]
    )
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent._visible_known_tool_names = MagicMock(return_value=set())
    agent._execute_tool_batch = AsyncMock()
    agent.observability_store = MagicMock()
    agent.observability_store.log_llm_call = AsyncMock(return_value="event-id")

    outcome = await agent._run_reflection_phase(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "rename the feature"},
            {"role": "assistant", "content": "Renamed."},
        ],
        final_content="Renamed.",
        session_id="session-1238",
    )

    assert outcome.skipped is False
    assert outcome.tool_calls_executed == 3

    # The reflection LLM call must have used only the three fact tools, not the
    # github tool that was filtered out.
    llm_args = agent.llm_service.generate_with_messages.await_args
    passed_tools = [t["function"]["name"] for t in llm_args.kwargs["tools"]]
    assert set(passed_tools) == RESERVED_FACT_TOOL_NAMES
    assert llm_args.kwargs["messages"][0]["content"] == REFLECTION_SYSTEM_PROMPT
    assert llm_args.kwargs["session_id"] == "session-1238"

    # All three saves dispatched through the normal tool-execution path so
    # PRE/POST hooks and observability fire identically to user-driven saves.
    agent._execute_tool_batch.assert_awaited_once()
    batch_args = agent._execute_tool_batch.await_args.args
    dispatched_tool_calls = batch_args[0]
    assert len(dispatched_tool_calls) == 3
    assert [tc.name for tc in dispatched_tool_calls] == [
        "save_fact",
        "save_fact",
        "strategy_add_pattern",
    ]

    # Observability got one row for the reflection LLM call (instrumented like
    # any other phase per #1239).
    agent.observability_store.log_llm_call.assert_awaited_once()
    obs_kwargs = agent.observability_store.log_llm_call.await_args.kwargs
    assert obs_kwargs["provider"] == "reflection"
    assert obs_kwargs["metadata"]["phase"] == "reflection"
    assert obs_kwargs["metadata"]["tool_calls_count"] == 3
    assert obs_kwargs["agent_did"] == "did:web:test-agent"


@pytest.mark.asyncio
async def test_reflection_no_facts_yields_zero_saves(monkeypatch):
    """When the turn surfaces nothing structural, no saves fire."""
    monkeypatch.delenv("KESTREL_REFLECTION_DISABLED", raising=False)

    agent = MagicMock()
    agent.did = "did:web:test"
    _bind_reflection_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content="", tool_calls=None)
    )
    agent._build_all_tools = MagicMock(return_value=[_fact_tool_schema("save_fact")])
    agent._execute_tool_batch = AsyncMock()
    agent.observability_store = MagicMock()
    agent.observability_store.log_llm_call = AsyncMock(return_value="ok")

    outcome = await agent._run_reflection_phase(
        messages=[{"role": "user", "content": "hi"}],
        final_content="hello",
        session_id="s",
    )

    assert outcome.skipped is False
    assert outcome.tool_calls_executed == 0
    agent._execute_tool_batch.assert_not_called()
    agent.observability_store.log_llm_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_reflection_env_disabled_short_circuits(monkeypatch):
    monkeypatch.setenv("KESTREL_REFLECTION_DISABLED", "1")

    agent = MagicMock()
    _bind_reflection_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock()
    agent._build_all_tools = MagicMock(return_value=[_fact_tool_schema("save_fact")])
    agent._execute_tool_batch = AsyncMock()

    outcome = await agent._run_reflection_phase(
        messages=[{"role": "user", "content": "x"}],
        final_content="y",
        session_id="s",
    )

    assert outcome.skipped is True
    assert outcome.skip_reason == "env_disabled"
    agent.llm_service.generate_with_messages.assert_not_called()
    agent._execute_tool_batch.assert_not_called()


@pytest.mark.asyncio
async def test_reflection_skipped_when_no_fact_tools_loaded(monkeypatch):
    """Agents without the three memory/strategy tools skip cleanly."""
    monkeypatch.delenv("KESTREL_REFLECTION_DISABLED", raising=False)

    agent = MagicMock()
    _bind_reflection_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock()
    agent._build_all_tools = MagicMock(
        return_value=[_fact_tool_schema("unrelated_tool"), _fact_tool_schema("github_issue_view")]
    )

    outcome = await agent._run_reflection_phase(
        messages=[{"role": "user", "content": "x"}],
        final_content="y",
        session_id="s",
    )

    assert outcome.skipped is True
    assert outcome.skip_reason == "no_fact_tools_available"
    agent.llm_service.generate_with_messages.assert_not_called()


@pytest.mark.asyncio
async def test_reflection_llm_failure_does_not_break_main_turn(monkeypatch):
    """A crashing reflection LLM must never raise to the caller."""
    monkeypatch.delenv("KESTREL_REFLECTION_DISABLED", raising=False)

    agent = MagicMock()
    agent.did = "did:web:test"
    _bind_reflection_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(side_effect=RuntimeError("LLM down"))
    agent._build_all_tools = MagicMock(return_value=[_fact_tool_schema("save_fact")])
    agent._execute_tool_batch = AsyncMock()
    agent.observability_store = MagicMock()
    agent.observability_store.log_llm_call = AsyncMock(return_value="ok")

    outcome = await agent._run_reflection_phase(
        messages=[{"role": "user", "content": "x"}],
        final_content="y",
        session_id="s",
    )

    assert outcome.skipped is True
    assert outcome.skip_reason == "llm_call_failed"
    assert outcome.error == "LLM down"
    # The error must still be observed
    agent.observability_store.log_llm_call.assert_awaited_once()
    obs_kwargs = agent.observability_store.log_llm_call.await_args.kwargs
    assert obs_kwargs["success"] is False
    assert obs_kwargs["error_message"] == "LLM down"


@pytest.mark.asyncio
async def test_finalize_turn_runs_reflection_then_returns_content(monkeypatch):
    """_finalize_turn is the shared exit-point wrapper for the non-streaming handler."""
    monkeypatch.delenv("KESTREL_REFLECTION_DISABLED", raising=False)

    agent = MagicMock()
    _bind_reflection_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content="", tool_calls=None)
    )
    agent._build_all_tools = MagicMock(return_value=[_fact_tool_schema("save_fact")])
    agent._execute_tool_batch = AsyncMock()
    agent.observability_store = MagicMock()
    agent.observability_store.log_llm_call = AsyncMock(return_value="ok")

    result = await agent._finalize_turn(
        final_content="The deed is done.",
        messages=[{"role": "user", "content": "do it"}],
        session_id="s",
    )

    assert result == "The deed is done."
    agent.llm_service.generate_with_messages.assert_awaited_once()


def test_filter_reflection_tools_keeps_only_three():
    schemas = [
        _fact_tool_schema("save_fact"),
        _fact_tool_schema("strategy_add_pattern"),
        _fact_tool_schema("strategy_add_blocker"),
        _fact_tool_schema("strategy_add_decision"),
        _fact_tool_schema("memory_pin"),
        _fact_tool_schema("github_issue_view"),
    ]
    kept = {t["function"]["name"] for t in filter_reflection_tools(schemas)}
    assert kept == RESERVED_FACT_TOOL_NAMES


def test_format_turn_transcript_includes_tool_results_and_final_content():
    transcript = format_turn_transcript(
        messages=[
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "rename the package"},
            {"role": "assistant", "content": "I'll do that.", "tool_calls": [
                {"function": {"name": "rename_agent_core", "arguments": {"from": "a", "to": "b"}}}
            ]},
            {"role": "tool", "content": "renamed: kestrel_sovereign/features/foo -> bar"},
        ],
        final_content="Done. The package now lives at bar.",
    )

    # system prompt of the main turn must not leak — reflection has its own
    assert "sys prompt" not in transcript
    assert "rename the package" in transcript
    assert "rename_agent_core" in transcript
    assert "renamed: kestrel_sovereign/features/foo -> bar" in transcript
    assert "Done. The package now lives at bar." in transcript


def test_reflection_disabled_reads_env(monkeypatch):
    monkeypatch.delenv("KESTREL_REFLECTION_DISABLED", raising=False)
    assert reflection_disabled() is False

    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("KESTREL_REFLECTION_DISABLED", truthy)
        assert reflection_disabled() is True, f"expected disabled for {truthy!r}"

    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("KESTREL_REFLECTION_DISABLED", falsy)
        assert reflection_disabled() is False, f"expected enabled for {falsy!r}"
