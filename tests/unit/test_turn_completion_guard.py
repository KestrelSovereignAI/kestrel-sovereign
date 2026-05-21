"""Regression coverage for premature turn-yield repair (#1237)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall


def _tool_schema(name: str = "github_issue_view") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _bind_turn_completion_helpers(agent):
    agent._repair_premature_turn_yield = (
        OrchestratorEngineMixin._repair_premature_turn_yield.__get__(agent)
    )
    agent._signals_unfinished_tool_work = OrchestratorEngineMixin._signals_unfinished_tool_work
    agent._append_missing_tool_call_repair = OrchestratorEngineMixin._append_missing_tool_call_repair


def test_assistant_tool_history_preserves_provider_reasoning_from_raw_dict():
    agent = MagicMock()
    agent._build_tool_calls_msg = OrchestratorEngineMixin._build_tool_calls_msg
    agent._extract_response_reasoning_content = (
        OrchestratorEngineMixin._extract_response_reasoning_content
    )
    agent._build_assistant_tool_history_msg = (
        OrchestratorEngineMixin._build_assistant_tool_history_msg.__get__(agent)
    )

    msg = agent._build_assistant_tool_history_msg(
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"q": "hi"})],
            raw={"reasoning_content": "I need to call lookup."},
        )
    )

    assert msg["reasoning_content"] == "I need to call lookup."
    assert msg["tool_calls"][0]["function"]["arguments"] == {"q": "hi"}


def test_assistant_tool_history_preserves_provider_reasoning_from_openai_response():
    raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(reasoning_content="Use the tool.")
            )
        ]
    )

    response = LLMResponse(
        tool_calls=[ToolCall(id="call_1", name="lookup", arguments={})],
        raw=raw,
    )

    reasoning = OrchestratorEngineMixin._extract_response_reasoning_content(response)

    assert reasoning == "Use the tool."


@pytest.mark.asyncio
async def test_no_tool_continuation_gets_one_repair_step():
    agent = MagicMock()
    _bind_turn_completion_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="github_issue_view", arguments={})],
            ),
            LLMResponse(content="Issue loaded.", tool_calls=None),
        ]
    )
    agent._build_tool_calls_msg = MagicMock(
        return_value=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "github_issue_view", "arguments": "{}"},
            }
        ]
    )
    agent._execute_tool_batch = AsyncMock()
    agent._build_all_tools = MagicMock(return_value=[])
    agent._prune_orchestrator_messages = MagicMock(side_effect=lambda msgs, _tools: msgs)

    handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)
    result = await handler(
        response=LLMResponse(content="Let me check the GitHub issue.", tool_calls=None),
        feature_tools=[_tool_schema()],
        system_prompt="sys",
        force_local_only=False,
        effective_model="gpt-5.4",
        user_message="has this been done yet?",
        session_id="session-123",
    )

    assert result == "Issue loaded."
    assert agent.llm_service.generate_with_messages.await_count == 2
    repair_call = agent.llm_service.generate_with_messages.await_args_list[0].kwargs
    assert repair_call["tools"] == [_tool_schema()]
    assert repair_call["session_id"] == "session-123"
    assert repair_call["messages"][-2] == {
        "role": "assistant",
        "content": "Let me check the GitHub issue.",
    }
    assert "did not emit a tool call" in repair_call["messages"][-1]["content"]
    agent._execute_tool_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_no_tool_answer_does_not_repair():
    agent = MagicMock()
    _bind_turn_completion_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock()

    handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)
    result = await handler(
        response=LLMResponse(content="No, it is still open.", tool_calls=None),
        feature_tools=[_tool_schema()],
        system_prompt="sys",
        force_local_only=False,
        effective_model="gpt-5.4",
        user_message="has this been done yet?",
        session_id="session-123",
    )

    assert result == "No, it is still open."
    agent.llm_service.generate_with_messages.assert_not_awaited()


class _FeatureForTurnCompletion(Feature):
    tool_description = "test feature"

    async def initialize(self):
        return None


@pytest.mark.asyncio
async def test_feature_subagent_no_tool_continuation_gets_repair_step():
    tool = MagicMock()
    tool.name = "talon_claim"
    tool.execute = AsyncMock(return_value={"success": True, "claimed": 1237})

    agent = MagicMock()
    agent.hooks_manager = None
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="talon_claim", arguments={})],
            ),
            LLMResponse(content="Talon claimed the issue.", tool_calls=None),
        ]
    )
    feature = _FeatureForTurnCompletion(agent)
    feature.get_tools = MagicMock(return_value=[tool])

    result = await feature._handle_feature_tool_calls(
        response=LLMResponse(content="Let me use Talon for that.", tool_calls=None),
        tools=[_tool_schema("talon_claim")],
        system_prompt="sys",
        user_prompt="Task: claim issue 1237",
    )

    assert result == "Talon claimed the issue."
    assert agent.llm_service.generate_with_messages.await_count == 2
    repair_call = agent.llm_service.generate_with_messages.await_args_list[0].kwargs
    assert repair_call["messages"][1] == {
        "role": "user",
        "content": "Task: claim issue 1237",
    }
    assert "did not emit a tool call" in repair_call["messages"][-1]["content"]
    tool.execute.assert_awaited_once_with()
