"""Regression coverage for premature turn-yield repair (#1237)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall


def _tool_schema(name: str = "example_tool") -> dict:
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


def test_assistant_tool_history_omits_empty_provider_reasoning():
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
            tool_calls=[ToolCall(id="call_1", name="lookup", arguments={})],
            raw={"reasoning_content": ""},
        )
    )

    assert "reasoning_content" not in msg


def test_assistant_tool_history_omits_reasoning_without_tool_calls():
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
            content="done",
            tool_calls=None,
            raw={"reasoning_content": "No replay for text-only turns."},
        )
    )

    assert msg == {"role": "assistant", "content": "done"}


@pytest.mark.asyncio
async def test_no_tool_continuation_gets_one_repair_step():
    agent = MagicMock()
    _bind_turn_completion_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="example_tool", arguments={})],
            ),
            LLMResponse(content="Issue loaded.", tool_calls=None),
        ]
    )
    agent._build_tool_calls_msg = MagicMock(
        return_value=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "example_tool", "arguments": "{}"},
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


def test_tool_call_emitted_as_text_detected():
    """Literal tool-call markup in assistant text is recognized as unfinished
    tool work (root cause of the `kestrel ask` tools-as-text bug)."""
    xml = (
        '<function_calls><invoke name="todo_add">'
        '<parameter name="title">x</parameter></invoke></function_calls>'
    )
    assert OrchestratorEngineMixin._tool_call_emitted_as_text(xml)
    assert OrchestratorEngineMixin._signals_unfinished_tool_work(xml)
    # Other inline-syntax dialects also count.
    assert OrchestratorEngineMixin._signals_unfinished_tool_work(
        '<tool_call>{"name": "todo_add"}</tool_call>'
    )
    assert OrchestratorEngineMixin._signals_unfinished_tool_work(
        '<invoke name="todo_add">'
    )
    # Plain prose without markup is NOT flagged by the text-syntax detector.
    assert not OrchestratorEngineMixin._tool_call_emitted_as_text(
        "I added the todo and it now has id 4."
    )


@pytest.mark.asyncio
async def test_tool_call_as_text_gets_repaired_and_executed():
    """A model that writes <function_calls>/<invoke> markup as TEXT (no
    structured tool_use) is given one repair turn that re-emits a real tool
    call, which then executes — instead of silently fabricating success.

    This is the regression for the `kestrel ask` -> /api/agent/invoke path
    returning literal tool-call syntax with narrated fake results.
    """
    agent = MagicMock()
    _bind_turn_completion_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        side_effect=[
            # Repair turn: model now emits a REAL structured tool call.
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="todo_add", arguments={"title": "x"})],
            ),
            # Follow-up after the tool result: final answer.
            LLMResponse(content="Added todo, id 4.", tool_calls=None),
        ]
    )
    agent._build_tool_calls_msg = MagicMock(
        return_value=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "todo_add", "arguments": '{"title": "x"}'},
            }
        ]
    )
    agent._execute_tool_batch = AsyncMock()
    agent._build_all_tools = MagicMock(return_value=[])
    agent._prune_orchestrator_messages = MagicMock(side_effect=lambda msgs, _tools: msgs)

    tools_as_text = (
        '<function_calls><invoke name="todo_add">'
        '<parameter name="title">x</parameter></invoke></function_calls>\n'
        "Done — got ID 4. No tool errors."
    )

    handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)
    result = await handler(
        response=LLMResponse(content=tools_as_text, tool_calls=None),
        feature_tools=[_tool_schema("todo_add")],
        system_prompt="sys",
        force_local_only=False,
        effective_model="claude-opus-4-8",
        user_message="add a todo titled x",
        session_id="session-123",
    )

    assert result == "Added todo, id 4."
    assert agent.llm_service.generate_with_messages.await_count == 2
    # The repair turn used the sterner tools-as-text directive.
    repair_call = agent.llm_service.generate_with_messages.await_args_list[0].kwargs
    assert "written as plain text" in repair_call["messages"][-1]["content"]
    assert "fabricated" in repair_call["messages"][-1]["content"]
    # The re-emitted structured call was actually executed.
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
async def test_feature_subagent_tool_history_preserves_provider_reasoning():
    tool = MagicMock()
    tool.name = "health_check"
    tool.execute = AsyncMock(return_value={"status": "ok"})

    agent = MagicMock()
    agent.hooks_manager = None
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content="Health is ok.", tool_calls=None)
    )
    feature = _FeatureForTurnCompletion(agent)
    feature.get_tools = MagicMock(return_value=[tool])

    result = await feature._handle_feature_tool_calls(
        response=LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="health_check", arguments={})],
            raw={"reasoning_content": "Need a health probe."},
        ),
        tools=[_tool_schema("health_check")],
        system_prompt="sys",
        user_prompt="Task: health",
    )

    assert result == "Health is ok."
    continuation_messages = agent.llm_service.generate_with_messages.await_args.kwargs[
        "messages"
    ]
    assert continuation_messages[2]["role"] == "assistant"
    assert continuation_messages[2]["reasoning_content"] == "Need a health probe."
    assert continuation_messages[2]["content"] == ""
    assert continuation_messages[2]["tool_calls"][0]["function"]["arguments"] == {}
    tool.execute.assert_awaited_once_with()


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
