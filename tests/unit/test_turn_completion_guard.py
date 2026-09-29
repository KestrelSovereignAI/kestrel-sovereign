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
    agent._execute_tool_batch_at_stop_boundary = (
        OrchestratorEngineMixin._execute_tool_batch_at_stop_boundary.__get__(agent)
    )
    agent._build_all_tools = MagicMock(return_value=[])
    agent._prune_orchestrator_messages = MagicMock(side_effect=lambda msgs, _tools, **_kw: msgs)

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
    assert "made no tool call" in repair_call["messages"][-1]["content"]
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


def test_tool_call_as_text_detector_ignores_documentation_and_examples():
    """The detector requires a real invocation shape and exempts code, so an
    answer that merely DISCUSSES or quotes the markup is not treated as an
    unexecuted tool call (codex P2: avoid needless repair turns)."""
    # Bare tag mentions in prose — no invocation shape.
    assert not OrchestratorEngineMixin._tool_call_emitted_as_text(
        "Use the <invoke> element inside a <function_calls> block to call a tool."
    )
    assert not OrchestratorEngineMixin._tool_call_emitted_as_text(
        "The <tool_call> and <tool_use> tags wrap structured calls."
    )
    # A real invocation, but shown as a fenced code example — documentation.
    assert not OrchestratorEngineMixin._tool_call_emitted_as_text(
        'Here is the syntax:\n```\n<invoke name="todo_add">'
        '<parameter name="title">x</parameter></invoke>\n```'
    )
    # A real invocation quoted in an inline code span — still documentation.
    assert not OrchestratorEngineMixin._tool_call_emitted_as_text(
        'You write `<invoke name="todo_add">` to call it.'
    )
    # But a raw, real invocation (not in code) still triggers.
    assert OrchestratorEngineMixin._tool_call_emitted_as_text(
        '<invoke name="todo_add"><parameter name="title">x</parameter></invoke>'
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
    agent._execute_tool_batch_at_stop_boundary = (
        OrchestratorEngineMixin._execute_tool_batch_at_stop_boundary.__get__(agent)
    )
    agent._build_all_tools = MagicMock(return_value=[])
    agent._prune_orchestrator_messages = MagicMock(side_effect=lambda msgs, _tools, **_kw: msgs)

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
    tool.name = "launch_job"
    tool.execute = AsyncMock(return_value={"success": True, "claimed": 1237})

    agent = MagicMock()
    agent.hooks_manager = None
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="launch_job", arguments={})],
            ),
            LLMResponse(content="The job was launched.", tool_calls=None),
        ]
    )
    feature = _FeatureForTurnCompletion(agent)
    feature.get_tools = MagicMock(return_value=[tool])

    result = await feature._handle_feature_tool_calls(
        response=LLMResponse(
            content="Let me use the job launcher for that.", tool_calls=None
        ),
        tools=[_tool_schema("launch_job")],
        system_prompt="sys",
        user_prompt="Task: claim issue 1237",
    )

    assert result == "The job was launched."
    assert agent.llm_service.generate_with_messages.await_count == 2
    repair_call = agent.llm_service.generate_with_messages.await_args_list[0].kwargs
    assert repair_call["messages"][1] == {
        "role": "user",
        "content": "Task: claim issue 1237",
    }
    assert "made no tool call" in repair_call["messages"][-1]["content"]
    tool.execute.assert_awaited_once_with()


# --- A finished answer survives the repair --------------------------------
#
# Live case, 2026-09-29: an orchestrating agent answered a status request with
# a full report whose body named its next steps ("when CI finishes I will run
# the codex review"). The match fired the repair five times in one turn, and
# each no-tool repair reply ("This turn is done...") replaced the report, so
# the caller received only that reply and the report was lost. The pattern
# cannot tell that plan from an unfinished announcement, so the repair must be
# harmless when it fires on a finished answer.

_STATUS_REPORT = (
    "Nothing is running. #3380 is held until CI shows its PostgreSQL cases run.\n\n"
    "When CI on #3380 finishes, I will run the codex review and then check the "
    "merge guard against the head SHA.\n\n"
    "Nothing is waiting on you."
)


@pytest.mark.parametrize(
    "content",
    [
        "The issue is open.\n\nLet me check the GitHub issue comments.",
        # An announcement followed by a closing line (codex review r3).
        "I'll check the GitHub issue now.\n\nPlease wait.",
    ],
)
def test_an_unmade_announced_call_is_unfinished_work(content):
    assert OrchestratorEngineMixin._signals_unfinished_tool_work(content) is True
    assert Feature._signals_unfinished_tool_work(content) is True


@pytest.mark.asyncio
async def test_finished_report_survives_a_repair_it_did_not_need():
    agent = MagicMock()
    _bind_turn_completion_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content="[answer complete]", tool_calls=None)
    )

    handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)
    result = await handler(
        response=LLMResponse(content=_STATUS_REPORT, tool_calls=None),
        feature_tools=[_tool_schema()],
        system_prompt="sys",
        force_local_only=False,
        effective_model="claude-opus-5-5",
        user_message="what is your status?",
        session_id="session-123",
    )

    assert result == _STATUS_REPORT
    assert agent.llm_service.generate_with_messages.await_count == 1


_ENDS_WITH_PLAN = (
    "#3380 is held until CI is green.\n\n"
    "When CI finishes, I will run the codex review on the new head."
)


async def _repair_no_tool(reply: str):
    agent = MagicMock()
    _bind_turn_completion_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content=reply, tool_calls=None)
    )
    handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)
    result = await handler(
        response=LLMResponse(content=_ENDS_WITH_PLAN, tool_calls=None),
        feature_tools=[_tool_schema()],
        system_prompt="sys",
        force_local_only=False,
        effective_model="claude-opus-5-5",
        user_message="what is your status?",
        session_id="session-123",
    )
    return agent, result


@pytest.mark.asyncio
async def test_confirmed_answer_is_delivered_as_written():
    agent, result = await _repair_no_tool("[answer complete]")

    assert result == _ENDS_WITH_PLAN
    assert agent.llm_service.generate_with_messages.await_count == 1
    prompt = agent.llm_service.generate_with_messages.await_args.kwargs["messages"][-1]
    assert prompt["role"] == "user"
    assert "not a message from the user" in prompt["content"]
    assert "[answer complete]" in prompt["content"]


@pytest.mark.asyncio
async def test_repair_reply_is_added_to_the_answer_not_substituted():
    agent, result = await _repair_no_tool(
        "[answer complete]\n\nThe watch on #3380 is set."
    )

    assert result == f"{_ENDS_WITH_PLAN}\n\nThe watch on #3380 is set."


@pytest.mark.asyncio
async def test_unconfirmed_repair_reply_is_the_new_answer():
    """A reply without the marker is a new answer, not an addition: appending
    "I cannot reach GitHub" to "I will run the review" would contradict itself
    (codex review r4)."""
    agent, result = await _repair_no_tool("I cannot reach GitHub right now.")

    assert result == "I cannot reach GitHub right now."


@pytest.mark.asyncio
async def test_tool_call_markup_is_replaced_by_the_repaired_answer():
    """Markup written as text executed nothing, so it is not kept as an answer."""
    agent = MagicMock()
    _bind_turn_completion_helpers(agent)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content="I could not add the todo.", tool_calls=None)
    )
    handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)
    result = await handler(
        response=LLMResponse(
            content='<invoke name="todo_add"><parameter name="title">x</parameter></invoke>',
            tool_calls=None,
        ),
        feature_tools=[_tool_schema("todo_add")],
        system_prompt="sys",
        force_local_only=False,
        effective_model="claude-opus-5-5",
        user_message="add a todo titled x",
        session_id="session-123",
    )

    assert result == "I could not add the todo."


@pytest.mark.asyncio
async def test_already_streamed_answer_yields_only_the_addition():
    repaired = LLMResponse(content="[answer complete]", tool_calls=None)
    settled = OrchestratorEngineMixin._settle_repaired_turn(
        _ENDS_WITH_PLAN, repaired, original_delivered=True,
    )
    assert settled.content == ""

    repaired = LLMResponse(content="[answer complete] Watch set.", tool_calls=None)
    settled = OrchestratorEngineMixin._settle_repaired_turn(
        _ENDS_WITH_PLAN, repaired, original_delivered=True,
    )
    assert settled.content == "Watch set."

    repaired = LLMResponse(content="I cannot reach GitHub.", tool_calls=None)
    settled = OrchestratorEngineMixin._settle_repaired_turn(
        _ENDS_WITH_PLAN, repaired, original_delivered=True,
    )
    assert settled.content == "I cannot reach GitHub."


@pytest.mark.asyncio
async def test_feature_subagent_confirmed_answer_is_kept():
    agent = MagicMock()
    agent.hooks_manager = None
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content="[answer complete]", tool_calls=None)
    )
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
    assert agent.llm_service.generate_with_messages.await_count == 1


def test_settled_repair_keeps_runtime_attributes():
    """Adapters and the service attach non-field attributes to the response;
    settling the content must not drop them (codex review r1 P1)."""
    repaired = LLMResponse(content="[answer complete]", tool_calls=None)
    repaired.model = "claude-opus-5-5"
    repaired.provider = "anthropic:plan"

    settled = OrchestratorEngineMixin._settle_repaired_turn(
        _ENDS_WITH_PLAN, repaired, original_delivered=False,
    )

    assert settled.content == _ENDS_WITH_PLAN
    assert settled.model == "claude-opus-5-5"
    assert settled.provider == "anthropic:plan"


def test_repair_that_ran_tools_inline_goes_on_as_the_turn():
    """A codex-routed repair that executed tools inline acted; its answer and
    its executed_tool_calls reach the breadcrumb path unchanged."""
    executed = [{"name": "get_github_issue", "arguments": {}, "result": "ok"}]
    repaired = LLMResponse(content="Issue 3380 is open.", tool_calls=None)
    repaired.executed_tool_calls = executed

    settled = OrchestratorEngineMixin._settle_repaired_turn(
        _ENDS_WITH_PLAN, repaired, original_delivered=False,
    )

    assert settled is repaired
    assert settled.content == "Issue 3380 is open."
    assert settled.executed_tool_calls == executed


def _streaming_agent(stream_items, repair_reply):
    agent = MagicMock()
    agent.features = {}
    agent._direct_tools = {}
    agent._build_assistant_tool_history_msg = MagicMock(
        return_value={"role": "assistant", "content": "", "tool_calls": []}
    )
    agent._execute_tool_batch = AsyncMock()
    agent._execute_tool_batch_at_stop_boundary = (
        OrchestratorEngineMixin._execute_tool_batch_at_stop_boundary.__get__(agent)
    )
    agent._build_all_tools = MagicMock(return_value=[_tool_schema()])
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent._visible_known_tool_names = MagicMock(return_value=set())
    agent._known_tool_names = MagicMock(return_value=set())
    agent._make_inline_tool_executor = MagicMock(return_value=None)
    agent._prune_orchestrator_messages = MagicMock(
        side_effect=lambda msgs, _tools, **_kw: msgs
    )
    agent.is_request_cancelled = MagicMock(return_value=False)
    _bind_turn_completion_helpers(agent)

    async def _fake_stream(*_args, **_kwargs):
        for item in stream_items:
            yield item

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = _fake_stream
    agent.llm_service.generate_with_messages = AsyncMock(
        return_value=LLMResponse(content=repair_reply, tool_calls=None)
    )
    return agent


async def _drain_streaming(agent):
    handler = OrchestratorEngineMixin._handle_orchestrator_response_streaming.__get__(agent)
    chunks = []
    async for chunk in handler(
        response=LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="example_tool", arguments={})],
        ),
        feature_tools=[_tool_schema()],
        system_prompt="sys",
        force_local_only=False,
        effective_model="claude-opus-5-5",
        user_message="what is your status?",
        tool_events=[],
        tool_results=[],
        session_id="s-1",
    ):
        if isinstance(chunk, str):
            chunks.append(chunk)
    return "".join(chunks)


@pytest.mark.asyncio
async def test_streamed_answer_is_not_repeated_after_a_confirmed_repair():
    agent = _streaming_agent(
        [_ENDS_WITH_PLAN, LLMResponse(content=_ENDS_WITH_PLAN, tool_calls=None)],
        "[answer complete]",
    )

    text = await _drain_streaming(agent)

    assert agent.llm_service.generate_with_messages.await_count == 1
    assert text.count(_ENDS_WITH_PLAN) == 1
    assert "[answer complete]" not in text


@pytest.mark.asyncio
async def test_unstreamed_answer_is_delivered_after_a_confirmed_repair():
    """A terminal response whose text never streamed as chunks has not reached
    the client, so the confirmed answer is delivered whole (codex review r2)."""
    agent = _streaming_agent(
        [LLMResponse(content=_ENDS_WITH_PLAN, tool_calls=None)],
        "[answer complete]",
    )

    text = await _drain_streaming(agent)

    assert agent.llm_service.generate_with_messages.await_count == 1
    assert _ENDS_WITH_PLAN in text
    assert "[answer complete]" not in text
