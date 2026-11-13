"""Scripted streaming collaborators around a real native Agent and primary SQL."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.agent.context_manager import ContextResult


async def streaming_agent(native_tool_effect_agent, *, persisted=None, connected_backend=None):
    agent = await native_tool_effect_agent(did="did:test:streaming-component", connected_backend=connected_backend)
    return prepare_streaming_agent(agent, persisted=persisted)


def prepare_streaming_agent(agent, *, persisted=None, tools=None):
    """Script context/model collaborators while retaining an existing native host."""
    context = ContextResult(system_prompt="system", messages=[], total_tokens=0,
                            budget_summary={}, dynamic_user_context="ctx")
    agent.context_manager = SimpleNamespace(build_context=AsyncMock(return_value=context))
    agent.user_prompt_template = "{context}{query}"
    agent._cached_features_prompt = ""
    agent.extension = None
    agent.operator_signal_producer = None
    agent._maybe_audit = AsyncMock(return_value=None)
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._get_governing_constitution = AsyncMock(return_value="constitution")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[] if tools is None else tools)
    agent._post_response_pipeline = AsyncMock(return_value=None)
    write = agent.privacy_agent.add_conversation

    async def record(role, content, **kwargs):
        result = await write(role, content, **kwargs)
        if persisted is not None:
            persisted.append({"role": role, "content": content, **kwargs})
        return result

    agent.privacy_agent.add_conversation = AsyncMock(side_effect=record)
    return agent


async def collect_native_stream(agent, operation, *, chunks=None):
    """One explicit input owns a private streaming component through settlement."""
    from tests.utils.native_tool_effects import drive_native

    observed = [] if chunks is None else chunks

    async def collect():
        async for chunk in operation():
            observed.append(chunk)
        return observed

    return await drive_native(agent, collect)


def install_input_checkpoint_provider(agent, registered, monkeypatch, *, followup=False, dispatch='inline',
                                      before_tools=None, tool_markers=False, interleaved_parts=False):
    """Script only model I/O; all governed tools and their receipts remain real."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall, ToolCallStarted
    from kestrel_sovereign.tools.effect_journal import ToolInputSuspended
    from kestrel_sovereign.agent.streaming import _build_tool_sentinel
    from kestrel_sovereign.agent.parts import build_part_sentinel

    first = registered['record_marker']
    suspended = []
    provider_calls = []
    async def stream(**kwargs):
        provider_calls.append(True)
        executed = []
        yield 'UNFINISHED-PRE-INPUT-PROSE'
        if interleaved_parts:
            yield build_part_sentinel({'type': 'notice', 'data': 'BEFORE-BATCH-' + str(len(provider_calls))})
        if before_tools is not None:
            await before_tools()
        if followup and len(provider_calls) == 1:
            yield LLMResponse(content='', stop_reason='tool_calls', tool_calls=[ToolCall(
                id='owned-first-batch', name=first.binding.name,
                arguments={'marker': 'completed-before-input'})])
            return
        names = ('open_marker',) if followup else ('record_marker', 'open_marker')
        if dispatch == 'batch':
            yield LLMResponse(content='', stop_reason='tool_calls', tool_calls=[ToolCall(
                id='owned-batch-' + str(index), name=registered[name].binding.name,
                arguments={'marker': 'completed-before-input' if name == 'record_marker' else 'must-wait'})
                for index, name in enumerate(names)])
            return
        for index, name in enumerate(names):
            registered_tool = registered[name]
            if tool_markers:
                yield ToolCallStarted(index=index, id='owned-inline-' + str(index), name=registered_tool.binding.name)
            try:
                outcome = await kwargs['tool_executor'](registered_tool.binding.name,
                    {'marker': 'completed-before-input' if name == 'record_marker' else 'must-wait'})
            except ToolInputSuspended as signal:
                suspended.append(signal)
                raise
            if tool_markers:
                yield _build_tool_sentinel('done', registered_tool.binding.name)
            executed.append(dict(id='owned-inline-' + str(index), name=registered_tool.binding.name,
                arguments=outcome[0], result=outcome[1], dispatch_id=outcome.dispatch_id))
        final = LLMResponse(content='UNFINISHED-POST-INPUT-PROSE', stop_reason='completed')
        final.executed_tool_calls = executed
        yield final

    monkeypatch.setattr(agent.llm_service, 'stream_with_tool_detection', stream)

    async def generate(**kwargs):
        final = None
        async for item in stream(**kwargs):
            if isinstance(item, LLMResponse):
                final = item
        assert final is not None, 'Scripted provider did not finish or suspend'
        return final

    monkeypatch.setattr(agent.llm_service, 'generate_with_messages', generate)

    return provider_calls, suspended
