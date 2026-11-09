"""Active-session privacy transition contracts."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.privacy import PrivacyMode


class _StreamProbe:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.force_local_only_values = []

    async def stream_with_tool_detection(self, **kwargs):
        from kestrel_sovereign.llm.adapter import LLMResponse

        self.force_local_only_values.append(kwargs["force_local_only"])
        self.started.set()
        yield "hello"
        await self.release.wait()
        yield " world"
        yield LLMResponse(content="hello world", tool_calls=[])


async def _make_streaming_agent(native_tool_effect_agent):
    from kestrel_sovereign.agent.context_manager import ContextResult

    agent = await native_tool_effect_agent(did="did:test:privacy-active-session")
    stream = _StreamProbe()
    agent.llm_service.stream_with_tool_detection = stream.stream_with_tool_detection
    agent.storage.set_privacy_mode = MagicMock(wraps=agent.storage.set_privacy_mode)
    context = ContextResult(system_prompt="system", messages=[], total_tokens=0, budget_summary={})
    agent.context_manager = SimpleNamespace(build_context=AsyncMock(return_value=context))
    agent.user_prompt_template = "{context}{query}"
    agent._cached_features_prompt = ""
    agent.extension = None
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._maybe_audit = AsyncMock(return_value=None)
    agent._get_governing_constitution = AsyncMock(return_value="constitution")
    agent.check_solvency = AsyncMock(return_value="gpt-5-mini")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._post_response_pipeline = AsyncMock()
    return agent, stream


@pytest.mark.asyncio
async def test_privacy_transition_waits_for_active_stream_before_switching_modes(native_tool_effect_agent):
    agent, stream = await _make_streaming_agent(native_tool_effect_agent)

    chunks = []

    async def consume_stream():
        async for chunk in agent.process_input_streaming("hello"):
            chunks.append(chunk)

    stream_task = asyncio.create_task(consume_stream())
    await stream.started.wait()
    await asyncio.sleep(0)

    transition_task = asyncio.create_task(
        agent.set_privacy_mode_with_effects(PrivacyMode.ISOLATED)
    )
    await asyncio.sleep(0)

    assert transition_task.done() is False
    assert agent._privacy_mode == PrivacyMode.NORMAL
    assert stream.force_local_only_values == [False]

    stream.release.set()
    await stream_task
    transition = await transition_task

    assert chunks == ["hello", " world"]
    assert agent.context_manager.build_context.await_args.kwargs["include_briefing"] is True
    assert agent._session_briefed is True
    assert transition.allows_cloud_llm is False
    assert agent._privacy_mode == PrivacyMode.ISOLATED
    agent.storage.set_privacy_mode.assert_called_once_with(PrivacyMode.ISOLATED)


@pytest.mark.asyncio
async def test_streaming_omits_briefing_after_session_was_briefed(native_tool_effect_agent):
    agent, stream = await _make_streaming_agent(native_tool_effect_agent)
    agent._session_briefed = True
    stream.release.set()

    chunks = [
        chunk
        async for chunk in agent.process_input_streaming(
            "second turn",
            session_id="session-1",
        )
    ]

    assert chunks == ["hello", " world"]
    assert agent.context_manager.build_context.await_args.kwargs["include_briefing"] is False
    assert agent._session_briefed is True
