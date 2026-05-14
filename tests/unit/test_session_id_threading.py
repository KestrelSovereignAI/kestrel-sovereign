"""Verify ``session_id`` threads from agent loop to LLMService callers (#821).

Two layers:

- Signature-level: every public method that should accept ``session_id`` does.
  This catches accidental rename / drop in future refactors.
- Behavioral: ``OrchestratorEngineMixin._handle_orchestrator_response`` actually
  passes ``session_id`` to ``llm_service.generate_with_messages`` when it loops
  on tool results. The mocking pattern follows ``test_denied_tools_dispatch.py``
  — bind the mixin method to a MagicMock agent and stub the heavy collaborators.
"""

import inspect

import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.llm.streaming import StreamingMixin


class TestSignatures:
    def test_generate_with_messages_accepts_session_id(self):
        sig = inspect.signature(LLMService.generate_with_messages)
        assert "session_id" in sig.parameters
        # Default must be None so existing callers don't break.
        assert sig.parameters["session_id"].default is None

    def test_stream_with_messages_accepts_session_id(self):
        sig = inspect.signature(StreamingMixin.stream_with_messages)
        assert "session_id" in sig.parameters
        assert sig.parameters["session_id"].default is None

    def test_stream_with_tool_detection_accepts_session_id(self):
        sig = inspect.signature(StreamingMixin.stream_with_tool_detection)
        assert "session_id" in sig.parameters
        assert sig.parameters["session_id"].default is None

    def test_orchestrator_handlers_accept_session_id(self):
        for name in (
            "_handle_orchestrator_response",
            "_handle_orchestrator_response_streaming",
        ):
            sig = inspect.signature(getattr(OrchestratorEngineMixin, name))
            assert "session_id" in sig.parameters, f"{name} missing session_id"
            assert sig.parameters["session_id"].default is None


@pytest.mark.asyncio
class TestOrchestratorThreadsSessionId:
    """``_handle_orchestrator_response`` threads session_id into the LLM call."""

    def _bind_handler(self, agent):
        # The reflection phase (#1238) lives on the same mixin and is invoked
        # from every final-return path. Bind it as a real method; with no
        # fact-save tools loaded it short-circuits and doesn't perturb the
        # session_id assertions.
        agent._run_reflection_phase = (
            OrchestratorEngineMixin._run_reflection_phase.__get__(agent)
        )
        agent._log_reflection_call = (
            OrchestratorEngineMixin._log_reflection_call.__get__(agent)
        )
        agent._finalize_turn = OrchestratorEngineMixin._finalize_turn.__get__(agent)
        return OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)

    async def test_session_id_passed_to_generate_with_messages(self):
        # Initial response has a tool call so the loop body runs at least once.
        initial = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="some_tool", arguments={})],
        )
        # Second LLM response has no tool calls → loop exits cleanly.
        terminating = LLMResponse(content="done", tool_calls=None)

        agent = MagicMock()
        agent.llm_service = MagicMock()
        agent.llm_service.generate_with_messages = AsyncMock(return_value=terminating)
        # Stub all collaborators that would otherwise need a full agent.
        agent.features = {}
        agent._direct_tools = {}
        agent._build_tool_calls_msg = MagicMock(
            return_value=[
                {"id": "call_1", "type": "function", "function": {"name": "some_tool", "arguments": "{}"}}
            ]
        )
        agent._execute_tool_batch = AsyncMock()  # no-op
        agent._build_all_tools = MagicMock(return_value=[])
        agent._prune_orchestrator_messages = MagicMock(side_effect=lambda msgs, _tools: msgs)

        handler = self._bind_handler(agent)
        result = await handler(
            response=initial,
            feature_tools=[],
            system_prompt="sys",
            force_local_only=False,
            effective_model="gpt-5.4",
            user_message="hi",
            session_id="session-xyz",
        )

        assert result == "done"
        agent.llm_service.generate_with_messages.assert_awaited_once()
        kwargs = agent.llm_service.generate_with_messages.await_args.kwargs
        assert kwargs["session_id"] == "session-xyz"

    async def test_string_response_short_circuits_without_calling_llm(self):
        # Sanity: when the orchestrator's first response is already a string
        # (no tool calls in flight), we never hit generate_with_messages, so
        # session_id has nothing to do — but the parameter must still be
        # accepted without error.
        agent = MagicMock()
        agent.llm_service = MagicMock()
        agent.llm_service.generate_with_messages = AsyncMock()

        handler = self._bind_handler(agent)
        result = await handler(
            response="already-final",
            feature_tools=[],
            system_prompt="sys",
            force_local_only=False,
            effective_model="gpt-5.4",
            session_id="session-xyz",
        )

        assert result == "already-final"
        agent.llm_service.generate_with_messages.assert_not_awaited()
