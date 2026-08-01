"""Tool continuations must carry the turn's conversation history (#2841).

The turn's FIRST provider call is built by the caller as
``[system, *ContextResult.messages, user]``. Both orchestrator handlers then
re-derive their own array for every post-tool continuation. Before #2841 they
seeded it as ``[system, user]`` only, so the synthesis call — the one that
writes the text the user actually reads — answered from a blank conversation:
it denied that earlier turns had happened and re-asked for information it had
already been given.

These tests assert the continuation's message array reconstructs the prefix the
first call used, on both transports, and that the prune boundary follows it.
"""

import inspect

import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall


HISTORY = [
    {"role": "user", "content": "Remember this codeword: ZEBRA-9042."},
    {"role": "assistant", "content": "ok"},
]


def _tool_call_response():
    """An initial response with one tool call, so the loop body runs once."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name="some_tool", arguments={})],
    )


def _base_agent():
    agent = MagicMock()
    agent.features = {}
    agent._direct_tools = {}
    agent._build_assistant_tool_history_msg = MagicMock(
        return_value={"role": "assistant", "content": "", "tool_calls": []}
    )
    agent._execute_tool_batch = AsyncMock()
    agent._build_all_tools = MagicMock(return_value=[])
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent._visible_known_tool_names = MagicMock(return_value=set())
    agent._make_inline_tool_executor = MagicMock(return_value=None)
    # Identity prune so the assertion sees exactly what the handler built.
    agent._prune_orchestrator_messages = MagicMock(
        side_effect=lambda msgs, _tools, **_kw: msgs
    )
    return agent


class TestSignatures:
    def test_both_handlers_accept_conversation_history(self):
        for name in (
            "_handle_orchestrator_response",
            "_handle_orchestrator_response_streaming",
        ):
            sig = inspect.signature(getattr(OrchestratorEngineMixin, name))
            assert "conversation_history" in sig.parameters, (
                f"{name} cannot receive the turn's history"
            )
            assert sig.parameters["conversation_history"].default is None


@pytest.mark.asyncio
class TestNonStreamingHistoryContinuation:
    async def test_history_continuation_reaches_generate_with_messages(self):
        agent = _base_agent()
        agent.llm_service = MagicMock()
        agent.llm_service.generate_with_messages = AsyncMock(
            return_value=LLMResponse(content="done", tool_calls=None)
        )

        handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)
        await handler(
            response=_tool_call_response(),
            feature_tools=[],
            system_prompt="sys",
            force_local_only=False,
            effective_model="claude-opus-5",
            user_message="what codeword did I give you?",
            session_id="s-1",
            conversation_history=HISTORY,
        )

        agent.llm_service.generate_with_messages.assert_awaited()
        sent = agent.llm_service.generate_with_messages.await_args.kwargs["messages"]

        assert sent[0]["role"] == "system"
        # The prior turns are present, in order, between system and the
        # current user turn — not merely present somewhere.
        assert sent[1:3] == HISTORY
        assert sent[3] == {
            "role": "user",
            "content": "what codeword did I give you?",
        }

    async def test_history_continuation_omitted_when_no_history(self):
        """No history (e.g. first turn of a session) keeps the legacy shape."""
        agent = _base_agent()
        agent.llm_service = MagicMock()
        agent.llm_service.generate_with_messages = AsyncMock(
            return_value=LLMResponse(content="done", tool_calls=None)
        )

        handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(agent)
        await handler(
            response=_tool_call_response(),
            feature_tools=[],
            system_prompt="sys",
            force_local_only=False,
            effective_model="claude-opus-5",
            user_message="hi",
            session_id="s-1",
        )

        sent = agent.llm_service.generate_with_messages.await_args.kwargs["messages"]
        assert sent[0]["role"] == "system"
        assert sent[1] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
class TestStreamingHistoryContinuation:
    async def test_history_continuation_reaches_stream_with_tool_detection(self):
        agent = _base_agent()

        captured = {}

        async def _fake_stream(*_args, **kwargs):
            captured["messages"] = kwargs["messages"]
            yield LLMResponse(content="done", tool_calls=None)

        agent.llm_service = MagicMock()
        agent.llm_service.stream_with_tool_detection = _fake_stream
        agent.is_request_cancelled = MagicMock(return_value=False)

        handler = (
            OrchestratorEngineMixin._handle_orchestrator_response_streaming.__get__(
                agent
            )
        )
        async for _chunk in handler(
            response=_tool_call_response(),
            feature_tools=[],
            system_prompt="sys",
            force_local_only=False,
            effective_model="claude-opus-5",
            user_message="what codeword did I give you?",
            tool_events=[],
            tool_results=[],
            session_id="s-1",
            conversation_history=HISTORY,
        ):
            pass

        sent = captured["messages"]
        assert sent[0]["role"] == "system"
        assert sent[1:3] == HISTORY
        assert sent[3] == {
            "role": "user",
            "content": "what codeword did I give you?",
        }


class TestPruneShedsHistoryUnderPressure:
    """Replaying history made tool truncation an insufficient pressure valve.

    A long session's history can exceed the ceiling on its own, with no
    oversized tool result left to reclaim. The prune must then shed the oldest
    prior turns — never ``system``, never the current user turn.
    """

    @staticmethod
    def _prune(agent_limit):
        agent = MagicMock()
        agent.llm_service = MagicMock()
        agent.llm_service._context_limit = agent_limit
        return OrchestratorEngineMixin._prune_orchestrator_messages.__get__(agent)

    def test_oldest_history_is_shed_when_tool_truncation_is_not_enough(self):
        # Long history, and a tool result too small to reclaim anything.
        long_history = [
            {"role": "user", "content": f"turn-{i} " + "y" * 400}
            for i in range(10)
        ]
        messages = (
            [{"role": "system", "content": "sys"}]
            + long_history
            + [{"role": "user", "content": "current ask"}]
            + [
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "call_1", "content": "small"},
            ]
        )
        prune = self._prune(agent_limit=500)

        result = prune(
            messages, [],
            protected_prefix=1 + len(long_history) + 1,
            history_len=len(long_history),
        )

        kept_history = [m for m in result if m["content"].startswith("turn-")]
        # Something was shed...
        assert len(kept_history) < len(long_history)
        # ...and it was the OLDEST, so the newest survives.
        assert kept_history == long_history[len(long_history) - len(kept_history):]
        # Structural messages are untouched.
        assert result[0] == {"role": "system", "content": "sys"}
        assert {"role": "user", "content": "current ask"} in result
        assert result[-1]["role"] == "tool"

    def test_history_is_kept_when_it_already_fits(self):
        messages = (
            [{"role": "system", "content": "sys"}]
            + HISTORY
            + [{"role": "user", "content": "current ask"}]
        )
        prune = self._prune(agent_limit=131072)

        result = prune(
            messages, [], protected_prefix=4, history_len=len(HISTORY)
        )
        assert result == messages

    def test_defaults_preserve_the_legacy_no_history_shape(self):
        sig = inspect.signature(
            OrchestratorEngineMixin._prune_orchestrator_messages
        )
        assert sig.parameters["protected_prefix"].default == 2
        assert sig.parameters["history_len"].default == 0


class TestPrefixBoundsSurviveShedding:
    """Bounds must be re-derived per pass, not cached from seed time.

    A multi-iteration tool turn prunes between iterations. Once a prune sheds
    replayed history the array is shorter, so seed-time indexes reach past the
    current user turn — the next pass would treat this turn's own request and
    ``tool_use`` as sheddable history and orphan the matching ``tool_result``.
    """

    def test_bounds_track_history_that_was_already_shed(self):
        history = [{"role": "user", "content": f"turn-{i}"} for i in range(6)]
        user_turn = {"role": "user", "content": "current ask"}
        history_ids = frozenset(id(m) for m in history)

        seeded = [{"role": "system", "content": "sys"}] + history + [user_turn]
        prefix, hist_len = OrchestratorEngineMixin._prefix_bounds(
            seeded, history_ids, True
        )
        assert (prefix, hist_len) == (8, 6)
        assert seeded[prefix - 1] is user_turn

        # Simulate a prune having shed the 4 oldest replayed turns.
        after_shed = (
            [{"role": "system", "content": "sys"}]
            + history[4:]
            + [user_turn]
            + [{"role": "assistant", "content": "", "tool_calls": []},
               {"role": "tool", "tool_call_id": "c1", "content": "r"}]
        )
        prefix2, hist_len2 = OrchestratorEngineMixin._prefix_bounds(
            after_shed, history_ids, True
        )
        assert (prefix2, hist_len2) == (4, 2)
        # The boundary still lands exactly on the current user turn, so the
        # tool exchange stays in the prunable middle.
        assert after_shed[prefix2 - 1] is user_turn
        assert after_shed[prefix2]["role"] == "assistant"

    def test_bounds_without_a_user_message(self):
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        history_ids = frozenset(id(m) for m in history)
        seeded = [{"role": "system", "content": "sys"}] + history
        assert OrchestratorEngineMixin._prefix_bounds(
            seeded, history_ids, False
        ) == (3, 2)

    def test_seed_reports_history_identity(self):
        history = [{"role": "user", "content": "a"}]
        messages, history_ids = OrchestratorEngineMixin._seed_orchestrator_messages(
            "sys", history, "ask", "TEST"
        )
        assert messages[1] is history[0]
        assert history_ids == frozenset({id(history[0])})
