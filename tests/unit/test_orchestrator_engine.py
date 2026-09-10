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
from unittest.mock import AsyncMock, MagicMock, patch

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
    agent._execute_tool_batch_at_stop_boundary = (
        OrchestratorEngineMixin._execute_tool_batch_at_stop_boundary.__get__(agent)
    )
    agent._build_all_tools = MagicMock(return_value=[])
    agent._visible_features_by_tool_name = MagicMock(return_value={})
    agent._visible_known_tool_names = MagicMock(return_value=set())
    agent._known_tool_names = MagicMock(return_value=set())
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

    def test_stale_history_is_shed_before_the_live_tool_result_is_destroyed(self):
        """Priority: lose old turns before this turn's tool output.

        The synthesis call exists to report what the tool returned. Truncating
        that result while a stale exchange would have freed the same bytes
        leaves the answer without the lookup it was asked for.
        """
        history = [
            {"role": "user", "content": f"old-{i} " + "y" * 500}
            for i in range(6)
        ]
        tool_result = {
            "role": "tool", "tool_call_id": "c1", "content": "R" * 1200,
        }
        messages = (
            [{"role": "system", "content": "sys"}]
            + history
            + [{"role": "user", "content": "current ask"}]
            + [{"role": "assistant", "content": "", "tool_calls": []}, tool_result]
        )
        prune = self._prune(agent_limit=900)

        result = prune(
            messages, [],
            protected_prefix=1 + len(history) + 1,
            history_len=len(history),
        )

        # The tool result the answer depends on survives intact...
        assert result[-1]["content"] == "R" * 1200, (
            "live tool output was truncated while stale history was still present"
        )
        # ...because stale history paid for it.
        kept_history = [m for m in result if m["content"].startswith("old-")]
        assert len(kept_history) < len(history)

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

    def test_shed_advances_to_a_user_boundary(self):
        """A shed must not leave a detached assistant reply at the head.

        History alternates user/assistant. Dropping an odd number of rows
        strands an answer with no question — and providers reject a replayed
        span that opens on an assistant turn.
        """
        # Uneven row sizes so the fit-loop stops at both odd and even counts;
        # a single fixed limit can land on a user boundary by luck.
        history = []
        for i in range(8):
            history.append(
                {"role": "user", "content": f"q{i} " + "z" * (400 if i == 0 else 90)}
            )
            history.append({"role": "assistant", "content": f"a{i} " + "z" * 90})

        base = (
            [{"role": "system", "content": "sys"}]
            + history
            + [{"role": "user", "content": "current ask"}]
            + [
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "c1", "content": "small"},
            ]
        )
        agent = MagicMock()
        agent.llm_service = MagicMock(spec=[])
        prune = OrchestratorEngineMixin._prune_orchestrator_messages.__get__(agent)

        checked = 0
        for limit in range(120, 900, 7):
            result = prune(
                [dict(m) for m in base], [],
                context_limit=limit,
                protected_prefix=1 + len(history) + 1,
                history_len=len(history),
            )
            kept = [
                m for m in result[1:]
                if m.get("content", "")[:1] in ("q", "a")
            ]
            if not kept:
                continue
            checked += 1
            assert kept[0]["role"] == "user", (
                f"at context_limit={limit} the replayed span opens on a "
                f"detached {kept[0]['role']} turn: {kept[0]['content'][:20]!r}"
            )
        assert checked > 5, "sweep never exercised a partial shed"


class TestContextLimitResolution:
    """Pruning must size against the model actually serving the turn."""

    def test_limit_comes_from_the_model_when_service_has_none(self):
        service = MagicMock(spec=[])  # no _context_limit attribute
        resolve = OrchestratorEngineMixin._resolve_orchestrator_context_limit

        from kestrel_sovereign.agent.token_counter import get_token_counter

        expected = get_token_counter("claude-opus-5").get_context_limit()
        assert resolve(service, "claude-opus-5") == expected
        # A real model window must not silently collapse to the legacy default
        # unless that genuinely IS its window.
        assert expected > 0

    def test_explicit_service_limit_wins(self):
        service = MagicMock()
        service._context_limit = 4096
        assert OrchestratorEngineMixin._resolve_orchestrator_context_limit(
            service, "claude-opus-5"
        ) == 4096

    def test_falls_back_to_the_active_route_when_the_turn_names_no_model(self):
        """``effective_model`` is None for every wallet-less agent.

        ``check_solvency()`` returns None when there is no wallet, so most
        turns arrive here with no model at all. Dropping to the default then
        would mis-size the majority of continuations.
        """
        from kestrel_sovereign.agent.token_counter import get_token_counter

        service = MagicMock(spec=["get_active_model_id"])
        service.get_active_model_id.return_value = "claude-opus-5"

        assert OrchestratorEngineMixin._resolve_orchestrator_context_limit(
            service, None
        ) == get_token_counter("claude-opus-5").get_context_limit()
        service.get_active_model_id.assert_called_once()

    def test_route_qualified_selection_beats_the_bare_model_id(self):
        """A route-level cap must not be lost to the bare model id.

        ``get_active_model_id()`` returns just the model, so a plan route
        serving a smaller window than the model's own would be sized against
        the model's full window.
        """
        service = MagicMock(
            spec=["get_active_model_selection", "get_active_model_id"]
        )
        service.get_active_model_selection.return_value = {
            "model": "anthropic:plan/claude-opus-5"
        }
        service.get_active_model_id.return_value = "claude-opus-5"

        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter"
        ) as counter:
            counter.return_value.get_context_limit.return_value = 4242
            assert OrchestratorEngineMixin._resolve_orchestrator_context_limit(
                service, None
            ) == 4242

        counter.assert_called_once_with("anthropic:plan/claude-opus-5")

    def test_bare_override_still_picks_up_the_route_cap(self):
        """A bare/provider-only override names the model, not the route.

        A route can carry a per-turn cap far below the model's own window, so
        a truthy-but-unqualified override must not bypass the route lookup.
        """
        service = MagicMock(spec=["get_active_model_selection", "get_active_model_id"])
        service.get_active_model_selection.return_value = {
            "model": "openai:plan/gpt-5.5"
        }
        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter"
        ) as counter:
            counter.return_value.get_context_limit.return_value = 20480
            assert OrchestratorEngineMixin._resolve_orchestrator_context_limit(
                service, "openai/gpt-5.5"
            ) == 20480
        counter.assert_called_once_with("openai:plan/gpt-5.5")

    def test_a_different_override_model_is_respected_not_overridden(self):
        """The route cap only applies when it IS this model's route."""
        service = MagicMock(spec=["get_active_model_selection", "get_active_model_id"])
        service.get_active_model_selection.return_value = {
            "model": "openai:plan/gpt-5.5"
        }
        with patch(
            "kestrel_sovereign.agent.token_counter.get_token_counter"
        ) as counter:
            counter.return_value.get_context_limit.return_value = 999
            OrchestratorEngineMixin._resolve_orchestrator_context_limit(
                service, "claude-opus-5"
            )
        counter.assert_called_once_with("claude-opus-5")

    def test_falls_back_only_when_the_route_is_unresolved(self):
        from kestrel_sovereign.agent.orchestrator_engine import (
            _DEFAULT_ORCHESTRATOR_CONTEXT_LIMIT,
        )

        service = MagicMock(spec=["get_active_model_id"])
        service.get_active_model_id.return_value = "auto"
        assert OrchestratorEngineMixin._resolve_orchestrator_context_limit(
            service, None
        ) == _DEFAULT_ORCHESTRATOR_CONTEXT_LIMIT

        bare = MagicMock(spec=[])
        assert OrchestratorEngineMixin._resolve_orchestrator_context_limit(
            bare, None
        ) == _DEFAULT_ORCHESTRATOR_CONTEXT_LIMIT


@pytest.mark.asyncio
class TestContinuationReplaysRenderedPrompt:
    """The continuation's last user turn must be the bytes the first call sent.

    The streaming path passes RAW user input as ``user_message`` because that
    string also becomes a dispatched subagent's "User's original request".
    The provider prefix needs the RENDERED prompt (memories + RAG + lazy
    hint), so the two travel separately.
    """

    async def test_rendered_prompt_is_used_over_raw_user_message(self):
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
        async for _ in handler(
            response=_tool_call_response(),
            feature_tools=[],
            system_prompt="sys",
            force_local_only=False,
            effective_model="claude-opus-5",
            user_message="raw ask",
            tool_events=[],
            tool_results=[],
            session_id="s-1",
            conversation_history=HISTORY,
            continuation_user_content="RENDERED: memories+rag\n\nraw ask",
        ):
            pass

        sent = captured["messages"]
        assert sent[3] == {
            "role": "user",
            "content": "RENDERED: memories+rag\n\nraw ask",
        }, "continuation lost this turn's retrieved context"

    async def test_falls_back_to_user_message_when_not_supplied(self):
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
            user_message="plain ask",
            session_id="s-1",
            conversation_history=HISTORY,
        )
        sent = agent.llm_service.generate_with_messages.await_args.kwargs["messages"]
        assert sent[3] == {"role": "user", "content": "plain ask"}


class TestPruneThreadsTheTurnModel:
    def test_prune_consults_the_resolver_with_the_turn_model(self):
        agent = MagicMock()
        agent.llm_service = MagicMock(spec=[])
        prune = OrchestratorEngineMixin._prune_orchestrator_messages.__get__(agent)

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 40000},
        ]
        with patch.object(
            OrchestratorEngineMixin,
            "_resolve_orchestrator_context_limit",
            return_value=100,
        ) as spy:
            result = prune(messages, [], model="some-route/some-model")

        spy.assert_called_once_with(agent.llm_service, "some-route/some-model")
        # The resolved (tiny) window must actually drive pruning — the 131072
        # default would have waved this payload straight through.
        assert "truncated" in result[-1]["content"].lower()


@pytest.mark.asyncio
class TestRepairTurnIsPruned:
    """A repair turn calls the provider directly, with no pruning of its own.

    Harmless while the array was ``[system, user]``; once history is replayed a
    turn that fit the first call can overflow on the repair call.
    """

    async def test_repair_path_prunes_the_history_bearing_array(self):
        agent = _base_agent()
        agent.llm_service = MagicMock()
        agent.llm_service.generate_with_messages = AsyncMock(
            return_value=LLMResponse(content="done", tool_calls=None)
        )
        agent._repair_premature_turn_yield = AsyncMock(
            return_value=LLMResponse(content="repaired", tool_calls=None)
        )
        pruned = MagicMock(side_effect=lambda msgs, _tools, **_kw: msgs)
        agent._prune_orchestrator_messages = pruned

        # No tool calls, but text that trips the unfinished-tool-work guard.
        with patch.object(
            OrchestratorEngineMixin,
            "_signals_unfinished_tool_work",
            return_value=True,
        ):
            handler = OrchestratorEngineMixin._handle_orchestrator_response.__get__(
                agent
            )
            await handler(
                response=LLMResponse(content="let me check that", tool_calls=None),
                feature_tools=[{"function": {"name": "t"}}],
                system_prompt="sys",
                force_local_only=False,
                effective_model="claude-opus-5",
                user_message="ask",
                session_id="s-1",
                conversation_history=HISTORY,
            )

        pruned.assert_called_once()
        kwargs = pruned.call_args.kwargs
        # Sized against the turn's model, with the prefix bounds that describe
        # the replayed history.
        assert kwargs["model"] == "claude-opus-5"
        assert kwargs["history_len"] == len(HISTORY)
        assert kwargs["protected_prefix"] == 1 + len(HISTORY) + 1
        agent._repair_premature_turn_yield.assert_awaited()


class TestSeedIdentity:
    def test_seed_reports_history_identity(self):
        history = [{"role": "user", "content": "a"}]
        messages, history_ids = OrchestratorEngineMixin._seed_orchestrator_messages(
            "sys", history, "ask", "TEST"
        )
        assert messages[1] is history[0]
        assert history_ids == frozenset({id(history[0])})
