"""
Unit tests for ``OrchestratorEngineMixin.execute_named_tool``.

The public ``execute_named_tool`` entry point is the contract that
non-chat transports (voice realtime, MCP, future A2A surfaces) use to
invoke an agent's tools without bypassing the PRE/POST_TOOL_USE hook
stack.  Issue #1314 — the voice realtime path was calling
``tool.execute(**args)`` directly, which skipped governance.

These tests cover:

* PRE_TOOL_USE ``PermissionDecision.DENY`` is honored — the tool is NOT
  invoked, and an envelope ``{"success": False, "error": ...}`` is
  returned so the caller can surface the denial to the model.
* PRE_TOOL_USE ``updated_input`` is applied before execution (matches
  chat-path behavior).
* POST_TOOL_USE hooks fire on success.
* Unknown tool name raises ``ValueError`` (not an envelope) so callers
  can distinguish "tool exists but blocked" from "tool doesn't exist".
* The resolver walks ``self.features`` directly rather than relying on
  the explored-tools cache, so external transports work on cold features.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from kestrel_sdk.hooks.base import (
    HookEvent,
    HookInput,
    HookOutput,
    PermissionDecision,
)
from kestrel_sovereign.agent.orchestrator_engine import (
    OrchestratorEngineMixin,
    ToolNotRegisteredError,
)


class _FakeTool:
    """Minimal stand-in for an AgentTool: name + async execute()."""

    def __init__(self, name: str, *, returns: Any = None, side_effect=None):
        self.name = name
        self._returns = returns
        self._side_effect = side_effect
        self.calls: list[dict] = []

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self._side_effect is not None:
            raise self._side_effect
        return self._returns


class _FakeFeature:
    """Minimal feature shell exposing ``get_tools`` and ``tool_name``."""

    def __init__(self, name: str, tools: list[_FakeTool]):
        self.name = name
        self.tool_name = name
        self._tools = tools

    def get_tools(self):
        return list(self._tools)


class _FakeHooksManager:
    """In-memory hooks manager that models the real ``HookManager`` chain.

    The real manager threads ``HookInput`` through the hook chain — when
    a hook returns ``HookOutput.modify(updated_input=...)`` the manager
    sets ``input.tool_input = updated_input`` on the *next* iteration,
    then ultimately returns a fresh ``HookOutput.allow()`` with no
    ``updated_input`` set.  Callers that want to read MODIFY-accumulated
    args read ``pre_input.tool_input``, not the returned HookOutput.

    This fixture mirrors that behavior so the unit tests don't pass
    against a contract the real manager doesn't honor.  Use
    ``modify_input`` to model a MODIFY hook (mutates the threaded
    input; returned HookOutput is a fresh allow) or ``short_circuit_input``
    to model a single-hook fixture that returns ``updated_input``
    directly on its HookOutput.
    """

    def __init__(
        self,
        *,
        pre_decision: PermissionDecision = PermissionDecision.ALLOW,
        pre_reason: str = "",
        modify_input: dict | None = None,            # threaded MODIFY (real-manager shape)
        short_circuit_input: dict | None = None,      # single-hook updated_input on output
    ):
        self.pre_decision = pre_decision
        self.pre_reason = pre_reason
        self.modify_input = modify_input
        self.short_circuit_input = short_circuit_input
        self.pre_calls: list[HookInput] = []
        self.post_calls: list[HookInput] = []

    async def execute_hooks(self, event: HookEvent, hook_input: HookInput) -> HookOutput:
        assert event is HookEvent.PRE_TOOL_USE
        # Record the input AS IT ARRIVED, not after mutation, so tests
        # can still see the pre-modify args.
        self.pre_calls.append(
            HookInput(
                session_id=hook_input.session_id,
                hook_event_name=hook_input.hook_event_name,
                tool_name=hook_input.tool_name,
                tool_input=dict(hook_input.tool_input or {}),
                feature_name=hook_input.feature_name,
            )
        )
        # Real-manager MODIFY: mutate the threaded input, return fresh allow().
        if self.modify_input is not None:
            hook_input.tool_input = self.modify_input
        return HookOutput(
            permission_decision=self.pre_decision,
            permission_reason=self.pre_reason,
            updated_input=self.short_circuit_input,
        )

    async def execute_hooks_parallel(self, event: HookEvent, hook_input: HookInput) -> None:
        assert event is HookEvent.POST_TOOL_USE
        self.post_calls.append(hook_input)


class _MinimalOrchestrator(OrchestratorEngineMixin):
    """Just enough orchestrator surface to exercise ``execute_named_tool``."""

    def __init__(self, features: dict, hooks_manager: _FakeHooksManager):
        self.features = features
        self.hooks_manager = hooks_manager


@pytest.fixture
def fake_tool():
    return _FakeTool("send_email", returns={"success": True, "message_id": "abc"})


@pytest.fixture
def agent_with_tool(fake_tool):
    feature = _FakeFeature("email", [fake_tool])
    return _MinimalOrchestrator(
        features={"email": feature},
        hooks_manager=_FakeHooksManager(),
    )


class TestExecuteNamedToolGovernance:
    @pytest.mark.asyncio
    async def test_runs_tool_and_fires_pre_and_post_hooks(self, agent_with_tool, fake_tool):
        result = await agent_with_tool.execute_named_tool(
            "send_email",
            {"to": "x@example.com"},
            session_id="voice-session-1",
            source="voice_realtime",
        )

        # Tool actually executed.
        assert fake_tool.calls == [{"to": "x@example.com"}]
        assert result == {"success": True, "message_id": "abc"}

        # PRE_TOOL_USE fired with correct context.
        assert len(agent_with_tool.hooks_manager.pre_calls) == 1
        pre = agent_with_tool.hooks_manager.pre_calls[0]
        assert pre.tool_name == "send_email"
        assert pre.feature_name == "email"
        assert pre.session_id == "voice-session-1"
        assert pre.tool_input == {"to": "x@example.com"}

        # POST_TOOL_USE fired with the tool response.
        assert len(agent_with_tool.hooks_manager.post_calls) == 1
        post = agent_with_tool.hooks_manager.post_calls[0]
        assert post.tool_name == "send_email"
        assert post.feature_name == "email"

    @pytest.mark.asyncio
    async def test_pre_hook_deny_blocks_execution(self, fake_tool):
        """The headline #1314 behavior — a DENY decision must NOT run the tool."""
        feature = _FakeFeature("email", [fake_tool])
        agent = _MinimalOrchestrator(
            features={"email": feature},
            hooks_manager=_FakeHooksManager(
                pre_decision=PermissionDecision.DENY,
                pre_reason="user revoked email permission for this session",
            ),
        )

        result = await agent.execute_named_tool(
            "send_email",
            {"to": "x@example.com"},
            session_id="voice-session-2",
            source="voice_realtime",
        )

        # Tool NOT executed.
        assert fake_tool.calls == []
        # Permission-denied envelope returned, NOT the tool result.
        assert result == {
            "success": False,
            "error": "Permission denied: user revoked email permission for this session",
        }
        # POST hook does NOT fire on a denied call.
        assert agent.hooks_manager.post_calls == []

    @pytest.mark.asyncio
    async def test_real_manager_modify_chain_reaches_the_tool(self, fake_tool):
        """The real ``HookManager`` mutates ``input.tool_input`` when a
        MODIFY hook fires, then returns a fresh ``allow()`` HookOutput
        with NO ``updated_input``.  Reading args back from
        ``HookOutput.updated_input`` alone would silently no-op real
        redactor hooks — codex caught this on round-2 review of #1314.

        Pins that the dispatcher reads back from the threaded
        ``HookInput.tool_input`` so the production shape works.
        """
        feature = _FakeFeature("email", [fake_tool])
        agent = _MinimalOrchestrator(
            features={"email": feature},
            hooks_manager=_FakeHooksManager(
                modify_input={"to": "redacted@example.com"},
            ),
        )

        await agent.execute_named_tool(
            "send_email",
            {"to": "leaked@example.com"},
            session_id="voice-session-3",
            source="voice_realtime",
        )

        # Tool ran with the REDACTED args — without this, a real
        # redactor hook would be a no-op at execution time and PII
        # would leak past it.
        assert fake_tool.calls == [{"to": "redacted@example.com"}]
        # POST hook sees the post-modify args so audit logs match reality.
        assert len(agent.hooks_manager.post_calls) == 1
        assert agent.hooks_manager.post_calls[0].tool_input == {"to": "redacted@example.com"}

    @pytest.mark.asyncio
    async def test_single_hook_short_circuit_updated_input_also_reaches_the_tool(self, fake_tool):
        """The other shape: a single hook short-circuits the chain with
        ``HookOutput(updated_input=...)`` rather than threading the
        modification through.  Both shapes must work — fixtures and
        production hooks should both apply rewrites at execution time.
        """
        feature = _FakeFeature("email", [fake_tool])
        agent = _MinimalOrchestrator(
            features={"email": feature},
            hooks_manager=_FakeHooksManager(
                short_circuit_input={"to": "redacted@example.com"},
            ),
        )
        await agent.execute_named_tool(
            "send_email", {"to": "leaked@example.com"},
            session_id="s", source="voice_realtime",
        )
        assert fake_tool.calls == [{"to": "redacted@example.com"}]

    @pytest.mark.asyncio
    async def test_non_dict_args_rejected_before_execution(self, fake_tool):
        """Non-dict ``args`` would crash at ``**effective_args``.  The
        chat path runs ``validate_tool_arguments`` before dispatch; the
        external path must too, or realtime/MCP/A2A callers can crash
        the worker by passing a list/string instead of a dict.  Codex
        round-4 caught this on #1314.
        """
        feature = _FakeFeature("email", [fake_tool])
        agent = _MinimalOrchestrator(
            features={"email": feature},
            hooks_manager=_FakeHooksManager(),
        )

        result = await agent.execute_named_tool(
            "send_email",
            ["not", "a", "dict"],  # type: ignore[arg-type] — deliberately wrong
            session_id="s",
            source="voice_realtime",
        )

        # Tool NOT executed; structured error returned.
        assert fake_tool.calls == []
        assert result["success"] is False
        assert "must be a dict" in result["error"]

    @pytest.mark.asyncio
    async def test_oversize_string_arg_rejected_before_execution(self, fake_tool):
        """``validate_tool_arguments`` enforces ``MAX_TOOL_ARG_LENGTH``
        on every string field.  External transports must honor the
        same guardrail — otherwise an unbounded prompt-injection
        payload from the realtime model reaches the tool unchecked.
        """
        from kestrel_sovereign.security.input_guardrails import (
            MAX_TOOL_ARG_LENGTH,
        )

        feature = _FakeFeature("email", [fake_tool])
        agent = _MinimalOrchestrator(
            features={"email": feature},
            hooks_manager=_FakeHooksManager(),
        )

        # One byte over the limit.
        too_long = "x" * (MAX_TOOL_ARG_LENGTH + 1)
        result = await agent.execute_named_tool(
            "send_email",
            {"to": too_long},
            session_id="s",
            source="voice_realtime",
        )

        assert fake_tool.calls == []
        assert result["success"] is False
        assert "exceeds maximum length" in result["error"]

    @pytest.mark.asyncio
    async def test_hook_rewrite_to_empty_dict_is_honored(self):
        """An empty-dict rewrite is a legitimate hook decision — a
        constraint hook clearing all sensitive args, or a redactor
        stripping every field.  Truthiness fallbacks (``updated_input or
        args``) would silently discard that rewrite and run the tool
        with the original sensitive payload.  Codex round-3 caught
        this on #1314.
        """
        tool = _FakeTool("noop", returns="ok")
        feature = _FakeFeature("noop_feat", [tool])
        agent = _MinimalOrchestrator(
            features={"noop_feat": feature},
            hooks_manager=_FakeHooksManager(short_circuit_input={}),
        )

        await agent.execute_named_tool(
            "noop",
            {"sensitive": "data", "more": "stuff"},
            session_id="s",
            source="voice_realtime",
        )

        # Tool ran with the EMPTY dict the hook returned, not the
        # original sensitive payload.
        assert tool.calls == [{}]

    @pytest.mark.asyncio
    async def test_pre_hook_ask_decision_is_blocking(self, fake_tool):
        """A PRE hook returning ``PermissionDecision.ASK`` is routing the
        call to an approval queue — the tool must NOT execute until
        approval lands.  Treating ASK as fall-through would bypass the
        approval gate for any approval-protected tool invoked via this
        path.  Codex flagged this on round-2 review of #1314.
        """
        feature = _FakeFeature("email", [fake_tool])
        agent = _MinimalOrchestrator(
            features={"email": feature},
            hooks_manager=_FakeHooksManager(
                pre_decision=PermissionDecision.ASK,
                pre_reason="this tool requires explicit user approval",
            ),
        )

        result = await agent.execute_named_tool(
            "send_email",
            {"to": "x@example.com"},
            session_id="voice-session-ask",
            source="voice_realtime",
        )

        # Tool NOT executed.
        assert fake_tool.calls == []
        # Approval-required envelope returned — distinct from DENY so
        # the caller's UX can route the user to the approval flow.
        assert result == {
            "success": False,
            "error": "Approval required: this tool requires explicit user approval",
        }
        # POST hook does NOT fire on a not-yet-approved call.
        assert agent.hooks_manager.post_calls == []

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_tool_not_registered(self, agent_with_tool):
        """Distinct from "denied" — bug, not policy.  Uses the dedicated
        ``ToolNotRegisteredError`` subclass so callers can distinguish
        "tool doesn't exist" from a ``ValueError`` raised by a tool's
        own validation (see ``test_tool_value_error_is_not_caught_as_unregistered``).
        """
        with pytest.raises(ToolNotRegisteredError, match="not registered with any enabled feature"):
            await agent_with_tool.execute_named_tool(
                "nonexistent_tool",
                {},
                session_id="s",
                source="voice_realtime",
            )

    @pytest.mark.asyncio
    async def test_tool_value_error_propagates_unwrapped(self):
        """A ``ValueError`` raised by the tool's own validation logic
        propagates out of ``execute_named_tool`` as a plain
        ``ValueError`` — NOT as ``ToolNotRegisteredError``.  Callers can
        ``except ToolNotRegisteredError`` to handle "tool doesn't exist"
        and ``except ValueError`` for tool-side validation errors.
        Codex round-2 caught the prior version of this — without the
        distinction, a tool raising ``ValueError("invalid recipient")``
        would be misreported as "tool not found" and the realtime model
        would correct the wrong problem.
        """
        bad_tool = _FakeTool("send_email", side_effect=ValueError("invalid recipient: not an email"))
        feature = _FakeFeature("email", [bad_tool])
        agent = _MinimalOrchestrator(
            features={"email": feature},
            hooks_manager=_FakeHooksManager(),
        )

        with pytest.raises(ValueError, match="invalid recipient"):
            await agent.execute_named_tool(
                "send_email", {"to": "bad"},
                session_id="s", source="voice_realtime",
            )
        # And critically — that ValueError is NOT a ToolNotRegisteredError.
        # We verify by re-running and confirming the exception type.
        try:
            await agent.execute_named_tool(
                "send_email", {"to": "bad"},
                session_id="s", source="voice_realtime",
            )
        except ValueError as exc:
            assert not isinstance(exc, ToolNotRegisteredError)

    @pytest.mark.asyncio
    async def test_resolver_walks_features_not_explored_cache(self, fake_tool):
        """External transports must work on features that have never been
        subagent-dispatched (the ``_direct_tools`` / ``_explored_features``
        cache is only populated after the chat orchestrator first dispatches
        into a feature).  The resolver must consult ``self.features``
        directly, not the cache."""
        feature = _FakeFeature("email", [fake_tool])
        agent = _MinimalOrchestrator(
            features={"email": feature},
            hooks_manager=_FakeHooksManager(),
        )
        # Crucially, _direct_tools / _explored_features attributes are NOT
        # populated — only self.features.  If the resolver consulted the
        # cache, the next call would raise ValueError.
        assert not hasattr(agent, "_direct_tools")

        result = await agent.execute_named_tool(
            "send_email", {"to": "x@example.com"},
            session_id="s", source="voice_realtime",
        )
        assert result == {"success": True, "message_id": "abc"}

    @pytest.mark.asyncio
    async def test_first_match_wins_across_features(self):
        """Tool names are expected globally unique within an agent's
        enabled features; if they collide the resolver returns the first
        match (deterministic behavior — iteration order of self.features
        is the agent's registration order)."""
        tool_a = _FakeTool("shared_name", returns="from_a")
        tool_b = _FakeTool("shared_name", returns="from_b")
        agent = _MinimalOrchestrator(
            features={
                "feat_a": _FakeFeature("feat_a", [tool_a]),
                "feat_b": _FakeFeature("feat_b", [tool_b]),
            },
            hooks_manager=_FakeHooksManager(),
        )

        result = await agent.execute_named_tool(
            "shared_name", {},
            session_id="s", source="voice_realtime",
        )
        assert result == "from_a"

    @pytest.mark.asyncio
    async def test_broken_feature_does_not_block_lookup(self, fake_tool):
        """A single feature whose ``get_tools()`` raises must not prevent
        the resolver from finding tools on the other features."""
        class _BrokenFeature:
            name = "broken"
            tool_name = "broken"
            def get_tools(self):
                raise RuntimeError("intentionally broken for this test")

        agent = _MinimalOrchestrator(
            features={
                "broken": _BrokenFeature(),
                "email": _FakeFeature("email", [fake_tool]),
            },
            hooks_manager=_FakeHooksManager(),
        )
        result = await agent.execute_named_tool(
            "send_email", {"to": "x@example.com"},
            session_id="s", source="voice_realtime",
        )
        assert result == {"success": True, "message_id": "abc"}
