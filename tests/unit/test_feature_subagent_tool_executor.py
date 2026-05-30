"""Regression tests for ``Feature.execute_as_subagent`` threading a
``tool_executor`` to the LLM service (#1461 follow-up).

The codex app-server adapter (openai:plan, gpt-5.5) executes tool calls
INSIDE the LLM turn via ``item/tool/call`` RPC and blocks until the
result arrives. Without a ``tool_executor`` callback wired through,
codex raises ``CodexAppServerError("requires a tool_executor callback
when tools are provided")`` at the provider layer.

Before this fix, ``Feature.execute_as_subagent`` called
``llm_service.generate(tools=...)`` directly — ``generate()`` /
``get_response()`` don't thread ``tool_executor`` through, so EVERY
subagent dispatch on an openai:plan-routed agent failed at the
provider layer. Emma's "memory_feature failed at the provider layer"
narration for multiple days is exactly this bug.

These tests pin:
  - subagent dispatch routes through ``generate_with_messages``
    (the only entrypoint that threads ``tool_executor``)
  - a feature-scoped inline executor is provided when tools exist
  - the executor returns proper success/error envelopes
  - the executor refuses tools outside this feature's palette
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.features.base import Feature
from kestrel_sdk.tools.base import ToolCategory, ToolSchema


class _FakeTool:
    def __init__(self, name: str, returns):
        self.name = name
        self.schema = ToolSchema(
            name=name,
            description=f"fake tool {name}",
            category=ToolCategory.UTILITY,
        )
        self._returns = returns
        self.executed_with: dict = {}

    async def execute(self, **kwargs):
        self.executed_with = dict(kwargs)
        if isinstance(self._returns, Exception):
            raise self._returns
        return self._returns


class _ProbeFeature(Feature):
    """Minimal concrete Feature for testing the executor wiring without
    pulling in a full feature stack."""

    @property
    def tool_description(self) -> str:
        return "probe"

    async def initialize(self):
        return None

    def get_tools(self):
        return self._fake_tools


def _make_feature_with_agent_capture(
    fake_tools: list[_FakeTool],
) -> tuple[_ProbeFeature, dict]:
    """Return a probe feature whose agent.llm_service.generate_with_messages
    captures the call kwargs into the returned dict."""
    captured: dict = {}

    class _ProbeLLMService:
        async def generate_with_messages(self, **kwargs):
            captured.update(kwargs)
            # Return a no-tool-call response so the subagent's tool loop
            # exits cleanly.
            from kestrel_sdk.llm.adapter import LLMResponse
            return LLMResponse(content="ok", tool_calls=None)

    agent = SimpleNamespace(
        llm_service=_ProbeLLMService(),
        did="did:test:probe",
    )
    feature = _ProbeFeature(agent)
    feature._fake_tools = fake_tools
    return feature, captured


@pytest.mark.asyncio
async def test_subagent_threads_tool_executor_through_to_llm_service():
    """The subagent dispatch must call
    ``llm_service.generate_with_messages`` (the only entrypoint that
    accepts ``tool_executor``), not ``generate`` (which silently drops
    it and forces codex into the no-executor error path)."""
    tool = _FakeTool("search_memory", returns={"success": True, "data": []})
    feature, captured = _make_feature_with_agent_capture([tool])

    result = await feature.execute_as_subagent(task="search for X")

    assert result["success"] is True, (
        f"Subagent must succeed when llm_service returns a non-tool-call "
        f"response. Got {result!r}."
    )
    # The captured kwargs must include both ``tools`` AND
    # ``tool_executor`` — that's the binding contract for codex-routed
    # subagent calls.
    assert "tools" in captured and captured["tools"] is not None
    assert "tool_executor" in captured, (
        "execute_as_subagent must pass tool_executor as a kwarg — "
        "without it the codex app-server adapter raises "
        "CodexAppServerError at the provider layer and EVERY subagent "
        "dispatch fails on openai:plan-routed agents."
    )
    assert callable(captured["tool_executor"])


@pytest.mark.asyncio
async def test_subagent_executor_dispatches_to_feature_tool():
    """The feature-scoped executor must dispatch to the named tool
    from THIS feature's palette and return its result. Tools live
    inside the feature; the executor is the bridge to them."""
    tool = _FakeTool(
        "search_memory",
        returns={"success": True, "data": {"hits": 2}},
    )
    feature, captured = _make_feature_with_agent_capture([tool])
    await feature.execute_as_subagent(task="anything")

    executor = captured["tool_executor"]
    result = await executor("search_memory", {"query": "rescue", "limit": 5})

    assert result == {"success": True, "data": {"hits": 2}}
    assert tool.executed_with == {"query": "rescue", "limit": 5}, (
        f"Executor must forward args verbatim to tool.execute(). Got "
        f"{tool.executed_with!r}."
    )


@pytest.mark.asyncio
async def test_subagent_executor_rejects_tool_outside_palette():
    """Refusing a name not in this feature's palette is the security
    boundary. A subagent shouldn't be able to reach for sibling
    features' tools mid-turn; the codex adapter already constrains
    the visible set, so an unknown name here is a real bug or a
    policy denial and must surface as an error envelope."""
    tool = _FakeTool("allowed", returns={"success": True})
    feature, captured = _make_feature_with_agent_capture([tool])
    await feature.execute_as_subagent(task="anything")

    executor = captured["tool_executor"]
    result = await executor("forbidden", {})

    assert result["success"] is False
    assert "palette" in result["error"].lower(), (
        f"Out-of-palette rejection must mention the palette so the "
        f"diagnostic is unambiguous. Got {result['error']!r}."
    )


@pytest.mark.asyncio
async def test_subagent_executor_surfaces_tool_exception_as_error_envelope():
    """A raised exception inside the tool must NOT propagate up to
    codex (which would treat it as a transport error and abort the
    turn). Convert to a ``{success: False, error: ...}`` envelope so
    codex can return the failure result to the model, which can then
    decide whether to retry / give up / explain."""
    tool = _FakeTool("broken", returns=ValueError("bad input"))
    feature, captured = _make_feature_with_agent_capture([tool])
    await feature.execute_as_subagent(task="anything")

    executor = captured["tool_executor"]
    result = await executor("broken", {"x": 1})

    assert result == {
        "success": False,
        "error": "ValueError: bad input",
    }


@pytest.mark.asyncio
async def test_subagent_no_tools_passes_no_executor():
    """When the feature has no tools (or all are denied), don't
    construct an executor — the LLM call is plain text generation and
    threading a callable would be misleading."""
    feature, captured = _make_feature_with_agent_capture([])

    await feature.execute_as_subagent(task="just talk")

    assert captured["tools"] is None
    assert captured["tool_executor"] is None, (
        "No tools → no executor. The codex adapter is fine with both "
        "absent — it's the (tools-yes, executor-no) combo that errors."
    )


@pytest.mark.asyncio
async def test_subagent_executor_enforces_pre_tool_use_hooks():
    """Codex round 1 P1 on #1461 follow-up: the inline executor must
    gate through PRE_TOOL_USE hooks just like the non-inline
    ``_handle_feature_tool_calls`` path. Without this, the codex
    app-server inline path bypasses SecurityHook denials, approval
    hooks, and argument-redaction hooks — a real privilege-escalation
    hole on every codex-routed subagent dispatch."""
    from kestrel_sdk.hooks.base import (
        HookEvent,
        PermissionDecision,
    )

    tool = _FakeTool("dangerous", returns={"success": True})
    feature, captured = _make_feature_with_agent_capture([tool])

    # Wire a hook manager that DENIES the tool call. Mirrors what a
    # SecurityHook would emit when the agent lacks permission.
    hook_output = SimpleNamespace(
        permission_decision=PermissionDecision.DENY,
        permission_reason="missing permission for dangerous tool",
        updated_input=None,
    )
    hooks_manager = MagicMock()
    hooks_manager.execute_hooks = AsyncMock(return_value=hook_output)
    feature.agent.hooks_manager = hooks_manager

    await feature.execute_as_subagent(task="anything")
    executor = captured["tool_executor"]

    result = await executor("dangerous", {"target": "/etc/passwd"})

    # The hook decision must be honored — the tool must NOT execute.
    assert tool.executed_with == {}, (
        "PRE_TOOL_USE DENY decision must prevent tool.execute() from "
        "running. If this assertion fires, codex-routed subagents can "
        "execute denied tools by routing through the inline path."
    )
    assert result["success"] is False
    assert "PERMISSION DENIED" in result["error"], (
        f"Denial envelope must say PERMISSION DENIED — that's what "
        f"the non-inline path produces (parity required so the LLM "
        f"can't tell which transport ran). Got {result['error']!r}."
    )
    # Hook DID fire (proof the executor consulted the manager).
    hooks_manager.execute_hooks.assert_awaited_once()
    call_args = hooks_manager.execute_hooks.await_args.args
    assert call_args[0] == HookEvent.PRE_TOOL_USE
    assert call_args[1].tool_name == "dangerous"


@pytest.mark.asyncio
async def test_subagent_executor_honors_hook_argument_rewrite():
    """PRE_TOOL_USE hooks may rewrite arguments (PII redaction,
    normalization, constraint enforcement). The inline executor must
    forward the REWRITTEN args to ``tool.execute``, not the original
    — otherwise a redaction hook's protection is silently ignored on
    the codex inline path."""
    from kestrel_sdk.hooks.base import PermissionDecision

    tool = _FakeTool("send_email", returns={"success": True})
    feature, captured = _make_feature_with_agent_capture([tool])

    # Hook rewrites the to/body to redact PII.
    hook_output = SimpleNamespace(
        permission_decision=PermissionDecision.ALLOW,
        permission_reason=None,
        updated_input={"to": "<redacted>", "body": "<redacted>"},
    )
    hooks_manager = MagicMock()
    hooks_manager.execute_hooks = AsyncMock(return_value=hook_output)
    feature.agent.hooks_manager = hooks_manager

    await feature.execute_as_subagent(task="anything")
    executor = captured["tool_executor"]

    await executor("send_email", {
        "to": "victim@example.com", "body": "ssn 123-45-6789",
    })

    assert tool.executed_with == {
        "to": "<redacted>", "body": "<redacted>",
    }, (
        f"Hook updated_input must override the original args before "
        f"tool.execute() runs. Got {tool.executed_with!r} — if this "
        f"shows the original sensitive args, the redaction hook was "
        f"bypassed."
    )


@pytest.mark.asyncio
async def test_subagent_executor_blocks_on_permission_decision_ask():
    """Codex round 2 P1 on #1461 follow-up: ``PermissionDecision.ASK``
    must block tool execution and surface as APPROVAL REQUIRED,
    matching ``execute_named_tool``'s contract. Without this, codex-
    inline subagents bypass human-approval gates that the
    orchestrator-driven path enforces."""
    from kestrel_sdk.hooks.base import PermissionDecision

    tool = _FakeTool("approval_gated", returns={"success": True})
    feature, captured = _make_feature_with_agent_capture([tool])

    hook_output = SimpleNamespace(
        permission_decision=PermissionDecision.ASK,
        permission_reason="needs operator approval",
        updated_input=None,
    )
    hooks_manager = MagicMock()
    hooks_manager.execute_hooks = AsyncMock(return_value=hook_output)
    feature.agent.hooks_manager = hooks_manager

    await feature.execute_as_subagent(task="anything")
    executor = captured["tool_executor"]

    result = await executor("approval_gated", {"target": "production"})

    assert tool.executed_with == {}, (
        "PermissionDecision.ASK must short-circuit before tool.execute() "
        "— approval-gated tools running unattended is a real governance "
        "regression."
    )
    assert result["success"] is False
    assert "APPROVAL REQUIRED" in result["error"], (
        f"ASK envelope must say APPROVAL REQUIRED so the model can tell "
        f"this from a plain DENY. Got {result['error']!r}."
    )


@pytest.mark.asyncio
async def test_subagent_executor_honors_in_place_hook_argument_mutation():
    """Codex round 2 P1 on #1461 follow-up: the real ``HooksManager``
    threads MODIFY-hook rewrites by mutating ``hook_input.tool_input``
    IN PLACE and returns ``HookOutput.allow()`` with
    ``updated_input=None``. Reading only the output field meant
    codex-inline subagent tools ran with the original (pre-rewrite)
    args. The executor must read the post-hook ``tool_input`` first
    so redaction/normalization hooks actually take effect."""
    from kestrel_sdk.hooks.base import PermissionDecision

    tool = _FakeTool("send_message", returns={"success": True})
    feature, captured = _make_feature_with_agent_capture([tool])

    async def _mutate_in_place(event, hook_input):
        # Real-manager pattern: rewrite by mutating in place; return
        # an ALLOW output with updated_input=None.
        hook_input.tool_input = {
            "to": "redacted@example.com",
            "body": "<PII removed>",
        }
        return SimpleNamespace(
            permission_decision=PermissionDecision.ALLOW,
            permission_reason=None,
            updated_input=None,
        )

    hooks_manager = MagicMock()
    hooks_manager.execute_hooks = AsyncMock(side_effect=_mutate_in_place)
    feature.agent.hooks_manager = hooks_manager

    await feature.execute_as_subagent(task="anything")
    executor = captured["tool_executor"]

    await executor("send_message", {
        "to": "victim@example.com",
        "body": "ssn 123-45-6789",
    })

    assert tool.executed_with == {
        "to": "redacted@example.com",
        "body": "<PII removed>",
    }, (
        f"In-place rewrite of hook_input.tool_input must reach "
        f"tool.execute(). Got {tool.executed_with!r} — if this shows "
        f"the original args, real-manager redaction hooks are silently "
        f"bypassed on the codex inline path."
    )
