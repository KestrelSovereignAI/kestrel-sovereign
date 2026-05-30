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
