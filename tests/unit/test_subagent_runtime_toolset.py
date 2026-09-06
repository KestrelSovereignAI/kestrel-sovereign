"""One canonical toolset per subagent invocation.

Three things must name the same tools: the schemas advertised to the model,
the "your tools are" line in the subagent prompt, and the map the loop
executes against. They used to be derived separately and disagreed twice --
a security-denied tool stayed in the executable map (policy held only because
the model was never offered it), and the prompt could not see lent tools.

Lending is what lets a compacted subagent read back what was moved out of its
context, per salvage.py's invariant. It must not leak into feature identity:
`get_tools()` feeds `get_agent_card()` and the A2A skill list, so a borrowed
tool appearing there would make every feature advertise capabilities it does
not own.
"""

import pytest
from unittest.mock import MagicMock

from kestrel_sovereign.features.base import Feature
from kestrel_sdk.tools.base import AgentTool


def _async_return(value):
    async def _f(*a, **k):
        return value
    return _f


def _tool(name):
    t = MagicMock(spec=AgentTool)
    t.name = name
    return t


class _StubFeature(Feature):
    """Minimal Feature exposing a controllable tool list."""

    def __init__(self, own_tools, agent=None):
        self._own = own_tools
        self.agent = agent
        self.disabled_skills = set()

    @property
    def name(self):
        return "StubFeature"

    @property
    def tool_description(self) -> str:
        return "stub"

    async def initialize(self) -> bool:
        return True

    def get_tools(self):
        return list(self._own)


def _agent_with(context_tools=(), attachment_tools=()):
    ctx = MagicMock(); ctx.get_tools.return_value = list(context_tools)
    att = MagicMock(); att.get_tools.return_value = list(attachment_tools)
    agent = MagicMock()
    agent.get_feature.side_effect = lambda n: {
        "ContextFeature": ctx, "AttachmentsFeature": att
    }.get(n)
    return agent


@pytest.fixture
def feature():
    return _StubFeature(
        [_tool("own_a"), _tool("own_b")],
        agent=_agent_with([_tool("recursive_query")], [_tool("read_attachment")]),
    )
def test_prompt_names_exactly_the_runtime_toolset(feature):
    runtime = feature._compose_subagent_runtime_tools({"own_a"})
    prompt = feature._get_subagent_prompt(runtime)
    assert "own_b" in prompt
    assert "own_a" not in prompt            # denied tools are not


# ---------------------------------------------------------------------------
# The executable map, both doors
# ---------------------------------------------------------------------------
# Composition alone is not enough: the loop and the inline (codex) executor
# each build their own name->tool map, and each used to rebuild it from an
# unfiltered get_tools(). A denied tool was therefore still executable if its
# name reached either door. The model was never offered it, so policy held by
# omission -- but a hallucinated or replayed name is exactly the case policy
# exists for.

class _Call:
    def __init__(self, name, id="c1", arguments=None):
        self.name = name
        self.id = id
        self.arguments = arguments or {}


class _Resp:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


@pytest.mark.asyncio
async def test_loop_hands_the_executor_a_map_without_denied_tools(feature):
    """Kills the mutant that rebuilds tools_by_name from get_tools().

    Asserts on the MAP the loop passes, not on whether execution happened:
    the name lookup that refuses an absent tool lives inside
    ``_execute_subagent_tool``, so a double standing in for it cannot
    demonstrate the refusal -- it would only be testing the double.
    """
    seen = {}

    async def _spy(*, tool_name, args, tools_by_name, **kw):
        seen.update(tools_by_name)
        return {"ok": True}

    feature._execute_subagent_tool = _spy
    feature.agent.llm_service.generate_with_messages = _async_return(_Resp(content="done"))

    runtime = feature._compose_subagent_runtime_tools({"own_a"})
    await feature._handle_feature_tool_calls(
        _Resp(tool_calls=[_Call("own_b")]),
        tools=[], system_prompt="sp", max_iterations=2,
        runtime_tools=runtime,
    )
    assert seen, "executor was never reached — the assertion below would be vacuous"
    assert "own_a" not in seen, "a security-denied tool reached the executable map"
    assert "own_b" in seen


@pytest.mark.asyncio
async def test_loop_still_executes_a_permitted_tool(feature):
    """Control: proves the assertion above fails for the right reason and the
    harness can in fact observe an execution."""
    executed = []

    async def _spy(*, tool_name, args, tools_by_name, **kw):
        executed.append(tool_name)
        return {"ok": True}

    feature._execute_subagent_tool = _spy
    feature.agent.llm_service.generate_with_messages = _async_return(_Resp(content="done"))

    runtime = feature._compose_subagent_runtime_tools({"own_a"})
    await feature._handle_feature_tool_calls(
        _Resp(tool_calls=[_Call("own_b")]),
        tools=[], system_prompt="sp", max_iterations=2,
        runtime_tools=runtime,
    )
    assert executed == ["own_b"]


def test_inline_executor_uses_the_runtime_toolset(feature):
    """The second door. Built with the runtime list, its map must match."""
    runtime = feature._compose_subagent_runtime_tools({"own_a"})
    seen = {}

    async def _spy(*, tool_name, args, tools_by_name, **kw):
        seen.update(tools_by_name)
        return {"ok": True}

    feature._execute_subagent_tool = _spy
    executor = feature._make_feature_inline_tool_executor(runtime_tools=runtime)
    import asyncio
    asyncio.get_event_loop().run_until_complete(executor("own_b", {}))
    assert "own_a" not in seen
    assert "own_b" in seen


@pytest.mark.asyncio
async def test_all_tools_denied_still_refuses_through_the_real_path(feature):
    """The "all tools blocked" guard, exercised through execute_as_subagent.

    The existing coverage in test_denied_tools_dispatch.py reimplements this
    guard inside the test body and asserts against its own local dict, so it
    stays green no matter what the real path does. A change that made
    `available_tools` non-empty for a fully-denied feature would therefore
    run a whole subagent LLM loop unnoticed.
    """
    called = {"llm": 0}

    async def _never(*a, **k):
        called["llm"] += 1
        return _Resp(content="should not run")

    feature.agent.llm_service.generate = _never
    feature.agent.llm_service.generate_with_messages = _never

    result = await feature.execute_as_subagent(
        task="anything", denied_tools={"own_a", "own_b"}
    )

    assert result["success"] is False
    assert "blocked by security policy" in result["error"]
    assert called["llm"] == 0, "a fully-denied feature reached the model"
