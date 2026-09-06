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


def test_runtime_toolset_keeps_the_features_own_tools(feature):
    names = [t.name for t in feature._compose_subagent_runtime_tools()]
    assert "own_a" in names and "own_b" in names


def test_runtime_toolset_lends_context_retrieval(feature):
    names = [t.name for t in feature._compose_subagent_runtime_tools()]
    assert "recursive_query" in names
    assert "read_attachment" in names


def test_denied_tools_are_absent_from_the_runtime_toolset(feature):
    names = [t.name for t in feature._compose_subagent_runtime_tools({"own_a"})]
    assert "own_a" not in names
    assert "own_b" in names


def test_a_denied_borrowed_tool_is_also_withheld(feature):
    """Security policy outranks lending."""
    names = [
        t.name for t in feature._compose_subagent_runtime_tools({"recursive_query"})
    ]
    assert "recursive_query" not in names


def test_borrowing_does_not_touch_feature_identity(feature):
    """THE cleanliness guarantee: get_tools() feeds get_agent_card() and the
    A2A skill list, so a lent tool must never appear there."""
    identity = [t.name for t in feature.get_tools()]
    assert identity == ["own_a", "own_b"]
    assert "recursive_query" not in identity
    assert "read_attachment" not in identity


def test_missing_context_features_are_not_an_error():
    """A host may not have Context/Attachments enabled."""
    bare = _StubFeature([_tool("own_a")], agent=MagicMock(get_feature=lambda n: None))
    assert [t.name for t in bare._compose_subagent_runtime_tools()] == ["own_a"]


def test_no_agent_at_all_is_not_an_error():
    orphan = _StubFeature([_tool("own_a")], agent=None)
    assert [t.name for t in orphan._compose_subagent_runtime_tools()] == ["own_a"]


def test_a_feature_owning_the_same_name_wins_over_the_lent_one():
    """The Context feature dispatching its own subagent must execute its own
    recursive_query, not a borrowed duplicate."""
    f = _StubFeature(
        [_tool("recursive_query")],
        agent=_agent_with([_tool("recursive_query")]),
    )
    tools = f._compose_subagent_runtime_tools()
    assert [t.name for t in tools].count("recursive_query") == 1
    assert tools[0] is f._own[0]


def test_prompt_names_exactly_the_runtime_toolset(feature):
    runtime = feature._compose_subagent_runtime_tools({"own_a"})
    prompt = feature._get_subagent_prompt(runtime)
    assert "own_b" in prompt
    assert "recursive_query" in prompt      # lent tools are named
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
    assert "recursive_query" in seen, "lent tools must be executable, not just advertised"


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
