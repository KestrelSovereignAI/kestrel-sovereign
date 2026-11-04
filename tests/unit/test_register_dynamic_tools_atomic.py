"""``register_dynamic_tools`` mounts a batch whole or not at all (#3923).

Before #3923 the owner's epoch and each tool's handle were written before the
tool's schema was converted, so a batch that raised part-way left a callable
handle no owner could retire, and expired the owner's retained bindings even
though the replacement never mounted.
"""

from copy import deepcopy
from unittest.mock import patch

import pytest
from kestrel_sdk.tools.base import ToolCategory, ToolParameter, ToolSchema

from kestrel_sovereign.agent.tool_allowlist import registry_generation
from kestrel_sovereign.kestrel_agent import KestrelAgent


class _Tool:
    def __init__(self, name):
        self.name = name
        self.schema = ToolSchema(
            name, f"{name} tool", ToolCategory.UTILITY,
            parameters=[ToolParameter("query", "string", "Query")],
        )

    async def execute(self, **args):
        return {"success": True, "result": args}


class _CachingSchema(ToolSchema):
    """A schema that hands back the one dict it caches, as a plugin may."""

    def to_openai_format(self):
        cached = self.__dict__.get("_cached_definition")
        if cached is None:
            cached = self.__dict__["_cached_definition"] = super().to_openai_format()
        return cached


class _CachingTool(_Tool):
    def __init__(self, name):
        super().__init__(name)
        self.schema = _CachingSchema(name, f"{name} tool", ToolCategory.UTILITY)


class _MalformedSchema:
    model_visible = True

    def to_openai_format(self):
        raise AttributeError("malformed schema")


class _MalformedSchemaTool(_Tool):
    def __init__(self, name="bad"):
        super().__init__(name)
        self.schema = _MalformedSchema()


class _UnreadableSchemaTool:
    """A plugin whose ``schema`` getter itself raises."""

    def __init__(self, name):
        self.name = name

    @property
    def schema(self):
        raise RuntimeError("schema unavailable")


@pytest.fixture
def agent():
    with patch("kestrel_sovereign.kestrel_agent.LLMService"):
        a = KestrelAgent(did="did:test:agent")
    a.features = {}
    return a


def _snapshot(agent):
    """Everything a registration can change, in order and by identity."""
    return {
        "handles": list(agent._direct_tools.items()),
        "definitions": deepcopy(agent._direct_tool_defs),
        "ownership": list(agent._tool_to_feature.items()),
        "epochs": list(agent._explored_features.items()),
        "pins": set(agent._pinned_features),
        "generation": registry_generation(agent),
    }


def _assert_no_orphans(agent):
    """Every handle has a mounted owner, and exactly the visible ones a schema."""
    assert set(agent._direct_tools) == set(agent._tool_to_feature)
    assert set(agent._tool_to_feature.values()) <= set(agent._explored_features)
    advertised = [d["function"]["name"] for d in agent._direct_tool_defs]
    assert len(advertised) == len(set(advertised))
    visible = {
        name for name, tool in agent._direct_tools.items()
        if getattr(tool.schema, "model_visible", True) is not False
    }
    assert set(advertised) == visible


def _mount(agent, owner, tools, **kwargs):
    bindings = []
    count = agent.register_dynamic_tools(
        owner, tools, on_registered=lambda _, binding: bindings.append(binding), **kwargs,
    )
    assert count == len(bindings)
    return bindings


@pytest.fixture
def populated(agent):
    """A registry holding a pinned owner and a retained binding."""
    pinned = _mount(agent, "pinned", [_Tool("keep_me")], pin=True)
    native = _mount(agent, "native", [_Tool("search"), _Tool("fetch")])
    return agent, pinned + native


def _rejected_batches():
    def iteration_raises():
        yield _Tool("first")
        raise OSError("tool listing failed")

    return {
        "first": lambda: [_MalformedSchemaTool(), _Tool("good"), _Tool("also_good")],
        "middle": lambda: [_Tool("good"), _MalformedSchemaTool(), _Tool("also_good")],
        "last": lambda: [_Tool("good"), _Tool("also_good"), _MalformedSchemaTool()],
        "schema_getter": lambda: [_Tool("good"), _UnreadableSchemaTool("bad")],
        "iteration": iteration_raises,
    }


_BATCHES = _rejected_batches()
_ERRORS = (AttributeError, RuntimeError, OSError)


@pytest.mark.parametrize("position", sorted(_BATCHES))
def test_rejected_batch_leaves_registry_unchanged(populated, position):
    agent, bindings = populated
    before = _snapshot(agent)
    with pytest.raises(_ERRORS):
        agent.register_dynamic_tools("review-owner", _BATCHES[position](), pin=True)
    assert _snapshot(agent) == before
    assert all(binding.is_current for binding in bindings)
    _assert_no_orphans(agent)


@pytest.mark.parametrize("position", sorted(_BATCHES))
def test_rejected_replacement_keeps_the_owners_bindings(populated, position):
    agent, _ = populated
    retained = _mount(agent, "mcp:views", [_Tool("view_update"), _Tool("view_read")])
    before = _snapshot(agent)
    with pytest.raises(_ERRORS):
        agent.register_dynamic_tools("mcp:views", _BATCHES[position]())
    assert _snapshot(agent) == before
    for binding in retained:
        assert binding.is_current
        binding.require_current()


def test_rejected_replacement_leaves_a_cached_definition_alone(populated):
    """The rename of a remounted tool must not reach the dict already mounted."""
    agent, bindings = populated
    same = _CachingTool("view_update")
    retained = _mount(agent, "mcp:views", [same])
    before = _snapshot(agent)
    with pytest.raises(AttributeError):
        agent.register_dynamic_tools("mcp:views", [same, _MalformedSchemaTool()])
    assert _snapshot(agent) == before
    assert same.schema.to_openai_format()["function"]["name"] == "view_update"
    assert all(binding.is_current for binding in bindings + retained)

    # Unregister finds the advertised schema by its mounted name.
    assert agent.unregister_dynamic_tools("mcp:views") == 1
    advertised = [d["function"]["name"] for d in agent._direct_tool_defs]
    assert "view_update" not in agent._direct_tools
    assert not [name for name in advertised if "view_update" in name]
    _assert_no_orphans(agent)


def test_a_shared_cached_definition_keeps_each_owners_name(agent):
    shared = _CachingTool("search")
    _mount(agent, "native", [shared])
    _mount(agent, "mcp:views", [shared])
    advertised = [d["function"]["name"] for d in agent._direct_tool_defs]
    assert advertised == ["search", "mcp_views__search"]
    assert shared.schema.to_openai_format()["function"]["name"] == "search"
    _assert_no_orphans(agent)

    assert agent.unregister_dynamic_tools("mcp:views") == 1
    assert [d["function"]["name"] for d in agent._direct_tool_defs] == ["search"]
    _assert_no_orphans(agent)


def test_unhashable_owner_is_refused_before_any_change(populated):
    agent, bindings = populated
    before = _snapshot(agent)
    with pytest.raises(TypeError):
        agent.register_dynamic_tools(["not", "hashable"], [_Tool("good")], pin=True)
    assert _snapshot(agent) == before
    assert all(binding.is_current for binding in bindings)


def test_registry_works_after_a_rejected_batch(populated):
    agent, _ = populated
    with pytest.raises(AttributeError):
        agent.register_dynamic_tools("review-owner", [_Tool("good"), _MalformedSchemaTool()])

    # A successful registration, including a collision with another owner.
    mounted = _mount(agent, "review-owner", [_Tool("good"), _Tool("search")])
    assert [binding.name for binding in mounted] == ["good", "review_owner__search"]
    assert agent._tool_to_feature["search"] == "native"
    assert agent._tool_to_feature["review_owner__search"] == "review-owner"
    _assert_no_orphans(agent)

    # Eviction still takes the oldest unpinned owner.
    agent.MAX_DIRECT_TOOLS = len(agent._direct_tools)
    newest = _mount(agent, "mcp:new", [_Tool("newest")])
    assert "native" not in agent._explored_features
    assert "search" not in agent._direct_tools
    assert "keep_me" in agent._direct_tools
    assert all(binding.is_current for binding in mounted + newest)
    _assert_no_orphans(agent)

    # Unregister retires the whole batch and leaves nothing behind.
    assert agent.unregister_dynamic_tools("review-owner") == 2
    assert not {"good", "review_owner__search"} & set(agent._direct_tools)
    assert not any(binding.is_current for binding in mounted)
    _assert_no_orphans(agent)


def test_names_are_reserved_within_one_batch(agent):
    mounted = _mount(agent, "mcp:x", [
        _Tool("dup"), _Tool("dup"), _Tool("dup"), _Tool("fetch.raw"), _Tool("fetch/raw"),
    ])
    assert [binding.name for binding in mounted] == [
        "dup", "mcp_x__dup", "mcp_x__dup_2", "fetch_raw", "fetch_raw_2",
    ]
    _assert_no_orphans(agent)


def test_callback_failure_propagates_after_a_committed_registration(agent):
    delivered = []

    def on_registered(tool, binding):
        if tool.name == "second":
            raise RuntimeError("callback failed")
        delivered.append(binding)

    generation = registry_generation(agent)
    with pytest.raises(RuntimeError, match="callback failed"):
        agent.register_dynamic_tools(
            "mcp:views", [_Tool("first"), _Tool("second"), _Tool("third")],
            on_registered=on_registered,
        )
    # The batch stays mounted whole: the failure stopped only the callbacks.
    assert [binding.name for binding in delivered] == ["first"]
    assert delivered[0].is_current
    owned = {name: agent._tool_to_feature.get(name) for name in ("first", "second", "third")}
    assert owned == dict.fromkeys(owned, "mcp:views")
    assert "mcp:views" in agent._explored_features
    assert registry_generation(agent) == generation + 1
    _assert_no_orphans(agent)

    assert agent.unregister_dynamic_tools("mcp:views") == 3
    assert not delivered[0].is_current
    assert agent._direct_tools == agent._tool_to_feature == {}
    _assert_no_orphans(agent)
