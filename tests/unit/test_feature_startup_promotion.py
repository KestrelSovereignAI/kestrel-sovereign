from unittest.mock import MagicMock

from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.features.save.feature import SaveFeature
from kestrel_sovereign.features.spawn.feature import SpawnFeature
from kestrel_sovereign.features.strategic_memory.feature import (
    StrategicMemoryFeature,
)
from kestrel_sovereign.features.tasks.feature import TaskFeature
from kestrel_sovereign.kestrel_agent import KestrelAgent


class PlainFeature(Feature):
    @property
    def tool_description(self):
        return "Plain test feature"

    async def initialize(self):
        pass


def test_feature_startup_promotion_defaults_to_disabled():
    feature = PlainFeature(MagicMock())

    assert feature.promote_tools_on_startup is False


def test_meta_features_opt_into_startup_direct_tools():
    agent = MagicMock()

    assert SpawnFeature(agent).promote_tools_on_startup is True
    assert TaskFeature(agent).promote_tools_on_startup is True
    assert PeersFeature(agent).promote_tools_on_startup is True


def test_save_and_strategic_memory_opt_into_startup_direct_tools():
    """#1578 (B): SaveFeature and StrategicMemoryFeature own durable
    operational tools (save_item, strategy_add_decision, etc.) that
    must be advertised from turn 1 so the LLM doesn't hit "not
    advertised" on first use. Same tier as Peers/Tasks/Spawn."""
    agent = MagicMock()
    assert SaveFeature(agent).promote_tools_on_startup is True
    assert StrategicMemoryFeature(agent).promote_tools_on_startup is True


def test_startup_promotion_stays_under_budget():
    """#1578 (B) startup-budget guardrail: the count of @tool methods
    surfaced by features opting into startup promotion must stay
    under a target so we don't silently chew the
    ``MAX_DIRECT_TOOLS = 60`` budget and force LRU eviction of
    other promoted-but-not-pinned features. Emma's reshape demanded
    this assertion."""
    agent = MagicMock()
    promoted_classes = (
        SaveFeature, StrategicMemoryFeature,
        SpawnFeature, TaskFeature, PeersFeature,
    )
    total = 0
    for cls in promoted_classes:
        feat = cls(agent)
        assert feat.promote_tools_on_startup is True
        try:
            total += len(list(feat.get_tools()))
        except Exception:
            # Some feature classes lazy-init tools in initialize();
            # if get_tools raises pre-init, count the @tool-decorated
            # methods by introspection instead so the budget assertion
            # is still meaningful for that class.
            tool_methods = [
                name for name in dir(cls)
                if callable(getattr(cls, name, None))
                and hasattr(getattr(cls, name), "_kestrel_tool_schema")
            ]
            total += len(tool_methods)
    # Target leaves headroom inside the 60-tool cap so explored-but-
    # not-pinned features have room to land without immediate
    # eviction. If this assertion ever fails, raise MAX_DIRECT_TOOLS
    # in tool_registry.py rather than removing it.
    assert total <= 50, (
        f"Startup-promoted features expose {total} direct tools; "
        f"budget is 50 (out of {60} cap). Tighten promotion scope or "
        f"raise MAX_DIRECT_TOOLS."
    )


def test_startup_promotion_uses_feature_descriptor_not_feature_names():
    promoted = MagicMock()
    promoted.name = "ArbitraryRuntimeFeature"
    promoted.tool_name = "arbitrary_runtime"
    promoted.promote_tools_on_startup = True

    regular = MagicMock()
    regular.name = "TaskFeature"
    regular.tool_name = "task_feature"
    regular.promote_tools_on_startup = False

    agent = object.__new__(KestrelAgent)
    agent.features = {
        promoted.name: promoted,
        regular.name: regular,
    }
    agent._pinned_features = set()
    calls = []
    agent._register_explored_feature_tools = calls.append

    agent._promote_startup_feature_tools()

    assert calls == [promoted]
    # #1580 (D): promoted features are also pinned so LRU eviction
    # can never silently drop them.
    assert agent._pinned_features == {"arbitrary_runtime"}
