from unittest.mock import MagicMock

from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.features.spawn.feature import SpawnFeature
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
    calls = []
    agent._register_explored_feature_tools = calls.append

    agent._promote_startup_feature_tools()

    assert calls == [promoted]
