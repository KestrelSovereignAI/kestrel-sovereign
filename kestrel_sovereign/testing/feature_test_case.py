"""
FeatureTestCase — async test base class for Kestrel features.

Provides automatic MockAgent setup, feature instantiation, and teardown.

Usage:
    from kestrel_sovereign.testing import FeatureTestCase
    from my_feature import MyFeature

    class TestMyFeature(FeatureTestCase):
        feature_class = MyFeature

        async def test_my_tool(self):
            result = await self.feature.my_tool(param="test")
            assert result["success"]
"""

import unittest
from typing import ClassVar, List, Optional, Type

from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.testing.mock_agent import MockAgent


class FeatureTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Base test class for testing Kestrel Features.

    Subclasses set ``feature_class`` to the Feature they want to test.
    The MockAgent and feature are created in setUp and torn down in tearDown.

    Class attributes:
        feature_class: The Feature subclass to test (required).
        use_storage: If True, creates MockAgent with in-memory SQLite storage.
        default_llm_response: Default response from the mock LLM service.
        llm_responses: Queue of responses the mock LLM will return in order.

    Instance attributes (available in tests):
        agent: The MockAgent instance.
        feature: The instantiated and initialized feature.
    """

    feature_class: ClassVar[Type[Feature]]
    use_storage: ClassVar[bool] = False
    default_llm_response: ClassVar[str] = "Mock LLM response"
    llm_responses: ClassVar[Optional[List[str]]] = None

    async def asyncSetUp(self) -> None:
        """Create MockAgent and initialize the feature."""
        if not hasattr(self, "feature_class") or self.feature_class is None:
            raise TypeError(
                f"{self.__class__.__name__} must set feature_class to a Feature subclass"
            )

        if self.use_storage:
            self.agent = await MockAgent.create(
                default_llm_response=self.default_llm_response,
                llm_responses=list(self.llm_responses) if self.llm_responses else None,
            )
        else:
            self.agent = MockAgent(
                default_llm_response=self.default_llm_response,
                llm_responses=list(self.llm_responses) if self.llm_responses else None,
            )

        self.feature = self.feature_class(self.agent)
        self.agent.features[self.feature.name] = self.feature
        await self.feature.initialize()

    async def asyncTearDown(self) -> None:
        """Shutdown feature and agent."""
        if hasattr(self, "feature"):
            await self.feature.shutdown()
        if hasattr(self, "agent"):
            await self.agent.shutdown()
