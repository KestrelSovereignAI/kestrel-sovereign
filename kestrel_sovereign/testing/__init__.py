"""
Kestrel Sovereign Testing Utilities.

Provides lightweight test fixtures for feature developers:

- MockAgent: A stubbed KestrelAgent that doesn't require real services
- FeatureTestCase: Base class for async feature tests with setup/teardown

Usage:
    from kestrel_sovereign.testing import MockAgent, FeatureTestCase

    class TestMyFeature(FeatureTestCase):
        feature_class = MyFeature

        async def test_my_tool(self):
            result = await self.feature.my_tool(param="test")
            assert result["success"]
"""

from kestrel_sovereign.testing.mock_agent import MockAgent
from kestrel_sovereign.testing.feature_test_case import FeatureTestCase

__all__ = ["MockAgent", "FeatureTestCase"]
