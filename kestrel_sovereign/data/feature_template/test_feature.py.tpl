"""Tests for {{class_name}}."""

import pytest

from kestrel_sovereign.testing import MockAgent
from {{pkg_name}}.feature import {{class_name}}


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def feature(mock_agent):
    return {{class_name}}(mock_agent)


def test_tool_description(feature):
    assert feature.tool_description


def test_get_tools(feature):
    tools = feature.get_tools()
    assert len(tools) >= 1
    names = [t.name for t in tools]
    assert "hello" in names


@pytest.mark.asyncio
async def test_hello(feature):
    result = await feature.hello(name="Kestrel")
    assert "Kestrel" in result
