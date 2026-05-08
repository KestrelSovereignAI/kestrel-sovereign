"""
Unit tests for the identity profile endpoints.

Tests PATCH /api/identity, POST /api/identity/avatar,
POST /api/identity/avatar/generate, and the description field on GET /api/identity.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO


# ---------------------------------------------------------------------------
# Shared mock helpers (same pattern as test_rename_command.py)
# ---------------------------------------------------------------------------

class MockNode:
    def __init__(self, node_id, properties=None, label=None, node_type="agent"):
        self.node_id = node_id
        self.properties = properties or {}
        self.label = label or ""
        self.node_type = node_type


class MockStorage:
    def __init__(self):
        self.nodes = {}
        self.files = MockFileStore()

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def add_node(self, node):
        self.nodes[node.node_id] = node


class MockFileStore:
    def __init__(self):
        self.stored = []

    async def store_avatar(self, image_data, agent_id, avatar_type="primary", source_url=None):
        content_hash = "abc123hash"
        self.stored.append({
            "image_data": image_data,
            "agent_id": agent_id,
            "avatar_type": avatar_type,
            "source_url": source_url,
        })
        return content_hash


class MockDB:
    def __init__(self):
        self.data = {}

    async def execute(self, query, params=None):
        if params and len(params) >= 4:
            key = (params[0], params[1])
            self.data[key] = params[2]

    async def fetchone(self, query, params=None):
        key = (params[0], params[1]) if params and len(params) >= 2 else None
        if key and key in self.data:
            return (self.data[key],)
        return None


class MockAgent:
    def __init__(self, agent_id="did:test:123", agent_name="TestAgent"):
        self.did = agent_id
        self.agent_id = agent_id
        self._agent_name = agent_name
        self.storage = MockStorage()
        self._raw_storage = MagicMock()
        self._raw_storage.db = MockDB()
        self.context_builder = MagicMock()
        self.bootstrap_service = MagicMock()
        self.bootstrap_service.agent_name = agent_name
        self.bootstrap_service.agent_data_path = None
        self.features = {}

        # Set up agent node
        self.storage.nodes[agent_id] = MockNode(
            agent_id,
            properties={"name": agent_name, "avatar_hash": None},
            label=agent_name,
        )


# ---------------------------------------------------------------------------
# Tests for rename_agent_core (extracted function)
# ---------------------------------------------------------------------------

class TestRenameAgentCore:
    @pytest.mark.asyncio
    async def test_rename_success(self):
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        agent = MockAgent()
        result, soul_updated = await rename_agent_core(agent, "NewName")

        assert "NewName" in result
        assert agent._agent_name == "NewName"
        assert agent.bootstrap_service.agent_name == "NewName"

    @pytest.mark.asyncio
    async def test_rename_empty_raises(self):
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        agent = MockAgent()
        with pytest.raises(ValueError, match="empty"):
            await rename_agent_core(agent, "")

    @pytest.mark.asyncio
    async def test_rename_too_long_raises(self):
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        agent = MockAgent()
        with pytest.raises(ValueError, match="too long"):
            await rename_agent_core(agent, "x" * 65)

    @pytest.mark.asyncio
    async def test_rename_whitespace_only_raises(self):
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        agent = MockAgent()
        with pytest.raises(ValueError, match="empty"):
            await rename_agent_core(agent, "   ")

    @pytest.mark.asyncio
    async def test_rename_strips_whitespace(self):
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        agent = MockAgent()
        result, _ = await rename_agent_core(agent, "  Trimmed  ")
        assert agent._agent_name == "Trimmed"

    @pytest.mark.asyncio
    async def test_rename_updates_node(self):
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        agent = MockAgent()
        await rename_agent_core(agent, "NodeTest")

        node = await agent.storage.get_node(agent.agent_id)
        assert node.properties["name"] == "NodeTest"
        assert node.label == "NodeTest"


# ---------------------------------------------------------------------------
# Tests for identity update endpoints (request/response shape)
# ---------------------------------------------------------------------------

class TestUpdateIdentityRequest:
    def test_valid_name(self):
        from kestrel_sovereign.endpoints.models import UpdateIdentityRequest

        req = UpdateIdentityRequest(name="Hello")
        assert req.name == "Hello"
        assert req.description is None

    def test_valid_description(self):
        from kestrel_sovereign.endpoints.models import UpdateIdentityRequest

        req = UpdateIdentityRequest(description="A helpful agent")
        assert req.description == "A helpful agent"
        assert req.name is None

    def test_name_too_long(self):
        from pydantic import ValidationError
        from kestrel_sovereign.endpoints.models import UpdateIdentityRequest

        with pytest.raises(ValidationError):
            UpdateIdentityRequest(name="x" * 65)

    def test_name_empty_string(self):
        from pydantic import ValidationError
        from kestrel_sovereign.endpoints.models import UpdateIdentityRequest

        with pytest.raises(ValidationError):
            UpdateIdentityRequest(name="")

    def test_description_too_long(self):
        from pydantic import ValidationError
        from kestrel_sovereign.endpoints.models import UpdateIdentityRequest

        with pytest.raises(ValidationError):
            UpdateIdentityRequest(description="x" * 501)


class TestGenerateAvatarRequest:
    def test_valid(self):
        from kestrel_sovereign.endpoints.models import GenerateAvatarRequest

        req = GenerateAvatarRequest(description="A friendly robot")
        assert req.description == "A friendly robot"
        assert req.num_outputs == 2

    def test_num_outputs_range(self):
        from pydantic import ValidationError
        from kestrel_sovereign.endpoints.models import GenerateAvatarRequest

        with pytest.raises(ValidationError):
            GenerateAvatarRequest(description="test", num_outputs=5)

        with pytest.raises(ValidationError):
            GenerateAvatarRequest(description="test", num_outputs=0)

    def test_description_required(self):
        from pydantic import ValidationError
        from kestrel_sovereign.endpoints.models import GenerateAvatarRequest

        with pytest.raises(ValidationError):
            GenerateAvatarRequest()


class TestSetAvatarUrlRequest:
    def test_valid(self):
        from kestrel_sovereign.endpoints.models import SetAvatarUrlRequest

        req = SetAvatarUrlRequest(url="https://example.com/avatar.png")
        assert req.url == "https://example.com/avatar.png"

    def test_empty_url(self):
        from pydantic import ValidationError
        from kestrel_sovereign.endpoints.models import SetAvatarUrlRequest

        with pytest.raises(ValidationError):
            SetAvatarUrlRequest(url="")
