"""
Unit tests for the identity profile endpoints.

Tests PATCH /api/identity, POST /api/identity/avatar,
POST /api/identity/avatar/generate, and the description field on GET /api/identity.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO
from starlette.responses import Response


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

    async def add_node(self, node, *, capability=None):
        # Mirrors the real AsyncStorage/wrapper envelope: trusted governance
        # callers (rename_agent_core) pass the control-plane capability (#2672).
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

    def resolve_effective_name(self, agent_node=None, *, default=None):
        # Mirror KestrelAgent.resolve_effective_name: the live in-memory name
        # is the session source of truth (a volatile rename updates it while
        # skipping the durable node), then the stored node, then ``default``
        # (#2672 review P2).
        live = getattr(self, "_agent_name", None)
        if isinstance(live, str) and live.strip():
            return live
        if agent_node is not None:
            props = getattr(agent_node, "properties", None) or {}
            name = props.get("name")
            if isinstance(name, str) and name.strip():
                return name
        return default


# ---------------------------------------------------------------------------
# Tests for rename_agent_core (extracted function)
# ---------------------------------------------------------------------------

class TestRenameAgentCore:
    @pytest.mark.asyncio
    async def test_rename_success(self):
        from kestrel_sovereign.features.bootstrap.feature import rename_agent_core

        agent = MockAgent()
        outcome = await rename_agent_core(agent, "NewName")

        assert outcome.success is True
        assert outcome.db_row_written is True
        assert outcome.graph_updated is True
        assert outcome.memory_updated is True
        assert outcome.soul_md_updated is False
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
        outcome = await rename_agent_core(agent, "  Trimmed  ")
        assert outcome.success is True
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
# PATCH /api/identity write-outcome contract in a volatile privacy mode
# (#2672 review P1/P2): a skipped durable write must report partial (207 /
# success:false), never a false 200/success:true.
# ---------------------------------------------------------------------------


class _VolatileStorage(MockStorage):
    """Mock storage whose privacy policy forbids durable user-content writes."""

    def __init__(self):
        super().__init__()
        from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config
        self.privacy_config = privacy_mode_to_config(PrivacyMode.EPHEMERAL)

    def allows_persistent_writes(self):
        return False


def _volatile_agent():
    agent = MockAgent()
    agent.storage = _VolatileStorage()
    agent.storage.nodes[agent.agent_id] = MockNode(
        agent.agent_id,
        properties={"name": "TestAgent", "description": "stored bio", "avatar_hash": None},
        label="TestAgent",
    )
    return agent


def _request_for(agent):
    req = MagicMock()
    req.state.agent = agent
    return req


class TestUpdateIdentityVolatileSkip:
    @pytest.mark.asyncio
    async def test_description_skip_reports_partial_not_success(self):
        """A description skipped for privacy returns 207 / success:false with an
        explicit skip flag — never a 200/success:true false confirmation (P2)."""
        from fastapi import Response
        from kestrel_sovereign.endpoints.models import (
            UpdateIdentityRequest,
            update_identity,
        )

        agent = _volatile_agent()
        response = Response()
        payload = await update_identity.__wrapped__(
            _request_for(agent), response,
            UpdateIdentityRequest(description="a fresh bio"),
        )

        assert response.status_code == 207
        assert payload["success"] is False
        assert payload["description_skipped_privacy"] is True
        assert "description_skipped_privacy" in payload["updated_fields"]
        assert "description" not in payload["updated_fields"]
        # The stored value is untouched (not the requested one).
        node = await agent.storage.get_node(agent.agent_id)
        assert node.properties["description"] == "stored bio"

    @pytest.mark.asyncio
    async def test_name_skip_reports_partial_not_success(self):
        """A name skipped for privacy returns 207 / success:false (P1)."""
        from fastapi import Response
        from kestrel_sovereign.endpoints.models import (
            UpdateIdentityRequest,
            update_identity,
        )

        agent = _volatile_agent()
        response = Response()
        payload = await update_identity.__wrapped__(
            _request_for(agent), response,
            UpdateIdentityRequest(name="NewName"),
        )

        assert response.status_code == 207
        assert payload["success"] is False
        assert "name_skipped_privacy" in payload["updated_fields"]
        # In-memory name applied, durable name unchanged.
        assert agent._agent_name == "NewName"
        node = await agent.storage.get_node(agent.agent_id)
        assert node.properties["name"] == "TestAgent"


class TestVolatileRenameLiveVsRestart:
    """A session-only (volatile-mode) rename is visible LIVE — on the PATCH
    response AND the A2A discovery card, which share one resolver — but leaves no
    durable trace, so a restarted agent (which reloads its name from the durable
    node) answers to the OLD name (#2672 review P2)."""

    @pytest.mark.asyncio
    async def test_session_rename_visible_live_but_absent_after_restart(self):
        from fastapi import Response
        from kestrel_sovereign.endpoints.models import (
            UpdateIdentityRequest,
            update_identity,
        )
        from kestrel_sovereign.kestrel_agent import KestrelAgent

        agent = _volatile_agent()  # durable stored name "TestAgent"
        response = Response()
        payload = await update_identity.__wrapped__(
            _request_for(agent), response,
            UpdateIdentityRequest(name="SessionName"),
        )

        # LIVE (API): the PATCH response reports the live session name, 207 partial.
        assert response.status_code == 207
        assert payload["success"] is False
        assert payload["name"] == "SessionName"
        assert agent._agent_name == "SessionName"

        # LIVE (A2A card): the real card builder shares resolve_effective_name, so
        # it advertises the same live session name — endpoint and card never
        # disagree after a volatile rename.
        card = await KestrelAgent.get_agent_card(agent)
        assert card.name == "SessionName"

        # DURABLE: the stored node was never renamed.
        node = await agent.storage.get_node(agent.agent_id)
        assert node.properties["name"] == "TestAgent"
        assert node.label == "TestAgent"

        # RESTART: a fresh agent reloads its live name from the durable node, which
        # still holds the OLD name — the session rename left no trace anywhere.
        restarted = _volatile_agent()
        restarted._agent_name = (
            await restarted.storage.get_node(restarted.agent_id)
        ).properties["name"]
        assert restarted._agent_name == "TestAgent"
        restarted_card = await KestrelAgent.get_agent_card(restarted)
        assert restarted_card.name == "TestAgent"


class TestMultiAgentDiscoveryLiveName:
    """The multi-agent ``GET /api/agents`` discovery route advertises each agent's
    LIVE display name (from ``get_agent_card`` → ``resolve_effective_name``) and
    exposes the AgentManager routing key separately as ``routing_name``. It must
    NOT overwrite the live name with the immutable key — the prior bug hid every
    session rename behind the registration name in host discovery (#2672 review
    P2)."""

    @pytest.mark.asyncio
    async def test_get_agents_reports_live_name_and_routing_key(self):
        from kestrel_sovereign.endpoints.models import get_agents

        # Registered under manager key "Emma" but renamed live to "RenamedLive"
        # (a volatile session rename that skipped the durable node write). The card
        # carries the live name; the manager key is the routing identity.
        card = MagicMock()
        card.model_dump.return_value = {
            "name": "RenamedLive",
            "description": "d",
            "url": "http://localhost:8888",
            "skills": [],
        }
        agent = MagicMock(agent_id="did:emma", is_demo=False)
        agent.get_agent_card = AsyncMock(return_value=card)

        manager = MagicMock()
        manager.list_agents.return_value = {"Emma": agent}

        request = MagicMock()
        request.app.state.agent_manager = manager
        request.app.state.demo_mode = False

        result = await get_agents(request, Response())

        assert result["mode"] == "multi_agent"
        entry = result["agents"][0]
        assert entry["name"] == "RenamedLive", "discovery advertises the LIVE name"
        assert entry["routing_name"] == "Emma", "manager key exposed for path routing"
        assert entry["id"] == "did:emma"

    @pytest.mark.asyncio
    async def test_get_agents_error_fallback_carries_routing_name(self):
        """When card generation fails the fallback entry still carries the routing
        key under BOTH ``name`` (best available) and ``routing_name`` so path
        construction never loses the manager key (#2672 review P2)."""
        from kestrel_sovereign.endpoints.models import get_agents

        agent = MagicMock(agent_id="did:emma", is_demo=False)
        agent.get_agent_card = AsyncMock(side_effect=RuntimeError("boom"))

        manager = MagicMock()
        manager.list_agents.return_value = {"Emma": agent}

        request = MagicMock()
        request.app.state.agent_manager = manager
        request.app.state.demo_mode = False

        result = await get_agents(request, Response())
        entry = result["agents"][0]
        assert entry["status"] == "error"
        assert entry["name"] == "Emma"
        assert entry["routing_name"] == "Emma"


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
