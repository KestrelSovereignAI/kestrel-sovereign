"""
Unit Tests for the KestrelClaw Bridge Feature (#157).

Tests:
- BridgeFeature lifecycle (initialize, shutdown)
- Database table creation (bridge_sessions, bridge_log)
- Session management (create, resume, prune stale)
- Invocation logging (inbound, outbound)
- Tool commands (!bridge status, !bridge connections, !bridge history)
- Capability discovery
- Tool discovery and command prefixes
- Router endpoint logic (invoke, stream, capabilities, health, session)
- Graceful degradation without database
- Protocol model validation
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from kestrel_sovereign.features.bridge.feature import (
    BridgeFeature,
    MAX_ACTIVE_SESSIONS,
    SESSION_IDLE_TIMEOUT_SECONDS,
)
from kestrel_sovereign.features.bridge.protocol import (
    BridgeCapabilitiesResponse,
    BridgeCapability,
    BridgeRequest,
    BridgeResponse,
    BridgeSession,
    ChannelType,
)
from kestrel_sovereign.features.bridge.router import _build_context_note


# ============================================================================
# Helpers
# ============================================================================


def _make_db(fetchone_data=None, fetchall_data=None):
    """Create a mock AsyncDatabase."""
    db = AsyncMock()
    db.fetchone = AsyncMock(return_value=fetchone_data if fetchone_data is not None else (0,))
    db.fetchall = AsyncMock(return_value=fetchall_data or [])
    db.execute = AsyncMock(return_value=0)
    db.table_exists = AsyncMock(return_value=True)
    return db


def _make_agent(db=None, agent_id="test-bridge-agent"):
    """Create a mock KestrelAgent with configurable components."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.did = agent_id

    storage = MagicMock()
    storage.db = db
    agent.storage = storage
    agent._raw_storage = None

    # LLM service
    llm_service = MagicMock()
    llm_service.provider = "anthropic"
    agent.llm_service = llm_service

    # Features dict (empty by default, tests add BridgeFeature)
    agent.features = {}

    # process_input mock
    agent.process_input = AsyncMock(return_value="Hello from the agent!")

    # process_input_streaming mock
    async def mock_streaming(*args, **kwargs):
        for chunk in ["Hello ", "from ", "streaming!"]:
            yield chunk

    agent.process_input_streaming = mock_streaming

    return agent


# ============================================================================
# Protocol Model Tests
# ============================================================================


class TestProtocolModels:
    def test_bridge_request_valid(self):
        req = BridgeRequest(message="Hello")
        assert req.message == "Hello"
        assert req.channel_type == ChannelType.API
        assert req.session_id is None
        assert req.context is None

    def test_bridge_request_full(self):
        req = BridgeRequest(
            session_id="gw-123",
            message="What's on this page?",
            context={"url": "https://example.com", "selected_text": "Lorem ipsum"},
            channel_type=ChannelType.BROWSER_EXTENSION,
            sender_id="user-456",
            model_override="anthropic/claude-3-opus",
            did="did:key:z6Mk...",
        )
        assert req.session_id == "gw-123"
        assert req.channel_type == ChannelType.BROWSER_EXTENSION
        assert req.context["url"] == "https://example.com"

    def test_bridge_request_rejects_empty_message(self):
        with pytest.raises(Exception):
            BridgeRequest(message="")

    def test_bridge_response_valid(self):
        resp = BridgeResponse(
            message="Here is the answer.",
            session_id="sess-1",
        )
        assert resp.message == "Here is the answer."
        assert resp.tool_results is None
        assert resp.metadata == {}

    def test_bridge_session_touch(self):
        session = BridgeSession(
            id="s1",
            agent_id="agent-1",
        )
        old_time = session.last_activity_at
        # Brief pause so the clock advances
        import time as _time
        _time.sleep(0.01)
        session.touch()
        assert session.last_activity_at >= old_time

    def test_channel_types(self):
        assert ChannelType.BROWSER_EXTENSION.value == "browser_extension"
        assert ChannelType.DISCORD.value == "discord"
        assert ChannelType.SLACK.value == "slack"
        assert ChannelType.API.value == "api"

    def test_capabilities_response(self):
        resp = BridgeCapabilitiesResponse(
            agent_id="agent-1",
            features=["BridgeFeature", "HeartbeatFeature"],
            capabilities=[
                BridgeCapability(
                    name="bridge_status",
                    description="Show status",
                    category="system",
                )
            ],
        )
        assert len(resp.capabilities) == 1
        assert "browser_extension" in resp.channel_types


# ============================================================================
# BridgeFeature Initialization Tests
# ============================================================================


class TestBridgeFeatureInitialize:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = BridgeFeature(agent)
        await feat.initialize()
        yield feat

    @pytest.mark.asyncio
    async def test_creates_sessions_table(self, feature):
        create_calls = [
            c for c in feature._db.execute.call_args_list
            if "CREATE TABLE" in str(c) and "bridge_sessions" in str(c)
        ]
        assert len(create_calls) == 1

    @pytest.mark.asyncio
    async def test_creates_log_table(self, feature):
        create_calls = [
            c for c in feature._db.execute.call_args_list
            if "CREATE TABLE" in str(c) and "bridge_log" in str(c)
        ]
        assert len(create_calls) == 1

    @pytest.mark.asyncio
    async def test_creates_indexes(self, feature):
        index_calls = [
            c for c in feature._db.execute.call_args_list
            if "CREATE INDEX" in str(c)
        ]
        # Should create: idx_bridge_sessions_gateway, idx_bridge_sessions_agent,
        # idx_bridge_log_agent, idx_bridge_log_session
        assert len(index_calls) == 4

    @pytest.mark.asyncio
    async def test_sets_agent_id(self, feature):
        assert feature._agent_id == "test-bridge-agent"

    @pytest.mark.asyncio
    async def test_initializes_empty_sessions(self, feature):
        assert feature._sessions == {}

    @pytest.mark.asyncio
    async def test_initializes_counters(self, feature):
        assert feature._invocation_count == 0


class TestBridgeFeatureInitializeWithoutDb:
    @pytest.mark.asyncio
    async def test_initialize_without_db(self):
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = BridgeFeature(agent)
        await feat.initialize()
        assert feat._db is None
        # Should not raise


# ============================================================================
# Session Management Tests
# ============================================================================


class TestSessionManagement:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = BridgeFeature(agent)
        await feat.initialize()
        yield feat

    @pytest.mark.asyncio
    async def test_create_new_session(self, feature):
        session = await feature.get_or_create_session(
            gateway_session_id="gw-1",
            channel_type=ChannelType.BROWSER_EXTENSION,
            sender_id="user-1",
        )
        assert session.id is not None
        assert session.gateway_session_id == "gw-1"
        assert session.channel_type == ChannelType.BROWSER_EXTENSION
        assert session.sender_id == "user-1"
        assert session.agent_id == "test-bridge-agent"

    @pytest.mark.asyncio
    async def test_resume_session_from_memory(self, feature):
        # Create first
        session1 = await feature.get_or_create_session(
            gateway_session_id="gw-2"
        )
        # Resume
        session2 = await feature.get_or_create_session(
            gateway_session_id="gw-2"
        )
        assert session1.id == session2.id

    @pytest.mark.asyncio
    async def test_resume_session_from_db(self, feature):
        """Session not in memory but found in database."""
        now = datetime.now(timezone.utc).isoformat()
        feature._db.fetchone = AsyncMock(return_value=(
            "db-session-id", "test-bridge-agent", "gw-db",
            "api", "user-db", now, now,
        ))

        session = await feature.get_or_create_session(
            gateway_session_id="gw-db"
        )
        assert session.id == "db-session-id"

    @pytest.mark.asyncio
    async def test_create_session_without_gateway_id(self, feature):
        session = await feature.get_or_create_session(
            gateway_session_id=None,
            channel_type=ChannelType.DISCORD,
        )
        assert session.id is not None
        assert session.gateway_session_id is None

    @pytest.mark.asyncio
    async def test_session_persisted_to_db(self, feature):
        await feature.get_or_create_session(gateway_session_id="gw-persist")
        insert_calls = [
            c for c in feature._db.execute.call_args_list
            if "INSERT INTO bridge_sessions" in str(c)
        ]
        assert len(insert_calls) == 1

    @pytest.mark.asyncio
    async def test_session_touch_updates_activity(self, feature):
        session = await feature.get_or_create_session(
            gateway_session_id="gw-touch"
        )
        old_activity = session.last_activity_at

        import time as _time
        _time.sleep(0.01)

        # Resume triggers touch
        session2 = await feature.get_or_create_session(
            gateway_session_id="gw-touch"
        )
        assert session2.last_activity_at >= old_activity

    @pytest.mark.asyncio
    async def test_prune_stale_sessions(self, feature):
        """Sessions older than the timeout are pruned."""
        old_time = datetime.now(timezone.utc) - timedelta(
            seconds=SESSION_IDLE_TIMEOUT_SECONDS + 100
        )
        feature._sessions["stale-1"] = BridgeSession(
            id="stale-session",
            agent_id="test-bridge-agent",
            gateway_session_id="stale-1",
            created_at=old_time,
            last_activity_at=old_time,
        )
        feature._sessions["fresh-1"] = BridgeSession(
            id="fresh-session",
            agent_id="test-bridge-agent",
            gateway_session_id="fresh-1",
        )

        await feature._prune_stale_sessions()
        assert "stale-1" not in feature._sessions
        assert "fresh-1" in feature._sessions


# ============================================================================
# Invocation Logging Tests
# ============================================================================


class TestInvocationLogging:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = BridgeFeature(agent)
        await feat.initialize()
        # Reset call tracking after initialization
        db.execute.reset_mock()
        yield feat

    @pytest.mark.asyncio
    async def test_log_inbound(self, feature):
        await feature.log_invocation(
            session_id="sess-1",
            direction="inbound",
            content_preview="Hello from gateway",
        )
        insert_calls = [
            c for c in feature._db.execute.call_args_list
            if "INSERT INTO bridge_log" in str(c)
        ]
        assert len(insert_calls) == 1
        assert feature._invocation_count == 1

    @pytest.mark.asyncio
    async def test_log_outbound_with_metrics(self, feature):
        await feature.log_invocation(
            session_id="sess-1",
            direction="outbound",
            content_preview="Agent response",
            tokens_used=150,
            duration_ms=320,
        )
        insert_calls = [
            c for c in feature._db.execute.call_args_list
            if "INSERT INTO bridge_log" in str(c)
        ]
        assert len(insert_calls) == 1

    @pytest.mark.asyncio
    async def test_log_truncates_preview(self, feature):
        long_content = "x" * 500
        await feature.log_invocation(
            session_id="sess-1",
            direction="inbound",
            content_preview=long_content,
        )
        # The execute was called -- we check the args
        call_args = feature._db.execute.call_args
        # The content_preview arg is at index 4 in the tuple (position 5)
        stored_preview = call_args[0][1][4]  # (sql, params)[params][4]
        assert len(stored_preview) <= 200

    @pytest.mark.asyncio
    async def test_log_without_db(self):
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = BridgeFeature(agent)
        await feat.initialize()

        # Should not raise
        await feat.log_invocation(
            session_id="sess-1",
            direction="inbound",
            content_preview="no db",
        )
        assert feat._invocation_count == 1

    @pytest.mark.asyncio
    async def test_log_db_failure_does_not_crash(self, feature):
        feature._db.execute = AsyncMock(side_effect=Exception("DB write error"))
        # Should not raise
        await feature.log_invocation(
            session_id="sess-1",
            direction="inbound",
            content_preview="should not crash",
        )
        assert feature._invocation_count == 1


# ============================================================================
# Tool Command Tests
# ============================================================================


class TestBridgeStatus:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = BridgeFeature(agent)
        await feat.initialize()
        yield feat

    @pytest.mark.asyncio
    async def test_returns_status(self, feature):
        result = await feature.bridge_status()
        assert result["status"] == "active"
        assert result["agent_id"] == "test-bridge-agent"
        assert "uptime_seconds" in result
        assert result["database_available"] is True
        assert result["total_invocations"] == 0

    @pytest.mark.asyncio
    async def test_counts_sessions(self, feature):
        feature._db.fetchone = AsyncMock(return_value=(5,))
        result = await feature.bridge_status()
        assert result["total_sessions_db"] == 5

    @pytest.mark.asyncio
    async def test_status_without_db(self):
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = BridgeFeature(agent)
        await feat.initialize()

        result = await feat.bridge_status()
        assert result["status"] == "active"
        assert result["database_available"] is False


class TestBridgeConnections:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = BridgeFeature(agent)
        await feat.initialize()
        yield feat

    @pytest.mark.asyncio
    async def test_returns_sessions_from_db(self, feature):
        now = datetime.now(timezone.utc).isoformat()
        feature._db.fetchall = AsyncMock(return_value=[
            ("s1", "gw-1", "browser_extension", "user-1", now, now),
            ("s2", "gw-2", "discord", None, now, now),
        ])
        result = await feature.bridge_connections()
        assert result["count"] == 2
        assert result["sessions"][0]["channel_type"] == "browser_extension"

    @pytest.mark.asyncio
    async def test_falls_back_to_memory(self, feature):
        feature._db.fetchall = AsyncMock(return_value=[])
        # Add a session to memory
        feature._sessions["gw-mem"] = BridgeSession(
            id="mem-session",
            agent_id="test-bridge-agent",
            gateway_session_id="gw-mem",
            channel_type=ChannelType.SLACK,
        )
        result = await feature.bridge_connections()
        assert result["count"] == 1
        assert result["sessions"][0]["channel_type"] == "slack"


class TestBridgeHistory:
    @pytest_asyncio.fixture
    async def feature(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = BridgeFeature(agent)
        await feat.initialize()
        yield feat

    @pytest.mark.asyncio
    async def test_returns_log_entries(self, feature):
        now = datetime.now(timezone.utc).isoformat()
        feature._db.fetchall = AsyncMock(return_value=[
            ("log-1", "sess-1", "inbound", "Hello", 0, 0, now),
            ("log-2", "sess-1", "outbound", "Hi there", 150, 320, now),
        ])
        result = await feature.bridge_history()
        assert result["count"] == 2
        assert result["entries"][0]["direction"] == "inbound"
        assert result["entries"][1]["tokens_used"] == 150

    @pytest.mark.asyncio
    async def test_empty_history(self, feature):
        result = await feature.bridge_history()
        assert result["count"] == 0
        assert result["entries"] == []


# ============================================================================
# Capability Discovery Tests
# ============================================================================


class TestCapabilityDiscovery:
    @pytest.mark.asyncio
    async def test_lists_all_feature_tools(self):
        db = _make_db()
        agent = _make_agent(db=db)

        # Add a mock feature with tools
        mock_feature = MagicMock()
        mock_tool = MagicMock()
        mock_tool.schema.name = "test_tool"
        mock_tool.schema.description = "A test tool"
        mock_tool.schema.category.value = "system"
        mock_tool.schema.command_prefix = "!test"
        mock_tool.schema.parameters = []
        mock_feature.get_tools.return_value = [mock_tool]

        agent.features = {"MockFeature": mock_feature}

        feat = BridgeFeature(agent)
        await feat.initialize()

        capabilities = feat.get_capabilities()
        assert len(capabilities) == 1
        assert capabilities[0]["name"] == "test_tool"
        assert capabilities[0]["feature"] == "MockFeature"

    @pytest.mark.asyncio
    async def test_handles_feature_tool_error(self):
        db = _make_db()
        agent = _make_agent(db=db)

        # Feature that throws when get_tools() is called
        bad_feature = MagicMock()
        bad_feature.get_tools.side_effect = Exception("broken")

        agent.features = {"BadFeature": bad_feature}

        feat = BridgeFeature(agent)
        await feat.initialize()

        # Should not raise
        capabilities = feat.get_capabilities()
        assert capabilities == []


# ============================================================================
# Tool Discovery Tests
# ============================================================================


class TestToolDiscovery:
    @pytest.mark.asyncio
    async def test_tools_registered(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = BridgeFeature(agent)
        await feat.initialize()

        tools = feat.get_tools()
        tool_names = {t.name for t in tools}

        assert "bridge_status" in tool_names
        assert "bridge_connections" in tool_names
        assert "bridge_history" in tool_names
        assert len(tool_names) == 3

    @pytest.mark.asyncio
    async def test_tool_description(self):
        agent = _make_agent()
        feat = BridgeFeature(agent)
        assert "bridge" in feat.tool_description.lower()
        assert "gateway" in feat.tool_description.lower()

    @pytest.mark.asyncio
    async def test_command_prefixes(self):
        db = _make_db()
        agent = _make_agent(db=db)
        feat = BridgeFeature(agent)
        await feat.initialize()

        tools = feat.get_tools()
        prefixes = {t.schema.command_prefix for t in tools}
        assert "!bridge status" in prefixes
        assert "!bridge connections" in prefixes
        assert "!bridge history" in prefixes


# ============================================================================
# Context Note Builder Tests
# ============================================================================


class TestBuildContextNote:
    def test_empty_context(self):
        body = BridgeRequest(message="hi")
        assert _build_context_note(body) == ""

    def test_url_context(self):
        body = BridgeRequest(
            message="summarize",
            context={"url": "https://example.com"},
        )
        note = _build_context_note(body)
        assert "URL: https://example.com" in note

    def test_selected_text_truncation(self):
        body = BridgeRequest(
            message="explain",
            context={"selected_text": "x" * 600},
        )
        note = _build_context_note(body)
        assert len(note) < 600

    def test_multiple_context_fields(self):
        body = BridgeRequest(
            message="help",
            context={
                "url": "https://example.com",
                "page_title": "Test Page",
                "channel_name": "#general",
                "custom_field": "custom_value",
            },
        )
        note = _build_context_note(body)
        assert "URL: https://example.com" in note
        assert "Page: Test Page" in note
        assert "Channel: #general" in note
        assert "custom_field: custom_value" in note


# ============================================================================
# Router Integration Tests (using mock request/app)
# ============================================================================


class TestRouterHelpers:
    """Test the router helper functions and basic endpoint logic."""

    @pytest.mark.asyncio
    async def test_get_bridge_feature_missing_agent(self):
        from kestrel_sovereign.features.bridge.router import _get_bridge_feature

        request = MagicMock()
        request.app.state = MagicMock(spec=[])  # no 'agent' attribute

        with pytest.raises(Exception) as exc_info:
            _get_bridge_feature(request)
        assert "503" in str(exc_info.value.status_code)

    @pytest.mark.asyncio
    async def test_get_bridge_feature_missing_bridge(self):
        from kestrel_sovereign.features.bridge.router import _get_bridge_feature

        request = MagicMock()
        agent = MagicMock()
        agent.features = {}
        request.app.state.agent = agent

        with pytest.raises(Exception) as exc_info:
            _get_bridge_feature(request)
        assert "503" in str(exc_info.value.status_code)

    @pytest.mark.asyncio
    async def test_get_bridge_feature_success(self):
        from kestrel_sovereign.features.bridge.router import _get_bridge_feature

        db = _make_db()
        agent = _make_agent(db=db)
        bridge = BridgeFeature(agent)
        await bridge.initialize()
        agent.features = {"BridgeFeature": bridge}

        request = MagicMock()
        request.app.state.agent = agent

        resolved_agent, resolved_bridge = _get_bridge_feature(request)
        assert resolved_agent is agent
        assert resolved_bridge is bridge


# ============================================================================
# Router get_router() Tests
# ============================================================================


class TestGetRouter:
    def test_returns_api_router(self):
        from kestrel_sovereign.features.bridge.router import get_router

        router = get_router()
        # Check it's an APIRouter with the expected prefix
        assert router.prefix == "/api/bridge"
        assert "bridge" in router.tags

    def test_has_expected_routes(self):
        from kestrel_sovereign.features.bridge.router import get_router

        router = get_router()
        route_paths = {r.path for r in router.routes}

        # Routes include the router prefix
        assert "/api/bridge/invoke" in route_paths
        assert "/api/bridge/stream" in route_paths
        assert "/api/bridge/capabilities" in route_paths
        assert "/api/bridge/health" in route_paths
        assert "/api/bridge/session" in route_paths


# ============================================================================
# Graceful Degradation Tests
# ============================================================================


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_all_commands_work_without_db(self):
        """All tool commands work even without a database."""
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = BridgeFeature(agent)
        await feat.initialize()

        # bridge_status
        status = await feat.bridge_status()
        assert status["status"] == "active"
        assert status["database_available"] is False

        # bridge_connections (empty)
        conns = await feat.bridge_connections()
        assert conns["count"] == 0

        # bridge_history (empty)
        history = await feat.bridge_history()
        assert history["count"] == 0

    @pytest.mark.asyncio
    async def test_session_works_without_db(self):
        """Session management works with in-memory only."""
        agent = _make_agent(db=None)
        agent.storage.db = None
        feat = BridgeFeature(agent)
        await feat.initialize()

        session = await feat.get_or_create_session(
            gateway_session_id="gw-no-db"
        )
        assert session.id is not None

        # Can resume from memory
        session2 = await feat.get_or_create_session(
            gateway_session_id="gw-no-db"
        )
        assert session.id == session2.id

    @pytest.mark.asyncio
    async def test_db_error_on_session_create_does_not_crash(self):
        """If DB persist fails during session creation, we still get a session."""
        db = _make_db()
        agent = _make_agent(db=db)
        feat = BridgeFeature(agent)
        await feat.initialize()

        # Make INSERT fail
        feat._db.execute = AsyncMock(side_effect=Exception("DB error"))

        session = await feat.get_or_create_session(
            gateway_session_id="gw-error"
        )
        # Should still return a valid in-memory session
        assert session.id is not None
        assert "gw-error" in feat._sessions
