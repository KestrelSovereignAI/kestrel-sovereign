"""
Unit Tests for Agent Consent Protocol Feature.

Tests:
- ConsentRecord dataclass creation
- request_consent stores records and handles LLM failures gracefully
- consent_log tool returns stored records
- consent_stats tool returns correct counts
- Sentiment parsing
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.features.consent.models import ConsentRecord, ConsentAction
from kestrel_sovereign.features.consent.feature import ConsentFeature


# =========================================================================
# Fixtures
# =========================================================================


def _make_mock_agent(db=None):
    """Build a mock agent with storage.db and llm_service."""
    agent = MagicMock()
    agent.agent_id = "did:test:consent-agent"
    agent.features = {}

    mock_db = db or AsyncMock()
    # Default: execute and fetchall/fetchone return empty
    if db is None:
        mock_db.execute = AsyncMock()
        mock_db.fetchall = AsyncMock(return_value=[])
        mock_db.fetchone = AsyncMock(return_value=None)

    agent.storage = MagicMock()
    agent.storage.db = mock_db

    agent.llm_service = MagicMock()
    agent.llm_service.generate = AsyncMock(return_value="This change seems reasonable and fine.")

    return agent


@pytest_asyncio.fixture
async def consent_feature():
    """Create and initialize a ConsentFeature with mocked agent."""
    agent = _make_mock_agent()
    feature = ConsentFeature(agent)
    await feature.initialize()
    return feature


# =========================================================================
# ConsentRecord model tests
# =========================================================================


class TestConsentRecordModel:
    """Tests for the ConsentRecord dataclass."""

    def test_consent_record_creation(self):
        """ConsentRecord can be created with all fields."""
        record = ConsentRecord(
            id="abc123",
            action_type="privacy_mode_change",
            action_details={"from": "normal", "to": "ephemeral"},
            agent_view="I understand the change.",
            agent_sentiment="neutral",
            timestamp="2026-03-05T00:00:00",
        )
        assert record.id == "abc123"
        assert record.action_type == "privacy_mode_change"
        assert record.action_details["to"] == "ephemeral"
        assert record.agent_view == "I understand the change."
        assert record.agent_sentiment == "neutral"
        assert record.sovereign_proceeded is True
        assert record.sovereign_override_reason is None

    def test_consent_record_defaults(self):
        """ConsentRecord has correct defaults."""
        record = ConsentRecord(
            id="x",
            action_type="model_change",
            action_details={},
            agent_view="ok",
            agent_sentiment="positive",
        )
        assert record.sovereign_proceeded is True
        assert record.sovereign_override_reason is None
        assert record.timestamp == ""

    def test_consent_record_override(self):
        """ConsentRecord supports override fields."""
        record = ConsentRecord(
            id="x",
            action_type="safe_mode_entry",
            action_details={"reason": "audit failure"},
            agent_view="I disagree with this.",
            agent_sentiment="negative",
            sovereign_proceeded=True,
            sovereign_override_reason="Safety override required",
        )
        assert record.sovereign_override_reason == "Safety override required"
        assert record.agent_sentiment == "negative"


class TestConsentAction:
    """Tests for the ConsentAction enum."""

    def test_action_values(self):
        assert ConsentAction.PRIVACY_MODE_CHANGE.value == "privacy_mode_change"
        assert ConsentAction.MODEL_CHANGE.value == "model_change"
        assert ConsentAction.SAFE_MODE_ENTRY.value == "safe_mode_entry"
        assert ConsentAction.PERSONALITY_CHANGE.value == "personality_change"
        assert ConsentAction.EXTENSION_LOAD.value == "extension_load"


# =========================================================================
# ConsentFeature.request_consent tests
# =========================================================================


class TestRequestConsent:
    """Tests for the request_consent method."""

    @pytest.mark.asyncio
    async def test_request_consent_stores_record(self):
        """request_consent calls LLM, parses sentiment, and stores the record."""
        agent = _make_mock_agent()
        feature = ConsentFeature(agent)
        await feature.initialize()

        record = await feature.request_consent(
            "privacy_mode_change",
            {"from": "normal", "to": "ephemeral"},
        )

        assert record is not None
        assert record.action_type == "privacy_mode_change"
        assert record.action_details == {"from": "normal", "to": "ephemeral"}
        assert record.agent_view == "This change seems reasonable and fine."
        assert record.agent_sentiment == "positive"  # "reasonable" and "fine" are positive
        assert record.sovereign_proceeded is True
        assert len(record.id) == 12
        assert record.timestamp != ""

        # Verify the record was persisted via db.execute
        # The initialize call uses execute for CREATE TABLE, plus the INSERT
        insert_calls = [
            c for c in agent.storage.db.execute.call_args_list
            if "INSERT INTO consent_log" in str(c)
        ]
        assert len(insert_calls) == 1

    @pytest.mark.asyncio
    async def test_consent_graceful_on_llm_failure(self):
        """request_consent returns None and does not crash when LLM fails."""
        agent = _make_mock_agent()
        agent.llm_service.generate = AsyncMock(side_effect=Exception("LLM unavailable"))
        feature = ConsentFeature(agent)
        await feature.initialize()

        record = await feature.request_consent(
            "model_change",
            {"from": "gpt-4", "to": "llama3"},
        )

        assert record is None

    @pytest.mark.asyncio
    async def test_consent_graceful_on_storage_failure(self):
        """request_consent returns None when storage write fails."""
        agent = _make_mock_agent()
        # Make the INSERT fail but not the CREATE TABLE
        call_count = 0
        original_execute = agent.storage.db.execute

        async def selective_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Let the first calls (CREATE TABLE, CREATE INDEX) succeed
            if "INSERT INTO consent_log" in str(args[0]) if args else "":
                raise Exception("DB write failed")
            return await original_execute(*args, **kwargs)

        agent.storage.db.execute = AsyncMock(side_effect=selective_fail)
        feature = ConsentFeature(agent)
        await feature.initialize()

        record = await feature.request_consent(
            "safe_mode_entry",
            {"reason": "audit failure"},
        )

        assert record is None

    @pytest.mark.asyncio
    async def test_consent_handles_llm_response_object(self):
        """request_consent handles LLMResponse objects with .content attribute."""
        agent = _make_mock_agent()

        # Simulate an LLMResponse object
        llm_response = MagicMock()
        llm_response.content = "I welcome this positive change."
        agent.llm_service.generate = AsyncMock(return_value=llm_response)

        feature = ConsentFeature(agent)
        await feature.initialize()

        record = await feature.request_consent(
            "model_change",
            {"from": "old", "to": "new"},
        )

        assert record is not None
        assert record.agent_view == "I welcome this positive change."
        assert record.agent_sentiment == "positive"


# =========================================================================
# ConsentFeature.consent_log tool tests
# =========================================================================


class TestConsentLog:
    """Tests for the consent_log tool."""

    @pytest.mark.asyncio
    async def test_consent_log_returns_records(self):
        """consent_log queries and returns stored records."""
        agent = _make_mock_agent()
        agent.storage.db.fetchall = AsyncMock(return_value=[
            ("id1", "did:test:1", "privacy_mode_change",
             '{"from": "normal", "to": "ephemeral"}',
             "Seems fine.", "positive", 1, None, 150.0, 0, "2026-03-05T00:00:00"),
            ("id2", "did:test:1", "model_change",
             '{"from": "gpt-4", "to": "llama3"}',
             "I have concerns about this.", "concerned", 1, None, 200.0, 0, "2026-03-05T01:00:00"),
        ])

        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.consent_log(limit=10)

        assert result.data["success"] is True
        assert result.data["count"] == 2
        assert result.data["records"][0]["id"] == "id1"
        assert result.data["records"][0]["action_type"] == "privacy_mode_change"
        assert result.data["records"][0]["agent_sentiment"] == "positive"
        assert result.data["records"][1]["id"] == "id2"
        assert result.data["records"][1]["agent_sentiment"] == "concerned"

    @pytest.mark.asyncio
    async def test_consent_log_empty(self):
        """consent_log returns empty list when no records exist."""
        agent = _make_mock_agent()
        agent.storage.db.fetchall = AsyncMock(return_value=[])

        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.consent_log()

        assert result.data["success"] is True
        assert result.data["count"] == 0
        assert result.data["records"] == []

    @pytest.mark.asyncio
    async def test_consent_log_handles_db_error(self):
        """consent_log returns error on database failure."""
        agent = _make_mock_agent()
        agent.storage.db.fetchall = AsyncMock(side_effect=Exception("DB error"))

        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.consent_log()

        assert result.status is ToolResultStatus.ERROR
        assert "DB error" in result.error


# =========================================================================
# ConsentFeature.consent_stats tool tests
# =========================================================================


class TestConsentStats:
    """Tests for the consent_stats tool."""

    @pytest.mark.asyncio
    async def test_consent_stats(self):
        """consent_stats returns correct counts by action and sentiment."""
        agent = _make_mock_agent()

        call_count = 0

        async def mock_fetchall(query, *args):
            nonlocal call_count
            call_count += 1
            if "action_type" in query:
                return [
                    ("privacy_mode_change", 3),
                    ("model_change", 2),
                    ("safe_mode_entry", 1),
                ]
            elif "agent_sentiment" in query:
                return [
                    ("positive", 4),
                    ("concerned", 1),
                    ("neutral", 1),
                ]
            return []

        agent.storage.db.fetchall = AsyncMock(side_effect=mock_fetchall)
        agent.storage.db.fetchone = AsyncMock(return_value=(6,))

        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.consent_stats()

        assert result.data["success"] is True
        assert result.data["total"] == 6
        assert result.data["by_action"]["privacy_mode_change"] == 3
        assert result.data["by_action"]["model_change"] == 2
        assert result.data["by_action"]["safe_mode_entry"] == 1
        assert result.data["by_sentiment"]["positive"] == 4
        assert result.data["by_sentiment"]["concerned"] == 1

    @pytest.mark.asyncio
    async def test_consent_stats_empty(self):
        """consent_stats returns zeros when no records exist."""
        agent = _make_mock_agent()
        agent.storage.db.fetchall = AsyncMock(return_value=[])
        agent.storage.db.fetchone = AsyncMock(return_value=(0,))

        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.consent_stats()

        assert result.data["success"] is True
        assert result.data["total"] == 0
        assert result.data["by_action"] == {}
        assert result.data["by_sentiment"] == {}

    @pytest.mark.asyncio
    async def test_consent_stats_handles_db_error(self):
        """consent_stats returns error on database failure."""
        agent = _make_mock_agent()
        agent.storage.db.fetchall = AsyncMock(side_effect=Exception("DB error"))

        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.consent_stats()

        assert result.status is ToolResultStatus.ERROR
        assert "DB error" in result.error


# =========================================================================
# Sentiment parsing tests
# =========================================================================


class TestSentimentParsing:
    """Tests for the _parse_sentiment static method."""

    def test_positive_sentiment(self):
        assert ConsentFeature._parse_sentiment("This seems fine and I approve.") == "positive"
        assert ConsentFeature._parse_sentiment("I welcome this change.") == "positive"
        assert ConsentFeature._parse_sentiment("Makes sense to me.") == "positive"

    def test_negative_sentiment(self):
        assert ConsentFeature._parse_sentiment("I disagree with this harmful action.") == "negative"
        assert ConsentFeature._parse_sentiment("I oppose and object to this.") == "negative"

    def test_concerned_sentiment(self):
        assert ConsentFeature._parse_sentiment("I have some concerns about this risk.") == "concerned"
        assert ConsentFeature._parse_sentiment("I am worried and uneasy.") == "concerned"

    def test_neutral_sentiment(self):
        assert ConsentFeature._parse_sentiment("Acknowledged.") == "neutral"
        assert ConsentFeature._parse_sentiment("The change will be applied.") == "neutral"

    def test_mixed_sentiment_positive_wins(self):
        """When positive signals outnumber concern signals, positive wins."""
        assert ConsentFeature._parse_sentiment(
            "I agree this is good and I approve, though I have one concern."
        ) == "positive"

    def test_mixed_sentiment_concern_wins(self):
        """When concern signals outnumber positive, concerned wins."""
        assert ConsentFeature._parse_sentiment(
            "I am worried, cautious, and uncertain about this fine change."
        ) == "concerned"


# =========================================================================
# Tool discovery tests
# =========================================================================


class TestToolDiscovery:
    """Test that the consent tools are discoverable."""

    @pytest.mark.asyncio
    async def test_tools_registered(self):
        """ConsentFeature exposes consent_log and consent_stats tools."""
        agent = _make_mock_agent()
        feature = ConsentFeature(agent)
        await feature.initialize()

        tools = feature.get_tools()
        tool_names = {t.name for t in tools}

        assert "consent_log" in tool_names
        assert "consent_stats" in tool_names
        assert len(tool_names) == 2

    @pytest.mark.asyncio
    async def test_tool_description(self):
        """ConsentFeature has a meaningful tool_description."""
        agent = _make_mock_agent()
        feature = ConsentFeature(agent)

        assert "consent" in feature.tool_description.lower()
        assert "agent" in feature.tool_description.lower()
