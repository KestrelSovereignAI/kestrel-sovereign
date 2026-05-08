"""
Unit Tests for Consent Protocol Strict Timeout.

Tests:
- Timeout after CONSENT_TIMEOUT_SECONDS triggers fail-open
- Timeout is recorded in consent_log with timed_out=True
- duration_ms is tracked for successful requests
- Errors trigger fail-open (action proceeds)
- consent_stats includes timeout rate, avg duration, p95 duration
"""

import asyncio
import json
import time

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

from kestrel_sovereign.features.consent.models import ConsentRecord
from kestrel_sovereign.features.consent.feature import (
    ConsentFeature,
    CONSENT_TIMEOUT_SECONDS,
)


# =========================================================================
# Fixtures
# =========================================================================


def _make_mock_agent(db=None):
    """Build a mock agent with storage.db and llm_service."""
    agent = MagicMock()
    agent.agent_id = "did:test:consent-agent"
    agent.features = {}

    mock_db = db or AsyncMock()
    if db is None:
        mock_db.execute = AsyncMock()
        mock_db.fetchall = AsyncMock(return_value=[])
        mock_db.fetchone = AsyncMock(return_value=None)

    agent.storage = MagicMock()
    agent.storage.db = mock_db

    agent.llm_service = MagicMock()
    agent.llm_service.generate = AsyncMock(
        return_value="This change seems reasonable and fine."
    )

    return agent


@pytest_asyncio.fixture
async def consent_feature():
    """Create and initialize a ConsentFeature with mocked agent."""
    agent = _make_mock_agent()
    feature = ConsentFeature(agent)
    await feature.initialize()
    return feature


# =========================================================================
# test_consent_timeout_proceeds
# =========================================================================


class TestConsentTimeoutProceeds:
    """Verify that a slow LLM triggers a timeout and the caller gets None."""

    @pytest.mark.asyncio
    async def test_consent_timeout_proceeds(self):
        """Mock LLM to take 10s, verify timeout fires and returns None."""
        agent = _make_mock_agent()

        async def slow_llm(**kwargs):
            await asyncio.sleep(10)  # Well beyond CONSENT_TIMEOUT_SECONDS
            return "I approve."

        agent.llm_service.generate = AsyncMock(side_effect=slow_llm)
        feature = ConsentFeature(agent)
        await feature.initialize()

        start = time.monotonic()
        result = await feature.request_consent(
            "privacy_mode_change",
            {"from": "normal", "to": "ephemeral"},
        )
        elapsed = time.monotonic() - start

        # Should return None (fail-open)
        assert result is None
        # Should have timed out in roughly CONSENT_TIMEOUT_SECONDS, not 10s
        assert elapsed < CONSENT_TIMEOUT_SECONDS + 1.0

    @pytest.mark.asyncio
    async def test_consent_timeout_under_limit_succeeds(self):
        """LLM that responds quickly should succeed normally."""
        agent = _make_mock_agent()

        async def fast_llm(**kwargs):
            await asyncio.sleep(0.01)
            return "This is fine and I approve."

        agent.llm_service.generate = AsyncMock(side_effect=fast_llm)
        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.request_consent(
            "model_change",
            {"from": "gpt-4", "to": "llama3"},
        )

        assert result is not None
        assert result.action_type == "model_change"
        assert result.timed_out is False
        assert result.duration_ms is not None
        assert result.duration_ms > 0


# =========================================================================
# test_consent_timeout_recorded
# =========================================================================


class TestConsentTimeoutRecorded:
    """Verify that timeout events are persisted to consent_log."""

    @pytest.mark.asyncio
    async def test_consent_timeout_recorded(self):
        """On timeout, a record with timed_out=True is stored."""
        agent = _make_mock_agent()

        async def slow_llm(**kwargs):
            await asyncio.sleep(10)
            return "I approve."

        agent.llm_service.generate = AsyncMock(side_effect=slow_llm)
        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.request_consent(
            "safe_mode_entry",
            {"reason": "audit failure"},
        )

        assert result is None

        # Find the INSERT call for the timeout record
        insert_calls = [
            c for c in agent.storage.db.execute.call_args_list
            if c.args and "INSERT INTO consent_log" in str(c.args[0])
        ]
        assert len(insert_calls) >= 1

        # The most recent INSERT should be the timeout record
        last_insert = insert_calls[-1]
        params = last_insert.args[1]  # positional tuple
        # params layout: (id, agent_id, action_type, action_details,
        #                  agent_view, agent_sentiment, sovereign_proceeded,
        #                  sovereign_override_reason, duration_ms, timed_out, created_at)
        agent_view = params[4]
        agent_sentiment = params[5]
        duration_ms = params[8]
        timed_out = params[9]

        assert agent_view == "[TIMEOUT]"
        assert agent_sentiment == "timeout"
        assert duration_ms == CONSENT_TIMEOUT_SECONDS * 1000
        assert timed_out == 1  # stored as integer


# =========================================================================
# test_consent_duration_tracked
# =========================================================================


class TestConsentDurationTracked:
    """Verify that duration_ms is recorded for successful requests."""

    @pytest.mark.asyncio
    async def test_consent_duration_tracked(self):
        """duration_ms should be a positive number for a successful consent."""
        agent = _make_mock_agent()

        async def timed_llm(**kwargs):
            await asyncio.sleep(0.05)  # 50ms
            return "This is fine and reasonable."

        agent.llm_service.generate = AsyncMock(side_effect=timed_llm)
        feature = ConsentFeature(agent)
        await feature.initialize()

        record = await feature.request_consent(
            "model_change",
            {"from": "gpt-4", "to": "claude-3"},
        )

        assert record is not None
        assert record.duration_ms is not None
        assert record.duration_ms >= 40  # At least ~40ms given the sleep
        assert record.timed_out is False

        # Verify the INSERT stored the duration
        insert_calls = [
            c for c in agent.storage.db.execute.call_args_list
            if c.args and "INSERT INTO consent_log" in str(c.args[0])
        ]
        assert len(insert_calls) >= 1
        last_insert = insert_calls[-1]
        params = last_insert.args[1]
        stored_duration = params[8]
        stored_timed_out = params[9]

        assert stored_duration >= 40
        assert stored_timed_out == 0


# =========================================================================
# test_consent_fail_open_on_error
# =========================================================================


class TestConsentFailOpenOnError:
    """Verify that LLM errors result in None and the action proceeds."""

    @pytest.mark.asyncio
    async def test_consent_fail_open_on_error(self):
        """When the LLM raises, request_consent returns None."""
        agent = _make_mock_agent()
        agent.llm_service.generate = AsyncMock(
            side_effect=RuntimeError("LLM service unavailable")
        )
        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.request_consent(
            "privacy_mode_change",
            {"from": "normal", "to": "isolated"},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_consent_fail_open_on_storage_error(self):
        """When storage INSERT fails, request_consent returns None."""
        agent = _make_mock_agent()

        async def insert_fails(*args, **kwargs):
            if args and "INSERT INTO consent_log" in str(args[0]):
                raise RuntimeError("DB write failed")
            # Let CREATE TABLE / ALTER TABLE succeed

        agent.storage.db.execute = AsyncMock(side_effect=insert_fails)
        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.request_consent(
            "safe_mode_entry",
            {"reason": "test"},
        )

        assert result is None


# =========================================================================
# test_consent_stats_includes_metrics
# =========================================================================


class TestConsentStatsIncludesMetrics:
    """Verify consent_stats exposes timeout rate and avg duration."""

    @pytest.mark.asyncio
    async def test_consent_stats_includes_metrics(self):
        """Stats should include avg_duration_ms, timeout_count, timeout_rate, etc."""
        agent = _make_mock_agent()

        fetchall_responses = {
            "action_type": [
                ("privacy_mode_change", 3),
                ("model_change", 2),
            ],
            "agent_sentiment": [
                ("positive", 3),
                ("concerned", 1),
                ("timeout", 1),
            ],
        }

        async def mock_fetchall(query, *args):
            if "action_type" in query:
                return fetchall_responses["action_type"]
            elif "agent_sentiment" in query:
                return fetchall_responses["agent_sentiment"]
            return []

        agent.storage.db.fetchall = AsyncMock(side_effect=mock_fetchall)

        # Mock fetchone calls in order:
        # 1. Total count
        # 2. AVG(duration_ms)
        # 3. COUNT where duration_ms IS NOT NULL (for p95 calc)
        # 4. P95 duration row
        # 5. Timeout count
        # 6. Error count (same query as timeout in current impl)
        agent.storage.db.fetchone = AsyncMock(
            side_effect=[
                (5,),        # total
                (120.5,),    # avg duration
                (4,),        # count with duration
                (250.0,),    # p95 duration
                (1,),        # timeout count
                (1,),        # error count
            ]
        )

        feature = ConsentFeature(agent)
        await feature.initialize()

        from kestrel_sdk.tools.result import ToolResultStatus
        result = await feature.consent_stats()

        # 5 records is below the threshold (>=10) for the
        # "high timeout rate" PARTIAL surface, so this is OK.
        assert result.status is ToolResultStatus.OK
        assert result.data["total"] == 5
        assert result.data["avg_duration_ms"] == 120.5
        assert result.data["p95_duration_ms"] == 250.0
        assert result.data["timeout_count"] == 1
        assert result.data["timeout_rate"] == 0.2  # 1/5
        assert result.data["error_count"] == 1
        assert result.data["error_rate"] == 0.2
        assert result.data["by_action"]["privacy_mode_change"] == 3
        assert result.data["by_sentiment"]["timeout"] == 1

    @pytest.mark.asyncio
    async def test_consent_stats_no_duration_data(self):
        """Stats handle the case where no duration data exists (all NULL)."""
        agent = _make_mock_agent()

        agent.storage.db.fetchall = AsyncMock(return_value=[])
        agent.storage.db.fetchone = AsyncMock(
            side_effect=[
                (0,),     # total
                (None,),  # avg duration (no rows)
                (0,),     # count with duration
                # p95 query skipped since duration_count=0
                (0,),     # timeout count
                (0,),     # error count
            ]
        )

        feature = ConsentFeature(agent)
        await feature.initialize()

        from kestrel_sdk.tools.result import ToolResultStatus
        result = await feature.consent_stats()

        assert result.status is ToolResultStatus.OK
        assert result.data["total"] == 0
        assert result.data["avg_duration_ms"] is None
        assert result.data["p95_duration_ms"] is None
        assert result.data["timeout_count"] == 0
        assert result.data["timeout_rate"] == 0.0
        assert result.data["error_count"] == 0
        assert result.data["error_rate"] == 0.0


# =========================================================================
# Integration point verification
# =========================================================================


class TestIntegrationPointsFailOpen:
    """
    Verify that callers of request_consent handle None returns correctly,
    so the change always proceeds regardless of consent outcome.
    """

    @pytest.mark.asyncio
    async def test_set_privacy_mode_proceeds_on_consent_none(self):
        """
        set_privacy_mode in kestrel_agent.py uses fire-and-forget (create_task)
        when the loop is running. Verify the pattern handles None.
        """
        # This tests the contract: request_consent returning None should not
        # raise or block. We verify by calling it and checking None is fine.
        agent = _make_mock_agent()
        agent.llm_service.generate = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        feature = ConsentFeature(agent)
        await feature.initialize()

        # request_consent catches TimeoutError internally via wait_for
        result = await feature.request_consent(
            "privacy_mode_change",
            {"from": "normal", "to": "ephemeral"},
        )
        assert result is None
        # The caller would proceed -- no exception raised

    @pytest.mark.asyncio
    async def test_enter_safe_mode_proceeds_on_consent_none(self):
        """
        enter_safe_mode wraps request_consent in try/except and always
        proceeds. Verify the consent side behaves correctly.
        """
        agent = _make_mock_agent()
        agent.llm_service.generate = AsyncMock(
            side_effect=RuntimeError("Connection refused")
        )
        feature = ConsentFeature(agent)
        await feature.initialize()

        result = await feature.request_consent(
            "safe_mode_entry",
            {"reason": "constitution audit failure"},
        )
        assert result is None


# =========================================================================
# CONSENT_TIMEOUT_SECONDS constant
# =========================================================================


class TestConsentTimeoutConstant:
    """Verify the timeout constant is set correctly."""

    def test_timeout_is_five_seconds(self):
        assert CONSENT_TIMEOUT_SECONDS == 5.0

    def test_timeout_is_positive(self):
        assert CONSENT_TIMEOUT_SECONDS > 0
