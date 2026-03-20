"""Contract tests for consent integration at feature call sites."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.features.model.feature import ModelAgent
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyMode


class _LoopStub:
    def __init__(self, running: bool):
        self._running = running
        self.created = []
        self.awaited = []

    def is_running(self):
        return self._running

    def create_task(self, coro):
        self.created.append(coro)
        coro.close()
        return MagicMock()

    def run_until_complete(self, coro):
        self.awaited.append(coro)
        coro.close()
        return None


def _make_consent():
    consent = MagicMock()
    consent.request_consent = AsyncMock(return_value=None)
    return consent


class TestPrivacyConsentCaller:
    def test_set_privacy_mode_requests_consent_with_sync_loop(self, tmp_path, monkeypatch):
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            privacy_mode=PrivacyMode.NORMAL,
        )
        agent.storage = MagicMock()
        agent.storage.set_privacy_mode = MagicMock()
        agent.privacy_agent = MagicMock()
        agent.privacy_agent.set_mode = MagicMock()
        consent = _make_consent()
        agent.features = {"ConsentFeature": consent}

        loop = _LoopStub(running=False)
        monkeypatch.setattr("kestrel_sovereign.kestrel_agent.asyncio.get_event_loop", lambda: loop)

        agent.set_privacy_mode(PrivacyMode.EPHEMERAL)

        assert len(loop.awaited) == 1
        consent.request_consent.assert_called_once_with(
            "privacy_mode_change",
            {"from": "normal", "to": "ephemeral"},
        )
        agent.storage.set_privacy_mode.assert_called_once_with(PrivacyMode.EPHEMERAL)
        agent.privacy_agent.set_mode.assert_called_once_with(PrivacyMode.EPHEMERAL)

    def test_set_privacy_mode_requests_consent_with_running_loop(self, tmp_path, monkeypatch):
        agent = KestrelAgent(
            did="did:test:123",
            storage_path=str(tmp_path / "test.db"),
            privacy_mode=PrivacyMode.NORMAL,
        )
        agent.storage = MagicMock()
        agent.storage.set_privacy_mode = MagicMock()
        agent.privacy_agent = MagicMock()
        agent.privacy_agent.set_mode = MagicMock()
        consent = _make_consent()
        agent.features = {"ConsentFeature": consent}

        loop = _LoopStub(running=True)
        monkeypatch.setattr("kestrel_sovereign.kestrel_agent.asyncio.get_event_loop", lambda: loop)

        agent.set_privacy_mode(PrivacyMode.ISOLATED)

        assert len(loop.created) == 1
        assert len(loop.awaited) == 0
        consent.request_consent.assert_called_once_with(
            "privacy_mode_change",
            {"from": "normal", "to": "isolated"},
        )
        agent.storage.set_privacy_mode.assert_called_once_with(PrivacyMode.ISOLATED)
        agent.privacy_agent.set_mode.assert_called_once_with(PrivacyMode.ISOLATED)


class TestModelConsentCaller:
    @pytest.mark.asyncio
    async def test_model_change_requests_consent_before_switch(self):
        parent_agent = MagicMock()
        parent_agent.storage = MagicMock()
        parent_agent.storage.get_conversation_history = AsyncMock(return_value=[])
        parent_agent.features = {"ConsentFeature": _make_consent()}

        model_agent = ModelAgent(agent=parent_agent)
        model_agent.llm_service = MagicMock()
        model_agent.llm_service.get_model_preference = MagicMock(
            return_value={"provider": "openai", "model": "gpt-5"}
        )
        model_agent.llm_service.set_model_preference = MagicMock()

        result = await model_agent.set_model("openai/gpt-5-mini")

        assert result["success"] is True
        parent_agent.features["ConsentFeature"].request_consent.assert_awaited_once_with(
            "model_change",
            {"from": "gpt-5", "to": "gpt-5-mini", "provider": "openai"},
        )
        model_agent.llm_service.set_model_preference.assert_called_once_with("gpt-5-mini", "openai")

    @pytest.mark.asyncio
    async def test_model_change_proceeds_when_consent_fails(self):
        parent_agent = MagicMock()
        parent_agent.storage = MagicMock()
        parent_agent.storage.get_conversation_history = AsyncMock(return_value=[])
        consent = _make_consent()
        consent.request_consent = AsyncMock(side_effect=RuntimeError("consent unavailable"))
        parent_agent.features = {"ConsentFeature": consent}

        model_agent = ModelAgent(agent=parent_agent)
        model_agent.llm_service = MagicMock()
        model_agent.llm_service.get_model_preference = MagicMock(
            return_value={"provider": "openai", "model": "gpt-5"}
        )
        model_agent.llm_service.set_model_preference = MagicMock()

        result = await model_agent.set_model("openai/gpt-5-mini")

        assert result["success"] is True
        model_agent.llm_service.set_model_preference.assert_called_once_with("gpt-5-mini", "openai")


class _ConstitutionHarness(ConstitutionMixin):
    def _get_timestamp(self):
        return "2026-03-19T12:00:00Z"


class TestSafeModeConsentCaller:
    @pytest.mark.asyncio
    async def test_enter_safe_mode_records_consent_and_system_event(self):
        harness = _ConstitutionHarness()
        harness.features = {"ConsentFeature": _make_consent()}
        harness.privacy_agent = MagicMock()
        harness.privacy_agent.add_conversation = AsyncMock()
        harness._safe_mode = False

        await harness.enter_safe_mode("integrity failure")

        harness.features["ConsentFeature"].request_consent.assert_awaited_once_with(
            "safe_mode_entry",
            {"reason": "integrity failure"},
        )
        assert harness._safe_mode is True
        harness.privacy_agent.add_conversation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_enter_safe_mode_proceeds_when_consent_fails(self):
        consent = _make_consent()
        consent.request_consent = AsyncMock(side_effect=RuntimeError("consent unavailable"))

        harness = _ConstitutionHarness()
        harness.features = {"ConsentFeature": consent}
        harness.privacy_agent = MagicMock()
        harness.privacy_agent.add_conversation = AsyncMock()
        harness._safe_mode = False

        await harness.enter_safe_mode("integrity failure")

        assert harness._safe_mode is True
        harness.privacy_agent.add_conversation.assert_awaited_once()
