"""Per-agent ``[privacy] computer_access`` opt-in via kestrel.toml (#956).

Two real bugs this PR fixes:

1. ``PrivacyConfig.computer_access`` defaults False and the design comment
   in ``privacy.py`` says it "must be opted into explicitly by setting the
   flag after preset construction" — but no production code did that
   until now. There was no way for the Sovereign to enable computer-use
   on a running agent.

2. ``ComputerUseFeature._privacy_allows`` does
   ``getattr(self.agent, "privacy_config", None)``, but the agent stored
   the config on ``agent.privacy_agent.privacy_config`` (different
   attribute path), so the lookup always returned None and the gate
   always denied. Adding the ``KestrelAgent.privacy_config`` property
   makes the lookup succeed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pytest

from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.privacy import PrivacyConfig, PrivacyMode


def _make_agent(storage_path: Path | None) -> KestrelAgent:
    return KestrelAgent(
        did="did:test:privacy",
        privacy_mode=PrivacyMode.NORMAL,
        storage_path=str(storage_path) if storage_path else None,
    )


class TestTomlOptInRead:
    def test_toml_with_computer_access_true(self, tmp_path):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "kestrel.toml").write_text(
            '[agent]\nname = "Test"\n\n[privacy]\ncomputer_access = true\n'
        )
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        assert agent._privacy_computer_access is True

    def test_toml_without_privacy_section_defaults_false(self, tmp_path):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "kestrel.toml").write_text('[agent]\nname = "Test"\n')
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        assert agent._privacy_computer_access is False

    def test_toml_with_computer_access_false(self, tmp_path):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "kestrel.toml").write_text(
            '[privacy]\ncomputer_access = false\n'
        )
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        assert agent._privacy_computer_access is False

    def test_no_toml_file_defaults_false(self, tmp_path):
        agent_dir = tmp_path / "agent_no_toml"
        agent_dir.mkdir()
        agent = _make_agent(agent_dir / "kestrel_prime.db")
        assert agent._privacy_computer_access is False

    def test_no_storage_path_defaults_false(self):
        agent = _make_agent(storage_path=None)
        assert agent._privacy_computer_access is False

    def test_malformed_toml_warns_and_defaults_false(self, tmp_path, caplog):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "kestrel.toml").write_text("this is = not [valid toml at all")
        with caplog.at_level(logging.WARNING):
            agent = _make_agent(agent_dir / "kestrel_prime.db")
        assert agent._privacy_computer_access is False
        assert any(
            "[privacy]" in rec.getMessage() for rec in caplog.records
        )


class TestPrivacyConfigProperty:
    """``KestrelAgent.privacy_config`` must delegate to
    ``self.privacy_agent.privacy_config`` once initialized — that's the
    attribute path ``ComputerUseFeature._privacy_allows`` reads."""

    def test_property_returns_none_before_initialize(self, tmp_path):
        # Before ``initialize()``, ``self.privacy_agent`` is None — the
        # property should be defensive about that and return None instead
        # of raising AttributeError.
        agent = _make_agent(tmp_path / "kestrel_prime.db")
        assert agent.privacy_agent is None
        assert agent.privacy_config is None

    @pytest.mark.asyncio
    async def test_toml_opt_in_enables_computer_access_after_initialize(
        self, tmp_path, monkeypatch
    ):
        # End-to-end: kestrel.toml sets computer_access=true → after
        # ``initialize()``, ``agent.privacy_config.computer_access`` is True.
        # This is the invariant ``ComputerUseFeature._privacy_allows`` needs.
        agent_dir = tmp_path / "live_agent"
        agent_dir.mkdir()
        (agent_dir / "kestrel.toml").write_text('[privacy]\ncomputer_access = true\n')

        # Avoid heavyweight feature discovery + LLM init by stubbing them.
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
        from kestrel_sovereign.storage.async_storage import AsyncStorage

        class _StubLLMService:
            async def initialize(self): ...
            def get_active_model_id(self): return "stub"
            def list_routes(self): return []
            async def shutdown(self): ...

        agent = KestrelAgent(
            did="did:test:e2e",
            privacy_mode=PrivacyMode.NORMAL,
            storage_path=str(agent_dir / "kestrel_prime.db"),
            llm_service=_StubLLMService(),
        )
        # Just exercise the privacy-init slice, not the whole initialize().
        # Replicating that slice keeps the test fast and focused on #956.
        agent._raw_storage = AsyncStorage(
            str(agent_dir / "kestrel_prime.db"),
            agent_id=agent.did,
        )
        await agent._raw_storage.initialize()
        agent.storage = PrivacyEnforcingStorage(agent._raw_storage, PrivacyMode.NORMAL)

        # Run the same construction logic ``initialize()`` uses for the
        # privacy agent. The branch is in kestrel_agent.py around the
        # ``self.privacy_agent = PrivacyAgent(...)`` line.
        from kestrel_sovereign.features.privacy.feature import PrivacyAgent
        if agent._privacy_computer_access:
            from kestrel_sovereign.privacy import PrivacyConfig, privacy_mode_to_config
            base_cfg = privacy_mode_to_config(agent._privacy_mode)
            opted_in = PrivacyConfig(
                storage=base_cfg.storage,
                llm_location=base_cfg.llm_location,
                shareable=base_cfg.shareable,
                computer_access=True,
            )
            agent.privacy_agent = PrivacyAgent(agent._raw_storage, opted_in)
        else:
            agent.privacy_agent = PrivacyAgent(agent._raw_storage, agent._privacy_mode)

        # The two invariants ComputerUseFeature gate 2 needs:
        cfg = agent.privacy_config
        assert cfg is not None, "privacy_config property must return the live config"
        assert cfg.computer_access is True, (
            "[privacy] computer_access = true in kestrel.toml must reach "
            "the live PrivacyConfig"
        )
        assert cfg.allows_computer_access() is True

        await agent._raw_storage.close()

    @pytest.mark.asyncio
    async def test_default_omitted_keeps_computer_access_false(self, tmp_path):
        # Mirror of above without the toml flag. Existing behavior preserved.
        agent_dir = tmp_path / "default_agent"
        agent_dir.mkdir()
        (agent_dir / "kestrel.toml").write_text('[agent]\nname = "Default"\n')

        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
        from kestrel_sovereign.storage.async_storage import AsyncStorage
        from kestrel_sovereign.features.privacy.feature import PrivacyAgent

        agent = KestrelAgent(
            did="did:test:default",
            privacy_mode=PrivacyMode.NORMAL,
            storage_path=str(agent_dir / "kestrel_prime.db"),
        )
        agent._raw_storage = AsyncStorage(
            str(agent_dir / "kestrel_prime.db"),
            agent_id=agent.did,
        )
        await agent._raw_storage.initialize()
        agent.privacy_agent = PrivacyAgent(agent._raw_storage, agent._privacy_mode)

        cfg = agent.privacy_config
        assert cfg is not None
        assert cfg.computer_access is False  # preset default — unchanged
        assert cfg.allows_computer_access() is False

        await agent._raw_storage.close()
