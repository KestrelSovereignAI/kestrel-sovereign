"""Consistency tests for canonical privacy presets."""

from unittest.mock import MagicMock

from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.features.privacy import PrivacyAgent
from kestrel_sovereign.privacy import (
    PRIVACY_PRESETS,
    PrivacyMode,
    get_privacy_preset,
    privacy_config_to_mode,
    privacy_mode_to_config,
)


def test_privacy_presets_match_canonical_flag_combinations():
    assert get_privacy_preset("ephemeral").storage == "none"
    assert get_privacy_preset("ephemeral").llm_location == "local"
    assert get_privacy_preset("ephemeral").shareable is False

    assert get_privacy_preset("isolated").storage == "temp"
    assert get_privacy_preset("isolated").llm_location == "local"
    assert get_privacy_preset("isolated").shareable is False

    assert get_privacy_preset("anonymous").storage == "pii_redacted"
    assert get_privacy_preset("anonymous").processing == "local"
    assert get_privacy_preset("anonymous").llm_location == "local"
    assert get_privacy_preset("anonymous").allows_cloud_llm() is False
    assert get_privacy_preset("anonymous").shareable is False
    assert get_privacy_preset("anonymous").assurance == "pii_redacted"

    assert get_privacy_preset("normal").storage == "full"
    assert get_privacy_preset("normal").llm_location == "cloud"
    assert get_privacy_preset("normal").shareable is False

    assert get_privacy_preset("public").storage == "full"
    assert get_privacy_preset("public").llm_location == "cloud"
    assert get_privacy_preset("public").shareable is True
    assert get_privacy_preset("public").sharing == "public"

    assert get_privacy_preset("deidentified").storage == "deidentified"
    assert get_privacy_preset("deidentified").processing == "trusted"
    assert get_privacy_preset("deidentified").sharing == "research"
    assert get_privacy_preset("deidentified").assurance == "safe_harbor"
    assert get_privacy_preset("deidentified").audit == "required"
    assert get_privacy_preset("deidentified").requires_audit() is True


def test_scrubbed_storage_is_legacy_alias_for_pii_redacted():
    from kestrel_sovereign.privacy import PrivacyConfig

    cfg = PrivacyConfig(storage="scrubbed", llm_location="local", shareable=False)

    assert cfg.storage == "pii_redacted"
    assert cfg.requires_anonymization() is True
    assert privacy_config_to_mode(cfg) == PrivacyMode.ANONYMOUS


def test_legacy_deidentified_aliases_do_not_downgrade_to_normal():
    from kestrel_sovereign.privacy import PrivacyConfig

    cfg = PrivacyConfig(storage="deidentified", llm_location="local", shareable=False)

    assert cfg.processing == "trusted"
    assert cfg.sharing == "research"
    assert cfg.assurance == "safe_harbor"
    assert cfg.audit == "required"
    assert privacy_config_to_mode(cfg) == PrivacyMode.DEIDENTIFIED


def test_privacy_mode_round_trips_to_named_presets():
    for mode in PrivacyMode:
        config = privacy_mode_to_config(mode)
        assert privacy_config_to_mode(config) == mode
        assert config == PRIVACY_PRESETS[mode.value]


def test_privacy_agent_set_mode_reports_canonical_meanings():
    agent = PrivacyAgent(MagicMock())

    assert "local LLM only" in agent.set_mode("ephemeral")
    assert "temporary session storage" in agent.set_mode("isolated")
    assert "PII removed" in agent.set_mode("anonymous")
    assert "local LLM only" in agent.set_mode("anonymous")
    assert "standard persistence" in agent.set_mode("normal")
    assert "shared and exported publicly" in agent.set_mode("public")
    assert "Safe Harbor evidence" in agent.set_mode("deidentified")


def test_privacy_agent_handle_get_privacy_mode_matches_canonical_descriptions():
    """Privacy mode display logic lives in PrivacyAgent.handle_get_privacy_mode()."""
    expectations = {
        PrivacyMode.EPHEMERAL: "Nothing stored, local LLM only",
        PrivacyMode.ISOLATED: "Temporary session storage, local LLM only",
        PrivacyMode.ANONYMOUS: "Local LLM only, stored with PII removed",
        PrivacyMode.NORMAL: "Standard persistent storage",
        PrivacyMode.PUBLIC: "Shareable and exportable",
        PrivacyMode.DEIDENTIFIED: "Research sharing with Safe Harbor evidence",
    }

    for mode, expected in expectations.items():
        pa = PrivacyAgent(MagicMock(), initial_mode=mode)
        status = pa.handle_get_privacy_mode()
        assert expected in status, f"Expected '{expected}' in status for {mode}, got: {status}"


def test_command_handler_delegates_get_privacy_mode_to_privacy_agent():
    """CommandHandler delegates to PrivacyAgent.handle_get_privacy_mode()."""
    pa = PrivacyAgent(MagicMock(), initial_mode=PrivacyMode.NORMAL)
    agent = MagicMock()
    agent.privacy_agent = pa
    handler = CommandHandler(agent)

    status = handler._cmd_get_privacy_mode("!get-privacy-mode")
    assert "Standard persistent storage" in status


def test_deidentified_status_reports_evidence_required_not_persistent():
    pa = PrivacyAgent(MagicMock(), initial_mode=PrivacyMode.DEIDENTIFIED)

    status = pa.get_detailed_status()

    assert status["privacy_mode"] == "deidentified"
    assert status["persistent_storage"] is False
    assert status["backup_status"] == "disabled"
    assert status["backup_encryption"] == "required"
    assert status["privacy_config"]["assurance"] == "safe_harbor"
    assert status["privacy_config"]["audit"] == "required"
