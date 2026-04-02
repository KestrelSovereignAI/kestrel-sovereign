"""Consistency tests for canonical privacy presets."""

from unittest.mock import MagicMock

from kestrel_sovereign.command_handler import CommandHandler
from kestrel_sovereign.features.privacy import PrivacyAgent
from kestrel_sovereign.privacy import PRIVACY_PRESETS, PrivacyMode, get_privacy_preset


def test_privacy_presets_match_canonical_flag_combinations():
    assert get_privacy_preset("ephemeral").storage == "none"
    assert get_privacy_preset("ephemeral").llm_location == "local"
    assert get_privacy_preset("ephemeral").shareable is False

    assert get_privacy_preset("isolated").storage == "temp"
    assert get_privacy_preset("isolated").llm_location == "local"
    assert get_privacy_preset("isolated").shareable is False

    assert get_privacy_preset("anonymous").storage == "scrubbed"
    assert get_privacy_preset("anonymous").llm_location == "cloud"
    assert get_privacy_preset("anonymous").shareable is False

    assert get_privacy_preset("normal").storage == "full"
    assert get_privacy_preset("normal").llm_location == "cloud"
    assert get_privacy_preset("normal").shareable is False

    assert get_privacy_preset("public").storage == "full"
    assert get_privacy_preset("public").llm_location == "cloud"
    assert get_privacy_preset("public").shareable is True


def test_privacy_mode_round_trips_to_named_presets():
    for mode in PrivacyMode:
        config = mode.to_config()
        assert PrivacyMode.from_config(config) == mode
        assert config == PRIVACY_PRESETS[mode.value]


def test_privacy_agent_set_mode_reports_canonical_meanings():
    agent = PrivacyAgent(MagicMock())

    assert "local LLM only" in agent.set_mode("ephemeral")
    assert "temporary session storage" in agent.set_mode("isolated")
    assert "PII removed" in agent.set_mode("anonymous")
    assert "cloud LLM allowed" in agent.set_mode("anonymous")
    assert "standard persistence" in agent.set_mode("normal")
    assert "shared and exported publicly" in agent.set_mode("public")


def test_privacy_agent_handle_get_privacy_mode_matches_canonical_descriptions():
    """Privacy mode display logic lives in PrivacyAgent.handle_get_privacy_mode()."""
    expectations = {
        PrivacyMode.EPHEMERAL: "Nothing stored, local LLM only",
        PrivacyMode.ISOLATED: "Temporary session storage, local LLM only",
        PrivacyMode.ANONYMOUS: "Stored with PII removed, cloud LLM allowed",
        PrivacyMode.NORMAL: "Standard persistent storage",
        PrivacyMode.PUBLIC: "Shareable and exportable",
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
