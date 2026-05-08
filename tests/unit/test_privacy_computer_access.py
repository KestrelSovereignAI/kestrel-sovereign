"""Tests for the ``computer_access`` flag on ``PrivacyConfig`` (#832)."""

import pytest

from kestrel_sovereign.privacy import (
    PRIVACY_PRESETS,
    PrivacyConfig,
    PrivacyMode,
    get_privacy_preset,
    privacy_config_to_mode,
    privacy_mode_to_config,
)


def test_default_is_false():
    assert PrivacyConfig().computer_access is False


def test_allows_helper():
    assert PrivacyConfig().allows_computer_access() is False
    assert PrivacyConfig(computer_access=True).allows_computer_access() is True


@pytest.mark.parametrize("name", list(PRIVACY_PRESETS.keys()))
def test_all_presets_default_false(name):
    assert PRIVACY_PRESETS[name].computer_access is False


@pytest.mark.parametrize("name", list(PRIVACY_PRESETS.keys()))
def test_get_privacy_preset_returns_copy_with_flag_false(name):
    cfg = get_privacy_preset(name)
    assert cfg.computer_access is False


def test_privacy_mode_to_config_defaults_false():
    assert privacy_mode_to_config("normal").computer_access is False
    assert privacy_mode_to_config(PrivacyMode.NORMAL).computer_access is False


def test_from_config_ignores_computer_access():
    """Toggling computer_access must NOT change preset identity."""
    cfg = PrivacyConfig(
        storage="full", llm_location="cloud", shareable=False, computer_access=True
    )
    # Equivalent of "normal" preset, just with computer_access on.
    assert privacy_config_to_mode(cfg) is PrivacyMode.NORMAL


def test_get_privacy_preset_unknown_raises():
    with pytest.raises(ValueError):
        get_privacy_preset("not-a-preset")
