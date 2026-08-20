"""Host-side provisioning of feature configuration (#3008).

Extracting Talon broke it on every existing agent because the package
correctly requires the host to supply explicit paths and the host had no
declarative mechanism to supply them. These cover the mechanism that closes
that gap, including the spelling that silently does nothing.
"""

from __future__ import annotations

import logging

import pytest

from kestrel_sovereign.kestrel_agent import KestrelAgent


def _agent(tmp_path):
    """An agent stub carrying only what the config lookup depends on."""
    agent = KestrelAgent.__new__(KestrelAgent)
    agent.storage_path = str(tmp_path / "kestrel_prime.db")
    return agent


def _write_agent_toml(tmp_path, body: str) -> None:
    (tmp_path / "kestrel.toml").write_text(body, encoding="utf-8")


class _ConfigurableFeature:
    config_schema = {"type": "object"}

    def __init__(self):
        self.applied = None

    async def set_config(self, config):
        self.applied = config


class TalonCoordinatorFeature(_ConfigurableFeature):
    """Named to match the real class so the registry resolves it to 'talon'."""


def test_declared_config_is_addressed_by_package_name(tmp_path):
    _write_agent_toml(
        tmp_path,
        '[features.talon.config]\nconfig_path = "/srv/kestrel/kestrel.toml"\n',
    )
    declared = _agent(tmp_path)._declared_feature_config("TalonCoordinatorFeature")
    assert declared == {"config_path": "/srv/kestrel/kestrel.toml"}


def test_class_name_spelling_is_not_read_but_is_reported(tmp_path, caplog):
    """The confusable spelling must not fail silently — that is the bug."""
    _write_agent_toml(
        tmp_path,
        '[features.TalonCoordinatorFeature.config]\nconfig_path = "/srv/k.toml"\n',
    )
    with caplog.at_level(logging.WARNING):
        declared = _agent(tmp_path)._declared_feature_config(
            "TalonCoordinatorFeature"
        )
    assert declared == {}
    assert "is never read" in caplog.text
    assert "[features.talon.config]" in caplog.text


def test_absent_file_and_absent_block_are_not_errors(tmp_path):
    agent = _agent(tmp_path)
    assert agent._declared_feature_config("TalonCoordinatorFeature") == {}
    _write_agent_toml(tmp_path, '[llm]\nallow_paid_fallback = false\n')
    assert agent._declared_feature_config("TalonCoordinatorFeature") == {}


def test_unknown_feature_class_resolves_to_no_package(tmp_path):
    _write_agent_toml(tmp_path, '[features.talon.config]\nconfig_path = "/x"\n')
    assert _agent(tmp_path)._declared_feature_config("NotAFeature") == {}


@pytest.mark.asyncio
async def test_declared_config_is_applied_to_the_feature(tmp_path):
    _write_agent_toml(
        tmp_path, '[features.talon.config]\nconfig_path = "/srv/k.toml"\n'
    )
    feature = TalonCoordinatorFeature()
    await _agent(tmp_path)._apply_host_feature_config(feature)
    assert feature.applied == {"config_path": "/srv/k.toml"}


@pytest.mark.asyncio
async def test_feature_without_a_schema_is_left_alone(tmp_path):
    _write_agent_toml(
        tmp_path, '[features.talon.config]\nconfig_path = "/srv/k.toml"\n'
    )

    feature = TalonCoordinatorFeature()
    feature.config_schema = None
    await _agent(tmp_path)._apply_host_feature_config(feature)
    assert feature.applied is None


@pytest.mark.asyncio
async def test_a_rejected_config_is_named_and_does_not_stop_boot(tmp_path, caplog):
    """A capability gap: boot degraded and say so, rather than refusing."""
    _write_agent_toml(
        tmp_path, '[features.talon.config]\nbogus_key = "nope"\n'
    )

    # The lookup keys on the class name, so a subclass would resolve to no
    # package and this would pass without ever reaching set_config.
    feature = TalonCoordinatorFeature()

    async def _reject(config):
        raise ValueError("Unknown Talon feature configuration key(s): bogus_key")

    feature.set_config = _reject
    with caplog.at_level(logging.ERROR):
        await _agent(tmp_path)._apply_host_feature_config(feature)

    assert "bogus_key" in caplog.text
    assert "unconfigured" in caplog.text
