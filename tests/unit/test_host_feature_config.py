"""Host-side provisioning of feature configuration (#3008).

Extracting Talon broke it on every existing agent because the package
correctly requires the host to supply explicit paths and the host had no
declarative mechanism to supply them. These cover the mechanism that closes
that gap, including the spelling that silently does nothing.
"""

from __future__ import annotations

import logging

import pytest

from kestrel_sovereign.features.config_validation import FeatureConfigInvalid
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
    """Advertises the real class name so the registry resolves it to 'talon'."""

    name = "TalonCoordinatorFeature"


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
    assert declared is None
    assert "is never read" in caplog.text
    assert "[features.talon.config]" in caplog.text


def test_absent_file_and_absent_block_are_not_errors(tmp_path):
    agent = _agent(tmp_path)
    assert agent._declared_feature_config("TalonCoordinatorFeature") is None
    _write_agent_toml(tmp_path, '[llm]\nallow_paid_fallback = false\n')
    assert agent._declared_feature_config("TalonCoordinatorFeature") is None


def test_unknown_feature_class_resolves_to_no_package(tmp_path):
    _write_agent_toml(tmp_path, '[features.talon.config]\nconfig_path = "/x"\n')
    assert _agent(tmp_path)._declared_feature_config("NotAFeature") is None


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
async def test_a_rejected_config_propagates_rather_than_being_swallowed(tmp_path):
    """A rejected block does NOT leave the feature unconfigured.

    A feature that validates before replacing its active config — Talon does —
    keeps running its PREVIOUS configuration when a new block is refused.
    Swallowing the rejection would leave the agent running settings the
    operator did not declare and believes they replaced, which is the silent
    divergence this mechanism exists to end.
    """
    _write_agent_toml(
        tmp_path, '[features.talon.config]\nconfig_path = "/srv/k.toml"\n'
    )
    feature = TalonCoordinatorFeature()

    async def _reject(config):
        raise ValueError("Talon config_path must be a non-empty string")

    feature.set_config = _reject

    with pytest.raises(ValueError, match="config_path"):
        await _agent(tmp_path)._apply_host_feature_config(feature)


@pytest.mark.asyncio
async def test_declared_values_are_validated_against_the_feature_schema(tmp_path):
    """The rule the HTTP route applies, applied before anything persists."""
    _write_agent_toml(tmp_path, "[features.talon.config]\nconfig_path = 123\n")
    feature = TalonCoordinatorFeature()
    feature.config_schema = {
        "type": "object",
        "properties": {"config_path": {"type": "string"}},
    }

    with pytest.raises(FeatureConfigInvalid, match="must be string"):
        await _agent(tmp_path)._apply_host_feature_config(feature)

    assert feature.applied is None


@pytest.mark.asyncio
async def test_isolated_features_resolve_by_advertised_name(tmp_path):
    """An isolated feature is a ProxyFeature; only .name carries its identity.

    Keying on type(feature).__name__ finds no registry package for any isolated
    feature, so a valid block would be silently ignored — the same silence this
    mechanism closes.
    """
    _write_agent_toml(
        tmp_path, '[features.talon.config]\nconfig_path = "/srv/k.toml"\n'
    )

    class ProxyFeature(_ConfigurableFeature):
        name = "TalonCoordinatorFeature"

    feature = ProxyFeature()
    await _agent(tmp_path)._apply_host_feature_config(feature)

    assert feature.applied == {"config_path": "/srv/k.toml"}


# --- The call site, on a real agent -----------------------------------------


class _RegistrationTaskManager:
    def __init__(self) -> None:
        self.agents: dict = {}

    def register_agent(self, *, agent_card, handler, command_prefixes):
        self.agents[agent_card.name] = handler

    def unregister_agent(self, name):
        self.agents.pop(name, None)


@pytest.mark.asyncio
async def test_registration_fails_when_declared_config_is_rejected(tmp_path):
    """The feature must not end up loaded with configuration nobody declared.

    Exercised through the real ``_register_feature`` rather than the helper,
    because the decision that matters — leave it out of ``agent.features``
    rather than register it — lives at the call site.
    """
    from kestrel_sovereign.features.base import Feature
    from kestrel_sovereign.signals.registry import SourceRegistry
    from kestrel_sovereign.waits import WaitRegistry

    _write_agent_toml(
        tmp_path, '[features.talon.config]\nconfig_path = "/srv/k.toml"\n'
    )

    class TalonCoordinatorFeature(Feature):
        name = "TalonCoordinatorFeature"
        config_schema = {"type": "object"}

        @property
        def tool_description(self) -> str:
            return "test double"

        async def initialize(self) -> None:
            return None

        async def set_config(self, config):
            raise ValueError("Talon config_path must be a non-empty string")

    agent = KestrelAgent(
        did="did:test:hostconfig", storage_path=str(tmp_path / "kestrel_prime.db")
    )
    agent.task_manager = _RegistrationTaskManager()
    agent.signal_registry = SourceRegistry()
    agent.wait_registry = WaitRegistry()
    agent.features = {}

    feature = TalonCoordinatorFeature(agent)
    with pytest.raises(ValueError, match="config_path"):
        await agent._register_feature(feature)

    assert "TalonCoordinatorFeature" not in agent.features


@pytest.mark.asyncio
async def test_an_explicitly_empty_block_is_applied_not_ignored(tmp_path):
    """Emptying the table means "clear it", not "I said nothing"."""
    _write_agent_toml(tmp_path, "[features.talon.config]\n")
    feature = TalonCoordinatorFeature()

    await _agent(tmp_path)._apply_host_feature_config(feature)

    assert feature.applied == {}


def test_an_unreadable_agent_toml_fails_closed(tmp_path):
    """A malformed file is not an absent one.

    Degrading to "no declaration" would let an initialized feature keep the
    configuration the operator believes they replaced.
    """
    from kestrel_sovereign.kestrel_agent import HostFeatureConfigError

    _write_agent_toml(tmp_path, "[features.talon.config\nconfig_path = ")

    with pytest.raises(HostFeatureConfigError, match="could not be read"):
        _agent(tmp_path)._declared_feature_config("TalonCoordinatorFeature")


@pytest.mark.asyncio
async def test_cancellation_during_config_application_tears_the_feature_down(
    tmp_path,
):
    """CancelledError is a BaseException and bypasses ``except Exception``.

    initialize() has already run and the feature is not yet in agent.features,
    so boot rollback cannot find it — its tasks would outlive the failed boot.
    """
    import asyncio

    from kestrel_sovereign.features.base import Feature
    from kestrel_sovereign.signals.registry import SourceRegistry
    from kestrel_sovereign.waits import WaitRegistry

    _write_agent_toml(
        tmp_path, '[features.talon.config]\nconfig_path = "/srv/k.toml"\n'
    )

    class TalonCoordinatorFeature(Feature):
        name = "TalonCoordinatorFeature"
        config_schema = {"type": "object"}

        @property
        def tool_description(self) -> str:
            return "test double"

        async def initialize(self) -> None:
            return None

        async def set_config(self, config):
            raise asyncio.CancelledError()

    agent = KestrelAgent(
        did="did:test:cancel", storage_path=str(tmp_path / "kestrel_prime.db")
    )
    agent.task_manager = _RegistrationTaskManager()
    agent.signal_registry = SourceRegistry()
    agent.wait_registry = WaitRegistry()
    agent.features = {}

    torn_down = []
    original = agent._shutdown_failed_feature

    async def _record(f):
        torn_down.append(f)
        await original(f)

    agent._shutdown_failed_feature = _record

    feature = TalonCoordinatorFeature(agent)
    with pytest.raises(asyncio.CancelledError):
        await agent._register_feature(feature)

    assert torn_down == [feature]
    assert "TalonCoordinatorFeature" not in agent.features
