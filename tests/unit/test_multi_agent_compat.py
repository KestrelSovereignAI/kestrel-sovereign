"""Tests for the Rookery -> MultiAgent backward-compat shims."""

from __future__ import annotations

import logging

import pytest

from kestrel_sovereign.multi_agent import compat


@pytest.fixture(autouse=True)
def _reset_latches():
    compat.reset_deprecation_latches()
    yield
    compat.reset_deprecation_latches()


# ---- env var compat --------------------------------------------------------


def test_get_config_env_value_prefers_new_name():
    env = {
        "KESTREL_MULTI_AGENT_CONFIG": "/new/path.toml",
        "KESTREL_ROOKERY_CONFIG": "/legacy/path.toml",
    }
    assert compat.get_config_env_value(env) == "/new/path.toml"


def test_get_config_env_value_falls_back_to_legacy(caplog):
    env = {"KESTREL_ROOKERY_CONFIG": "/legacy/path.toml"}
    with caplog.at_level(logging.WARNING):
        result = compat.get_config_env_value(env)
    assert result == "/legacy/path.toml"
    assert any("DEPRECATED" in r.message and "KESTREL_ROOKERY_CONFIG" in r.message
               for r in caplog.records)


def test_get_config_env_value_returns_none_when_neither_set():
    assert compat.get_config_env_value({}) is None


def test_legacy_env_var_warning_fires_only_once(caplog):
    env = {"KESTREL_ROOKERY_CONFIG": "/legacy.toml"}
    with caplog.at_level(logging.WARNING):
        compat.get_config_env_value(env)
        compat.get_config_env_value(env)
        compat.get_config_env_value(env)
    deprecation_warnings = [r for r in caplog.records
                            if "DEPRECATED" in r.message
                            and "KESTREL_ROOKERY_CONFIG" in r.message]
    assert len(deprecation_warnings) == 1


def test_is_config_env_var_set_detects_either():
    assert compat.is_config_env_var_set({"KESTREL_MULTI_AGENT_CONFIG": "x"})
    assert compat.is_config_env_var_set({"KESTREL_ROOKERY_CONFIG": "x"})
    assert not compat.is_config_env_var_set({})


def test_write_config_env_value_clears_legacy():
    env = {
        "KESTREL_MULTI_AGENT_CONFIG": "old-new",
        "KESTREL_ROOKERY_CONFIG": "old-legacy",
    }
    compat.write_config_env_value(env, "fresh")
    assert env["KESTREL_MULTI_AGENT_CONFIG"] == "fresh"
    assert "KESTREL_ROOKERY_CONFIG" not in env


# ---- config filename compat -----------------------------------------------


def test_find_existing_config_prefers_new(tmp_path):
    (tmp_path / "multi_agent.toml").write_text("# new")
    (tmp_path / "rookery.toml").write_text("# legacy")
    found = compat.find_existing_config(tmp_path)
    assert found == tmp_path / "multi_agent.toml"


def test_find_existing_config_falls_back_to_legacy(tmp_path, caplog):
    (tmp_path / "rookery.toml").write_text("# legacy")
    with caplog.at_level(logging.WARNING):
        found = compat.find_existing_config(tmp_path)
    assert found == tmp_path / "rookery.toml"
    assert any("DEPRECATED" in r.message and "rookery.toml" in r.message
               for r in caplog.records)


def test_find_existing_config_returns_none_when_neither_exists(tmp_path):
    assert compat.find_existing_config(tmp_path) is None


# ---- deployment_mode compat -----------------------------------------------


def test_normalize_deployment_mode_canonicalises_legacy(caplog):
    with caplog.at_level(logging.WARNING):
        result = compat.normalize_deployment_mode("rookery")
    assert result == "multi_agent"
    assert any("DEPRECATED" in r.message and "rookery" in r.message
               for r in caplog.records)


def test_normalize_deployment_mode_passes_through_modern_value():
    assert compat.normalize_deployment_mode("multi_agent") == "multi_agent"


def test_normalize_deployment_mode_passes_through_agent_default():
    assert compat.normalize_deployment_mode("agent") == "agent"


def test_normalize_deployment_mode_handles_none():
    assert compat.normalize_deployment_mode(None) == "agent"


# ---- profile alias compat (lives in deploy.core, but exercised here) ------


def test_legacy_profile_alias_resolves_with_warning(caplog):
    """`get_profile('rookery-dev')` should resolve to `multi-agent-dev`."""
    from kestrel_sovereign.features.deploy.core import (
        DeployManagerCore,
        DeploymentProfile,
    )

    mgr = DeployManagerCore.__new__(DeployManagerCore)
    mgr.profiles = {
        "multi-agent-dev": DeploymentProfile(
            provider="cloudrun",
            service_name="kestrel-multi-agent-dev",
            region="us-central1",
            deployment_mode="multi_agent",
        ),
    }

    with caplog.at_level(logging.WARNING):
        profile = mgr.get_profile("rookery-dev")
    assert profile.service_name == "kestrel-multi-agent-dev"
    assert any("DEPRECATED" in r.message and "rookery-dev" in r.message
               for r in caplog.records)


def test_unknown_profile_still_raises():
    from kestrel_sovereign.features.deploy.core import (
        DeployManagerCore,
        DeployManagerError,
    )

    mgr = DeployManagerCore.__new__(DeployManagerCore)
    mgr.profiles = {}

    with pytest.raises(DeployManagerError):
        mgr.get_profile("does-not-exist")


# ---- end-to-end: loader picks up legacy filename --------------------------


def test_multi_agent_config_load_finds_legacy_filename(tmp_path, caplog,
                                                       monkeypatch):
    """MultiAgentConfig.load() with no path should fall back to rookery.toml."""
    from kestrel_sovereign.multi_agent.config import MultiAgentConfig

    monkeypatch.chdir(tmp_path)
    (tmp_path / "rookery.toml").write_text(
        '[host]\nport = 9999\n[agents]\n', encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        config = MultiAgentConfig.load()

    assert config.host.port == 9999
    assert any("DEPRECATED" in r.message and "rookery.toml" in r.message
               for r in caplog.records)


def test_multi_agent_config_load_explicit_new_path_falls_back_to_legacy_sibling(
    tmp_path, caplog,
):
    """If a caller passes an explicit `multi_agent.toml` path that doesn't
    exist but a sibling `rookery.toml` does, load() must use the legacy
    file. Codex review (PR #998 round 1 of rookery rename) caught this:
    the CLI passes the new filename explicitly, so the previous compat
    branch only firing when path=None left existing installs broken.
    """
    from kestrel_sovereign.multi_agent.config import MultiAgentConfig

    (tmp_path / "rookery.toml").write_text(
        '[host]\nport = 7777\n[agents]\n', encoding="utf-8",
    )
    explicit_new_path = tmp_path / "multi_agent.toml"
    assert not explicit_new_path.exists()

    with caplog.at_level(logging.WARNING):
        config = MultiAgentConfig.load(
            explicit_new_path, auto_discover_fallback=False,
        )

    assert config.host.port == 7777
    assert any("DEPRECATED" in r.message and "rookery.toml" in r.message
               for r in caplog.records)


def test_resolve_multi_agent_path_falls_back_to_legacy_when_no_env_var(
    tmp_path, monkeypatch, caplog,
):
    """server.py:resolve_multi_agent_path returns the legacy path when
    no env var is set and only rookery.toml exists at cwd. Without this
    fallback, the lifespan startup gate (`multi_agent_path.exists()`)
    is false and the server boots in single-agent mode even though the
    operator has a multi-agent config — codex review caught this."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rookery.toml").write_text("[host]\nport = 5555\n[agents]\n")

    from server import resolve_multi_agent_path

    with caplog.at_level(logging.WARNING):
        result = resolve_multi_agent_path({})

    assert result.name == "rookery.toml"
    assert result.exists()
