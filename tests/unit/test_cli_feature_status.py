"""Tests for `kestrel feature status` — host install + per-agent loaded view.

`status` answers "is my environment set up right, per agent?": it shows whether
the manifest's packages are installed in the venv, and — for each running
agent — whether each feature is actually loaded (vs installed-but-not-loaded,
which signals a needed restart or an allowlist exclusion). These tests pin the
behaviour without a real venv, network, or running host.
"""

from __future__ import annotations

import subprocess  # noqa: F401  (imported for parity with sibling tests)
import types

import pytest

from kestrel_sovereign import cli


def _info(name, package, features):
    return types.SimpleNamespace(name=name, package=package, features=features, git="")


@pytest.fixture
def fake_registry(monkeypatch):
    registry = {
        "github": _info("github", "kestrel-feature-github", ["GitHubFeature"]),
        "voice": _info("voice", "kestrel-feature-voice", ["VoiceFeature"]),
        # Provider: registry lists provider classes under `features`, none of
        # which end in "Feature" — so it must NOT be treated as a per-agent feature.
        "voice_openai": _info(
            "voice_openai", "kestrel-voice-openai",
            ["OpenAITTSProvider", "OpenAISTTProvider"],
        ),
    }
    import kestrel_sovereign.feature_registry as fr

    monkeypatch.setattr(fr, "load_registry", lambda: registry)
    return registry


def _versions(monkeypatch, mapping):
    import importlib.metadata as md

    def fake_version(name):
        if name in mapping:
            return mapping[name]
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", fake_version)


def _fake_host(monkeypatch, tmp_path, agents=("Emma", "Claw"), port=8888):
    """Stub MultiAgentConfig + project dir so per-agent enumeration works."""
    local = {a: types.SimpleNamespace(port=8801) for a in agents}
    cfg = types.SimpleNamespace(
        host=types.SimpleNamespace(port=port),
        get_local_agents=lambda: local,
    )
    fake_cls = types.SimpleNamespace(load=lambda _p: cfg)
    monkeypatch.setattr(cli, "MultiAgentConfig", fake_cls)
    monkeypatch.setattr(cli, "_get_project_dir", lambda: tmp_path)
    return local


def _args(manifest, agent=None):
    return types.SimpleNamespace(manifest=str(manifest), agent=agent)


# --- helpers ---------------------------------------------------------------


def test_registry_info_for_resolves_dist_name_and_short_name(fake_registry):
    # short name
    assert cli._registry_info_for("voice", fake_registry).package == "kestrel-feature-voice"
    # raw dist name (what manifests commonly use)
    assert cli._registry_info_for("kestrel-feature-voice", fake_registry).name == "voice"
    # unknown
    assert cli._registry_info_for("nope", fake_registry) is None


def test_query_agent_feature_catalog_parses_status(monkeypatch):
    import httpx

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"features": [
                {"package": "kestrel-feature-voice", "status": "enabled"},
                {"package": "kestrel-feature-github", "status": "installed"},
                {"package": "", "status": "enabled"},  # skipped (no package)
            ]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    out = cli._query_agent_feature_catalog("http://x", "key")
    assert out == {
        "kestrel-feature-voice": "enabled",
        "kestrel-feature-github": "installed",
    }


def test_query_agent_feature_catalog_network_error_returns_none(monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.RequestError("down")

    monkeypatch.setattr(httpx, "get", boom)
    assert cli._query_agent_feature_catalog("http://x", "") is None


@pytest.mark.parametrize(
    "payload",
    [[], "nope", {"features": None}, {"features": ["x", 1]}, {"features": [{"package": []}]}],
)
def test_query_agent_feature_catalog_malformed_json_never_crashes(monkeypatch, payload):
    import httpx

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return payload

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    # Non-dict / null-features → None; a list with junk entries → {} (no crash).
    result = cli._query_agent_feature_catalog("http://x", "")
    assert result is None or result == {}


# --- install section -------------------------------------------------------


def test_status_install_satisfied_and_agents_offline(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname="kestrel-feature-github"\n'
        '[[feature]]\nname="kestrel-feature-voice"\n'
    )
    _versions(monkeypatch, {"kestrel-feature-github": "0.2.0", "kestrel-feature-voice": "0.2.1"})
    _fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_detect_running_agent_server", lambda *a: None)  # nothing running

    rc = cli.cmd_feature_status(_args(manifest))
    out = capsys.readouterr().out

    assert rc == 0
    assert "manifest satisfied" in out
    assert "(offline)" in out
    assert "No running host found" in out


def test_status_install_missing_prompts_sync(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname="kestrel-feature-voice"\n')
    _versions(monkeypatch, {})  # not installed
    _fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_detect_running_agent_server", lambda *a: None)

    rc = cli.cmd_feature_status(_args(manifest))
    out = capsys.readouterr().out

    assert rc == 0
    assert "not installed" in out
    assert "kestrel feature sync" in out


# --- per-agent section -----------------------------------------------------


def test_status_per_agent_enabled_vs_installed_drift(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        '[[feature]]\nname="kestrel-feature-github"\n'
        '[[feature]]\nname="kestrel-feature-voice"\n'
    )
    _versions(monkeypatch, {"kestrel-feature-github": "0.2.0", "kestrel-feature-voice": "0.2.1"})
    _fake_host(monkeypatch, tmp_path, agents=("Emma",))
    monkeypatch.setattr(cli, "_detect_running_agent_server", lambda *a: ("http://h/api/agents/Emma", "k"))
    # Emma has github loaded but voice merely installed (needs restart)
    monkeypatch.setattr(
        cli, "_query_agent_feature_catalog",
        lambda base, key: {"kestrel-feature-github": "enabled", "kestrel-feature-voice": "installed"},
    )

    rc = cli.cmd_feature_status(_args(manifest))
    out = capsys.readouterr().out

    assert rc == 0
    assert "✓ enabled" in out
    assert "⚠ installed" in out
    assert "installed-but-not-loaded" in out  # drift hint fired


def test_status_provider_has_no_per_agent_column(monkeypatch, fake_registry, tmp_path, capsys):
    # voice_openai is a provider (no Feature class) -> install section only.
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname="kestrel-voice-openai"\n')
    _versions(monkeypatch, {"kestrel-voice-openai": "0.1.0"})
    _fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_detect_running_agent_server", lambda *a: None)

    rc = cli.cmd_feature_status(_args(manifest))
    out = capsys.readouterr().out

    assert rc == 0
    assert "kestrel-voice-openai" in out  # shown in install section
    assert "no loadable feature packages in the manifest" in out  # no per-agent columns


def test_status_install_editable_mismatch_flags_drift(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname="kestrel-feature-voice"\neditable="/want/voice"\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.2.1"})  # installed...
    monkeypatch.setattr(cli, "_editable_install_path", lambda d: "/other/voice")  # ...wrong checkout
    _fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_detect_running_agent_server", lambda *a: None)

    rc = cli.cmd_feature_status(_args(manifest))
    out = capsys.readouterr().out

    assert rc == 0
    assert "needs reinstall" in out
    assert "drift from the manifest" in out


def test_status_install_extras_flags_ensure(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname="kestrel-feature-voice"\nextras=["local"]\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.2.1"})  # base present, extras unprobed
    _fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_detect_running_agent_server", lambda *a: None)

    rc = cli.cmd_feature_status(_args(manifest))
    out = capsys.readouterr().out

    assert rc == 0
    assert "needs ensure" in out


def test_status_uninstalled_feature_still_gets_agent_row(monkeypatch, fake_registry, tmp_path, capsys):
    # Feature declared + registry-known but NOT installed: must still appear as a
    # per-agent column (classification is registry-based, not install-based).
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname="kestrel-feature-voice"\n')
    _versions(monkeypatch, {})  # not installed
    _fake_host(monkeypatch, tmp_path, agents=("Emma",))
    monkeypatch.setattr(cli, "_detect_running_agent_server", lambda *a: ("http://h/api/agents/Emma", "k"))
    monkeypatch.setattr(cli, "_query_agent_feature_catalog", lambda b, k: {})  # agent has none

    rc = cli.cmd_feature_status(_args(manifest))
    out = capsys.readouterr().out

    assert rc == 0
    assert "voice" in out          # column present despite not installed
    assert "✗ —" in out            # Emma doesn't have it loaded


def test_status_specific_agent_not_found(monkeypatch, fake_registry, tmp_path, capsys):
    manifest = tmp_path / "m.toml"
    manifest.write_text('[[feature]]\nname="kestrel-feature-voice"\n')
    _versions(monkeypatch, {"kestrel-feature-voice": "0.2.1"})
    _fake_host(monkeypatch, tmp_path, agents=("Emma", "Claw"))

    rc = cli.cmd_feature_status(_args(manifest, agent="Bogus"))
    out = capsys.readouterr().out

    assert rc == 1
    assert "not found" in out


def test_status_no_manifest_still_lists_agents(monkeypatch, fake_registry, tmp_path, capsys):
    _fake_host(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_detect_running_agent_server", lambda *a: None)

    rc = cli.cmd_feature_status(_args(tmp_path / "missing.toml"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "Run `kestrel feature sync --capture`" in out
