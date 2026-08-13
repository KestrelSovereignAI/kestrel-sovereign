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
from kestrel_sovereign.feature_registry import PackageBoundary


def _info(
    name,
    package,
    features,
    boundary=PackageBoundary.FEATURE_PACKAGE,
):
    return types.SimpleNamespace(
        name=name,
        package=package,
        features=features,
        boundary=boundary,
        git="",
    )


@pytest.fixture
def fake_registry(monkeypatch):
    registry = {
        "github": _info("github", "kestrel-feature-github", ["GitHubFeature"]),
        "voice": _info("voice", "kestrel-feature-voice", ["VoiceFeature"]),
        # Provider implementations have their own explicit boundary and are
        # never treated as per-agent Feature lifecycle classes.
        "voice_openai": _info(
            "voice_openai", "kestrel-voice-openai",
            [],
            PackageBoundary.PROVIDER_PACKAGE,
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


def test_status_core_entry_is_source_policy_not_a_lifecycle_column(
    monkeypatch, fake_registry, tmp_path, capsys
):
    """A captured `kestrel-sovereign` entry declares where CORE comes from.

    It cannot be rendered as one per-agent feature column: dozens of bundled
    features share the `kestrel-sovereign` package, so the entry resolves to
    whichever bundled row the registry happens to list first (`identity`), and
    the agent's catalog collapses every bundled feature's status under that one
    package key, last-writer-wins. A column would therefore report an arbitrary
    bundled feature's state under core's name — here it would call Emma's
    ENABLED Identity feature merely "installed" and raise a phantom
    needs-restart hint (issue #2949).
    """
    from kestrel_sovereign.cli_features import CORE_DISTRIBUTION

    # Mirror the real registry: several bundled rows share the core package.
    fake_registry["identity"] = _info(
        "identity", CORE_DISTRIBUTION, ["IdentityFeature"], PackageBoundary.BUNDLED,
    )
    fake_registry["memory"] = _info(
        "memory", CORE_DISTRIBUTION, ["MemoryFeature"], PackageBoundary.BUNDLED,
    )
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        f'[[feature]]\nname="{CORE_DISTRIBUTION}"\neditable="/src/core"\n'
        '[[feature]]\nname="kestrel-feature-voice"\n'
    )
    _versions(monkeypatch, {CORE_DISTRIBUTION: "0.52.0", "kestrel-feature-voice": "0.2.1"})
    monkeypatch.setattr(cli, "_editable_install_path", lambda d: "/src/core")
    _fake_host(monkeypatch, tmp_path, agents=("Emma",))
    monkeypatch.setattr(
        cli, "_detect_running_agent_server", lambda *a: ("http://h/api/agents/Emma", "k"),
    )
    # Identity IS enabled on Emma; a later bundled row wins the package key.
    monkeypatch.setattr(
        cli, "_query_agent_feature_catalog",
        lambda b, k: {CORE_DISTRIBUTION: "installed", "kestrel-feature-voice": "enabled"},
    )

    rc = cli.cmd_feature_status(_args(manifest))
    out = capsys.readouterr().out

    assert rc == 0
    # Core's SOURCE is reported, at the level a source policy lives at.
    assert CORE_DISTRIBUTION in out
    assert "manifest satisfied" in out
    # ...but never as a lifecycle column, under core's name or identity's.
    per_agent = out.split("Loaded per agent")[1]
    assert "identity" not in per_agent
    assert CORE_DISTRIBUTION not in per_agent
    # The real feature column is unaffected, and no phantom restart hint fires.
    assert "✓ enabled" in per_agent
    assert "installed-but-not-loaded" not in out


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
