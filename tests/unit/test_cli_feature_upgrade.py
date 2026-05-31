"""Tests for `kestrel feature upgrade`.

The command discovers which packages back the loaded features from their live
entry-points (no curated requirements file), skips editable/dev installs (they
track a local checkout), and pip-upgrades the rest. These tests pin that
behaviour without touching the real network or pip.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from kestrel_sovereign import cli


def _dist(name, version, editable_path=None, entries=None):
    return {
        "dist": name,
        "version": version,
        "editable_path": editable_path,
        "entries": entries or [f"features:{name}"],
    }


@pytest.fixture
def fake_registry(monkeypatch):
    """Stub load_registry so name resolution + git fallback have data."""
    registry = {
        "github": types.SimpleNamespace(
            package="kestrel-feature-github",
            git="https://github.com/KestrelSovereignAI/kestrel-feature-github.git",
            features=["GitHubFeature"],
        ),
        "voice": types.SimpleNamespace(
            package="kestrel-feature-voice",
            git="https://github.com/KestrelSovereignAI/kestrel-feature-voice.git",
            features=["VoiceFeature"],
        ),
    }
    import kestrel_sovereign.feature_registry as fr

    monkeypatch.setattr(fr, "load_registry", lambda: registry)
    return registry


class _PipSpy:
    """Records pip invocations and returns a canned success result."""

    def __init__(self, installed_line=""):
        self.calls = []
        self.installed_line = installed_line

    def __call__(self, cmd, capture_output=True, text=True):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout=self.installed_line, stderr=""
        )


def test_parse_pip_installed_version_extracts_only_matching_package():
    out = "Successfully installed kestrel-feature-github-0.2.0 httpx-0.27.0"
    assert cli._parse_pip_installed_version(out, "kestrel-feature-github") == "0.2.0"
    # underscores/dashes normalise
    assert cli._parse_pip_installed_version(out, "kestrel_feature_github") == "0.2.0"
    assert cli._parse_pip_installed_version("Requirement already satisfied", "x") is None


def test_editable_install_path_reads_direct_url(monkeypatch):
    import kestrel_sovereign.cli as climod

    class _Dist:
        def __init__(self, payload):
            self._payload = payload

        def read_text(self, name):
            return self._payload

    import importlib.metadata as md

    editable = '{"url": "file:///Volumes/x/pkg", "dir_info": {"editable": true}}'
    monkeypatch.setattr(md, "distribution", lambda n: _Dist(editable))
    assert climod._editable_install_path("pkg") == "/Volumes/x/pkg"

    regular = '{"url": "https://pypi/...", "dir_info": {}}'
    monkeypatch.setattr(md, "distribution", lambda n: _Dist(regular))
    assert climod._editable_install_path("pkg") is None


def test_upgrade_dry_run_changes_nothing(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [
            _dist("kestrel-feature-github", "0.1.0"),
            _dist("kestrel-feature-voice", "0.1.0", editable_path="/src/voice"),
        ],
    )
    spy = _PipSpy()
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=True))

    assert rc == 0
    assert spy.calls == []  # dry-run never shells out
    out = capsys.readouterr().out
    assert "would upgrade" in out
    assert "skip (editable -> /src/voice)" in out


def test_upgrade_skips_editable_and_pip_upgrades_others(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [
            _dist("kestrel-feature-github", "0.1.0"),
            _dist("kestrel-feature-voice", "0.1.0", editable_path="/src/voice"),
        ],
    )
    spy = _PipSpy(installed_line="Successfully installed kestrel-feature-github-0.2.0")
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    # exactly one pip call — the non-editable package
    assert len(spy.calls) == 1
    assert spy.calls[0][-3:] == ["install", "--upgrade", "kestrel-feature-github"]
    out = capsys.readouterr().out
    assert "upgraded -> 0.2.0" in out
    assert "1 package(s) upgraded" in out
    assert "kestrel-feature-voice" in out and "skip (editable" in out


def test_upgrade_subset_by_name_only_targets_match(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [
            _dist("kestrel-feature-github", "0.1.0"),
            _dist("kestrel-feature-voice", "0.1.0"),
        ],
    )
    spy = _PipSpy(installed_line="Successfully installed kestrel-feature-github-0.2.0")
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(
        types.SimpleNamespace(names=["github"], dry_run=False)
    )

    assert rc == 0
    assert len(spy.calls) == 1
    assert spy.calls[0][-1] == "kestrel-feature-github"


def test_upgrade_unmatched_name_is_reported(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("kestrel-feature-github", "0.1.0")],
    )
    spy = _PipSpy()
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(
        types.SimpleNamespace(names=["does-not-exist"], dry_run=True)
    )

    assert rc == 1
    assert spy.calls == []
    assert "not an installed extension package" in capsys.readouterr().out


def test_upgrade_falls_back_to_git_on_pip_failure(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(
        cli,
        "_installed_extension_distributions",
        lambda: [_dist("kestrel-feature-github", "0.1.0")],
    )

    calls = []

    def flaky_run(cmd, capture_output=True, text=True):
        calls.append(cmd)
        # First (PyPI) attempt fails; git fallback succeeds.
        if "git+" in cmd[-1]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="Successfully installed kestrel-feature-github-0.2.0", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No matching distribution")

    monkeypatch.setattr(cli.subprocess, "run", flaky_run)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    assert len(calls) == 2
    assert calls[1][-1].startswith("git+https://github.com/")


def test_upgrade_no_installed_extensions(monkeypatch, fake_registry, capsys):
    monkeypatch.setattr(cli, "_installed_extension_distributions", lambda: [])
    spy = _PipSpy()
    monkeypatch.setattr(cli.subprocess, "run", spy)

    rc = cli.cmd_feature_upgrade(types.SimpleNamespace(names=[], dry_run=False))

    assert rc == 0
    assert spy.calls == []
    assert "No installed feature/provider packages" in capsys.readouterr().out
