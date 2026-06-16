"""Tests for the ``kestrel update`` reconcile step (#1788).

Covers the execution + wiring half (the planning half is in
``test_feature_reconcile.py``): a missing allowlisted feature gets installed, an
unresolvable class aborts with a clear error, ``--dry-run`` mutates nothing, and
editable vs PyPI update modes dispatch to ``git pull`` vs ``pip --upgrade``.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kestrel_sovereign import cli, cli_lifecycle
from kestrel_sovereign import feature_reconcile as fr
from kestrel_sovereign.feature_registry import FeaturePackageInfo
from kestrel_sovereign.multi_agent.config import LocalAgentConfig, MultiAgentConfig


def _registry():
    return {
        "voice": FeaturePackageInfo(
            name="voice", package="kestrel-feature-voice",
            git="https://example/voice.git",
            features=["VoiceFeature"], description="", core=False,
        ),
    }


def _ok(stdout="", stderr="", rc=0):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Patch the shared seams so reconcile runs against a fake host."""
    ma = MultiAgentConfig(agents={
        "emma": LocalAgentConfig(
            data_dir="agent_data/emma", port=8801, features=["VoiceFeature"],
        ),
    })
    monkeypatch.setattr(cli.MultiAgentConfig, "load", classmethod(lambda c, p, **k: ma))
    monkeypatch.setattr(
        "kestrel_sovereign.feature_registry.load_registry", lambda *a, **k: _registry(),
    )
    # No source map on disk by default.
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: tmp_path / "missing.toml")
    return tmp_path


# --------------------------------------------------------------------------
# _run_feature_reconcile
# --------------------------------------------------------------------------

def test_reconcile_installs_missing_allowlisted_feature(patched, capsys):
    """An allowlisted class missing from the venv is installed from its
    registry PyPI source."""
    install_calls = []

    def fake_install(pip_args):
        install_calls.append(pip_args)
        return _ok(stdout="Successfully installed kestrel-feature-voice-0.3.0")

    with patch("importlib.metadata.version", side_effect=__import__("importlib.metadata", fromlist=["PackageNotFoundError"]).PackageNotFoundError), \
         patch.object(cli, "_editable_install_path", lambda p: None), \
         patch.object(cli, "_extension_install_run", fake_install):
        rc = cli_lifecycle._run_feature_reconcile(
            patched, manifest_override=None, dry_run=False,
            allow_dirty=False, continue_on_error=False, prefer=None,
        )

    assert rc == 0
    assert install_calls == [["kestrel-feature-voice"]]
    out = capsys.readouterr().out
    assert "kestrel-feature-voice" in out


def test_reconcile_errors_on_unresolvable_class(monkeypatch, tmp_path, capsys):
    """A class no registry package provides is a hard error (no blind
    fallback) — reconcile returns non-zero and names the class."""
    ma = MultiAgentConfig(agents={
        "x": LocalAgentConfig(data_dir="d/x", port=8801, features=["GhostFeature"]),
    })
    monkeypatch.setattr(cli.MultiAgentConfig, "load", classmethod(lambda c, p, **k: ma))
    monkeypatch.setattr(
        "kestrel_sovereign.feature_registry.load_registry", lambda *a, **k: _registry(),
    )
    monkeypatch.setattr(cli, "_host_manifest_path", lambda ns: tmp_path / "missing.toml")

    rc = cli_lifecycle._run_feature_reconcile(
        tmp_path, manifest_override=None, dry_run=False,
        allow_dirty=False, continue_on_error=False, prefer=None,
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "GhostFeature" in err


def test_reconcile_dry_run_mutates_nothing(patched):
    """--dry-run previews the install without running pip/git."""
    pnf = __import__("importlib.metadata", fromlist=["PackageNotFoundError"]).PackageNotFoundError
    with patch("importlib.metadata.version", side_effect=pnf), \
         patch.object(cli, "_editable_install_path", lambda p: None), \
         patch.object(cli, "_extension_install_run") as install, \
         patch.object(cli_lifecycle, "_editable_git_pull") as pull:
        rc = cli_lifecycle._run_feature_reconcile(
            patched, manifest_override=None, dry_run=True,
            allow_dirty=False, continue_on_error=False, prefer=None,
        )
    assert rc == 0
    install.assert_not_called()
    pull.assert_not_called()


# --------------------------------------------------------------------------
# _execute_reconcile_action
# --------------------------------------------------------------------------

def test_execute_editable_update_runs_git_pull_only():
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="update", mode="editable",
        source="/co/voice", required_by=["VoiceFeature"],
    )
    with patch.object(cli_lifecycle, "_editable_git_pull", return_value=(0, "Updated.")) as pull, \
         patch.object(cli, "_extension_install_run") as install:
        ok, detail = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False)
    assert ok is True
    pull.assert_called_once()
    install.assert_not_called()  # present editable: pull is enough, no reinstall


def test_execute_editable_install_pulls_then_links():
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="install", mode="editable",
        source="/co/voice",
    )
    with patch.object(cli_lifecycle, "_editable_git_pull", return_value=(0, "")) as pull, \
         patch.object(cli, "_extension_install_run", return_value=_ok()) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False)
    assert ok is True
    pull.assert_called_once()
    install.assert_called_once()
    assert install.call_args[0][0][0] == "-e"


def test_execute_editable_pull_failure_reports_and_skips_link():
    """A non-fast-forward / dirty collision from the pull is reported and the
    pip link is NOT attempted."""
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="install", mode="editable",
        source="/co/voice",
    )
    with patch.object(cli_lifecycle, "_editable_git_pull", return_value=(2, "REFUSED — dirty")) as pull, \
         patch.object(cli, "_extension_install_run") as install:
        ok, detail = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False)
    assert ok is False
    assert "REFUSED" in detail
    install.assert_not_called()


def test_execute_pypi_update_runs_upgrade():
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="update", mode="pypi",
        source="kestrel-feature-voice>=0.3,<0.4",
    )
    with patch.object(cli, "_extension_install_run", return_value=_ok()) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(action, {}, allow_dirty=False)
    assert ok is True
    assert install.call_args[0][0] == ["--upgrade", "kestrel-feature-voice>=0.3,<0.4"]


def test_execute_pypi_falls_back_to_git_url():
    action = fr.ReconcileAction(
        package="kestrel-feature-voice", op="install", mode="pypi",
        source="kestrel-feature-voice",
    )
    results = [_ok(rc=1, stderr="not found"), _ok(rc=0)]
    with patch.object(cli, "_extension_install_run", side_effect=results) as install:
        ok, _ = cli_lifecycle._execute_reconcile_action(
            action, {"kestrel-feature-voice": "https://example/voice.git"},
            allow_dirty=False,
        )
    assert ok is True
    assert install.call_count == 2
    assert install.call_args[0][0] == ["git+https://example/voice.git"]
