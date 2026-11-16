"""``kestrel update`` refuses, before changing anything, an install it cannot finish (#2923).

Measured on windows-latest (Windows journey): with the host stopped and its
port closed, ``kestrel update`` failed with ``failed to remove file
...Scripts/kestrel.exe ... (os error 32)``, after the pull. While any
``kestrel.exe`` runs, Windows refuses to replace, delete or rename it, and the
update runs inside one. These tests exercise the Windows decision on any OS;
the Windows behaviour itself is proven by the journey harness.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign import cli, windows_launchers
from tests.unit.test_cli_update import _ns, stub_project_dir  # noqa: F401 (fixture)


@pytest.fixture
def scripts(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "venv" / "Scripts"
    directory.mkdir(parents=True)
    (directory / "kestrel.exe").write_bytes(b"launcher")
    monkeypatch.setattr(windows_launchers, "_ENABLED", True)
    return directory


def test_a_launcher_held_open_is_reported(scripts):
    assert windows_launchers.launchers_in_use(scripts, ("kestrel",), held=lambda p: True) == [
        scripts / "kestrel.exe"
    ]


def test_a_launcher_nobody_holds_is_not_reported(scripts):
    assert windows_launchers.launchers_in_use(scripts, ("kestrel",), held=lambda p: False) == []


def test_a_missing_launcher_is_not_probed(scripts):
    probed = []
    windows_launchers.launchers_in_use(scripts, ("absent",), held=probed.append)
    assert probed == []


def test_off_windows_nothing_is_looked_up(monkeypatch):
    """No metadata lookup, no probe: an update off Windows is unchanged."""
    monkeypatch.setattr(windows_launchers, "_ENABLED", False)
    monkeypatch.setattr(windows_launchers, "console_script_names", pytest.fail)
    assert windows_launchers.launchers_in_use(held=pytest.fail) == []


def test_core_installs_the_kestrel_console_script():
    assert "kestrel" in windows_launchers.console_script_names()


def test_the_refusal_names_a_command_that_runs_the_same_update():
    message = windows_launchers.refusal([Path("C:/v/Scripts/kestrel.exe")],
                                        ["update", "--no-restart", "Emma"])
    command = message.splitlines()[2].strip()
    assert command == subprocess.list2cmdline(
        [sys.executable, "-m", "kestrel_sovereign.cli", "update", "--no-restart", "Emma"])
    assert "kestrel.exe" in message and "Nothing has been changed" in message


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing")
def test_the_probe_sees_a_handle_that_denies_delete_sharing(tmp_path):
    """A plain open() on Windows shares read/write but not delete, like a running image."""
    target = tmp_path / "kestrel.exe"
    target.write_bytes(b"x")
    assert windows_launchers._held_open_without_delete_sharing(target) is False
    with open(target, "rb"):
        assert windows_launchers._held_open_without_delete_sharing(target) is True


# -- wiring: `kestrel update` refuses before the pull ------------------------------


def _update(stub_project_dir, monkeypatch, *, held: bool, **ns):
    monkeypatch.setattr(windows_launchers, "_ENABLED", True)
    monkeypatch.setattr(
        windows_launchers, "launchers_in_use",
        lambda *a, **k: [Path("C:/v/Scripts/kestrel.exe")] if held else [],
    )
    steps: list[str] = []
    with patch.object(cli, "_project_dir_is_git", lambda _: True), \
         patch.object(cli, "_git_working_tree_dirty", lambda _: (False, "")), \
         patch.object(cli, "_run_git_fast_forward", lambda *_: steps.append("pull") or (0, "")), \
         patch.object(cli, "_run_uv_sync", lambda _: steps.append("install") or (0, "")), \
         patch.object(cli, "_run_uv_pip_install_editable",
                      lambda *a, **k: steps.append("install") or (0, "")), \
         patch.object(cli, "cmd_feature_sync", lambda args: 0), \
         patch.object(cli, "cmd_restart", lambda args: 0):
        rc = cli.cmd_update(_ns(**ns))
    return rc, steps


def test_an_update_whose_launcher_is_in_use_refuses_before_the_pull(
    stub_project_dir, monkeypatch, capsys
):
    rc, steps = _update(stub_project_dir, monkeypatch, held=True)

    assert rc == 2
    assert steps == []
    assert "kestrel_sovereign.cli update" in capsys.readouterr().err


def test_an_update_run_through_python_proceeds(stub_project_dir, monkeypatch):
    rc, steps = _update(stub_project_dir, monkeypatch, held=False)

    assert rc == 0
    assert "install" in steps


@pytest.mark.parametrize("ns", [{"install": False}, {"dry_run": True}],
                         ids=["no-install", "dry-run"])
def test_an_update_that_installs_nothing_is_not_refused(stub_project_dir, monkeypatch, ns):
    rc, steps = _update(stub_project_dir, monkeypatch, held=True, **ns)

    assert rc == 0
    assert "install" not in steps
