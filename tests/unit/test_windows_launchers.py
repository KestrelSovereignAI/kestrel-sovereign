"""``kestrel update`` refuses, before changing anything, an install it cannot finish (#2923).

Measured on windows-latest (Windows journey): with the host stopped and its
port closed, ``kestrel update`` failed with ``failed to remove file
...Scripts/kestrel.exe ... (os error 32)``, after the pull. While any
``kestrel.exe`` runs, Windows refuses to replace, delete or rename it, and the
update runs inside one. These tests exercise the Windows decision on any OS;
the Windows behaviour itself is proven by the journey harness.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign import cli, cli_lifecycle, windows_launchers
from tests.unit import test_cli_update
from tests.unit.test_cli_update import _ns

# The update tests' fixture, rebound here so pytest collects it for this module.
stub_project_dir = test_cli_update.stub_project_dir


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


def test_the_printed_command_is_the_fresh_interpreter_command_with_P():
    """-P keeps the working directory off sys.path, so pasting it in another
    checkout still updates the checkout this venv runs (review round 2)."""
    command = cli_lifecycle._python_module_command(["update", "--no-restart"])
    assert command == [sys.executable, "-P", "-m", "kestrel_sovereign.cli", "update", "--no-restart"]


def test_cmd_line_quotes_the_program_so_spaces_and_ampersands_are_literal():
    line = windows_launchers.cmd_line(["C:/R&D/Ann Lee/python.exe", "-P", "-m", "x", "update"])
    assert line == '"C:/R&D/Ann Lee/python.exe" -P -m x update'


def test_powershell_line_uses_the_call_operator_and_single_quotes():
    line = windows_launchers.powershell_line(
        ["C:/Users/Ann Lee/v/python.exe", "-P", "-m", "kestrel_sovereign.cli", "update", "Emma's agent"])
    assert line == "& 'C:/Users/Ann Lee/v/python.exe' -P -m kestrel_sovereign.cli update 'Emma''s agent'"


def test_run_by_the_launcher_the_refusal_names_both_shell_forms():
    message = windows_launchers.refusal([Path("C:/v/Scripts/kestrel.exe")],
                                        ["C:/v/Scripts/python.exe", "-P", "-m", "kestrel_sovereign.cli", "update"],
                                        run_by_launcher=True)
    assert "cmd:" in message and "PowerShell:" in message and "Nothing has been changed" in message


def test_held_by_another_command_the_refusal_does_not_loop_back_to_python():
    message = windows_launchers.refusal([Path("C:/v/Scripts/kestrel.exe")],
                                        ["C:/v/Scripts/python.exe", "-P", "-m", "kestrel_sovereign.cli", "update"],
                                        run_by_launcher=False)
    assert "another `kestrel` command" in message
    assert "Run the update through Python instead" not in message


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing")
def test_the_probe_sees_a_handle_that_denies_delete_sharing(tmp_path):
    """A plain open() on Windows shares read/write but not delete, as current uv launchers' handles do."""
    target = tmp_path / "kestrel.exe"
    target.write_bytes(b"x")
    assert windows_launchers._held_open_without_delete_sharing(target) is False
    with open(target, "rb"):
        assert windows_launchers._held_open_without_delete_sharing(target) is True


# -- wiring: `kestrel update` refuses before the pull ------------------------------


def _update(stub_project_dir, monkeypatch, *, held: bool, probe_error=None, **ns):
    monkeypatch.setattr(windows_launchers, "_ENABLED", True)

    def launchers_in_use(*a, **k):
        if probe_error is not None:
            raise probe_error
        return [Path("C:/v/Scripts/kestrel.exe")] if held else []

    monkeypatch.setattr(windows_launchers, "launchers_in_use", launchers_in_use)
    monkeypatch.setattr(windows_launchers, "run_by_launcher", lambda launchers: True)
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


@pytest.mark.parametrize("ns", [{"install": False, "features": False}, {"dry_run": True}],
                         ids=["no-install-no-features", "dry-run"])
def test_an_update_that_changes_no_packages_is_not_refused(stub_project_dir, monkeypatch, ns):
    rc, steps = _update(stub_project_dir, monkeypatch, held=True, **ns)

    assert rc == 0
    assert "install" not in steps


def test_feature_sync_alone_is_refused_because_it_can_reinstall_core(stub_project_dir, monkeypatch):
    rc, steps = _update(stub_project_dir, monkeypatch, held=True, install=False)

    assert rc == 2 and steps == []


def test_a_pypi_core_without_feature_sync_is_not_refused(stub_project_dir, monkeypatch):
    """No source checkout: the install step is skipped, so it needs no launcher."""
    monkeypatch.setattr(cli, "_resolve_source_checkout", lambda: None)
    rc, _steps = _update(stub_project_dir, monkeypatch, held=True, features=False)

    assert rc == 0


def test_a_probe_that_errors_refuses_clearly(stub_project_dir, monkeypatch, capsys):
    rc, steps = _update(stub_project_dir, monkeypatch, held=True,
                        probe_error=PermissionError(5, "Access is denied"))

    assert rc == 2 and steps == []
    assert "could not check whether" in capsys.readouterr().err
