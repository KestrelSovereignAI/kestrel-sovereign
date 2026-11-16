"""``kestrel update`` can replace the ``kestrel.exe`` that is running it (#2923).

Measured on windows-latest (Windows journey, run 43921983963): with the host
stopped and its port closed, ``kestrel update`` still failed with
``failed to remove file ...Scripts/kestrel.exe ... (os error 32)``. The lock
was the update's own launcher. These tests exercise the Windows path on any
OS. The Windows lock itself is proven by the journey harness, not here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kestrel_sovereign import cli, windows_launchers
from tests.unit.test_cli_update import _ns, stub_project_dir  # noqa: F401 (fixture)

NAMES = ("kestrel",)


@pytest.fixture
def scripts(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "venv" / "Scripts"
    directory.mkdir(parents=True)
    (directory / "kestrel.exe").write_bytes(b"old launcher")
    monkeypatch.setattr(windows_launchers, "_ENABLED", True)
    return directory


def test_the_running_launcher_is_renamed_aside(scripts):
    moved = windows_launchers.move_aside(scripts, NAMES)

    assert not (scripts / "kestrel.exe").exists()
    [launcher] = moved
    assert launcher.aside.read_bytes() == b"old launcher"
    assert launcher.aside.parent == scripts


def test_a_launcher_the_install_replaced_is_left_for_the_next_update(scripts):
    moved = windows_launchers.move_aside(scripts, NAMES)
    (scripts / "kestrel.exe").write_bytes(b"new launcher")

    assert windows_launchers.restore_missing(moved) == []
    assert (scripts / "kestrel.exe").read_bytes() == b"new launcher"
    assert moved[0].aside.exists()


def test_a_launcher_the_install_did_not_replace_is_put_back(scripts):
    moved = windows_launchers.move_aside(scripts, NAMES)

    assert windows_launchers.restore_missing(moved) == [scripts / "kestrel.exe"]
    assert (scripts / "kestrel.exe").read_bytes() == b"old launcher"


def test_the_next_update_removes_replaced_launchers_and_keeps_running_ones(scripts):
    stale = scripts / "kestrel.exe.replaced-1-1"
    running = scripts / "kestrel.exe.replaced-2-2"
    unrelated = scripts / "other.exe.replaced-3-3"
    for path in (stale, running, unrelated):
        path.write_bytes(b"x")
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self == running:
            raise PermissionError(32, "in use")
        return real_unlink(self, *args, **kwargs)

    with patch.object(Path, "unlink", unlink):
        removed, in_use = windows_launchers.remove_replaced(scripts, NAMES)

    assert removed == [stale] and in_use == [running]
    assert not stale.exists() and running.exists() and unrelated.exists()


def test_a_rename_that_fails_restores_what_moved_and_raises(scripts):
    (scripts / "second.exe").write_bytes(b"second")
    real_rename = Path.rename

    def rename(self, target):
        if self.name == "second.exe":
            raise PermissionError(5, "denied")
        return real_rename(self, target)

    with patch.object(Path, "rename", rename), pytest.raises(PermissionError):
        windows_launchers.move_aside(scripts, ("kestrel", "second"))

    assert (scripts / "kestrel.exe").read_bytes() == b"old launcher"
    assert not list(scripts.glob("*.replaced-*"))


def test_off_windows_nothing_moves(scripts, monkeypatch):
    monkeypatch.setattr(windows_launchers, "_ENABLED", False)
    assert windows_launchers.move_aside(scripts, NAMES) == []
    assert windows_launchers.remove_replaced(scripts, NAMES) == ([], [])
    assert (scripts / "kestrel.exe").exists()


def test_core_installs_the_kestrel_console_script():
    assert "kestrel" in windows_launchers.console_script_names()


# -- wiring: both install paths of `kestrel update` ------------------------------


def _update_with_install(stub_project_dir, scripts, monkeypatch, *, use_lock: bool, install_rc: int):
    if use_lock:
        (stub_project_dir / "uv.lock").touch()
    monkeypatch.setattr(windows_launchers, "scripts_dir", lambda prefix=None: scripts)
    monkeypatch.setattr(windows_launchers, "console_script_names", lambda distribution=None: NAMES)
    seen: list[bool] = []

    def install(*_args, **_kwargs):
        # What uv does once the running launcher is out of its way.
        seen.append((scripts / "kestrel.exe").exists())
        if install_rc == 0:
            (scripts / "kestrel.exe").write_bytes(b"new launcher")
        return install_rc, "" if install_rc == 0 else "install failed"

    with patch.object(cli, "_project_dir_is_git", lambda _: True), \
         patch.object(cli, "_git_working_tree_dirty", lambda _: (False, "")), \
         patch.object(cli, "_run_git_fast_forward", lambda *_: (0, "")), \
         patch.object(cli, "_run_uv_sync", install), \
         patch.object(cli, "_run_uv_pip_install_editable", install), \
         patch.object(cli, "cmd_feature_sync", lambda args: 0), \
         patch.object(cli, "cmd_restart", lambda args: 0):
        rc = cli.cmd_update(_ns())
    return rc, seen


@pytest.mark.parametrize("use_lock", [True, False], ids=["uv-sync", "uv-pip-install"])
def test_update_installs_with_the_running_launcher_out_of_the_way(
    stub_project_dir, scripts, monkeypatch, use_lock
):
    rc, seen = _update_with_install(stub_project_dir, scripts, monkeypatch,
                                    use_lock=use_lock, install_rc=0)

    assert rc == 0
    assert seen == [False]
    assert (scripts / "kestrel.exe").read_bytes() == b"new launcher"


@pytest.mark.parametrize("use_lock", [True, False], ids=["uv-sync", "uv-pip-install"])
def test_a_failed_install_leaves_a_working_kestrel_command(
    stub_project_dir, scripts, monkeypatch, use_lock
):
    rc, _seen = _update_with_install(stub_project_dir, scripts, monkeypatch,
                                     use_lock=use_lock, install_rc=1)

    assert rc == 1
    assert (scripts / "kestrel.exe").read_bytes() == b"old launcher"
