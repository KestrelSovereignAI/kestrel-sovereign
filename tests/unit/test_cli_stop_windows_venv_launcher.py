"""``kestrel stop`` attributes a Windows host's listener through its venv launcher.

A Windows venv's ``Scripts\\python.exe`` is a launcher. It starts the base
interpreter that ``pyvenv.cfg`` ``home`` names as a child and waits on it, so
the host's PID record names the launcher while the listening socket belongs to
the child. Measured on windows-latest (Windows journey, run 43921270420): PID
record 1520, port 8888 owned by 3684, a base ``python.exe`` whose parent is
1520. The ownership check therefore refused every Windows host, and
``kestrel stop`` printed "local Stop authority could not be proven".
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from kestrel_sovereign import cli_stop

PORT = 8888


def _listener(port: int = PORT) -> SimpleNamespace:
    return SimpleNamespace(status=psutil.CONN_LISTEN, laddr=("0.0.0.0", port), raddr=())


class _FakeProcess:
    def __init__(self, exe: Path, *, connections=(), children=(), exe_error=None):
        self._exe = exe
        self._connections = list(connections)
        self._children = list(children)
        self._exe_error = exe_error

    def exe(self) -> str:
        if self._exe_error is not None:
            raise self._exe_error
        return str(self._exe)

    def children(self):
        return list(self._children)

    def net_connections(self, kind: str = "tcp"):
        return list(self._connections)


@pytest.fixture
def venv(tmp_path: Path) -> SimpleNamespace:
    """A Windows-shaped venv whose pyvenv.cfg points at a base install."""
    base = tmp_path / "uv-python" / "cpython-3.13.18-windows-x86_64-none"
    base.mkdir(parents=True)
    root = tmp_path / "checkout" / ".venv"
    (root / "Scripts").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text(
        f"home = {base}\nimplementation = CPython\nversion_info = 3.13.18\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        launcher=root / "Scripts" / "python.exe",
        interpreter=base / "python.exe",
        base=base,
    )


def _patch_processes(monkeypatch, processes: dict[int, _FakeProcess], *, windows=True):
    monkeypatch.setattr(cli_stop, "_IS_WINDOWS", windows)
    monkeypatch.setattr(psutil, "Process", lambda pid: processes[pid])


def test_a_listener_owned_by_the_launchers_interpreter_belongs_to_the_host(venv, monkeypatch):
    interpreter = _FakeProcess(venv.interpreter, connections=[_listener()])
    launcher = _FakeProcess(venv.launcher, children=[interpreter])
    _patch_processes(monkeypatch, {1520: launcher})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is True


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privilege on Windows")
def test_a_home_reached_through_a_link_still_matches_the_interpreter(tmp_path, monkeypatch):
    """uv's pyvenv.cfg names a minor-version junction; the image resolves past it.

    Run 43922732579 on windows-latest: home ``cpython-3.13-windows-x86_64-none``,
    interpreter image ``cpython-3.13.18-windows-x86_64-none\\python.exe``. A
    symlink stands in for the junction here.
    """
    real = tmp_path / "uv-python" / "cpython-3.13.18-windows-x86_64-none"
    real.mkdir(parents=True)
    (real / "python.exe").write_bytes(b"")
    link = tmp_path / "uv-python" / "cpython-3.13-windows-x86_64-none"
    link.symlink_to(real, target_is_directory=True)
    root = tmp_path / "checkout" / ".venv"
    (root / "Scripts").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text(f"home = {link}\n", encoding="utf-8")

    interpreter = _FakeProcess(real / "python.exe", connections=[_listener()])
    launcher = _FakeProcess(root / "Scripts" / "python.exe", children=[interpreter])
    _patch_processes(monkeypatch, {1520: launcher})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is True


def test_a_home_that_does_not_exist_matches_nothing(venv, monkeypatch):
    (venv.launcher.parent.parent / "pyvenv.cfg").write_text(
        f"home = {venv.base / 'gone'}\n", encoding="utf-8"
    )
    interpreter = _FakeProcess(venv.interpreter, connections=[_listener()])
    launcher = _FakeProcess(venv.launcher, children=[interpreter])
    _patch_processes(monkeypatch, {1520: launcher})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is False


def test_a_child_running_anything_but_the_base_interpreter_is_not_the_host(venv, monkeypatch):
    stranger = _FakeProcess(venv.base / "evil.exe", connections=[_listener()])
    other_python = _FakeProcess(Path("C:/Other/Python313/python.exe"), connections=[_listener()])
    launcher = _FakeProcess(venv.launcher, children=[stranger, other_python])
    _patch_processes(monkeypatch, {1520: launcher})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is False


def test_children_of_a_process_that_is_not_a_venv_launcher_are_ignored(venv, monkeypatch):
    interpreter = _FakeProcess(venv.interpreter, connections=[_listener()])
    not_a_launcher = _FakeProcess(venv.base / "python.exe", children=[interpreter])
    _patch_processes(monkeypatch, {1520: not_a_launcher})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is False


def test_a_python_beside_a_pyvenv_cfg_but_outside_scripts_is_not_a_launcher(venv, monkeypatch):
    elsewhere = venv.launcher.parent.parent / "Tools" / "python.exe"
    interpreter = _FakeProcess(venv.interpreter, connections=[_listener()])
    not_a_launcher = _FakeProcess(elsewhere, children=[interpreter])
    _patch_processes(monkeypatch, {1520: not_a_launcher})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is False


def test_a_launcher_without_pyvenv_cfg_attributes_nothing(venv, monkeypatch):
    (venv.launcher.parent.parent / "pyvenv.cfg").unlink()
    interpreter = _FakeProcess(venv.interpreter, connections=[_listener()])
    launcher = _FakeProcess(venv.launcher, children=[interpreter])
    _patch_processes(monkeypatch, {1520: launcher})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is False


def test_an_unreadable_child_is_skipped_not_trusted(venv, monkeypatch):
    unreadable = _FakeProcess(venv.interpreter, connections=[_listener()],
                              exe_error=psutil.AccessDenied(3684))
    launcher = _FakeProcess(venv.launcher, children=[unreadable])
    _patch_processes(monkeypatch, {1520: launcher})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is False


def test_off_windows_the_recorded_process_alone_is_consulted(venv, monkeypatch):
    interpreter = _FakeProcess(venv.interpreter, connections=[_listener()])
    launcher = _FakeProcess(venv.launcher, children=[interpreter])
    _patch_processes(monkeypatch, {1520: launcher}, windows=False)

    assert cli_stop._listener_is_owned_by(PORT, 1520) is False


def test_the_recorded_process_owning_the_listener_still_works(venv, monkeypatch):
    direct = _FakeProcess(venv.interpreter, connections=[_listener()])
    _patch_processes(monkeypatch, {1520: direct})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is True


def test_an_established_flow_on_the_launchers_interpreter_is_attributed(venv, monkeypatch):
    """The per-request socket check goes through the same seam."""
    flow = SimpleNamespace(status=psutil.CONN_ESTABLISHED, laddr=("127.0.0.1", PORT),
                           raddr=("127.0.0.1", 50123))
    interpreter = _FakeProcess(venv.interpreter, connections=[flow])
    launcher = _FakeProcess(venv.launcher, children=[interpreter])
    _patch_processes(monkeypatch, {1520: launcher})

    assert flow in cli_stop._process_connections(1520)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("home = C:\\Py\\base\nversion = 3.13\n", Path("C:\\Py\\base")),
        ("HOME=C:\\Py\\base\n", Path("C:\\Py\\base")),
        ("version = 3.13\n", None),
        ("home =   \n", None),
    ],
)
def test_pyvenv_home(tmp_path, text, expected):
    cfg = tmp_path / "pyvenv.cfg"
    cfg.write_text(text, encoding="utf-8")
    assert cli_stop._pyvenv_home(cfg) == expected


def test_a_recorded_process_whose_image_is_unreadable_keeps_its_own_sockets(monkeypatch):
    """psutil.AccessDenied on exe() must not refuse a host that owns its socket."""
    direct = _FakeProcess(Path("C:/py/python.exe"), connections=[_listener()],
                          exe_error=psutil.AccessDenied(1520))
    _patch_processes(monkeypatch, {1520: direct})

    assert cli_stop._listener_is_owned_by(PORT, 1520) is True


def test_an_undecodable_pyvenv_cfg_has_no_home(tmp_path):
    cfg = tmp_path / "pyvenv.cfg"
    cfg.write_bytes(b"home = C:\\\xff\xfe\n")
    assert cli_stop._pyvenv_home(cfg) is None


def test_a_bom_before_home_is_read(tmp_path):
    cfg = tmp_path / "pyvenv.cfg"
    cfg.write_bytes(b"\xef\xbb\xbfhome = C:\\Py\n")
    assert cli_stop._pyvenv_home(cfg) == Path("C:\\Py")
