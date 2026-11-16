"""Refuse an update that cannot replace the ``kestrel.exe`` running it (#2923).

On Windows a running ``kestrel.exe`` (a uv console-script launcher) can be
neither replaced, deleted nor renamed. Measured on windows-latest (Windows
journey, run 43925328757): with no ``kestrel.exe`` running, the launcher
renamed fine; while any ``kestrel`` command ran, the rename failed with
``WinError 32``. ``kestrel update`` itself runs inside ``kestrel.exe``, and every
install that reinstalls ``kestrel-sovereign`` (``uv sync`` after a version bump,
``uv pip install -e .``) must replace that file, so it failed with ``os error
32`` whether or not an agent was running. #2923 blamed the agent; it was the
update's own launcher.

The launcher cannot step out of the way of its own process, so this does what
pip does for ``pip install --upgrade pip`` on Windows: an update that would have
to replace a launcher in use refuses before it changes anything, and names the
command that can do it (``python -m kestrel_sovereign.cli update``, which holds
no launcher open).

Everywhere but Windows this finds nothing: POSIX replaces a running file.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

DISTRIBUTION = "kestrel-sovereign"
# Read at call time, so tests can exercise the Windows path on any OS.
_ENABLED = os.name == "nt"

_DELETE = 0x00010000
_FILE_SHARE_ALL = 0x00000001 | 0x00000002 | 0x00000004  # read | write | delete
_OPEN_EXISTING = 3
_ERROR_FILE_NOT_FOUND = 2
_ERROR_SHARING_VIOLATION = 32


def console_script_names(distribution: str = DISTRIBUTION) -> tuple[str, ...]:
    """The console scripts ``distribution`` installs (``kestrel`` for core)."""
    entry_points = importlib.metadata.distribution(distribution).entry_points
    return tuple(sorted(ep.name for ep in entry_points.select(group="console_scripts")))


def scripts_dir(prefix: str | os.PathLike[str] = sys.prefix) -> Path:
    """The directory a Windows venv keeps its launchers in."""
    return Path(prefix) / "Scripts"


def launchers_in_use(
    directory: Path | None = None,
    names: Sequence[str] | None = None,
    *,
    held: Callable[[Path], bool] | None = None,
) -> list[Path]:
    """This distribution's launchers an install could not replace right now.

    Off Windows nothing is looked up and nothing is found.
    """
    if not _ENABLED:
        return []
    directory = scripts_dir() if directory is None else directory
    names = console_script_names() if names is None else names
    probe = _held_open_without_delete_sharing if held is None else held
    launchers = [directory / f"{name}.exe" for name in names]
    return [launcher for launcher in launchers if launcher.exists() and probe(launcher)]


def _held_open_without_delete_sharing(path: Path) -> bool:
    """Whether some process holds ``path`` so that it cannot be renamed or deleted.

    Opens the file for DELETE access, sharing everything: Windows grants it
    only if every existing handle allows deletion, which is exactly what
    replacing the file needs. Nothing is changed and the handle is closed.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.CreateFileW(str(path), _DELETE, _FILE_SHARE_ALL, None,
                                  _OPEN_EXISTING, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        error = ctypes.get_last_error()
        if error == _ERROR_SHARING_VIOLATION:
            return True
        if error == _ERROR_FILE_NOT_FOUND:
            return False
        raise ctypes.WinError(error)
    kernel32.CloseHandle(handle)
    return False


def python_module_command(argv: Sequence[str]) -> str:
    """The same ``kestrel`` command, run through this venv's interpreter."""
    return subprocess.list2cmdline([sys.executable, "-m", "kestrel_sovereign.cli", *argv])


def refusal(in_use: Sequence[Path], argv: Sequence[str]) -> str:
    names = ", ".join(path.name for path in in_use)
    return (
        f"kestrel update: refusing to start: {names} is running, and Windows does not let an "
        "install replace a running launcher, so the install step would fail "
        "(os error 32) after the pull. Nothing has been changed.\n"
        "Run the update through Python instead, which holds no launcher open:\n"
        f"    {python_module_command(argv)}\n"
        "If another `kestrel` command is also running (logs --follow, a shell), close it first."
    )
