"""Let an install replace the ``kestrel.exe`` launcher that is running it (#2923).

On Windows a running executable cannot be deleted or overwritten, but it can
be renamed. ``kestrel update`` runs inside ``kestrel.exe``, the console-script
launcher, which waits on the interpreter it started for as long as the update
runs. Any install that reinstalls ``kestrel-sovereign`` (``uv sync`` after a
version bump, ``uv pip install -e .``) must replace that launcher, and failed
with ``os error 32``. That happened on every such update, whether or not an
agent was running: measured on windows-latest with the host stopped and its
port closed.

So before the install, this module renames the distribution's launchers aside.
The install then writes new ones in their place. If the install leaves no new
launcher behind (a failure, an interrupt), the old one is put back, so the venv
is never left without a ``kestrel`` command. A renamed copy that is still running
cannot be deleted; the next update removes it.

Everywhere but Windows this is a no-op: POSIX replaces a running file in place.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DISTRIBUTION = "kestrel-sovereign"
_ASIDE_MARKER = ".exe.replaced-"
# Read at call time, so tests can exercise the Windows path on any OS.
_ENABLED = os.name == "nt"


@dataclass(frozen=True)
class MovedLauncher:
    original: Path
    aside: Path


def console_script_names(distribution: str = DISTRIBUTION) -> tuple[str, ...]:
    """The console scripts ``distribution`` installs (``kestrel`` for core)."""
    entry_points = importlib.metadata.distribution(distribution).entry_points
    return tuple(sorted(ep.name for ep in entry_points.select(group="console_scripts")))


def scripts_dir(prefix: str | os.PathLike[str] = sys.prefix) -> Path:
    """The directory a Windows venv keeps its launchers in."""
    return Path(prefix) / "Scripts"


def move_aside(directory: Path, names: tuple[str, ...]) -> list[MovedLauncher]:
    """Rename each ``<name>.exe`` in ``directory`` aside; what was moved.

    A rename that fails raises: an install that would then hit the lock is
    better refused here, naming the file, than reported as ``os error 32``
    from inside ``uv``.
    """
    if not _ENABLED:
        return []
    moved: list[MovedLauncher] = []
    stamp = f"{os.getpid()}-{time.time_ns()}"
    try:
        for name in names:
            original = directory / f"{name}.exe"
            if not original.exists():
                continue
            aside = directory / f"{name}{_ASIDE_MARKER}{stamp}"
            original.rename(aside)
            moved.append(MovedLauncher(original, aside))
    except OSError:
        restore_missing(moved)
        raise
    return moved


def restore_missing(moved: list[MovedLauncher]) -> list[Path]:
    """Put back each launcher the install did not replace; the paths restored."""
    restored = []
    for launcher in moved:
        if launcher.original.exists() or not launcher.aside.exists():
            continue
        launcher.aside.rename(launcher.original)
        restored.append(launcher.original)
    return restored


def remove_replaced(directory: Path, names: tuple[str, ...]) -> tuple[list[Path], list[Path]]:
    """Delete launchers an earlier update renamed aside: (removed, still in use).

    One still in use is a process from that update still running; it is left
    for the next update and reported, not treated as an error.
    """
    if not _ENABLED:
        return [], []
    removed: list[Path] = []
    in_use: list[Path] = []
    for name in names:
        for aside in sorted(directory.glob(f"{name}{_ASIDE_MARKER}*")):
            try:
                aside.unlink()
            except PermissionError:
                in_use.append(aside)
            else:
                removed.append(aside)
    if in_use:
        logger.info("Kept %d replaced launcher(s) still in use: %s", len(in_use),
                    ", ".join(p.name for p in in_use))
    return removed, in_use
