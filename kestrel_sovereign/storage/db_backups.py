"""Where a reanchor's copy of an agent database goes, and how (#3785).

``kestrel constitution reanchor --force`` copies a SQLite agent database
before it writes. The copies used to land beside the database as
``kestrel_prime.db.backup-<YYYYmmdd-HHMMSS>``, where nothing ever moved one:
on the production fleet they reached 18.6 GB in 24 files under
``agent_data/``. They now go under the backup root
(:func:`kestrel_sovereign.paths.backup_dir`), laid out like the rest of that
tree::

    <backup root>/<YYYY-MM-DD>-reanchor/<agent data dir>/kestrel_prime.db.backup-<YYYYmmdd-HHMMSS>

:func:`write_reanchor_backup` writes each copy under a staging name in its
final directory, flushes it to disk, and only then publishes it under its
final name, which is never replaced. A copy that fails part-way leaves
nothing under that name.

Nothing here deletes a copy: which copies to keep is the Sovereign's call.
"""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
import time
from pathlib import Path

_EVENT_DATE_FORMAT = "%Y-%m-%d"
_STAMP_FORMAT = "%Y%m%d-%H%M%S"
# Never ``.backup-``: that suffix is how a finished copy is recognized.
_STAGING_INFIX = ".partial-backup-"
_COPY_CHUNK_BYTES = 1 << 20
_DIRECTORY_MODE = 0o700
#: Copies of one database published within one second before giving up.
_MAX_COPIES_PER_SECOND = 100


def reanchor_backup_path(db_path: Path, backup_root: Path) -> Path:
    """Where a reanchor copies *db_path* now, below *backup_root*."""
    moment = time.localtime()
    database = Path(os.path.abspath(db_path))
    return (
        Path(backup_root)
        / f"{time.strftime(_EVENT_DATE_FORMAT, moment)}-reanchor"
        / database.parent.name
        / f"{database.name}.backup-{time.strftime(_STAMP_FORMAT, moment)}"
    )


def _fsync_directory(path: Path) -> None:
    """Make a new directory entry in *path* durable (POSIX only)."""
    if os.name == "nt":  # pragma: no cover - directory fsync is POSIX-specific
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_directory(path: Path) -> None:
    """Create *path* (never its parent) as ``0700``; accept an existing one."""
    try:
        path.mkdir(mode=_DIRECTORY_MODE)
    except FileExistsError:
        if path.is_dir():
            return
        reason = (
            "is a symbolic link to a directory that does not exist (an "
            "unmounted volume?)"
            if path.is_symlink()
            else "exists but is not a directory"
        )
        raise NotADirectoryError(errno.ENOTDIR, reason, str(path)) from None
    _fsync_directory(path.parent)


def write_reanchor_backup(db_path: Path, backup_root: Path) -> Path:
    """Copy *db_path* under *backup_root* and return the copy's path.

    The backup root is created when missing, but never its parent: a root on
    a volume that is not mounted is refused rather than recreated on the disk
    beneath the mount point. The bytes go to a staging file in the copy's
    final directory and are flushed to disk; only then is the file linked
    under its final name, the atomic rename that never replaces an existing
    file. A second copy within the same second is published as
    ``<name>-1`` (then ``-2``, ...) instead. Like ``shutil.copy2``, the copy
    keeps the database's mode and mtime.

    Raises ``OSError`` when the copy cannot be written.
    """
    db_path = Path(db_path)
    backup_root = Path(backup_root)
    final = reanchor_backup_path(db_path, backup_root)
    try:
        _make_directory(backup_root)
    except FileNotFoundError:
        raise FileNotFoundError(
            errno.ENOENT,
            "the backup root's parent directory does not exist (an unmounted "
            "volume?); create it, or set KESTREL_BACKUP_DIR to another "
            "directory",
            str(backup_root),
        ) from None
    _make_directory(final.parent.parent)
    _make_directory(final.parent)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f"{db_path.name}{_STAGING_INFIX}", dir=final.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            with open(db_path, "rb") as source:
                shutil.copyfileobj(source, target, _COPY_CHUNK_BYTES)
            target.flush()
            os.fsync(target.fileno())
        shutil.copystat(db_path, staging)
        published = _publish(staging, final)
        _fsync_directory(final.parent)
    finally:
        staging.unlink(missing_ok=True)
    return published


def _publish(staging: Path, final: Path) -> Path:
    """Link *staging* under *final*, or the first free ``<final>-<n>``."""
    for attempt in range(_MAX_COPIES_PER_SECOND):
        candidate = final.with_name(f"{final.name}-{attempt}") if attempt else final
        try:
            os.link(staging, candidate, follow_symlinks=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(
        errno.EEXIST,
        f"{_MAX_COPIES_PER_SECOND} reanchor copies already carry this stamp, "
        "and none is ever overwritten; retry in a second",
        str(final),
    )
