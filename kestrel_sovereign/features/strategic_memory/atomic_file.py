"""Content digests and atomic replacement for strategic memory's YAML files.

STRATEGY.yaml and STRATEGY_LEDGER.yaml are canonical records an operator may
edit by hand while the agent runs. Each writer keeps the digest of the bytes it
last read or wrote, checks the file still has it before writing, and replaces
the file so it holds the old text or the new, never part of either (#2273,
#3636).
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import stat
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def file_digest(raw: Optional[bytes]) -> Optional[str]:
    """Content hash of a file's bytes; ``None`` when there is no file."""
    return None if raw is None else hashlib.sha256(raw).hexdigest()


def read_file_bytes(path: Path) -> Optional[bytes]:
    """The file's bytes, or ``None`` when it does not exist."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def replace_atomically(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` so it holds the old text or the new, never part.

    Staged in a sibling file, fsynced, and moved over the target with
    ``os.replace``. Writing the target in place truncates it first, so a
    failure part-way through (a full disk, a kill) left a partial file behind.
    Writes through a symlink rather than replacing the link, and keeps the
    existing file's permission bits. A file this process may not write is
    refused as the in-place write refused it: ``os.replace`` needs only the
    directory to be writable, and would otherwise override a read-only file.
    ``newline=""`` keeps the bytes on disk equal to ``content`` on every
    platform, so the digest of what was written matches the next read of it.
    """
    target = Path(os.path.realpath(path))
    try:
        mode: Optional[int] = stat.S_IMODE(os.stat(target).st_mode)
    except FileNotFoundError:
        mode = None  # a new file keeps the umask-derived mode write_text gives it
    if mode is not None and not os.access(target, os.W_OK):
        raise PermissionError(errno.EACCES, "Permission denied", str(target))
    staging = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        staging.write_text(content, encoding="utf-8", newline="")
        descriptor = os.open(staging, os.O_RDWR)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if mode is not None:
            os.chmod(staging, mode)
        os.replace(staging, target)
    except BaseException:
        try:
            staging.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logger.warning(
                "Could not remove staging file %s: %s", staging, cleanup_error
            )
        raise
