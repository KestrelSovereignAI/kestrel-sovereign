"""Bounded reads of a file that must be a regular file.

A configuration file an agent can replace (the Sovereign trust root, the
corpus exclusion manifest) can be replaced by a FIFO. ``open()`` on a FIFO
blocks until something opens it for writing, so a reader that opened it
plainly would hang, and on the event loop would stop the whole host (#3893).

The file is opened with ``O_NONBLOCK`` and checked on the open descriptor:
a FIFO, device, socket or directory is refused at once, and what is read is
what was checked. On a regular file the flag is inert.
"""

from __future__ import annotations

import os
import stat

_READ_CHUNK_BYTES = 64 * 1024


class NotARegularFileError(OSError):
    """The path names a FIFO, device, socket or directory, not a regular file."""


def read_regular_file(path: str | os.PathLike[str], *, max_bytes: int) -> bytes:
    """Read at most ``max_bytes + 1`` bytes of the regular file at ``path``.

    One byte past the limit is read so the caller can tell a file over it
    from one at it. Raises ``FileNotFoundError`` when nothing is at ``path``,
    :class:`NotARegularFileError` when something other than a regular file
    is, and ``OSError`` when it cannot be read.
    """
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise NotARegularFileError(f"{os.fspath(path)} is not a regular file")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)
