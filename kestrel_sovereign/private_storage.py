"""POSIX-oriented custody primitives for sensitive host-runtime files.

The helpers in this module establish a narrow, reusable contract shared by
host-owned stores: real directories, regular single-link files, no leaf
symlink traversal, and owner-only modes.  They intentionally do not decide
*where* a feature stores data or how a feature migrates it.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class PrivateStorageError(RuntimeError):
    """Sensitive local storage cannot be opened with exclusive custody."""


def _lock_private_file_descriptor(descriptor: int) -> Any:
    """Take one blocking cross-process exclusive lock."""

    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class _Overlapped(ctypes.Structure):
            _fields_ = [
                ("Internal", ctypes.c_size_t),
                ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD),
                ("OffsetHigh", wintypes.DWORD),
                ("hEvent", wintypes.HANDLE),
            ]

        overlapped = _Overlapped()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.LockFileEx(
            wintypes.HANDLE(msvcrt.get_osfhandle(descriptor)),
            0x00000002,  # LOCKFILE_EXCLUSIVE_LOCK
            0,
            1,
            0,
            ctypes.byref(overlapped),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return overlapped

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return None


def _unlock_private_file_descriptor(descriptor: int, token: Any) -> None:
    """Release a lock obtained by ``_lock_private_file_descriptor``."""

    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.UnlockFileEx(
            wintypes.HANDLE(msvcrt.get_osfhandle(descriptor)),
            0,
            1,
            0,
            ctypes.byref(token),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def absolute_without_following_leaf(path: Path) -> Path:
    """Return an absolute normalized path without resolving a leaf symlink."""
    return Path(os.path.abspath(path.expanduser()))


def path_exists(path: Path) -> bool:
    """Return whether ``path`` exists as a directory entry, including links."""
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _create_missing_private_directories(path: Path, *, label: str) -> None:
    """Create every missing path component as ``0700``.

    ``Path.mkdir(parents=True, mode=...)`` applies ``mode`` only to the leaf;
    under a permissive umask it can briefly create intermediate custody
    directories as ``0777``. Build the missing suffix one component at a time
    so every object is private at creation, and validate a concurrent creator.
    """
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            cursor.lstat()
            break
        except FileNotFoundError:
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                break
            cursor = parent
        except OSError as exc:
            raise PrivateStorageError(
                f"cannot inspect {label} directory {cursor}: {exc}"
            ) from exc

    for directory in reversed(missing):
        try:
            directory.mkdir(mode=PRIVATE_DIRECTORY_MODE, exist_ok=False)
            st = directory.lstat()
        except FileExistsError:
            # Validate an object created after the missing-path walk.
            st = directory.lstat()
        except OSError as exc:
            raise PrivateStorageError(
                f"cannot create private {label} directory {directory}: {exc}"
            ) from exc
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise PrivateStorageError(
                f"{label} custody path must be a real directory, not a link or "
                f"special file: {directory}"
            )
        try:
            directory.chmod(PRIVATE_DIRECTORY_MODE)
        except OSError as exc:
            raise PrivateStorageError(
                f"cannot restrict {label} directory {directory} to mode 0700: "
                f"{exc}"
            ) from exc


def ensure_private_directory(path: Path, *, label: str = "storage") -> None:
    """Create or validate one real directory, then enforce mode ``0700``."""
    try:
        st = path.lstat()
    except FileNotFoundError:
        _create_missing_private_directories(path, label=label)
        st = path.lstat()
    except OSError as exc:
        raise PrivateStorageError(
            f"cannot inspect {label} directory {path}: {exc}"
        ) from exc

    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise PrivateStorageError(
            f"{label} custody path must be a real directory, not a link or "
            f"special file: {path}"
        )
    try:
        path.chmod(PRIVATE_DIRECTORY_MODE)
    except OSError as exc:
        raise PrivateStorageError(
            f"cannot restrict {label} directory {path} to mode 0700: {exc}"
        ) from exc


def require_private_directory(path: Path, *, label: str = "storage") -> None:
    """Validate an operator-owned directory without changing its permissions.

    Explicit file overrides may point beneath directories Kestrel does not own.
    Mutating such a parent (for example ``/data`` or ``/tmp``) would be unsafe,
    so overrides must provide a dedicated directory that is already ``0700``.
    """
    try:
        st = path.lstat()
    except OSError as exc:
        raise PrivateStorageError(
            f"cannot inspect {label} directory {path}: {exc}"
        ) from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise PrivateStorageError(
            f"{label} custody path must be a real directory, not a link or "
            f"special file: {path}"
        )
    if os.name != "nt" and stat.S_IMODE(st.st_mode) != PRIVATE_DIRECTORY_MODE:
        raise PrivateStorageError(
            f"{label} directory {path} must have mode 0700; found "
            f"{stat.S_IMODE(st.st_mode):04o}"
        )


def _open_private_file(
    path: Path,
    flags: int,
    *,
    label: str,
    harden: bool,
) -> int:
    """Open one private file and either harden or validate its custody mode."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and path_exists(path):  # pragma: no cover - Windows fallback
        try:
            if stat.S_ISLNK(path.lstat().st_mode):
                raise PrivateStorageError(
                    f"{label} custody file must not be a symbolic link: {path}"
                )
        except OSError as exc:
            raise PrivateStorageError(
                f"cannot inspect private {label} file {path}: {exc}"
            ) from exc
    try:
        fd = os.open(path, flags | nofollow, PRIVATE_FILE_MODE)
    except OSError as exc:
        raise PrivateStorageError(
            f"cannot open private {label} file {path}: {exc}"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise PrivateStorageError(f"{label} custody file is not regular: {path}")
        if st.st_nlink != 1:
            raise PrivateStorageError(
                f"{label} custody file has {st.st_nlink} hard links; exclusive "
                f"custody cannot be established: {path}"
            )
        if harden:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, PRIVATE_FILE_MODE)
            else:  # pragma: no cover - Windows has no POSIX mode enforcement
                path.chmod(PRIVATE_FILE_MODE)
        elif os.name != "nt" and stat.S_IMODE(st.st_mode) != PRIVATE_FILE_MODE:
            raise PrivateStorageError(
                f"{label} custody file {path} must have mode 0600; found "
                f"{stat.S_IMODE(st.st_mode):04o}"
            )
        return fd
    except (OSError, PrivateStorageError):
        os.close(fd)
        raise


def open_private_file(
    path: Path,
    flags: int,
    *,
    label: str = "storage",
) -> int:
    """Open a non-link, single-link regular file and enforce mode ``0600``."""

    return _open_private_file(path, flags, label=label, harden=True)


def open_private_file_for_validation(
    path: Path,
    flags: int = os.O_RDONLY,
    *,
    label: str = "storage",
) -> int:
    """Open an existing private file without changing its custody metadata."""

    mutating_flags = os.O_CREAT | os.O_TRUNC | os.O_APPEND | os.O_WRONLY
    if flags & mutating_flags:
        raise ValueError("validation-only private file open cannot mutate the file")
    return _open_private_file(path, flags, label=label, harden=False)


def ensure_private_file(path: Path, *, label: str = "storage") -> None:
    """Securely create or harden a regular file without writing content."""
    fd = open_private_file(path, os.O_RDWR | os.O_CREAT, label=label)
    os.close(fd)


@contextmanager
def exclusive_private_file_lock(
    path: Path,
    *,
    label: str = "storage",
) -> Iterator[None]:
    """Serialize one private-file protocol across processes.

    The lock file is durable protocol structure rather than authoritative
    payload. Keeping it after release lets every process contend on the same
    inode and makes process death release ownership without a stale sentinel.
    """

    ensure_private_directory(path.parent, label=label)
    descriptor = open_private_file(
        path,
        os.O_RDWR | os.O_CREAT,
        label=f"{label} lock",
    )
    try:
        token = _lock_private_file_descriptor(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise PrivateStorageError(
            f"cannot lock private {label} file {path}: {exc}"
        ) from exc

    try:
        yield
    finally:
        try:
            _unlock_private_file_descriptor(descriptor, token)
        except OSError:
            # Closing the descriptor releases the ownership even when an
            # explicit unlock reports a teardown-only failure.
            pass
        os.close(descriptor)


__all__ = [
    "PRIVATE_DIRECTORY_MODE",
    "PRIVATE_FILE_MODE",
    "PrivateStorageError",
    "absolute_without_following_leaf",
    "ensure_private_directory",
    "ensure_private_file",
    "exclusive_private_file_lock",
    "open_private_file",
    "open_private_file_for_validation",
    "path_exists",
    "require_private_directory",
]
