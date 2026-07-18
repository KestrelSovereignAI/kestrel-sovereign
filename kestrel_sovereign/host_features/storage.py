"""Private path resolution and migration for fleet/host feature SQLite state."""

from __future__ import annotations

import errno
import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Optional

from kestrel_sovereign.paths import host_data_dir, project_dir
from kestrel_sovereign.private_storage import (
    PRIVATE_FILE_MODE,
    PrivateStorageError,
    absolute_without_following_leaf,
    ensure_private_directory,
    ensure_private_file,
    open_private_file,
    path_exists,
    require_private_directory,
)

HOST_DB_PATH_ENV = "KESTREL_HOST_DB_PATH"
HOST_FEATURE_DB_FILENAME = "host-features.db"
LEGACY_HOST_DB_FILENAME = "kestrel_host.db"
SQLITE_AUXILIARY_SUFFIXES = ("-wal", "-shm", "-journal")

# Public host-domain name while retaining the shared primitive's exception
# identity, so callers catch failures from every custody operation uniformly.
HostStorageError = PrivateStorageError

logger = logging.getLogger(__name__)


def host_database_path(db_path: Optional[str] = None) -> tuple[Path, bool]:
    """Return ``(absolute path, uses implicit default)`` without filesystem I/O."""
    explicit = db_path or os.environ.get(HOST_DB_PATH_ENV)
    if explicit:
        return absolute_without_following_leaf(Path(explicit)), False
    return host_data_dir() / HOST_FEATURE_DB_FILENAME, True


def legacy_host_database_path() -> Path:
    """Return the pre-#2610 project-root host-feature database location."""
    return absolute_without_following_leaf(project_dir() / LEGACY_HOST_DB_FILENAME)


def sqlite_family(path: Path) -> tuple[Path, ...]:
    """Main SQLite file plus every sensitive on-disk auxiliary it may create."""
    return (path, *(Path(f"{path}{suffix}") for suffix in SQLITE_AUXILIARY_SUFFIXES))


def _family_exists(path: Path) -> bool:
    return any(path_exists(member) for member in sqlite_family(path))


def _harden_existing_family(path: Path, *, label: str) -> None:
    existing = [member for member in sqlite_family(path) if path_exists(member)]
    if not existing:
        return
    if path not in existing:
        raise HostStorageError(
            f"{label} has SQLite auxiliary files but no main database: {path}"
        )
    for member in existing:
        ensure_private_file(member, label=label)


def validate_sqlite_family_private(path: Path, *, label: str = "host database") -> None:
    """Fail if an opened SQLite family is not regular, exclusive, and ``0600``."""
    if not path_exists(path):
        raise HostStorageError(f"{label} main file is missing: {path}")
    for member in sqlite_family(path):
        if not path_exists(member):
            continue
        try:
            st = member.lstat()
        except OSError as exc:
            raise HostStorageError(
                f"cannot inspect {label} file {member}: {exc}"
            ) from exc
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise HostStorageError(
                f"{label} file must be regular and not a link: {member}"
            )
        if st.st_nlink != 1:
            raise HostStorageError(
                f"{label} file has {st.st_nlink} hard links; exclusive custody "
                f"cannot be established: {member}"
            )
        if os.name != "nt" and stat.S_IMODE(st.st_mode) != PRIVATE_FILE_MODE:
            raise HostStorageError(
                f"{label} file {member} must have mode 0600; found "
                f"{stat.S_IMODE(st.st_mode):04o}"
            )


def _copy_database_across_filesystems(source: Path, destination: Path) -> None:
    """Publish a stopped database through a private, fsynced staging file."""
    staging_fd = -1
    staging: Optional[Path] = None
    try:
        staging_fd, staging_name = tempfile.mkstemp(
            prefix=".host-features-migrate-",
            dir=destination.parent,
        )
        staging = Path(staging_name)
        if hasattr(os, "fchmod"):
            os.fchmod(staging_fd, PRIVATE_FILE_MODE)
    except OSError as exc:
        if staging_fd >= 0:
            os.close(staging_fd)
        if staging is not None:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass
        raise HostStorageError(
            f"cannot create private host database migration staging file: {exc}"
        ) from exc

    source_fd: Optional[int] = None
    try:
        source_fd = open_private_file(source, os.O_RDONLY, label="legacy host database")
        with os.fdopen(source_fd, "rb", closefd=True) as source_file:
            source_fd = None
            with os.fdopen(staging_fd, "wb", closefd=True) as staging_file:
                staging_fd = -1
                shutil.copyfileobj(source_file, staging_file)
                staging_file.flush()
                os.fsync(staging_file.fileno())
        ensure_private_file(staging, label="host database migration")
        os.replace(staging, destination)
        validate_sqlite_family_private(destination)
        _fsync_directory(destination.parent)
        source.unlink()
        _fsync_directory(source.parent)
    except (OSError, shutil.Error, HostStorageError) as exc:
        if source_fd is not None:
            os.close(source_fd)
        if staging_fd >= 0:
            os.close(staging_fd)
        try:
            if staging is not None and path_exists(staging):
                staging.unlink()
        except OSError:
            pass
        raise HostStorageError(
            f"cannot safely migrate host database from {source} to "
            f"{destination}: {exc}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    """Durably order a migration rename/unlink on POSIX filesystems."""
    if os.name == "nt":  # pragma: no cover - directory fsync is POSIX-specific
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise HostStorageError(
            f"cannot fsync host database directory {path}: {exc}"
        ) from exc


def _migrate_legacy_database(destination: Path) -> None:
    legacy = legacy_host_database_path()
    if legacy == destination or not _family_exists(legacy):
        return

    # Contain the old disclosure before deciding whether migration is safe.
    _harden_existing_family(legacy, label="legacy host database")

    active_sidecars = [
        member
        for member in sqlite_family(legacy)[1:]
        if path_exists(member)
    ]
    if active_sidecars:
        names = ", ".join(member.name for member in active_sidecars)
        raise HostStorageError(
            f"legacy host database {legacy} still has SQLite sidecars ({names}); "
            "another Kestrel process may be using it. Stop every old host and "
            "restart to migrate after a clean SQLite shutdown"
        )

    if _family_exists(destination):
        _harden_existing_family(destination, label="host database destination")
        raise HostStorageError(
            f"both legacy host database {legacy} and destination {destination} "
            "contain state; the host store is disabled rather than guessing or "
            "merging SQLite histories. Back up both files, choose the "
            "authoritative database, and move the other aside"
        )

    try:
        os.replace(legacy, destination)
        validate_sqlite_family_private(destination)
        _fsync_directory(destination.parent)
        if legacy.parent != destination.parent:
            _fsync_directory(legacy.parent)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise HostStorageError(
                f"cannot migrate host database from {legacy} to {destination}: {exc}"
            ) from exc
        _copy_database_across_filesystems(legacy, destination)

    logger.warning(
        "Migrated legacy host-feature database from %s to private host-data "
        "location %s.",
        legacy,
        destination,
    )


def prepare_host_database(db_path: Optional[str] = None) -> Path:
    """Resolve, migrate, and securely pre-create the host-feature database.

    Pre-creating the main file as ``0600`` is the secure-at-creation boundary
    for SQLite on POSIX. The standard Unix VFS creates WAL/journal/SHM files
    with the main database's exact mode, independent of the process umask.
    """
    destination, uses_default = host_database_path(db_path)
    parent = destination.parent
    if uses_default:
        ensure_private_directory(parent, label="host data")
        _migrate_legacy_database(destination)
    elif path_exists(parent):
        # Never chmod an operator's shared parent such as /data or /tmp.
        require_private_directory(parent, label="host database")
    else:
        ensure_private_directory(parent, label="host database")

    _harden_existing_family(destination, label="host database")
    ensure_private_file(destination, label="host database")
    validate_sqlite_family_private(destination)
    return destination


__all__ = [
    "HOST_DB_PATH_ENV",
    "HOST_FEATURE_DB_FILENAME",
    "LEGACY_HOST_DB_FILENAME",
    "HostStorageError",
    "host_database_path",
    "legacy_host_database_path",
    "prepare_host_database",
    "sqlite_family",
    "validate_sqlite_family_private",
]
