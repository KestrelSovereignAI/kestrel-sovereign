"""Private path resolution and migration for fleet/host feature SQLite state."""

from __future__ import annotations

import errno
import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Mapping, Optional

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
DERIVED_HOST_DB_PATH_ENV = "KESTREL_DERIVED_HOST_DB_PATH"
AGENT_DB_PATH_ENV = "KESTREL_DB_PATH"
HOST_FEATURE_DB_FILENAME = "host-features.db"
LEGACY_HOST_DB_FILENAME = "kestrel_host.db"
SQLITE_AUXILIARY_SUFFIXES = ("-wal", "-shm", "-journal")

# Public host-domain name while retaining the shared primitive's exception
# identity, so callers catch failures from every custody operation uniformly.
HostStorageError = PrivateStorageError

logger = logging.getLogger(__name__)


def _runtime_path(value: str, env: Mapping[str, str], base_dir: Path) -> Path:
    """Resolve one path using the target runtime's home and working directory."""

    runtime_home = env.get("HOME") or env.get("USERPROFILE")
    if runtime_home and value == "~":
        candidate = Path(runtime_home)
    elif runtime_home and value.startswith(("~/", "~\\")):
        candidate = Path(runtime_home) / value[2:]
    else:
        candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return absolute_without_following_leaf(candidate)


def _default_host_database_path(
    env: Mapping[str, str], base_dir: Path
) -> Path:
    """Resolve the private default from a described runtime, without mutating it."""

    configured_home = env.get("KESTREL_HOME")
    if configured_home:
        root = _runtime_path(configured_home, env, base_dir)
    else:
        runtime_home = env.get("HOME") or env.get("USERPROFILE")
        root = (
            _runtime_path(runtime_home, env, base_dir) / ".kestrel"
            if runtime_home
            else absolute_without_following_leaf(Path.home() / ".kestrel")
        )
    return root / "host-data" / HOST_FEATURE_DB_FILENAME


def host_database_path(
    db_path: Optional[str] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    base_dir: Optional[Path] = None,
) -> tuple[Path, bool]:
    """Return ``(absolute path, uses implicit default)`` without filesystem I/O.

    ``env`` and ``base_dir`` let launchers and offline validation resolve the
    path for the runtime they are describing.  Omitting them preserves the
    live-process contract: ``os.environ`` plus the current working directory.
    """

    runtime_env = os.environ if env is None else env
    runtime_base = absolute_without_following_leaf(base_dir or Path.cwd())
    explicit = db_path or runtime_env.get(HOST_DB_PATH_ENV)
    if explicit:
        return _runtime_path(explicit, runtime_env, runtime_base), False
    agent_data_root = runtime_env.get(AGENT_DB_PATH_ENV)
    if agent_data_root:
        return (
            _runtime_path(agent_data_root, runtime_env, runtime_base)
            / "host-data"
            / HOST_FEATURE_DB_FILENAME,
            False,
        )
    return _default_host_database_path(runtime_env, runtime_base), True


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
        source_fd = open_private_file(
            source,
            os.O_RDONLY,
            label="prior host database",
        )
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


def _hold_evidence_family(path: Path) -> tuple[Path, ...]:
    """Return every Hold witness that must move atomically with ``path``."""

    # Lazy import avoids making private host-path resolution depend on Hold at
    # module import time.  Hold itself imports this module for its read-only
    # SQLite readiness checks.
    from kestrel_sovereign.hold.state import (
        hold_history_anchor_path,
        hold_initialization_witness_path,
        hold_sqlite_custody_marker_path,
    )

    history = hold_history_anchor_path(path)
    return (
        hold_initialization_witness_path(path),
        history,
        Path(f"{history}.pending"),
        Path(f"{history}.bootstrap"),
        Path(f"{history}.lock"),
        hold_sqlite_custody_marker_path(path),
    )


def validate_host_database_migration_readiness(
    destination: Path,
    sources: tuple[tuple[str, Path], ...],
) -> Optional[tuple[str, Path]]:
    """Read-only validation of the migration runtime would perform.

    Returns the one stopped, pre-Hold database that may be migrated.  Hold's
    database and external witnesses form one custody unit, so an initialized
    Hold store is explicitly refused instead of moving only the SQLite member
    and silently stranding its authority evidence.
    """

    destination = absolute_without_following_leaf(destination)
    occupied: list[tuple[str, Path]] = []
    for label, raw_source in sources:
        source = absolute_without_following_leaf(raw_source)
        if source == destination:
            continue
        family = tuple(
            member for member in sqlite_family(source) if path_exists(member)
        )
        evidence = tuple(
            member for member in _hold_evidence_family(source) if path_exists(member)
        )
        if not family and not evidence:
            continue
        if source not in family:
            remnants = ", ".join(member.name for member in (*family, *evidence))
            raise HostStorageError(
                f"{label} {source} is missing while custody remnants remain "
                f"({remnants}); the host store is disabled rather than "
                "treating prior state as a first boot"
            )
        try:
            source_stat = source.lstat()
        except OSError as exc:
            raise HostStorageError(f"cannot inspect {label} {source}: {exc}") from exc
        if (
            stat.S_ISLNK(source_stat.st_mode)
            or not stat.S_ISREG(source_stat.st_mode)
            or source_stat.st_nlink != 1
        ):
            raise HostStorageError(
                f"{label} must be a regular, exclusively linked file: {source}"
            )
        active_sidecars = family[1:]
        if active_sidecars:
            names = ", ".join(member.name for member in active_sidecars)
            raise HostStorageError(
                f"{label} {source} still has SQLite sidecars ({names}); another "
                "Kestrel process may be using it. Stop every old host and "
                "restart to migrate after a clean SQLite shutdown"
            )
        if evidence:
            names = ", ".join(member.name for member in evidence)
            raise HostStorageError(
                f"{label} {source} has Hold custody evidence ({names}); move the "
                "stopped database, its hold-* witnesses, and .hold-custody "
                "directory together rather than migrating only the database"
            )
        occupied.append((label, source))

    if len(occupied) > 1:
        locations = ", ".join(str(source) for _label, source in occupied)
        raise HostStorageError(
            "multiple prior host databases contain state "
            f"({locations}); the host store is disabled rather than guessing "
            "which history is authoritative"
        )
    if occupied and _family_exists(destination):
        label, source = occupied[0]
        raise HostStorageError(
            f"both {label} {source} and destination {destination} contain state; "
            "the host store is disabled rather than guessing or merging SQLite "
            "histories. Back up both files, choose the authoritative database, "
            "and move the other aside"
        )
    return occupied[0] if occupied else None


def _migrate_prior_database(
    destination: Path,
    sources: tuple[tuple[str, Path], ...],
) -> None:
    # Contain historical disclosures before reporting why migration cannot
    # continue. This preserves the existing runtime contract: a stopped 0644
    # legacy family is restricted even when live sidecars or a second history
    # make automatic migration unsafe.
    for label, source in sources:
        if absolute_without_following_leaf(source) != destination and _family_exists(
            source
        ):
            _harden_existing_family(source, label=label)
    if _family_exists(destination):
        _harden_existing_family(destination, label="host database destination")

    selected = validate_host_database_migration_readiness(destination, sources)
    if selected is None:
        return
    label, source = selected

    try:
        os.replace(source, destination)
        validate_sqlite_family_private(destination)
        _fsync_directory(destination.parent)
        if source.parent != destination.parent:
            _fsync_directory(source.parent)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise HostStorageError(
                f"cannot migrate host database from {source} to {destination}: {exc}"
            ) from exc
        _copy_database_across_filesystems(source, destination)

    logger.warning(
        "Migrated prior host-feature database from %s to private host-data "
        "location %s.",
        source,
        destination,
    )


def prepare_host_database(db_path: Optional[str] = None) -> Path:
    """Resolve, migrate, and securely pre-create the host-feature database.

    Pre-creating the main file as ``0600`` is the secure-at-creation boundary
    for SQLite on POSIX. The standard Unix VFS creates WAL/journal/SHM files
    with the main database's exact mode, independent of the process umask.
    """
    destination, uses_default = host_database_path(db_path)
    configured_host_path = os.environ.get(HOST_DB_PATH_ENV)
    derived_host_path = os.environ.get(DERIVED_HOST_DB_PATH_ENV)
    launcher_derived_override = bool(
        not db_path
        and configured_host_path
        and derived_host_path == configured_host_path
    )
    explicit_override = bool(
        db_path or (configured_host_path and not launcher_derived_override)
    )
    previous_default = host_data_dir() / HOST_FEATURE_DB_FILENAME
    follows_agent_data_root = not explicit_override and destination != previous_default
    parent = destination.parent
    if uses_default:
        ensure_private_directory(parent, label="host data")
    elif path_exists(parent):
        # Never chmod an operator's shared parent such as /data or /tmp.
        require_private_directory(parent, label="host database")
    else:
        ensure_private_directory(parent, label="host database")

    if not explicit_override:
        sources: list[tuple[str, Path]] = []
        if follows_agent_data_root:
            sources.append(
                (
                    "previous default host database",
                    previous_default,
                )
            )
        sources.append(("legacy host database", legacy_host_database_path()))
        _migrate_prior_database(destination, tuple(sources))

    _harden_existing_family(destination, label="host database")
    ensure_private_file(destination, label="host database")
    validate_sqlite_family_private(destination)
    return destination


__all__ = [
    "AGENT_DB_PATH_ENV",
    "DERIVED_HOST_DB_PATH_ENV",
    "HOST_DB_PATH_ENV",
    "HOST_FEATURE_DB_FILENAME",
    "LEGACY_HOST_DB_FILENAME",
    "HostStorageError",
    "host_database_path",
    "legacy_host_database_path",
    "prepare_host_database",
    "sqlite_family",
    "validate_host_database_migration_readiness",
    "validate_sqlite_family_private",
]
