"""
Kestrel Compute Feature - Trash Manager.

Manages the trash folder for deleted files, providing restore and cleanup
operations.
"""

import ctypes
import errno
import logging
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .destructive_policy import DEFAULT_TRASH_DIR, DestructiveOperationPolicy

logger = logging.getLogger(__name__)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _entry_deleted_at(path: Path) -> Optional[datetime]:
    """Parse one trash entry timestamp, falling back to its modification time."""
    parts = path.name.split("_", 2)
    if len(parts) >= 2:
        try:
            return datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def _secure_restore_supported() -> bool:
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and sys.platform in {"darwin", "linux"}
        and all(
            function in os.supports_dir_fd
            for function in (
                os.mkdir,
                os.open,
                os.readlink,
                os.rmdir,
                os.stat,
            )
        )
    )


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _same_file_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _open_directory_chain(path: Path, *, create: bool = False) -> int:
    """Open an absolute directory without following any component symlink."""
    if not path.is_absolute():
        raise ValueError(f"Expected an absolute directory path: {path}")

    descriptor = os.open(path.anchor, _directory_flags())
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o777, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent creator won the race.  The no-follow open
                    # below deterministically accepts only a real directory.
                    pass
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _rename_noreplace(
    source: str,
    destination: str,
    *,
    source_dir_fd: int,
    destination_dir_fd: int,
    expected_source_stat: os.stat_result,
) -> None:
    """Atomically rename without replacing a concurrently-created target."""
    current_source_stat = os.stat(
        source,
        dir_fd=source_dir_fd,
        follow_symlinks=False,
    )
    if not _same_file_identity(expected_source_stat, current_source_stat):
        raise PermissionError("Trash item changed while restore was running")

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)

    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_dir_fd,
            encoded_source,
            destination_dir_fd,
            encoded_destination,
            0x00000004,  # RENAME_EXCL
        )
    else:
        try:
            rename = libc.renameat2
        except AttributeError as exc:
            raise PermissionError(
                "Atomic no-replace trash restore is unavailable"
            ) from exc
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_dir_fd,
            encoded_source,
            destination_dir_fd,
            encoded_destination,
            1,  # RENAME_NOREPLACE
        )

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), source)


@dataclass
class TrashItem:
    """An item in the trash folder."""

    name: str  # Original filename
    path: Path  # Current path in trash
    original_path: Optional[str]  # Original location (if known)
    deleted_at: datetime  # When it was deleted
    size_bytes: int  # File/folder size
    is_dir: bool  # Whether it's a directory

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "path": str(self.path),
            "original_path": self.original_path,
            "deleted_at": self.deleted_at.isoformat(),
            "size_bytes": self.size_bytes,
            "is_dir": self.is_dir,
        }


class TrashManager:
    """
    Manage the trash folder for safe file deletion.

    The trash folder contains timestamped subdirectories with deleted files.
    This allows restore operations and cleanup of old items.

    Example:
        manager = TrashManager()

        # List recent trash items
        items = manager.list_items(days=7)

        # Restore an item
        manager.restore(item.path, "/original/location")

        # Empty old trash
        deleted_count = manager.empty(older_than_days=30)
    """

    def __init__(
        self,
        trash_dir: Optional[Path] = None,
        current_agent_data_path: Optional[str | Path] = None,
    ):
        """
        Initialize the trash manager.

        Args:
            trash_dir: Path to trash directory (default: ~/.kestrel/trash)
            current_agent_data_path: This agent's own data directory. Restore
                targets under other agent_data children are rejected.
        """
        self.trash_dir = trash_dir or DEFAULT_TRASH_DIR
        self._policy = DestructiveOperationPolicy(
            trash_dir=self.trash_dir,
            current_agent_data_path=current_agent_data_path,
        )

    def ensure_trash_dir(self) -> None:
        """Create trash directory if it doesn't exist."""
        self.trash_dir.mkdir(parents=True, exist_ok=True)

    def list_items(
        self,
        days: Optional[int] = None,
        limit: int = 100,
    ) -> List[TrashItem]:
        """
        List items in the trash folder.

        Args:
            days: Only show items from the last N days (None = all)
            limit: Maximum number of items to return

        Returns:
            List of TrashItems, newest first
        """
        items: List[TrashItem] = []

        if not self.trash_dir.exists():
            return items

        cutoff = None
        if days is not None:
            cutoff = datetime.now() - timedelta(days=days)

        # Iterate through timestamped subdirectories
        subdirs = sorted(self.trash_dir.iterdir(), reverse=True)

        for subdir in subdirs:
            if not subdir.is_dir():
                continue

            deleted_at = _entry_deleted_at(subdir)
            if deleted_at is None:
                continue

            if cutoff and deleted_at < cutoff:
                continue

            # List files in this trash subdirectory
            try:
                for item_path in subdir.iterdir():
                    try:
                        stat = item_path.stat()
                        is_dir = item_path.is_dir()

                        if is_dir:
                            size = self._get_dir_size(item_path)
                        else:
                            size = stat.st_size

                        items.append(
                            TrashItem(
                                name=item_path.name,
                                path=item_path,
                                original_path=None,  # We don't track original path currently
                                deleted_at=deleted_at,
                                size_bytes=size,
                                is_dir=is_dir,
                            )
                        )

                        if len(items) >= limit:
                            return items

                    except OSError as e:
                        logger.warning(f"Error reading trash item {item_path}: {e}")
                        continue

            except OSError as e:
                logger.warning(f"Error reading trash subdir {subdir}: {e}")
                continue

        return items

    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of a directory."""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def restore(
        self,
        trash_path: Path,
        destination: Optional[str] = None,
    ) -> Path:
        """
        Restore an item from trash.

        Args:
            trash_path: Path to the item in trash
            destination: Where to restore (default: current directory)

        Returns:
            Path to the restored item

        Raises:
            FileNotFoundError: If trash item doesn't exist
            FileExistsError: If destination already exists
            PermissionError: If the source is not contained by the trash root
        """
        # Determine destination
        if destination:
            dest_path = Path(destination)
        else:
            dest_path = Path.cwd() / trash_path.name

        self._restore_anchored(trash_path, dest_path)
        logger.info("Restored %s to %s", trash_path.name, dest_path)
        return dest_path

    def _restore_anchored(self, trash_path: Path, dest_path: Path) -> None:
        """Rename one direct trash item through opened, no-follow directories.

        The source operation directory is opened relative to an already-open
        trash-root descriptor.  Later renames or symlink swaps of either lexical
        path therefore cannot redirect the move to an outside host file.
        """
        if not _secure_restore_supported():
            raise PermissionError(
                "Secure trash restore is unavailable on this operating system"
            )

        configured_root = _absolute_path(self.trash_dir)
        source_path = _absolute_path(trash_path)
        try:
            relative_source = source_path.relative_to(configured_root)
        except ValueError as exc:
            raise PermissionError(
                f"Refusing to restore source outside trash directory: {trash_path}"
            ) from exc

        # TrashManager lists and restores immediate children of one generated
        # operation directory. Rejecting deeper paths keeps the descriptor walk
        # finite and excludes both traversal and the root audit log.
        if len(relative_source.parts) != 2:
            raise PermissionError(
                f"Refusing to restore source outside trash directory: {trash_path}"
            )
        operation_name, item_name = relative_source.parts

        try:
            resolved_root = configured_root.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Trash directory not found: {self.trash_dir}"
            ) from exc
        expected_root_stat = resolved_root.stat()

        absolute_destination = _absolute_path(dest_path)
        resolved_destination_parent = absolute_destination.parent.resolve(strict=False)
        resolved_destination = resolved_destination_parent / absolute_destination.name
        try:
            expected_destination_parent_stat = resolved_destination_parent.stat()
        except FileNotFoundError:
            expected_destination_parent_stat = None
        # Validate before the secure walk because it may create missing parent
        # components. Validate again against the descriptor-backed path before
        # the rename below.
        self._policy.assert_agent_data_deletion_allowed(
            resolved_destination,
            "trash_restore",
        )

        root_fd = operation_fd = destination_fd = None
        try:
            root_fd = _open_directory_chain(resolved_root)
            if not _same_file_identity(expected_root_stat, os.fstat(root_fd)):
                raise PermissionError(
                    "Trash directory changed while restore was starting"
                )
            try:
                expected_operation_stat = os.stat(
                    operation_name,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
                operation_fd = os.open(
                    operation_name,
                    _directory_flags(),
                    dir_fd=root_fd,
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Trash item not found: {trash_path}") from exc
            except (NotADirectoryError, OSError) as exc:
                raise PermissionError(
                    f"Refusing unsafe trash operation directory: {operation_name}"
                ) from exc
            if not _same_file_identity(
                expected_operation_stat,
                os.fstat(operation_fd),
            ):
                raise PermissionError(
                    "Trash operation directory changed while restore was starting"
                )

            try:
                source_stat = os.stat(
                    item_name,
                    dir_fd=operation_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Trash item not found: {trash_path}") from exc
            if stat.S_ISLNK(source_stat.st_mode):
                link_target = Path(os.readlink(item_name, dir_fd=operation_fd))
                if not link_target.is_absolute():
                    link_target = resolved_root / operation_name / link_target
                try:
                    resolved_link_target = link_target.resolve(strict=True)
                except (FileNotFoundError, OSError, RuntimeError) as exc:
                    raise PermissionError(
                        f"Could not safely resolve trash source: {trash_path}"
                    ) from exc
                if not (
                    resolved_link_target == resolved_root
                    or resolved_link_target.is_relative_to(resolved_root)
                ):
                    raise PermissionError(
                        "Refusing to restore source outside trash directory: "
                        f"{trash_path}"
                    )

            destination_fd = _open_directory_chain(
                resolved_destination_parent,
                create=True,
            )
            if expected_destination_parent_stat is not None and not _same_file_identity(
                expected_destination_parent_stat,
                os.fstat(destination_fd),
            ):
                raise PermissionError(
                    "Restore destination changed while restore was starting"
                )
            # Destination authorization is deliberately checked after the
            # no-follow directory walk. The path below names the directory
            # descriptor that the atomic rename will actually use.
            self._policy.assert_agent_data_deletion_allowed(
                resolved_destination,
                "trash_restore",
            )

            try:
                _rename_noreplace(
                    item_name,
                    absolute_destination.name,
                    source_dir_fd=operation_fd,
                    destination_dir_fd=destination_fd,
                    expected_source_stat=source_stat,
                )
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Destination already exists: {dest_path}"
                ) from exc

            # The source directory descriptor stays anchored even if its
            # lexical path changes while restore is running.
            try:
                os.rmdir(operation_name, dir_fd=root_fd)
            except OSError:
                pass
        finally:
            for descriptor in (destination_fd, operation_fd, root_fd):
                if descriptor is not None:
                    os.close(descriptor)

    def empty(
        self,
        older_than_days: int = 30,
        dry_run: bool = False,
    ) -> int:
        """
        Permanently delete old trash items.

        Args:
            older_than_days: Delete items older than this many days
            dry_run: If True, don't actually delete, just count

        Returns:
            Number of items deleted
        """
        if not self.trash_dir.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=older_than_days)
        deleted_count = 0

        # Iterate through timestamped subdirectories
        for subdir in list(self.trash_dir.iterdir()):
            if not subdir.is_dir():
                continue

            deleted_at = _entry_deleted_at(subdir)
            if deleted_at is None:
                continue

            if deleted_at < cutoff:
                if self._contains_agent_database(subdir):
                    logger.warning(
                        "Refusing to permanently delete trash directory with agent database: %s",
                        subdir,
                    )
                    continue
                if dry_run:
                    # Count items in directory
                    try:
                        deleted_count += sum(1 for _ in subdir.iterdir())
                    except OSError:
                        deleted_count += 1
                else:
                    try:
                        item_count = sum(1 for _ in subdir.iterdir())
                        shutil.rmtree(subdir)
                        deleted_count += item_count
                        logger.info(f"Deleted trash directory: {subdir.name}")
                    except OSError as e:
                        logger.warning(f"Error deleting trash directory {subdir}: {e}")

        return deleted_count

    def _contains_agent_database(self, path: Path) -> bool:
        """Return True for agent database files or directories containing one."""
        protected_names = {
            "kestrel_prime.db",
            "kestrel_prime.db-wal",
            "kestrel_prime.db-shm",
        }
        if path.name in protected_names:
            return True
        if not path.is_dir():
            return False
        try:
            for protected_name in protected_names:
                if any(path.rglob(protected_name)):
                    return True
        except OSError:
            return False
        return False

    def get_stats(self) -> dict:
        """
        Get statistics about the trash folder.

        Returns:
            Dictionary with item_count, total_size_bytes, oldest_item_age_days
        """
        if not self.trash_dir.exists():
            return {
                "item_count": 0,
                "total_size_bytes": 0,
                "oldest_item_age_days": None,
            }

        items = self.list_items(limit=10000)

        if not items:
            return {
                "item_count": 0,
                "total_size_bytes": 0,
                "oldest_item_age_days": None,
            }

        total_size = sum(item.size_bytes for item in items)
        oldest = min(items, key=lambda x: x.deleted_at)
        age_days = (datetime.now() - oldest.deleted_at).days

        return {
            "item_count": len(items),
            "total_size_bytes": total_size,
            "oldest_item_age_days": age_days,
        }

    def format_size(self, size_bytes: int) -> str:
        """Format bytes as human-readable size."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


# Public compatibility API: this singleton has no in-tree consumers, but it has
# been exported from ``features.compute`` and may be used by external feature
# packages.  Retain it until a separately-versioned deprecation can be made.
_trash_manager: Optional[TrashManager] = None


def get_trash_manager() -> TrashManager:
    """Get the global trash manager instance."""
    global _trash_manager
    if _trash_manager is None:
        _trash_manager = TrashManager()
    return _trash_manager
