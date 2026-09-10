"""Standalone Python safe-delete runtime injected into compute scripts.

This module deliberately has no Kestrel imports.  Its exact source is executed
in an isolated namespace ahead of a user script, which lets Docker/UV scripts
use the same audited implementation without requiring the Kestrel package in
their execution environment.
"""

import builtins as _builtins
from contextvars import ContextVar as _ContextVar
from datetime import datetime as _datetime, timezone as _timezone
import io as _io
import json as _json
import os as _os
from pathlib import Path as _Path
import shutil as _shutil
import stat as _stat
import tempfile as _tempfile
import unicodedata as _unicodedata


class _KestrelAgentDataProtectionError(PermissionError):
    pass


def _is_relative_to(path: _Path, parent: _Path) -> bool:
    """Use component-aware containment; string prefixes are never boundaries."""
    return path == parent or path.is_relative_to(parent)


def _nearest_existing_path(path: _Path) -> _Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _alternate_case(name: str) -> str | None:
    for position, character in enumerate(name):
        swapped = character.swapcase()
        if swapped != character:
            return name[:position] + swapped + name[position + 1 :]
    return None


def _filesystem_is_case_insensitive(path: _Path) -> bool | None:
    if _os.name == "nt":
        return True
    existing = _nearest_existing_path(path)
    if existing.is_dir():
        try:
            candidates = existing.iterdir()
        except OSError:
            return None
    else:
        candidates = iter((existing,))
    try:
        for candidate in candidates:
            alternate_name = _alternate_case(candidate.name)
            if alternate_name is None:
                continue
            alternate = candidate.with_name(alternate_name)
            try:
                return candidate.samefile(alternate)
            except FileNotFoundError:
                return False
            except OSError:
                continue
    except OSError:
        return None
    return None


def _combined_case_insensitivity(*results: bool | None) -> bool | None:
    if any(result is True for result in results):
        return True
    if all(result is False for result in results):
        return False
    return None


def _normalized_parts(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_unicodedata.normalize("NFD", part) for part in parts)


def _parts_overlap(
    first: tuple[str, ...],
    second: tuple[str, ...],
    *,
    case_insensitive: bool | None,
) -> bool:
    first_normalized = _normalized_parts(first)
    second_normalized = _normalized_parts(second)
    shorter = min(len(first_normalized), len(second_normalized))
    if first_normalized[:shorter] == second_normalized[:shorter]:
        return True
    if case_insensitive is False:
        return False
    return tuple(part.casefold() for part in first_normalized[:shorter]) == tuple(
        part.casefold() for part in second_normalized[:shorter]
    )


def _same_existing_path(first: _Path, second: _Path) -> bool:
    try:
        return first.samefile(second)
    except OSError:
        return False


def _aliased_ancestor_suffixes(
    first: _Path,
    second: _Path,
) -> tuple[tuple[str, ...], tuple[str, ...], _Path, _Path] | None:
    first_ancestor = _nearest_existing_path(first)
    second_ancestor = _nearest_existing_path(second)
    if not _same_existing_path(first_ancestor, second_ancestor):
        return None
    return (
        first.relative_to(first_ancestor).parts,
        second.relative_to(second_ancestor).parts,
        first_ancestor,
        second_ancestor,
    )


def _paths_overlap_by_filesystem_identity(first: _Path, second: _Path) -> bool:
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    if _is_relative_to(first, second) or _is_relative_to(second, first):
        return True
    if first.exists() and second.exists() and _same_existing_path(first, second):
        return True
    aliased = _aliased_ancestor_suffixes(first, second)
    if aliased is not None:
        first_suffix, second_suffix, first_ancestor, second_ancestor = aliased
        if _parts_overlap(
            first_suffix,
            second_suffix,
            case_insensitive=_combined_case_insensitivity(
                _filesystem_is_case_insensitive(first_ancestor),
                _filesystem_is_case_insensitive(second_ancestor),
            ),
        ):
            return True
    return _parts_overlap(
        first.parts,
        second.parts,
        case_insensitive=_combined_case_insensitivity(
            _filesystem_is_case_insensitive(first),
            _filesystem_is_case_insensitive(second),
        ),
    )


def _is_agent_data_path(path: _Path) -> bool:
    return "agent_data" in path.parts


def _unique_trash_subdir(trash_root: _Path) -> _Path:
    """Create an OS-allocated exclusive directory for one moved item."""
    trash_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    prefix = _datetime.now().strftime("%Y%m%d_%H%M%S_%f_")
    return _Path(_tempfile.mkdtemp(prefix=prefix, dir=trash_root))


def install_safe_delete_runtime(
    trash_dir: str,
    current_agent_data_path: str | None,
    host_control_data_path: str,
    deletable_prefixes: list[str],
    workdir: str | None,
) -> None:
    """Patch common Python deletion APIs so removal always goes to trash.

    Direct deletion remains available in executor-owned temporary roots and
    this agent's own data.  The source is resolved again in the child at the
    operation boundary; a path whose ownership changed falls back to trash or
    is rejected by the cross-agent policy.
    """
    trash_root = _Path(trash_dir).expanduser()
    current_agent_data = (
        _Path(current_agent_data_path).expanduser().resolve(strict=False)
        if current_agent_data_path
        else None
    )
    host_control_data = (
        _Path(host_control_data_path).expanduser().resolve(strict=False)
    )
    authorized_workdir = _Path(workdir) if workdir else None
    configured_prefixes = tuple(
        _Path(prefix).expanduser() for prefix in deletable_prefixes
    )
    audit_log = trash_root / "agent_data_access_audit.jsonl"

    original_unlink = _os.unlink
    original_link = _os.link
    original_rename = _os.rename
    original_replace = _os.replace
    original_truncate = _os.truncate
    original_os_open = _os.open
    original_open = _builtins.open
    original_io_open = _io.open
    original_rmtree = _shutil.rmtree
    original_path_open = _Path.open
    internal_filesystem_operation = _ContextVar(
        "_kestrel_internal_filesystem_operation",
        default=False,
    )

    def audit_agent_data(path: _Path, action: str, decision: str, reason: str) -> None:
        if not _is_agent_data_path(path):
            return
        entry = {
            "timestamp": _datetime.now(_timezone.utc).isoformat(),
            "action": action,
            "decision": decision,
            "reason": reason,
            "path": str(path),
            "current_agent_data_path": (
                str(current_agent_data) if current_agent_data else None
            ),
        }
        try:
            audit_log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with original_open(audit_log, "a", encoding="utf-8") as handle:
                handle.write(_json.dumps(entry, sort_keys=True) + "\n")
        except OSError:
            # Audit persistence is best-effort inside constrained execution
            # environments; the access decision itself remains fail-closed.
            pass

    def assert_agent_data_allowed(path: _Path, action: str) -> None:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            metadata = None
        except OSError as exc:
            raise _KestrelAgentDataProtectionError(
                f"Refusing to {action} path whose hard-link custody cannot be "
                f"verified: {path}"
            ) from exc
        if (
            metadata is not None
            and _stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink != 1
        ):
            audit_agent_data(
                path,
                action,
                "blocked",
                "ambiguous_hard_link_custody",
            )
            raise _KestrelAgentDataProtectionError(
                f"Refusing to {action} multiply-linked file with ambiguous "
                f"Hold custody: {path}"
            )
        if _paths_overlap_by_filesystem_identity(path, host_control_data):
            audit_agent_data(path, action, "blocked", "host_hold_custody")
            raise _KestrelAgentDataProtectionError(
                f"Refusing to {action} host Hold custody: {path}"
            )
        if not _is_agent_data_path(path):
            return
        if current_agent_data is not None and _is_relative_to(path, current_agent_data):
            audit_agent_data(path, action, "allowed", "own_agent_data")
            return
        audit_agent_data(path, action, "blocked", "other_agent_data")
        raise _KestrelAgentDataProtectionError(
            f"Refusing to {action} another agent's data: {path}"
        )

    def direct_delete_root(path: _Path) -> _Path | None:
        """Return the concrete root that owns ``path`` at operation time."""
        if _paths_overlap_by_filesystem_identity(path, host_control_data):
            return None
        for configured_prefix in configured_prefixes:
            prefix_parent = configured_prefix.parent.resolve(strict=False)
            try:
                relative = path.relative_to(prefix_parent)
            except ValueError:
                continue
            if relative.parts and relative.parts[0].startswith(configured_prefix.name):
                return prefix_parent / relative.parts[0]

        if authorized_workdir is not None and _is_relative_to(
            path,
            authorized_workdir,
        ):
            return authorized_workdir
        if current_agent_data is not None and _is_relative_to(
            path,
            current_agent_data,
        ):
            return current_agent_data
        return None

    def source_paths(path) -> tuple[_Path, _Path, bool]:
        lexical = _Path(path).expanduser()
        if not lexical.exists() and not lexical.is_symlink():
            raise FileNotFoundError(str(lexical))
        is_symlink = lexical.is_symlink()
        resolved = lexical.resolve(strict=False)
        assert_agent_data_allowed(resolved, "delete")
        if is_symlink:
            # Deleting a symlink moves the entry itself, so its physical
            # location must be authorized too, not only the followed target:
            # a link parked inside another agent's data would otherwise
            # escape that directory through its own removal.
            entry = lexical.parent.resolve(strict=False) / lexical.name
            assert_agent_data_allowed(entry, "delete")
        return lexical, resolved, is_symlink

    def move_to_trash(lexical: _Path, resolved: _Path, is_symlink: bool) -> None:
        trash_subdir = _unique_trash_subdir(trash_root)
        source = lexical if is_symlink else resolved
        token = internal_filesystem_operation.set(True)
        try:
            _shutil.move(str(source), str(trash_subdir / lexical.name))
        except BaseException:
            try:
                trash_subdir.rmdir()
            except OSError:
                pass
            raise
        finally:
            internal_filesystem_operation.reset(token)

    def safe_remove(path, *args, **kwargs):
        """Delete an owned file directly; otherwise move it to trash."""
        if internal_filesystem_operation.get():
            return original_unlink(path, *args, **kwargs)
        if args or kwargs:
            raise ValueError(
                "Safe deletion does not support dir_fd or extra os.remove arguments"
            )
        lexical, resolved, is_symlink = source_paths(path)
        if not is_symlink and direct_delete_root(resolved) is not None:
            return original_unlink(resolved)
        move_to_trash(lexical, resolved, is_symlink)

    def safe_rmtree(path, *args, **kwargs):
        if internal_filesystem_operation.get():
            return original_rmtree(path, *args, **kwargs)
        if args or kwargs:
            raise ValueError(
                "Safe deletion does not support shutil.rmtree callbacks/options"
            )
        lexical, resolved, is_symlink = source_paths(path)
        if is_symlink:
            raise OSError("Cannot call rmtree on a symbolic link")
        if direct_delete_root(resolved) is not None:
            token = internal_filesystem_operation.set(True)
            try:
                return original_rmtree(resolved)
            finally:
                internal_filesystem_operation.reset(token)
        move_to_trash(lexical, resolved, is_symlink=False)

    def safe_rename(src, dst, *args, **kwargs):
        if internal_filesystem_operation.get():
            return original_rename(src, dst, *args, **kwargs)
        if args or kwargs:
            raise ValueError(
                "Safe rename does not support dir_fd or extra os.rename arguments"
            )
        # Resolved paths authorize the operation only; the original operands
        # are passed through so symlink operands keep os.rename semantics
        # (rename the link itself, replace a link destination).
        assert_agent_data_allowed(_Path(src).resolve(strict=False), "rename")
        assert_agent_data_allowed(_Path(dst).resolve(strict=False), "rename")
        return original_rename(src, dst)

    def safe_replace(src, dst, *args, **kwargs):
        if internal_filesystem_operation.get():
            return original_replace(src, dst, *args, **kwargs)
        if args or kwargs:
            raise ValueError(
                "Safe replace does not support dir_fd or extra os.replace arguments"
            )
        # Authorize on resolved paths; operate on the original operands so
        # symlink semantics match os.replace.
        assert_agent_data_allowed(_Path(src).resolve(strict=False), "replace")
        assert_agent_data_allowed(_Path(dst).resolve(strict=False), "replace")
        return original_replace(src, dst)

    def safe_link(src, dst, *args, **kwargs):
        if internal_filesystem_operation.get():
            return original_link(src, dst, *args, **kwargs)
        if args or kwargs:
            raise ValueError(
                "Safe hard-link creation does not support dir_fd/options"
            )
        assert_agent_data_allowed(_Path(src).resolve(strict=False), "hard_link")
        assert_agent_data_allowed(_Path(dst).resolve(strict=False), "hard_link")
        return original_link(src, dst)

    def safe_truncate(path, length, *args, **kwargs):
        try:
            resolved = _Path(path).expanduser().resolve(strict=False)
        except TypeError:
            return original_truncate(path, length, *args, **kwargs)
        assert_agent_data_allowed(resolved, "truncate")
        return original_truncate(resolved, length, *args, **kwargs)

    def safe_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
            try:
                resolved = _Path(file).expanduser().resolve(strict=False)
            except TypeError:
                return original_open(file, mode, *args, **kwargs)
            assert_agent_data_allowed(resolved, "open_write")
            return original_open(resolved, mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)

    def safe_io_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
            try:
                resolved = _Path(file).expanduser().resolve(strict=False)
            except TypeError:
                return original_io_open(file, mode, *args, **kwargs)
            assert_agent_data_allowed(resolved, "open_write")
            return original_io_open(resolved, mode, *args, **kwargs)
        return original_io_open(file, mode, *args, **kwargs)

    def safe_os_open(file, flags, mode=0o777, *, dir_fd=None):
        mutation_flags = (
            _os.O_WRONLY
            | _os.O_RDWR
            | _os.O_APPEND
            | _os.O_CREAT
            | _os.O_TRUNC
            # O_TMPFILE is DEFINED as __O_TMPFILE|O_DIRECTORY on Linux, so
            # masking with it whole matches a plain read-only directory open
            # (O_RDONLY|O_DIRECTORY) and routes a READ through the write
            # guard -- and, with dir_fd set, fails every anchored relative
            # open outright. Keep only its write-intent bit; a real O_TMPFILE
            # caller also passes O_WRONLY/O_RDWR, which this mask covers.
            | (getattr(_os, "O_TMPFILE", 0) & ~getattr(_os, "O_DIRECTORY", 0))
        )
        if isinstance(flags, int) and flags & mutation_flags:
            try:
                lexical = _Path(file).expanduser()
            except TypeError:
                return original_os_open(file, flags, mode, dir_fd=dir_fd)
            if dir_fd is not None and not lexical.is_absolute():
                raise ValueError(
                    "Safe write open does not support relative paths with dir_fd"
                )
            resolved = lexical.resolve(strict=False)
            assert_agent_data_allowed(resolved, "os_open_write")
            return original_os_open(resolved, flags, mode, dir_fd=dir_fd)
        return original_os_open(file, flags, mode, dir_fd=dir_fd)

    def path_safe_open(self, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
            resolved = _Path(self).expanduser().resolve(strict=False)
            assert_agent_data_allowed(resolved, "open_write")
            return original_path_open(resolved, mode, *args, **kwargs)
        return original_path_open(self, mode, *args, **kwargs)

    def path_safe_unlink(self, missing_ok=False):
        try:
            safe_remove(str(self))
        except FileNotFoundError:
            if not missing_ok:
                raise

    def path_safe_rename(self, target):
        assert_agent_data_allowed(_Path(self).resolve(strict=False), "rename")
        assert_agent_data_allowed(_Path(target).resolve(strict=False), "rename")
        original_rename(self, target)
        return _Path(target)

    def path_safe_replace(self, target):
        assert_agent_data_allowed(_Path(self).resolve(strict=False), "replace")
        assert_agent_data_allowed(_Path(target).resolve(strict=False), "replace")
        original_replace(self, target)
        return _Path(target)

    _os.remove = safe_remove
    _os.unlink = safe_remove
    _os.link = safe_link
    _os.rename = safe_rename
    _os.replace = safe_replace
    _os.truncate = safe_truncate
    _os.open = safe_os_open
    _builtins.open = safe_open
    _io.open = safe_io_open
    _shutil.rmtree = safe_rmtree

    # Path is an alias of the platform's concrete Path class.  Patching that
    # class is sufficient; assigning both Path and PosixPath/WindowsPath would
    # duplicate the same mutation on supported Python versions.
    _Path.open = path_safe_open
    _Path.unlink = path_safe_unlink
    _Path.rename = path_safe_rename
    _Path.replace = path_safe_replace
