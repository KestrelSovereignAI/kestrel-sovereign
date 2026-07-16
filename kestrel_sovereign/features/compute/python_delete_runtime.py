"""Standalone Python safe-delete runtime injected into compute scripts.

This module deliberately has no Kestrel imports.  Its exact source is executed
in an isolated namespace ahead of a user script, which lets Docker/UV scripts
use the same audited implementation without requiring the Kestrel package in
their execution environment.
"""

import builtins as _builtins
from contextvars import ContextVar as _ContextVar
from datetime import datetime as _datetime, timezone as _timezone
import json as _json
import os as _os
from pathlib import Path as _Path
import shutil as _shutil
import tempfile as _tempfile


class _KestrelAgentDataProtectionError(PermissionError):
    pass


def _is_relative_to(path: _Path, parent: _Path) -> bool:
    """Use component-aware containment; string prefixes are never boundaries."""
    return path == parent or path.is_relative_to(parent)


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
    authorized_workdir = _Path(workdir) if workdir else None
    configured_prefixes = tuple(
        _Path(prefix).expanduser() for prefix in deletable_prefixes
    )
    audit_log = trash_root / "agent_data_access_audit.jsonl"

    original_unlink = _os.unlink
    original_rename = _os.rename
    original_replace = _os.replace
    original_truncate = _os.truncate
    original_open = _builtins.open
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

    def safe_truncate(path, length, *args, **kwargs):
        try:
            resolved = _Path(path).expanduser().resolve(strict=False)
        except TypeError:
            return original_truncate(path, length, *args, **kwargs)
        assert_agent_data_allowed(resolved, "truncate")
        return original_truncate(resolved, length, *args, **kwargs)

    def safe_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "w" in mode:
            try:
                resolved = _Path(file).expanduser().resolve(strict=False)
            except TypeError:
                return original_open(file, mode, *args, **kwargs)
            assert_agent_data_allowed(resolved, "open_truncate")
            return original_open(resolved, mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)

    def path_safe_open(self, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "w" in mode:
            resolved = _Path(self).expanduser().resolve(strict=False)
            assert_agent_data_allowed(resolved, "open_truncate")
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
    _os.rename = safe_rename
    _os.replace = safe_replace
    _os.truncate = safe_truncate
    _builtins.open = safe_open
    _shutil.rmtree = safe_rmtree

    # Path is an alias of the platform's concrete Path class.  Patching that
    # class is sufficient; assigning both Path and PosixPath/WindowsPath would
    # duplicate the same mutation on supported Python versions.
    _Path.open = path_safe_open
    _Path.unlink = path_safe_unlink
    _Path.rename = path_safe_rename
    _Path.replace = path_safe_replace
