"""
Kestrel Compute Feature - Destructive Operation Policy.

Rewrites rm and other destructive operations to use trash folder instead
of permanent deletion. This ensures user data is never permanently lost
by agent actions.
"""

import logging
import os
import re
import shlex
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# Default trash directory
DEFAULT_TRASH_DIR = Path(os.environ.get(
    "KESTREL_TRASH_DIR",
    os.path.expanduser("~/.kestrel/trash")
))

# Directories where true deletion is allowed (agent's temp workspace)
# Include both /tmp and /private/tmp for macOS compatibility
DEFAULT_DELETABLE_PREFIXES = [
    "/tmp/kestrel_compute_",
    "/tmp/kestrel_scratch_",
    "/private/tmp/kestrel_compute_",
    "/private/tmp/kestrel_scratch_",
]

AGENT_DATA_DIR_NAME = "agent_data"


class AgentDataProtectionError(PermissionError):
    """Raised when compute tries to delete another agent's data directory."""


def _resolve_path(path: str | Path) -> Path:
    """Resolve paths for prefix checks without requiring the target to exist."""
    return Path(path).expanduser().resolve(strict=False)


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _contains_agent_data_segment(path: Path) -> bool:
    return AGENT_DATA_DIR_NAME in path.parts


class DestructiveOperationPolicy:
    """
    Policy for handling rm and other destructive operations.
    
    Rules:
    1. `rm` is NEVER executed directly
    2. `rm` is rewritten to `mv <target> ~/.kestrel/trash/<timestamp>_<basename>`
    3. EXCEPTION: Files in agent's temp workspace can be truly deleted
    
    Example:
        policy = DestructiveOperationPolicy()
        
        # Rewrite a command
        safe_cmd = policy.rewrite_rm("rm -rf /data/old_files")
        # Returns: mkdir -p ~/.kestrel/trash/20251201_143022 && mv /data/old_files ~/.kestrel/trash/20251201_143022/
        
        # Rewrite an entire bash script
        safe_script = policy.rewrite_script(script_content, "bash", "/tmp/kestrel_compute_abc123")
    """
    
    def __init__(
        self,
        trash_dir: Optional[Path] = None,
        deletable_prefixes: Optional[List[str]] = None,
        current_agent_data_path: Optional[str | Path] = None,
    ):
        """
        Initialize the policy.
        
        Args:
            trash_dir: Path to trash directory (default: ~/.kestrel/trash)
            deletable_prefixes: Paths where true deletion is allowed
            current_agent_data_path: This agent's own data directory. Paths
                under other agent_data children are untouchable.
        """
        self.trash_dir = trash_dir or DEFAULT_TRASH_DIR
        self.deletable_prefixes = deletable_prefixes or DEFAULT_DELETABLE_PREFIXES
        self.current_agent_data_path = (
            _resolve_path(current_agent_data_path)
            if current_agent_data_path
            else None
        )
        self.agent_data_audit_log = self.trash_dir / "agent_data_access_audit.jsonl"

    def audit_agent_data_access(
        self,
        path: str | Path,
        action: str,
        decision: str,
        reason: str,
    ) -> None:
        """Append a best-effort audit row for attempts touching agent_data."""
        try:
            resolved = _resolve_path(path)
        except Exception:
            resolved = Path(str(path))

        if not _contains_agent_data_segment(resolved):
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "decision": decision,
            "reason": reason,
            "path": str(resolved),
            "current_agent_data_path": (
                str(self.current_agent_data_path)
                if self.current_agent_data_path
                else None
            ),
        }

        logger.warning(
            "agent_data access audit: action=%s decision=%s path=%s reason=%s",
            action,
            decision,
            resolved,
            reason,
        )
        try:
            self.agent_data_audit_log.parent.mkdir(parents=True, exist_ok=True)
            with self.agent_data_audit_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning("Could not write agent_data access audit log: %s", exc)

    def is_own_agent_data_path(self, path: str | Path) -> bool:
        """Return True when path is inside this agent's own data directory."""
        if self.current_agent_data_path is None:
            return False
        try:
            resolved = _resolve_path(path)
        except Exception:
            return False
        return (
            resolved == self.current_agent_data_path
            or _path_is_relative_to(resolved, self.current_agent_data_path)
        )

    def is_agent_data_path(self, path: str | Path) -> bool:
        """Return True when path is inside an agent_data tree."""
        try:
            resolved = _resolve_path(path)
        except Exception:
            resolved = Path(str(path))
        return _contains_agent_data_segment(resolved)

    def assert_agent_data_deletion_allowed(
        self,
        path: str | Path,
        action: str = "delete",
    ) -> None:
        """Block deletion/trashing of another agent's data."""
        if not self.is_agent_data_path(path):
            return
        if self.is_own_agent_data_path(path):
            self.audit_agent_data_access(path, action, "allowed", "own_agent_data")
            return

        self.audit_agent_data_access(path, action, "blocked", "other_agent_data")
        raise AgentDataProtectionError(
            f"Refusing to {action} another agent's data: {_resolve_path(path)}"
        )
    
    def is_deletable_path(self, path: str, script_workdir: Optional[str] = None) -> bool:
        """
        Check if a path is in a deletable (temp) location.
        
        Args:
            path: The path to check
            script_workdir: The script's working directory (also deletable)
            
        Returns:
            True if path can be truly deleted, False if should go to trash
        """
        # Expand user home
        if path.startswith("~"):
            path = os.path.expanduser(path)
        
        # Resolve to absolute path
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            resolved = path
        
        # Check against deletable prefixes. Also check the original (pre-resolve)
        # path so Unix-style temp paths like /tmp/kestrel_compute_* are recognised
        # on Windows, where Path.resolve() turns them into C:\tmp\... variants.
        for prefix in self.deletable_prefixes:
            if resolved.startswith(prefix) or path.startswith(prefix):
                return True
        
        # Check against script workdir
        if script_workdir and resolved.startswith(script_workdir):
            return True

        if self.is_own_agent_data_path(resolved):
            return True
        
        return False
    
    def rewrite_rm(self, command: str, script_workdir: Optional[str] = None) -> str:
        """
        Rewrite rm commands to mv to trash.
        
        Args:
            command: Original command containing rm
            script_workdir: If set, files in this dir can be truly deleted
            
        Returns:
            Rewritten command (mv to trash) or original if in temp workspace
        """
        # Parse rm command
        # Match: rm [-options] target1 [target2 ...]
        rm_match = re.match(r'^(\s*)rm\s+(-[rfivI]+\s+)?(.+)$', command)
        if not rm_match:
            return command
        
        indent = rm_match.group(1) or ""
        options = rm_match.group(2) or ""
        targets_str = rm_match.group(3).strip()
        
        # Parse targets (handle quoted paths)
        try:
            target_paths = shlex.split(targets_str)
        except ValueError:
            # If parsing fails, treat as single target
            target_paths = [targets_str]
        
        # Check if ALL targets are in deletable prefixes
        for target in target_paths:
            self.assert_agent_data_deletion_allowed(
                self._resolve_shell_target(target, script_workdir),
                "rm",
            )

        all_deletable = all(self.is_deletable_path(t, script_workdir) for t in target_paths)
        
        if all_deletable:
            # Allow true deletion for temp files
            logger.debug(f"Allowing true rm for temp paths: {target_paths}")
            return command
        
        # Rewrite to mv to trash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trash_subdir = self.trash_dir / timestamp
        
        # Build safe command
        # Quote paths properly to handle spaces
        quoted_targets = " ".join(shlex.quote(t) for t in target_paths)
        
        safe_command = (
            f"{indent}mkdir -p {shlex.quote(str(trash_subdir))} && "
            f"mv {quoted_targets} {shlex.quote(str(trash_subdir))}/"
        )
        
        logger.info(f"Rewrote rm to trash: {targets_str} -> {trash_subdir}")
        return safe_command

    def _resolve_shell_target(
        self,
        target: str,
        script_workdir: Optional[str] = None,
    ) -> str:
        if os.path.isabs(target) or not script_workdir:
            return target
        return str(Path(script_workdir) / target)

    def assert_shell_command_allowed(
        self,
        line: str,
        script_workdir: Optional[str] = None,
    ) -> None:
        """Reject shell access to another agent's data path in any pipeline segment."""
        stripped = line.strip()
        if not stripped:
            return

        try:
            lexer = shlex.shlex(stripped, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            parts = list(lexer)
        except ValueError:
            parts = stripped.split()

        for token in parts:
            if not token or set(token) <= {";", "&", "|"}:
                continue
            for target in self._shell_token_candidate_paths(token):
                self.assert_agent_data_deletion_allowed(
                    self._resolve_shell_target(target, script_workdir),
                    "shell",
                )

        for match in re.finditer(r'(?:^|[^0-9])(?:>>?|<>)\s*([^\s;&|]+)', line):
            target = match.group(1).strip()
            if target:
                self.assert_agent_data_deletion_allowed(
                    self._resolve_shell_target(target, script_workdir),
                    "redirect",
                )

    def _shell_token_candidate_paths(self, token: str) -> List[str]:
        """Extract possible path payloads from a shell token."""
        candidates = [token]

        if "=" in token:
            _, value = token.split("=", 1)
            if value:
                candidates.append(value)

        stripped = token.lstrip("0123456789<>")
        if stripped and stripped != token:
            candidates.append(stripped)

        return candidates
    
    def rewrite_bash_script(self, content: str, workdir: Optional[str] = None) -> str:
        """
        Rewrite all destructive operations in a bash script.
        
        Args:
            content: The script content
            workdir: The script's working directory
            
        Returns:
            Rewritten script with rm -> mv transformations
        """
        lines = content.split('\n')
        rewritten = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip comments and empty lines
            if stripped.startswith('#') or not stripped:
                rewritten.append(line)
                continue
            
            # Check for rm command (not in a string or comment)
            if re.match(r'^(\s*)rm\s', line):
                rewritten.append(self.rewrite_rm(line, workdir))
            else:
                self.assert_shell_command_allowed(line, workdir)
                rewritten.append(line)
        
        return '\n'.join(rewritten)
    
    def get_python_safe_remove_helper(self, workdir: Optional[str] = None) -> str:
        """
        Get Python code to inject at the top of scripts for safe deletion.
        
        This uses sys.modules manipulation to persistently patch os.remove, 
        os.unlink, shutil.rmtree and pathlib.Path.unlink to use trash folder 
        instead of permanent deletion.
        
        Args:
            workdir: The script's working directory
            
        Returns:
            Python code to prepend to scripts
        """
        # Use repr() so Windows paths with backslashes (e.g. C:\Users\...) don't
        # produce invalid string-escape sequences (\U, \n, \t) when embedded in
        # the generated Python source.
        trash_dir_literal = repr(str(self.trash_dir))
        workdir_literal = repr(workdir or "")
        prefixes_str = repr(self.deletable_prefixes)
        current_agent_data_literal = repr(
            str(self.current_agent_data_path) if self.current_agent_data_path else ""
        )

        return f'''
# === KESTREL SAFE DELETION WRAPPER ===
# This code ensures deletions go to trash instead of being permanent
# Uses sys.modules patching to make changes persistent across imports

import sys as _kestrel_sys

# Import modules directly BEFORE any user code runs
import shutil as _kestrel_shutil_original
import os as _kestrel_os_original
import builtins as _kestrel_builtins_original
import json as _kestrel_json
from pathlib import Path as _KestrelPathOriginal
from datetime import datetime as _kestrel_datetime, timezone as _kestrel_timezone

_KESTREL_TRASH_DIR = _KestrelPathOriginal({trash_dir_literal})
_KESTREL_WORKDIR = {workdir_literal}
_KESTREL_DELETABLE_PREFIXES = {prefixes_str}
_KESTREL_CURRENT_AGENT_DATA = _KestrelPathOriginal({current_agent_data_literal}).expanduser().resolve() if {current_agent_data_literal} else None
_KESTREL_AGENT_DATA_AUDIT_LOG = _KESTREL_TRASH_DIR / "agent_data_access_audit.jsonl"
_KESTREL_PATCHED = False

class _KestrelAgentDataProtectionError(PermissionError):
    pass

def _kestrel_is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def _kestrel_is_agent_data_path(path) -> bool:
    return "agent_data" in _KestrelPathOriginal(path).parts

def _kestrel_is_own_agent_data(path) -> bool:
    if _KESTREL_CURRENT_AGENT_DATA is None:
        return False
    return path == _KESTREL_CURRENT_AGENT_DATA or _kestrel_is_relative_to(path, _KESTREL_CURRENT_AGENT_DATA)

def _kestrel_audit_agent_data(path, action, decision, reason):
    if not _kestrel_is_agent_data_path(path):
        return
    entry = {{
        "timestamp": _kestrel_datetime.now(_kestrel_timezone.utc).isoformat(),
        "action": action,
        "decision": decision,
        "reason": reason,
        "path": str(path),
        "current_agent_data_path": str(_KESTREL_CURRENT_AGENT_DATA) if _KESTREL_CURRENT_AGENT_DATA else None,
    }}
    try:
        _KESTREL_AGENT_DATA_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _KESTREL_AGENT_DATA_AUDIT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(_kestrel_json.dumps(entry, sort_keys=True) + "\\n")
    except Exception:
        pass

def _kestrel_assert_agent_data_allowed(path, action):
    if not _kestrel_is_agent_data_path(path):
        return
    if _kestrel_is_own_agent_data(path):
        _kestrel_audit_agent_data(path, action, "allowed", "own_agent_data")
        return
    _kestrel_audit_agent_data(path, action, "blocked", "other_agent_data")
    raise _KestrelAgentDataProtectionError(f"Refusing to {{action}} another agent's data: {{path}}")

def _kestrel_is_deletable(path: str) -> bool:
    """Check if path can be truly deleted."""
    try:
        resolved = str(_KestrelPathOriginal(path).expanduser().resolve())
    except Exception:
        resolved = path
    resolved_path = _KestrelPathOriginal(resolved)
    
    for prefix in _KESTREL_DELETABLE_PREFIXES:
        if resolved.startswith(prefix):
            return True
    
    if _KESTREL_WORKDIR and resolved.startswith(_KESTREL_WORKDIR):
        return True

    if _kestrel_is_own_agent_data(resolved_path):
        return True
    
    return False

# Store original functions ONCE
_kestrel_original_remove = _kestrel_os_original.remove
_kestrel_original_unlink = _kestrel_os_original.unlink
_kestrel_original_rename = _kestrel_os_original.rename
_kestrel_original_replace = _kestrel_os_original.replace
_kestrel_original_truncate = _kestrel_os_original.truncate
_kestrel_original_open = _kestrel_builtins_original.open
_kestrel_original_rmtree = _kestrel_shutil_original.rmtree
_kestrel_original_path_open = _KestrelPathOriginal.open
_kestrel_original_path_unlink = _KestrelPathOriginal.unlink
_kestrel_original_path_rename = _KestrelPathOriginal.rename
_kestrel_original_path_replace = _KestrelPathOriginal.replace

def _kestrel_safe_remove(path, *args, **kwargs):
    """Move to trash instead of deleting (unless in temp workspace)."""
    p = _KestrelPathOriginal(path).expanduser().resolve()
    _kestrel_assert_agent_data_allowed(p, "delete")
    
    if _kestrel_is_deletable(str(p)):
        # Allow true deletion for temp files
        if p.is_dir():
            import os as _kestrel_os_module
            _saved_remove = _kestrel_os_module.remove
            _saved_unlink = _kestrel_os_module.unlink
            try:
                _kestrel_os_module.remove = _kestrel_original_remove
                _kestrel_os_module.unlink = _kestrel_original_unlink
                _kestrel_original_rmtree(str(p))
            finally:
                _kestrel_os_module.remove = _saved_remove
                _kestrel_os_module.unlink = _saved_unlink
        else:
            _kestrel_original_unlink(str(p))
    else:
        # Move to trash
        timestamp = _kestrel_datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trash_subdir = _KESTREL_TRASH_DIR / timestamp
        trash_subdir.mkdir(parents=True, exist_ok=True)
        _kestrel_shutil_original.move(str(p), str(trash_subdir / p.name))

def _kestrel_safe_rmtree(path, *args, **kwargs):
    """Safe rmtree that moves to trash."""
    _kestrel_safe_remove(path)

def _kestrel_safe_rename(src, dst, *args, **kwargs):
    """Block renames that move or overwrite another agent's data."""
    src_path = _KestrelPathOriginal(src).expanduser().resolve()
    dst_path = _KestrelPathOriginal(dst).expanduser().resolve()
    _kestrel_assert_agent_data_allowed(src_path, "rename")
    _kestrel_assert_agent_data_allowed(dst_path, "rename")
    return _kestrel_original_rename(src, dst, *args, **kwargs)

def _kestrel_safe_replace(src, dst, *args, **kwargs):
    """Block replaces that move or overwrite another agent's data."""
    src_path = _KestrelPathOriginal(src).expanduser().resolve()
    dst_path = _KestrelPathOriginal(dst).expanduser().resolve()
    _kestrel_assert_agent_data_allowed(src_path, "replace")
    _kestrel_assert_agent_data_allowed(dst_path, "replace")
    return _kestrel_original_replace(src, dst, *args, **kwargs)

def _kestrel_safe_truncate(path, length, *args, **kwargs):
    """Block truncate calls against another agent's data."""
    try:
        p = _KestrelPathOriginal(path).expanduser().resolve()
    except TypeError:
        return _kestrel_original_truncate(path, length, *args, **kwargs)
    _kestrel_assert_agent_data_allowed(p, "truncate")
    return _kestrel_original_truncate(path, length, *args, **kwargs)

def _kestrel_safe_open(file, mode="r", *args, **kwargs):
    """Block builtins.open calls that would truncate another agent's data."""
    if isinstance(mode, str) and "w" in mode:
        try:
            p = _KestrelPathOriginal(file).expanduser().resolve()
        except TypeError:
            return _kestrel_original_open(file, mode, *args, **kwargs)
        _kestrel_assert_agent_data_allowed(p, "open_truncate")
    return _kestrel_original_open(file, mode, *args, **kwargs)

def _kestrel_path_safe_open(self, mode="r", *args, **kwargs):
    """Block Path.open calls that would truncate another agent's data."""
    if isinstance(mode, str) and "w" in mode:
        p = _KestrelPathOriginal(self).expanduser().resolve()
        _kestrel_assert_agent_data_allowed(p, "open_truncate")
    return _kestrel_original_path_open(self, mode, *args, **kwargs)

def _kestrel_path_safe_unlink(self, missing_ok=False):
    """Safe Path.unlink that moves to trash."""
    try:
        _kestrel_safe_remove(str(self))
    except FileNotFoundError:
        if not missing_ok:
            raise

def _kestrel_path_safe_rename(self, target):
    """Safe Path.rename that blocks another agent's data."""
    source = _KestrelPathOriginal(self).expanduser().resolve()
    dest = _KestrelPathOriginal(target).expanduser().resolve()
    _kestrel_assert_agent_data_allowed(source, "rename")
    _kestrel_assert_agent_data_allowed(dest, "rename")
    return _kestrel_original_path_rename(self, target)

def _kestrel_path_safe_replace(self, target):
    """Safe Path.replace that blocks another agent's data."""
    source = _KestrelPathOriginal(self).expanduser().resolve()
    dest = _KestrelPathOriginal(target).expanduser().resolve()
    _kestrel_assert_agent_data_allowed(source, "replace")
    _kestrel_assert_agent_data_allowed(dest, "replace")
    return _kestrel_original_path_replace(self, target)

def _kestrel_apply_patches():
    """Apply patches to os, shutil, and pathlib modules in sys.modules."""
    global _KESTREL_PATCHED
    if _KESTREL_PATCHED:
        return
    
    # Patch os module
    import os
    os.remove = _kestrel_safe_remove
    os.unlink = _kestrel_safe_remove
    os.rename = _kestrel_safe_rename
    os.replace = _kestrel_safe_replace
    os.truncate = _kestrel_safe_truncate

    import builtins
    builtins.open = _kestrel_safe_open
    
    # Patch shutil module  
    import shutil
    shutil.rmtree = _kestrel_safe_rmtree
    
    # Patch pathlib.Path class
    import pathlib
    pathlib.Path.open = _kestrel_path_safe_open
    pathlib.Path.unlink = _kestrel_path_safe_unlink
    pathlib.Path.rename = _kestrel_path_safe_rename
    pathlib.Path.replace = _kestrel_path_safe_replace
    # Also patch PurePath descendants
    if hasattr(pathlib, 'PosixPath'):
        pathlib.PosixPath.open = _kestrel_path_safe_open
        pathlib.PosixPath.unlink = _kestrel_path_safe_unlink
        pathlib.PosixPath.rename = _kestrel_path_safe_rename
        pathlib.PosixPath.replace = _kestrel_path_safe_replace
    if hasattr(pathlib, 'WindowsPath'):
        pathlib.WindowsPath.open = _kestrel_path_safe_open
        pathlib.WindowsPath.unlink = _kestrel_path_safe_unlink
        pathlib.WindowsPath.rename = _kestrel_path_safe_rename
        pathlib.WindowsPath.replace = _kestrel_path_safe_replace
    
    _KESTREL_PATCHED = True

# Apply patches immediately
_kestrel_apply_patches()

# === END KESTREL SAFE DELETION WRAPPER ===

'''
    
    def rewrite_python_script(self, content: str, workdir: Optional[str] = None) -> str:
        """
        Rewrite a Python script to use safe deletion.
        
        Prepends the safe_remove helper to the script.
        
        Args:
            content: The script content
            workdir: The script's working directory
            
        Returns:
            Rewritten script with safe deletion wrapper
        """
        helper = self.get_python_safe_remove_helper(workdir)
        
        # Handle shebang - keep it at the top
        if content.startswith('#!'):
            lines = content.split('\n', 1)
            shebang = lines[0]
            rest = lines[1] if len(lines) > 1 else ""
            return f"{shebang}\n{helper}\n{rest}"
        
        return f"{helper}\n{content}"
    
    def rewrite_script(
        self,
        content: str,
        language: str,
        workdir: Optional[str] = None,
    ) -> str:
        """
        Rewrite a script to use safe deletion based on language.
        
        Args:
            content: The script content
            language: "bash" or "python"
            workdir: The script's working directory
            
        Returns:
            Rewritten script
        """
        if language == "bash":
            return self.rewrite_bash_script(content, workdir)
        elif language == "python":
            return self.rewrite_python_script(content, workdir)
        else:
            logger.warning(f"Unknown language '{language}', not rewriting")
            return content


def rewrite_script_for_safety(
    content: str,
    language: str,
    workdir: Optional[str] = None,
) -> str:
    """
    Convenience function to rewrite a script for safe deletion.
    
    Args:
        content: The script content
        language: "bash" or "python"
        workdir: The script's working directory
        
    Returns:
        Rewritten script
    """
    policy = DestructiveOperationPolicy()
    return policy.rewrite_script(content, language, workdir)
