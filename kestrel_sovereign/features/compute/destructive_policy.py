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
from datetime import datetime
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
    ):
        """
        Initialize the policy.
        
        Args:
            trash_dir: Path to trash directory (default: ~/.kestrel/trash)
            deletable_prefixes: Paths where true deletion is allowed
        """
        self.trash_dir = trash_dir or DEFAULT_TRASH_DIR
        self.deletable_prefixes = deletable_prefixes or DEFAULT_DELETABLE_PREFIXES
    
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
        
        # Check against deletable prefixes
        for prefix in self.deletable_prefixes:
            if resolved.startswith(prefix):
                return True
        
        # Check against script workdir
        if script_workdir and resolved.startswith(script_workdir):
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
        all_deletable = all(
            self.is_deletable_path(t, script_workdir)
            for t in target_paths
        )
        
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
        trash_dir_str = str(self.trash_dir)
        workdir_str = workdir or ""
        prefixes_str = repr(self.deletable_prefixes)
        
        return f'''
# === KESTREL SAFE DELETION WRAPPER ===
# This code ensures deletions go to trash instead of being permanent
# Uses sys.modules patching to make changes persistent across imports

import sys as _kestrel_sys

# Import modules directly BEFORE any user code runs
import shutil as _kestrel_shutil_original
import os as _kestrel_os_original
from pathlib import Path as _KestrelPathOriginal
from datetime import datetime as _kestrel_datetime

_KESTREL_TRASH_DIR = _KestrelPathOriginal("{trash_dir_str}")
_KESTREL_WORKDIR = "{workdir_str}"
_KESTREL_DELETABLE_PREFIXES = {prefixes_str}
_KESTREL_PATCHED = False

def _kestrel_is_deletable(path: str) -> bool:
    """Check if path can be truly deleted."""
    try:
        resolved = str(_KestrelPathOriginal(path).expanduser().resolve())
    except Exception:
        resolved = path
    
    for prefix in _KESTREL_DELETABLE_PREFIXES:
        if resolved.startswith(prefix):
            return True
    
    if _KESTREL_WORKDIR and resolved.startswith(_KESTREL_WORKDIR):
        return True
    
    return False

# Store original functions ONCE
_kestrel_original_remove = _kestrel_os_original.remove
_kestrel_original_unlink = _kestrel_os_original.unlink
_kestrel_original_rmtree = _kestrel_shutil_original.rmtree
_kestrel_original_path_unlink = _KestrelPathOriginal.unlink

def _kestrel_safe_remove(path, *args, **kwargs):
    """Move to trash instead of deleting (unless in temp workspace)."""
    p = _KestrelPathOriginal(path).expanduser().resolve()
    
    if _kestrel_is_deletable(str(p)):
        # Allow true deletion for temp files
        if p.is_dir():
            _kestrel_original_rmtree(str(p))
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

def _kestrel_path_safe_unlink(self, missing_ok=False):
    """Safe Path.unlink that moves to trash."""
    try:
        _kestrel_safe_remove(str(self))
    except FileNotFoundError:
        if not missing_ok:
            raise

def _kestrel_apply_patches():
    """Apply patches to os, shutil, and pathlib modules in sys.modules."""
    global _KESTREL_PATCHED
    if _KESTREL_PATCHED:
        return
    
    # Patch os module
    import os
    os.remove = _kestrel_safe_remove
    os.unlink = _kestrel_safe_remove
    
    # Patch shutil module  
    import shutil
    shutil.rmtree = _kestrel_safe_rmtree
    
    # Patch pathlib.Path class
    import pathlib
    pathlib.Path.unlink = _kestrel_path_safe_unlink
    # Also patch PurePath descendants
    if hasattr(pathlib, 'PosixPath'):
        pathlib.PosixPath.unlink = _kestrel_path_safe_unlink
    if hasattr(pathlib, 'WindowsPath'):
        pathlib.WindowsPath.unlink = _kestrel_path_safe_unlink
    
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
