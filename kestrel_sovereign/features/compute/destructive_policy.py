"""Canonical destructive-operation policy for the compute feature.

Shell syntax-tree handling lives in :mod:`shell_rewriter`; the standalone
Python child-process runtime lives in :mod:`python_delete_runtime`.  This
module owns the shared configuration, agent-data policy, and public rewrite
surface only.
"""

import ast
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shlex
from typing import Optional

from . import python_delete_runtime
from .shell_rewriter import ShellRewriteError, ShellScriptRewriter


logger = logging.getLogger(__name__)


DEFAULT_TRASH_DIR = Path(
    os.environ.get("KESTREL_TRASH_DIR", os.path.expanduser("~/.kestrel/trash"))
)

# Each value denotes a parent plus a generated-directory name prefix, not an
# arbitrary string prefix.  ``is_deletable_path`` resolves both sides and uses
# path components so symlinks and prefix siblings cannot escape containment.
DEFAULT_DELETABLE_PREFIXES = [
    "/tmp/kestrel_compute_",
    "/tmp/kestrel_scratch_",
    "/private/tmp/kestrel_compute_",
    "/private/tmp/kestrel_scratch_",
]

AGENT_DATA_DIR_NAME = "agent_data"
_PYTHON_ENCODING_COOKIE = re.compile(r"coding[:=]\s*[-\w.]+")


class AgentDataProtectionError(PermissionError):
    """Raised when compute tries to mutate another agent's data directory."""


def _resolve_path(path: str | Path) -> Path:
    """Resolve paths for component checks without requiring the target."""
    return Path(path).expanduser().resolve(strict=False)


def _absolute_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _contains_agent_data_segment(path: Path) -> bool:
    return AGENT_DATA_DIR_NAME in path.parts


class DestructiveOperationPolicy:
    """Govern shell/Python deletion and cross-agent filesystem access.

    ``is_deletable_path`` remains a public classification helper for callers
    that need to reason about executor-owned workspaces.  Rewriters preserve
    the established direct-delete authorization for those workspaces and the
    current agent's own data, but revalidate the owning root in the child
    immediately before the filesystem operation.
    """

    def __init__(
        self,
        trash_dir: Optional[Path] = None,
        deletable_prefixes: Optional[list[str]] = None,
        current_agent_data_path: Optional[str | Path] = None,
    ) -> None:
        self.trash_dir = trash_dir or DEFAULT_TRASH_DIR
        self.deletable_prefixes = (
            list(deletable_prefixes)
            if deletable_prefixes is not None
            else list(DEFAULT_DELETABLE_PREFIXES)
        )
        self.current_agent_data_path = (
            _resolve_path(current_agent_data_path) if current_agent_data_path else None
        )
        self.agent_data_audit_log = self.trash_dir / "agent_data_access_audit.jsonl"

    def audit_agent_data_access(
        self,
        path: str | Path,
        action: str,
        decision: str,
        reason: str,
    ) -> None:
        """Append a best-effort audit row for attempts touching agent data."""
        try:
            resolved = _resolve_path(path)
        except (OSError, RuntimeError, ValueError):
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
        if self.current_agent_data_path is None:
            return False
        try:
            resolved = _resolve_path(path)
        except (OSError, RuntimeError, ValueError):
            return False
        return resolved == self.current_agent_data_path or resolved.is_relative_to(
            self.current_agent_data_path
        )

    def is_agent_data_path(self, path: str | Path) -> bool:
        try:
            resolved = _resolve_path(path)
        except (OSError, RuntimeError, ValueError):
            resolved = Path(str(path))
        return _contains_agent_data_segment(resolved)

    def assert_agent_data_deletion_allowed(
        self,
        path: str | Path,
        action: str = "delete",
    ) -> None:
        if not self.is_agent_data_path(path):
            return
        if self.is_own_agent_data_path(path):
            self.audit_agent_data_access(path, action, "allowed", "own_agent_data")
            return
        self.audit_agent_data_access(path, action, "blocked", "other_agent_data")
        raise AgentDataProtectionError(
            f"Refusing to {action} another agent's data: {_resolve_path(path)}"
        )

    def is_deletable_path(
        self,
        path: str,
        script_workdir: Optional[str] = None,
    ) -> bool:
        """Return whether ``path`` resolves inside an executor-owned workspace."""
        return self._direct_delete_root(path, script_workdir) is not None

    def _direct_delete_root(
        self,
        path: str | Path,
        script_workdir: Optional[str] = None,
    ) -> Optional[Path]:
        """Return the exact root authorizing direct deletion, if any."""
        try:
            resolved_path = _resolve_path(path)
        except (OSError, RuntimeError, ValueError):
            return None

        for configured_prefix in self.deletable_prefixes:
            prefix_path = Path(configured_prefix).expanduser()
            prefix_parent = _resolve_path(prefix_path.parent)
            try:
                relative = resolved_path.relative_to(prefix_parent)
            except ValueError:
                continue
            if relative.parts and relative.parts[0].startswith(prefix_path.name):
                return prefix_parent / relative.parts[0]

        if script_workdir:
            workdir_path = _resolve_path(script_workdir)
            if resolved_path == workdir_path or resolved_path.is_relative_to(
                workdir_path
            ):
                return workdir_path

        if self.is_own_agent_data_path(resolved_path):
            return self.current_agent_data_path
        return None

    def rewrite_rm(
        self,
        command: str,
        script_workdir: Optional[str] = None,
    ) -> str:
        """Rewrite a shell fragment through the canonical syntax-tree path."""
        return self.rewrite_bash_script(command, script_workdir)

    def _resolve_shell_target(
        self,
        target: str,
        script_workdir: Optional[str],
    ) -> str:
        if os.path.isabs(target) or not script_workdir:
            return target
        return str(Path(script_workdir) / target)

    def assert_shell_command_allowed(
        self,
        command: str,
        script_workdir: Optional[str] = None,
    ) -> None:
        """Reject static shell paths that target another agent's data."""
        stripped = command.strip()
        if not stripped:
            return
        try:
            lexer = shlex.shlex(stripped, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            parts = list(lexer)
        except ValueError as exc:
            raise ShellRewriteError(
                f"Could not classify shell command paths: {exc}"
            ) from exc

        for token in parts:
            if not token or set(token) <= {";", "&", "|"}:
                continue
            for target in self._shell_token_candidate_paths(token):
                self.assert_agent_data_deletion_allowed(
                    self._resolve_shell_target(target, script_workdir),
                    "shell",
                )

        for match in re.finditer(r"(?:^|[^0-9])(?:>>?|<>)\s*([^\s;&|]+)", command):
            target = match.group(1).strip()
            if target:
                self.assert_agent_data_deletion_allowed(
                    self._resolve_shell_target(target, script_workdir),
                    "redirect",
                )

    @staticmethod
    def _shell_token_candidate_paths(token: str) -> list[str]:
        candidates = [token]
        if "=" in token:
            _, value = token.split("=", 1)
            if value:
                candidates.append(value)
        stripped = token.lstrip("0123456789<>")
        if stripped and stripped != token:
            candidates.append(stripped)
        return candidates

    def rewrite_bash_script(
        self,
        content: str,
        workdir: Optional[str] = None,
        *,
        runtime_trash_dir: Optional[str | Path] = None,
    ) -> str:
        """Rewrite shell deletion using a concrete Bash syntax tree."""
        trash_dir = _absolute_path(runtime_trash_dir or self.trash_dir)
        rewriter = ShellScriptRewriter(
            trash_dir=trash_dir,
            workdir=workdir,
            assert_delete_allowed=lambda target: (
                self.assert_agent_data_deletion_allowed(
                    self._resolve_shell_target(target, workdir),
                    "rm",
                )
            ),
            assert_command_allowed=lambda command: self.assert_shell_command_allowed(
                command,
                workdir,
            ),
            direct_delete_root=lambda target: (
                str(root)
                if (
                    root := self._direct_delete_root(
                        self._resolve_shell_target(target, workdir),
                        workdir,
                    )
                )
                is not None
                else None
            ),
            current_agent_data_root=self.current_agent_data_path,
        )
        return rewriter.rewrite(content)

    def get_python_safe_remove_helper(
        self,
        workdir: Optional[str] = None,
        *,
        runtime_trash_dir: Optional[str | Path] = None,
    ) -> str:
        """Build a bootstrap from the standalone canonical Python runtime."""
        runtime_source = Path(python_delete_runtime.__file__).read_text(
            encoding="utf-8"
        )
        trash_dir = str(_absolute_path(runtime_trash_dir or self.trash_dir))
        current_agent_data = (
            str(self.current_agent_data_path)
            if self.current_agent_data_path is not None
            else None
        )
        runtime_workdir = str(_resolve_path(workdir)) if workdir else None
        return (
            "# === KESTREL SAFE DELETION RUNTIME ===\n"
            "# Compatibility names: _kestrel_safe_remove, _KESTREL_TRASH_DIR\n"
            "_kestrel_runtime_namespace = {\n"
            "    '__name__': '_kestrel_safe_delete_runtime',\n"
            "}\n"
            f"exec({runtime_source!r}, _kestrel_runtime_namespace)\n"
            "_kestrel_runtime_namespace['install_safe_delete_runtime'](\n"
            f"    {trash_dir!r},\n"
            f"    {current_agent_data!r},\n"
            f"    {self.deletable_prefixes!r},\n"
            f"    {runtime_workdir!r},\n"
            ")\n"
            "del _kestrel_runtime_namespace\n"
            "# === END KESTREL SAFE DELETION RUNTIME ===\n"
        )

    def rewrite_python_script(
        self,
        content: str,
        workdir: Optional[str] = None,
        *,
        runtime_trash_dir: Optional[str | Path] = None,
    ) -> str:
        """Install safe deletion after legal module prologue statements."""
        helper = self.get_python_safe_remove_helper(
            workdir,
            runtime_trash_dir=runtime_trash_dir,
        )
        insertion_offset = self._python_runtime_insertion_offset(content)
        prefix = content[:insertion_offset]
        suffix = content[insertion_offset:]
        separator = "" if not prefix or prefix.endswith("\n") else "\n"
        return f"{prefix}{separator}{helper}{suffix}"

    @staticmethod
    def _python_runtime_insertion_offset(content: str) -> int:
        try:
            module = ast.parse(content)
        except SyntaxError as exc:
            raise ValueError(f"Cannot instrument invalid Python source: {exc}") from exc

        insertion_line = 0
        body_index = 0
        if (
            module.body
            and isinstance(module.body[0], ast.Expr)
            and isinstance(module.body[0].value, ast.Constant)
            and isinstance(module.body[0].value.value, str)
        ):
            insertion_line = module.body[0].end_lineno or module.body[0].lineno
            body_index = 1

        while body_index < len(module.body):
            node = module.body[body_index]
            if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
                break
            insertion_line = node.end_lineno or node.lineno
            body_index += 1

        lines = content.splitlines(keepends=True)
        if insertion_line == 0:
            if lines and lines[0].startswith("#!"):
                insertion_line = 1
            for index, line in enumerate(lines[:2], start=1):
                if _PYTHON_ENCODING_COOKIE.search(line):
                    insertion_line = max(insertion_line, index)
        return sum(len(line) for line in lines[:insertion_line])

    def rewrite_script(
        self,
        content: str,
        language: str,
        workdir: Optional[str] = None,
        *,
        runtime_trash_dir: Optional[str | Path] = None,
    ) -> str:
        if language == "bash":
            return self.rewrite_bash_script(
                content,
                workdir,
                runtime_trash_dir=runtime_trash_dir,
            )
        if language == "python":
            return self.rewrite_python_script(
                content,
                workdir,
                runtime_trash_dir=runtime_trash_dir,
            )
        logger.warning("Unknown language %r, not rewriting", language)
        return content


def rewrite_script_for_safety(
    content: str,
    language: str,
    workdir: Optional[str] = None,
) -> str:
    """Rewrite one script with the default destructive-operation policy."""
    return DestructiveOperationPolicy().rewrite_script(content, language, workdir)


__all__ = [
    "AgentDataProtectionError",
    "DEFAULT_DELETABLE_PREFIXES",
    "DEFAULT_TRASH_DIR",
    "DestructiveOperationPolicy",
    "ShellRewriteError",
    "rewrite_script_for_safety",
]
